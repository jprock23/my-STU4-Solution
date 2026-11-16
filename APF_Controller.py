#############################################################################################
# DON'T CHANGE ANY OF THE BELOW. NECESSARY FOR CORRECTLY JOINING THE COMPETITION SIMULATION #
#############################################################################################

import os, sys, time, datetime, traceback
import spaceteams as st
def custom_exception_handler(exctype, value, tb):
    error_message = "".join(traceback.format_exception(exctype, value, tb))
    st.logger_fatal(error_message)
sys.excepthook = custom_exception_handler
st.connect_to_sim(sys.argv)

import numpy as np
import numpy.typing as npt

this = st.GetThisSystem()

doSpawning = st.SimGlobals.GetSimEntity().GetParam(st.VarType.bool, "DoSpawning")
if doSpawning:
    while True:
        time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))

phobos: st.Entity = this.GetParam(st.VarType.entityRef, "Phobos")
phobosFrame = phobos.GetBodyFixedFrame()
shuttle: st.Entity = this.GetParam(st.VarType.entityRef, "Shuttle")
shuttleFrame = shuttle.GetBodyFixedFrame()
target: st.Entity = this.GetParam(st.VarType.entityRef, "Target")

topCamera: st.Entity = this.GetParam(st.VarType.entityRef, "TopCamera")
topCameraFrame = topCamera.GetBodyFixedFrame()
bottomCamera: st.Entity = this.GetParam(st.VarType.entityRef, "BottomCamera")
bottomCameraFrame = bottomCamera.GetBodyFixedFrame()


def GetNearestDebrisLocations(N: int) -> list[npt.NDArray]:
    allDebrisInfo: list[npt.NDArray] = shuttle.GetParamArray(st.VarType.doubleM3x3, 
                                                            ["Radar", "SortedDebrisData"])
    nearest = allDebrisInfo[:min(N, len(allDebrisInfo))]
    return nearest


def PanTiltCamera(camera: st.Entity, yaw_deg: float, pitch_deg: float) -> None:
    if camera == topCamera:
        yaw_deg = np.clip(yaw_deg, -89.0, 89.0)
        pitch_deg = np.clip(pitch_deg, -50.0, 0.0)
    elif camera == bottomCamera:
        yaw_deg = np.clip(yaw_deg, -89.0, 89.0)
        pitch_deg = np.clip(pitch_deg, 0.0, 50.0)
    camera.setRotation_DCM(st.math.Euler321_to_DCM([np.deg2rad(yaw_deg), np.deg2rad(pitch_deg), 0.0]), shuttleFrame)


def SetCommand(accelCommand: npt.NDArray, angAccelCommand: npt.NDArray) -> None:
    shuttle.SetParam(st.VarType.doubleV3, ["JetSelection", "CommandedAccel"], accelCommand)
    shuttle.SetParam(st.VarType.doubleV3, ["JetSelection", "CommandedAngAccel"], angAccelCommand)


#############################################################################################
# DON'T CHANGE ANY OF THE ABOVE. NECESSARY FOR CORRECTLY JOINING THE COMPETITION SIMULATION #
#############################################################################################


#########################################################
# EXAMPLE USER CODE (NOTE: RUNNING A LOOP IS NECESSARY) #
#########################################################


import copy
from PIL import Image

#### Custom imports
from LQR import TranslationLQR
from apf_planner import APFPlanner, APFConfig, Obstacle
from debris_tracker import DebrisTracker, TrackerConfig
####

USE_TRACKER = True      # False -> fall back to radar_to_obstacles (zero velocity)
RADAR_MAX_RANGE = 150.0 # m, shared by the tracker and the fallback

# Shuttle-frame vector -> Phobos-frame axes (rotation only, no translation).
# Reads the global shuttleFrame at call time, so it uses the frame refreshed
# at the top of each tick.
def shuttle_to_phobos(v_body):
    return np.asarray(st.frames.FramedVector(v_body, shuttleFrame).ExprIn(phobosFrame), float)


def radar_to_obstacles(debris_info, own_pos, max_range=150.0):
    obs = []
    for d in debris_info:
        loc_body = np.asarray(d[0, :], float)
        if np.linalg.norm(loc_body) > max_range:
            continue                                   # list is sorted nearest-first
        loc_world = own_pos + np.asarray(
            st.frames.FramedVector(loc_body, shuttleFrame).ExprIn(phobosFrame), float)
        obs.append(Obstacle(position=loc_world,
                            velocity=np.zeros(3),
                            radius=0.5 * d[1, 2],      # diameter -> radius
                            position_sigma=1.0))
    return obs
####

###############################
# Get initial simulation time #
###############################

t0_sim: st.timestamp = st.SimGlobals.GetSimEntity().GetParam(st.VarType.timestamp, ["Time", "StartDateTAI"])
t0: datetime.datetime = t0_sim.as_datetime()

##################
# Start the loop #
##################
##########
DT        = 1.0/this.GetParam(st.VarType.double, "LoopFreqHz")    # MUST match how often the code below runs
MAX_ACCEL = 0.5    # m/s^2 -- set from the shuttle's real thrust/mass
V_MAX     = 5.0    # approach-speed cap, m/s (see table)
G_PHOBOS  = 0.0057 # m/s^2, set to 0.0 to disable gravity feedforward

max_pos_err, max_vel_err = 5.0, 2.0
Q = np.diag([1/max_pos_err**2]*3 + [1/max_vel_err**2]*3)
R = np.diag([1/MAX_ACCEL**2]*3)
lqr = TranslationLQR(dt=DT, max_accel=MAX_ACCEL, Q=Q, R=R)
cfg = APFConfig(cruise_speed=V_MAX, max_speed=V_MAX,
                goal_slowdown_radius=2 * V_MAX**2 / (2 * MAX_ACCEL),   # 2x stopping distance
                deadband_distance=40.0, influence_radius=40.0)
planner = APFPlanner(None, cfg)
planner.reset()

# Values from the offline tests (test_debris_tracker.py, 10 Hz, ~0.2 m noise).
# Re-tune once you know the real radar period and noise.
tracker_cfg = TrackerConfig(max_range=RADAR_MAX_RANGE,
                            beta=0.05,
                            max_misses=6,
                            sigma_growth_per_miss=0.3)
tracker = DebrisTracker(shuttle_to_phobos, tracker_cfg)
tracker.reset()
last_debris_raw = None      # to skip ticks where the radar hasn't refreshed
obstacles = []
radar_updates = 0


kp, kd = lqr.K[0, 0], lqr.K[0, 3]
E_MAX = V_MAX * kd / kp   # largest position error the LQR is allowed to see
###########
t_prev = t0

# ---- adaptive-gain dt estimate ----
DT_MIN, DT_MAX = 0.02, 2.0         # clamp so one huge spike can't produce a weird gain
DT_ALPHA = 0.2                     # smoothing: weight of the newest measurement
dt_est = DT                        # start at the design period
# -----------------------------------

# ---- timing instrumentation ----
LOG_EVERY = 10                     # log every N iterations
wall_prev = time.perf_counter()
timing_max = {}                    # worst time per section since the last log
def _mark(name, start):
    """Record time since `start` for section `name`; return a new start time."""
    now = time.perf_counter()
    timing_max[name] = max(timing_max.get(name, 0.0), now - start)
    return now
# --------------------------------

iteration = 0
imageNumber = 0
exit_flag: bool = False
while not exit_flag:

    ################################
    # Wait between loop iterations #
    ################################
    
    w = time.perf_counter()
    time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))
    w = _mark("sleep", w)

    ###################################
    # Get elapsed time from sim start #
    ###################################

    t_now: datetime.datetime = st.SimGlobals.SimClock.GetTimeNow().as_datetime()
    t_from_start: float = (t_now - t0).total_seconds()

    # sim-clock dt vs wall-clock dt for the whole previous iteration
    sim_dt = (t_now - t_prev).total_seconds()
    t_prev = t_now
    wall_now = time.perf_counter()
    wall_dt = wall_now - wall_prev
    wall_prev = wall_now
    timing_max["sim_dt"] = max(timing_max.get("sim_dt", 0.0), sim_dt)
    timing_max["wall_dt"] = max(timing_max.get("wall_dt", 0.0), wall_dt)

    # Smoothed loop-period estimate for the adaptive gain. Skip iteration 0:
    # t_prev starts at the sim's START time, so the first sim_dt is meaningless.
    if iteration > 0:
        dt_est = (1 - DT_ALPHA) * dt_est + DT_ALPHA * float(np.clip(sim_dt, DT_MIN, DT_MAX))
    w = _mark("clock", w)


    # debrisInfo = GetNearestDebrisLocations(4)
    # if len(debrisInfo) > 0:
    #     closestDebris = debrisInfo[0]
    #     closestDebris_loc = closestDebris[0, :]
    #     closestDebris_closestPoint = closestDebris[1, 0]
    #     closestDebris_rangeRate = closestDebris[1, 1]
    #     closestDebris_diameter = closestDebris[1, 2]
    #     st.OnScreenLogMessage(f"Nearest debris: {np.linalg.norm(closestDebris_loc):.1f} m", "UserAPIPrintout", st.Severity.Info)
    # w = _mark("debris", w)

    ##################################
    # YOUR CONTROLLER CODE GOES HERE #
    ##################################

    # ===== every tick =====
    shuttleFrame = shuttle.GetBodyFixedFrame()

    shuttle_state = shuttle.getLocVelAcc()
    pos = np.asarray(shuttle_state.loc_WRT_ExprIn(phobosFrame), dtype=float)
    vel = np.asarray(shuttle_state.vel_WRT_ExprIn(phobosFrame), dtype=float)

    target_state = target.getLocVelAcc()
    tgt_pos = np.asarray(target_state.loc_WRT_ExprIn(phobosFrame), dtype=float)
    tgt_vel = np.asarray(target_state.vel_WRT_ExprIn(phobosFrame), dtype=float)

    dist = np.linalg.norm(tgt_pos - pos)

    N = 200
    debrisInfo = GetNearestDebrisLocations(N)
    w = _mark("radar", w)

    if USE_TRACKER:
        # If the radar runs slower than this loop, the same frame comes back
        # several ticks in a row. Feeding a repeat to the tracker would look
        # like every rock froze (residual = -velocity*dt), dragging velocity
        # estimates toward zero. Only update on a new frame; otherwise keep
        # the last obstacle list (the planner re-predicts from velocities).
        debris_raw = np.asarray(debrisInfo, float) if len(debrisInfo) else np.zeros((0, 3, 3))
        is_new_frame = (last_debris_raw is None
                        or debris_raw.shape != last_debris_raw.shape
                        or not np.array_equal(debris_raw, last_debris_raw))
        if is_new_frame:
            obstacles = tracker.update(debrisInfo, pos, vel, t_from_start)
            last_debris_raw = debris_raw
            radar_updates += 1
    else:
        obstacles = radar_to_obstacles(debrisInfo, pos, max_range=RADAR_MAX_RANGE)
    w = _mark("tracker", w)

    planner.set_goal(tgt_pos)
    v_ref = planner.compute(pos, vel, obstacles, dt_est)
    v_cmd = tgt_vel + v_ref

    g = -G_PHOBOS * pos / np.linalg.norm(pos)

    accel_world = lqr.compute(pos, vel, pos, v_cmd, g, dt=dt_est)   # phobosFrame
    accel_body = st.frames.FramedVector(accel_world, phobosFrame).ExprIn(shuttleFrame)
    msg = (f"lin_accel_x: {accel_body[0]}, lin_accel_y: {accel_body[1]}, lin_accel_z: {accel_body[2]}")
    st.OnScreenLogMessage(msg, "Commands", st.Severity.Info)

    SetCommand(accel_body, np.zeros(3))
    w = _mark("control", w)

    # ---- timing log: worst value of each section over the last LOG_EVERY iterations ----
    # if iteration % LOG_EVERY == 0:
    #     msg = (f"[it {iteration}] sim_dt={timing_max.get('sim_dt', 0):.3f} "
    #            f"wall_dt={timing_max.get('wall_dt', 0):.3f} (design {DT:.3f}, "
    #            f"gain dt {dt_est:.3f}) | "
    #            + " ".join(f"{k}={v*1000:.0f}ms" for k, v in timing_max.items()
    #                       if k not in ("sim_dt", "wall_dt"))
    #            + f" | dist={dist:.0f}m")
    #     st.OnScreenLogMessage(msg, "Timing", st.Severity.Info)
    #     print(msg, flush=True)     # also goes to the console/log file for later
    #     timing_max.clear()

    # ---- tracker log ----
    # radar_updates / LOG_EVERY ticks tells you the real radar rate:
    # e.g. 10 updates in 10 ticks = radar runs at the loop rate; 2 = 1/5 of it.
    # if USE_TRACKER and iteration % LOG_EVERY == 0:
    #     msg = (f"[it {iteration}] radar_updates={radar_updates}/{LOG_EVERY} "
    #            f"obs={len(obstacles)} | {tracker.format_debug()}")
    #     st.OnScreenLogMessage(msg, "Tracker", st.Severity.Info)
    #     print(msg, flush=True)
    #     radar_updates = 0

    iteration += 1

st.leave_sim()