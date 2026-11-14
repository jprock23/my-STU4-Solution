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

#### Custom imports
from LQR import TranslationLQR
####


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
lqr = TranslationLQR(dt=DT, max_accel=MAX_ACCEL, Q=Q, R=R)   # keywords: safe for either arg order

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

    #####################################
    # Example for getting vehicle state #
    #####################################

    locVelAcc = shuttle.getLocVelAcc()
    loc = locVelAcc.loc_WRT_ExprIn(phobosFrame)
    vel = locVelAcc.vel_WRT_ExprIn(phobosFrame)
    rot = shuttle.getRotation().Quat_WRT(phobosFrame)
    angvel = shuttle.getAngVelocity().WRT_ExprIn(phobosFrame)

    #####################################################
    # Example for getting additional vehicle properties #
    #####################################################

    mass = shuttle.getMass()
    inertiaBodyFrame = shuttle.getInertiaTensor(shuttleFrame)
    damage: float = shuttle.GetParam(st.VarType.double, ["Stats", "Damage"])
    fuelUsed: float = shuttle.GetParam(st.VarType.double, ["Stats", "FuelUsed"])
    w = _mark("state", w)

    ##################################################################
    # Example for rotating the top camera 30 deg left and 10 deg up, #
    # and the bottom camera 20 deg right and 15 deg down.            #
    ##################################################################

    if iteration % 20 == 0:
        PanTiltCamera(topCamera, 30.0, -10.0)
        PanTiltCamera(bottomCamera, -20.0, 15.0)
    w = _mark("pantilt", w)

    ####################################################################################
    # Example for obtaining an image from the top camera with FOV = 60 deg and EV = 12 #
    ####################################################################################

    if iteration % 20 == 0:  # Don't hot-loop the image captures
        cameraParams = st.CaptureImageProperties()
        cameraParams.nonphys_EV = 10.0
        cameraParams.FOV = 60.0
        cameraParams.ResolutionX = 512
        cameraParams.ResolutionY = 512
        image = st.camera.CaptureImageBlocking(topCamera, cameraParams)

        if image is not None:
            imageR = np.asarray(image.as_RGB8().PixelsR).reshape((cameraParams.ResolutionX, cameraParams.ResolutionY))
            imageG = np.asarray(image.as_RGB8().PixelsG).reshape((cameraParams.ResolutionX, cameraParams.ResolutionY))
            imageB = np.asarray(image.as_RGB8().PixelsB).reshape((cameraParams.ResolutionX, cameraParams.ResolutionY))
            imageRGB = Image.fromarray(np.dstack((imageR, imageG, imageB)).astype(np.uint8))

            imageNumber += 1
        else:
            st.OnScreenLogMessage("Failed to capture camera image :(", "CameraNoCapture", st.Severity.Info)
    w = _mark("camera", w)

    ###################################################################################
    # Example for getting the positions, closest points, range rates, and diameters   #
    # of the nearest 4 pieces of debris and printing the distance to the closest one. #
    ###################################################################################

    debrisInfo = GetNearestDebrisLocations(4)
    if len(debrisInfo) > 0:
        closestDebris = debrisInfo[0]
        closestDebris_loc = closestDebris[0, :]
        closestDebris_closestPoint = closestDebris[1, 0]
        closestDebris_rangeRate = closestDebris[1, 1]
        closestDebris_diameter = closestDebris[1, 2]
        st.OnScreenLogMessage(f"Nearest debris: {np.linalg.norm(closestDebris_loc):.1f} m", "UserAPIPrintout", st.Severity.Info)
    w = _mark("debris", w)

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

    # Clamp: if the target is far, aim at a point E_MAX away along the line to it
    to_target = tgt_pos - pos
    dist = np.linalg.norm(to_target)
    if dist > E_MAX:
        ref_pos = pos + to_target * (E_MAX / dist)
    else:
        ref_pos = tgt_pos

    # Gravity points toward Phobos' center
    r = np.linalg.norm(pos)
    g = -G_PHOBOS * pos / r if r > 0 else np.zeros(3)

    # Adaptive gain: LQR re-designed for the measured period (cached, ~free).
    # E_MAX stays fixed from the nominal gain so the speed cap doesn't jitter.
    accel_world = lqr.compute(pos, vel, ref_pos, tgt_vel, g, dt=dt_est)   # phobosFrame
    accel_body = st.frames.FramedVector(accel_world, phobosFrame).ExprIn(shuttleFrame)
    SetCommand(accel_body, np.zeros(3))
    w = _mark("control", w)

    # ---- timing log: worst value of each section over the last LOG_EVERY iterations ----
    if iteration % LOG_EVERY == 0:
        msg = (f"[it {iteration}] sim_dt={timing_max.get('sim_dt', 0):.3f} "
               f"wall_dt={timing_max.get('wall_dt', 0):.3f} (design {DT:.3f}, "
               f"gain dt {dt_est:.3f}) | "
               + " ".join(f"{k}={v*1000:.0f}ms" for k, v in timing_max.items()
                          if k not in ("sim_dt", "wall_dt"))
               + f" | dist={dist:.0f}m")
        st.OnScreenLogMessage(msg, "Timing", st.Severity.Info)
        print(msg, flush=True)     # also goes to the console/log file for later
        timing_max.clear()

    iteration += 1

st.leave_sim()