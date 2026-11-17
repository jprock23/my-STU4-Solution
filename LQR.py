import numpy as np
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are


class TranslationLQR:
    """
    Double-integrator LQR with named weight profiles.

        x = [px, py, pz, vx, vy, vz]   (6x1 state)
        u = [ax, ay, az]               (3x1 input)

    Profiles let one controller switch between gain sets (e.g. "tight" near debris,
    "loose" in open space) without re-solving the Riccati equation every tick: gains
    are cached per (profile, dt).

    The constructor's Q, R become the "default" profile, so existing calls like
    lqr.compute(pos, vel, ref_pos, ref_vel, g, dt=dt_est) behave exactly as before.
    """

    def __init__(self, dt, max_accel, Q, R):
        self.dt = dt
        self.max_accel = max_accel

        # d/dt p = v,  d/dt v = u
        self.A = np.block([[np.zeros((3, 3)), np.eye(3)],
                           [np.zeros((3, 3)), np.zeros((3, 3))]])   # 6x6
        self.B = np.block([[np.zeros((3, 3))],
                           [np.eye(3)]])                            # 6x3

        # profile name -> dict(Q, R, u_deadband)
        self._profiles = {}
        self._K_cache = {}
        self.add_profile("default", Q, R)
        self.K = self.gain_for(self.dt)             # nominal gain of the default profile

    # ------------------------------------------------------------ profiles

    def add_profile(self, name, Q, R, u_deadband=0.0):
        """
        Register a weight set.
        u_deadband: m/s^2. If the total command (after gravity feedforward is removed)
                    is smaller than this, send only the gravity term. Stops the
                    thrusters from chasing sensor noise during long coasts.
        """
        self._profiles[name] = {"Q": np.asarray(Q, float),
                                "R": np.asarray(R, float),
                                "u_deadband": float(u_deadband)}
        # drop cached gains for this profile
        self._K_cache = {k: v for k, v in self._K_cache.items() if k[0] != name}

    def set_weights(self, Q, R):
        """Kept for compatibility: replaces the default profile."""
        self.add_profile("default", Q, R, self._profiles.get("default", {}).get("u_deadband", 0.0))
        self.K = self.gain_for(self.dt)

    def _solve_gain(self, Q, R, dt):
        Ad, Bd, *_ = cont2discrete(
            (self.A, self.B, np.eye(6), np.zeros((6, 3))), dt)
        P = solve_discrete_are(Ad, Bd, Q, R)
        return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

    def gain_for(self, dt, profile="default"):
        dt_key = max(round(dt, 2), 0.01)
        key = (profile, dt_key)
        if key not in self._K_cache:
            p = self._profiles[profile]
            self._K_cache[key] = self._solve_gain(p["Q"], p["R"], dt_key)
        return self._K_cache[key]

    def pd_gains(self, profile="default", dt=None):
        """(kp, kd) of one axis, e.g. for clamping position error: E_MAX = V_MAX * kd / kp."""
        K = self.gain_for(self.dt if dt is None else dt, profile)
        return K[0, 0], K[0, 3]

    # ------------------------------------------------------------- control

    def compute(self, pos, vel, ref_pos, ref_vel=np.zeros(3), g=np.zeros(3),
                dt=None, profile="default", a_ff=None):
        """
        World-frame acceleration command.
            dt=None    -> nominal gain
            profile    -> which weight set to use
            a_ff       -> reference acceleration feedforward (e.g. from the planner)
        """
        K = self.gain_for(self.dt if dt is None else dt, profile)
        e = np.concatenate([pos - ref_pos, vel - ref_vel])

        u_fb = -K @ e
        if a_ff is not None:
            u_fb = u_fb + np.asarray(a_ff, float)

        # Deadband applies to the feedback + feedforward part only, never to gravity
        if np.linalg.norm(u_fb) < self._profiles[profile]["u_deadband"]:
            u_fb = np.zeros(3)

        u = u_fb - g
        n = np.linalg.norm(u)
        if n > self.max_accel:
            u *= self.max_accel / n
        return u