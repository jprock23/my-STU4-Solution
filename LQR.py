import numpy as np
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are


class TranslationLQR:
    def __init__(self, dt, max_accel, Q, R):
        ################################
        # x = [px, py, pz, vx, vy, vz]   (6x1 state)
        # u = [ax, ay, az]               (3x1 input)
        #################################
        self.dt = dt
        self.max_accel = max_accel

        # d/dt p = v,  d/dt v = u
        self.A = np.block([[np.zeros((3, 3)), np.eye(3)],
                           [np.zeros((3, 3)), np.zeros((3, 3))]])   # 6x6
        self.B = np.block([[np.zeros((3, 3))],
                           [np.eye(3)]])                            # 6x3

        self._K_cache = {}
        self.set_weights(Q, R)

    def _solve_gain(self, dt):
        Ad, Bd, *_ = cont2discrete(
            (self.A, self.B, np.eye(6), np.zeros((6, 3))), dt)
        P = solve_discrete_are(Ad, Bd, self.Q, self.R)
        return np.linalg.solve(self.R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

    def set_weights(self, Q, R):
        self.Q, self.R = Q, R
        self._K_cache = {}                       # weights changed: old gains invalid
        self.K = self.gain_for(self.dt)          # nominal gain (used for E_MAX etc.)

    def gain_for(self, dt):
        """LQR gain for loop period dt. Binned to 10 ms and cached, so after the
        first few ticks this is just a dict lookup."""
        key = max(round(dt, 2), 0.01)
        if key not in self._K_cache:
            self._K_cache[key] = self._solve_gain(key)
        return self._K_cache[key]

    def compute(self, pos, vel, ref_pos, ref_vel=np.zeros(3), g=np.zeros(3), dt=None):
        """World-frame acceleration command. dt=None uses the nominal gain."""
        K = self.K if dt is None else self.gain_for(dt)
        e = np.concatenate([pos - ref_pos, vel - ref_vel])
        u = -K @ e - g
        n = np.linalg.norm(u)
        if n > self.max_accel:
            u *= self.max_accel / n
        return u