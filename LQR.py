import numpy as np
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are

class TranslationLQR:
    def __init__(self, dt, max_accel, Q, R):
        ################################
        # X = [px, py, pz, vx, vy, vz]
        # B = [ax, ay, az]
        #################################
        self.dt = dt
        self.max_accel = max_accel
        
        self.A = np.block([[np.zeros((3,3)), np.eye(3)], [np.zeros((3,3)), np.zeros((3,3))]]) # px = vx*t, py = vy*t, pz = vz*t, all other equations are null
        self.B = np.block(np.zeros(3,3), np.eye(3)) # vx = ax*t, vy=ay*t, vz=az*t

        self.Ad, self.Bd, *_ = cont2discrete((self.A, self.B, np.eye(6), np.zeros((3,3))), self.dt)
        self.set_weights(Q, R)

    def set_weights(self, Q, R):
        self.Q, self.R = Q, R
        P = solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        self.K = np.linalg.solve(R + self.Bd.T @ P @ self.Bd, self.Bd.T @ P @ self.Ad)

    def compute(self, pos, vel, ref_pos, ref_vel=np.zeros(3), g=np.zeros(3)):
        e = np.concatenate([pos - ref_pos, vel - ref_vel])
        u = -self.K @ e - g
        n = np.linalg.norm(u)

        if n > self.max_accel:
            u *= self.max_accel / n
        return u



