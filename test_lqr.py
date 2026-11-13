"""
Offline sanity check for TranslationLQR.

Runs three tests without the sim:
  1. Single waypoint from an offset start (convergence, overshoot, settle time)
  2. Same, with Phobos gravity added and cancelled by feedforward
  3. A chain of waypoints with fly-through velocity (like the sphere planner)

The "truth" plant is integrated with small substeps (RK4) instead of reusing
Ad/Bd, so the test isn't just checking the controller against its own model.

Usage:  python test_lqr.py          (saves lqr_test.png and prints a summary)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are


# --------------------------------------------------------------------------
# Controller (same as the class we discussed)
# --------------------------------------------------------------------------
class TranslationLQR:
    def __init__(self, dt, Q, R, max_accel):
        self.dt = dt
        self.max_accel = max_accel
        I3, Z3 = np.eye(3), np.zeros((3, 3))
        A = np.block([[Z3, I3], [Z3, Z3]])
        B = np.block([[Z3], [I3]])
        self.Ad, self.Bd, *_ = cont2discrete(
            (A, B, np.eye(6), np.zeros((6, 3))), dt)
        self.set_weights(Q, R)

    def set_weights(self, Q, R):
        self.Q, self.R = Q, R
        P = solve_discrete_are(self.Ad, self.Bd, Q, R)
        self.K = np.linalg.solve(R + self.Bd.T @ P @ self.Bd, self.Bd.T @ P @ self.Ad)

    def compute(self, pos, vel, ref_pos, ref_vel=np.zeros(3), g=np.zeros(3)):
        e = np.concatenate([pos - ref_pos, vel - ref_vel])
        u = -self.K @ e - g
        n = np.linalg.norm(u)
        if n > self.max_accel:
            u *= self.max_accel / n
        return u


# --------------------------------------------------------------------------
# Truth plant: double integrator + gravity, RK4 with substeps
# --------------------------------------------------------------------------
def plant_step(x, u, g, dt, substeps=10):
    h = dt / substeps
    f = lambda s: np.concatenate([s[3:], u + g])   # u held constant (ZOH)
    for _ in range(substeps):
        k1 = f(x); k2 = f(x + h/2*k1); k3 = f(x + h/2*k2); k4 = f(x + h*k3)
        x = x + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return x


def run(ctrl, x0, waypoints, g, T, switch_radius=None, fly_speed=0.0):
    """waypoints: list of 3-vectors. Last one is the final target (stop there)."""
    dt = ctrl.dt
    n = int(T / dt)
    x = x0.copy()
    wp_i = 0
    X, U, WP = [], [], []
    dv_used = 0.0           # fuel proxy: integral of |a| dt (delta-v)
    for _ in range(n):
        wp = waypoints[wp_i]
        last = wp_i == len(waypoints) - 1

        # fly-through velocity toward the next waypoint (zero at the final one)
        if not last and fly_speed > 0:
            d = waypoints[wp_i + 1] - wp
            ref_vel = fly_speed * d / np.linalg.norm(d)
        else:
            ref_vel = np.zeros(3)

        u = ctrl.compute(x[:3], x[3:], wp, ref_vel, g)
        X.append(x.copy()); U.append(u.copy()); WP.append(wp_i)
        dv_used += np.linalg.norm(u) * dt
        x = plant_step(x, u, g, dt)

        # Switch when inside the radius OR once we've passed the waypoint along
        # the direction to the next one. The second condition matters: with a
        # nonzero ref_vel at a FIXED ref_pos, the LQR does not fly through the
        # point -- it parks where kp*pos_err balances kd*vel_err, which is
        # about kd*v_ref/kp past the waypoint (≈7.8 m with these gains). That
        # can sit outside the switch radius and the shuttle stalls there.
        if not last and switch_radius is not None:
            d = waypoints[wp_i + 1] - wp
            passed = np.dot(x[:3] - wp, d) > 0
            if passed or np.linalg.norm(x[:3] - wp) < switch_radius:
                wp_i += 1
    t = np.arange(n) * dt
    return t, np.array(X), np.array(U), np.array(WP), dv_used


def settle_time(t, X, target, tol=0.1):
    err = np.linalg.norm(X[:, :3] - target, axis=1)
    outside = np.where(err > tol)[0]
    if len(outside) == 0:
        return 0.0
    if outside[-1] == len(t) - 1:
        return None                                 # never settled
    return t[outside[-1] + 1]


def overshoot(X, x0, target):
    """Max distance past the target, measured along the start->target line."""
    d = target - x0[:3]
    d_hat = d / np.linalg.norm(d)
    progress = (X[:, :3] - x0[:3]) @ d_hat
    return max(0.0, progress.max() - np.linalg.norm(d))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- tuning (same numbers as the Bryson's rule example) ----
    dt = 0.1
    max_pos_err, max_vel_err, max_accel = 5.0, 2.0, 0.5
    Q = np.diag([1/max_pos_err**2]*3 + [1/max_vel_err**2]*3)
    R = np.diag([1/max_accel**2]*3)
    ctrl = TranslationLQR(dt, Q, R, max_accel)

    # ---- basic checks on the gain ----
    print("K shape:", ctrl.K.shape)
    eig = np.linalg.eigvals(ctrl.Ad - ctrl.Bd @ ctrl.K)
    print("closed-loop |eig| max: %.4f  (must be < 1 for stability)"
          % np.abs(eig).max())
    assert np.abs(eig).max() < 1, "closed loop is unstable!"
    print("K (x-axis row): kp = %.3f, kd = %.3f" % (ctrl.K[0, 0], ctrl.K[0, 3]))

    g_phobos = np.array([0.0, 0.0, -0.0057])       # m/s^2, roughly
    x0 = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    target = np.array([40.0, -20.0, 15.0])

    # ---- Test 1: single waypoint, no gravity ----
    t1, X1, U1, _, dv1 = run(ctrl, x0, [target], np.zeros(3), T=120)
    # ---- Test 2: with gravity (feedforward on) ----
    t2, X2, U2, _, dv2 = run(ctrl, x0, [target], g_phobos, T=120)
    # ---- Test 2b: gravity present but feedforward OFF (shows the drift) ----
    class NoFF(TranslationLQR):
        def compute(self, pos, vel, ref_pos, ref_vel=np.zeros(3), g=np.zeros(3)):
            return super().compute(pos, vel, ref_pos, ref_vel, np.zeros(3))
    ctrl_noff = NoFF(dt, Q, R, max_accel)
    def run_noff():
        x = x0.copy(); X = []
        for _ in range(int(120/dt)):
            u = ctrl_noff.compute(x[:3], x[3:], target)
            X.append(x.copy()); x = plant_step(x, u, g_phobos, dt)
        return np.array(X)
    X2b = run_noff()

    # ---- Test 3: waypoint chain with fly-through ----
    chain = [np.array(p, float) for p in
             [(15, 5, 3), (30, -5, 8), (45, 5, 12), (60, -5, 16), (75, 0, 20)]]
    t3, X3, U3, WP3, dv3 = run(ctrl, x0, chain, g_phobos, T=200,
                               switch_radius=3.0, fly_speed=1.5)

    # ---- summary ----
    print("\n%-34s %10s %10s %10s %10s" %
          ("test", "settle[s]", "overshoot", "final err", "delta-v"))
    for name, t, X, tgt, dv in [
            ("1 single wp, no gravity", t1, X1, target, dv1),
            ("2 single wp, gravity + FF", t2, X2, target, dv2),
            ("3 waypoint chain, fly-through", t3, X3, chain[-1], dv3)]:
        st = settle_time(t, X, tgt)
        print("%-34s %10s %9.2fm %9.3fm %8.2fm/s" % (
            name, "never" if st is None else "%.1f" % st,
            overshoot(X, x0, tgt), np.linalg.norm(X[-1, :3] - tgt), dv))
    print("2b steady-state error with gravity but NO feedforward: %.3f m"
          % np.linalg.norm(X2b[-1, :3] - target))
    # Corner cutting: the path must stay inside your free-space spheres, so
    # compare these numbers against the sphere radii your planner produces.
    miss = [np.linalg.norm(X3[:, :3] - w, axis=1).min() for w in chain]
    print("test 3 closest approach to each waypoint [m]:",
          np.round(miss, 2))
    print("max |u| commanded (test 1): %.3f m/s^2 (limit %.3f)"
          % (np.linalg.norm(U1, axis=1).max(), max_accel))

    # ---- plots ----
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(2, 2, 1)
    for i, lbl in enumerate("xyz"):
        ax.plot(t1, X1[:, i], label=lbl)
        ax.axhline(target[i], ls="--", lw=0.8, color="gray")
    ax.set_title("Test 1: position vs time"); ax.set_xlabel("t [s]")
    ax.set_ylabel("m"); ax.legend()

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t1, np.linalg.norm(U1, axis=1), label="|u|")
    for i, lbl in enumerate("xyz"):
        ax.plot(t1, U1[:, i], lw=0.8, label="u" + lbl)
    ax.axhline(max_accel, ls="--", color="red", lw=0.8, label="limit")
    ax.set_title("Test 1: commanded acceleration"); ax.set_xlabel("t [s]")
    ax.set_ylabel("m/s²"); ax.legend()

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t2, np.linalg.norm(X2[:, :3] - target, axis=1), label="with FF")
    ax.plot(t2, np.linalg.norm(X2b[:, :3] - target, axis=1), label="no FF")
    ax.set_yscale("log"); ax.set_title("Test 2: distance to target (gravity on)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("m"); ax.legend()

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    ax.plot(X3[:, 0], X3[:, 1], X3[:, 2], label="path")
    c = np.array(chain)
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], color="red", label="waypoints")
    ax.set_title("Test 3: waypoint chain"); ax.legend()

    plt.tight_layout()
    plt.savefig("lqr_test.png", dpi=110)
    print("\nsaved lqr_test.png")