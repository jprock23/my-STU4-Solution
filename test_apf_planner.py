"""
Offline sanity test for APFPlanner. No sim required.

Shuttle dynamics are a crude first-order lag standing in for the LQR:
    vel += (v_ref - vel) * dt / tau
Run:  python test_apf_planner.py
(expects your planner in apf_planner.py next to this file)
"""
import contextlib
import io
import sys

import numpy as np

from apf_planner import APFPlanner, Obstacle

SHUTTLE_R = 2.0


def run(obstacles, goal=np.array([300.0, 0, 0]), dt=0.1, T=120.0, tau=1.0, label=""):
    p = APFPlanner(goal)
    p.reset()
    pos, vel = np.zeros(3), np.zeros(3)
    obs = [Obstacle(o.position.copy(), o.velocity, o.radius) for o in obstacles]
    min_clear, escape_ticks, k = np.inf, 0, 0
    for k in range(int(T / dt)):
        v_ref = p.compute(pos, vel, obs, dt)
        if np.any(np.isnan(v_ref)):
            print(f"{label:26s} NaN in v_ref at t={k*dt:.1f}s")
            return None
        escape_ticks += p._escape_dir is not None
        vel += (v_ref - vel) * dt / tau
        pos += vel * dt
        for o in obs:
            o.position = o.position + o.velocity * dt
            min_clear = min(min_clear, np.linalg.norm(o.position - pos) - o.radius - SHUTTLE_R)
        if np.linalg.norm(goal - pos) < 1.0:
            break
    t = k * dt
    print(f"{label:26s} time={t:6.1f}s  min_clearance={min_clear:7.2f} m  escape_ticks={escape_ticks}")
    return t, min_clear


def zeros():
    return np.zeros(3)


if __name__ == "__main__":
    run([], label="no obstacles")
    run([Obstacle(np.array([150.0, 0, 0]), zeros(), 5.0)], label="static rock on line")
    run([Obstacle(np.array([150.0, 0.5, 0]), zeros(), 5.0)], label="static rock slightly off")
    run([Obstacle(np.array([150.0, -60, 0]), np.array([0, 4.0, 0]), 5.0)], label="crossing rock")
    run([Obstacle(np.array([150.0, 0, 0]), np.array([-5.0, 0, 0]), 5.0)], label="rock coming head-on")
    run([Obstacle(np.array([150.0, 6, 0]), zeros(), 4.0),
         Obstacle(np.array([150.0, -6, 0]), zeros(), 4.0)], label="too-narrow two-rock gap")

    # Random debris fields
    rng = np.random.default_rng(0)
    times, clears = [], []
    for _ in range(30):
        field = [Obstacle(np.array([rng.uniform(40, 260), rng.uniform(-25, 25), rng.uniform(-25, 25)]),
                          rng.normal(0, 1.5, 3), rng.uniform(2, 6)) for _ in range(25)]
        with contextlib.redirect_stdout(io.StringIO()):
            res = run(field)
        if res:
            times.append(res[0]); clears.append(res[1])
    print(f"\nrandom fields x30: worst clearance={min(clears):.2f} m, "
          f"mean time={np.mean(times):.1f}s, max time={max(times):.1f}s")