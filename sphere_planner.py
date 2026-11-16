"""
sphere_planner.py — Sphere-corridor planner for Escape from Phobos

Pipeline (one call per loop iteration):
    DebrisTracker.update() -> List[Obstacle] -> FreeSpaceModel
        -> SpherePlanner (A* over overlapping free spheres) -> CorridorTrajectory -> LQR reference

Modes (chosen every step, see SphereCorridorPlanner.step):
    LIFTOFF    climb straight off the surface until clear of the ground
    EMERGENCY  a rock is on a collision course right now -> APF fallback or built-in dodge
    DIRECT     straight line to the intercept point is clear -> skip the corridor
    CORRIDOR   follow the A* sphere corridor

Output: (p_ref, v_ref, a_ff) in the Phobos frame for the LQR. a_ff is always zero
(no polynomial trajectory fit; see CorridorTrajectory).

All positions/velocities are in the Phobos frame unless noted. Units: m, s, rad.

Things to verify in the sim (marked VERIFY):
    - shuttle_radius, max_accel, cruise_speed
    - Phobos frame origin is the moon's center (surface model "auto" checks this)
    - camera yaw/pitch sign conventions
"""

from __future__ import annotations

import csv
import heapq
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

from apf_planner import Obstacle   # position, velocity, radius, position_sigma (Phobos frame)


# =============================================================================
# Config
# =============================================================================

@dataclass
class PlannerConfig:
    # --- Geometry ---
    shuttle_radius: float = 3.0          # VERIFY: shuttle bounding radius
    safety_margin: float = 2.0           # extra padding on every obstacle
    min_sphere_radius: float = 4.0       # reject spheres the shuttle can't fit through
    max_sphere_radius: float = 60.0      # cap growth
    min_overlap: float = 1.5             # required overlap depth between parent/child (m)

    # --- Search ---
    samples_per_expansion: int = 10      # candidate directions per expanded sphere
    goal_bias: float = 0.6               # fraction of samples biased toward the goal
    goal_cone: float = 0.3               # spread of goal-biased samples (std of perturbation)
    max_expansions: int = 150            # per planning cycle
    max_plan_time_s: float = 0.05        # wall-clock budget per cycle
    prune_depth_frac: float = 0.5        # skip child if its center is this deep inside an existing sphere
    swept_samples: int = 3               # time samples when checking a sphere stays free

    # --- Cost weights ---
    w_time: float = 1.0
    w_clearance: float = 5.0             # penalty ~ 1 / radius
    w_turn: float = 2.0                  # penalty on direction change (proxy for delta-v)

    # --- Dynamics ---
    cruise_speed: float = 10.0           # VERIFY/tune: max speed relative to Phobos along the path
    max_accel: float = 1.0               # VERIFY: shuttle accel limit (m/s^2)
    brake_accel_frac: float = 0.6        # plan braking at this fraction of max_accel (margin)
    arrive_radius: float = 5.0           # m, consider the target reached inside this
    rendezvous_blend_dist: float = 80.0  # m, blend in target velocity over this distance

    # --- Flyby (success = get within success_radius; no velocity matching) ---
    flyby: bool = False                  # True -> no braking, no rendezvous blend, coast in
    success_radius: float = 50.0         # m, mission success distance
    success_margin: float = 15.0         # m, aim this far inside success_radius

    # --- Waypoint following ---
    waypoint_accept_radius: float = 3.0  # advance when this close to a waypoint...
    # ...or when past the plane through the waypoint perpendicular to the segment
    track_cross_track: bool = True       # p_ref = closest point on path (False -> p_ref = pos)

    # --- Sensing ---
    radar_range: float = 140.0           # keep <= tracker max_range (150); unknown beyond this
    treat_unknown_as_occupied: bool = True
    sigma_growth_rate: float = 0.5       # m/s, extra position uncertainty per second of prediction
    sigma_scale: float = 2.0             # inflate obstacles by this many sigmas

    # --- Phobos surface ---
    surface_mode: str = "auto"           # "auto" | "ellipsoid" | "plane" | "off"
    phobos_axes: tuple = (13000.0, 11400.0, 9100.0)  # m, triaxial ellipsoid semi-axes
    plane_up: tuple = (0.0, 0.0, 1.0)    # VERIFY: local "up" if the ellipsoid model is unusable
    ground_margin: float = 3.0           # m, keep sphere surfaces this far above launch-height ground
    liftoff_height: float = 15.0         # m, climb straight up this far before planning
    liftoff_speed: float = 3.0           # m/s

    # --- Threat / emergency ---
    threat_horizon_s: float = 8.0        # look this far ahead for closest approach
    emergency_tca_s: float = 3.0         # emergency if predicted miss < 0 within this time
    dodge_speed: float = 4.0             # m/s, built-in dodge (when no APF fallback)

    # --- Direct mode ---
    direct_check_step: float = 5.0       # m, sampling step along the straight line

    # --- Replanning ---
    replan_every_n: int = 5              # iterations between full replans
    commit_spheres: int = 2              # re-check the first N spheres every iteration
    max_plan_failures: int = 5           # consecutive failures before dropping the old corridor

    # --- Cameras ---
    yaw_sign: float = 1.0                # VERIFY: +1 if +yaw turns toward body +y
    pitch_sign: float = -1.0             # VERIFY: starter uses negative pitch = up for top cam
    cam_yaw_limit: float = 89.0
    top_pitch_range: tuple = (-50.0, 0.0)
    bottom_pitch_range: tuple = (0.0, 50.0)

    # --- Logging ---
    log_csv_path: Optional[str] = None   # write every new corridor here (None = off)


# =============================================================================
# Small helpers
# =============================================================================

def _unit(v: npt.NDArray) -> npt.NDArray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def _any_perpendicular(v: npt.NDArray) -> npt.NDArray:
    a = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _unit(np.cross(v, a))


# =============================================================================
# Phobos surface model
# =============================================================================

class SurfaceModel:
    """
    Height above the launch site's ground level, so spheres never dig into Phobos.

    'ellipsoid': altitude from Phobos's triaxial ellipsoid, offset so the launch point is 0.
                 Real terrain differs from the ellipsoid by up to a few hundred meters, but
                 using the launch point as the zero reference makes it accurate locally.
    'plane':     height along a fixed 'up' axis from the launch point.
    'auto':      ellipsoid if the start position is ~one Phobos radius from the origin
                 (i.e. the Phobos frame is centered on the moon), else plane.
    """

    def __init__(self, cfg: PlannerConfig):
        self.cfg = cfg
        self.mode = "off" if cfg.surface_mode == "off" else None
        self.axes = np.asarray(cfg.phobos_axes, dtype=float)
        self.start_alt = 0.0
        self.start_pos = np.zeros(3)
        self.warning = ""

    def initialize(self, start_pos: npt.NDArray) -> None:
        self.start_pos = np.array(start_pos, dtype=float)
        mode = self.cfg.surface_mode
        if mode == "auto":
            r = np.linalg.norm(start_pos)
            if 0.6 * self.axes.min() < r < 1.2 * self.axes.max():
                mode = "ellipsoid"
            else:
                mode = "plane"
                self.warning = (f"surface: |start|={r:.0f} m doesn't look like Phobos-centered; "
                                f"using plane with up={self.cfg.plane_up} (VERIFY)")
        self.mode = mode
        if mode == "ellipsoid":
            self.start_alt = self._ellipsoid_alt(self.start_pos)

    def _ellipsoid_alt(self, p: npt.NDArray) -> float:
        k = np.linalg.norm(p / self.axes)          # 1 on the surface
        r = np.linalg.norm(p)
        return r * (1.0 - 1.0 / max(k, 1e-9))     # radial distance above the ellipsoid

    def height(self, p: npt.NDArray) -> float:
        """Height above launch-site ground (m). +inf if disabled."""
        if self.mode in (None, "off"):
            return np.inf
        if self.mode == "ellipsoid":
            return self._ellipsoid_alt(p) - self.start_alt
        up = _unit(np.asarray(self.cfg.plane_up, dtype=float))
        return float(np.dot(p - self.start_pos, up))

    def up(self, p: npt.NDArray) -> npt.NDArray:
        if self.mode == "ellipsoid":
            return _unit(p / (self.axes ** 2))     # ellipsoid normal
        return _unit(np.asarray(self.cfg.plane_up, dtype=float))


# =============================================================================
# Free-space model
# =============================================================================

@dataclass
class Sphere:
    center: npt.NDArray
    radius: float
    t_arrival: float                     # estimated absolute sim time the shuttle reaches center


class FreeSpaceModel:
    """Answers 'how big a free sphere fits here, at time t?'"""

    def __init__(self, cfg: PlannerConfig, surface: SurfaceModel):
        self.cfg = cfg
        self.surface = surface
        self.obstacles: list[Obstacle] = []
        self.shuttle_pos = np.zeros(3)
        self.t_snapshot = 0.0
        self.extra_obstacles: list[tuple[npt.NDArray, float]] = []  # static (center, radius), e.g. from depth

    def set_snapshot(self, obstacles: list[Obstacle], shuttle_pos: npt.NDArray, t: float) -> None:
        # Copy: the tracker's Obstacles share arrays with its Tracks, which it mutates in place.
        self.obstacles = [Obstacle(position=np.array(o.position, dtype=float),
                                   velocity=np.array(o.velocity, dtype=float),
                                   radius=float(o.radius),
                                   position_sigma=float(o.position_sigma))
                          for o in obstacles]
        self.shuttle_pos = np.array(shuttle_pos, dtype=float)
        self.t_snapshot = t

    def predicted(self, o: Obstacle, t: float) -> tuple[npt.NDArray, float]:
        """(predicted center, inflated radius) of obstacle o at time t."""
        dt = max(0.0, t - self.t_snapshot)
        c = o.position + o.velocity * dt
        sigma = o.position_sigma + self.cfg.sigma_growth_rate * dt
        inflate = (o.radius + self.cfg.shuttle_radius
                   + self.cfg.safety_margin + self.cfg.sigma_scale * sigma)
        return c, inflate

    def clearance(self, p: npt.NDArray, t: float) -> float:
        """Radius of the largest free sphere centered at p at time t (shuttle size already included)."""
        best = self.cfg.max_sphere_radius
        for o in self.obstacles:
            c, inflate = self.predicted(o, t)
            best = min(best, np.linalg.norm(p - c) - inflate)

        for c, r in self.extra_obstacles:
            best = min(best, np.linalg.norm(p - c) - r - self.cfg.shuttle_radius - self.cfg.safety_margin)

        # Ground: launch point is height 0 (shuttle center sits on the pad)
        best = min(best, self.surface.height(p) - self.cfg.ground_margin)

        if self.cfg.treat_unknown_as_occupied:
            best = min(best, self.cfg.radar_range - np.linalg.norm(p - self.shuttle_pos))

        return best

    def swept_radius(self, p: npt.NDArray, t_enter: float, t_exit: float) -> float:
        """Smallest clearance at p over the window the shuttle occupies the sphere."""
        n = max(self.cfg.swept_samples, 1)
        ts = np.linspace(t_enter, t_exit, n) if n > 1 else [t_enter]
        return min(self.clearance(p, float(t)) for t in ts)

    def segment_clear(self, a: npt.NDArray, b: npt.NDArray, t0: float, speed: float) -> bool:
        """Is the straight line a->b free when flown at `speed` starting at t0?"""
        L = np.linalg.norm(b - a)
        n = max(int(np.ceil(L / self.cfg.direct_check_step)), 1)
        for k in range(n + 1):
            s = k / n
            p = a + s * (b - a)
            t = t0 + s * L / max(speed, 1e-3)
            # Ignore the unknown-space cap here; direct mode is only used within radar range
            c = self._clearance_known(p, t)
            if c < 0.0:
                return False
        return True

    def _clearance_known(self, p: npt.NDArray, t: float) -> float:
        best = np.inf
        for o in self.obstacles:
            c, inflate = self.predicted(o, t)
            best = min(best, np.linalg.norm(p - c) - inflate)
        for c, r in self.extra_obstacles:
            best = min(best, np.linalg.norm(p - c) - r - self.cfg.shuttle_radius - self.cfg.safety_margin)
        best = min(best, self.surface.height(p) - self.cfg.ground_margin)
        return best


# =============================================================================
# Sphere A* planner
# =============================================================================

@dataclass(order=True)
class _Node:
    f: float
    idx: int = field(compare=False)


class SpherePlanner:
    def __init__(self, cfg: PlannerConfig, space: FreeSpaceModel):
        self.cfg = cfg
        self.space = space
        self.rng = np.random.default_rng(0)
        self.last_stats: dict = {}

    def plan(self, start: npt.NDArray, start_vel: npt.NDArray,
             goal: npt.NDArray, t_now: float) -> tuple[list[Sphere] | None, bool]:
        """
        A* over a graph of overlapping free spheres.
        Returns (corridor, reached_goal). corridor is None on failure.
        reached_goal=False means a partial corridor toward the most promising sphere.
        """
        cfg = self.cfg
        deadline = time.perf_counter() + cfg.max_plan_time_s

        r0 = self.space.clearance(start, t_now)
        if r0 < 0.5 * cfg.min_sphere_radius:
            self.last_stats = {"fail": "start_blocked", "r0": r0}
            return None, False

        spheres: list[Sphere] = [Sphere(start.copy(), r0, t_now)]
        parent: list[int] = [-1]
        g: list[float] = [0.0]
        heading: list[npt.NDArray] = [_unit(start_vel) if np.linalg.norm(start_vel) > 0.5
                                      else _unit(goal - start)]

        open_heap = [_Node(self._h(start, goal), 0)]
        best_idx, best_h = 0, self._h(start, goal)
        expansions = 0

        while open_heap and expansions < cfg.max_expansions and time.perf_counter() < deadline:
            i = heapq.heappop(open_heap).idx
            s = spheres[i]
            expansions += 1

            if self._reaches_goal(s, goal):
                self.last_stats = {"expansions": expansions, "nodes": len(spheres), "reached": True}
                return self._reconstruct(spheres, parent, i), True

            for c in self._sample_children(s, goal):
                if self._is_duplicate(c, spheres):
                    continue
                t_arr = s.t_arrival + np.linalg.norm(c - s.center) / cfg.cruise_speed
                r_now = self.space.clearance(c, t_arr)
                if r_now < cfg.min_sphere_radius:
                    continue
                t_exit = t_arr + 2.0 * r_now / cfg.cruise_speed
                r = self.space.swept_radius(c, t_arr, t_exit)   # shrink to what stays free
                child = Sphere(c, r, t_arr)
                if not self._valid_child(s, child):
                    continue

                new_heading = _unit(c - s.center)
                j = len(spheres)
                spheres.append(child)
                parent.append(i)
                heading.append(new_heading)
                g.append(g[i] + self._edge_cost(s, child, heading[i], new_heading))

                h = self._h(c, goal)
                heapq.heappush(open_heap, _Node(g[j] + h, j))
                if h < best_h:
                    best_idx, best_h = j, h

        self.last_stats = {"expansions": expansions, "nodes": len(spheres), "reached": False}
        if best_idx > 0:
            return self._reconstruct(spheres, parent, best_idx), False
        self.last_stats["fail"] = "no_progress"
        return None, False

    # ---- helpers ----

    def _sample_children(self, s: Sphere, goal: npt.NDArray) -> list[npt.NDArray]:
        """Candidate centers just inside the surface of s: goal-biased plus spread-out directions."""
        cfg = self.cfg
        to_goal = _unit(goal - s.center)
        n_goal = int(round(cfg.samples_per_expansion * cfg.goal_bias))
        dirs = [to_goal]                               # always try straight at the goal
        for k in range(1, cfg.samples_per_expansion):
            if k < n_goal:
                d = to_goal + cfg.goal_cone * self.rng.normal(size=3)
            else:
                d = self.rng.normal(size=3)
            dirs.append(_unit(d))
        step = max(s.radius - cfg.min_overlap, cfg.min_overlap)
        return [s.center + step * d for d in dirs]

    def _is_duplicate(self, c: npt.NDArray, spheres: list[Sphere]) -> bool:
        """Center lies deep inside an existing sphere -> that region is already explored."""
        frac = self.cfg.prune_depth_frac
        for s in spheres:
            if np.linalg.norm(c - s.center) < frac * s.radius:
                return True
        return False

    def _valid_child(self, parent: Sphere, child: Sphere) -> bool:
        cfg = self.cfg
        if child.radius < cfg.min_sphere_radius:
            return False
        overlap = parent.radius + child.radius - np.linalg.norm(child.center - parent.center)
        return overlap >= cfg.min_overlap

    def _edge_cost(self, a: Sphere, b: Sphere, h_prev: npt.NDArray, h_new: npt.NDArray) -> float:
        cfg = self.cfg
        dist = np.linalg.norm(b.center - a.center)
        turn = 1.0 - float(np.dot(h_prev, h_new))        # 0 straight, 2 reversal
        return (cfg.w_time * dist / cfg.cruise_speed
                + cfg.w_clearance / max(b.radius, 1e-3)
                + cfg.w_turn * turn)

    def _h(self, p: npt.NDArray, goal: npt.NDArray) -> float:
        return self.cfg.w_time * np.linalg.norm(goal - p) / self.cfg.cruise_speed

    def _reaches_goal(self, s: Sphere, goal: npt.NDArray) -> bool:
        d = np.linalg.norm(goal - s.center)
        if self.cfg.flyby:
            aim = self.cfg.success_radius - self.cfg.success_margin
            return d - s.radius <= aim         # some point of the sphere is within aim radius
        return d <= s.radius

    @staticmethod
    def _reconstruct(spheres, parent, i) -> list[Sphere]:
        path = []
        while i != -1:
            path.append(spheres[i])
            i = parent[i]
        return path[::-1]


# =============================================================================
# Goal selection (intercept point)
# =============================================================================

def predict_intercept(shuttle_pos: npt.NDArray, target_pos: npt.NDArray,
                      target_vel: npt.NDArray, cruise_speed: float,
                      target_acc: Optional[npt.NDArray] = None,
                      iters: int = 8) -> tuple[npt.NDArray, float]:
    """
    Fixed-point iteration: where will the target be when we get there?
    Uses p + v t + 0.5 a t^2 when target_acc is given (captures the curvature of the
    target's path in the rotating Phobos frame over moderate horizons).
    Returns (intercept_point, time_to_go).
    """
    a = np.zeros(3) if target_acc is None else np.asarray(target_acc, dtype=float)

    def at(tgo):
        return target_pos + target_vel * tgo + 0.5 * a * tgo * tgo

    tgo = np.linalg.norm(target_pos - shuttle_pos) / cruise_speed
    for _ in range(iters):
        tgo_new = np.linalg.norm(at(tgo) - shuttle_pos) / cruise_speed
        if abs(tgo_new - tgo) < 1e-3:
            tgo = tgo_new
            break
        tgo = 0.5 * (tgo + tgo_new)                 # damped, avoids oscillation
    return at(tgo), tgo


# =============================================================================
# Corridor -> reference for LQR
# =============================================================================

class CorridorTrajectory:
    """
    Converts the sphere chain into a (p_ref, v_ref, a_ff) reference.

    Waypoints sit at the centers of the overlap 'doorways' between consecutive spheres.
    Speed along the path is limited by:
        - cruise_speed
        - stopping distance to the end of the corridor (if it doesn't reach the goal yet),
          so the shuttle can always stop before unknown space
        - braking toward the target (relative speed <= sqrt(2 a d))
    Velocity blends toward the target's velocity in the last rendezvous_blend_dist meters.

    Not implemented: a smooth polynomial fit through the corridor (min-snap QP). a_ff is zero.
    """

    def __init__(self, cfg: PlannerConfig):
        self.cfg = cfg
        self.waypoints: list[npt.NDArray] = []
        self.idx = 0
        self.reaches_goal = False
        self.seg_start = np.zeros(3)

    def set_corridor(self, corridor: list[Sphere], goal: npt.NDArray,
                     reaches_goal: bool, pos: npt.NDArray) -> None:
        wps = [self._overlap_center(a, b) for a, b in zip(corridor[:-1], corridor[1:])]
        if reaches_goal and self.cfg.flyby:
            last = corridor[-1]
            to_g = goal - last.center
            d = np.linalg.norm(to_g)
            wps.append(last.center + _unit(to_g) * min(last.radius, d))
        elif reaches_goal:
            wps.append(np.array(goal, dtype=float))
        else:
            wps.append(corridor[-1].center.copy())    # stop at the last known-free sphere
        self.waypoints = wps
        self.idx = 0
        self.reaches_goal = reaches_goal
        self.seg_start = np.array(pos, dtype=float)

    def set_direct(self, goal: npt.NDArray, pos: npt.NDArray) -> None:
        self.waypoints = [np.array(goal, dtype=float)]
        self.idx = 0
        self.reaches_goal = True
        self.seg_start = np.array(pos, dtype=float)

    @staticmethod
    def _overlap_center(a: Sphere, b: Sphere) -> npt.NDArray:
        """Center of the circle where spheres a and b intersect."""
        d_vec = b.center - a.center
        d = np.linalg.norm(d_vec)
        if d < 1e-9:
            return a.center.copy()
        x = (d * d + a.radius ** 2 - b.radius ** 2) / (2 * d)
        x = np.clip(x, 0.0, d)
        return a.center + (x / d) * d_vec

    def _advance(self, pos: npt.NDArray) -> None:
        """Move to the next waypoint when close to it or past its plane (no stopping)."""
        while self.idx < len(self.waypoints) - 1:
            wp = self.waypoints[self.idx]
            seg_dir = _unit(wp - self.seg_start)
            close = np.linalg.norm(wp - pos) < self.cfg.waypoint_accept_radius
            passed = np.dot(pos - wp, seg_dir) > 0.0
            if close or passed:
                self.seg_start = wp
                self.idx += 1
            else:
                break

    def remaining_length(self, pos: npt.NDArray) -> float:
        if not self.waypoints:
            return 0.0
        L = np.linalg.norm(self.waypoints[self.idx] - pos)
        for a, b in zip(self.waypoints[self.idx:-1], self.waypoints[self.idx + 1:]):
            L += np.linalg.norm(b - a)
        return L

    def reference(self, pos: npt.NDArray, vel: npt.NDArray,
                  target_pos: npt.NDArray, target_vel: npt.NDArray
                  ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        cfg = self.cfg
        if not self.waypoints:
            return pos.copy(), vel.copy(), np.zeros(3)

        self._advance(pos)
        wp = self.waypoints[self.idx]
        to_wp = wp - pos
        dir_path = _unit(to_wp) if np.linalg.norm(to_wp) > 1e-6 else _unit(target_pos - pos)

        a_brake = cfg.brake_accel_frac * cfg.max_accel
        L_path = self.remaining_length(pos)
        d_target = np.linalg.norm(target_pos - pos)

        # Speed along the path (relative to Phobos)
        speed = cfg.cruise_speed
        if not self.reaches_goal:
            speed = min(speed, np.sqrt(2.0 * a_brake * max(L_path, 0.0)))

        if cfg.flyby:
            aim = cfg.success_radius - cfg.success_margin
            if d_target <= aim:
                v_ref = vel.copy()                  # inside: stop thrusting, coast
            else:
                v_ref = dir_path * speed            # no braking toward the target
            if cfg.track_cross_track:
                a, b = self.seg_start, wp
                ab = b - a
                s = np.clip(np.dot(pos - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
                p_ref = a + s * ab
            else:
                p_ref = pos.copy()
            return p_ref, v_ref, np.zeros(3)

        # Rendezvous blend: weight w goes 0 -> 1 over the last blend distance
        w = float(np.clip(1.0 - d_target / cfg.rendezvous_blend_dist, 0.0, 1.0))
        if self.reaches_goal:
            # relative speed toward the target limited by braking distance
            rel_speed = min(cfg.cruise_speed, np.sqrt(2.0 * a_brake * max(d_target - cfg.arrive_radius, 0.0)))
            v_far = dir_path * speed
            v_near = target_vel + _unit(target_pos - pos) * rel_speed
            v_ref = (1.0 - w) * v_far + w * v_near
            # far away, still respect braking relative to the target
            v_rel = v_ref - target_vel
            if np.linalg.norm(v_rel) > rel_speed and d_target < 4.0 * cfg.rendezvous_blend_dist:
                v_ref = target_vel + _unit(v_rel) * rel_speed
        else:
            v_ref = dir_path * speed

        if d_target <= cfg.arrive_radius:
            v_ref = np.array(target_vel, dtype=float)

        # Position reference: closest point on the current segment (cross-track correction)
        if cfg.track_cross_track:
            a, b = self.seg_start, wp
            ab = b - a
            s = np.clip(np.dot(pos - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
            p_ref = a + s * ab
        else:
            p_ref = pos.copy()

        return p_ref, v_ref, np.zeros(3)


# =============================================================================
# Threat assessment / emergency dodge
# =============================================================================

@dataclass
class Threat:
    obstacle: Obstacle
    tca: float               # s, time of closest approach (clipped to [0, horizon])
    miss: float              # m, surface-to-surface miss distance (negative = collision)
    miss_vec: npt.NDArray    # from rock center to shuttle at closest approach


def assess_threats(obstacles: list[Obstacle], pos: npt.NDArray, vel: npt.NDArray,
                   cfg: PlannerConfig) -> list[Threat]:
    """Closest approach of each obstacle assuming both keep current velocity. Sorted worst first."""
    out = []
    for o in obstacles:
        r = pos - o.position                      # rock -> shuttle
        v = vel - o.velocity
        vv = float(np.dot(v, v))
        tca = 0.0 if vv < 1e-9 else float(np.clip(-np.dot(r, v) / vv, 0.0, cfg.threat_horizon_s))
        miss_vec = r + v * tca
        miss = (np.linalg.norm(miss_vec) - o.radius - cfg.shuttle_radius
                - cfg.sigma_scale * (o.position_sigma + cfg.sigma_growth_rate * tca))
        out.append(Threat(o, tca, miss, miss_vec))
    out.sort(key=lambda th: th.miss + 2.0 * th.tca)
    return out


def dodge_velocity(threat: Threat, vel: npt.NDArray, cfg: PlannerConfig) -> npt.NDArray:
    """Push sideways out of the rock's path (perpendicular to relative velocity)."""
    rel_v = vel - threat.obstacle.velocity
    away = threat.miss_vec
    if np.linalg.norm(rel_v) > 1e-6:
        away = away - np.dot(away, _unit(rel_v)) * _unit(rel_v)
    if np.linalg.norm(away) < 1e-3:
        away = _any_perpendicular(rel_v if np.linalg.norm(rel_v) > 1e-6 else np.array([1.0, 0, 0]))
    return vel + _unit(away) * cfg.dodge_speed


# =============================================================================
# Camera pointing
# =============================================================================

def _yaw_pitch(v_body: npt.NDArray, cfg: PlannerConfig) -> tuple[float, float]:
    x, y, z = v_body
    yaw = cfg.yaw_sign * np.degrees(np.arctan2(y, x))
    pitch = cfg.pitch_sign * np.degrees(np.arctan2(z, np.hypot(x, y)))
    return yaw, pitch


def choose_camera_aim(look_points: list[npt.NDArray], shuttle_pos: npt.NDArray,
                      to_body: Callable[[npt.NDArray], npt.NDArray],
                      cfg: PlannerConfig) -> dict[str, tuple[float, float]]:
    """
    look_points: Phobos-frame points in priority order (e.g. corridor lookahead, worst threat).
    to_body: rotates a Phobos-frame vector into the shuttle body frame.
    Each camera takes the highest-priority point it can reach within its limits;
    if none, it points at the clipped direction of the top priority point.
    Returns {"top": (yaw, pitch), "bottom": (yaw, pitch)} for PanTiltCamera.
    """
    angles = [_yaw_pitch(np.asarray(to_body(p - shuttle_pos), dtype=float), cfg) for p in look_points]
    ranges = {"top": cfg.top_pitch_range, "bottom": cfg.bottom_pitch_range}
    result, used = {}, set()
    for cam, (lo, hi) in ranges.items():
        choice = None
        for k, (yaw, pitch) in enumerate(angles):
            if k in used:
                continue
            if abs(yaw) <= cfg.cam_yaw_limit and lo <= pitch <= hi:
                choice = k
                break
        if choice is not None:
            used.add(choice)
            result[cam] = angles[choice]
        elif angles:
            yaw, pitch = angles[0]
            result[cam] = (float(np.clip(yaw, -cfg.cam_yaw_limit, cfg.cam_yaw_limit)),
                           float(np.clip(pitch, lo, hi)))
        else:
            result[cam] = (0.0, (lo + hi) / 2.0)
    return result


# =============================================================================
# Top-level glue
# =============================================================================

class SphereCorridorPlanner:
    """
    Call step() every loop with the tracker's obstacles; feed the result to your LQR.

        obstacles = tracker.update(debris_info, pos, vel, t)
        p_ref, v_ref, a_ff = sphere.step(t, pos, vel, obstacles, tgt_pos, tgt_vel, dt, tgt_acc)
        accel_world = lqr.compute(pos, vel, p_ref, v_ref, ...)

    fallback: your APFPlanner (needs set_goal(goal) and compute(pos, vel, obstacles, dt)
    returning a velocity relative to the target). Used in EMERGENCY mode and when no
    corridor exists. Without it, a built-in sideways dodge is used.
    """

    def __init__(self, cfg: PlannerConfig | None = None, fallback=None):
        self.cfg = cfg or PlannerConfig()
        self.surface = SurfaceModel(self.cfg)
        self.space = FreeSpaceModel(self.cfg, self.surface)
        self.planner = SpherePlanner(self.cfg, self.space)
        self.traj = CorridorTrajectory(self.cfg)
        self.fallback = fallback
        self.reset()

    def reset(self) -> None:
        self.corridor: list[Sphere] = []
        self.reaches_goal = False
        self.iter = 0
        self.plan_failures = 0
        self.mode = "INIT"
        self.goal = np.zeros(3)
        self.threats: list[Threat] = []
        self.liftoff_done = False
        self.surface.mode = None if self.cfg.surface_mode != "off" else "off"
        self.n_plans = 0

    # ------------------------------------------------------------------ main

    def step(self, t: float, pos: npt.NDArray, vel: npt.NDArray,
             obstacles: list[Obstacle],
             target_pos: npt.NDArray, target_vel: npt.NDArray, dt: float,
             target_acc: Optional[npt.NDArray] = None,
             obstacles_t: Optional[float] = None
             ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        obstacles_t: sim time the obstacle states were estimated at (the time of the
        last tracker.update). Defaults to t. Pass it when you reuse the previous
        obstacle list on ticks where the radar didn't refresh.
        """
        cfg = self.cfg
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)
        target_vel = np.asarray(target_vel, dtype=float)

        if self.surface.mode is None:
            self.surface.initialize(pos)

        self.space.set_snapshot(obstacles, pos, t if obstacles_t is None else obstacles_t)
        if obstacles_t is not None and obstacles_t < t:
            # bring obstacle states forward to now so threat checks use current positions
            for o in self.space.obstacles:
                o.position = o.position + o.velocity * (t - obstacles_t)
                o.position_sigma += cfg.sigma_growth_rate * (t - obstacles_t)
            self.space.t_snapshot = t
        self.goal, _ = predict_intercept(pos, target_pos, target_vel, cfg.cruise_speed, target_acc)
        self.threats = assess_threats(self.space.obstacles, pos, vel, cfg)
        self.iter += 1

        # 1. EMERGENCY: something will hit us soon at current velocity
        worst = self.threats[0] if self.threats else None
        if worst is not None and worst.miss < 0.0 and worst.tca < cfg.emergency_tca_s:
            self.mode = "EMERGENCY"
            self.corridor = []                    # force a fresh plan afterwards
            return self._fallback_ref(pos, vel, obstacles, target_vel, dt, worst)

        # 2. LIFTOFF: climb off the surface before planning
        if not self.liftoff_done:
            if self.surface.height(pos) >= cfg.liftoff_height or self.surface.mode == "off":
                self.liftoff_done = True
            else:
                self.mode = "LIFTOFF"
                up = self.surface.up(pos)
                return pos.copy(), up * cfg.liftoff_speed, np.zeros(3)

        # 3. DIRECT: straight line to the intercept is clear (within radar range)
        d_goal = np.linalg.norm(self.goal - pos)
        check_end = self.goal if d_goal <= cfg.radar_range else pos + _unit(self.goal - pos) * cfg.radar_range
        if self.space.segment_clear(pos, check_end, t, max(np.linalg.norm(vel), cfg.cruise_speed)):
            if self.mode != "DIRECT" or self.iter % cfg.replan_every_n == 0:
                self.traj.set_direct(self.goal, pos)
            self.mode = "DIRECT"
            self.corridor = []
            return self.traj.reference(pos, vel, target_pos, target_vel)

        # 4. CORRIDOR
        need_replan = (self.mode != "CORRIDOR"
                       or not self.corridor
                       or self.iter % cfg.replan_every_n == 0
                       or not self._committed_still_safe(t))
        if need_replan:
            corridor, reached = self.planner.plan(pos, vel, self.goal, t)
            if corridor and len(corridor) >= 1:
                self.corridor = corridor
                self.reaches_goal = reached
                self.traj.set_corridor(corridor, self.goal, reached, pos)
                self.plan_failures = 0
                self.n_plans += 1
                if cfg.log_csv_path:
                    log_corridor_csv(cfg.log_csv_path, t, corridor, self.goal, reached)
            else:
                self.plan_failures += 1
                if self.plan_failures >= cfg.max_plan_failures:
                    self.corridor = []

        if not self.corridor:
            self.mode = "FALLBACK"
            return self._fallback_ref(pos, vel, obstacles, target_vel, dt, worst)

        self.mode = "CORRIDOR"
        return self.traj.reference(pos, vel, target_pos, target_vel)

    # ------------------------------------------------------------- helpers

    def _fallback_ref(self, pos, vel, obstacles, target_vel, dt, worst: Optional[Threat]):
        if self.fallback is not None:
            self.fallback.set_goal(self.goal)
            v_rel = self.fallback.compute(pos, vel, obstacles, dt)
            return pos.copy(), target_vel + np.asarray(v_rel, dtype=float), np.zeros(3)
        if worst is not None and worst.miss < 0.0:
            return pos.copy(), dodge_velocity(worst, vel, self.cfg), np.zeros(3)
        # nothing threatening: crawl toward the goal
        return pos.copy(), _unit(self.goal - pos) * min(2.0, self.cfg.cruise_speed), np.zeros(3)

    def _committed_still_safe(self, t: float) -> bool:
        """Re-check the first few spheres against fresh obstacle estimates."""
        for s in self.corridor[: self.cfg.commit_spheres]:
            tt = max(t, s.t_arrival)
            if self.space.clearance(s.center, tt) < 0.8 * s.radius:
                return False
        return True

    def camera_aims(self, pos: npt.NDArray,
                    to_body: Callable[[npt.NDArray], npt.NDArray]) -> dict[str, tuple[float, float]]:
        """
        Where to point the cameras this frame. Priority: corridor lookahead, worst threat, goal.
        to_body example:
            lambda v: np.asarray(st.frames.FramedVector(v, phobosFrame).ExprIn(shuttleFrame), float)
        """
        pts = []
        if self.traj.waypoints:
            k = min(self.traj.idx + 1, len(self.traj.waypoints) - 1)
            pts.append(self.traj.waypoints[k])
        if self.threats:
            pts.append(self.threats[0].obstacle.position)
        pts.append(self.goal)
        return choose_camera_aim(pts, np.asarray(pos, dtype=float), to_body, self.cfg)

    def debug_line(self) -> str:
        w = self.threats[0] if self.threats else None
        thr = f"miss={w.miss:.1f}m tca={w.tca:.1f}s" if w else "no threats"
        st_ = self.planner.last_stats
        return (f"mode={self.mode} spheres={len(self.corridor)} reach={self.reaches_goal} "
                f"wp={self.traj.idx}/{len(self.traj.waypoints)} fails={self.plan_failures} "
                f"exp={st_.get('expansions', 0)} {thr}"
                + (f" | {self.surface.warning}" if self.surface.warning else ""))


# =============================================================================
# Debug helpers
# =============================================================================

def log_corridor(corridor: list[Sphere], logger=print) -> None:
    for k, s in enumerate(corridor):
        logger(f"[{k}] c={np.round(s.center, 1)} r={s.radius:.1f} t={s.t_arrival:.1f}")


def log_corridor_csv(path: str, t: float, corridor: list[Sphere],
                     goal: npt.NDArray, reached: bool) -> None:
    """Append one row per sphere: t, index, cx, cy, cz, r, t_arrival, gx, gy, gz, reached."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["t", "k", "cx", "cy", "cz", "r", "t_arrival", "gx", "gy", "gz", "reached"])
        for k, s in enumerate(corridor):
            w.writerow([f"{t:.3f}", k, *np.round(s.center, 3), round(s.radius, 3),
                        round(s.t_arrival, 3), *np.round(goal, 3), int(reached)])