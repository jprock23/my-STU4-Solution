"""
Implementation of an Artificial Potenial Field based path planner. Treats the target as a negative charge that gives off
an attractive potential that pulls the shuttle towards it, while debris is positive charge that has a repulsive potential,
pushing the shuttle away
"""

from dataclasses import dataclass, field
from typing import List, Optional
 
import numpy as np
 
 
# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
 
@dataclass
class Obstacle:
    """One tracked debris object, in the same frame as the shuttle state."""
    position: np.ndarray            # (3,) m
    velocity: np.ndarray            # (3,) m/s  (zeros if unknown)
    radius: float                   # m, estimated physical radius
    position_sigma: float = 0.0     # m, uncertainty used to inflate the margin
 
 
@dataclass
class APFConfig:
    """Tuning parameters. Start here when tuning for time vs fuel."""
    # Attractive term
    cruise_speed: float = 10.0          # m/s, speed along the goal line
    goal_slowdown_radius: float = 50.0  # m, start slowing down inside this distance
 
    # Repulsive term
    shuttle_radius: float = 2.0         # m
    safety_margin: float = 3.0          # m, added to every obstacle
    influence_radius: float = 60.0      # m (d0), surface distance where repulsion starts
    k_rep: float = 50.0                  # repulsive gain
    prediction_horizon: float = 10.0    # s, max lookahead for closest approach
    deadband_distance: float = 15.0     # m, ignore obstacles whose closest approach is beyond this
 
    # Output shaping
    max_speed: float = 15.0             # m/s, cap on |v_ref|
    filter_alpha: float = 0.3           # 0..1, low-pass weight on the new value
 
    # Local-minimum handling
    stuck_progress_threshold: float = 0.5  # m/s, progress toward goal below this counts as stuck
    stuck_time_threshold: float = 2.0      # s, how long before tangential escape kicks in
    tangential_gain: float = 0.5
 
 
# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
 
def _clamp(val, lb, ub):
    """ Clamps val between a lower bound(lb) and an upper bound(ub)"""
    return max(lb, min(val, ub))


class APFPlanner:
    def __init__(self, goal: np.ndarray, config: Optional[APFConfig] = None):
        self.goal = np.asarray(goal, dtype=float)
        self.cfg = config or APFConfig()
 
        # Internal state
        self._v_ref_filtered = np.zeros(3)
        self._stuck_timer = 0.0
        self._escape_dir: Optional[np.ndarray] = None  # keep one escape direction while stuck
  
    def set_goal(self, goal: np.ndarray) -> None:
        """Update the goal (e.g. the moving return spacecraft, or a waypoint
        from a higher-level planner later on)."""
        self.goal = np.asarray(goal, dtype=float)
 
    def reset(self) -> None:
        """Clear filter and stuck-detection state (call at the start of a run)."""
        self._v_ref_filtered = np.zeros(3)
        self._stuck_timer = 0.0
        self._escape_dir: Optional[np.ndarray] = None
 
    def compute(self,
                pos: np.ndarray,
                vel: np.ndarray,
                obstacles: List[Obstacle],
                dt: float) -> np.ndarray:
        """Main entry point. Returns v_ref (3,) for the LQR to track.
 
        Suggested flow:
            v_att  = self._attractive(pos)
            v_rep  = sum(self._repulsive(pos, vel, o) for o in obstacles)
            v_cmd  = v_att + v_rep
            v_cmd += self._tangential_escape(pos, vel, v_att, v_rep, dt)
            v_cmd  = self._cap(v_cmd)
            return self._filter(v_cmd)
        """

        if self.goal is None:
            return np.zeros(3)

        v_att = self._attractive(pos)
        v_rep = sum((self._repulsive(pos, vel, obs) for obs in obstacles), np.zeros(3))
        v_cmd = v_att + v_rep + self._tangential_escape(pos, vel, v_att, v_rep, dt)
        v_cmd = self._cap(v_cmd)

        return self._filter(v_cmd)
    
    # ------------------------------------------------------- Attractive term
 
    def _attractive(self, pos: np.ndarray) -> np.ndarray:
        """Constant-speed pull toward the goal.
 
        Direction: unit vector from pos to goal.
        Speed:     cfg.cruise_speed, scaled down inside goal_slowdown_radius
                   so you arrive without overshooting.
        """
        dist = self.goal - pos
        d = np.linalg.norm(dist)
        if d < 1e-9:
            return np.zeros(3)
        return self.cfg.cruise_speed * min(1.0, d / self.cfg.goal_slowdown_radius) * (dist / d)

    # -------------------------------------------------------- Repulsive term
 
    def _closest_approach(self,
                          pos: np.ndarray,
                          vel: np.ndarray,
                          obs: Obstacle,
                          own_vel_guess: Optional[np.ndarray] = None):
        """Predict when and where the obstacle gets closest to the shuttle.
 
        Use relative position r = obs.position - pos and relative velocity
        v = obs.velocity - (own_vel_guess or vel).
        t* = clamp(-(r . v) / (v . v), 0, cfg.prediction_horizon)
 
        Returns:
            t_star   (float)        time of closest approach, s
            r_star   (np.ndarray)   relative position at t_star (obstacle - shuttle)
        """
        r = obs.position - pos # relative position
        v = obs.velocity - vel # relative velocity
        t_star = _clamp(-(np.dot(r, v)/(v@v)), 0, self.cfg.prediction_horizon) if v@v > 1e-9 else 0.0
        r_star = r + v * t_star

        return t_star, r_star
 
    def _surface_distance(self, center_dist: float, obs: Obstacle) -> float:
        """Center distance minus obstacle radius, shuttle radius,
        safety margin and an uncertainty inflation (e.g. 2 * position_sigma)."""
        return center_dist - (obs.radius + self.cfg.shuttle_radius + self.cfg.safety_margin + (2 * obs.position_sigma))
 
    def _repulsive(self,
                   pos: np.ndarray,
                   vel: np.ndarray,
                   obs: Obstacle) -> np.ndarray:
        """Repulsive velocity contribution from one obstacle.
 
        Steps:
            1. t*, r* = self._closest_approach(...)
            2. d = self._surface_distance(|r*|, obs)
            3. If d > cfg.deadband_distance or d > cfg.influence_radius: return 0
            4. magnitude ~ k_rep * (1/d - 1/d0)   (clamp d to a small positive value)
            5. direction = -r* / |r*|   (push away from the predicted position)
            6. Optionally weight by urgency, e.g. scale down for large t*.
        """
        t_star, r_star = self._closest_approach(pos, vel, obs)
        d = self._surface_distance(np.linalg.norm(r_star), obs)
        if d > self.cfg.deadband_distance or d > self.cfg.influence_radius:
            return np.zeros(3)

        d0 = min(self.cfg.influence_radius, self.cfg.deadband_distance)
        d = max(d, 0.1)
        mag = self.cfg.k_rep * (1/d - 1/d0)

        r_now = obs.position - pos
        radial = -r_now / max(np.linalg.norm(r_now), 1e-9)

        v_rel = obs.velocity - vel
        side = np.zeros(3)
        if v_rel @ v_rel > 1e-9:
            v_hat = v_rel / np.linalg.norm(v_rel)
            perp = r_star - (r_star @ v_hat) * v_hat
            if np.linalg.norm(perp) > 1e-3:
                side = -perp / np.linalg.norm(perp)
            else: # object is straight ahead, move sideways
                n = np.cross(v_hat, [0, 0, 1])
                if np.linalg.norm(n) < 1e-9:
                    n = np.cross(v_hat, [1, 0, 0])
                side = n / np.linalg.norm(n)

        direction = side + 0.5 * radial
        nd = np.linalg.norm(direction)
        direction = direction / nd if nd > 1e-9 else radial

        vel_rep = mag * direction * 1/(1 + t_star)

        return vel_rep
 
    # --------------------------------------------------- Local-minimum escape
 
    def _is_stuck(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> bool:
        """Update self._stuck_timer using progress toward the goal
        (vel . goal_dir < cfg.stuck_progress_threshold) and return True
        once it exceeds cfg.stuck_time_threshold."""
        to_goal = self.goal - pos
        dist = np.linalg.norm(to_goal)
        progress = vel @ (to_goal / dist) if dist > 1e-9 else 0.0
        if progress < self.cfg.stuck_progress_threshold and dist > self.cfg.goal_slowdown_radius:
            self._stuck_timer += dt
        else:
            self._stuck_timer = 0.0

        return self._stuck_timer > self.cfg.stuck_time_threshold 
        
 
    def _tangential_escape(self,
                           pos: np.ndarray,
                           vel: np.ndarray,
                           v_att: np.ndarray,
                           v_rep: np.ndarray,
                           dt: float) -> np.ndarray:
        """Sideways nudge when attractive and repulsive terms cancel.
 
        Direction: perpendicular to both v_att and v_rep, e.g.
            n = cross(v_att, v_rep); if |n| ~ 0, pick any vector perpendicular to v_att.
        Keep self._escape_dir fixed while stuck so you don't flip sides every tick.
        Clear it once progress resumes.
        """
        if not self._is_stuck(pos, vel, dt):
            self._escape_dir = None
            return np.zeros(3)

        if self._escape_dir is None:
            n = np.cross(v_att, v_rep)
            if np.linalg.norm(n) < 1e-9:
                n = np.cross(v_att, [0, 0, 1])
            if np.linalg.norm(n) < 1e-9:          # v_att was along z-axis
                n = np.cross(v_att, [1, 0, 0])
            self._escape_dir = n / np.linalg.norm(n)

        return self.cfg.tangential_gain * self.cfg.cruise_speed * self._escape_dir


            
    # ---------------------------------------------------------- Output shaping
 
    def _cap(self, v: np.ndarray) -> np.ndarray:
        """Limit |v| to cfg.max_speed, keeping its direction."""
        if np.sqrt(v @ v) <= self.cfg.max_speed:
            return v

        direction = v / np.linalg.norm(v)
        return self.cfg.max_speed * direction
 
    def _filter(self, v: np.ndarray) -> np.ndarray:
        """First-order low-pass:
            filtered = alpha * v + (1 - alpha) * filtered
        Store it in self._v_ref_filtered and return it."""

        self._v_ref_filtered = self.cfg.filter_alpha * v + (1 - self.cfg.filter_alpha) * self._v_ref_filtered
        return self._v_ref_filtered
 
    # ------------------------------------------------------------- Debugging
 
    def debug_info(self) -> dict:
        """Return the last tick's terms (v_att, v_rep, escape, stuck_timer,
        nearest obstacle distance) for logging/plots while tuning."""
        # TODO
        return {}
 
 
# ---------------------------------------------------------------------------
# Example usage inside the control loop (sketch)
# ---------------------------------------------------------------------------
#
# planner = APFPlanner(goal=target_pos)
# planner.reset()
#
# while running:
#     pos, vel = get_estimated_state()
#     obstacles = tracker.update(radar_returns)     # -> List[Obstacle]
#     planner.set_goal(get_target_position())       # if the target moves
#     v_ref = planner.compute(pos, vel, obstacles, dt)
#     thrust = lqr.compute(pos, vel, v_ref)         # LQR tracks v_ref
#     send_thrust(thrust)
 