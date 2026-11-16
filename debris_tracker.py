"""
Debris tracker for Escape from Phobos.

Turns raw radar frames into a List[Obstacle] with estimated velocities,
which feeds APFPlanner.compute(). Replaces radar_to_obstacles().

Pipeline per radar frame:
    SortedDebrisData (shuttle frame) -> convert to Phobos frame -> Detections
    existing tracks -> predict forward by dt
    detections <-> predicted tracks -> gated nearest-neighbor association
    matched   -> alpha-beta update (position + velocity)
    unmatched detections -> new tentative tracks (velocity seeded from range rate)
    unmatched tracks     -> miss count++, grow uncertainty, drop if too many misses
    confirmed tracks -> List[Obstacle]

Everything inside the tracker is in the Phobos frame (the same frame as
the shuttle `pos` / `vel` you give the planner). Never difference positions
in the shuttle frame: that frame moves and rotates with you.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from apf_planner import Obstacle


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """One radar return for one frame, already converted to the Phobos frame."""
    position: np.ndarray            # (3,) m, Phobos frame
    range_rate: float               # m/s, from d[1, 1] (check the sign convention)
    radius: float                   # m, 0.5 * d[1, 2]
    los_unit: np.ndarray            # (3,) unit vector shuttle -> rock, Phobos frame


@dataclass
class Track:
    """One debris object followed across frames."""
    track_id: int
    position: np.ndarray            # (3,) m, Phobos frame (filtered)
    velocity: np.ndarray            # (3,) m/s, Phobos frame (filtered)
    radius: float                   # m, smoothed over hits
    position_sigma: float           # m, grows on misses, shrinks on hits
    hits: int = 1                   # total frames this track was matched
    misses: int = 0                 # consecutive frames with no match
    age: float = 0.0                # s since the track was created
    confirmed: bool = False         # True once hits >= cfg.confirm_hits


@dataclass
class TrackerConfig:
    """Tuning parameters for the tracker."""
    # Detection filtering
    max_range: float = 150.0            # m, ignore returns farther than this (list is nearest-first)

    # Association
    gate_distance: float = 5.0          # m, max distance between prediction and detection to match
    gate_sigma_scale: float = 3.0       # widen the gate by this * track.position_sigma

    # Alpha-beta filter gains (0..1)
    alpha: float = 0.5                  # position correction weight
    beta: float = 0.2                   # velocity correction weight (scaled by 1/dt)

    # Track lifecycle
    confirm_hits: int = 2               # hits before a track is handed to the planner
    max_misses: int = 3                 # consecutive misses before a track is deleted
    coast_confirmed_tracks: bool = True # keep giving the planner coasted tracks while missing

    # Uncertainty
    initial_sigma: float = 2.0          # m, sigma for a brand-new track
    min_sigma: float = 0.5              # m, floor after many hits
    sigma_growth_per_miss: float = 1.0  # m, added each missed frame
    sigma_decay: float = 0.8            # multiply sigma by this on each hit

    # Velocity seeding for new tracks
    use_range_rate_seed: bool = True    # seed new-track velocity from range rate along line of sight
    range_rate_closing_negative: bool = True  # True if negative range rate means closing

    # Radius smoothing
    radius_alpha: float = 0.3           # low-pass weight on the new radius measurement


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class DebrisTracker:
    def __init__(self,
                 to_phobos_frame: Callable[[np.ndarray], np.ndarray],
                 config: Optional[TrackerConfig] = None):
        """
        to_phobos_frame: function that rotates a shuttle-frame vector into
            Phobos-frame axes (no translation), e.g.
                lambda v: np.asarray(
                    st.frames.FramedVector(v, shuttleFrame).ExprIn(phobosFrame), float)
            Passed in so this file doesn't depend on the sim's API and can be
            tested offline with a fake (e.g. identity).
        """
        self.to_phobos_frame = to_phobos_frame
        self.cfg = config or TrackerConfig()

        # Internal state
        self._tracks: Dict[int, Track] = {}
        self._next_id = 0
        self._last_time: Optional[float] = None
        self._last_debug: dict = {}

    # ------------------------------------------------------------------ API

    def reset(self) -> None:
        """Clear all tracks and timing state (call at the start of a run)."""
        self._tracks = {}
        self._next_id = 0
        self._last_time = 0.0
        self._last_debug = {}
        self._last_time = None

    def update(self,
               debris_info,
               own_pos: np.ndarray,
               own_vel: np.ndarray,
               t: float) -> List[Obstacle]:
        """Main entry point. Call once per radar frame.

        debris_info: SortedDebrisData from the sim, each entry a 3x3 matrix
                     (row 0 = location in shuttle frame,
                      row 1 = [closest point, range rate, diameter]).
        own_pos, own_vel: shuttle state in the Phobos frame.
        t: sim time in seconds (use sim time, not wall clock).

        Suggested flow:
            dt = self._compute_dt(t)
            detections = self._convert_detections(debris_info, own_pos)
            self._predict(dt)
            matches, unmatched_dets, unmatched_tracks = self._associate(detections)
            for tid, di in matches:        self._update_track(self._tracks[tid], detections[di], dt)
            for di in unmatched_dets:      self._spawn_track(detections[di], own_vel)
            for tid in unmatched_tracks:   self._mark_missed(self._tracks[tid])
            self._prune()
            return self._to_obstacles()
        """
        detections = self._convert_detections(debris_info, own_pos)
        dt = self._compute_dt(t)
        if dt > 0:
            self._predict(dt)
        matches, unmatched_dets, unmatched_tracks = self._associate(detections)

        for t_id, d_i in matches:
            self._update_track(self._tracks[t_id], detections[d_i], dt)
        for d_i in unmatched_dets:
            self._spawn_track(detections[d_i], own_vel)
        for t_id in unmatched_tracks:
            self._mark_missed(self._tracks[t_id])

        self._prune(own_pos)
        return self._to_obstacles()

    def tracks(self) -> List[Track]:
        """All current tracks (confirmed and tentative), for logging/plots."""

        return list(self._tracks.values())

    # ------------------------------------------------------------ Timing

    def _compute_dt(self, t: float) -> float:
        """Time since the last update. First call returns 0.
        Guard against dt <= 0 (repeated or out-of-order frames): skip
        prediction and the velocity update in that case."""
        if self._last_time == None:
            self._last_time = t
            return 0.0        
        dt = t - self._last_time
        self._last_time = t
        return dt

    # ------------------------------------------------------- Detections

    def _convert_detections(self,
                            debris_info,
                            own_pos: np.ndarray) -> List[Detection]:
        """Turn raw radar entries into Detections in the Phobos frame.

        For each entry d:
            loc_body = d[0, :]
            stop once |loc_body| > cfg.max_range (list is sorted nearest-first)
            offset   = self.to_phobos_frame(loc_body)
            position = own_pos + offset
            los_unit = offset / |offset|
            radius   = 0.5 * d[1, 2]
            range_rate = d[1, 1]
        """
        detections = []
        for d in debris_info:
            loc_body = d[0, :]
            if np.linalg.norm(loc_body) >= self.cfg.max_range:
                break
            offset = self.to_phobos_frame(loc_body)
            position = own_pos + offset
            los_unit = offset / np.linalg.norm(offset)
            radius = 0.5 * d[1,2]
            range_rate = d[1,1]
            detections.append(Detection(position, range_rate, radius, los_unit))
        return detections

    # ------------------------------------------------------- Prediction

    def _predict(self, dt: float) -> None:
        """Constant-velocity prediction for every track:
            position += velocity * dt
            age      += dt
        Optionally grow position_sigma a little with dt here too."""

        for track in self._tracks.values():
            track.position += track.velocity * dt
            track.age += dt

    # ------------------------------------------------------ Association

    def _gate(self, track: Track) -> float:
        """Match radius for this track:
            cfg.gate_distance + cfg.gate_sigma_scale * track.position_sigma
        Tracks that have been missing for a while get a wider gate."""
        return self.cfg.gate_distance + self.cfg.gate_sigma_scale * track.position_sigma


    def _associate(self,
                   detections: List[Detection]
                   ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy gated nearest-neighbor.

        1. Build all (distance, track_id, det_index) pairs with
           distance = |track.position - det.position| <= self._gate(track).
           Optionally also require similar radius (e.g. within 50%).
        2. Sort by distance, take pairs in order, skipping any track or
           detection that is already used.

        Returns:
            matches            list of (track_id, det_index)
            unmatched_dets     list of det_index
            unmatched_tracks   list of track_id
        """

        pairs = []

        for t_id, track in self._tracks.items():
            gate = self._gate(track)
            for d_id, det in enumerate(detections):
                dist = np.linalg.norm(track.position - det.position)
                if dist > gate:
                    continue
                r_small = min(track.radius, max(det.radius, 1e-6))
                r_big = max(track.radius, det.radius)

                if r_big/r_small > 1.5:
                    continue
                pairs.append((dist, t_id, d_id))

        pairs.sort(key=lambda p: p[0])
        matches = []
        used_tracks, used_dets = set(), set()

        for dist, t_id, d_id in pairs:
            if t_id in used_tracks or d_id in used_dets:
                continue
            matches.append((t_id, d_id))
            used_tracks.add(t_id)
            used_dets.add(d_id)

        unmatched_dets = [d_id for d_id in range(len(detections)) if d_id not in used_dets]
        unmatched_tracks = [t_id for t_id in self._tracks if t_id not in used_tracks]

        return matches, unmatched_dets, unmatched_tracks

    # ----------------------------------------------------------- Update

    def _update_track(self, track: Track, det: Detection, dt: float) -> None:
        """Alpha-beta correction with a matched detection.

            residual  = det.position - track.position        (track already predicted)
            position += cfg.alpha * residual
            velocity += (cfg.beta / dt) * residual           (skip if dt <= 0)
            radius    = radius_alpha * det.radius + (1 - radius_alpha) * radius
            sigma     = max(cfg.min_sigma, sigma * cfg.sigma_decay)
            hits += 1, misses = 0
            confirmed = hits >= cfg.confirm_hits
        """
        residual = det.position - track.position
        track.position += self.cfg.alpha * residual
        if dt > 0:
            track.velocity += (self.cfg.beta / dt) * residual
        track.radius = self.cfg.radius_alpha * det.radius + (1 - self.cfg.radius_alpha) * track.radius
        track.position_sigma = max(self.cfg.min_sigma, track.position_sigma * self.cfg.sigma_decay)
        track.hits += 1
        track.misses = 0
        track.confirmed = track.hits >= self.cfg.confirm_hits


    def _range_rate_velocity(self,
                             det: Detection,
                             own_vel: np.ndarray) -> np.ndarray:
        """Velocity guess for a brand-new track from range rate only.

            v_rock ~ own_vel + range_rate * los_unit

        Only the line-of-sight component is real; the sideways part is just
        own_vel. Flip the sign of range_rate if cfg.range_rate_closing_negative
        doesn't match the sim. Return zeros if cfg.use_range_rate_seed is False.
        """
        if not self.cfg.use_range_rate_seed:
            return np.zeros(3)

        rr = det.range_rate if self.cfg.range_rate_closing_negative else -det.range_rate

        return own_vel + rr * det.los_unit


    def _spawn_track(self, det: Detection, own_vel: np.ndarray) -> None:
        """Create a tentative Track from an unmatched detection."""
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = Track(
            track_id=tid,
            position=det.position.copy(),
            velocity=self._range_rate_velocity(det, own_vel),
            radius=det.radius,
            position_sigma=self.cfg.initial_sigma,
            confirmed=self.cfg.confirm_hits <= 1,
        )


    def _mark_missed(self, track: Track) -> None:
        """No detection this frame: misses += 1, sigma += cfg.sigma_growth_per_miss.
        The track keeps coasting on its predicted position."""
        track.misses += 1
        track.position_sigma += self.cfg.sigma_growth_per_miss


    def _prune(self, own_pos:  np.ndarray) -> None:
        """Delete tracks with misses > cfg.max_misses.
        Also delete tentative tracks that missed even once (likely noise),
        and optionally tracks far beyond cfg.max_range from the shuttle."""
        to_delete = []

        for t_id, track in self._tracks.items():
            misses_flag = track.misses > self.cfg.max_misses
            dist_flag = np.linalg.norm(track.position - own_pos) > 1.5 * self.cfg.max_range
            tent_flag = (not track.confirmed) and track.misses > 0
            if misses_flag or dist_flag or tent_flag:
                to_delete.append(t_id)

        for t_id in to_delete:
            del self._tracks[t_id]

        self._last_debug["deleted"] = len(to_delete)

    # ----------------------------------------------------------- Output

    def _to_obstacles(self) -> List[Obstacle]:
        """Confirmed tracks -> Obstacle list for the planner.
        Include coasting tracks (misses > 0) if cfg.coast_confirmed_tracks,
        their larger position_sigma already inflates the planner's margin."""

        obstacles = []
        for track in self._tracks.values():
            if not track.confirmed:
                continue
            if track.misses > 0 and not self.cfg.coast_confirmed_tracks:
                continue
            obstacles.append(Obstacle(position=track.position,
                                    velocity=track.velocity,
                                    radius=track.radius,
                                    position_sigma=track.position_sigma))
        return obstacles
    # -------------------------------------------------------- Debugging
   
    def _record_debug(self, t, dt, own_pos, detections, matches,
                      unmatched_dets, unmatched_tracks, deleted, obstacles) -> None:
        """Store this frame's stats. Called at the end of update()."""
        confirmed = [tr for tr in self._tracks.values() if tr.confirmed]
        coasting = [tr for tr in confirmed if tr.misses > 0]
 
        nearest_dist = float("inf")
        nearest_speed = 0.0
        nearest_closing = 0.0
        nearest_id = -1
        for tr in confirmed:
            rel = tr.position - own_pos
            center_dist = np.linalg.norm(rel)
            surface_dist = center_dist - tr.radius
            if surface_dist < nearest_dist:
                nearest_dist = surface_dist
                nearest_speed = float(np.linalg.norm(tr.velocity))
                # rock velocity along the line of sight (negative = moving toward the shuttle's position)
                nearest_closing = float(tr.velocity @ rel / max(center_dist, 1e-9))
                nearest_id = tr.track_id
 
        self._last_debug = {
            "t": t,
            "dt": dt,
            "n_detections": len(detections),
            "n_matched": len(matches),
            "n_new": len(unmatched_dets),
            "n_missed": len(unmatched_tracks),
            "n_deleted": deleted,
            "n_tracks": len(self._tracks),
            "n_confirmed": len(confirmed),
            "n_coasting": len(coasting),
            "n_obstacles": len(obstacles),
            "nearest_id": nearest_id,
            "nearest_surface_dist": nearest_dist,
            "nearest_speed": nearest_speed,
            "nearest_radial_vel": nearest_closing,
        }
 
    def debug_info(self) -> dict:
        """Last frame's stats for logging. Keys:
            t, dt                  sim time and time since the previous frame
            n_detections           radar returns inside max_range
            n_matched              returns matched to an existing track
            n_new                  returns that started a new track
            n_missed               tracks with no return this frame
            n_deleted              tracks removed by _prune
            n_tracks               all tracks (tentative + confirmed)
            n_confirmed            confirmed tracks
            n_coasting             confirmed tracks currently missing
            n_obstacles            obstacles handed to the planner
            nearest_id             track ID of the closest confirmed rock (-1 if none)
            nearest_surface_dist   m, center distance minus rock radius
            nearest_speed          m/s, |velocity| of that rock (Phobos frame)
            nearest_radial_vel     m/s, its velocity along the line of sight
        Once the tracker settles, n_new should be ~0 except when rocks come
        into range. A steady n_new > 0 means the gate is too tight.
        """
        return dict(self._last_debug)
 
    def format_debug(self) -> str:
        """One-line summary for your existing log line."""
        d = self._last_debug
        if not d:
            return "tracker: no data"
        return (f"det={d['n_detections']} match={d['n_matched']} new={d['n_new']} "
                f"miss={d['n_missed']} del={d['n_deleted']} conf={d['n_confirmed']} "
                f"coast={d['n_coasting']} near={d['nearest_surface_dist']:.1f}m "
                f"v={d['nearest_speed']:.2f}m/s")

# ---------------------------------------------------------------------------
# Example usage inside the control loop (sketch)
# ---------------------------------------------------------------------------
#
# to_phobos = lambda v: np.asarray(
#     st.frames.FramedVector(v, shuttleFrame).ExprIn(phobosFrame), float)
# tracker = DebrisTracker(to_phobos, TrackerConfig(max_range=150.0))
# tracker.reset()
# planner.reset()
#
# while running:
#     pos, vel = get_estimated_state()                         # Phobos frame
#     debris_info = <SortedDebrisData from the radar>
#     obstacles = tracker.update(debris_info, pos, vel, sim_time)
#     planner.set_goal(tgt_pos)
#     v_ref = planner.compute(pos, vel, obstacles, dt_est)
#     v_cmd = tgt_vel + v_ref
#     accel_world = lqr.compute(pos, vel, pos, v_cmd, g, dt=dt_est)
#
# Offline test idea: simulate a few rocks moving in straight lines, feed
# their positions (plus noise, with random dropouts) through update() with
# to_phobos_frame = identity and own_pos = 0, and check that each estimated
# velocity converges to the true one within ~5 frames and track IDs don't
# swap when two rocks pass close to each other.