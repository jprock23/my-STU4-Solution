"""
score_cruise.py - score-driven burn/coast governor for open-space flight.

Ported from the ideas in the teammate's AccelerationGuidance (score_cruise mode):

  * Every tick, price "burn 2 more seconds" against "coast from here" using the
    competition score, and keep burning only while the extra burn raises the score.
    There is NO fixed cruise speed: the speed falls out of the time/fuel trade.
  * Coasting is only allowed if a ballistic (gravity-only) prediction actually enters
    the 50 m success sphere, with a margin (8 m to start coasting, 4 m to keep coasting)
    so the decision doesn't chatter at the edge.
  * If a coast would miss sideways, fire a short sideways TRIM instead of more forward
    thrust. A trim never brakes forward progress.
  * While coasting, command EXACTLY zero thrust - no gravity feedforward. Gravity is
    already inside the arrival prediction, so cancelling it only burns fuel.
  * The fuel price per m/s of delta-v is learned from the sim's FuelUsed stat.

Use it only when the path is clear (your DIRECT mode). Near debris, keep your
sphere planner + tight LQR in charge.

Frames: any inertial-ish frame, as long as everything is in the same one
(Phobos frame in the main loop). Units: m, s, kg.
"""
from dataclasses import dataclass
import math

import numpy as np


# ---------------------------------------------------------------- scoring model
# Same formula the teammate validated against the local scorer. Double-check these
# three constants against the official rules if you have time.
NOMINAL_TIME = 120.0      # s
NOMINAL_FUEL = 75.0       # kg
MAX_DAMAGE = 209200.0     # J
SUCCESS_RADIUS = 50.0     # m


def score(elapsed, fuel, damage=0.0):
    time_score = NOMINAL_TIME / elapsed if elapsed > 0.0 else 0.0
    return 50.0 + 100.0 * (time_score - max(fuel / NOMINAL_FUEL, 0.0) - 2.0 * damage / MAX_DAMAGE)


# ----------------------------------------------------------- ballistic geometry
def _unit(v, fallback=(1.0, 0.0, 0.0)):
    n = float(np.linalg.norm(v))
    return np.asarray(v, float) / n if n > 1e-9 else np.array(fallback, float)


def _limited(v, maximum):
    v = np.asarray(v, float)
    return v * min(1.0, maximum / max(float(np.linalg.norm(v)), 1e-12))


def travel_time(distance, speed, accel):
    """Time to cover `distance` in 1-D from `speed` under constant `accel`."""
    if distance <= 0.0:
        return 0.0
    if abs(accel) < 1e-9:
        return distance / speed if speed > 1e-6 else math.inf
    disc = speed * speed + 2.0 * accel * distance
    if disc < 0.0:
        return math.inf
    roots = [(-speed + s * math.sqrt(disc)) / accel for s in (-1.0, 1.0)]
    return min((t for t in roots if t >= 0.0), default=math.inf)


def ballistic_entry(rel_pos, rel_vel, accel, radius, horizon):
    """
    First time a constant-acceleration path enters a sphere around the target.

    rel_pos = target - ship, rel_vel = ship_vel - target_vel, accel = acceleration
    of the ship relative to the target (gravity, plus thrust if burning).
    Solves |rel_pos - v t - a t^2/2|^2 = radius^2 (a quartic). inf if no entry.
    """
    r, v, a = (np.asarray(x, float) for x in (rel_pos, rel_vel, accel))
    if not np.all(np.isfinite(np.r_[r, v, a])) or radius <= 0.0 or horizon <= 0.0:
        return math.inf
    if r @ r <= radius * radius:
        return 0.0
    coeffs = np.array([0.25 * (a @ a), v @ a, v @ v - r @ a, -2.0 * (r @ v), r @ r - radius * radius])
    coeffs = np.trim_zeros(coeffs, "f")
    if len(coeffs) < 2:
        return math.inf
    roots = np.roots(coeffs)
    hits = [float(z.real) for z in roots if abs(z.imag) < 1e-6 and 0.0 <= z.real <= horizon]
    return min(hits, default=math.inf)


def finite_burn_entry(rel_pos, rel_vel, gravity, thrust, duration, radius, horizon=600.0):
    """Arrival time if we burn `thrust` for `duration` s and then coast."""
    r, v, g, a = (np.asarray(x, float) for x in (rel_pos, rel_vel, gravity, thrust))
    if np.linalg.norm(r) <= radius:
        return 0.0
    duration = min(max(float(duration), 0.0), horizon)
    during = ballistic_entry(r, v, g + a, radius, duration)
    if math.isfinite(during):
        return during
    r2 = r - v * duration - 0.5 * (g + a) * duration ** 2
    v2 = v + (g + a) * duration
    return duration + ballistic_entry(r2, v2, g, radius, horizon - duration)


def nonbraking(accel, rel_vel, rel_pos):
    """Remove any component of `accel` that would slow progress toward the target."""
    axis = _unit(rel_pos)
    cmd = accel - min(0.0, float(accel @ axis)) * axis
    if rel_vel @ axis > 0.0 and rel_vel @ rel_vel > 1e-8:
        cmd = cmd - min(0.0, float(cmd @ rel_vel)) / float(rel_vel @ rel_vel) * rel_vel
    return cmd


# --------------------------------------------------------------------- governor
@dataclass
class CruiseDecision:
    mode: str              # "BURN" | "COAST" | "TRIM" | "ARRIVED"
    accel: np.ndarray      # world-frame thrust acceleration to command (no gravity term)
    eta: float             # predicted seconds until entering the success sphere
    advantage: float       # score(burn 2 s more) - score(coast now)
    reason: str


class CruiseGovernor:
    """
    Parameters
    ----------
    burn_accel      m/s^2 used for main and trim burns. The teammate used
                    min(0.12, 0.5 * MaxAccel_m_s2) - their measurements showed the jet
                    allocator gets less efficient above that.
    fuel_per_dv     kg per m/s, first guess. Ideal is mass / (80 * 9.8); the teammate
                    starts at 1.3x ideal. learn_fuel() refines it from FuelUsed.
    burn_lookahead  how many seconds of extra burn to price each tick (teammate: 2 s).
    """

    def __init__(self, burn_accel, fuel_per_dv, burn_lookahead=2.0,
                 enter_margin=8.0, hold_margin=4.0, latch_hysteresis=0.05,
                 fuel_reserve_kg=1.0, trim_max_fraction=0.58, ideal_fuel_per_dv=None):
        self.burn_accel = float(burn_accel)
        self.fuel_per_dv = float(fuel_per_dv)
        # Floor for the learned value (the physical minimum, mass / (80 * 9.8)).
        self.ideal_fuel_per_dv = float(ideal_fuel_per_dv if ideal_fuel_per_dv is not None
                                       else fuel_per_dv / 1.3)
        self.burn_lookahead = burn_lookahead
        self.enter_margin, self.hold_margin = enter_margin, hold_margin
        self.latch_hysteresis = latch_hysteresis
        self.fuel_reserve_kg = fuel_reserve_kg
        self.trim_max_fraction = trim_max_fraction

        self.coast_latched = False      # "score says stop burning", with hysteresis
        self.coasting = False           # ballistic path enters the sphere -> zero thrust
        self.last = None
        # fuel learning accumulators
        self._fuel_mark = None
        self._dv_since_mark = 0.0

    # ----------------------------------------------------------- score cutoff
    def _score_says_coast(self, r, v, g, now, fuel, entry, radius):
        axis = _unit(r)
        # If an unpowered path misses laterally, price forward motion on the radial
        # projection; the sideways miss is handled by a trim, not by more speed.
        if not math.isfinite(entry):
            r = np.linalg.norm(r) * axis
            v = (v @ axis) * axis
            g = (g @ axis) * axis
            entry = travel_time(np.linalg.norm(r) - radius, v @ axis, g @ axis)
        if not math.isfinite(entry) or entry > 600.0 or v @ axis <= 0.0:
            return False, -math.inf, "acquire closing speed"
        if fuel + self.fuel_per_dv + self.fuel_reserve_kg >= NOMINAL_FUEL:
            return True, -math.inf, "fuel reserve"
        duration = min(self.burn_lookahead, entry)
        thrust = self.burn_accel * axis
        powered = finite_burn_entry(r, v, g, thrust, duration, radius)
        if not math.isfinite(powered):
            return True, -math.inf, "burn does not help"
        extra_fuel = self.fuel_per_dv * self.burn_accel * min(duration, powered)
        advantage = score(now + powered, fuel + extra_fuel) - score(now + entry, fuel)
        coast = advantage <= (self.latch_hysteresis if self.coast_latched else 0.0)
        return coast, advantage, ("score favors coast" if coast else "score favors burn")

    # ------------------------------------------------------------------ main
    def update(self, now, rel_pos, rel_vel, gravity, fuel_used):
        """
        now        mission seconds since sim start (same clock the scorer uses)
        rel_pos    target_pos - ship_pos
        rel_vel    ship_vel - target_vel
        gravity    gravitational acceleration at the ship (same frame)
        fuel_used  kg, from shuttle Stats/FuelUsed
        """
        r = np.asarray(rel_pos, float)
        v = np.asarray(rel_vel, float)
        g = np.asarray(gravity, float)
        dist = float(np.linalg.norm(r))

        if dist <= SUCCESS_RADIUS:
            self.coasting = True
            self.last = CruiseDecision("ARRIVED", np.zeros(3), 0.0, 0.0, "arrived")
            return self.last

        # Hysteresis on the capture test: need 8 m of margin to start coasting,
        # keep coasting while there's still 4 m.
        capture = SUCCESS_RADIUS - (self.hold_margin if self.coasting else self.enter_margin)
        entry = ballistic_entry(r, v, g, capture, 600.0)
        approaching = float(r @ v) > 0.0

        self.coast_latched, advantage, reason = self._score_says_coast(
            r, v, g, now, fuel_used, entry, capture)

        # 1) Score says stop AND gravity-only path arrives -> pure coast.
        self.coasting = self.coast_latched and approaching and math.isfinite(entry)
        if self.coasting:
            self.last = CruiseDecision("COAST", np.zeros(3), entry, advantage, reason)
            return self.last

        axis = _unit(r)
        speed = float(v @ axis)
        g_along = float(g @ axis)
        lat_v = v - speed * axis
        lat_g = g - g_along * axis

        # 2) Score says stop, but the coast misses sideways -> sideways trim only.
        if self.coast_latched:
            eta = travel_time(max(0.0, dist - capture), speed, g_along)
            eta = float(np.clip(eta, 1.0, 600.0))
            dv = nonbraking(-(lat_v + 0.5 * lat_g * eta), v, r)
            accel = self.burn_accel * _unit(dv) if np.linalg.norm(dv) > 1e-6 else np.zeros(3)
            self.last = CruiseDecision("TRIM", accel, eta, advantage, "lateral miss")
            return self.last

        # 3) Score says keep burning -> full burn toward the target, plus a bounded
        #    sideways correction folded into the same burn so it never brakes.
        eta = float(np.clip(travel_time(max(0.0, dist - capture), speed,
                                        g_along + self.burn_accel), 1.0, 600.0))
        accel = self.burn_accel * axis
        corr_eta = min(eta, 120.0)
        lateral_miss = lat_v * corr_eta + 0.5 * lat_g * corr_eta ** 2
        if np.linalg.norm(lateral_miss) > capture:
            correction = -(lat_v + 0.5 * lat_g * corr_eta) / max(3.0, 0.25 * corr_eta)
            accel = accel + _limited(correction, self.trim_max_fraction * self.burn_accel)
        accel = nonbraking(_limited(accel, self.burn_accel), v, r)
        self.last = CruiseDecision("BURN", accel, eta, advantage, reason)
        return self.last

    # ------------------------------------------------------------ fuel model
    def note_command(self, accel_world, dt):
        """Call every tick with the thrust you actually commanded (any mode)."""
        self._dv_since_mark += float(np.linalg.norm(accel_world)) * dt

    def learn_fuel(self, fuel_used):
        """
        Call about once a second with FuelUsed. Refines kg per m/s from real burns.
        Clipped to [ideal, 3 x ideal] and smoothed so one odd window can't wreck it.
        """
        if self._fuel_mark is None:
            self._fuel_mark, self._dv_since_mark = fuel_used, 0.0
            return
        fuel_delta = fuel_used - self._fuel_mark
        if self._dv_since_mark > 0.03 and fuel_delta > 0.0:
            measured = float(np.clip(fuel_delta / self._dv_since_mark,
                                     self.ideal_fuel_per_dv, 3.0 * self.ideal_fuel_per_dv))
            self.fuel_per_dv = 0.9 * self.fuel_per_dv + 0.1 * measured
        self._fuel_mark, self._dv_since_mark = fuel_used, 0.0