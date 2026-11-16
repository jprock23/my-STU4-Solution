"""
Offline tests for DebrisTracker (no sim needed).

A fake radar generates SortedDebrisData-style 3x3 entries from simulated
rocks, with noise, dropouts and a moving/rotating shuttle, and the tests
check what the tracker does with them.

Run:
    python test_debris_tracker.py            # all tests, prints PASS/FAIL
    python test_debris_tracker.py -v         # also prints per-test details
    python test_debris_tracker.py --plot     # plus a plot of one scenario (needs matplotlib)
    pytest test_debris_tracker.py            # also works

Set RADAR_DT to the radar period you see in the sim before trusting the
numbers: the tracker's behavior (velocity noise, gate size) depends on it.
"""

import sys
from dataclasses import dataclass
from typing import List

import numpy as np

from debris_tracker import DebrisTracker, TrackerConfig
from apf_planner import Obstacle


RADAR_DT = 0.1        # s between radar frames  <-- set from your sim's timing log

# Settings applied to every test, on top of TrackerConfig's defaults.
# These are the values the tests suggest for a 10 Hz radar with ~0.2 m noise;
# copy them into TrackerConfig once they also behave in the sim.
#   beta 0.2 -> 0.05: beta/dt turns position noise into velocity noise, so a
#       fast radar needs a small beta (0.2 at 10 Hz gave ~1 m/s velocity error).
#   max_misses 3 -> 6: 3 frames at 10 Hz is only 0.3 s of coasting.
#   sigma_growth_per_miss 1.0 -> 0.3: 1 m per frame widens the gate by
#       3 m per missed frame, which lets a coasting track grab a neighbor.
TUNING = dict(beta=0.05, max_misses=6, sigma_growth_per_miss=0.3)
VERBOSE = "-v" in sys.argv


# ---------------------------------------------------------------------------
# Fake world
# ---------------------------------------------------------------------------

@dataclass
class Rock:
    p0: np.ndarray          # m, Phobos frame at t = 0
    v: np.ndarray           # m/s, Phobos frame
    radius: float           # m

    def pos(self, t):
        return self.p0 + self.v * t


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class FakeWorld:
    """Rocks plus a shuttle that can translate and spin about z.

    radar(t) returns a list of 3x3 arrays like SortedDebrisData:
        row 0 = rock center in the shuttle (body) frame
        row 1 = [closest point (unused here), range rate, diameter]
        row 2 = zeros
    sorted nearest-first.
    """

    def __init__(self, rocks: List[Rock],
                 shuttle_p0=(0.0, 0.0, 0.0), shuttle_v=(0.0, 0.0, 0.0),
                 spin_rate=0.0, pos_noise=0.0, rr_noise=0.0,
                 dropout=0.0, closing_negative=True, seed=0):
        self.rocks = rocks
        self.shuttle_p0 = np.asarray(shuttle_p0, float)
        self.shuttle_v = np.asarray(shuttle_v, float)
        self.spin_rate = spin_rate            # rad/s about z
        self.pos_noise = pos_noise            # m, per axis
        self.rr_noise = rr_noise              # m/s
        self.dropout = dropout                # probability a return is missing
        self.closing_negative = closing_negative
        self.rng = np.random.default_rng(seed)
        self.R = np.eye(3)                    # current body -> Phobos rotation

    # The tracker calls this. It reads the rotation for the current frame,
    # like FramedVector(...).ExprIn(phobosFrame) does in the sim.
    def to_phobos_frame(self, v_body):
        return self.R @ np.asarray(v_body, float)

    def shuttle_pos(self, t):
        return self.shuttle_p0 + self.shuttle_v * t

    def radar(self, t):
        self.R = rot_z(self.spin_rate * t)
        own_pos = self.shuttle_pos(t)
        entries = []
        for rock in self.rocks:
            if self.rng.random() < self.dropout:
                continue
            rel = rock.pos(t) - own_pos
            rng = np.linalg.norm(rel)
            rr = (rock.v - self.shuttle_v) @ rel / rng
            if not self.closing_negative:
                rr = -rr
            rr += self.rng.normal(0.0, self.rr_noise) if self.rr_noise else 0.0
            body = self.R.T @ rel
            if self.pos_noise:
                body = body + self.rng.normal(0.0, self.pos_noise, 3)
            entries.append(np.array([body,
                                     [rng - rock.radius, rr, 2 * rock.radius],
                                     [0.0, 0.0, 0.0]]))
        entries.sort(key=lambda e: np.linalg.norm(e[0]))
        return entries


def run(world: FakeWorld, tracker: DebrisTracker, duration: float, dt=RADAR_DT,
        on_frame=None):
    """Step the world and the tracker. on_frame(t, obstacles) is called each frame."""
    tracker.reset()
    obstacles = []
    n = int(round(duration / dt))
    for k in range(n + 1):
        t = k * dt
        debris = world.radar(t)
        obstacles = tracker.update(debris, world.shuttle_pos(t), world.shuttle_v, t)
        if on_frame:
            on_frame(t, obstacles)
    return obstacles


def nearest_track(tracker, p, confirmed_only=True):
    best, best_d = None, float("inf")
    for tr in tracker.tracks():
        if confirmed_only and not tr.confirmed:
            continue
        d = np.linalg.norm(tr.position - p)
        if d < best_d:
            best, best_d = tr, d
    return best, best_d


def make_tracker(world, **cfg):
    return DebrisTracker(world.to_phobos_frame, TrackerConfig(**{**TUNING, **cfg}))


def log(msg):
    if VERBOSE:
        print("      " + msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_stationary_rock():
    """One still rock, no noise: one track, confirmed on frame 2, velocity ~0."""
    world = FakeWorld([Rock(np.array([40.0, 5.0, 0.0]), np.zeros(3), 3.0)])
    tr = make_tracker(world)

    tr.reset()
    tr.update(world.radar(0.0), world.shuttle_pos(0.0), world.shuttle_v, 0.0)
    assert len(tr.tracks()) == 1, "first frame should create one track"
    assert not tr.tracks()[0].confirmed, "track should be tentative after one hit"
    obs = tr.update(world.radar(RADAR_DT), world.shuttle_pos(RADAR_DT),
                    world.shuttle_v, RADAR_DT)
    assert len(obs) == 1, "track should be confirmed and output after two hits"

    run(world, tr, 2.0)
    assert len(tr.tracks()) == 1, f"expected 1 track, got {len(tr.tracks())}"
    v = tr.tracks()[0].velocity
    log(f"velocity = {v}")
    assert np.linalg.norm(v) < 1e-6, f"stationary rock has velocity {v}"


def test_velocity_convergence():
    """Several rocks in straight lines with noise: velocities converge."""
    rng = np.random.default_rng(1)
    rocks = [Rock(rng.uniform([30, -40, -20], [90, 40, 20]),
                  rng.uniform(-3, 3, 3), rng.uniform(1, 4)) for _ in range(6)]
    world = FakeWorld(rocks, pos_noise=0.2, seed=2)
    tr = make_tracker(world)
    run(world, tr, 3.0)

    t_end = 3.0
    assert len(tr.tracks()) == len(rocks), \
        f"expected {len(rocks)} tracks, got {len(tr.tracks())} (duplicates or losses)"
    for i, rock in enumerate(rocks):
        track, d = nearest_track(tr, rock.pos(t_end))
        v_err = np.linalg.norm(track.velocity - rock.v)
        log(f"rock {i}: pos err {d:.2f} m, vel err {v_err:.2f} m/s")
        assert d < 1.0, f"rock {i}: position error {d:.2f} m"
        assert v_err < 0.5, f"rock {i}: velocity error {v_err:.2f} m/s"


def test_range_rate_seed_head_on():
    """Rock coming straight at the shuttle: first-frame velocity is exact."""
    rock = Rock(np.array([60.0, 0.0, 0.0]), np.array([-4.0, 0.0, 0.0]), 2.0)
    world = FakeWorld([rock])
    tr = make_tracker(world)
    tr.reset()
    tr.update(world.radar(0.0), world.shuttle_pos(0.0), world.shuttle_v, 0.0)
    v = tr.tracks()[0].velocity
    log(f"seed velocity = {v}, true = {rock.v}")
    assert np.allclose(v, rock.v, atol=1e-6), f"seed {v} != true {rock.v}"


def test_range_rate_sign_flag():
    """If the radar reports closing as positive, the config flag fixes it."""
    rock = Rock(np.array([60.0, 0.0, 0.0]), np.array([-4.0, 0.0, 0.0]), 2.0)

    world = FakeWorld([rock], closing_negative=False)
    tr = make_tracker(world, range_rate_closing_negative=False)
    tr.reset()
    tr.update(world.radar(0.0), world.shuttle_pos(0.0), world.shuttle_v, 0.0)
    v_good = tr.tracks()[0].velocity

    tr_wrong = make_tracker(world, range_rate_closing_negative=True)
    tr_wrong.reset()
    tr_wrong.update(world.radar(0.0), world.shuttle_pos(0.0), world.shuttle_v, 0.0)
    v_bad = tr_wrong.tracks()[0].velocity

    log(f"flag right: {v_good}, flag wrong: {v_bad}")
    assert np.allclose(v_good, rock.v, atol=1e-6), f"flag set: seed {v_good}"
    assert not np.allclose(v_bad, rock.v, atol=0.1), "wrong flag should give the wrong seed"


def test_rotating_moving_shuttle():
    """Shuttle translating and spinning: a still rock must stay still in the
    Phobos frame, with a single track. Catches frame-conversion mistakes."""
    rock = Rock(np.array([50.0, 20.0, 5.0]), np.zeros(3), 3.0)
    world = FakeWorld([rock], shuttle_v=(2.0, 0.5, 0.0),
                      spin_rate=np.radians(30), pos_noise=0.1, seed=3)
    tr = make_tracker(world)
    new_after_settle = []

    def on_frame(t, obs):
        if t > 0.5:
            new_after_settle.append(tr.debug_info()["n_new"])

    run(world, tr, 4.0, on_frame=on_frame)
    track, d = nearest_track(tr, rock.pos(4.0))
    log(f"pos err {d:.2f} m, |v| {np.linalg.norm(track.velocity):.2f} m/s, "
        f"tracks {len(tr.tracks())}")
    assert len(tr.tracks()) == 1, f"expected 1 track, got {len(tr.tracks())}"
    assert sum(new_after_settle) == 0, "new tracks spawned after settling (gate too tight or frame bug)"
    assert np.linalg.norm(track.velocity) < 0.5, \
        f"still rock looks like it moves at {track.velocity}"


def test_dropouts():
    """30% of returns missing: tracks coast through gaps, no duplicates pile up."""
    rng = np.random.default_rng(4)
    rocks = [Rock(rng.uniform([30, -40, -20], [90, 40, 20]),
                  rng.uniform(-2, 2, 3), rng.uniform(1, 4)) for _ in range(5)]
    world = FakeWorld(rocks, pos_noise=0.2, dropout=0.3, seed=5)
    tr = make_tracker(world)
    counts = []
    run(world, tr, 5.0, on_frame=lambda t, obs: counts.append(len(obs)))

    total_ids = tr._next_id
    log(f"ids created: {total_ids}, obstacles per frame (last 10): {counts[-10:]}")
    # A tentative track that misses its 2nd frame dies and restarts (by design),
    # so a few extra IDs are expected; a confirmed track should never be lost.
    assert total_ids <= 2 * len(rocks), f"{total_ids} track IDs for {len(rocks)} rocks"
    assert max(counts) <= len(rocks), f"more obstacles ({max(counts)}) than rocks"
    assert np.mean(counts[-20:]) >= len(rocks) - 1, "tracks lost during dropouts"


def test_crossing_rocks_keep_ids():
    """Two same-size rocks crossing 4 m apart: IDs must not swap."""
    a = Rock(np.array([60.0, -20.0, 0.0]), np.array([0.0, 4.0, 0.0]), 2.0)
    b = Rock(np.array([64.0, 20.0, 0.0]), np.array([0.0, -4.0, 0.0]), 2.0)
    world = FakeWorld([a, b], pos_noise=0.1, seed=6)
    tr = make_tracker(world)
    ids_a, ids_b = [], []

    def on_frame(t, obs):
        if t < 1.0:
            return
        ta, _ = nearest_track(tr, a.pos(t))
        tb, _ = nearest_track(tr, b.pos(t))
        ids_a.append(ta.track_id)
        ids_b.append(tb.track_id)

    run(world, tr, 10.0, on_frame=on_frame)
    log(f"ids near A: {sorted(set(ids_a))}, near B: {sorted(set(ids_b))}")
    assert len(set(ids_a)) == 1, f"rock A changed track IDs: {sorted(set(ids_a))}"
    assert len(set(ids_b)) == 1, f"rock B changed track IDs: {sorted(set(ids_b))}"
    assert ids_a[0] != ids_b[0], "both rocks share one track"


def test_rock_leaves_range():
    """A rock flying away is deleted once it's out of range for good."""
    rock = Rock(np.array([100.0, 0.0, 0.0]), np.array([30.0, 0.0, 0.0]), 2.0)
    world = FakeWorld([rock])
    tr = make_tracker(world, max_range=150.0)
    run(world, tr, 4.0)    # ends at 220 m
    log(f"tracks left: {len(tr.tracks())}")
    assert len(tr.tracks()) == 0, "track should be deleted after leaving range"


def test_repeated_and_backwards_frames():
    """Same timestamp twice, or time going backwards: no crash, no velocity jump."""
    rock = Rock(np.array([40.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 2.0)
    world = FakeWorld([rock])
    tr = make_tracker(world)
    run(world, tr, 1.0)
    v_before = tr.tracks()[0].velocity.copy()

    t = 1.0
    for t_bad in (t, t - 0.05):
        tr.update(world.radar(t), world.shuttle_pos(t), world.shuttle_v, t_bad)
    v_after = tr.tracks()[0].velocity
    log(f"v before {v_before}, after {v_after}")
    assert len(tr.tracks()) == 1
    assert np.allclose(v_before, v_after), "velocity changed on a dt <= 0 frame"


def test_obstacle_output():
    """Output is a list of Obstacle with sane fields; tentative tracks excluded."""
    rock = Rock(np.array([40.0, 0.0, 0.0]), np.zeros(3), 2.5)
    world = FakeWorld([rock])
    tr = make_tracker(world)
    tr.reset()
    obs0 = tr.update(world.radar(0.0), world.shuttle_pos(0.0), world.shuttle_v, 0.0)
    assert isinstance(obs0, list) and len(obs0) == 0, "tentative track was output"
    obs = run(world, tr, 1.0)
    assert len(obs) == 1 and isinstance(obs[0], Obstacle), f"bad output: {obs}"
    o = obs[0]
    assert np.shape(o.position) == (3,) and np.shape(o.velocity) == (3,)
    assert abs(o.radius - 2.5) < 1e-6, f"radius {o.radius} != 2.5"
    assert o.position_sigma > 0


def test_debug_info():
    """debug_info has the expected keys and consistent counts."""
    rocks = [Rock(np.array([40.0, 0.0, 0.0]), np.zeros(3), 2.0),
             Rock(np.array([60.0, 10.0, 0.0]), np.zeros(3), 3.0)]
    world = FakeWorld(rocks)
    tr = make_tracker(world)
    run(world, tr, 1.0)
    d = tr.debug_info()
    log(tr.format_debug())
    for key in ("n_detections", "n_matched", "n_new", "n_missed", "n_deleted",
                "n_tracks", "n_confirmed", "n_coasting", "n_obstacles",
                "nearest_id", "nearest_surface_dist", "nearest_speed"):
        assert key in d, f"missing key {key}"
    assert d["n_detections"] == 2 and d["n_matched"] == 2 and d["n_new"] == 0
    assert d["n_confirmed"] == 2 and d["n_obstacles"] == 2
    assert abs(d["nearest_surface_dist"] - 38.0) < 1e-6


# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------

def plot_scenario():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(1)
    rocks = [Rock(rng.uniform([30, -40, -20], [90, 40, 20]),
                  rng.uniform(-3, 3, 3), rng.uniform(1, 4)) for _ in range(6)]
    world = FakeWorld(rocks, pos_noise=0.2, dropout=0.2, seed=2)
    tr = make_tracker(world)
    times, errs = [], {i: [] for i in range(len(rocks))}

    def on_frame(t, obs):
        times.append(t)
        for i, rock in enumerate(rocks):
            track, _ = nearest_track(tr, rock.pos(t))
            errs[i].append(np.linalg.norm(track.velocity - rock.v) if track else np.nan)

    run(world, tr, 5.0, on_frame=on_frame)
    for i, e in errs.items():
        plt.plot(times, e, label=f"rock {i}")
    plt.xlabel("time (s)")
    plt.ylabel("velocity error (m/s)")
    plt.title(f"Tracker velocity error (radar dt = {RADAR_DT} s, 20% dropouts)")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {test.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if "--plot" in sys.argv:
        plot_scenario()
    sys.exit(1 if failed else 0)