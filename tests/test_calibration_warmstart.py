"""
Tests for research/calibration_warmstart.py.

No network dependency: build_calibration_samples() takes plain (epoch,
price) tuples, so these tests construct synthetic tick series directly
rather than fetching real history. The point of these tests is (1) the
replay produces valid probabilities and binary outcomes with no look-ahead
corruption within a single sample, (2) an obviously-trending series labels
outcomes in the direction that's actually true (i.e. the labeling logic
isn't accidentally reversed), and (3) the JSON round-trip and
apply-to-pipeline plumbing are lossless.
"""
import json

import numpy as np

from decision.rise_fall_decision_engine import RiseFallSymbolPipeline
from pricing.contracts import FALL, RISE
from research.calibration_warmstart import (
    apply_samples_to_pipeline,
    build_calibration_samples,
    jsonable_to_samples,
    load_samples_json,
    samples_to_jsonable,
    save_samples_json,
)

ALL_KEYS = {(RISE, "t"), (RISE, "m"), (FALL, "t"), (FALL, "m")}


def _synthetic_ticks(n: int, drift: float, seed: int = 0, start_epoch: int = 1_700_000_000) -> list[tuple[int, float]]:
    """n one-second-spaced ticks with a log-price random walk of the given
    per-step drift -- enough ticks at n=3000 to cover ~50 one-minute bars,
    comfortably above build_calibration_samples()'s MIN_HISTORY_LENGTH for
    both tick and minute resolution."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, 0.001, size=n)
    log_prices = np.cumsum(log_returns) + np.log(100.0)
    prices = np.exp(log_prices)
    return [(start_epoch + i, float(p)) for i, p in enumerate(prices)]


# ---------------------------------------------------------------------------
# build_calibration_samples
# ---------------------------------------------------------------------------

def test_build_calibration_samples_returns_all_four_keys():
    ticks = _synthetic_ticks(3000, drift=0.0002)
    samples = build_calibration_samples(ticks, [5, 10], [1, 3], n_sims=200, sample_every=20,
                                         rng=np.random.default_rng(1))
    assert set(samples.keys()) == ALL_KEYS


def test_build_calibration_samples_produces_valid_probabilities_and_binary_outcomes():
    ticks = _synthetic_ticks(3000, drift=0.0002)
    samples = build_calibration_samples(ticks, [5, 10], [1, 3], n_sims=200, sample_every=20,
                                         rng=np.random.default_rng(1))
    total = 0
    for key, pairs in samples.items():
        for raw_prob, outcome in pairs:
            assert 0.0 <= raw_prob <= 1.0, f"{key}: {raw_prob!r} out of [0,1]"
            assert outcome in (0, 1), f"{key}: {outcome!r} not binary"
            total += 1
    assert total > 0  # this trend/length combination must actually produce samples somewhere


def test_build_calibration_samples_labels_outcomes_correctly_on_a_strong_uptrend():
    """A strong, obvious uptrend should settle RISE candidates as wins far
    more often than losses, and FALL candidates the opposite way -- if the
    outcome-labeling logic in build_calibration_samples() had the sign
    backwards, this would come out flipped."""
    ticks = _synthetic_ticks(3000, drift=0.003)  # strong, unambiguous uptrend
    samples = build_calibration_samples(ticks, [5, 10], [1, 3], n_sims=200, sample_every=10,
                                         rng=np.random.default_rng(2))

    rise_outcomes = [o for _, o in samples[(RISE, "t")]] + [o for _, o in samples[(RISE, "m")]]
    fall_outcomes = [o for _, o in samples[(FALL, "t")]] + [o for _, o in samples[(FALL, "m")]]
    assert len(rise_outcomes) > 10 and len(fall_outcomes) > 10
    assert sum(rise_outcomes) / len(rise_outcomes) > 0.8  # RISE should win most of the time
    assert sum(fall_outcomes) / len(fall_outcomes) < 0.2  # FALL should lose most of the time


def test_build_calibration_samples_empty_for_too_short_history():
    ticks = _synthetic_ticks(15, drift=0.0)  # far short of MIN_HISTORY_LENGTH=20 + any duration
    samples = build_calibration_samples(ticks, [5, 10], [1, 3], n_sims=100, sample_every=5,
                                         rng=np.random.default_rng(0))
    assert all(pairs == [] for pairs in samples.values())


def test_build_calibration_samples_empty_when_no_duration_candidates_given():
    ticks = _synthetic_ticks(3000, drift=0.0)
    samples = build_calibration_samples(ticks, [], [], n_sims=100, sample_every=20,
                                         rng=np.random.default_rng(0))
    assert all(pairs == [] for pairs in samples.values())


def test_build_calibration_samples_smaller_stride_yields_at_least_as_many_samples():
    ticks = _synthetic_ticks(3000, drift=0.0002)
    dense = build_calibration_samples(ticks, [5, 10], [1, 3], n_sims=100, sample_every=5,
                                       rng=np.random.default_rng(3))
    sparse = build_calibration_samples(ticks, [5, 10], [1, 3], n_sims=100, sample_every=25,
                                        rng=np.random.default_rng(3))
    assert len(dense[(RISE, "t")]) >= len(sparse[(RISE, "t")])


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

def test_samples_to_jsonable_and_back_round_trips():
    samples = {
        (RISE, "t"): [(0.8, 1), (0.6, 0)],
        (RISE, "m"): [],
        (FALL, "t"): [(0.3, 0)],
        (FALL, "m"): [(0.55, 1), (0.9, 1)],
    }
    restored = jsonable_to_samples(samples_to_jsonable(samples))
    assert restored == samples


def test_samples_to_jsonable_uses_string_keys():
    samples = {(RISE, "t"): [(0.8, 1)], (RISE, "m"): [], (FALL, "t"): [], (FALL, "m"): []}
    jsonable = samples_to_jsonable(samples)
    assert set(jsonable.keys()) == {"CALL_t", "CALL_m", "PUT_t", "PUT_m"}
    json.dumps(jsonable)  # must be genuinely JSON-serializable, not just dict-shaped


def test_save_and_load_samples_json_round_trips(tmp_path):
    samples = {
        (RISE, "t"): [(0.8, 1), (0.6, 0)],
        (RISE, "m"): [(0.7, 1)],
        (FALL, "t"): [],
        (FALL, "m"): [(0.4, 0)],
    }
    path = str(tmp_path / "warmstart" / "R_100.json")  # nested dir -- must be created if missing
    save_samples_json(samples, path)
    loaded = load_samples_json(path)
    assert loaded == samples


# ---------------------------------------------------------------------------
# apply_samples_to_pipeline
# ---------------------------------------------------------------------------

def test_apply_samples_to_pipeline_feeds_the_matching_trackers():
    pipeline = RiseFallSymbolPipeline("1HZ10V")
    samples = {
        (RISE, "t"): [(0.8, 1), (0.6, 0), (0.9, 1)],
        (RISE, "m"): [],
        (FALL, "t"): [(0.3, 0)],
        (FALL, "m"): [],
    }

    applied = apply_samples_to_pipeline(pipeline, samples)

    assert applied == {"CALL_t": 3, "CALL_m": 0, "PUT_t": 1, "PUT_m": 0}  # counts reported for every matched key, including empty ones
    assert list(pipeline.calibration[(RISE, "t")]._raw) == [0.8, 0.6, 0.9]
    assert list(pipeline.calibration[(RISE, "t")]._outcome) == [1, 0, 1]
    assert list(pipeline.calibration[(FALL, "t")]._raw) == [0.3]
    assert len(pipeline.calibration[(RISE, "m")]._raw) == 0
    assert len(pipeline.calibration[(FALL, "m")]._raw) == 0


def test_apply_samples_to_pipeline_skips_unknown_keys_without_raising():
    pipeline = RiseFallSymbolPipeline("1HZ10V")
    samples = {("SOMETHING_ELSE", "t"): [(0.5, 1)]}

    applied = apply_samples_to_pipeline(pipeline, samples)

    assert applied == {}  # nothing matched -- no error, no partial state


def test_apply_samples_to_pipeline_can_trigger_a_real_fit():
    """End-to-end sanity check: enough warm-started samples should actually
    flip is_calibrated to True, exactly as enough live settled trades
    would -- apply_samples_to_pipeline uses the same .record() call, not a
    separate code path that merely resembles it."""
    pipeline = RiseFallSymbolPipeline("1HZ10V", min_calibration_samples=50)
    rng = np.random.default_rng(4)
    raw = rng.uniform(0.4, 0.95, size=200)
    outcome = (rng.uniform(0, 1, size=200) < raw).astype(int)  # roughly well-calibrated synthetic data
    samples = {(RISE, "t"): list(zip(raw.tolist(), outcome.tolist())),
               (RISE, "m"): [], (FALL, "t"): [], (FALL, "m"): []}

    assert pipeline.calibration[(RISE, "t")].is_calibrated is False
    apply_samples_to_pipeline(pipeline, samples)
    assert pipeline.calibration[(RISE, "t")].is_calibrated is True
