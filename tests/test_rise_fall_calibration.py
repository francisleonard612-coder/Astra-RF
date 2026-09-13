"""
Tests for two more additions to decision/rise_fall_decision_engine.py:

1. Per-resolution calibration split: self.calibration is keyed by
   (contract_type, duration_unit) -- four independent CalibrationTrackers
   per symbol, not two pooled across tick/minute.
2. The calibration-quality gate (min_calibration_quality): once a
   candidate's calibrator has fit, its quality_score() must also clear
   min_calibration_quality, on top of (not instead of) the confidence gate.
"""
import asyncio

import numpy as np

import pricing.payout as payout_module
from decision.rise_fall_decision_engine import RiseFallSymbolPipeline, summary_reason
from pricing.contracts import FALL, RISE


def _reset_payout_module_state():
    payout_module._quote_cache.clear()
    payout_module._rate_limit_cooldown_until = 0.0
    payout_module._consecutive_rate_limit_failures = 0


class FakeDerivClient:
    def __init__(self, payout_for_contract_type: dict[str, float]):
        self.payout_for_contract_type = payout_for_contract_type
        self.calls = []

    async def get_proposal(self, symbol, contract_type, barrier, stake, duration, duration_unit, currency):
        self.calls.append((contract_type, duration, duration_unit, stake))
        assert barrier is None
        return {
            "payout": self.payout_for_contract_type.get(contract_type, 2.0),
            "ask_price": stake, "id": f"prop-{contract_type}-{duration}{duration_unit}-{stake}",
            "spot": 100.0, "longcode": "rises" if contract_type == RISE else "falls",
        }


def _seed_trending_history(pipeline: RiseFallSymbolPipeline, seed: int, drift: float) -> None:
    rng = np.random.default_rng(seed)
    tick_returns = rng.normal(drift, 0.001, size=300)
    minute_returns = rng.normal(drift, 0.002, size=300)
    pipeline.price_series.tick_log_returns.extend(tick_returns.tolist())
    pipeline.price_series.minute_log_returns.extend(minute_returns.tolist())


# ---------------------------------------------------------------------------
# Per-resolution calibration split
# ---------------------------------------------------------------------------

def test_calibration_has_four_independent_trackers_per_symbol():
    pipeline = RiseFallSymbolPipeline("1HZ10V")
    assert set(pipeline.calibration.keys()) == {(RISE, "t"), (RISE, "m"), (FALL, "t"), (FALL, "m")}
    # confirm they're genuinely independent objects, not the same tracker
    # referenced under multiple keys
    assert len({id(t) for t in pipeline.calibration.values()}) == 4


def test_tick_and_minute_outcomes_for_the_same_side_do_not_share_a_tracker():
    pipeline = RiseFallSymbolPipeline("1HZ10V")
    pipeline._pending = (RISE, "t", 0.9)
    pipeline.record_outcome(won=True)
    assert list(pipeline.calibration[(RISE, "t")]._outcome) == [1]
    assert list(pipeline.calibration[(RISE, "m")]._outcome) == []  # same side, other resolution -- untouched

    pipeline._pending = (RISE, "m", 0.4)
    pipeline.record_outcome(won=False)
    assert list(pipeline.calibration[(RISE, "m")]._outcome) == [0]
    assert list(pipeline.calibration[(RISE, "t")]._outcome) == [1]  # unaffected by the minute-side update


# ---------------------------------------------------------------------------
# Calibration-quality gate
# ---------------------------------------------------------------------------

def test_calibration_quality_gate_disabled_by_default():
    pipeline = RiseFallSymbolPipeline("1HZ10V")
    assert pipeline.min_calibration_quality == 0.0


def test_quality_gate_does_not_block_an_uncalibrated_tracker_even_with_a_high_threshold(monkeypatch):
    """quality_score() returns a fixed low placeholder (0.3) before a
    tracker has fit -- the gate must NOT apply that placeholder, or every
    symbol would be blocked from its very first trade for no real reason."""
    import decision.rise_fall_decision_engine as rfde
    _reset_payout_module_state()

    def fake_mc(returns, direction, durations, n_sims=2000, rng=None):
        return durations[0], 0.85

    monkeypatch.setattr(rfde, "monte_carlo_duration", fake_mc)

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03,
                                           min_confidence=0.70, min_calibration_quality=0.9)
        for tracker in pipeline.calibration.values():
            assert tracker.is_calibrated is False  # precondition: nothing has fit yet
        client = FakeDerivClient({RISE: 3.0, FALL: 3.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)

        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                            rng=np.random.default_rng(0))
        assert decision.decision != "NO_TRADE"  # not blocked despite min_calibration_quality=0.9

    asyncio.run(run())


def test_quality_gate_blocks_a_fitted_but_poorly_calibrated_candidate(monkeypatch):
    """Once a tracker HAS fit (is_calibrated=True), a low quality_score()
    must block it even though the raw/calibrated probabilities themselves
    would otherwise clear min_confidence and min_edge comfortably."""
    import decision.rise_fall_decision_engine as rfde
    _reset_payout_module_state()

    def fake_mc(returns, direction, durations, n_sims=2000, rng=None):
        return durations[0], 0.85

    monkeypatch.setattr(rfde, "monte_carlo_duration", fake_mc)

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03,
                                           min_confidence=0.70, min_calibration_quality=0.5)
        # Force every calibrator to look fitted (is_calibrated=True) but
        # badly calibrated (quality_score well under the 0.5 gate) --
        # sidesteps needing hundreds of real samples to reach this state
        # organically. calibrate()'s own exception handling falls back to
        # a temperature-scaled raw_prob when _calibrator.predict(...) fails
        # (see models/calibration.py), so this doesn't crash evaluate().
        for tracker in pipeline.calibration.values():
            tracker._calibrator = object()  # any non-None value flips is_calibrated True
            tracker.quality_score = lambda: 0.1

        client = FakeDerivClient({RISE: 3.0, FALL: 3.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)

        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                            rng=np.random.default_rng(0))

        assert decision.decision == "NO_TRADE"
        assert "no candidate cleared min_calibration_quality=0.5" in decision.reason
        assert summary_reason(decision) == "poor_calibration_quality"
        assert client.calls == []  # blocked before any real quote was fetched

    asyncio.run(run())


def test_quality_gate_allows_a_fitted_and_well_calibrated_candidate(monkeypatch):
    import decision.rise_fall_decision_engine as rfde
    _reset_payout_module_state()

    def fake_mc(returns, direction, durations, n_sims=2000, rng=None):
        return durations[0], 0.85

    monkeypatch.setattr(rfde, "monte_carlo_duration", fake_mc)

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03,
                                           min_confidence=0.70, min_calibration_quality=0.5)
        for tracker in pipeline.calibration.values():
            tracker._calibrator = object()
            tracker.quality_score = lambda: 0.95  # well above the gate

        client = FakeDerivClient({RISE: 3.0, FALL: 3.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)

        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                            rng=np.random.default_rng(0))
        assert decision.decision != "NO_TRADE"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Probe trades -- the escape hatch for the lockout the quality gate above
# would otherwise create (see "PROBE TRADES" docstring in
# decision/rise_fall_decision_engine.py)
# ---------------------------------------------------------------------------

def test_probe_stays_blocked_before_reaching_the_interval(monkeypatch):
    import decision.rise_fall_decision_engine as rfde
    _reset_payout_module_state()

    def fake_mc(returns, direction, durations, n_sims=2000, rng=None):
        return durations[0], 0.85

    monkeypatch.setattr(rfde, "monte_carlo_duration", fake_mc)

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03,
                                           min_confidence=0.70, min_calibration_quality=0.5,
                                           calibration_quality_probe_interval=5)
        for tracker in pipeline.calibration.values():
            tracker._calibrator = object()  # is_calibrated=True
            tracker.quality_score = lambda: 0.1  # poor -- well under the 0.5 gate

        client = FakeDerivClient({RISE: 3.0, FALL: 3.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)

        for _ in range(4):  # interval - 1 calls -- never reaches the probe threshold
            decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                                rng=np.random.default_rng(0))
            assert decision.decision == "NO_TRADE"
            assert "min_calibration_quality" in decision.reason

        assert client.calls == []  # never even fetched a real quote across all 4 blocked cycles

    asyncio.run(run())


def test_probe_fires_at_the_interval_and_resets_only_the_winning_keys_streak(monkeypatch):
    import decision.rise_fall_decision_engine as rfde
    _reset_payout_module_state()

    def fake_mc(returns, direction, durations, n_sims=2000, rng=None):
        return durations[0], 0.85

    monkeypatch.setattr(rfde, "monte_carlo_duration", fake_mc)

    async def run():
        interval = 5
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03,
                                           min_confidence=0.70, min_calibration_quality=0.5,
                                           calibration_quality_probe_interval=interval)
        for tracker in pipeline.calibration.values():
            tracker._calibrator = object()
            tracker.quality_score = lambda: 0.1

        client = FakeDerivClient({RISE: 3.0, FALL: 3.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)

        decision = None
        for _ in range(interval):
            decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                                rng=np.random.default_rng(0))

        assert decision.decision != "NO_TRADE"
        assert decision.is_calibration_probe is True
        assert decision.stake == 1.0  # base_stake

        winning_key = (decision.contract_type, decision.duration_unit)
        assert pipeline._quality_gate_blocked_streak[winning_key] == 0
        # every OTHER key also reached the interval this same cycle (identical
        # fake probability/quality across all four) and was also attempted as
        # a probe, but wasn't chosen -- its streak must NOT have been reset
        for key, streak in pipeline._quality_gate_blocked_streak.items():
            if key != winning_key:
                assert streak == interval

    asyncio.run(run())


def test_probe_interval_of_zero_disables_probing_permanently(monkeypatch):
    import decision.rise_fall_decision_engine as rfde
    _reset_payout_module_state()

    def fake_mc(returns, direction, durations, n_sims=2000, rng=None):
        return durations[0], 0.85

    monkeypatch.setattr(rfde, "monte_carlo_duration", fake_mc)

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03,
                                           min_confidence=0.70, min_calibration_quality=0.5,
                                           calibration_quality_probe_interval=0)
        for tracker in pipeline.calibration.values():
            tracker._calibrator = object()
            tracker.quality_score = lambda: 0.1

        client = FakeDerivClient({RISE: 3.0, FALL: 3.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)

        for _ in range(200):  # far beyond any reasonable interval -- must never fire
            decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                                rng=np.random.default_rng(0))
            assert decision.decision == "NO_TRADE"

        assert client.calls == []

    asyncio.run(run())


def test_probe_ignores_martingale_escalation_and_uses_base_stake(monkeypatch):
    import decision.rise_fall_decision_engine as rfde
    _reset_payout_module_state()

    def fake_mc(returns, direction, durations, n_sims=2000, rng=None):
        return durations[0], 0.85

    monkeypatch.setattr(rfde, "monte_carlo_duration", fake_mc)

    async def run():
        interval = 3
        pipeline = RiseFallSymbolPipeline(
            "1HZ10V", base_stake=1.0, min_edge=0.03, min_confidence=0.70, min_calibration_quality=0.5,
            calibration_quality_probe_interval=interval,
            staking_enabled=True, staking_progression_factor=2.0, staking_max_steps=4, staking_max_stake=100.0,
        )
        # escalate the martingale progression well past base_stake
        pipeline.staking.load_state("1HZ10V", step=3, consecutive_losses=3)
        assert pipeline.staking.current_stake("1HZ10V") == 8.0  # would apply to a non-probe trade

        for tracker in pipeline.calibration.values():
            tracker._calibrator = object()
            tracker.quality_score = lambda: 0.1

        client = FakeDerivClient({RISE: 3.0, FALL: 3.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)

        decision = None
        for _ in range(interval):
            decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                                rng=np.random.default_rng(0))

        assert decision.decision != "NO_TRADE"
        assert decision.is_calibration_probe is True
        assert decision.stake == 1.0  # base_stake, NOT the escalated 8.0

    asyncio.run(run())


def test_probe_streak_not_reset_if_the_probe_attempt_fails_a_later_gate(monkeypatch):
    """A probe that clears the quality gate but then fails confidence must
    NOT reset its streak -- it hasn't actually generated a fresh sample, so
    it should remain eligible to try again next cycle rather than waiting
    through another full interval."""
    import decision.rise_fall_decision_engine as rfde
    _reset_payout_module_state()

    def fake_mc(returns, direction, durations, n_sims=2000, rng=None):
        return durations[0], 0.85  # clears the quality bypass, but won't clear a 0.99 confidence floor

    monkeypatch.setattr(rfde, "monte_carlo_duration", fake_mc)

    async def run():
        interval = 3
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03,
                                           min_confidence=0.99, min_calibration_quality=0.5,
                                           calibration_quality_probe_interval=interval)
        for tracker in pipeline.calibration.values():
            tracker._calibrator = object()
            tracker.quality_score = lambda: 0.1

        client = FakeDerivClient({RISE: 3.0, FALL: 3.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)

        for _ in range(interval + 2):  # push well past the interval
            decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                                rng=np.random.default_rng(0))
            assert decision.decision == "NO_TRADE"  # confidence gate still blocks every attempt

        assert client.calls == []  # confidence gate rejects before any quote fetch, probe or not
        for streak in pipeline._quality_gate_blocked_streak.values():
            assert streak >= interval  # never reset -- no probe ever actually traded

    asyncio.run(run())
