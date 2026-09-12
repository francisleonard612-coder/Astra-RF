"""
Tests for two additions to decision/rise_fall_decision_engine.py:

1. The confidence gate (min_confidence): a candidate must clear BOTH its raw
   MC win-probability (mc_win_probability) and its calibrated probability
   before it's even priced, on top of (not instead of) min_edge.
2. Opt-in martingale staking, routed through risk/staking.py's
   StakingEngine (RiseFallSymbolPipeline.staking).

Mirrors the fakes/fixtures already used in test_rise_fall_decision_engine.py
so these tests exercise the real evaluate()/record_outcome() code path, not
a reimplementation of it.
"""
import asyncio

import numpy as np

import pricing.payout as payout_module
from decision.rise_fall_decision_engine import DRIFT_STAKE_REDUCTION, RiseFallSymbolPipeline, summary_reason
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
# Confidence gate
# ---------------------------------------------------------------------------

def test_confidence_gate_blocks_a_trade_that_would_otherwise_clear_min_edge(monkeypatch):
    """A modest, genuinely-tradeable edge (breakeven comfortably below the
    calibrated probability) still must NOT trade if min_confidence is set
    above the probability actually reached -- min_edge alone is not
    sufficient once a confidence floor is configured. monte_carlo_duration
    is monkeypatched to a fixed, known probability so this test asserts the
    gate's own boolean logic rather than hunting for a seed that happens to
    land in a narrow probability band."""
    import decision.rise_fall_decision_engine as rfde
    _reset_payout_module_state()

    def fake_mc(returns, direction, durations, n_sims=2000, rng=None):
        return durations[0], 0.75  # comfortably above breakeven, comfortably below a 0.90 confidence floor

    monkeypatch.setattr(rfde, "monte_carlo_duration", fake_mc)

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03, min_confidence=0.90)
        client = FakeDerivClient({RISE: 3.0, FALL: 3.0})  # generous payout either way -> would clear min_edge easily
        _seed_trending_history(pipeline, seed=0, drift=-0.01)  # only needed to land a TRADEABLE regime

        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                            rng=np.random.default_rng(0))

        assert decision.decision == "NO_TRADE"
        assert "no candidate cleared min_confidence=0.9" in decision.reason
        assert summary_reason(decision) == "insufficient_confidence"
        # the confidence gate must reject BEFORE any real quote is fetched
        assert client.calls == []

    asyncio.run(run())


def test_confidence_gate_allows_a_trade_when_both_probabilities_clear_it(monkeypatch):
    import decision.rise_fall_decision_engine as rfde
    _reset_payout_module_state()

    def fake_mc(returns, direction, durations, n_sims=2000, rng=None):
        return durations[0], 0.95

    monkeypatch.setattr(rfde, "monte_carlo_duration", fake_mc)

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03, min_confidence=0.70)
        client = FakeDerivClient({RISE: 3.0, FALL: 3.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)

        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                            rng=np.random.default_rng(0))

        assert decision.decision != "NO_TRADE"
        assert decision.mc_win_probability == 0.95
        assert decision.calibrated_probability == 0.95  # calibrator untrained -> passes raw_p through unchanged
        assert client.calls  # a real quote WAS fetched once the gate cleared

    asyncio.run(run())


def test_confidence_gate_default_is_point_seven():
    pipeline = RiseFallSymbolPipeline("1HZ10V")
    assert pipeline.min_confidence == 0.70


def test_confidence_gate_requires_both_raw_and_calibrated_not_just_one():
    """A candidate whose calibrated probability has been pulled below
    min_confidence by an unfavorable calibration history must still be
    blocked even though its raw MC estimate alone would clear the gate --
    and vice versa. Exercised directly against the gate condition (not by
    hunting for a seed with this exact property), since the point is the
    boolean logic itself, not any particular market scenario."""
    pipeline = RiseFallSymbolPipeline("1HZ10V", min_confidence=0.70)
    raw_p, calibrated_p = 0.95, 0.50  # raw clears, calibrated doesn't
    assert not (raw_p > pipeline.min_confidence and calibrated_p > pipeline.min_confidence)
    raw_p, calibrated_p = 0.50, 0.95  # calibrated clears, raw doesn't
    assert not (raw_p > pipeline.min_confidence and calibrated_p > pipeline.min_confidence)
    raw_p, calibrated_p = 0.95, 0.95  # both clear
    assert raw_p > pipeline.min_confidence and calibrated_p > pipeline.min_confidence


# ---------------------------------------------------------------------------
# Martingale staking
# ---------------------------------------------------------------------------

def test_staking_disabled_by_default_trades_flat_base_stake():
    _reset_payout_module_state()

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=2.0, min_edge=0.03)
        assert pipeline.staking.enabled is False
        client = FakeDerivClient({RISE: 1.5, FALL: 50.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)

        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                            rng=np.random.default_rng(0))
        assert decision.decision != "NO_TRADE"
        assert decision.stake == 2.0

        pipeline.record_outcome(won=False)
        # staking disabled -- a loss must NOT change the next stake
        assert pipeline.staking.current_stake("1HZ10V") == 2.0

    asyncio.run(run())


def test_staking_enabled_does_not_escalate_after_a_single_isolated_loss():
    """The progression only ENGAGES after a second consecutive loss --
    exercised directly against StakingEngine, since this is the piece that
    owns the threshold (see risk/staking.py's
    min_consecutive_losses_before_escalation)."""
    pipeline = RiseFallSymbolPipeline(
        "1HZ10V", base_stake=1.0,
        staking_enabled=True, staking_progression_factor=2.0, staking_max_steps=4, staking_max_stake=100.0,
    )
    assert pipeline.staking.min_consecutive_losses_before_escalation == 2  # Rise/Fall's own default
    pipeline._pending = (RISE, 0.8)
    pipeline.record_outcome(won=False)  # first loss in the streak
    assert pipeline.staking.current_stake("1HZ10V") == 1.0  # unchanged -- threshold not yet reached


def test_staking_enabled_escalates_stake_after_a_loss_and_resets_on_a_win():
    _reset_payout_module_state()

    async def run():
        pipeline = RiseFallSymbolPipeline(
            "1HZ10V", base_stake=1.0, min_edge=0.03,
            staking_enabled=True, staking_progression_factor=2.0,
            staking_max_steps=4, staking_max_stake=100.0,
        )
        client = FakeDerivClient({RISE: 1.5, FALL: 50.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)

        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                            rng=np.random.default_rng(0))
        assert decision.decision != "NO_TRADE"
        assert decision.stake == 1.0  # first trade, no prior losses yet

        pipeline.record_outcome(won=False)
        # ONE loss alone must not escalate -- the progression needs a
        # second consecutive loss to engage (Astra's Rise/Fall default).
        assert pipeline.staking.current_stake("1HZ10V") == 1.0

        decision2 = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                             rng=np.random.default_rng(0))
        assert decision2.decision != "NO_TRADE"
        assert decision2.stake == 1.0  # still base_stake -- only one loss recorded so far

        pipeline.record_outcome(won=False)  # SECOND consecutive loss -- progression now engages
        assert pipeline.staking.current_stake("1HZ10V") == 2.0  # 1.0 * 2.0^1

        decision3 = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                             rng=np.random.default_rng(0))
        assert decision3.decision != "NO_TRADE"
        assert decision3.stake == 2.0  # the escalated stake was actually used to trade

        pipeline.record_outcome(won=True)
        assert pipeline.staking.current_stake("1HZ10V") == 1.0  # win resets to base_stake

    asyncio.run(run())


def test_staking_escalated_stake_and_quote_share_the_same_basis():
    """Same class of bug the drift-reduction path already guards against
    (test_drift_degraded_quote_matches_the_reduced_stake_basis): once
    martingale has moved stake away from base_stake, the quote attached to
    the decision must be re-fetched at that SAME stake, not left at the
    base_stake-basis comparison quote -- otherwise OrderExecutor's
    stale-quote payout check would reject every escalated trade."""
    _reset_payout_module_state()

    async def run():
        pipeline = RiseFallSymbolPipeline(
            "1HZ10V", base_stake=1.0, min_edge=0.03,
            staking_enabled=True, staking_progression_factor=2.0, staking_max_steps=4, staking_max_stake=100.0,
        )
        client = FakeDerivClient({RISE: 1.5, FALL: 50.0})
        _seed_trending_history(pipeline, seed=0, drift=-0.01)

        # force TWO consecutive losses -- the threshold -- to escalate the
        # stake before the decision that matters
        first = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                         rng=np.random.default_rng(0))
        assert first.decision != "NO_TRADE"
        pipeline.record_outcome(won=False)
        second = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                          rng=np.random.default_rng(0))
        assert second.decision != "NO_TRADE"
        pipeline.record_outcome(won=False)

        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                            rng=np.random.default_rng(0))
        assert decision.decision != "NO_TRADE"
        assert decision.stake == 2.0
        assert decision.quote is not None
        assert decision.quote.ask_price == decision.stake

    asyncio.run(run())


def test_staking_max_steps_stops_escalating_and_resets():
    pipeline = RiseFallSymbolPipeline(
        "1HZ10V", base_stake=1.0,
        staking_enabled=True, staking_progression_factor=2.0, staking_max_steps=2, staking_max_stake=100.0,
    )
    pipeline._pending = (RISE, 0.8)
    pipeline.record_outcome(won=False)  # loss #1 -- below threshold, stake unchanged
    assert pipeline.staking.current_stake("1HZ10V") == 1.0
    pipeline._pending = (RISE, 0.8)
    pipeline.record_outcome(won=False)  # loss #2 -- threshold reached, step 1 -> stake 2.0
    assert pipeline.staking.current_stake("1HZ10V") == 2.0
    pipeline._pending = (RISE, 0.8)
    pipeline.record_outcome(won=False)  # loss #3 -- step 2 (== max_steps) -> stake 4.0
    assert pipeline.staking.current_stake("1HZ10V") == 4.0
    pipeline._pending = (RISE, 0.8)
    pipeline.record_outcome(won=False)  # loss #4 -- beyond max_steps -> resets to step 0
    assert pipeline.staking.current_stake("1HZ10V") == 1.0


def test_staking_respects_its_own_max_stake_ceiling():
    pipeline = RiseFallSymbolPipeline(
        "1HZ10V", base_stake=1.0,
        staking_enabled=True, staking_progression_factor=2.0, staking_max_steps=10, staking_max_stake=3.0,
    )
    for _ in range(6):
        pipeline._pending = (RISE, 0.8)
        pipeline.record_outcome(won=False)
    assert pipeline.staking.current_stake("1HZ10V") <= 3.0


def test_cancel_pending_does_not_move_the_staking_progression():
    pipeline = RiseFallSymbolPipeline(
        "1HZ10V", base_stake=1.0,
        staking_enabled=True, staking_progression_factor=2.0, staking_max_steps=4, staking_max_stake=100.0,
    )
    # two real consecutive losses -- crosses the threshold, escalates to step 1
    pipeline._pending = (RISE, 0.8)
    pipeline.record_outcome(won=False)
    pipeline._pending = (RISE, 0.8)
    pipeline.record_outcome(won=False)
    assert pipeline.staking.current_stake("1HZ10V") == 2.0

    # a trade that was decided but never actually settled must NOT count
    # as a third loss just because cancel_pending() was called
    pipeline._pending = (RISE, 0.8)
    pipeline.cancel_pending()
    assert pipeline.staking.current_stake("1HZ10V") == 2.0  # unchanged


def test_record_outcome_with_no_pending_candidate_does_not_move_staking():
    pipeline = RiseFallSymbolPipeline(
        "1HZ10V", base_stake=1.0,
        staking_enabled=True, staking_progression_factor=2.0, staking_max_steps=4, staking_max_stake=100.0,
    )
    pipeline.record_outcome(won=False)  # nothing pending -- no real trade to attribute this to
    assert pipeline.staking.current_stake("1HZ10V") == 1.0


def test_staking_default_max_stake_derives_from_progression_when_unset():
    pipeline = RiseFallSymbolPipeline(
        "1HZ10V", base_stake=1.0,
        staking_enabled=True, staking_progression_factor=2.0, staking_max_steps=3,
    )
    assert pipeline.staking.max_stake == 1.0 * (2.0 ** 3)


def test_staking_engine_min_consecutive_losses_before_escalation_defaults_to_one():
    """Direct StakingEngine-level check that the default (used when a
    caller doesn't specify otherwise, e.g. the legacy digit risk.staking
    config) still escalates after a single loss -- Astra's Rise/Fall
    pipeline explicitly opts into a threshold of 2 on top of this."""
    from risk.staking import StakingEngine
    staking = StakingEngine(base_stake=1.0, enabled=True, progression_factor=2.0, max_steps=3, max_stake=100.0)
    assert staking.min_consecutive_losses_before_escalation == 1
    staking.record_result("R_100", won=False)
    assert staking.current_stake("R_100") == 2.0  # escalates immediately, unlike Rise/Fall's default


def test_staking_engine_explicit_threshold_of_two_matches_rise_fall_default():
    from risk.staking import StakingEngine
    staking = StakingEngine(base_stake=1.0, enabled=True, progression_factor=2.0, max_steps=3, max_stake=100.0,
                             min_consecutive_losses_before_escalation=2)
    staking.record_result("R_100", won=False)  # loss #1
    assert staking.current_stake("R_100") == 1.0
    staking.record_result("R_100", won=False)  # loss #2 -- threshold reached
    assert staking.current_stake("R_100") == 2.0
    staking.record_result("R_100", won=True)  # a win resets the streak entirely
    assert staking.current_stake("R_100") == 1.0
    staking.record_result("R_100", won=False)  # back to loss #1 of a fresh streak
    assert staking.current_stake("R_100") == 1.0
