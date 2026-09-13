import asyncio

import numpy as np

import pricing.payout as payout_module
from decision.rise_fall_decision_engine import RiseFallSymbolPipeline
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
        self.calls.append((contract_type, duration, duration_unit))
        return {
            "payout": self.payout_for_contract_type.get(contract_type, 2.0),
            "ask_price": stake, "id": f"prop-{contract_type}-{duration}{duration_unit}",
            "spot": 100.0, "longcode": "rises" if contract_type == RISE else "falls",
        }


def _seed_price_history(pipeline: RiseFallSymbolPipeline, seed: int, drift: float) -> None:
    """Mirrors tests/test_rise_fall_decision_engine.py's own
    _seed_trending_history -- populates PriceSeries directly (not through
    observe_tick()/push()), so this alone does NOT feed the new signal
    models. Use _seed_conviction_layers() alongside this when a test needs
    a specific conviction reading."""
    rng = np.random.default_rng(seed)
    tick_returns = rng.normal(drift, 0.001, size=300)
    minute_returns = rng.normal(drift, 0.002, size=300)
    pipeline.price_series.tick_log_returns.extend(tick_returns.tolist())
    pipeline.price_series.minute_log_returns.extend(minute_returns.tolist())


def _seed_conviction_layers(pipeline: RiseFallSymbolPipeline, unit: str, direction: int) -> None:
    """Directly drives the three momentum-family layers (tick_markov,
    hawkes_momentum, kalman_trend -- the TREND_* regime's mapped layers in
    RF_REGIME_LAYERS) toward a strong, unanimous vote in `direction`, via
    their real push() methods (not by poking .vote() or private state)."""
    sign = 1 if direction > 0 else -1
    price = 100.0
    for i in range(120):
        r = sign * 0.002
        pipeline.tick_markov[unit].push(r)
        pipeline.hawkes[unit].push(r if i % 20 != 0 else sign * 0.01)  # occasional "jump" to excite Hawkes
        price *= (1 + r)
        pipeline.kalman[unit].push(price)


def test_shadow_mode_computes_and_logs_conviction_without_affecting_the_decision():
    """Default conviction_shadow_only=True: layer_votes/conviction should
    be populated on the returned decision, but the trade itself must be
    identical to what it would be with no conviction wiring at all.
    Recipe (real upward drift + a dramatically RISE-favoring payout, seed=0
    confirmed tradeable/TREND_QUIET) mirrors
    tests/test_rise_fall_decision_engine.py's own
    test_dramatic_mispricing_produces_a_trade_regardless_of_regime_noise,
    mirrored for RISE instead of FALL -- both raw MC confidence (from real
    drift) and edge (from the mispriced payout) need to clear, not just one."""
    _reset_payout_module_state()

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03)
        client = FakeDerivClient({RISE: 50.0, FALL: 1.5})
        _seed_price_history(pipeline, seed=0, drift=0.01)
        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                            rng=np.random.default_rng(0))
        assert decision.decision == "TRADE_RISE"
        assert decision.conviction_applied is False
        # layer_votes were computed (not None) even though nothing fed them yet
        assert decision.layer_votes is not None
        assert set(decision.layer_votes) == {"tick_markov", "ou_zscore", "hawkes_momentum", "kalman_trend"}

    asyncio.run(run())


def test_live_mode_blocks_a_trade_the_conviction_layers_disagree_with():
    """conviction_shadow_only=False: even a candidate that clears MC
    confidence, calibration quality, and a dramatic mispricing edge must
    NOT trade if the resolution's mapped layers unanimously vote the
    OTHER direction -- the conviction direction gate applies before the
    edge-cleared candidate is ever allowed to become `best`."""
    _reset_payout_module_state()

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03,
                                           conviction_shadow_only=False)
        # Same recipe as the shadow-mode test -- MC/edge alone would pick TRADE_RISE
        client = FakeDerivClient({RISE: 50.0, FALL: 1.5})
        _seed_price_history(pipeline, seed=0, drift=0.01)
        # Force the momentum layers to unanimously vote FALL -- disagreeing
        # with what MC/edge alone would have picked
        _seed_conviction_layers(pipeline, unit="t", direction=-1)
        _seed_conviction_layers(pipeline, unit="m", direction=-1)
        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                            rng=np.random.default_rng(0))
        assert decision.decision == "NO_TRADE"
        assert decision.conviction_direction == -1
        assert "conviction" in decision.reason or "no candidate cleared min_edge" not in decision.reason

    asyncio.run(run())


def test_live_mode_trades_when_conviction_agrees_with_the_mispriced_direction():
    """Same mispricing as above, but the conviction layers now agree with
    RISE -- the trade should go through."""
    _reset_payout_module_state()

    async def run():
        pipeline = RiseFallSymbolPipeline("1HZ10V", base_stake=1.0, min_edge=0.03,
                                           conviction_shadow_only=False)
        client = FakeDerivClient({RISE: 50.0, FALL: 1.5})
        _seed_price_history(pipeline, seed=0, drift=0.01)
        _seed_conviction_layers(pipeline, unit="t", direction=1)
        _seed_conviction_layers(pipeline, unit="m", direction=1)
        decision = await pipeline.evaluate(client, "USD", [5, 10], [1, 3], n_sims=2000,
                                            rng=np.random.default_rng(0))
        assert decision.decision == "TRADE_RISE"
        assert decision.conviction_applied is True
        assert decision.conviction_direction == 1
        assert decision.conviction is not None and decision.conviction > 0

    asyncio.run(run())
