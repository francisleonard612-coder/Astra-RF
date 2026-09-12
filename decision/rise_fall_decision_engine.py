"""
Rise/Fall decision engine core.

Pipeline, per evaluation:

1. PriceSeries (tick + minute log-returns) -> Hurst + vol-vs-baseline
   (regime/hurst_volatility.py) -> classify_regime()
   (decision/regime_conviction.py). RANGE_VOLATILE/NEUTRAL are skipped --
   "chop, no edge" and "no regime, no justified layer set" are legitimate
   reasons to sit out, independent of anything downstream.
2. Monte Carlo win-probability per (direction, duration-unit), via
   monte_carlo_duration() -- tick and minute returns scanned SEPARATELY,
   never pooled (see that module's docstring for why).
3. Each MC probability is calibrated through a per-symbol/side
   CalibrationTracker before being treated as a true probability.
4. Real Deriv quote + compute_edge() (pricing/edge.py, unmodified --
   already fully generic) -- mispricing against Deriv's ACTUAL payout is
   the trade gate, not the raw MC/calibrated probability alone.
5. DriftDetector.check_all() -- degraded halves stake (matching the
   source's own DRIFT_STAKE_REDUCTION=0.50 response: a caution signal,
   not a hard block, consistent with how mild a single drift fire is
   treated everywhere else in the ported material).

Deliberately NOT wired in here: decision/regime_conviction.py's
regime_votes()/compute_conviction() multi-layer voting and conviction
sizing. That machinery is designed for genuinely diverse, regime-
differentiated signals (Hurst, OU, Hawkes, RSI, ...) -- Astra only has TWO
signals for Rise/Fall right now (MC probability at tick and minute
resolution), and they're not independent evidence, they're the same method
at two granularities. Forcing them through compute_conviction()'s
conviction_min_voters>=3 default would mean it never fires at all; forcing
a lower threshold would misapply a tool built for a richer signal set than
exists yet. Stake here is a simple, flat, edge-independent size instead
(risk/staking.py's existing pattern) until Astra has enough genuinely
independent Rise/Fall signals to route through regime_conviction properly
-- wiring that in is a deliberate future decision, not an oversight.

Also NOT wired in here: OnlineMetaLearner (models/online_meta_learner.py)
-- same reasoning, needs a real feature vector this engine doesn't build
yet.

STAKE SIZING WARNING, carried forward from the source material's own
documented incident: base_stake here must be a FIXED value (e.g.
risk/staking.py's base_stake), never balance * some_percentage. The source
had a real production incident where a balance-scaled base stake and the
risk allocator's independent sizing disagreed by 527x, caught only by a
hard ceiling. Anchor stake to a fixed "smallest sane trade" and let the
risk engine bound it independently -- don't let two different parts of the
system both try to scale with balance.

MULTIPLE-COMPARISONS WARNING, found while testing this wiring (not present
in any single ported component -- it's an emergent property of how
evaluate() combines them): four candidates get evaluated per cycle (RISE
and FALL, at both tick and minute resolution), and the best-edge one is
selected. Each candidate independently has some chance of clearing
min_edge purely from MC estimation noise on a short recent-window sample
(the same phenomenon pricing/monte_carlo_duration.py's own tests
characterize -- a single realization's recent-window sample mean can look
like real drift even under zero true drift). Taking the max edge across
four such candidates inflates the overall false-positive rate well above
what min_edge alone would suggest for a single comparison -- measured
directly: on pure noise with fairly-priced quotes, roughly HALF of
independent histories produced a "trade" despite there being no real edge
anywhere. min_edge, min_calibration_samples, or n_sims likely all need to
be more conservative here than they would for evaluating a single candidate
in isolation, and this should be re-measured against real (not synthetic)
return data before trusting a specific min_edge value in production.

CONFIDENCE GATE: a candidate only clears this stage if BOTH its raw MC
win-probability and its calibrated probability clear min_confidence. Both
are already "probability of this contract_type's own favorable direction"
(RISE's raw_prob is P(price up), FALL's is P(price down) -- see
record_outcome()'s docstring), so no direction flip is needed for FALL.
Found from a live deployment log where every "Executing trade" line showed
mc_win_probability == calibrated_probability bit for bit -- calibration
hadn't collected enough samples to ever fit, so this gate was comparing the
same number to itself twice with no calibrator involved at all.

CALIBRATION-QUALITY GATE: once a candidate's calibrator HAS fit
(CalibrationTracker.is_calibrated == True), this additionally requires its
quality_score() (an ECE-based 0..1 reliability score) to clear
min_calibration_quality. Never enforced before a calibrator has fit, so it
doesn't block trading during a fresh symbol's cold start. 0.0 (the default)
disables this gate entirely.

MARTINGALE STAKING: routed through risk/staking.py's StakingEngine
(self.staking on RiseFallSymbolPipeline). OFF by default -- see that
module's own documented incident from a sibling bot before enabling it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ingestion.deriv_client import DerivClient
from models.calibration import CalibrationTracker
from models.drift_detector import DriftDetector
from pricing.contracts import FALL, RISE
from pricing.edge import compute_edge
from pricing.monte_carlo_duration import DEFAULT_MC_SIMULATIONS, monte_carlo_duration
from pricing.payout import ContractQuote, get_quote
from state.price_series import PriceSeries
from decision.regime_conviction import TRADEABLE_REGIMES, classify_regime
from regime.hurst_volatility import hurst_rs, realized_vol_and_baseline
from risk.staking import StakingEngine

DRIFT_STAKE_REDUCTION = 0.50  # matches the source's own response to a degraded DriftDetector


@dataclass
class RiseFallDecision:
    symbol: str
    decision: str  # "TRADE_RISE" | "TRADE_FALL" | "NO_TRADE"
    regime: str
    reason: str
    contract_type: str | None = None
    duration: int | None = None
    duration_unit: str | None = None
    stake: float | None = None
    edge: float | None = None
    mc_win_probability: float | None = None
    calibrated_probability: float | None = None
    drift_degraded: bool = False
    quote: ContractQuote | None = None  # the quote the decision was actually made against --
                                         # threaded through so a caller can build a TradeIntent
                                         # without a redundant refetch


def summary_reason(decision: RiseFallDecision) -> str:
    """A short, stable category for app/tick_summary.py's NO_TRADE-reason
    bucketing. decision.reason itself is a free-form descriptive sentence
    (good for the detailed "Executing trade" log line), not the short comma-
    joined token format tick_summary.extract_no_trade_reasons() expects --
    feeding the full sentence straight in would split on every comma inside
    it and produce useless, garbled buckets."""
    if decision.regime not in TRADEABLE_REGIMES:
        return f"regime:{decision.regime}"
    if "still awaiting settlement" in decision.reason:
        return "pending_settlement"
    if "below confidence gate" in decision.reason:
        return "insufficient_confidence"
    if "below calibration quality gate" in decision.reason:
        return "poor_calibration_quality"
    if "no candidate cleared min_edge" in decision.reason:
        return "insufficient_edge"
    return "unknown"


class RiseFallSymbolPipeline:
    def __init__(self, symbol: str, base_stake: float = 1.0,
                 min_edge: float = 0.03, min_calibration_samples: int = 200,
                 min_confidence: float = 0.70,
                 min_calibration_quality: float = 0.0,
                 staking_enabled: bool = False,
                 staking_progression_factor: float = 2.0,
                 staking_max_steps: int = 4,
                 staking_max_stake: float | None = None,
                 staking_min_consecutive_losses: int = 2):
        self.symbol = symbol
        self.base_stake = base_stake
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.min_calibration_quality = min_calibration_quality
        self.price_series = PriceSeries(symbol=symbol)
        self.calibration: dict[str, CalibrationTracker] = {
            RISE: CalibrationTracker(min_samples=min_calibration_samples),
            FALL: CalibrationTracker(min_samples=min_calibration_samples),
        }
        self.drift = DriftDetector()

        # staking_max_stake=None (the default whenever the config doesn't
        # set rise_fall.staking.max_stake) falls back to the stake the
        # configured progression would reach naturally at staking_max_steps,
        # so StakingEngine always gets a real numeric ceiling instead of
        # None (which would break its `min(stake, max_stake)` comparison).
        resolved_max_stake = (
            staking_max_stake if staking_max_stake is not None
            else round(base_stake * (staking_progression_factor ** staking_max_steps), 2)
        )
        self.staking = StakingEngine(
            base_stake=base_stake, enabled=staking_enabled,
            progression_factor=staking_progression_factor,
            max_steps=staking_max_steps, max_stake=resolved_max_stake,
            min_consecutive_losses_before_escalation=staking_min_consecutive_losses,
        )

        # Tracks ONLY the single candidate actually traded this cycle, not
        # every candidate evaluate() considered. "Did price go up" is
        # specific to a (duration, duration_unit) window -- price can rise
        # over 5 ticks and fall over 3 minutes in the same cycle, so a
        # single shared outcome can't correctly calibrate multiple
        # candidates from one evaluate() call. Restricting to the one
        # candidate that was actually bought sidesteps this: its outcome
        # (won/lost) is directly known from settlement, no separate
        # per-candidate price tracking needed.
        self._pending: tuple[str, float] | None = None  # (contract_type, raw_prob)

    def observe_tick(self, epoch: int, price: float) -> None:
        self.price_series.push(epoch, price)

    def cancel_pending(self) -> None:
        """Clears pending state WITHOUT recording any calibration outcome or
        feeding CUSUM -- for when evaluate() produced a trade decision but
        the actual buy never went through (quote unavailable at execution,
        stale-quote drift, the buy request itself being rejected, dry_run,
        or a settlement that came back with an unknown outcome/timeout).
        Recording a fabricated won/lost here would corrupt calibration with
        data that was never real. Without calling this in exactly these
        cases, the pending-slot guard at the top of evaluate() would
        permanently lock this symbol out -- nothing would ever call
        record_outcome() to clear it, since no contract was ever bought."""
        self._pending = None

    def record_outcome(self, won: bool) -> None:
        """Call once the trade this pipeline's last evaluate() call produced
        has settled. `won` alone is the correct calibration outcome, with NO
        sign adjustment for which side was traded: each contract_type's
        `raw_prob` (see evaluate()) is already a probability of THAT type's
        OWN favorable direction -- RISE's raw_prob is P(price up), FALL's is
        P(price down) -- so `won` directly answers "did the event raw_prob
        was a probability of actually happen" for whichever side was
        traded, with no flip required.

        No separate entry/exit spot lookup needed either: for a standard
        (non-equals) Rise/Fall contract on a continuous-price synthetic
        index, an exact tie has effectively zero probability, so `won`
        alone fully determines which direction the price actually moved.

        Also feeds the CUSUM drift check on the same outcome, and the
        martingale StakingEngine (if enabled) so a loss can escalate the
        next stake and a win can reset it.
        """
        if self._pending is not None:
            contract_type, raw_prob = self._pending
            outcome = 1 if won else 0
            self.calibration[contract_type].record(raw_prob, outcome)
            self._pending = None
            self.drift.update_cusum(won)
            self.staking.record_result(self.symbol, won)

    async def evaluate(self, client: DerivClient, currency: str,
                        tick_candidate_durations: list[int], minute_candidate_durations: list[int],
                        n_sims: int = DEFAULT_MC_SIMULATIONS,
                        rng: np.random.Generator | None = None) -> RiseFallDecision:
        rng = rng or np.random.default_rng()
        tick_returns = np.array(self.price_series.tick_log_returns, dtype=float)
        minute_returns = np.array(self.price_series.minute_log_returns, dtype=float)
        hurst = hurst_rs(minute_returns) if len(minute_returns) >= 50 else hurst_rs(tick_returns)
        sigma_now, sigma_baseline = realized_vol_and_baseline(
            minute_returns if len(minute_returns) >= 60 else tick_returns
        )
        regime, regime_reason = classify_regime(hurst, sigma_now, sigma_baseline)
        if regime not in TRADEABLE_REGIMES:
            return RiseFallDecision(self.symbol, "NO_TRADE", regime, regime_reason)

        if self._pending is not None:
            # A previous trade from this pipeline is still awaiting
            # settlement (execution/orders.py's OrderExecutor decouples
            # placing from settling -- a trade can genuinely still be
            # in-flight when evaluate() runs again on a later tick, if
            # risk_engine's max_concurrent_trades allows more than one open
            # position). _pending is a single shared slot, not a queue --
            # producing a new TRADE decision here would overwrite it before
            # record_outcome() fires for the FIRST trade, corrupting
            # calibration for both. One Rise/Fall position open at a time
            # per symbol avoids the race entirely; risk_engine's own
            # max_concurrent_trades cap still governs how many symbols can
            # have a position open simultaneously.
            return RiseFallDecision(self.symbol, "NO_TRADE", regime,
                                     "previous trade still awaiting settlement")

        # (decision, edge, raw_prob) for the best candidate found so far --
        # raw_prob is carried alongside purely so it can be stashed in
        # self._pending once a final winner is chosen, without recomputing
        # or re-running the MC scan.
        best: tuple[RiseFallDecision, float, float] | None = None
        for returns, durations, unit in (
            (tick_returns, tick_candidate_durations, "t"),
            (minute_returns, minute_candidate_durations, "m"),
        ):
            if len(returns) < 20 or not durations:
                continue

            for direction, contract_type in ((1, RISE), (-1, FALL)):
                dur, raw_p = monte_carlo_duration(returns, direction, durations, n_sims=n_sims, rng=rng)

                # CONFIDENCE GATE -- checked on the raw MC estimate first
                # (cheap, no calibrator call needed) before spending a
                # get_quote() call on a candidate that can't pass regardless
                # of price. See module docstring for why no direction flip
                # is needed here.
                if raw_p < self.min_confidence:
                    continue

                tracker = self.calibration[contract_type]
                calibrated_p = tracker.calibrate(raw_p)
                if calibrated_p < self.min_confidence:
                    continue

                # CALIBRATION-QUALITY GATE -- only enforced once this
                # contract_type's calibrator has actually fit, and only if
                # the gate is enabled (min_calibration_quality > 0.0). See
                # module docstring for the "compared the same number to
                # itself twice" bug this specifically catches.
                if (self.min_calibration_quality > 0.0
                        and getattr(tracker, "is_calibrated", False)
                        and tracker.quality_score() < self.min_calibration_quality):
                    continue

                quote = await get_quote(client, self.symbol, contract_type, None, self.base_stake,
                                         dur, unit, currency)
                if quote is None:
                    continue

                edge_result = compute_edge(quote, calibrated_p)
                if edge_result.edge < self.min_edge:
                    continue

                if best is None or edge_result.edge > best[1]:
                    # Stake now comes from StakingEngine (flat base_stake
                    # whenever staking is disabled, escalated per
                    # risk/staking.py's progression otherwise).
                    stake = self.staking.current_stake(self.symbol)
                    live_confidence = calibrated_p if direction > 0 else (1 - calibrated_p)
                    degraded = self.drift.check_all(returns, live_confidence)
                    if degraded:
                        stake = round(stake * DRIFT_STAKE_REDUCTION, 2)

                    if stake != self.base_stake:
                        # Re-fetch at the FINAL stake, not the base_stake
                        # quote already in hand: OrderExecutor re-quotes at
                        # intent.stake right before buying and compares
                        # payouts to detect a stale quote (see
                        # execution/orders.py). Payout scales with stake for
                        # these contracts, so leaving intent.quote at the
                        # base_stake amount while intent.stake is a
                        # drift-reduced or staking-escalated amount would
                        # look like a payout mismatch -- OrderExecutor would
                        # reject the trade as stale every time either
                        # applies, silently defeating "trade a different
                        # size", not "don't trade at all".
                        final_quote = await get_quote(client, self.symbol, contract_type, None,
                                                       stake, dur, unit, currency)
                        if final_quote is None:
                            continue
                    else:
                        final_quote = quote

                    decision = RiseFallDecision(
                        self.symbol, f"TRADE_{'RISE' if direction > 0 else 'FALL'}", regime,
                        f"{regime_reason}; edge={edge_result.edge:.4f} at {dur}{unit}",
                        contract_type=contract_type, duration=dur, duration_unit=unit,
                        stake=stake, edge=edge_result.edge,
                        mc_win_probability=raw_p, calibrated_probability=calibrated_p,
                        drift_degraded=degraded, quote=final_quote,
                    )
                    best = (decision, edge_result.edge, raw_p)

        if best is None:
            return RiseFallDecision(self.symbol, "NO_TRADE", regime,
                                     f"{regime_reason}; no candidate cleared min_edge={self.min_edge}")

        decision, _, raw_p = best
        # Stash pending state ONLY for the candidate actually being traded --
        # see record_outcome()'s docstring for why calibrating any
        # non-traded candidate evaluated this same cycle would be wrong
        # (their outcomes aren't the same fact, since they can span
        # different durations/resolutions).
        self._pending = (decision.contract_type, raw_p)
        return decision
