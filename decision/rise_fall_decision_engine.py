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
  3. Each MC probability is calibrated through a per-symbol/side/resolution
     CalibrationTracker before being treated as a true probability (tick and
     minute candidates for the same side are calibrated SEPARATELY -- see
     "PER-RESOLUTION CALIBRATION" below).
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

CONFIDENCE GATE (min_confidence, default 0.70): a second, independent gate
alongside min_edge. min_edge alone accepts a candidate whenever the
calibrated probability clears Deriv's breakeven probability by min_edge --
which, at a generously-priced quote, can fire at a fairly low absolute
probability (e.g. calibrated_p=0.55 against a breakeven of 0.45 clears
min_edge=0.06 easily). This gate additionally requires BOTH the raw MC
win-probability estimate (mc_win_probability) AND the calibrated
probability to independently exceed min_confidence before a candidate is
even considered -- checked before the real quote fetch, so a candidate that
fails it never spends an API call. Requiring both (not just the calibrated
one) means a candidate can't pass on calibration alone if the underlying MC
simulation itself was unconvincing, and vice versa. This does not replace
min_edge -- a candidate must still clear both gates.

PER-RESOLUTION CALIBRATION: self.calibration is keyed by (contract_type,
duration_unit) -- (RISE,"t"), (RISE,"m"), (FALL,"t"), (FALL,"m") -- four
independent CalibrationTrackers per symbol, not two. An earlier version
pooled tick and minute candidates for the same side into one tracker per
contract_type; that's wrong for the same reason
pricing/monte_carlo_duration.py's own docstring gives for never mixing tick
and minute returns into one MC call -- they're different sampling
processes with different noise characteristics, so a raw_prob of 0.8 from a
5-tick candidate and a raw_prob of 0.8 from a 3-minute candidate are not
draws from the same underlying reliability curve, and pooling them dilutes
(and can actively distort) both curves' fit compared to calibrating them
separately. This does mean each tracker sees roughly a quarter of a
symbol's settled trades rather than half, so min_calibration_samples worth
of real data now takes proportionally longer to accumulate per tracker --
see "CALIBRATION WARM-START" below for closing that gap without waiting on
live trades alone.

CALIBRATION-QUALITY GATE (min_calibration_quality, default 0.0/disabled):
found from a live deployment log where EVERY "Executing trade" line showed
mc_win_probability == calibrated_probability bit for bit -- the calibrator
hadn't collected enough samples to ever fit, so min_confidence was
comparing the same number to itself twice, not two independent signals.
Once a candidate's CalibrationTracker HAS fit (is_calibrated=True), this
gate additionally requires its quality_score() (models/calibration.py's
ECE-based 0..1 reliability score) to clear min_calibration_quality before
that candidate is considered -- a fitted-but-poorly-calibrated curve is
arguably worse than no calibration at all, since it looks authoritative
while still being wrong. Deliberately does NOT apply before is_calibrated
is True: quality_score() returns a fixed low placeholder before 30 Brier
samples exist regardless of is_calibrated, and gating on that placeholder
would just block all trading during every symbol's cold start for no
real reason. Off by default (0.0) because it's a new behavior change on
top of an already-conservative default confidence gate; tick_summary now
logs each tracker's quality_score(), so a sensible non-zero value can be
chosen from real observed numbers instead of guessed blind.

PROBE TRADES (calibration_quality_probe_interval, default 50) -- fixes a
lockout the quality gate above otherwise creates: a candidate blocked by
quality_score() never gets a real quote, never gets stashed in
self._pending, and therefore never settles into a fresh calibration
sample -- record_outcome() has nothing to record for it, so quality_score()
can never move off whatever bad reading tripped the gate in the first
place. Left alone, that's a PERMANENT lock for that specific
(contract_type, resolution), fixable only from outside (a fresh warm-start
or Supabase restore overwriting its buffer). Instead: every consecutive
cycle a given (contract_type, resolution) candidate gets blocked by the
quality gate increments a per-key streak counter
(self._quality_gate_blocked_streak); once that streak reaches
calibration_quality_probe_interval, the NEXT such candidate is let through
past the quality gate only -- min_confidence and min_edge still apply in
full below, so a probe can't trade purely because it's "due" for one, it
still has to look genuinely tradeable by every other measure. The streak is
NOT reset just because a probe was attempted -- only once a probe actually
becomes the traded decision (self._pending gets set for that exact key,
meaning a real settlement will eventually feed calibration.record() a fresh
sample). A probe that gets crowded out by a better non-probe candidate, or
that fails confidence/edge/quote after clearing the quality gate, keeps its
streak exactly where it was and is eligible to try again next cycle --
deliberately more persistent than a fixed period, since the point is
"keep trying until it actually gets a fresh sample," not "try once every N
cycles regardless of whether that attempt achieved anything." A probe's
stake is forced to base_stake regardless of martingale escalation (see
"MARTINGALE STAKING" below) -- being on probation because its own
calibration reliability is in question is exactly the wrong moment to also
be trading an escalated size. calibration_quality_probe_interval=0 disables
probing entirely, restoring the original permanent-lockout behavior for
anyone who deliberately wants that.

CALIBRATION WARM-START (research/calibration_warmstart.py,
scripts/warm_start_calibration.py): waiting for live trades alone to fill
four separate per-resolution calibrators (500 samples each by default) at
a few trades an hour could take weeks. Historical tick data Astra can
already fetch (the same client.get_history() call app/main.py's
seed_symbol() uses to warm PriceSeries) contains everything needed to
compute the exact same (raw MC probability, actual outcome) pairs a live
trade would eventually produce -- replay past ticks through the same
monte_carlo_duration() call evaluate() itself uses, then look at the
(already-known, historical) future ticks to see whether the predicted
direction actually happened. This is NOT a backtest of trading performance
(no edge, no stake sizing, no P&L, no min_edge/min_confidence gating) --
it only produces (raw_prob, outcome) pairs to pre-fill each
CalibrationTracker's buffers via the exact same .record() call a live
settlement uses, so a symbol's calibrators have something to work with
from its FIRST live trade onward instead of starting cold. See
app/main.py's calibration_warm_start_dir wiring for how this gets loaded
at startup, and both files' own docstrings for the replay logic and CLI.

MARTINGALE STAKING (opt-in, off by default): stake sizing can be routed
through risk/staking.py's StakingEngine (self.staking) instead of always
trading base_stake flat. See that module's own docstring for the
documented reason this is opt-in, not default, elsewhere in this account.
When enabled, self.staking.current_stake(symbol) replaces base_stake as the
starting point for a trade's stake. The progression does NOT engage after a
single isolated loss: it only starts escalating once a SECOND consecutive
loss follows the first (staking_min_consecutive_losses=2 by default here --
see StakingEngine's min_consecutive_losses_before_escalation), and from
there escalates by progression_factor on each further consecutive loss
(capped at staking_max_steps escalations and staking_max_stake in absolute
terms), resetting to base_stake on any win. record_outcome() feeds it the
same settled won/lost outcome used for calibration. Critically, this means
the comparison quote fetched per candidate (at base_stake, purely to
compute a stake-invariant edge) can no longer be assumed to match the
actual trade's stake once staking has moved away from base_stake -- exactly
the same "quote basis must match decision.stake" hazard the drift-reduction
path below was already written to avoid, so the final-quote re-fetch
condition now checks for either cause, not just drift degradation. As with
the STAKE SIZING WARNING above: this progression multiplies base_stake, it
never reads account balance, so it can't reproduce that specific incident --
but it can absolutely still lose money faster than flat staking if the
underlying edge doesn't hold, per risk/staking.py's own live-trade evidence
from a sibling bot. The risk engine's max_stake remains the final,
independent backstop regardless of what this progression computes (see
app/main.py, which now checks risk_engine against decision.stake -- the
actual post-staking, post-drift-reduction amount -- rather than a
pre-evaluate estimate).
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
from risk.staking import StakingEngine
from state.price_series import PriceSeries

from decision.regime_conviction import TRADEABLE_REGIMES, classify_regime
from regime.hurst_volatility import hurst_rs, realized_vol_and_baseline

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
    is_calibration_probe: bool = False  # True if this candidate only got through because the
                                         # "PROBE TRADES" escape hatch bypassed the quality gate --
                                         # see decision/rise_fall_decision_engine.py's module docstring


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
    if "no candidate cleared min_calibration_quality" in decision.reason:
        return "poor_calibration_quality"
    if "no candidate cleared min_confidence" in decision.reason:
        return "insufficient_confidence"
    if "no candidate cleared min_edge" in decision.reason:
        return "insufficient_edge"
    return "unknown"


class RiseFallSymbolPipeline:
    def __init__(self, symbol: str, base_stake: float = 1.0,
                 min_edge: float = 0.03, min_calibration_samples: int = 200,
                 min_confidence: float = 0.70, min_calibration_quality: float = 0.0,
                 calibration_quality_probe_interval: int = 50,
                 staking_enabled: bool = False, staking_progression_factor: float = 2.0,
                 staking_max_steps: int = 4, staking_max_stake: float | None = None,
                 staking_min_consecutive_losses: int = 2):
        self.symbol = symbol
        self.base_stake = base_stake
        self.min_edge = min_edge
        # Confidence gate: a candidate must clear BOTH its raw MC
        # win-probability estimate and its calibrated probability -- see
        # this module's "CONFIDENCE GATE" docstring above.
        self.min_confidence = min_confidence
        # Calibration-quality gate: only enforced once a candidate's
        # calibrator has actually fit -- see "CALIBRATION-QUALITY GATE"
        # docstring above. 0.0 disables this gate (the default).
        self.min_calibration_quality = min_calibration_quality
        # Probe-trade escape hatch for the lockout the quality gate above
        # would otherwise create -- see "PROBE TRADES" docstring above. 0
        # disables probing (permanent lockout once blocked).
        self.calibration_quality_probe_interval = calibration_quality_probe_interval
        self.price_series = PriceSeries(symbol=symbol)
        # Keyed by (contract_type, duration_unit) -- FOUR independent
        # trackers per symbol, not two. See "PER-RESOLUTION CALIBRATION"
        # docstring above for why tick and minute candidates for the same
        # side must not share a calibrator.
        self.calibration: dict[tuple[str, str], CalibrationTracker] = {
            (RISE, "t"): CalibrationTracker(min_samples=min_calibration_samples),
            (RISE, "m"): CalibrationTracker(min_samples=min_calibration_samples),
            (FALL, "t"): CalibrationTracker(min_samples=min_calibration_samples),
            (FALL, "m"): CalibrationTracker(min_samples=min_calibration_samples),
        }
        # How many consecutive cycles each (contract_type, unit) candidate
        # has been blocked by the quality gate -- see "PROBE TRADES"
        # docstring above for exactly when this increments vs. resets.
        self._quality_gate_blocked_streak: dict[tuple[str, str], int] = {key: 0 for key in self.calibration}
        self.drift = DriftDetector()
        # Opt-in martingale staking -- see this module's "MARTINGALE
        # STAKING" docstring above and risk/staking.py's own warning.
        # Defaults to disabled (flat base_stake), matching Astra's
        # established default-off posture for progression staking.
        self.staking = StakingEngine(
            base_stake=base_stake,
            enabled=staking_enabled,
            progression_factor=staking_progression_factor,
            max_steps=staking_max_steps,
            max_stake=(staking_max_stake if staking_max_stake is not None
                       else base_stake * (staking_progression_factor ** staking_max_steps)),
            # A single isolated loss does NOT escalate the stake -- the
            # progression only engages once a second loss follows
            # consecutively. See risk/staking.py's StakingEngine docstring.
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
        self._pending: tuple[str, str, float] | None = None  # (contract_type, unit, raw_prob)

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
        traded, with no flip required. (An earlier version of this method
        flipped the outcome for FALL specifically, reasoning from "did price
        go up" as a shared fact -- wrong, and caught by
        test_record_outcome_calibrates_only_the_traded_candidate_fall:
        FALL's raw_prob was never a probability of price going up in the
        first place, so there was nothing to flip.)

        No separate entry/exit spot lookup needed either: for a standard
        (non-equals) Rise/Fall contract on a continuous-price synthetic
        index, an exact tie has effectively zero probability, so `won`
        alone fully determines which direction the price actually moved.

        Also feeds the CUSUM drift check on the same outcome.
        """
        if self._pending is not None:
            contract_type, unit, raw_prob = self._pending
            outcome = 1 if won else 0
            self.calibration[(contract_type, unit)].record(raw_prob, outcome)
            # Same restriction as calibration above: only a real, settled
            # outcome for the candidate actually traded should move the
            # martingale progression. A no-op when staking is disabled
            # (StakingEngine.current_stake always returns base_stake then).
            self.staking.record_result(self.symbol, won)
        self._pending = None
        self.drift.update_cusum(won)

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
        # Tracked purely for a more precise NO_TRADE reason below -- lets
        # summary_reason() (and app/tick_summary.py's per-gate breakdown)
        # distinguish "nothing was even confident enough to price" from
        # "priced fine, but no real edge" instead of collapsing both into
        # one generic bucket.
        any_cleared_quality = False
        any_cleared_confidence = False
        for returns, durations, unit in (
            (tick_returns, tick_candidate_durations, "t"),
            (minute_returns, minute_candidate_durations, "m"),
        ):
            if len(returns) < 20 or not durations:
                continue
            for direction, contract_type in ((1, RISE), (-1, FALL)):
                dur, raw_p = monte_carlo_duration(returns, direction, durations, n_sims=n_sims, rng=rng)
                cal = self.calibration[(contract_type, unit)]
                calibrated_p = cal.calibrate(raw_p)
                streak_key = (contract_type, unit)

                # CALIBRATION-QUALITY GATE: only enforced once this specific
                # (contract_type, resolution) calibrator has actually fit --
                # see "CALIBRATION-QUALITY GATE" docstring above for why not
                # before. is_probe tracks whether THIS candidate only got
                # through via the "PROBE TRADES" escape hatch -- see that
                # docstring for exactly when the streak that gates it
                # increments vs. resets (NOT simply "attempted a probe").
                is_probe = False
                if cal.is_calibrated and cal.quality_score() < self.min_calibration_quality:
                    self._quality_gate_blocked_streak[streak_key] += 1
                    if (self.calibration_quality_probe_interval <= 0
                            or self._quality_gate_blocked_streak[streak_key]
                            < self.calibration_quality_probe_interval):
                        continue
                    is_probe = True
                else:
                    self._quality_gate_blocked_streak[streak_key] = 0
                any_cleared_quality = True

                # CONFIDENCE GATE: both the raw MC estimate and the
                # calibrated probability must independently clear
                # min_confidence before this candidate is even worth a real
                # quote fetch. See this module's "CONFIDENCE GATE"
                # docstring for why both, not just the calibrated one.
                if raw_p <= self.min_confidence or calibrated_p <= self.min_confidence:
                    continue
                any_cleared_confidence = True

                quote = await get_quote(client, self.symbol, contract_type, None, self.base_stake,
                                         dur, unit, currency)
                if quote is None:
                    continue
                edge_result = compute_edge(quote, calibrated_p)
                if edge_result.edge < self.min_edge:
                    continue
                if best is None or edge_result.edge > best[1]:
                    # Martingale (opt-in, see "MARTINGALE STAKING" docstring
                    # above) replaces the flat base_stake starting point --
                    # a no-op back to base_stake whenever staking is
                    # disabled or the progression is currently at step 0.
                    # A probe candidate skips this entirely and always
                    # trades base_stake -- see "PROBE TRADES" docstring:
                    # being on probation because this candidate's own
                    # calibration reliability is in question is exactly the
                    # wrong moment to also be trading an escalated size.
                    stake = self.base_stake if is_probe else self.staking.current_stake(self.symbol)
                    live_confidence = calibrated_p if direction > 0 else (1 - calibrated_p)
                    degraded = self.drift.check_all(returns, live_confidence)
                    if degraded:
                        stake = round(stake * DRIFT_STAKE_REDUCTION, 2)
                    if stake != self.base_stake:
                        # Re-fetch at the FINAL stake, not the base_stake
                        # quote already in hand -- needed whenever stake has
                        # moved away from base_stake for ANY reason (drift
                        # reduction above, or martingale escalation/
                        # reduction just above it). OrderExecutor re-quotes
                        # at intent.stake right before buying and compares
                        # payouts to detect a stale quote (see
                        # execution/orders.py); payout scales with stake for
                        # these contracts, so leaving intent.quote at the
                        # base_stake amount while intent.stake is a
                        # different amount would look like a stale-quote
                        # payout mismatch and get the trade rejected at
                        # execution time -- or, worse for martingale
                        # specifically, silently understate the payout a
                        # calibration/EV figure was computed against.
                        final_quote = await get_quote(client, self.symbol, contract_type, None,
                                                       stake, dur, unit, currency)
                        if final_quote is None:
                            continue
                    else:
                        final_quote = quote
                    decision = RiseFallDecision(
                        self.symbol, f"TRADE_{'RISE' if direction > 0 else 'FALL'}", regime,
                        f"{regime_reason}; edge={edge_result.edge:.4f} at {dur}{unit}"
                        + (" [calibration probe]" if is_probe else ""),
                        contract_type=contract_type, duration=dur, duration_unit=unit,
                        stake=stake, edge=edge_result.edge,
                        mc_win_probability=raw_p, calibrated_probability=calibrated_p,
                        drift_degraded=degraded, quote=final_quote,
                        is_calibration_probe=is_probe,
                    )
                    best = (decision, edge_result.edge, raw_p)

        if best is None:
            if not any_cleared_quality:
                return RiseFallDecision(
                    self.symbol, "NO_TRADE", regime,
                    f"{regime_reason}; no candidate cleared min_calibration_quality={self.min_calibration_quality}")
            if not any_cleared_confidence:
                return RiseFallDecision(
                    self.symbol, "NO_TRADE", regime,
                    f"{regime_reason}; no candidate cleared min_confidence={self.min_confidence}")
            return RiseFallDecision(self.symbol, "NO_TRADE", regime,
                                     f"{regime_reason}; no candidate cleared min_edge={self.min_edge}")
        decision, _, raw_p = best
        # Stash pending state ONLY for the candidate actually being traded --
        # see record_outcome()'s docstring for why calibrating any
        # non-traded candidate evaluated this same cycle would be wrong
        # (their outcomes aren't the same fact, since they can span
        # different durations/resolutions).
        self._pending = (decision.contract_type, decision.duration_unit, raw_p)
        if decision.is_calibration_probe:
            # Only reset the blocked-streak once a probe ACTUALLY becomes
            # the traded decision (guaranteed a real settlement, and
            # therefore a fresh calibration sample, is coming) -- see
            # "PROBE TRADES" docstring above for why a merely-attempted
            # probe that got crowded out or failed a later gate must not
            # reset this.
            self._quality_gate_blocked_streak[(decision.contract_type, decision.duration_unit)] = 0
        return decision
