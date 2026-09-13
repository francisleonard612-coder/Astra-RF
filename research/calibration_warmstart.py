"""
Offline calibration warm-start for Rise/Fall.

WHY THIS EXISTS: a live deployment log showed every single "Executing
trade" line reporting calibrated_probability == mc_win_probability bit for
bit -- the calibrator hadn't collected enough samples (min_calibration_
samples, default 200-500) to ever fit, so decision/rise_fall_decision_
engine.py's confidence gate was comparing the exact same number to itself
twice, not two independent signals. Waiting for that many REAL settled
trades to accumulate live, split four ways across (contract_type,
resolution) since that change (see that module's "PER-RESOLUTION
CALIBRATION"), at maybe a few trades an hour, could take weeks before
calibration engages at all.

Historical tick data Astra can already fetch (the same client.get_history()
call app/main.py's seed_symbol() uses to warm PriceSeries) contains
everything needed to compute the exact same (raw MC probability, actual
outcome) pairs a live trade would eventually produce, WITHOUT waiting for a
live trade to settle: replay past ticks through the same
monte_carlo_duration() call evaluate() itself uses at each point, then look
at the (already-known, historical) future ticks to see whether the
predicted direction actually happened.

This is intentionally NOT a backtest of trading performance -- no edge, no
stake sizing, no P&L, no min_edge/min_confidence/min_calibration_quality
gating, no risk engine. It only produces (raw_prob, outcome) pairs to
pre-fill CalibrationTracker's buffers via the exact same .record() call a
live settlement uses (see apply_samples_to_pipeline below), so a symbol's
calibrators have real material to work with from its FIRST live trade
onward instead of starting cold. See backtest/warm_start_calibration.py for
the CLI that fetches history from Deriv and calls build_calibration_samples
below, and app/main.py's calibration_warm_start_dir wiring for how the
output gets loaded back in at startup.

NO LOOK-AHEAD WITHIN A SAMPLE, BUT THIS DOES USE FUTURE DATA ACROSS THE
REPLAY: at replay position i, the MC estimate uses only returns[:i] (exactly
what evaluate() would have seen live at that point), but the recorded
OUTCOME for that estimate necessarily reads returns[i:i+dur] -- ticks that,
in a live run, wouldn't exist yet. That's unavoidable and fine here (the
whole point is to use history we already have to manufacture calibration
samples faster), but it's why this must never be repurposed as a trading
backtest without also adding the edge/stake/gating logic evaluate() applies
live -- see backtest/simulator.py's own docstring for the same caution
about a different pipeline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from pricing.contracts import FALL, RISE
from pricing.monte_carlo_duration import DEFAULT_MC_SIMULATIONS, monte_carlo_duration
from state.price_series import PriceSeries

MIN_HISTORY_LENGTH = 20  # matches evaluate()'s own len(returns) < 20 skip

CalibrationKey = tuple[str, str]  # (contract_type, duration_unit)
SamplePairs = list[tuple[float, int]]  # [(raw_prob, outcome), ...]


def build_calibration_samples(
    ticks: list[tuple[int, float]],
    tick_candidate_durations: list[int],
    minute_candidate_durations: list[int],
    n_sims: int = DEFAULT_MC_SIMULATIONS,
    sample_every: int = 5,
    rng: np.random.Generator | None = None,
) -> dict[CalibrationKey, SamplePairs]:
    """
    ticks: chronological (epoch, price) pairs -- exactly what
        ingestion.deriv_client.DerivClient.get_history() returns per tick,
        just unpacked to plain tuples so this function has no network/client
        dependency and is trivially testable with synthetic data.
    sample_every: only replay every Nth valid position, not every single
        one -- controls cost, since each position runs up to 2 MC
        simulations (RISE and FALL) per resolution. Astra's live evaluate()
        runs on every tick; this doesn't need to match that density to be
        useful, since the goal is filling a calibration buffer, not
        reproducing live cadence.

    Returns {(contract_type, duration_unit): [(raw_prob, outcome), ...]}
    for all four keys decision/rise_fall_decision_engine.py's
    RiseFallSymbolPipeline.calibration uses. A key with an empty list means
    there wasn't enough history (or duration candidates) to produce any
    samples for it -- not an error.
    """
    rng = rng or np.random.default_rng()
    n = len(ticks)

    # max_tick_window/max_minute_window default to a few thousand -- large
    # enough for live tick-by-tick operation, but a warm-start replay can
    # easily exceed that if handed a big history batch. Size both windows to
    # the input so PriceSeries's internal deques never silently truncate
    # ticks out from under this replay.
    ps = PriceSeries(symbol="_calibration_warmstart", max_tick_window=n + 10, max_minute_window=n + 10)
    for epoch, price in ticks:
        ps.push(epoch, price)

    tick_returns = np.array(ps.tick_log_returns, dtype=float)
    minute_returns = np.array(ps.minute_log_returns, dtype=float)

    samples: dict[CalibrationKey, SamplePairs] = {
        (RISE, "t"): [], (FALL, "t"): [], (RISE, "m"): [], (FALL, "m"): [],
    }

    for returns, durations, unit in (
        (tick_returns, tick_candidate_durations, "t"),
        (minute_returns, minute_candidate_durations, "m"),
    ):
        if not durations or len(returns) < MIN_HISTORY_LENGTH:
            continue
        max_dur = max(durations)
        i = MIN_HISTORY_LENGTH
        while i + max_dur <= len(returns):
            history = returns[:i]
            for direction, contract_type in ((1, RISE), (-1, FALL)):
                dur, raw_p = monte_carlo_duration(history, direction, durations, n_sims=n_sims, rng=rng)
                # log(price[i+dur] / price[i]) == sum of the intervening
                # log-returns -- no need to reconstruct raw price levels to
                # know which way price actually moved over the window.
                future_move = float(np.sum(returns[i:i + dur]))
                if direction > 0:
                    outcome = 1 if future_move > 0 else 0
                else:
                    outcome = 1 if future_move < 0 else 0
                samples[(contract_type, unit)].append((raw_p, outcome))
            i += sample_every

    return samples


def samples_to_jsonable(samples: dict[CalibrationKey, SamplePairs]) -> dict[str, dict[str, list]]:
    """(contract_type, unit) tuple keys -> "CALL_t"/"PUT_m" string keys,
    since JSON object keys must be strings."""
    out: dict[str, dict[str, list]] = {}
    for (contract_type, unit), pairs in samples.items():
        out[f"{contract_type}_{unit}"] = {
            "raw": [p[0] for p in pairs],
            "outcome": [p[1] for p in pairs],
        }
    return out


def jsonable_to_samples(data: dict[str, dict[str, list]]) -> dict[CalibrationKey, SamplePairs]:
    out: dict[CalibrationKey, SamplePairs] = {}
    for key, payload in data.items():
        contract_type, unit = key.rsplit("_", 1)
        out[(contract_type, unit)] = list(zip(payload["raw"], payload["outcome"]))
    return out


def save_samples_json(samples: dict[CalibrationKey, SamplePairs], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(samples_to_jsonable(samples), f)


def load_samples_json(path: str) -> dict[CalibrationKey, SamplePairs]:
    with open(path, "r") as f:
        return jsonable_to_samples(json.load(f))


def extract_samples_from_pipeline(pipeline) -> dict[CalibrationKey, SamplePairs]:
    """Inverse of apply_samples_to_pipeline(): reads each of pipeline's
    CalibrationTrackers' current (raw, outcome) buffers back out. Used to
    persist a pipeline's LIVE, continuously-updated calibration state (e.g.
    to Supabase via database/repository.py's save_rise_fall_calibration_
    state, called after every settlement in app/main.py) -- distinct from
    the one-off historical replay build_calibration_samples() produces.
    Same duck-typing as apply_samples_to_pipeline: any object with a
    `.calibration` dict keyed by (contract_type, unit) -> CalibrationTracker
    works, no import of decision/rise_fall_decision_engine.py needed here.
    """
    return {key: list(zip(tracker._raw, tracker._outcome)) for key, tracker in pipeline.calibration.items()}


def apply_samples_to_pipeline(pipeline, samples: dict[CalibrationKey, SamplePairs]) -> dict[str, int]:
    """Feeds each (raw_prob, outcome) pair into the matching per-
    (contract_type, resolution) CalibrationTracker on `pipeline` via
    .record(), IN ORDER -- this is exactly the call a live settlement makes
    (see decision/rise_fall_decision_engine.py's record_outcome()), so it
    naturally triggers the tracker's own refit logic (min_samples/
    refit_every) exactly as if these had been real, live, settled trades.
    No special-casing needed here for how/when a tracker fits.

    `pipeline` is duck-typed (any object with a `.calibration` dict keyed by
    (contract_type, unit) -> CalibrationTracker) rather than type-hinted
    against RiseFallSymbolPipeline directly, so this module never needs to
    import decision/rise_fall_decision_engine.py -- avoids a circular
    import, since that module doesn't need to know this one exists.

    Returns {"CALL_t": n, ...} counts applied per key, for logging. A key
    present in `samples` but not on `pipeline.calibration` is skipped
    rather than raising -- keeps this forward-compatible with a warm-start
    file produced by an older/newer version of this module without an
    exact key-set match.
    """
    applied: dict[str, int] = {}
    for (contract_type, unit), pairs in samples.items():
        tracker = pipeline.calibration.get((contract_type, unit))
        if tracker is None:
            continue
        for raw_prob, outcome in pairs:
            tracker.record(raw_prob, outcome)
        applied[f"{contract_type}_{unit}"] = len(pairs)
    return applied


async def restore_or_generate_calibration(
    repo, pipeline, symbol: str, *,
    warm_start_dir: str | None,
    ticks: list[tuple[int, float]],
    tick_candidate_durations: list[int],
    minute_candidate_durations: list[int],
    auto_generate: bool = False,
    n_sims: int = DEFAULT_MC_SIMULATIONS,
    sample_every: int = 5,
    to_thread=None,
    on_file_error=None,
) -> tuple[str, dict[str, int]]:
    """One symbol's full calibration-restore priority chain, pulled out of
    app/main.py's startup sequence into a standalone function purely so it's
    unit-testable without booting the whole app (real Supabase client, real
    Deriv connection, etc.) -- app/main.py's main() just calls this once per
    symbol and logs based on the returned outcome tag.

    Priority order (stops at the first that produces something):
      1. Supabase (repo.load_rise_fall_calibration_state) -- the bot's own
         continuously-updated state from every previous settlement. Near-
         instant: just replays stored pairs through .record(), no MC.
      2. The file-based warm-start at {warm_start_dir}/{symbol}.json, if
         warm_start_dir is set and that file exists. A corrupt/unreadable
         file calls on_file_error(path, exc) if provided (app/main.py wires
         its logger through this -- this module never logs directly, same
         convention as the rest of it) and falls through to step 3/4 rather
         than raising -- a bad warm-start file must never block startup.
      3. Auto-generate (only if auto_generate=True AND ticks is non-empty):
         replay `ticks` through build_calibration_samples() -- genuinely
         CPU-bound -- and save the result to Supabase immediately, so this
         is a one-time cost per symbol; every later call hits step 1
         instead. Run through `to_thread` (defaults to asyncio.to_thread)
         rather than awaited inline, so a long replay can't block the
         event loop; tests can pass a synchronous stand-in to avoid a real
         thread hop.
      4. Nothing available -- pipeline starts cold, same as if none of this
         existed.

    Returns (outcome, applied) where outcome is one of "supabase", "file",
    "auto_generated", "cold" and applied is the same {"CALL_t": n, ...}
    dict apply_samples_to_pipeline() returns ({} for "cold").
    """
    if to_thread is None:
        import asyncio
        to_thread = asyncio.to_thread

    supabase_state = repo.load_rise_fall_calibration_state(symbol)
    if supabase_state:
        return "supabase", apply_samples_to_pipeline(pipeline, jsonable_to_samples(supabase_state))

    if warm_start_dir:
        path = os.path.join(warm_start_dir, f"{symbol}.json")
        if os.path.exists(path):
            try:
                samples = load_samples_json(path)
                return "file", apply_samples_to_pipeline(pipeline, samples)
            except Exception as exc:  # noqa: BLE001 -- a bad warm-start file must never block startup
                if on_file_error is not None:
                    on_file_error(path, exc)

    if auto_generate and ticks:
        samples = await to_thread(
            build_calibration_samples, ticks, tick_candidate_durations, minute_candidate_durations,
            n_sims=n_sims, sample_every=sample_every,
        )
        applied = apply_samples_to_pipeline(pipeline, samples)
        repo.save_rise_fall_calibration_state(symbol, samples_to_jsonable(samples))
        return "auto_generated", applied

    return "cold", {}
