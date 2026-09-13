"""
Astra entrypoint -- Rise/Fall.

Startup sequence:
  1. connect to Deriv (auto-resolving a demo or real account -- see
     app/config.py DerivConfig.use_real_account)
  2. discover symbols: ASTRA_SYMBOLS (see .env.example) or, if unset, every
     R_*/1HZ* synthetic index Deriv offers
  3. seed each symbol's PriceSeries from recent tick history where available
  4. subscribe to live ticks for every symbol
  5. spawn one independent worker task per symbol (no cross-symbol blocking)
  6. each worker: observe tick -> evaluate (regime -> MC -> calibration ->
     drift -> real-quote edge/EV, see decision/rise_fall_decision_engine.py)
     -> (maybe) trade -> repeat

This replaces the earlier digit-trading main.py entirely (see configs/
config.yaml's "LEGACY" section for what that left behind, still present but
unused). Monitoring dashboard/alerting is intentionally out of scope for
this build -- see app/logging_setup.py.
"""
from __future__ import annotations

import asyncio
import signal
import time

from app.config import get_config
from app.logging_setup import configure_logging, get_logger
from app.tick_summary import TickSummaryTracker
from database.repository import Repository
from database.supabase_client import make_supabase_client
from decision.rise_fall_decision_engine import RiseFallSymbolPipeline, summary_reason
from execution.orders import OrderExecutor, intent_from_rise_fall_decision
from ingestion.deriv_client import DerivClient
from pricing.contracts import FALL, RISE
from pricing.duration_grid import build_candidate_grid, filter_to_allowed
from pricing.monte_carlo_duration import DEFAULT_MC_SIMULATIONS
from research.calibration_warmstart import (
    extract_samples_from_pipeline, restore_or_generate_calibration, samples_to_jsonable,
)
from risk.risk_engine import RiskEngine

logger = get_logger("app.main")

TICK_SUMMARY_WINDOW = 150            # log a trade/no-trade-reason summary this often, per symbol
BALANCE_REFRESH_EVERY_N_TICKS = 200
EVALUATE_EVERY_N_TICKS = 1           # evaluate on every tick; raise this to throttle MC/API load per symbol


async def symbol_worker(symbol: str, client: DerivClient, pipeline: RiseFallSymbolPipeline,
                         repo: Repository, risk_engine: RiskEngine, order_executor: OrderExecutor, cfg) -> None:
    queue = await client.subscribe_ticks(symbol)
    currency = cfg.currency

    rf_cfg = cfg.get("rise_fall", default={})
    n_sims = rf_cfg.get("mc_simulations", DEFAULT_MC_SIMULATIONS)
    staking_cfg = rf_cfg.get("staking", {}) or {}
    tick_candidate_durations = cfg.get("contracts", "rise_fall", "candidate_tick_durations",
                                        default=[5, 10, 15, 20, 30])
    minute_candidate_durations = cfg.get("contracts", "rise_fall", "candidate_minute_durations",
                                          default=[1, 2, 3, 5, 10])

    log = get_logger("app.symbol_worker", symbol=symbol)

    # Cross-check the static config grid against Deriv's live per-symbol
    # limits before trading with it -- pricing/duration_grid.py's
    # filter_to_allowed() was built for exactly this but was never actually
    # called anywhere, which is why an out-of-range static candidate (e.g.
    # a tick duration above Deriv's CALL/PUT max of 10 ticks) was reaching
    # get_quote() on every single tick and getting rejected with "Proposal
    # request failed" / "Number of ticks must be between 1 and 10." forever,
    # since nothing ever removed it from the grid. Filtered against BOTH
    # RISE (CALL) and FALL (PUT) limits -- evaluate() shares one candidate
    # list across both directions, so a duration has to be valid for both
    # to be safe to hand it either one.
    try:
        contracts_for = await client.get_contracts_for(symbol)
    except Exception as exc:  # noqa: BLE001
        log.warning("contracts_for lookup failed; trading with unfiltered candidate durations",
                    extra={"extra_fields": {"error": str(exc)}})
        contracts_for = None

    if contracts_for:
        candidate_grid = build_candidate_grid(tick_candidate_durations, minute_candidate_durations)
        for contract_type in (RISE, FALL):
            candidate_grid = filter_to_allowed(candidate_grid, contracts_for, contract_type)
        filtered_ticks = sorted({c.duration for c in candidate_grid if c.duration_unit == "t"})
        filtered_minutes = sorted({c.duration for c in candidate_grid if c.duration_unit == "m"})
        dropped = (set(tick_candidate_durations) - set(filtered_ticks)) | \
                  (set(minute_candidate_durations) - set(filtered_minutes))
        if dropped:
            log.warning("Dropped candidate durations outside Deriv's live limits", extra={"extra_fields": {
                "dropped": sorted(dropped), "tick_candidates": filtered_ticks, "minute_candidates": filtered_minutes,
            }})
        tick_candidate_durations = filtered_ticks
        minute_candidate_durations = filtered_minutes
        if not tick_candidate_durations and not minute_candidate_durations:
            log.error("No candidate durations survived contracts_for filtering -- this symbol cannot trade "
                      "Rise/Fall until configs/config.yaml's rise_fall duration lists are corrected.")
    log.info("Worker started", extra={"extra_fields": {
        "base_stake": pipeline.base_stake, "min_edge": pipeline.min_edge,
        "min_confidence": pipeline.min_confidence,
        "staking_enabled": pipeline.staking.enabled,
        "tick_candidates": tick_candidate_durations, "minute_candidates": minute_candidate_durations,
        "seeded_prices": len(pipeline.price_series.prices),
    }})
    tick_summary = TickSummaryTracker(window_size=TICK_SUMMARY_WINDOW)

    tick_count = 0
    while True:
        tick = await queue.get()
        tick_count += 1

        pipeline.observe_tick(tick.epoch, tick.quote)
        if cfg.get("database", "persist_ticks", default=True):
            repo.insert_tick(symbol, tick.epoch, tick.quote, tick.digit)

        if tick_count % BALANCE_REFRESH_EVERY_N_TICKS == 0:
            try:
                balance = await client.get_balance()
                if balance and "balance" in balance:
                    risk_engine.set_equity(float(balance["balance"]))
            except Exception as exc:  # noqa: BLE001
                log.warning("Balance refresh failed", extra={"extra_fields": {"error": str(exc)}})

        if tick_count % EVALUATE_EVERY_N_TICKS != 0:
            continue

        decision = await pipeline.evaluate(
            client, currency, tick_candidate_durations, minute_candidate_durations, n_sims=n_sims,
        )

        # Gate against the ACTUAL stake this decision would trade at (after
        # martingale escalation/reset and any drift-degraded reduction),
        # never a pre-evaluate estimate -- decision.stake is only known
        # once evaluate() has run. Checking an earlier proxy (e.g. a flat
        # pipeline.base_stake) against risk.max_stake would silently let a
        # martingale-escalated stake through uninspected, which is exactly
        # the class of stake-mismatch bug decision/rise_fall_decision_
        # engine.py's own "STAKE SIZING WARNING" documents a real prior
        # incident about. No stake to check at all on a NO_TRADE decision.
        risk_ok, risk_reason = (True, None) if decision.decision == "NO_TRADE" \
            else risk_engine.check(decision.stake)

        if decision.decision != "NO_TRADE" and risk_ok:
            log.info("Executing trade", extra={"extra_fields": {
                "decision": decision.decision, "reason": decision.reason, "regime": decision.regime,
                "edge": decision.edge, "duration": f"{decision.duration}{decision.duration_unit}",
                "stake": decision.stake,
                "mc_win_probability": decision.mc_win_probability,
                "calibrated_probability": decision.calibrated_probability,
                "drift_degraded": decision.drift_degraded,
                "is_calibration_probe": decision.is_calibration_probe,
            }})
            intent = intent_from_rise_fall_decision(decision, currency)
            if intent is None:
                # Decided to trade, but couldn't even construct an intent
                # (missing quote/stake) -- no slot was ever reserved, and
                # evaluate() already stashed pending state for this
                # candidate that will now never settle. Cancel it rather
                # than leaving the pipeline permanently locked out (see
                # RiseFallSymbolPipeline.cancel_pending's docstring).
                pipeline.cancel_pending()
                tick_summary.record_trade(None, None)
            else:
                # Claim a concurrent-trade slot synchronously (no `await`
                # between the risk_ok check above and this reservation) --
                # see RiskEngine.reserve_trade_slot docstring for why the
                # ordering matters under asyncio. From here, OrderExecutor
                # owns releasing it: immediately if the buy itself never
                # goes through, or via the background settlement watcher
                # once the contract actually settles -- which, for a
                # Rise/Fall contract, can be minutes after this tick.
                risk_engine.reserve_trade_slot()

                def _on_settled(result, _log=log, _pipeline=pipeline, _summary=tick_summary,
                                 _repo=repo, _symbol=symbol, _persist_staking=staking_cfg.get(
                                     "persist_across_restarts", False)):
                    if result.won is not None:
                        _pipeline.record_outcome(result.won)
                        # Persist calibration state after every real,
                        # settled outcome -- not just at shutdown -- so a
                        # crash between saves loses at most the trades
                        # since the last settlement, not everything since
                        # the process started. See database/repository.py's
                        # save_rise_fall_calibration_state docstring and
                        # app/main.py's startup restore logic above.
                        _repo.save_rise_fall_calibration_state(
                            _symbol, samples_to_jsonable(extract_samples_from_pipeline(_pipeline)))
                        if _persist_staking:
                            step, consecutive_losses = _pipeline.staking.get_state(_symbol)
                            _repo.save_rise_fall_staking_state(_symbol, step, consecutive_losses)
                    else:
                        # settlement timed out / came back unknown -- no
                        # real outcome to record, but the pending slot
                        # still needs clearing or this symbol locks out
                        _pipeline.cancel_pending()
                    if result.error:
                        _log.warning("Trade did not settle cleanly", extra={"extra_fields": {
                            "error": result.error}})
                    _summary.record_trade(result.won, result.pnl)

                placed = await order_executor.place_trade(intent, prediction_id=None, on_settled=_on_settled)
                if placed.contract_id is None:
                    # No background watcher was scheduled (buy never went
                    # through, or dry_run) -- the outcome is already final,
                    # so report it now instead of waiting for a callback
                    # that will never fire. For a real failure (not
                    # dry_run), there's no real won/lost outcome to record,
                    # only that the pending decision needs clearing.
                    if cfg.dry_run:
                        _on_settled(placed)
                    else:
                        pipeline.cancel_pending()
                        if placed.error:
                            log.warning("Trade did not settle cleanly", extra={"extra_fields": {
                                "error": placed.error}})
                        tick_summary.record_trade(None, None)
        elif decision.decision != "NO_TRADE" and not risk_ok:
            log.info("Trade blocked by risk engine", extra={"extra_fields": {"reason": risk_reason}})
            repo.insert_risk_event(symbol, "trade_blocked", {"reason": risk_reason, "decision": decision.decision})
            pipeline.cancel_pending()  # decided to trade, but risk gate refused -- nothing to settle
            tick_summary.record_risk_blocked(risk_reason)
        else:
            tick_summary.record_no_trade(summary_reason(decision))

        if tick_summary.due():
            # Per-(contract_type, resolution) calibration diagnostics --
            # makes it visible from the logs alone whether a symbol's
            # calibrator has actually engaged yet (is_calibrated) and, once
            # it has, how reliable it is (quality_score's ECE-based 0..1
            # score -- see models/calibration.py). Without this,
            # "mc_win_probability" and "calibrated_probability" being
            # identical in an "Executing trade" log line is
            # indistinguishable from a real, validated calibration passing
            # raw probabilities straight through by coincidence -- this is
            # the one place that ambiguity gets resolved. Keyed as
            # "CALL_t"/"CALL_m"/"PUT_t"/"PUT_m" (string, not tuple -- JSON
            # object keys must be strings) since calibration is now split
            # per resolution, not just per contract_type -- see
            # decision/rise_fall_decision_engine.py's "PER-RESOLUTION
            # CALIBRATION".
            calibration_stats = {
                f"{contract_type}_{unit}": {
                    "n": pipeline.calibration[(contract_type, unit)].sample_size,
                    "is_calibrated": pipeline.calibration[(contract_type, unit)].is_calibrated,
                    "quality_score": round(pipeline.calibration[(contract_type, unit)].quality_score(), 3),
                    "rolling_log_loss": pipeline.calibration[(contract_type, unit)].rolling_log_loss(),
                }
                for contract_type in (RISE, FALL) for unit in ("t", "m")
            }
            summary = tick_summary.build_and_reset("rise_fall", len(pipeline.price_series.prices),
                                                    calibration_stats)
            log.info("Tick summary", extra={"extra_fields": {"event_type": "tick_summary", **summary.__dict__}})
            repo.insert_system_event("app.symbol_worker", "tick_summary", {"symbol": symbol, **summary.__dict__})


async def discover_symbols(client: DerivClient, cfg) -> list[str]:
    if cfg.symbol_override:
        logger.info("Using ASTRA_SYMBOLS override", extra={"extra_fields": {"symbols": cfg.symbol_override}})
        return cfg.symbol_override
    prefixes = cfg.get("symbols", "prefixes", default=["R_", "1HZ"])
    symbols = await client.get_active_synthetic_symbols(prefixes)
    logger.info("Discovered symbols", extra={"extra_fields": {"count": len(symbols), "symbols": symbols}})
    return symbols


async def seed_symbol(client: DerivClient, pipeline: RiseFallSymbolPipeline, count: int = 2000) -> list[tuple[int, float]]:
    """Returns the raw (epoch, price) ticks it seeded PriceSeries from, so a
    caller (see main()'s calibration auto-generate step) can reuse the same
    fetch instead of hitting client.get_history() a second time. Returns []
    on failure -- callers must treat that as "no history available", not
    raise."""
    try:
        history = await client.get_history(pipeline.symbol, count=count)
        for t in history:
            pipeline.observe_tick(t.epoch, t.quote)
        logger.info("Seeded symbol history", extra={"extra_fields": {
            "symbol": pipeline.symbol, "n": len(history),
            "tick_returns": len(pipeline.price_series.tick_log_returns),
            "minute_returns": len(pipeline.price_series.minute_log_returns),
        }})
        return [(t.epoch, t.quote) for t in history]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Seeding failed, will build history from live ticks only",
                        extra={"extra_fields": {"symbol": pipeline.symbol, "error": str(exc)}})
        return []


async def main() -> None:
    cfg = get_config()
    configure_logging(cfg.log_level)

    account_kind = "REAL MONEY" if cfg.deriv.use_real_account else "demo"
    logger.info("Starting Astra (Rise/Fall)", extra={"extra_fields": {
        "dry_run": cfg.dry_run, "account_kind": account_kind}})
    if cfg.deriv.use_real_account and cfg.dry_run:
        logger.warning("use_real_account=True with dry_run=True: connecting to the REAL account "
                        "but no orders will be placed.")
    elif cfg.deriv.use_real_account:
        logger.warning("use_real_account=True: this run trades REAL MONEY.")
    else:
        logger.info(f"Trading against the DEMO account (dry_run={cfg.dry_run}).")

    supabase = make_supabase_client(cfg.supabase.url, cfg.supabase.service_key, cfg.supabase.enabled)
    repo = Repository(supabase, persist_ticks=cfg.get("database", "persist_ticks", default=True))

    client = DerivClient(
        app_id=cfg.deriv.app_id, api_token=cfg.deriv.api_token,
        ws_url=cfg.deriv.ws_url, options_token_url=cfg.deriv.options_token_url,
        account_id=cfg.deriv.account_id, use_real_account=cfg.deriv.use_real_account,
    )
    await client.connect()
    repo.insert_system_event("app.main", "startup")

    symbols = await discover_symbols(client, cfg)
    if not symbols:
        logger.error("No symbols discovered -- nothing to trade. Check DERIV_APP_ID/token permissions.")
        return

    risk_cfg = cfg.get("risk", default={})
    risk_engine = RiskEngine(
        base_stake=risk_cfg.get("base_stake", 1.0), max_stake=risk_cfg.get("max_stake", 5.0),
        max_consecutive_losses=risk_cfg.get("max_consecutive_losses", 5),
        max_daily_loss=risk_cfg.get("max_daily_loss", 25.0), max_drawdown=risk_cfg.get("max_drawdown", 40.0),
        max_trades_per_day=risk_cfg.get("max_trades_per_day", 500),
        cooldown_seconds_after_max_losses=risk_cfg.get("cooldown_seconds_after_max_losses", 900),
        max_concurrent_trades=risk_cfg.get("max_concurrent_trades", 2),
    )

    # Shared across every symbol worker -- settlement watchers it schedules
    # run as background tasks independent of any one worker's tick loop
    # (see execution/orders.py), which is what lets a Rise/Fall contract
    # take minutes to settle without stalling tick ingestion/evaluation.
    order_executor = OrderExecutor(client, repo, risk_engine, dry_run=cfg.dry_run)

    rf_cfg = cfg.get("rise_fall", default={})
    staking_cfg = rf_cfg.get("staking", {}) or {}
    pipelines = {}
    for symbol in symbols:
        repo.upsert_symbol(symbol)
        pipelines[symbol] = RiseFallSymbolPipeline(
            symbol, base_stake=rf_cfg.get("base_stake", 1.0), min_edge=rf_cfg.get("min_edge", 0.03),
            min_calibration_samples=rf_cfg.get("min_calibration_samples", 200),
            # Confidence gate: only allow entries when BOTH mc_win_probability
            # and calibrated_probability clear this threshold (see
            # decision/rise_fall_decision_engine.py's "CONFIDENCE GATE").
            min_confidence=rf_cfg.get("min_confidence", 0.70),
            # Calibration-quality gate: once a candidate's calibrator has
            # fit, its quality_score() must also clear this (see
            # decision/rise_fall_decision_engine.py's "CALIBRATION-QUALITY
            # GATE"). 0.0 disables it -- see that docstring for why this is
            # off by default.
            min_calibration_quality=rf_cfg.get("min_calibration_quality", 0.0),
            # Probe-trade escape hatch for the lockout the quality gate
            # above would otherwise create -- see "PROBE TRADES" docstring.
            # 0 disables probing (permanent lockout once blocked).
            calibration_quality_probe_interval=rf_cfg.get("calibration_quality_probe_interval", 50),
            # Martingale staking -- opt-in, off by default (see
            # decision/rise_fall_decision_engine.py's "MARTINGALE STAKING"
            # and risk/staking.py's own documented warning).
            staking_enabled=staking_cfg.get("enabled", False),
            staking_progression_factor=staking_cfg.get("progression_factor", 2.0),
            staking_max_steps=staking_cfg.get("max_steps", 4),
            staking_max_stake=staking_cfg.get("max_stake"),
            staking_min_consecutive_losses=staking_cfg.get("min_consecutive_losses", 2),
        )

    # Fetch history ONCE per symbol -- feeds BOTH PriceSeries seeding
    # (always) and, if needed, calibration auto-generation below (only for
    # a symbol with no restorable state yet). One fetch, two uses, instead
    # of a second Deriv API round-trip. count is bumped when auto-generate
    # is enabled, since a meaningful minute-resolution replay needs
    # noticeably more raw ticks than PriceSeries alone would ask for.
    auto_generate = rf_cfg.get("calibration_warm_start_auto_generate", False)
    seed_count = rf_cfg.get("calibration_warm_start_auto_generate_count", 5000) if auto_generate else 2000
    history_by_symbol: dict[str, list[tuple[int, float]]] = {}

    async def _seed(symbol: str, pipeline: RiseFallSymbolPipeline) -> None:
        history_by_symbol[symbol] = await seed_symbol(client, pipeline, count=seed_count)

    await asyncio.gather(*[_seed(symbol, pipelines[symbol]) for symbol in symbols])

    # Restore calibration state, in priority order -- so every symbol's
    # worker begins its FIRST live evaluate() already at whatever warm
    # state is available, rather than warming up gradually during live
    # trading:
    #   1. Supabase (astra_rise_fall_calibration_state) -- the bot's OWN
    #      continuously-updated state from every previous settlement (see
    #      _on_settled below, which saves here after every trade). This is
    #      what actually survives a Railway restart/redeploy day to day,
    #      and loading it is near-instant (just replaying stored pairs
    #      through .record(), no MC simulation).
    #   2. The one-off file-based warm-start (calibration_warm_start_dir,
    #      research/calibration_warmstart.py / backtest/warm_start_
    #      calibration.py) -- for history that predates the bot ever
    #      running, or a manually-curated batch.
    #   3. Auto-generate (calibration_warm_start_auto_generate, opt-in, off
    #      by default) -- if NEITHER of the above had anything, replay the
    #      history batch already fetched above through the same MC call
    #      evaluate() itself uses, right now, before this symbol's worker
    #      starts. Saved to Supabase immediately afterward, so this is a
    #      ONE-TIME cost per symbol: every later restart hits step 1
    #      instead. This is genuinely CPU-bound (thousands of MC
    #      simulations) and runs in a thread (asyncio.to_thread), not
    #      inline on the event loop -- blocking the loop here for the
    #      seconds-to-minutes this can take would stall the recv-pump's
    #      ping/pong keepalive and risk tripping the exact "no close frame"
    #      disconnect ingestion/deriv_client.py's reconnect supervisor
    #      exists to recover from (see that module's docstring point 3) --
    #      recovering from a self-inflicted disconnect is a worse outcome
    #      than just taking the thread-hop here.
    #   4. Nothing available anywhere -- starts cold, same as before any of
    #      this existed.
    warm_start_dir = rf_cfg.get("calibration_warm_start_dir")
    tick_candidate_durations = cfg.get("contracts", "rise_fall", "candidate_tick_durations",
                                        default=[5, 10, 15, 20, 30])
    minute_candidate_durations = cfg.get("contracts", "rise_fall", "candidate_minute_durations",
                                          default=[1, 2, 3, 5, 10])

    async def _restore_or_generate_calibration(symbol: str, pipeline: RiseFallSymbolPipeline) -> None:
        t0 = time.monotonic()
        outcome, applied = await restore_or_generate_calibration(
            repo, pipeline, symbol,
            warm_start_dir=warm_start_dir,
            ticks=history_by_symbol.get(symbol) or [],
            tick_candidate_durations=tick_candidate_durations,
            minute_candidate_durations=minute_candidate_durations,
            auto_generate=auto_generate,
            n_sims=rf_cfg.get("calibration_warm_start_auto_generate_n_sims", 2000),
            sample_every=rf_cfg.get("calibration_warm_start_auto_generate_sample_every", 5),
            on_file_error=lambda path, exc: logger.warning(
                "Calibration warm-start load failed",
                extra={"extra_fields": {"symbol": symbol, "path": path, "error": str(exc)}}),
        )
        message = {
            "supabase": "Restored calibration state from Supabase",
            "file": "Loaded calibration warm-start from file",
            "auto_generated": "Auto-generated calibration warm-start",
            "cold": "No calibration state available for symbol -- starting cold",
        }[outcome]
        fields = {"symbol": symbol, "applied": applied}
        if outcome == "auto_generated":
            fields["seconds"] = round(time.monotonic() - t0, 1)
        logger.info(message, extra={"extra_fields": fields})

    await asyncio.gather(*[_restore_or_generate_calibration(symbol, pipelines[symbol]) for symbol in symbols])

    # Martingale progression state -- NOT restored by default even when
    # martingale itself is enabled. See risk/staking.py's StakingEngine
    # docstring and configs/config.yaml's rise_fall.staking.
    # persist_across_restarts for why this is a deliberate opt-in, not an
    # obvious win the way calibration restore above is.
    if staking_cfg.get("persist_across_restarts", False):
        for symbol, pipeline in pipelines.items():
            staking_state = repo.load_rise_fall_staking_state(symbol)
            if staking_state:
                pipeline.staking.load_state(
                    symbol, step=staking_state.get("step", 0),
                    consecutive_losses=staking_state.get("consecutive_losses", 0),
                )
                logger.info("Restored staking state from Supabase", extra={"extra_fields": {
                    "symbol": symbol, "step": staking_state.get("step", 0),
                    "consecutive_losses": staking_state.get("consecutive_losses", 0),
                }})

    workers = []
    for symbol in symbols:
        workers.append(asyncio.create_task(
            symbol_worker(symbol, client, pipelines[symbol], repo, risk_engine, order_executor, cfg),
            name=f"worker-{symbol}",
        ))

    stop_event = asyncio.Event()

    def _handle_signal():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass  # not available on some platforms (e.g. Windows)

    await stop_event.wait()

    for w in workers:
        w.cancel()
    # Cancelling worker tasks stops new trades from being placed, but any
    # settlement watchers already scheduled by order_executor.place_trade()
    # are independent asyncio tasks -- give them a bounded chance to finish
    # and persist their result before the connection they need goes away.
    await order_executor.shutdown()
    await client.close()
    repo.insert_system_event("app.main", "shutdown")


if __name__ == "__main__":
    asyncio.run(main())
