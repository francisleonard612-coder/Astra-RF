"""
Fetch historical Deriv ticks for one or more symbols and replay them through
research/calibration_warmstart.py to produce per-(contract_type, resolution)
(raw_prob, outcome) samples, then write one JSON file per symbol.

app/main.py loads these automatically at startup for any symbol whose file
exists under rise_fall.calibration_warm_start_dir (see configs/config.yaml
and decision/rise_fall_decision_engine.py's "CALIBRATION WARM-START"
docstring for why this exists).

Usage:
    python -m backtest.warm_start_calibration --symbols R_100,R_75

Writes data/calibration_warmstart/R_100.json, data/calibration_warmstart/R_75.json
by default -- override with --out.

NOTE on scope, same caution as backtest/simulator.py's own docstring for a
different pipeline: this produces CALIBRATION samples only, not a trading
backtest. No edge, stake sizing, or gating logic is applied here -- see
research/calibration_warmstart.py's own docstring for exactly what "uses
future data" means in this replay and why that's fine for THIS purpose but
would not be for a P&L backtest.
"""
from __future__ import annotations

import argparse
import asyncio
import os

from app.config import get_config
from ingestion.deriv_client import DerivClient
from research.calibration_warmstart import build_calibration_samples, save_samples_json


async def _run(symbols: list[str], count: int, n_sims: int, sample_every: int, out_dir: str, cfg) -> None:
    client = DerivClient(
        app_id=cfg.deriv.app_id, api_token=cfg.deriv.api_token,
        ws_url=cfg.deriv.ws_url, options_token_url=cfg.deriv.options_token_url,
        account_id=cfg.deriv.account_id, use_real_account=cfg.deriv.use_real_account,
    )
    await client.connect()
    tick_durations = cfg.get("contracts", "rise_fall", "candidate_tick_durations",
                              default=[5, 10, 15, 20, 30])
    minute_durations = cfg.get("contracts", "rise_fall", "candidate_minute_durations",
                                default=[1, 2, 3, 5, 10])
    try:
        for symbol in symbols:
            print(f"[{symbol}] fetching up to {count} historical ticks...")
            history = await client.get_history(symbol, count=count)
            ticks = [(t.epoch, t.quote) for t in history]
            print(f"[{symbol}] got {len(ticks)} ticks. Replaying through monte_carlo_duration "
                  f"(n_sims={n_sims}, sample_every={sample_every}) -- this is the slow part...")
            samples = build_calibration_samples(
                ticks, tick_durations, minute_durations, n_sims=n_sims, sample_every=sample_every,
            )
            counts = {f"{k[0]}_{k[1]}": len(v) for k, v in samples.items()}
            path = os.path.join(out_dir, f"{symbol}.json")
            save_samples_json(samples, path)
            print(f"[{symbol}] wrote {path}: {counts}")
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. R_100,R_75")
    parser.add_argument("--count", type=int, default=5000,
                         help="Historical ticks to fetch per symbol (Deriv's own history cap applies)")
    parser.add_argument("--n-sims", type=int, default=2000,
                         help="MC simulations per candidate -- deliberately lower than the live "
                              "default (see pricing/monte_carlo_duration.py's DEFAULT_MC_SIMULATIONS) "
                              "since this runs many more candidates than one live evaluate() call")
    parser.add_argument("--sample-every", type=int, default=5,
                         help="Only replay every Nth valid tick position -- cost control, see "
                              "build_calibration_samples()'s own docstring")
    parser.add_argument("--out", default="data/calibration_warmstart",
                         help="Output directory for per-symbol JSON files")
    args = parser.parse_args()

    cfg = get_config()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        parser.error("--symbols must name at least one symbol")

    asyncio.run(_run(symbols, args.count, args.n_sims, args.sample_every, args.out, cfg))


if __name__ == "__main__":
    main()
