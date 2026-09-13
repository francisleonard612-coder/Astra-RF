"""
Order-N Markov chain on tick-direction sign -- an independent Rise/Fall
signal layer for decision/regime_conviction.py's voting scheme.

Distinct mechanism from ou_zscore.py (a fitted continuous-state process)
and pricing/monte_carlo_duration.py (resampling the empirical return
distribution): this instead estimates P(next tick up | last k tick
directions) directly from observed transition frequencies. Same spirit as
the digit bot's models/markov.py, adapted from a 10-symbol digit alphabet
to a 3-symbol direction alphabet (up / down / flat), with the same order
backoff for the same reason: a higher-order context that hasn't been seen
often enough is worse evidence than a lower-order one that has.

INCREMENTAL COUNTS: push() updates a running transition-count table one
tick at a time. This is deliberately NOT implemented as "rescan
tick_log_returns and rebuild counts on every vote() call" -- that's
exactly the mistake state/rolling_state.py::SymbolState.push() was fixed
to avoid for the digit bot's own Markov model (see that module's
docstring for the profiling numbers). Built incrementally the first time
here instead of found the same way twice.

DATA AVAILABILITY: like ou_zscore.py, this learns from raw ticks, not
settled trade outcomes -- tick_summary logs showed sample_size climbing
into the thousands within minutes of a live run, versus calibration
trackers still sitting under 30 samples after days of real trading. Wire
push() to fire on every new PriceSeries.tick_log_returns append (a
SEPARATE instance for minute-resolution voting, fed from
minute_log_returns, if you want a minute-resolution vote too -- same
never-pool-tick-and-minute reasoning as everywhere else in this pipeline).
"""
from __future__ import annotations

from collections import defaultdict, deque


def _sign(log_return: float, flat_eps: float = 1e-9) -> int:
    if log_return > flat_eps:
        return 1
    if log_return < -flat_eps:
        return -1
    return 0


class TickMarkovModel:
    """One instance per (symbol, resolution). Feed it log-returns as they
    arrive via push() -- do not reconstruct it from a deque each cycle."""

    # Matches models/markov.py's MIN_CONTEXT_COUNT and its stated rationale:
    # a transition context needs at least this many observations before
    # it's trusted over backing off to the next-lower order.
    MIN_CONTEXT_COUNT = 20

    def __init__(self, max_order: int = 2, max_history: int = 2500):
        self.max_order = max_order
        self._history: deque = deque(maxlen=max_history)
        # _counts[order][context_tuple][next_sign] -> count
        self._counts: list[dict] = [defaultdict(lambda: defaultdict(int))
                                     for _ in range(max_order + 1)]

    def push(self, log_return: float) -> None:
        sign = _sign(log_return)
        for order in range(1, self.max_order + 1):
            if len(self._history) >= order:
                context = tuple(self._history)[-order:]
                self._counts[order][context][sign] += 1
        self._history.append(sign)

    def vote(self) -> float:
        """2*P(up | context) - 1, using the highest order whose current
        context has >= MIN_CONTEXT_COUNT observations, backing off toward
        order 1. Returns 0.0 (abstain) if no order has enough support."""
        for order in range(self.max_order, 0, -1):
            if len(self._history) < order:
                continue
            context = tuple(self._history)[-order:]
            counts = self._counts[order].get(context)
            if not counts:
                continue
            total = sum(counts.values())
            if total < self.MIN_CONTEXT_COUNT:
                continue
            p_up = counts.get(1, 0)
            p_down = counts.get(-1, 0)
            directional_total = p_up + p_down
            if directional_total <= 0:
                continue  # this context has only ever produced "flat" ticks
            return 2.0 * (p_up / directional_total) - 1.0
        return 0.0

    @property
    def sample_size(self) -> int:
        return len(self._history)
