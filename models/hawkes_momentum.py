"""
Signed self-exciting jump-clustering momentum -- an independent Rise/Fall
signal layer for decision/regime_conviction.py's voting scheme.

HONESTY NOTE: this is a lightweight, incrementally-updated intensity score
INSPIRED BY Hawkes self-exciting point processes, not a fitted (MLE)
Hawkes model. A properly fitted Hawkes process estimates its own
excitation kernel from data; this instead uses a fixed exponential decay
and a fixed excitation size, both configurable rather than learned.
That's a real simplification, made deliberately in the same spirit as
this codebase's other documented simplifications (adaptive per-digit
weighting instead of literally-separate per-digit models, deterministic
rule-based "research agents" instead of literal AI agents) -- it captures
the qualitatively important behavior (large moves cluster, and a recent
large move raises the odds of another one in the SAME direction) at a
fraction of the implementation and maintenance cost of a real MLE fit,
and it's honest about not being one.

WHY THIS IS A DIFFERENT SIGNAL from the other two new layers:
  - ou_zscore.py assumes reversion toward a fitted equilibrium -- exactly
    backwards for a market in the middle of a real directional move.
  - tick_markov.py conditions on the last k tick SIGNS regardless of
    size -- a run of ten tiny up-ticks and one huge up-tick look
    identical to it, as long as both are "up".
  - This layer conditions on MAGNITUDE (was this move unusually large
    relative to recent volatility) and CLUSTERING (have several such
    moves happened recently, in the same direction) -- information the
    other two structurally discard.

INCREMENTAL, O(1) PER TICK: intensity decays by a fixed factor on every
push() and gets excited only on an actual qualifying jump -- no
rescanning of tick history, matching this codebase's established
performance discipline (see ou_zscore.py and tick_markov.py's own
docstrings for the two prior instances of the "recomputed from scratch
every tick" bug already found and fixed elsewhere in this pipeline).
"""
from __future__ import annotations

import math


class HawkesMomentumModel:
    """One instance per (symbol, resolution) -- never mix tick and minute
    observations into one instance, same reasoning as everywhere else in
    this pipeline."""

    def __init__(self, decay: float = 0.90, jump_multiple: float = 2.0,
                 excitation: float = 1.0, vol_ewm_alpha: float = 0.05,
                 vote_scale: float = 2.0, warmup: int = 30):
        # decay: fraction of intensity retained each push() when no jump
        #   fires -- 0.90 gives a jump's contribution a half-life of
        #   about 6-7 observations (ln(0.5)/ln(0.90)).
        # jump_multiple: a |log_return| at least this many times the
        #   current EWMA volatility estimate counts as a "jump".
        # excitation: how much one qualifying jump adds to the signed
        #   intensity score (signed by that jump's own direction).
        # vol_ewm_alpha: smoothing factor for the running volatility
        #   estimate jumps are measured against -- self-relative, same
        #   reasoning as regime/detector.py's own vol-ratio thresholds
        #   (so this works unchanged across symbols of wildly different
        #   native volatility scale).
        self.decay = decay
        self.jump_multiple = jump_multiple
        self.excitation = excitation
        self.vol_ewm_alpha = vol_ewm_alpha
        self.vote_scale = vote_scale
        self.warmup = warmup

        self._intensity = 0.0
        self._ewm_var = 0.0
        self._n = 0

    def push(self, log_return: float) -> None:
        self._n += 1
        r2 = log_return * log_return
        if self._n == 1:
            self._ewm_var = r2
        else:
            self._ewm_var = (1 - self.vol_ewm_alpha) * self._ewm_var + self.vol_ewm_alpha * r2

        self._intensity *= self.decay

        vol = math.sqrt(self._ewm_var)
        if self._n > self.warmup and vol > 1e-12 and abs(log_return) >= self.jump_multiple * vol:
            sign = 1.0 if log_return > 0 else -1.0
            self._intensity += sign * self.excitation

    def vote(self) -> float:
        """Signed float in [-1, 1]. Positive = recent jump clustering has
        leaned upside (continuation, CALL/RISE); negative = downside
        clustering (PUT/FALL); near 0 = no recent jump activity, it's
        decayed away, or still warming up."""
        if self._n <= self.warmup:
            return 0.0
        return math.tanh(self._intensity / self.vote_scale)

    @property
    def sample_size(self) -> int:
        return self._n
