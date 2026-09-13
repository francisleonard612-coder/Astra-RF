"""
Local-linear-trend Kalman filter -- an independent Rise/Fall signal layer
for decision/regime_conviction.py's voting scheme.

Distinct mechanism from Hurst (a persistence EXPONENT recomputed from
scratch over a fixed lookback window) and from ou_zscore.py (a
mean-reverting equilibrium model): this maintains a running,
incrementally-updated ESTIMATE of the current local slope of log-price,
together with the filter's own uncertainty about that slope -- a
genuinely different state-space formulation, and one that naturally
adapts its effective lookback (a confident, well-tracked trend gets a
tighter effective window than a noisy one) rather than using one fixed
window for every symbol and condition.

STATE: [level, slope], both in log-price units per tick/minute.
  level_t = level_{t-1} + slope_{t-1}    (+ process noise q_level)
  slope_t = slope_{t-1}                  (+ process noise q_slope)
  observed z_t = level_t                 (+ observation noise r_obs)
This is the textbook "local linear trend" model (the state-space form of
double exponential smoothing) -- three free noise parameters with sane
defaults below, exposed for per-symbol tuning if warranted.

INCREMENTAL, O(1) PER OBSERVATION: a 2x2 Kalman filter's predict+update
step is a handful of scalar operations, not a window rescan -- cheap
enough to run every tick without the profiling concerns already
documented elsewhere in this codebase (see ou_zscore.py and
tick_markov.py's docstrings).

VOTE NORMALIZATION: uses the filter's OWN posterior uncertainty about the
slope (sqrt of the slope's variance in the state covariance) to scale the
vote, rather than a separately-computed value -- a slope of a given
magnitude is stronger evidence when the filter is confident about it than
when it isn't, and that's exactly the quantity the filter already tracks.
"""
from __future__ import annotations

import math


class KalmanTrendModel:
    """One instance per (symbol, resolution). Feed it prices via push()
    at whichever resolution (tick, or minute close) this instance
    tracks -- never mix the two, same reasoning as everywhere else in
    this pipeline."""

    def __init__(self, q_level: float = 1e-7, q_slope: float = 1e-9,
                 r_obs: float = 1e-6, vote_scale: float = 1.0, warmup: int = 20):
        self.q_level = q_level
        self.q_slope = q_slope
        self.r_obs = r_obs
        self.vote_scale = vote_scale
        self.warmup = warmup

        self._level: float | None = None
        self._slope: float = 0.0
        # 2x2 posterior covariance, row-major: [[p_ll, p_ls], [p_sl, p_ss]]
        self._p = [[1.0, 0.0], [0.0, 1.0]]
        self._n = 0

    def push(self, price: float) -> None:
        if price <= 0:
            return
        z = math.log(price)
        self._n += 1

        if self._level is None:
            self._level = z
            self._slope = 0.0
            return

        # --- predict --- F = [[1, 1], [0, 1]]; P_pred = F P F^T + Q
        level_pred = self._level + self._slope
        slope_pred = self._slope
        p_ll, p_ls = self._p[0][0], self._p[0][1]
        p_sl, p_ss = self._p[1][0], self._p[1][1]
        p_ll_pred = p_ll + p_ls + p_sl + p_ss + self.q_level
        p_ls_pred = p_ls + p_ss
        p_sl_pred = p_sl + p_ss
        p_ss_pred = p_ss + self.q_slope

        # --- update (observe z = level, H = [1, 0]) ---
        y = z - level_pred                 # innovation
        s = p_ll_pred + self.r_obs         # innovation covariance
        if s <= 1e-15:
            return
        k_level = p_ll_pred / s
        k_slope = p_sl_pred / s

        self._level = level_pred + k_level * y
        self._slope = slope_pred + k_slope * y

        self._p[0][0] = (1 - k_level) * p_ll_pred
        self._p[0][1] = (1 - k_level) * p_ls_pred
        self._p[1][0] = p_sl_pred - k_slope * p_ll_pred
        self._p[1][1] = p_ss_pred - k_slope * p_ls_pred

    def vote(self) -> float:
        """Signed float in [-1, 1]. Positive = current slope estimate
        leans upward (CALL/RISE); negative = downward (PUT/FALL);
        magnitude reflects both slope size AND the filter's confidence
        in it (slope divided by its own posterior std)."""
        if self._n <= self.warmup or self._level is None:
            return 0.0
        slope_std = math.sqrt(max(self._p[1][1], 1e-18))
        if slope_std <= 1e-12:
            return 0.0
        return math.tanh(self._slope / (slope_std * self.vote_scale))

    @property
    def sample_size(self) -> int:
        return self._n
