"""
Ornstein-Uhlenbeck mean-reversion z-score -- an independent Rise/Fall
signal layer for decision/regime_conviction.py's voting scheme.

Genuinely distinct mechanism from Hurst (regime/hurst_volatility.py, a
rescaled-range persistence exponent) and Monte Carlo resampling
(pricing/monte_carlo_duration.py, which resamples the empirical return
distribution directly). This instead FITS a parametric mean-reverting
process to log-price and measures how far away the CURRENT log-price sits
from that process's own equilibrium, in units of the process's own
equilibrium standard deviation -- a model-based deviation score, not a
resampling- or exponent-based one.

SCALE: fit on log-price, not raw price, for the same reason
regime/detector.py's thresholds are self-relative rather than absolute --
R_75 and 1HZ100V differ in native price scale by orders of magnitude (see
that module's own comment on this), and log-price differences are
scale-free the same way log-returns already are everywhere else in this
codebase.

FIT: discrete-time OU (Euler-Maruyama, dt = 1 observation) is exactly an
AR(1) regression:
    X[t] = a + b * X[t-1] + eps[t],   b = 1 - theta,   a = theta * mu
so theta and mu come out of one OLS fit -- no iterative solver, no scipy
dependency. The stationary variance of a discrete OU/AR(1) process is
resid_var / (1 - b^2); using that (rather than a separately-tracked
rolling std) means the "how wide is normal" estimate and the mean-
reversion strength estimate come from the same fit, so they can't quietly
disagree with each other.

REFIT DISCIPLINE: matches CalibrationTracker's own pattern
(models/calibration.py's min_samples/refit_every). Refitting on every
single tick from an OLS over the whole window would repeat the exact
"recomputed something O(window) every tick" mistake this codebase has
already found and fixed twice (state/rolling_state.py's Markov counts,
models/calibration.py's isotonic scoring -- see both modules' docstrings).
Fit refreshes every `refit_every` new observations; z-score lookups
between refits are O(1) off the cached theta/mu/sigma_eq.

ABSTENTION: vote() returns 0.0 (abstain -- excluded from
compute_conviction's agreement denominator entirely, not "no lean") until
min_samples observations have been seen, or if no OU fit has ever
succeeded (e.g. every window seen so far looked non-mean-reverting, b>=1).
This is a real advantage over waiting for calibration data: PriceSeries
already accumulates thousands of ticks per session (tick_summary logged
sample_size=2500 within the first ~15 minutes of a live run), so this
layer clears min_samples almost immediately -- unlike CalibrationTracker,
which needs actual SETTLED TRADE outcomes and was still sitting at n<30
after days of the account's history. Wire push() to fire on every
PriceSeries tick (or minute close, for a minute-resolution instance --
create two separate OUMeanReversionModel instances, same
PER-RESOLUTION-CALIBRATION reasoning as rise_fall_decision_engine.py's
four CalibrationTrackers), not on trade settlement.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class OUMeanReversionModel:
    """One instance per (symbol, resolution) -- same granularity as
    CalibrationTracker. Fit tick and minute log-price SEPARATELY, never
    pooled, for the same reason PER-RESOLUTION CALIBRATION in
    rise_fall_decision_engine.py never pools tick and minute candidates:
    different sampling processes, different noise characteristics.
    """
    window: int = 500          # log-price observations retained for refitting
    min_samples: int = 60      # below this, abstain entirely (vote=0.0)
    refit_every: int = 25      # re-run the OLS fit every N new observations
    vote_scale: float = 1.5    # z of this many equilibrium-sigmas -> vote ~= tanh(1.0) = 0.76

    _log_prices: deque = field(default_factory=lambda: deque(maxlen=500))
    _since_refit: int = 0
    _theta: float | None = field(default=None, repr=False)
    _mu: float | None = field(default=None, repr=False)
    _sigma_eq: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._log_prices = deque(self._log_prices, maxlen=self.window)

    def push(self, price: float) -> None:
        """Feed one new price observation -- call this alongside
        PriceSeries.push() (tick resolution) or on each minute-bar close
        (minute resolution), matching whichever this instance tracks."""
        if price <= 0:
            return
        self._log_prices.append(math.log(price))
        self._since_refit += 1
        if len(self._log_prices) >= self.min_samples and self._since_refit >= self.refit_every:
            self._refit()
            self._since_refit = 0

    def _refit(self) -> None:
        x = list(self._log_prices)
        x_prev, x_next = x[:-1], x[1:]
        n = len(x_prev)
        if n < 2:
            return
        mean_prev = sum(x_prev) / n
        mean_next = sum(x_next) / n
        cov = sum((p - mean_prev) * (nx - mean_next) for p, nx in zip(x_prev, x_next)) / n
        var_prev = sum((p - mean_prev) ** 2 for p in x_prev) / n
        if var_prev <= 1e-12:
            return  # degenerate (flat) window -- keep the previous fit rather than divide by ~0
        b = cov / var_prev
        a = mean_next - b * mean_prev
        # b >= 1: this window looks non-mean-reverting (or explosive) --
        # there's no equilibrium to score a deviation against. Decline to
        # update rather than report a nonsensical z-score; keep whatever
        # the last valid fit was (or stay unfitted, i.e. abstaining).
        if b >= 0.9999:
            return
        theta = 1.0 - b
        mu = a / theta
        residuals = [nx - (a + b * p) for p, nx in zip(x_prev, x_next)]
        resid_var = sum(r ** 2 for r in residuals) / max(1, n - 2)
        denom = 1.0 - b ** 2
        if denom <= 1e-9:
            return
        sigma_eq = math.sqrt(resid_var / denom)
        if sigma_eq <= 1e-12:
            return
        self._theta, self._mu, self._sigma_eq = theta, mu, sigma_eq

    def vote(self) -> float:
        """Signed float in [-1, 1]. Positive = currently below
        equilibrium, leans toward rising back up (CALL/RISE). Negative =
        currently above equilibrium, leans toward falling back down
        (PUT/FALL). 0.0 = abstain."""
        if self._theta is None or not self._log_prices:
            return 0.0
        current = self._log_prices[-1]
        z = (current - self._mu) / self._sigma_eq
        return -math.tanh(z / self.vote_scale)

    @property
    def is_fitted(self) -> bool:
        return self._theta is not None

    @property
    def sample_size(self) -> int:
        return len(self._log_prices)
