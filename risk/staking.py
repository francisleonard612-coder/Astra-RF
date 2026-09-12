"""
Stake progression, kept fully separate from the risk engine's hard limits so
a progression bug can never itself become a risk-limit bypass.

IMPORTANT CONTEXT FROM PRIOR BOTS: martingale progression was added to a
sibling bot (digit_over_bot) and, over a 278-trade live sample, the top
martingale stake tier lost the most money with no better win rate than the
flat-stake tier -- it amplified losses rather than compensating for them,
because the underlying per-trade edge didn't hold up. Astra defaults
staking.enabled=false (flat stake) for exactly that reason. Progression is
still implemented and available behind a config flag for anyone who wants to
re-test it, but it is not the default and should not be enabled without
looking at Astra's own live edge data first.

STAKE PRECISION BUG (found from a live deployment log, not theoretical):
current_stake() used to return base_stake * progression_factor**step
completely unrounded. Deriv's proposal endpoint rejects any stake with more
than 2 decimal places ("Stake can not have more than 2 decimal places."),
and base_stake * progression_factor**step lands on more than 2 decimals for
almost any progression_factor that isn't a power of 2 over a 2-decimal
base_stake -- e.g. base_stake=0.35, progression_factor=1.18 (the exact value
copied over from this account's own LEGACY digit-staking config) gives
0.35*1.18 = 0.413 at step=1. Once that happens, the escalated stake is
permanently rejected: get_quote() returns None for every future proposal at
that stake, evaluate() never has a decision to place, so record_outcome()
never fires to move the progression off that stake either -- the symbol
goes completely silent for the rest of the run, with nothing louder than a
per-tick "Proposal request failed" warning to show it (confirmed directly
against a production log: one symbol placed 4 trades in the first few
minutes, then silently placed zero more for the remaining ~18 minutes of
the run once its escalated stake hit this). Rounding here, once, at the
single place every caller reads the stake from, closes this off regardless
of what progression_factor is configured.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StakingState:
    consecutive_losses: int = 0
    step: int = 0


class StakingEngine:
    def __init__(self, base_stake: float, enabled: bool, progression_factor: float,
                 max_steps: int, max_stake: float, min_consecutive_losses_before_escalation: int = 1):
        self.base_stake = base_stake
        self.enabled = enabled
        self.progression_factor = progression_factor
        self.max_steps = max_steps
        self.max_stake = max_stake
        # How many consecutive losses must accumulate before the FIRST
        # escalation happens. Default 1 preserves the original behavior
        # (escalate immediately after a single loss). A caller can raise
        # this -- e.g. Astra's Rise/Fall pipeline uses 2, so one isolated
        # loss alone doesn't move the stake at all, and the progression only
        # engages once a second loss follows it consecutively. This does
        # NOT change what happens once escalation has started: every loss
        # after the threshold still steps up by progression_factor, same as
        # before.
        self.min_consecutive_losses_before_escalation = max(1, min_consecutive_losses_before_escalation)
        self._state: dict[str, StakingState] = {}

    def _get(self, symbol: str) -> StakingState:
        return self._state.setdefault(symbol, StakingState())

    def current_stake(self, symbol: str) -> float:
        if not self.enabled:
            return round(self.base_stake, 2)
        state = self._get(symbol)
        stake = self.base_stake * (self.progression_factor ** state.step)
        # See this module's "STAKE PRECISION BUG" docstring -- Deriv rejects
        # any stake with more than 2 decimal places, and the progression
        # above lands on more than 2 for almost any non-power-of-2 factor.
        # Round HERE, the one place every caller (evaluate()'s stake
        # computation, any future caller) reads the value from, rather than
        # leaving it to each call site to remember.
        return round(min(stake, self.max_stake), 2)

    def record_result(self, symbol: str, won: bool) -> None:
        state = self._get(symbol)
        if won:
            state.step = 0
            state.consecutive_losses = 0
            return
        state.consecutive_losses += 1
        if not self.enabled:
            return
        if state.consecutive_losses < self.min_consecutive_losses_before_escalation:
            # Not enough consecutive losses yet to engage the progression --
            # stake stays exactly where it is (base_stake, if this is the
            # very first loss in a fresh streak).
            return
        if state.step < self.max_steps:
            state.step += 1
        else:
            state.step = 0  # reset after max steps -- never climb forever
