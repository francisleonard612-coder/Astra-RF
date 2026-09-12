"""
Tests for:
1. database/repository.py's new save/load methods for Rise/Fall calibration
   and staking restart-recovery state.
2. risk/staking.py's StakingEngine.get_state()/load_state() accessors those
   repository methods are meant to round-trip.

No real Supabase connection: a minimal fake client stub mimics the
`.table(name).upsert(row).execute()` / `.select("*").eq(...).execute()`
chain Repository actually calls, matching the pattern Repository's own
`_safe()` wrapper is built around (catch, log, degrade -- never raise).
"""
from database.repository import Repository
from risk.staking import StakingEngine


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, store: dict, name: str):
        self._store = store
        self._name = name
        self._filter_symbol = None

    def upsert(self, row):
        self._pending_row = row
        return self

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, _field, value):
        self._filter_symbol = value
        return self

    def execute(self):
        if hasattr(self, "_pending_row"):
            self._store[self._pending_row["symbol"]] = self._pending_row
            return _FakeResult([self._pending_row])
        row = self._store.get(self._filter_symbol)
        return _FakeResult([row] if row else [])


class _FakeSupabaseClient:
    def __init__(self):
        self._tables: dict[str, dict] = {}

    def table(self, name):
        return _FakeTable(self._tables.setdefault(name, {}), name)


def _repo() -> Repository:
    return Repository(client=_FakeSupabaseClient())


# ---------------------------------------------------------------------------
# Calibration state
# ---------------------------------------------------------------------------

def test_calibration_state_round_trips_through_save_and_load():
    repo = _repo()
    calibration = {"CALL_t": {"raw": [0.8, 0.6], "outcome": [1, 0]}, "PUT_m": {"raw": [0.3], "outcome": [0]}}

    repo.save_rise_fall_calibration_state("R_100", calibration)
    loaded = repo.load_rise_fall_calibration_state("R_100")

    assert loaded == calibration


def test_calibration_state_load_returns_none_for_unknown_symbol():
    repo = _repo()
    assert repo.load_rise_fall_calibration_state("R_999") is None


def test_calibration_state_save_upserts_not_appends():
    """A second save for the same symbol must overwrite, not create a
    second row -- restart-recovery state is a snapshot, same pattern as
    astra_symbol_state / astra_digit_specialist_state."""
    repo = _repo()
    repo.save_rise_fall_calibration_state("R_100", {"CALL_t": {"raw": [0.1], "outcome": [0]}})
    repo.save_rise_fall_calibration_state("R_100", {"CALL_t": {"raw": [0.1, 0.9], "outcome": [0, 1]}})

    loaded = repo.load_rise_fall_calibration_state("R_100")

    assert loaded == {"CALL_t": {"raw": [0.1, 0.9], "outcome": [0, 1]}}


def test_calibration_state_degrades_gracefully_when_supabase_disabled():
    """Repository with no client (Supabase disabled) must not raise --
    same fail-soft contract as every other Repository method."""
    repo = Repository(client=None)
    repo.save_rise_fall_calibration_state("R_100", {"CALL_t": {"raw": [], "outcome": []}})
    assert repo.load_rise_fall_calibration_state("R_100") is None


# ---------------------------------------------------------------------------
# Staking state
# ---------------------------------------------------------------------------

def test_staking_state_round_trips_through_save_and_load():
    repo = _repo()
    repo.save_rise_fall_staking_state("R_100", step=2, consecutive_losses=3)

    loaded = repo.load_rise_fall_staking_state("R_100")

    assert loaded["step"] == 2
    assert loaded["consecutive_losses"] == 3


def test_staking_state_load_returns_none_for_unknown_symbol():
    repo = _repo()
    assert repo.load_rise_fall_staking_state("R_999") is None


# ---------------------------------------------------------------------------
# StakingEngine.get_state / load_state (what the repository methods above
# actually round-trip in app/main.py)
# ---------------------------------------------------------------------------

def test_get_state_reports_zero_for_a_symbol_never_touched():
    staking = StakingEngine(base_stake=1.0, enabled=True, progression_factor=2.0, max_steps=4, max_stake=100.0)
    assert staking.get_state("R_100") == (0, 0)


def test_get_state_does_not_allocate_state_as_a_side_effect():
    """Calling get_state() for an untouched symbol must not create an
    entry in the engine's internal state dict -- it should report (0, 0)
    without side effects, unlike current_stake()/record_result() which
    legitimately need to allocate on first use."""
    staking = StakingEngine(base_stake=1.0, enabled=True, progression_factor=2.0, max_steps=4, max_stake=100.0)
    staking.get_state("R_100")
    assert "R_100" not in staking._state


def test_load_state_restores_step_and_consecutive_losses():
    staking = StakingEngine(base_stake=1.0, enabled=True, progression_factor=2.0, max_steps=4, max_stake=100.0)

    staking.load_state("R_100", step=2, consecutive_losses=3)

    assert staking.get_state("R_100") == (2, 3)
    assert staking.current_stake("R_100") == 4.0  # 1.0 * 2.0^2 -- the restored step is actually used


def test_get_state_and_load_state_round_trip_through_record_result():
    """End-to-end: escalate via real record_result() calls, save the
    state, construct a FRESH engine (simulating a restart), restore via
    load_state(), and confirm it resumes exactly where it left off."""
    original = StakingEngine(base_stake=1.0, enabled=True, progression_factor=2.0, max_steps=4, max_stake=100.0,
                              min_consecutive_losses_before_escalation=1)
    original.record_result("R_100", won=False)
    original.record_result("R_100", won=False)
    step, consecutive_losses = original.get_state("R_100")
    assert (step, consecutive_losses) == (2, 2)

    restarted = StakingEngine(base_stake=1.0, enabled=True, progression_factor=2.0, max_steps=4, max_stake=100.0,
                               min_consecutive_losses_before_escalation=1)
    restarted.load_state("R_100", step=step, consecutive_losses=consecutive_losses)

    assert restarted.current_stake("R_100") == original.current_stake("R_100")
