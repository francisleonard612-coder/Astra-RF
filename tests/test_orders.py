import asyncio

import pricing.payout as payout_module
from execution.orders import OrderExecutor, TradeIntent, intent_from_rise_fall_decision
from pricing.contracts import RISE
from pricing.payout import ContractQuote
from risk.risk_engine import RiskEngine


def _reset_payout_module_state():
    # _quote_cache / rate-limit cooldown are module-level in pricing/payout.py
    # by design (the budget they guard is shared across the whole process,
    # not per-client) -- reset them between tests so one test's rate-limit
    # failure can't silently change another test's behavior.
    payout_module._quote_cache.clear()
    payout_module._rate_limit_cooldown_until = 0.0
    payout_module._consecutive_rate_limit_failures = 0


class FakeDerivClient:
    def __init__(self, payout=2.0, ask_price=1.0, contract_id=999, longcode="Volatility 100 Index rises",
                 profit=1.0, is_sold=True):
        self.payout = payout
        self.ask_price = ask_price
        self.contract_id = contract_id
        self.longcode = longcode
        self.profit = profit
        self.is_sold = is_sold
        self.buy_calls = []
        self.settlement_calls = []

    async def get_proposal(self, **kwargs):
        return {
            "payout": self.payout, "ask_price": self.ask_price, "id": "prop-1",
            "spot": 100.0, "longcode": self.longcode,
        }

    async def buy(self, proposal_id, price):
        self.buy_calls.append((proposal_id, price))
        if self.contract_id is None:
            return {}
        return {"contract_id": self.contract_id}

    async def wait_for_contract_settlement(self, contract_id, timeout=30.0):
        self.settlement_calls.append((contract_id, timeout))
        return {"is_sold": self.is_sold, "profit": self.profit if self.is_sold else None}


class FakeRepo:
    def __init__(self):
        self.inserted = []

    def insert_trade(self, result, prediction_id=None):
        self.inserted.append((result, prediction_id))


def _intent(**overrides) -> TradeIntent:
    quote = ContractQuote(symbol="R_100", contract_type=RISE, barrier=None, stake=1.0,
                           payout=2.0, ask_price=1.0, proposal_id="prop-0", spot=100.0,
                           longcode="Volatility 100 Index rises")
    defaults = dict(symbol="R_100", contract_type=RISE, stake=1.0, duration=1, duration_unit="m",
                     currency="USD", quote=quote, barrier=None)
    defaults.update(overrides)
    return TradeIntent(**defaults)


def _risk_engine(max_concurrent=2) -> RiskEngine:
    return RiskEngine(base_stake=1.0, max_stake=5.0, max_consecutive_losses=100, max_daily_loss=1000.0,
                       max_drawdown=1000.0, max_trades_per_day=10000, cooldown_seconds_after_max_losses=1,
                       max_concurrent_trades=max_concurrent)


def test_successful_trade_returns_immediately_then_settles_in_background():
    _reset_payout_module_state()

    async def run():
        client = FakeDerivClient(payout=2.0, contract_id=42, profit=1.5, is_sold=True)
        repo = FakeRepo()
        risk_engine = _risk_engine()
        executor = OrderExecutor(client, repo, risk_engine, dry_run=False)

        settled_results = []
        risk_engine.reserve_trade_slot()
        immediate = await executor.place_trade(_intent(), prediction_id=7,
                                                 on_settled=settled_results.append)

        # place_trade returns before settlement is known
        assert immediate.contract_id == 42
        assert immediate.won is None and immediate.pnl is None
        assert immediate.error is None
        assert risk_engine.state.open_trades == 1  # NOT yet released
        assert settled_results == []

        # force the background settlement watcher to completion
        assert len(executor._pending_tasks) == 1
        await list(executor._pending_tasks)[0]

        assert risk_engine.state.open_trades == 0  # released once settled
        assert len(settled_results) == 1
        assert settled_results[0].won is True
        assert settled_results[0].pnl == 1.5
        assert len(repo.inserted) == 1
        assert repo.inserted[0][1] == 7
        return client

    client = asyncio.run(run())
    assert client.buy_calls == [("prop-1", 1.0)]


def test_stale_quote_drift_fails_immediately_and_persists():
    _reset_payout_module_state()

    async def run():
        client = FakeDerivClient(payout=10.0)  # 2.0 -> 10.0 is far more than 15% drift
        repo = FakeRepo()
        risk_engine = _risk_engine()
        executor = OrderExecutor(client, repo, risk_engine, dry_run=False)

        risk_engine.reserve_trade_slot()
        result = await executor.place_trade(_intent(), prediction_id=1)

        assert result.contract_id is None
        assert result.error == "stale_quote_payout_drift"
        assert risk_engine.state.open_trades == 0  # released, not left dangling
        assert len(repo.inserted) == 1  # failed attempts are still logged
        assert client.buy_calls == []  # never got as far as buying
        assert executor._pending_tasks == set()

    asyncio.run(run())


def test_dry_run_never_buys_but_exercises_risk_bookkeeping():
    _reset_payout_module_state()

    async def run():
        client = FakeDerivClient(payout=2.0)
        repo = FakeRepo()
        risk_engine = _risk_engine()
        executor = OrderExecutor(client, repo, risk_engine, dry_run=True)

        risk_engine.reserve_trade_slot()
        result = await executor.place_trade(_intent(), prediction_id=3)

        assert result.contract_id is None
        assert result.pnl == 0.0
        assert result.error is None
        assert client.buy_calls == []
        assert risk_engine.state.open_trades == 0
        assert risk_engine.state.trades_today == 1  # dry-run trades still count
        assert len(repo.inserted) == 1

    asyncio.run(run())


def test_direction_mismatch_tripwire_blocks_the_trade():
    _reset_payout_module_state()

    async def run():
        # asked for RISE but Deriv's own longcode says the contract falls
        client = FakeDerivClient(payout=2.0, longcode="Volatility 100 Index falls")
        repo = FakeRepo()
        risk_engine = _risk_engine()
        executor = OrderExecutor(client, repo, risk_engine, dry_run=False)

        risk_engine.reserve_trade_slot()
        result = await executor.place_trade(_intent(contract_type=RISE), prediction_id=None)

        assert result.error == "direction_mismatch_tripwire"
        assert client.buy_calls == []
        assert risk_engine.state.open_trades == 0

    asyncio.run(run())


def test_shutdown_waits_for_pending_settlements_then_cancels_stragglers():
    _reset_payout_module_state()

    async def run():
        client = FakeDerivClient(payout=2.0, contract_id=1)
        repo = FakeRepo()
        risk_engine = _risk_engine()
        executor = OrderExecutor(client, repo, risk_engine, dry_run=False)

        async def never_settles(contract_id, timeout=30.0):
            await asyncio.sleep(9999)

        client.wait_for_contract_settlement = never_settles

        risk_engine.reserve_trade_slot()
        await executor.place_trade(_intent(), prediction_id=None)
        assert len(executor._pending_tasks) == 1

        stragglers = list(executor._pending_tasks)
        await executor.shutdown(timeout=0.2)
        await asyncio.sleep(0.05)  # let the loop actually process the cancellation
        assert all(t.done() for t in stragglers)

    asyncio.run(run())


class _FakeRiseFallDecision:
    def __init__(self, decision, stake, quote, contract_type=RISE, duration=5, duration_unit="t",
                 symbol="1HZ10V", conviction=None, conviction_direction=None, layer_votes=None):
        self.decision = decision
        self.stake = stake
        self.quote = quote
        self.contract_type = contract_type
        self.duration = duration
        self.duration_unit = duration_unit
        self.symbol = symbol
        # Conviction fields (decision/rise_fall_decision_engine.py's
        # "MULTI-LAYER VOTING") -- default None/None/None, matching a
        # decision made before conviction voting existed, or a probe
        # candidate (which never carries conviction). Tests that care about
        # these can override them explicitly.
        self.conviction = conviction
        self.conviction_direction = conviction_direction
        self.layer_votes = layer_votes


def test_intent_from_rise_fall_decision_builds_a_matching_intent():
    quote = ContractQuote(symbol="1HZ10V", contract_type=RISE, barrier=None, stake=2.0,
                           payout=4.0, ask_price=2.0, proposal_id="p1", spot=100.0)
    decision = _FakeRiseFallDecision("TRADE_RISE", stake=2.0, quote=quote,
                                      conviction=0.62, conviction_direction=1,
                                      layer_votes={"tick_markov": 0.4})

    intent = intent_from_rise_fall_decision(decision, currency="USD")

    assert intent is not None
    assert intent.symbol == "1HZ10V"
    assert intent.contract_type == RISE
    assert intent.stake == 2.0
    assert intent.duration == 5
    assert intent.duration_unit == "t"
    assert intent.currency == "USD"
    assert intent.barrier is None
    assert intent.quote is quote
    assert intent.conviction == 0.62
    assert intent.conviction_direction == 1
    assert intent.layer_votes == {"tick_markov": 0.4}


def test_intent_from_rise_fall_decision_none_on_no_trade():
    decision = _FakeRiseFallDecision("NO_TRADE", stake=None, quote=None)
    assert intent_from_rise_fall_decision(decision, currency="USD") is None


def test_intent_from_rise_fall_decision_none_without_a_quote():
    decision = _FakeRiseFallDecision("TRADE_RISE", stake=2.0, quote=None)
    assert intent_from_rise_fall_decision(decision, currency="USD") is None


def test_conviction_fields_survive_intent_through_to_the_settled_result():
    """End-to-end: conviction/conviction_direction/layer_votes set on a
    TradeIntent must ride along through the entire settlement flow (the
    immediate pending TradeResult AND the final settled one persisted by
    _watch_and_finalize) -- this is what decision/regime_conviction.py's
    conviction_outcome_report() needs present on completed astra_trades
    rows to check conviction against real win/loss outcomes."""
    _reset_payout_module_state()

    async def run():
        client = FakeDerivClient(payout=2.0, contract_id=42, profit=1.5, is_sold=True)
        repo = FakeRepo()
        risk_engine = _risk_engine()
        executor = OrderExecutor(client, repo, risk_engine, dry_run=False)
        intent = _intent(conviction=0.71, conviction_direction=1, layer_votes={"kalman_trend": 0.5})

        risk_engine.reserve_trade_slot()
        immediate = await executor.place_trade(intent, prediction_id=1)
        assert immediate.conviction == 0.71
        assert immediate.conviction_direction == 1
        assert immediate.layer_votes == {"kalman_trend": 0.5}

        await list(executor._pending_tasks)[0]  # force settlement

        settled_result, _ = repo.inserted[-1]
        assert settled_result.won is True
        assert settled_result.conviction == 0.71
        assert settled_result.conviction_direction == 1
        assert settled_result.layer_votes == {"kalman_trend": 0.5}

    asyncio.run(run())


def test_conviction_fields_default_to_none_for_digit_trades():
    """intent_from_digit_decision never sets these -- TradeIntent's
    defaults must make that a genuine no-op, not an error, for the digit
    decision engine's existing, unrelated call path."""
    intent = _intent()
    assert intent.conviction is None
    assert intent.conviction_direction is None
    assert intent.layer_votes is None


class _FakeInsertTable:
    def __init__(self, store: list):
        self._store = store

    def insert(self, row):
        self._row = row
        return self

    def execute(self):
        self._store.append(self._row)
        return self._row


class _FakeSupabaseClientForInsert:
    def __init__(self):
        self.rows: list = []

    def table(self, _name):
        return _FakeInsertTable(self.rows)


def test_repository_insert_trade_persists_conviction_fields():
    """database/repository.py's insert_trade must carry conviction/
    conviction_direction/layer_votes into the row it sends to Supabase --
    without this, TradeResult can have them, but astra_trades never would,
    and conviction_outcome_report() would have nothing to read."""
    from database.repository import Repository
    from execution.orders import TradeResult

    client = _FakeSupabaseClientForInsert()
    repo = Repository(client=client)
    result = TradeResult(symbol="R_100", contract_type=RISE, barrier=None, stake=1.0, payout=2.0,
                          contract_id=42, won=True, pnl=1.0, error=None,
                          conviction=0.55, conviction_direction=1, layer_votes={"ou_zscore": -0.2})

    repo.insert_trade(result, prediction_id=None)

    assert len(client.rows) == 1
    row = client.rows[0]
    assert row["conviction"] == 0.55
    assert row["conviction_direction"] == 1
    assert row["layer_votes"] == {"ou_zscore": -0.2}


def test_repository_insert_trade_conviction_fields_default_to_none():
    from database.repository import Repository
    from execution.orders import TradeResult

    client = _FakeSupabaseClientForInsert()
    repo = Repository(client=client)
    result = TradeResult(symbol="R_100", contract_type=RISE, barrier=None, stake=1.0, payout=2.0,
                          contract_id=42, won=False, pnl=-1.0, error=None)

    repo.insert_trade(result, prediction_id=None)

    row = client.rows[0]
    assert row["conviction"] is None
    assert row["conviction_direction"] is None
    assert row["layer_votes"] is None
