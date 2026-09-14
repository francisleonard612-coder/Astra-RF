"""
Regression tests for ingestion/deriv_client.py's recv-pump supervisor.

Root cause #1 being guarded against: a live deployment log showed
`_recv_pump` crash on `ConnectionClosedError: no close frame received or
sent`, log "Recv pump crashed", and simply return -- with nothing else in
the client automatically reconnecting the TICK path (ensure_connected()
only runs from the request/response path, which a pure tick-consuming
symbol_worker() loop never calls). The process kept running, but every
symbol's tick feed silently died forever.

Root cause #2 (docstring point 4 in deriv_client.py): a prior fix for #1
resubscribed every symbol BEFORE restarting the pump on the new socket.
Every resubscribe goes through _send(), which awaits a response future
that only a running _recv_pump() ever resolves -- with no pump running
yet, every single resubscribe after every reconnect was guaranteed to
hang until request_timeout. Confirmed against a real production log: a
clean OTP re-auth immediately followed by a burst of "Failed to
resubscribe ticks after reconnect" timeouts, one per symbol, spaced
~request_timeout apart.

These tests exercise both fixes (_run_recv_pump_forever supervising
self._pump_task -- the live _recv_pump() task, tracked as a Task so it can
be started fresh, concurrently with resubscribing, on every reconnect --
and the shared _reconnect_and_resubscribe() path) without a real network
connection, by monkeypatching the socket-opening and pump internals.
"""
import asyncio

from websockets.protocol import State as WsState

from ingestion.deriv_client import DerivClient


class _FakeWs:
    """state is checked by ensure_connected()/_reconnect_and_resubscribe().
    Also a trivial (immediately-exhausted) async iterator, so that a REAL
    _recv_pump() task started against one of these (as
    _reconnect_and_resubscribe() now does on every reconnect) exits
    cleanly instead of crashing on `async for raw in self._ws` -- tests
    that don't care about pump behavior can leave it unmocked."""
    def __init__(self, state=WsState.OPEN):
        self.state = state

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def _client() -> DerivClient:
    return DerivClient(app_id="1", api_token="tok", ws_url="wss://x", options_token_url="https://x")


async def _instant_sleep(*_args, **_kwargs) -> None:
    """Stand-in for asyncio.sleep in tests -- returns immediately instead
    of actually backing off, without recursing into itself (a lambda
    calling `asyncio.sleep(0)` after `asyncio.sleep` has been monkeypatched
    to that same lambda recurses forever)."""
    return None


async def _drain_pump_task(client: DerivClient) -> None:
    """Test teardown helper: await/cancel whatever self._pump_task ended up
    as, so a still-pending fake or real pump task doesn't outlive the test
    (asyncio warns loudly about tasks destroyed while pending)."""
    task = client._pump_task
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


# ---------------------------------------------------------------------------
# _run_recv_pump_forever
# ---------------------------------------------------------------------------

def test_recv_pump_forever_reconnects_after_the_pump_exits(monkeypatch):
    """The exact production scenario: the live pump task exits (crash or
    clean close, doesn't matter which -- _recv_pump already logs the
    difference internally and returns either way), and the supervisor must
    reconnect and resume, not just let the tick feed die silently forever.

    Mirrors what the real connect()/_reconnect_and_resubscribe() do: each
    "pump run" is its own Task assigned to self._pump_task, created fresh
    on every (re)connect -- see docstring point 4 for why the pump has to
    be a task the supervisor awaits, not a coroutine it calls directly."""
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)  # skip the real backoff delay

    async def run():
        client = _client()
        pump_calls = 0

        async def fake_pump_body():
            nonlocal pump_calls
            pump_calls += 1
            if pump_calls >= 2:
                client._closed = True  # stop the supervisor loop after this second exit
            return  # simulates _recv_pump exiting, exactly as it does after logging its own crash

        client._pump_task = asyncio.create_task(fake_pump_body())  # the initial connect()-started pump

        reconnect_calls = 0

        async def fake_reconnect():
            nonlocal reconnect_calls
            reconnect_calls += 1
            client._ws = _FakeWs()  # a real reconnect would leave a live socket behind
            # ...and start a fresh pump task on it, exactly as the real
            # _reconnect_and_resubscribe() does right after _open_socket().
            client._pump_task = asyncio.create_task(fake_pump_body())

        client._reconnect_and_resubscribe = fake_reconnect

        await client._run_recv_pump_forever()

        assert pump_calls == 2  # ran again (as a fresh task) after the first exit, then stopped once closed
        assert reconnect_calls == 1  # reconnected exactly once, between the two pump runs

    asyncio.run(run())


def test_recv_pump_forever_keeps_retrying_if_reconnect_itself_fails(monkeypatch):
    """A failed reconnect attempt (e.g. a transient OTP-exchange outage)
    must not kill the supervisor -- it should back off and retry the
    RECONNECT itself (not just fall through to awaiting a stale, already-
    finished pump task), not let one bad reconnect turn a recoverable
    outage into the same permanent-death bug this whole supervisor exists
    to fix."""
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    async def run():
        client = _client()
        pump_calls = 0

        async def fake_pump_body():
            nonlocal pump_calls
            pump_calls += 1
            if pump_calls >= 2:
                client._closed = True  # the SECOND pump run (after a successful reconnect) ends the test
            return

        client._pump_task = asyncio.create_task(fake_pump_body())

        reconnect_attempts = 0

        async def fake_reconnect():
            nonlocal reconnect_attempts
            reconnect_attempts += 1
            if reconnect_attempts == 1:
                raise RuntimeError("simulated transient OTP exchange failure")
            client._ws = _FakeWs()
            client._pump_task = asyncio.create_task(fake_pump_body())

        client._reconnect_and_resubscribe = fake_reconnect

        await client._run_recv_pump_forever()

        assert pump_calls == 2  # the first (pre-supervisor) pump run, plus the one after a successful reconnect
        assert reconnect_attempts == 2  # first failed, second succeeded -- the supervisor retried the RECONNECT
        #                                 itself (with backoff), rather than spinning on the already-dead pump task

    asyncio.run(run())


def test_recv_pump_forever_stops_immediately_once_closed():
    async def run():
        client = _client()
        client._closed = True  # already closed before the supervisor even starts
        pump_calls = 0

        async def fake_pump_body():
            nonlocal pump_calls
            pump_calls += 1

        client._pump_task = asyncio.create_task(fake_pump_body())

        await client._run_recv_pump_forever()
        await _drain_pump_task(client)

        assert pump_calls == 0  # never even awaited the pump once -- closed check comes first

    asyncio.run(run())


# ---------------------------------------------------------------------------
# _reconnect_and_resubscribe
# ---------------------------------------------------------------------------

def test_reconnect_and_resubscribe_skips_work_if_already_open():
    """Guards the race between _run_recv_pump_forever noticing the pump
    died and ensure_connected() noticing a closed socket from an ordinary
    request -- whichever wins the lock first does the work; the other must
    not redo it (double-resubscribing would create duplicate Deriv
    subscriptions)."""
    async def run():
        client = _client()
        client._ws = _FakeWs(WsState.OPEN)  # already open

        async def fail_if_called():
            raise AssertionError("_open_socket must not be called when already open")

        client._open_socket = fail_if_called

        await client._reconnect_and_resubscribe()  # must return immediately, no error

    asyncio.run(run())


def test_reconnect_and_resubscribe_resubscribes_every_tick_symbol():
    async def run():
        client = _client()
        client._ws = None  # not connected
        client._tick_queues = {"R_100": asyncio.Queue(), "R_75": asyncio.Queue()}
        client._subscription_ids = {"R_100": "stale-sub-id"}

        async def fake_open_socket():
            client._ws = _FakeWs(WsState.OPEN)

        resubscribed = []

        async def fake_send_subscribe_request(symbol):
            resubscribed.append(symbol)

        client._open_socket = fake_open_socket
        client._send_subscribe_request = fake_send_subscribe_request

        await client._reconnect_and_resubscribe()
        await _drain_pump_task(client)  # real _recv_pump() task started on the fake socket

        assert set(resubscribed) == {"R_100", "R_75"}
        assert "R_100" not in client._subscription_ids  # stale id cleared before resubscribing

    asyncio.run(run())


def test_reconnect_and_resubscribe_continues_past_a_symbol_that_fails():
    async def run():
        client = _client()
        client._ws = None
        client._tick_queues = {"BAD": asyncio.Queue(), "GOOD": asyncio.Queue()}

        async def fake_open_socket():
            client._ws = _FakeWs(WsState.OPEN)

        resubscribed = []

        async def fake_send_subscribe_request(symbol):
            if symbol == "BAD":
                raise RuntimeError("simulated subscribe failure")
            resubscribed.append(symbol)

        client._open_socket = fake_open_socket
        client._send_subscribe_request = fake_send_subscribe_request

        await client._reconnect_and_resubscribe()  # must not raise despite BAD failing
        await _drain_pump_task(client)  # real _recv_pump() task started on the fake socket

        assert resubscribed == ["GOOD"]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# ensure_connected
# ---------------------------------------------------------------------------

def test_ensure_connected_reconnects_when_socket_is_not_open():
    async def run():
        client = _client()
        client._ws = None
        called = []

        async def fake_reconnect():
            called.append(True)

        client._reconnect_and_resubscribe = fake_reconnect

        await client.ensure_connected()

        assert called == [True]

    asyncio.run(run())


def test_reconnect_and_resubscribe_pump_is_live_before_resubscribing():
    """Root cause #2, direct regression test: the pump task started inside
    _reconnect_and_resubscribe() must actually be running (not None, not
    already-done) DURING the resubscribe loop, not just created afterward
    -- otherwise every _send() call in that loop is racing a socket nothing
    is reading, and is guaranteed to hang until request_timeout for every
    symbol (see this file's module docstring, root cause #2)."""
    async def run():
        client = _client()
        client._ws = None
        client._tick_queues = {"1HZ10V": asyncio.Queue()}
        observed_pump_task_during_resubscribe = []

        async def fake_open_socket():
            client._ws = _FakeWs(WsState.OPEN)

        async def fake_send_subscribe_request(symbol):
            # Snapshot self._pump_task WHILE resubscribing is in progress --
            # this is exactly the window where the earlier, broken version
            # left self._pump_task unset/stale.
            observed_pump_task_during_resubscribe.append(client._pump_task)

        client._open_socket = fake_open_socket
        client._send_subscribe_request = fake_send_subscribe_request

        await client._reconnect_and_resubscribe()
        await _drain_pump_task(client)

        assert observed_pump_task_during_resubscribe == [client._pump_task]
        assert observed_pump_task_during_resubscribe[0] is not None

    asyncio.run(run())


def test_ensure_connected_does_nothing_when_already_open():
    async def run():
        client = _client()
        client._ws = _FakeWs(WsState.OPEN)

        async def fail_if_called():
            raise AssertionError("_reconnect_and_resubscribe must not be called when already open")

        client._reconnect_and_resubscribe = fail_if_called

        await client.ensure_connected()  # must return cleanly, no error

    asyncio.run(run())
