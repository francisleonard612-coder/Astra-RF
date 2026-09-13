"""
Regression tests for ingestion/deriv_client.py's recv-pump supervisor.

Root cause being guarded against: a live deployment log showed
`_recv_pump` crash on `ConnectionClosedError: no close frame received or
sent`, log "Recv pump crashed", and simply return -- with nothing else in
the client automatically reconnecting the TICK path (ensure_connected()
only runs from the request/response path, which a pure tick-consuming
symbol_worker() loop never calls). The process kept running, but every
symbol's tick feed silently died forever. These tests exercise the fix
(_run_recv_pump_forever supervising _recv_pump, and the shared
_reconnect_and_resubscribe() path) without a real network connection, by
monkeypatching the socket-opening and pump internals.
"""
import asyncio

from websockets.protocol import State as WsState

from ingestion.deriv_client import DerivClient


class _FakeWs:
    def __init__(self, state=WsState.OPEN):
        self.state = state


def _client() -> DerivClient:
    return DerivClient(app_id="1", api_token="tok", ws_url="wss://x", options_token_url="https://x")


async def _instant_sleep(*_args, **_kwargs) -> None:
    """Stand-in for asyncio.sleep in tests -- returns immediately instead
    of actually backing off, without recursing into itself (a lambda
    calling `asyncio.sleep(0)` after `asyncio.sleep` has been monkeypatched
    to that same lambda recurses forever)."""
    return None


# ---------------------------------------------------------------------------
# _run_recv_pump_forever
# ---------------------------------------------------------------------------

def test_recv_pump_forever_reconnects_after_the_pump_exits(monkeypatch):
    """The exact production scenario: _recv_pump() exits (crash or clean
    close, doesn't matter which -- _recv_pump already logs the difference
    internally and returns either way), and the supervisor must reconnect
    and resume, not just let the tick feed die silently forever."""
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)  # skip the real backoff delay

    async def run():
        client = _client()
        pump_calls = 0

        async def fake_recv_pump():
            nonlocal pump_calls
            pump_calls += 1
            if pump_calls >= 2:
                client._closed = True  # stop the supervisor loop after this second exit
            return  # simulates _recv_pump exiting, exactly as it does after logging its own crash

        reconnect_calls = 0

        async def fake_reconnect():
            nonlocal reconnect_calls
            reconnect_calls += 1
            client._ws = _FakeWs()  # a real reconnect would leave a live socket behind

        client._recv_pump = fake_recv_pump
        client._reconnect_and_resubscribe = fake_reconnect

        await client._run_recv_pump_forever()

        assert pump_calls == 2  # ran again after the first exit, then stopped once closed
        assert reconnect_calls == 1  # reconnected exactly once, between the two pump runs

    asyncio.run(run())


def test_recv_pump_forever_keeps_retrying_if_reconnect_itself_fails(monkeypatch):
    """A failed reconnect attempt (e.g. a transient OTP-exchange outage)
    must not kill the supervisor -- it should back off and try again, not
    let one bad reconnect turn a recoverable outage into the same
    permanent-death bug this whole supervisor exists to fix."""
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    async def run():
        client = _client()
        pump_calls = 0

        async def fake_recv_pump():
            nonlocal pump_calls
            pump_calls += 1
            if pump_calls >= 3:
                client._closed = True
            return

        reconnect_attempts = 0

        async def fake_reconnect():
            nonlocal reconnect_attempts
            reconnect_attempts += 1
            if reconnect_attempts == 1:
                raise RuntimeError("simulated transient OTP exchange failure")
            client._ws = _FakeWs()

        client._recv_pump = fake_recv_pump
        client._reconnect_and_resubscribe = fake_reconnect

        await client._run_recv_pump_forever()

        assert pump_calls == 3
        assert reconnect_attempts == 2  # first failed, second succeeded -- loop kept going

    asyncio.run(run())


def test_recv_pump_forever_stops_immediately_once_closed():
    async def run():
        client = _client()
        client._closed = True  # already closed before the supervisor even starts
        pump_calls = 0

        async def fake_recv_pump():
            nonlocal pump_calls
            pump_calls += 1

        client._recv_pump = fake_recv_pump

        await client._run_recv_pump_forever()

        assert pump_calls == 0  # never even ran the pump once

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


def test_ensure_connected_does_nothing_when_already_open():
    async def run():
        client = _client()
        client._ws = _FakeWs(WsState.OPEN)

        async def fail_if_called():
            raise AssertionError("_reconnect_and_resubscribe must not be called when already open")

        client._reconnect_and_resubscribe = fail_if_called

        await client.ensure_connected()  # must return cleanly, no error

    asyncio.run(run())
