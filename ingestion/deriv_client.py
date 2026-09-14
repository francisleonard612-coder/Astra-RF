"""
Deriv API client for Astra.

Two hard lessons from earlier bots in this account are baked in here:

1. AUTH: use Deriv's REST OTP Options API token exchange to get a
   pre-authenticated WebSocket URL, instead of the legacy
   connect-then-send-authorize-message flow. The legacy flow has produced
   401s at handshake time on this account before; the OTP exchange sidesteps
   that entirely.

   Current Deriv REST flow (as of the account-scoped Options API):
     - GET  {base}/trading/v1/options/accounts
           headers: Deriv-App-ID, Authorization: Bearer <token>
           -> list of accounts; we auto-pick one (see _resolve_account_id)
     - POST {base}/trading/v1/options/accounts/{accountId}/otp
           headers: Deriv-App-ID, Authorization: Bearer <token>
           (no JSON body)
           -> {"data": {"url": "wss://.../ws/demo?otp=..."}}
     - The OTP is valid for 120s and single-use, so the WS connection is
       opened immediately after the exchange.

2. NO INLINE AWAITS IN THE RECV LOOP: `_recv_pump` is the only coroutine that
   reads off the socket. If it ever `await`s a tick handler directly, and that
   handler calls something like `proposal()` or `buy()` which needs to read a
   response off the *same* socket, you get a self-deadlock -- the handler
   waits forever for a message that only `_recv_pump` can deliver, but
   `_recv_pump` is blocked awaiting the handler. This previously caused 100%
   of trade attempts to fail with "no response" / 1011 keepalive timeouts.
   The fix: `_recv_pump` never awaits handlers. Tick messages are pushed onto
   a per-symbol `asyncio.Queue` and consumed by independent worker tasks;
   request/response calls (proposal, buy, active_symbols, ticks_history) are
   resolved via a dict of `asyncio.Future`s keyed by req_id, and
   `_recv_pump` only ever does `future.set_result(...)`, never `await`.

3. RECV PUMP DEATH MUST NEVER BE A SILENT, PERMANENT OUTAGE: found from a
   real production log -- `_recv_pump` crashed on
   `ConnectionClosedError: no close frame received or sent` (Deriv or the
   network dropped the socket without a clean close handshake), logged
   "Recv pump crashed", and simply returned. Nothing else in this client
   reconnects the TICK path automatically: `ensure_connected()` only runs
   from `_send()`'s request/response path (proposal, buy, ticks_history,
   ...), which a pure tick-consuming `symbol_worker()` loop never calls --
   it just awaits its `asyncio.Queue` filling. So once the pump died, the
   process kept running (looked healthy in Railway) but every symbol's tick
   feed, and therefore all trading, silently stopped forever until a manual
   restart. `_run_recv_pump_forever()` (spawned by `connect()` instead of
   `_recv_pump` directly) supervises this: whenever `_recv_pump()` exits for
   any reason, it reconnects (backing off up to 30s between attempts) and
   resubscribes every live tick symbol before resuming the pump. This lives
   OUTSIDE `_recv_pump()` itself -- the reconnect only ever runs after the
   pump has already returned, never from within its loop -- so it doesn't
   violate rule 2 above.

   A follow-on bug from the same fix: `_reconnect_and_resubscribe()` used to
   open the new socket and immediately start sending `{"ticks": ..., "subscribe":
   1}` resubscribe requests through `_send()`, which blocks on a per-req_id
   `asyncio.Future` that only `_recv_pump()` ever resolves. But
   `_run_recv_pump_forever()` doesn't call `_recv_pump()` again until
   `_reconnect_and_resubscribe()` returns -- so nothing was reading the new
   socket while the resubscribes waited for their responses. Every resubscribe
   after a reconnect hung for the full `request_timeout` and failed with
   `TimeoutError` ("Failed to resubscribe ticks after reconnect"), leaving the
   symbol permanently unsubscribed even though the reconnect itself succeeded.
   Fix: `_reconnect_and_resubscribe()` now starts the pump reading the new
   socket (`self._pump_task = asyncio.create_task(self._recv_pump())`)
   *before* sending any resubscribe request, and `_run_recv_pump_forever()`
   awaits that same task instead of re-invoking `_recv_pump()` itself.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

import httpx
import websockets
from websockets.protocol import State as WsState

from app.logging_setup import get_logger

logger = get_logger("ingestion.deriv_client")


class DerivAuthError(RuntimeError):
    pass


class DerivRequestError(RuntimeError):
    def __init__(self, message: str, code: str | None = None, raw: dict | None = None):
        super().__init__(message)
        self.code = code
        self.raw = raw or {}


class _RateLimiter:
    """Sliding-window limiter for Deriv's shared per-connection budget.

    Deriv's docs (developers.deriv.com/docs/limits) count proposal,
    proposal_open_contract, buy, and sell against ONE shared budget of 360
    requests/minute per connection -- not 360 each. With several symbol
    workers sharing a single DerivClient/connection, each calling
    get_proposal() roughly once per tick, that budget was being blown
    through in seconds (observed: dozens of concurrent "You have reached
    the rate limit for proposal" errors within the same second).

    This waits BEFORE sending rather than firing and handling the rejection
    after the fact, per Deriv's own "pace your bursts" guidance -- cheaper
    than round-tripping a doomed request and matches how the proposal cache
    plus rate-limit backoff (see pricing/payout.py) already avoid most of
    this traffic and back off further once a rejection does slip through.
    """
    def __init__(self, max_per_window: int, window_seconds: float = 60.0):
        self.max_per_window = max_per_window
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                cutoff = now - self.window_seconds
                self._timestamps = [t for t in self._timestamps if t > cutoff]
                if len(self._timestamps) < self.max_per_window:
                    self._timestamps.append(now)
                    return
                sleep_for = self._timestamps[0] - cutoff
                await asyncio.sleep(max(sleep_for, 0.05))


@dataclass
class Tick:
    symbol: str
    epoch: int
    quote: float
    digit: int


class DerivClient:
    def __init__(self, app_id: str, api_token: str, ws_url: str, options_token_url: str,
                 account_id: str | None = None, use_real_account: bool = False,
                 request_timeout: float = 15.0):
        self.app_id = app_id
        self.api_token = api_token
        self.ws_url = ws_url  # unused by the current REST-OTP flow; kept only as a legacy fallback
        self.api_base_url = options_token_url.rstrip("/")
        self.account_id = account_id or None  # if unset, auto-resolved on first connect()
        # Which kind of account to auto-resolve to when account_id isn't pinned.
        # False (default) -> demo: trades place for real on Deriv's platform,
        # go through the exact same proposal/buy/settlement path as real
        # money, but settle against a demo account's play balance. This is
        # a materially different (and safer) thing than DRY_RUN, which never
        # calls buy() at all -- see app/config.py DRY_RUN docs.
        self.use_real_account = use_real_account
        self.request_timeout = request_timeout
        # Deriv's documented shared budget for {proposal, proposal_open_contract,
        # buy, sell} is 360/min per connection -- capped lower here (300) to
        # leave headroom for buy/sell/proposal_open_contract calls that share
        # the same budget outside of get_proposal()'s own traffic.
        self._quote_rate_limiter = _RateLimiter(max_per_window=300, window_seconds=60.0)
        self._rate_limited_keys = {"proposal", "proposal_open_contract", "buy", "sell"}

        self._ws: websockets.WebSocketClientProtocol | None = None
        self._req_id_counter = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._tick_queues: dict[str, asyncio.Queue] = {}
        self._tick_workers: dict[str, asyncio.Task] = {}
        self._subscription_ids: dict[str, str] = {}  # symbol -> deriv subscription id
        self._contract_queues: dict[int, asyncio.Queue] = {}
        self._contract_subscription_ids: dict[int, str] = {}  # contract_id -> deriv subscription id
        self._recv_task: asyncio.Task | None = None
        self._pump_task: asyncio.Task | None = None
        self._closed = False
        self._connect_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #
    async def connect(self) -> None:
        async with self._connect_lock:
            if not self.account_id:
                self.account_id = await self._resolve_account_id()
                logger.info("Auto-resolved Deriv account", extra={"extra_fields": {
                    "event_type": "account_resolved", "account_id": self.account_id,
                    "wanted": "real" if self.use_real_account else "demo",
                }})
            await self._open_socket()
            self._pump_task = asyncio.create_task(self._recv_pump(), name="deriv-recv-pump")
            # ONE long-lived supervisor task for the whole life of this
            # client, not re-spawned on every reconnect -- see this module's
            # docstring point 3 and _run_recv_pump_forever()'s own docstring
            # for why a fresh task per reconnect would risk two supervisors
            # racing to read the same (or a just-replaced) socket.
            self._recv_task = asyncio.create_task(self._run_recv_pump_forever(), name="deriv-recv-pump-supervisor")
            logger.info("Connected to Deriv", extra={"extra_fields": {"event_type": "ws_connected"}})

    async def _open_socket(self) -> None:
        """Exchanges a fresh (single-use) OTP and opens a new WebSocket
        connection, replacing self._ws. Does not touch self._recv_task or
        resubscribe anything -- callers (connect() for the very first
        connection, _reconnect_and_resubscribe() for every one after) own
        that."""
        auth_url = await self._exchange_otp(self.account_id)
        self._ws = await websockets.connect(auth_url, ping_interval=20, ping_timeout=20, close_timeout=5)
        self._closed = False

    async def _run_recv_pump_forever(self) -> None:
        """Supervises _recv_pump(): whenever it exits -- a clean close or,
        per this module's docstring point 3, a crash like
        ConnectionClosedError -- the socket is dead and every live tick
        subscription has silently stopped receiving pushes. Reconnects
        (exponential backoff, capped at 30s) and resubscribes every tick
        symbol before calling _recv_pump() again. Runs until close() sets
        self._closed.

        Deliberately does NOT create a new task or reassign self._recv_task
        on each reconnect -- this coroutine already IS self._recv_task, and
        cancelling/replacing it from inside itself would either self-cancel
        the very loop trying to recover, or leave two supervisors racing to
        read the same underlying socket. Reconnection happens in-place, in
        the same long-lived task connect() spawned once.
        """
        backoff = 1.0
        while not self._closed:
            await self._pump_task  # logs its own error and returns; only CancelledError propagates
            if self._closed:
                return
            logger.warning("Recv pump exited -- reconnecting", extra={"extra_fields": {
                "event_type": "recv_pump_reconnect", "backoff_seconds": round(backoff, 1)}})
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            try:
                await self._reconnect_and_resubscribe()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- a reconnect attempt failing must not kill the supervisor
                logger.error("Reconnect after recv pump exit failed", exc_info=exc, extra={"extra_fields": {
                    "event_type": "recv_pump_reconnect_failed"}})

    def _auth_headers(self) -> dict:
        return {"Deriv-App-ID": self.app_id, "Authorization": f"Bearer {self.api_token}"}

    async def _resolve_account_id(self) -> str:
        """Fetch the caller's Options accounts and pick one matching
        `self.use_real_account` (demo by default -- see __init__).

        Field-matching logic mirrors a separately-built and live-tested
        Deriv bot's account resolver rather than guessing at the schema:
        checks both `type` and `account_type` (whichever the response
        actually uses) against "real"/"demo", case-insensitively. If no
        account of the wanted kind is found, falls back to the first
        account returned and logs a warning -- better to trade on the
        wrong-but-visible account than silently fail to start.
        """
        if not self.api_token:
            raise DerivAuthError("DERIV_API_TOKEN is not set")
        url = f"{self.api_base_url}/trading/v1/options/accounts"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=self._auth_headers())
        if resp.status_code != 200:
            # Include token diagnostics (never the token itself) so an
            # expired/revoked token from Deriv is distinguishable in the
            # logs from a locally malformed one (e.g. trailing
            # newline/space pasted into the env var, which used to slip
            # through silently before DerivConfig started stripping
            # DERIV_APP_ID/DERIV_API_TOKEN).
            token_len = len(self.api_token)
            has_whitespace = self.api_token != self.api_token.strip()
            raise DerivAuthError(
                f"Fetching accounts failed: HTTP {resp.status_code} {resp.text[:300]} "
                f"(token_len={token_len}, token_has_surrounding_whitespace={has_whitespace}, "
                f"app_id={self.app_id!r})"
            )
        body = resp.json()
        accounts = body.get("data") or body.get("accounts") or (body if isinstance(body, list) else [])
        if not accounts:
            raise DerivAuthError(f"No Options accounts returned for this token: {body}")

        wanted = "real" if self.use_real_account else "demo"
        for acc in accounts:
            acc_type = str(acc.get("type") or acc.get("account_type") or "").lower()
            if acc_type == wanted:
                account_id = acc.get("account_id") or acc.get("id")
                if account_id:
                    return account_id

        first = accounts[0]
        account_id = first.get("account_id") or first.get("id")
        if not account_id:
            raise DerivAuthError(f"Account entry had no id field: {first}")
        logger.warning(f"No '{wanted}' account found, falling back to first returned account",
                        extra={"extra_fields": {
                            "event_type": "account_type_mismatch",
                            "wanted": wanted, "fallback_account_id": account_id,
                        }})
        return account_id

    async def _exchange_otp(self, account_id: str) -> str:
        """Exchange the long-lived API token for a pre-authenticated WS URL.

        OTP is valid for 120s and single-use, so the caller must open the
        WebSocket connection immediately after this returns.
        """
        if not self.api_token:
            raise DerivAuthError("DERIV_API_TOKEN is not set")
        url = f"{self.api_base_url}/trading/v1/options/accounts/{account_id}/otp"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, headers=self._auth_headers())
        if resp.status_code != 200:
            raise DerivAuthError(f"OTP token exchange failed: HTTP {resp.status_code} {resp.text[:300]}")
        body = resp.json()
        data = body.get("data", body)
        auth_url = data.get("url") or data.get("websocket_url") or data.get("ws_url")
        if not auth_url:
            # Fall back for older/legacy deployments that return a bare OTP instead
            # of a ready-to-use URL.
            otp = data.get("otp") or data.get("token")
            if not otp:
                raise DerivAuthError(f"OTP exchange response had no usable URL/token: {body}")
            auth_url = f"{self.ws_url}?app_id={self.app_id}&otp={otp}"
        return auth_url

    async def close(self) -> None:
        self._closed = True
        for task in list(self._tick_workers.values()):
            task.cancel()
        if self._recv_task:
            self._recv_task.cancel()
        if self._pump_task:
            self._pump_task.cancel()
        if self._ws is not None:
            await self._ws.close()

    async def ensure_connected(self) -> None:
        # websockets 14+ replaced the old boolean `.closed` property on the
        # connection object with a `.state` enum (websockets.protocol.State).
        # `.closed` doesn't exist on the ClientConnection this version returns
        # -- checking it raises AttributeError on every call and crashes the
        # bot right after every successful connect (confirmed against this
        # exact codebase: "'ClientConnection' object has no attribute
        # 'closed'"). requirements.txt pins `websockets>=12.0` with no upper
        # bound, so any environment installing a current version hits this.
        is_open = self._ws is not None and self._ws.state is WsState.OPEN
        if not is_open:
            await self._reconnect_and_resubscribe()

    async def _reconnect_and_resubscribe(self) -> None:
        """Shared reconnect path for BOTH triggers: _run_recv_pump_forever()
        noticing the pump itself died, and ensure_connected() noticing a
        closed socket from an ordinary request (_send()) before the pump
        has caught up. Whichever gets here first does the real work; the
        other finds the socket already open under the lock and returns
        immediately -- see the early-return check below."""
        async with self._connect_lock:
            if self._ws is not None and self._ws.state is WsState.OPEN:
                return  # someone else already reconnected while this caller waited for the lock
            logger.warning("Reconnecting to Deriv", extra={"extra_fields": {"event_type": "ws_reconnect"}})
            await self._open_socket()
            # Start the pump reading the NEW socket now, before resubscribing.
            # _send_subscribe_request() below goes through _send(), which
            # blocks on a per-req_id asyncio.Future that only _recv_pump()
            # resolves. _run_recv_pump_forever() doesn't await _recv_pump()
            # again until this function returns, so if the pump isn't
            # started here first, nothing is reading the new socket while
            # we wait for the resubscribe responses -- every resubscribe
            # would hang until _send()'s request_timeout and fail with
            # TimeoutError (this is what produced "Failed to resubscribe
            # ticks after reconnect" in production: the reconnect itself
            # succeeded, but the resubscribe sends had no reader).
            self._pump_task = asyncio.create_task(self._recv_pump(), name="deriv-recv-pump")
            # re-subscribe every symbol we were watching. Contract-update
            # subscriptions are NOT resubscribed here deliberately --
            # wait_for_contract_settlement()'s own bounded timeout + finally
            # already handles a subscription silently going quiet (returns
            # whatever partial state it has, or {} on timeout, which callers
            # already treat as an unknown/cancel-worthy outcome), so there's
            # no permanent-outage risk there the way there is for ticks.
            for symbol in list(self._tick_queues.keys()):
                self._subscription_ids.pop(symbol, None)
                try:
                    await self._send_subscribe_request(symbol)
                except Exception as exc:  # noqa: BLE001 -- one symbol failing to resubscribe must not block the rest
                    logger.error("Failed to resubscribe ticks after reconnect", exc_info=exc,
                                 extra={"extra_fields": {"symbol": symbol, "event_type": "resubscribe_failed"}})

    # ------------------------------------------------------------------ #
    # Low-level request/response
    # ------------------------------------------------------------------ #
    async def _send(self, payload: dict) -> dict:
        """Send a request and await its matching response. Never called from _recv_pump."""
        if self._rate_limited_keys.intersection(payload.keys()):
            await self._quote_rate_limiter.acquire()
        await self.ensure_connected()
        req_id = next(self._req_id_counter)
        payload = {**payload, "req_id": req_id}
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self._ws.send(json.dumps(payload))
            result = await asyncio.wait_for(fut, timeout=self.request_timeout)
        finally:
            self._pending.pop(req_id, None)
        if "error" in result:
            err = result["error"]
            raise DerivRequestError(err.get("message", "Deriv API error"), code=err.get("code"), raw=result)
        return result

    async def _recv_pump(self) -> None:
        """The ONLY coroutine allowed to read from the socket. Never awaits handlers."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.error("Malformed message from Deriv", extra={"extra_fields": {"raw": raw[:200]}})
                    continue

                msg_type = msg.get("msg_type")
                req_id = msg.get("req_id")

                if msg_type == "tick" and req_id is None:
                    self._route_tick(msg)
                    continue

                if msg_type == "proposal_open_contract" and req_id is None:
                    self._route_contract_update(msg)
                    continue

                if req_id is not None and req_id in self._pending:
                    fut = self._pending[req_id]
                    if not fut.done():
                        fut.set_result(msg)
                    # a "tick" response also carries the FIRST tick + a
                    # subscription id -- route that first tick too. Same
                    # reasoning for the first proposal_open_contract push
                    # that comes back as the subscribe call's own response.
                    if msg_type == "tick":
                        self._route_tick(msg)
                    elif msg_type == "proposal_open_contract":
                        self._route_contract_update(msg)
                    continue

                # Unmatched message (e.g. late subscription tick after we
                # stopped waiting) -- route ticks/contract updates, log
                # everything else.
                if msg_type == "tick":
                    self._route_tick(msg)
                elif msg_type == "proposal_open_contract":
                    self._route_contract_update(msg)
                else:
                    logger.debug("Unrouted message", extra={"extra_fields": {"msg_type": msg_type}})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Recv pump crashed", exc_info=exc,
                         extra={"extra_fields": {"event_type": "recv_pump_error"}})
            # fail every pending future so callers don't hang forever
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(exc)

    def _route_tick(self, msg: dict) -> None:
        tick = msg.get("tick")
        if not tick:
            return
        symbol = tick.get("symbol")
        queue = self._tick_queues.get(symbol)
        if queue is None:
            return
        quote = float(tick["quote"])
        digit = _last_digit(quote, tick.get("pip_size"))
        parsed = Tick(symbol=symbol, epoch=int(tick["epoch"]), quote=quote, digit=digit)
        try:
            queue.put_nowait(parsed)
        except asyncio.QueueFull:
            logger.warning("Tick queue full, dropping tick",
                            extra={"extra_fields": {"symbol": symbol, "event_type": "queue_overflow"}})
        if "id" in tick:
            self._subscription_ids[symbol] = tick["id"]

    def _route_contract_update(self, msg: dict) -> None:
        poc = msg.get("proposal_open_contract")
        if not poc:
            return
        contract_id = poc.get("contract_id")
        if contract_id is None:
            return
        queue = self._contract_queues.get(contract_id)
        if queue is not None:
            try:
                queue.put_nowait(poc)
            except asyncio.QueueFull:
                # Drop the OLDEST pending update rather than this one -- the
                # newest state is the one closer to the terminal (is_sold)
                # state wait_for_contract_settlement is actually waiting for.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                queue.put_nowait(poc)
        sub_id = msg.get("subscription", {}).get("id")
        if sub_id:
            self._contract_subscription_ids[contract_id] = sub_id

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def get_active_synthetic_symbols(self, prefixes: list[str]) -> list[str]:
        # "product_type" was removed from the request in Deriv's current
        # (non-legacy) Options API -- sending it causes a validation error.
        resp = await self._send({"active_symbols": "brief"})
        symbols = []
        for s in resp.get("active_symbols", []):
            if s.get("market") != "synthetic_index":
                continue
            # Response field renamed: "symbol" -> "underlying_symbol".
            code = s.get("underlying_symbol", "")
            if any(code.startswith(p) for p in prefixes):
                symbols.append(code)
        return sorted(set(symbols))

    async def get_contracts_for(self, symbol: str) -> dict:
        """Live per-symbol contract/barrier/duration limits -- see
        pricing/duration_grid.py's filter_to_allowed(), which cross-checks a
        candidate duration grid against this before it's trusted. Not in
        _rate_limited_keys: contracts_for shares no budget with
        proposal/proposal_open_contract/buy/sell."""
        resp = await self._send({"contracts_for": symbol})
        return resp.get("contracts_for", {})

    async def subscribe_ticks(self, symbol: str, queue_size: int = 2000) -> asyncio.Queue:
        if symbol not in self._tick_queues:
            self._tick_queues[symbol] = asyncio.Queue(maxsize=queue_size)
        await self._send_subscribe_request(symbol)
        return self._tick_queues[symbol]

    async def _send_subscribe_request(self, symbol: str) -> None:
        resp = await self._send({"ticks": symbol, "subscribe": 1})
        sub = resp.get("subscription", {})
        if sub.get("id"):
            self._subscription_ids[symbol] = sub["id"]

    async def get_history(self, symbol: str, count: int = 5000) -> list[Tick]:
        resp = await self._send({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "style": "ticks",
        })
        history = resp.get("history", {})
        prices = history.get("prices", [])
        times = history.get("times", [])
        out = []
        for t, p in zip(times, prices):
            p = float(p)
            out.append(Tick(symbol=symbol, epoch=int(t), quote=p, digit=_last_digit(p, None)))
        return out

    async def get_proposal(self, symbol: str, contract_type: str, barrier: int | None, stake: float,
                            duration: int, duration_unit: str, currency: str) -> dict:
        payload = {
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": currency,
            # Request field renamed: "symbol" -> "underlying_symbol" in
            # Deriv's current (non-legacy) Options API. Sending "symbol"
            # returns InputValidationFailed: Properties not allowed: symbol.
            "underlying_symbol": symbol,
            "duration": duration,
            "duration_unit": duration_unit,
        }
        # Rise/Fall (CALL/PUT) contracts have no barrier -- Deriv rejects a
        # barrier field entirely for these rather than ignoring it, so it
        # must be omitted, not sent as None/"None".
        if barrier is not None:
            payload["barrier"] = str(barrier)
        resp = await self._send(payload)
        return resp.get("proposal", {})

    async def buy(self, proposal_id: str, price: float) -> dict:
        resp = await self._send({"buy": proposal_id, "price": price})
        return resp.get("buy", {})

    async def get_balance(self) -> dict:
        resp = await self._send({"balance": 1})
        return resp.get("balance", {})

    async def subscribe_contract_updates(self, contract_id: int, queue_size: int = 50) -> asyncio.Queue:
        if contract_id not in self._contract_queues:
            self._contract_queues[contract_id] = asyncio.Queue(maxsize=queue_size)
        resp = await self._send({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1})
        # _recv_pump already routes this same response via its req_id branch
        # (which calls _route_contract_update too) -- this is a defensive
        # second attempt in case the queue was created after that routing
        # already ran, so the very first state is never lost.
        poc = resp.get("proposal_open_contract")
        if poc:
            try:
                self._contract_queues[contract_id].put_nowait(poc)
            except asyncio.QueueFull:
                pass
            sub_id = resp.get("subscription", {}).get("id") or poc.get("id")
            if sub_id:
                self._contract_subscription_ids[contract_id] = sub_id
        return self._contract_queues[contract_id]

    async def forget_contract_subscription(self, contract_id: int) -> None:
        sub_id = self._contract_subscription_ids.pop(contract_id, None)
        self._contract_queues.pop(contract_id, None)
        if sub_id:
            try:
                await self._send({"forget": sub_id})
            except DerivRequestError:
                pass  # already gone / never confirmed -- not worth failing settlement over

    async def wait_for_contract_settlement(self, contract_id: int, timeout: float = 30.0) -> dict:
        """Subscribes to proposal_open_contract pushes instead of polling.

        The old implementation polled with a plain (non-subscribed)
        proposal_open_contract request every 0.5s until is_sold -- fine for
        a ~2-4s digit contract (a handful of polls), but each poll is a
        fresh request against the SAME shared proposal/proposal_open_contract
        /buy/sell rate-limit budget that already caused a real production
        outage once (see pricing/payout.py). A multi-minute Rise/Fall
        contract at 0.5s polling is 600+ requests for ONE open position --
        exactly the load that budget can't absorb, especially with several
        positions open concurrently. Subscribing costs one request total;
        every update after that is pushed, not polled, per Deriv's own
        "subscribe instead of polling" guidance.
        """
        queue = await self.subscribe_contract_updates(contract_id)
        deadline = time.monotonic() + timeout
        contract: dict = {}
        try:
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    contract = await asyncio.wait_for(queue.get(), timeout=max(remaining, 0.1))
                except asyncio.TimeoutError:
                    break
                if contract.get("is_sold"):
                    break
        finally:
            await self.forget_contract_subscription(contract_id)
        return contract


def _last_digit(quote: float, pip_size: int | float | None) -> int:
    """Extract the last significant digit of a Deriv quote given its pip size."""
    if pip_size:
        decimals = len(str(pip_size).split(".")[-1]) if "." in str(pip_size) else 0
    else:
        s = f"{quote}"
        decimals = len(s.split(".")[-1]) if "." in s else 0
    scaled = round(quote * (10 ** decimals))
    return int(scaled % 10)
