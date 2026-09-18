"""
questrade.py — reading a Questrade account through their v1 API.

The whole file exists around one awkward fact: **Questrade refresh tokens are
single-use and self-rotating.** You exchange a refresh token for an access
token, and the response carries a NEW refresh token; the one you just used is
dead the instant the exchange succeeds. There is no way to re-read it, no
second copy, and no "list my tokens" endpoint. Lose the new one — crash between
the response and the write, two workers refreshing at once, a retry on a
request that actually succeeded — and the chain is broken for good. The user
has to go back to the App Hub and generate another by hand.

So three rules run through everything here:

  1. Persist first, use second. The new refresh token is written to storage
     before the access token is handed to any caller. A crash after the write
     costs nothing; a crash before it costs the connection.
  2. One refresh at a time, per user. A lock around the exchange, because two
     concurrent refreshes with the same token means one of them gets a dead
     token back and both may end up writing.
  3. Never retry an exchange. A timeout is not a failure — Questrade may have
     rotated the token and the response was simply lost. Retrying guarantees
     the second attempt fails with a token we can no longer replace. We record
     what happened and tell the user plainly.

Everything below the token layer is ordinary: `api_server` comes back with each
exchange and must be used as given (it is not always api01), access tokens last
30 minutes, and a 401 mid-session means refresh once and retry once.

Scope: read only. The client never calls an order endpoint, and the setup
instructions ask for read permissions only, so a leaked token cannot trade.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

LOGIN_URL = "https://login.questrade.com/oauth2/token"

#: Refresh this many seconds before the access token actually expires, so a
#: request never starts with a token that dies mid-flight.
EXPIRY_SLACK_S = 120

#: Questrade's documented ceilings: 30/s and 30,000/h for account calls,
#: 20/s and 15,000/h for market data. Nothing here comes close, but the
#: remaining-count header is surfaced so a future batch job can see it.
RATE_HEADER = "X-RateLimit-Remaining"

HTTP_TIMEOUT_S = 20

#: Questrade account types, as returned by v1/accounts. Mapped to display
#: names here rather than in the UI because the tax treatment is the point:
#: an RRSP loss is not harvestable and a TFSA gain is not taxable, so the
#: label has to survive all the way to the screen.
ACCOUNT_TYPES = {
    "Cash": "现金账户 Cash",
    "Margin": "保证金 Margin",
    "TFSA": "免税账户 TFSA",
    "RRSP": "退休账户 RRSP",
    "SRRSP": "配偶 RRSP",
    "LRRSP": "锁定 LRRSP",
    "LIRA": "锁定 LIRA",
    "LIF": "LIF",
    "RIF": "RIF",
    "SRIF": "配偶 RIF",
    "RESP": "教育金 RESP",
    "FRESP": "家庭 RESP",
    "FHSA": "首次购房 FHSA",
}


class QuestradeError(Exception):
    """Anything that went wrong talking to Questrade."""


class NeedsReconnect(QuestradeError):
    """
    The refresh-token chain is broken; only the user can fix it.

    Raised rather than returned so no caller can mistake it for an empty
    portfolio. The message is written to be shown verbatim.
    """


class RateLimited(QuestradeError):
    def __init__(self, reset_at: float | None = None):
        super().__init__("Questrade 接口调用过于频繁，请稍后再试。")
        self.reset_at = reset_at


@dataclass
class Tokens:
    """What has to survive between requests. `refresh` is the irreplaceable one."""
    refresh: str
    access: str = ""
    api_server: str = ""
    expires_at: float = 0.0
    connected_at: float = 0.0
    last_error: str = ""

    def usable(self, now: float) -> bool:
        return bool(self.access and self.api_server
                    and now < self.expires_at - EXPIRY_SLACK_S)

    def as_record(self) -> dict:
        return {"refresh_token": self.refresh, "access_token": self.access,
                "api_server": self.api_server, "expires_at": self.expires_at,
                "connected_at": self.connected_at, "last_error": self.last_error}

    @classmethod
    def from_record(cls, rec: dict) -> "Tokens":
        return cls(refresh=str(rec.get("refresh_token") or ""),
                   access=str(rec.get("access_token") or ""),
                   api_server=str(rec.get("api_server") or ""),
                   expires_at=float(rec.get("expires_at") or 0),
                   connected_at=float(rec.get("connected_at") or 0),
                   last_error=str(rec.get("last_error") or ""))


class TokenStore:
    """
    Where the rotating refresh token lives between requests.

    A protocol, not an implementation, so the client is testable without a
    database — and so the storage can move later without touching this file.
    `key` identifies the owner and is what the refresh lock is keyed on.
    """

    key: str = "default"

    def load(self) -> dict | None:
        raise NotImplementedError

    def save(self, record: dict) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


@dataclass
class MemoryStore(TokenStore):
    """For tests, and for a single-shot script."""
    key: str = "memory"
    record: dict | None = None
    writes: list = field(default_factory=list)

    def load(self) -> dict | None:
        return dict(self.record) if self.record else None

    def save(self, record: dict) -> None:
        self.record = dict(record)
        self.writes.append(dict(record))

    def clear(self) -> None:
        self.record = None


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


class Questrade:
    """
    One user's connection. Cheap to construct; the state lives in the store.

    `session` is anything with .get and .post shaped like requests'. Injected so
    the tests never open a socket, and so a future move to httpx is one line.
    """

    def __init__(self, store: TokenStore, *, session=None, now=time.time):
        self.store = store
        self.now = now
        self._session = session

    @property
    def session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    # ── tokens ───────────────────────────────────────────────────────────────
    def connect(self, manual_refresh_token: str) -> Tokens:
        """
        Adopt a refresh token generated by hand in the Questrade App Hub.

        Exchanged immediately rather than stored as given: a token that does not
        work must fail here, while the user is looking at the screen that asked
        for it, not four hours later inside a background refresh.
        """
        token = (manual_refresh_token or "").strip()
        if not token:
            raise NeedsReconnect("请输入 Questrade 刷新令牌 (refresh token)。")
        self.store.save(Tokens(refresh=token).as_record())
        tokens = self._exchange(token, first=True)
        return tokens

    def disconnect(self) -> None:
        self.store.clear()

    def status(self) -> dict:
        """Whether this user is connected — never the token itself."""
        rec = self.store.load()
        if not rec or not rec.get("refresh_token"):
            return {"connected": False, "reason": None}
        t = Tokens.from_record(rec)
        return {
            "connected": True,
            "api_server": t.api_server or None,
            "connected_at": _iso(t.connected_at) if t.connected_at else None,
            "access_expires_at": _iso(t.expires_at) if t.expires_at else None,
            "reason": t.last_error or None,
        }

    def _tokens(self) -> Tokens:
        rec = self.store.load()
        if not rec or not rec.get("refresh_token"):
            raise NeedsReconnect("尚未连接 Questrade —— 请先在页面上填入刷新令牌。")
        t = Tokens.from_record(rec)
        if t.usable(self.now()):
            return t

        # Serialised per user: two threads exchanging the same refresh token
        # means one of them is handed a token that is already dead.
        with _lock_for(self.store.key):
            rec = self.store.load() or {}
            t = Tokens.from_record(rec)
            if t.usable(self.now()):
                return t          # someone else refreshed while we waited
            if not t.refresh:
                raise NeedsReconnect("尚未连接 Questrade —— 请先在页面上填入刷新令牌。")
            return self._exchange(t.refresh, prior=t)

    def _exchange(self, refresh: str, *, first: bool = False,
                  prior: Tokens | None = None) -> Tokens:
        """
        Trade a refresh token for an access token, and persist what comes back.

        Deliberately not retried. A timeout here is ambiguous — Questrade may
        have rotated the token and lost the response on the way back — and a
        retry turns "probably fine" into "certainly broken".
        """
        try:
            data = self._token_request(refresh)
        except NeedsReconnect:
            self._mark_broken("刷新令牌已失效，请在 Questrade App Hub 重新生成。")
            raise
        except QuestradeError as exc:
            # The token MAY have been consumed. Say so rather than implying the
            # connection is intact.
            self._mark_broken(f"上次刷新未完成：{exc}")
            raise

        tokens = Tokens(
            refresh=str(data["refresh_token"]),
            access=str(data["access_token"]),
            api_server=str(data["api_server"]).rstrip("/") + "/",
            expires_at=self.now() + float(data.get("expires_in") or 1800),
            connected_at=(self.now() if first else
                          (prior.connected_at if prior else self.now())),
            last_error="",
        )
        # Written BEFORE the caller can use it. Rule 1 in the module docstring.
        self.store.save(tokens.as_record())
        return tokens

    def _token_request(self, refresh: str) -> dict:
        payload = {"grant_type": "refresh_token", "refresh_token": refresh}

        # A transport failure here is the dangerous case, not an ordinary one:
        # the request may have reached Questrade, rotated the token, and lost
        # the reply on the way back. Turned into a QuestradeError so the caller
        # records it against the connection instead of letting a bare
        # ConnectionError escape as an unexplained 500.
        def send(verb, **kw):
            try:
                return getattr(self.session, verb)(LOGIN_URL, timeout=HTTP_TIMEOUT_S, **kw)
            except QuestradeError:
                raise
            except Exception as exc:                               # noqa: BLE001
                raise QuestradeError(f"无法连接 Questrade 登录接口（{type(exc).__name__}）")

        # POST keeps the token out of the request line, and so out of any proxy
        # or access log that records URLs. Questrade documents the GET form, so
        # a routing-level rejection — and only that — falls back to it. A 400 is
        # a bad token, not a bad method, and must never be retried.
        r = send("post", data=payload)
        if r.status_code in (404, 405, 415):
            r = send("get", params=payload)

        if r.status_code in (400, 401):
            raise NeedsReconnect(
                "Questrade 拒绝了这个刷新令牌（可能已被使用或已过期）。"
                "请到 apphub.questrade.com 重新生成一个，再粘贴进来。")
        if r.status_code == 429:
            raise RateLimited()
        if r.status_code >= 400:
            raise QuestradeError(f"Questrade 登录接口返回 {r.status_code}")

        try:
            data = r.json()
        except Exception:                                          # noqa: BLE001
            raise QuestradeError("Questrade 登录接口返回了无法解析的内容")
        missing = {"access_token", "refresh_token", "api_server"} - set(data)
        if missing:
            raise QuestradeError(f"Questrade 返回缺少字段：{', '.join(sorted(missing))}")
        return data

    def _mark_broken(self, reason: str) -> None:
        rec = self.store.load()
        if rec:
            rec["last_error"] = reason
            rec["access_token"] = ""
            rec["expires_at"] = 0
            self.store.save(rec)

    # ── requests ─────────────────────────────────────────────────────────────
    def _get(self, path: str, params: dict | None = None, *, retry: bool = True):
        t = self._tokens()
        url = t.api_server + path.lstrip("/")
        r = self.session.get(url, params=params or {},
                             headers={"Authorization": f"Bearer {t.access}"},
                             timeout=HTTP_TIMEOUT_S)

        if r.status_code == 401 and retry:
            # The access token died early — revoked, or the clock drifted.
            # One forced refresh, one retry, then give up.
            with _lock_for(self.store.key):
                cur = Tokens.from_record(self.store.load() or {})
                if cur.access == t.access and cur.refresh:
                    self._exchange(cur.refresh, prior=cur)
            return self._get(path, params, retry=False)

        if r.status_code == 429:
            raise RateLimited()
        if r.status_code >= 400:
            raise QuestradeError(f"Questrade {path} 返回 {r.status_code}")
        try:
            return r.json()
        except Exception:                                          # noqa: BLE001
            raise QuestradeError(f"Questrade {path} 返回了无法解析的内容")

    # ── the endpoints this app uses ──────────────────────────────────────────
    def accounts(self) -> list[dict]:
        rows = self._get("v1/accounts").get("accounts") or []
        return [{
            "id": str(a.get("number") or ""),
            "type": str(a.get("type") or ""),
            "label": ACCOUNT_TYPES.get(str(a.get("type") or ""), str(a.get("type") or "账户")),
            "status": str(a.get("status") or ""),
            "primary": bool(a.get("isPrimary")),
            "client_type": str(a.get("clientAccountType") or ""),
        } for a in rows if a.get("number")]

    def positions(self, account_id: str) -> list[dict]:
        return self._get(f"v1/accounts/{account_id}/positions").get("positions") or []

    def balances(self, account_id: str) -> dict:
        return self._get(f"v1/accounts/{account_id}/balances") or {}

    def symbols(self, ids: list[int]) -> list[dict]:
        """Instrument detail for up to 100 symbolIds per call, as Questrade allows."""
        out: list[dict] = []
        ids = [int(i) for i in ids if i]
        for chunk in (ids[i:i + 100] for i in range(0, len(ids), 100)):
            got = self._get("v1/symbols", {"ids": ",".join(map(str, chunk))})
            out.extend(got.get("symbols") or [])
        return out

    def quotes(self, ids: list[int]) -> list[dict]:
        out: list[dict] = []
        ids = [int(i) for i in ids if i]
        for chunk in (ids[i:i + 100] for i in range(0, len(ids), 100)):
            got = self._get("v1/markets/quotes", {"ids": ",".join(map(str, chunk))})
            out.extend(got.get("quotes") or [])
        return out

    def candles(self, symbol_id: int, *, days: int = 365,
                interval: str = "OneDay") -> list[dict]:
        """
        Historical bars. NOTE: Questrade returns them UNADJUSTED for splits and
        dividends, so they are not interchangeable with the Yahoo series the
        indicators are calibrated on. Used for display, never for the risk
        statistics — see portfolio.py.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        got = self._get(f"v1/markets/candles/{int(symbol_id)}", {
            "startTime": _qt_time(start), "endTime": _qt_time(end),
            "interval": interval,
        })
        return got.get("candles") or []


def _qt_time(dt: datetime) -> str:
    """Questrade wants ISO-8601 WITH an offset; a bare 'Z' is rejected."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")
