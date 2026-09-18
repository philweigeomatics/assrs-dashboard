"""
questrade: the token chain, which is the only part that can break permanently.

Offline — a fake session stands in for requests. Everything here is about one
property: a Questrade refresh token is single-use, and the replacement arrives
exactly once, in the body of the response that killed the old one. Drop it and
the user has to go back to the App Hub by hand. So these tests care far more
about WHEN we write than about what we parse.

    python -m pytest api/tests/test_questrade.py -q
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import questrade as qt  # noqa: E402


class Response:
    def __init__(self, status=200, body=None, raises=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self._raises = raises

    def json(self):
        if self._raises:
            raise self._raises
        return self._body


class FakeSession:
    """Records every call, and replays scripted responses per URL."""

    def __init__(self, token_responses=None, api_responses=None, token_latency_s=0.0):
        self.token_latency_s = token_latency_s
        self.token_responses = list(token_responses or [])
        self.api_responses = dict(api_responses or {})
        self.calls: list[tuple[str, str, dict]] = []
        self.serial = 0

    def post(self, url, data=None, timeout=None, **kw):
        self.calls.append(("POST", url, dict(data or {})))
        return self._token()

    def get(self, url, params=None, headers=None, timeout=None, **kw):
        self.calls.append(("GET", url, dict(params or {})))
        if url.startswith(qt.LOGIN_URL):
            return self._token()
        for key, resp in self.api_responses.items():
            if key in url:
                return resp() if callable(resp) else resp
        return Response(404, {})

    def _token(self):
        # A real exchange is a network round trip. Without some latency the
        # threads in the concurrency test never overlap and the test passes
        # with the lock removed — which is exactly the bug it exists to catch.
        time.sleep(self.token_latency_s)
        if self.token_responses:
            nxt = self.token_responses.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        self.serial += 1
        return Response(200, {
            "access_token": f"acc{self.serial}",
            "refresh_token": f"ref{self.serial}",
            "api_server": "https://api07.iq.questrade.com/",
            "expires_in": 1800,
            "token_type": "Bearer",
        })

    @property
    def token_calls(self):
        return [c for c in self.calls if c[1].startswith(qt.LOGIN_URL)]


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def client(session=None, clock=None, store=None):
    store = store or qt.MemoryStore(key=f"k{id(session)}")
    return qt.Questrade(store, session=session or FakeSession(),
                        now=clock or Clock()), store


# ── connecting ───────────────────────────────────────────────────────────────
def test_the_pasted_token_is_exchanged_and_never_stored_as_given():
    """
    The token typed into the box is dead the moment it works. Storing it
    instead of the replacement would break the connection on the next call.
    """
    s = FakeSession()
    c, store = client(s)
    c.connect("manual-token-from-apphub")

    assert store.record["refresh_token"] == "ref1"
    assert store.record["refresh_token"] != "manual-token-from-apphub"
    assert s.token_calls[0][2]["refresh_token"] == "manual-token-from-apphub"


def test_the_replacement_is_written_before_anything_can_use_the_connection():
    """
    Rule one. If the process dies between the exchange and the write, the
    connection is gone for good — so the write has to come first.
    """
    s = FakeSession()
    c, store = client(s)
    c.connect("manual")

    # The store was written during connect(), before any API call existed.
    assert len(store.writes) >= 2                     # the pasted one, then the real one
    assert store.writes[-1]["refresh_token"] == "ref1"
    assert store.writes[-1]["access_token"] == "acc1"


def test_an_empty_token_is_refused_before_any_request():
    s = FakeSession()
    c, _ = client(s)
    with pytest.raises(qt.NeedsReconnect):
        c.connect("   ")
    assert s.calls == []


def test_a_rejected_token_says_what_to_do_about_it():
    s = FakeSession(token_responses=[Response(400, {"error": "invalid_grant"})])
    c, store = client(s)
    with pytest.raises(qt.NeedsReconnect, match="apphub"):
        c.connect("stale")
    assert store.record["last_error"]


# ── reusing and refreshing ───────────────────────────────────────────────────
def test_a_live_access_token_is_reused_rather_than_refreshed():
    s = FakeSession(api_responses={"v1/accounts": Response(200, {"accounts": []})})
    clock = Clock()
    c, _ = client(s, clock)
    c.connect("manual")

    before = len(s.token_calls)
    c.accounts()
    c.accounts()
    assert len(s.token_calls) == before, "refreshed while the token was still good"


def test_an_expiring_token_is_refreshed_exactly_once():
    s = FakeSession(api_responses={"v1/accounts": Response(200, {"accounts": []})})
    clock = Clock()
    c, store = client(s, clock)
    c.connect("manual")

    clock.t += 1800                       # past expiry
    c.accounts()
    assert len(s.token_calls) == 2
    assert store.record["refresh_token"] == "ref2"


def test_the_refresh_happens_before_the_token_actually_expires():
    """A token that dies mid-request is a failed request, not a late refresh."""
    s = FakeSession(api_responses={"v1/accounts": Response(200, {"accounts": []})})
    clock = Clock()
    c, _ = client(s, clock)
    c.connect("manual")

    clock.t += 1800 - qt.EXPIRY_SLACK_S + 1
    c.accounts()
    assert len(s.token_calls) == 2


def test_an_exchange_is_never_retried():
    """
    A timeout is ambiguous: Questrade may have rotated the token and lost the
    reply. Retrying turns "probably fine" into "certainly broken".
    """
    s = FakeSession(token_responses=[RuntimeError("connection reset")])
    c, store = client(s)
    c.store.save(qt.Tokens(refresh="live").as_record())

    with pytest.raises(Exception):
        c.accounts()
    assert len(s.token_calls) == 1
    assert "未完成" in store.record["last_error"]
    # The refresh token we still hold is NOT cleared — it may well be fine.
    assert store.record["refresh_token"] == "live"


def test_concurrent_callers_exchange_the_token_only_once():
    """
    Two threads refreshing the same token means one of them is handed a token
    that is already dead — and both may write.
    """
    s = FakeSession(api_responses={"v1/accounts": Response(200, {"accounts": []})},
                    token_latency_s=0.05)
    clock = Clock()
    store = qt.MemoryStore(key="concurrent-test")
    c = qt.Questrade(store, session=s, now=clock)
    c.connect("manual")
    clock.t += 1800

    errors = []

    def run():
        try:
            c.accounts()
        except Exception as exc:                                   # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(s.token_calls) == 2, "one connect + one shared refresh"


# ── requests ─────────────────────────────────────────────────────────────────
def test_the_api_server_from_the_response_is_the_one_used():
    """It is not always api01, and hardcoding it works right up until it doesn't."""
    s = FakeSession(api_responses={"v1/accounts": Response(200, {"accounts": []})})
    c, _ = client(s)
    c.connect("manual")
    c.accounts()

    url = [u for _, u, _ in s.calls if "v1/accounts" in u][0]
    assert url == "https://api07.iq.questrade.com/v1/accounts"


def test_a_token_rejected_mid_session_is_refreshed_once_and_the_call_retried():
    seen = {"n": 0}

    def accounts():
        seen["n"] += 1
        return Response(401, {}) if seen["n"] == 1 else Response(200, {"accounts": []})

    s = FakeSession(api_responses={"v1/accounts": accounts})
    c, _ = client(s)
    c.connect("manual")
    c.accounts()

    assert seen["n"] == 2
    assert len(s.token_calls) == 2


def test_a_second_rejection_gives_up_instead_of_looping():
    s = FakeSession(api_responses={"v1/accounts": Response(401, {})})
    c, _ = client(s)
    c.connect("manual")
    with pytest.raises(qt.QuestradeError):
        c.accounts()
    # connect + exactly one recovery attempt.
    assert len(s.token_calls) == 2


def test_the_token_is_posted_not_put_in_the_url():
    """A refresh token in a query string ends up in every proxy log there is."""
    s = FakeSession()
    c, _ = client(s)
    c.connect("manual")
    assert s.token_calls[0][0] == "POST"


def test_a_routing_rejection_falls_back_to_the_documented_get_form():
    """
    Questrade documents GET. A 405 means the method was wrong; a 400 means the
    TOKEN was wrong and must never be retried by another route.
    """
    s = FakeSession(token_responses=[Response(405, {})])
    c, _ = client(s)
    c.connect("manual")
    assert [m for m, _, _ in s.token_calls] == ["POST", "GET"]


def test_a_bad_token_is_not_retried_through_the_other_verb():
    s = FakeSession(token_responses=[Response(400, {"error": "invalid_grant"})])
    c, _ = client(s)
    with pytest.raises(qt.NeedsReconnect):
        c.connect("stale")
    assert len(s.token_calls) == 1


def test_calling_before_connecting_says_so():
    c, _ = client()
    with pytest.raises(qt.NeedsReconnect, match="尚未连接"):
        c.accounts()


# ── the endpoints ────────────────────────────────────────────────────────────
def test_accounts_are_labelled_by_their_tax_treatment():
    s = FakeSession(api_responses={"v1/accounts": Response(200, {"accounts": [
        {"number": "26598145", "type": "TFSA", "status": "Active", "isPrimary": True},
        {"number": "26598146", "type": "RRSP", "status": "Active"},
        {"number": "26598147", "type": "Margin", "status": "Active"},
        {"number": None, "type": "Margin"},
    ]})})
    c, _ = client(s)
    c.connect("manual")
    got = c.accounts()

    assert [a["id"] for a in got] == ["26598145", "26598146", "26598147"]
    assert "TFSA" in got[0]["label"] and "RRSP" in got[1]["label"]
    assert got[0]["primary"] is True


def test_symbols_are_requested_in_batches_of_a_hundred():
    calls = {"n": 0}

    def symbols():
        calls["n"] += 1
        return Response(200, {"symbols": [{"symbolId": 1}]})

    s = FakeSession(api_responses={"v1/symbols": symbols})
    c, _ = client(s)
    c.connect("manual")
    c.symbols(list(range(1, 251)))
    assert calls["n"] == 3


def test_candle_times_carry_an_offset():
    """Questrade rejects a bare 'Z'."""
    s = FakeSession(api_responses={"v1/markets/candles": Response(200, {"candles": []})})
    c, _ = client(s)
    c.connect("manual")
    c.candles(123, days=30)

    params = [p for _, u, p in s.calls if "candles" in u][0]
    assert params["startTime"].endswith("+00:00")
    assert params["endTime"].endswith("+00:00")


# ── what leaves the building ─────────────────────────────────────────────────
def test_status_never_exposes_the_token():
    s = FakeSession()
    c, _ = client(s)
    c.connect("manual")
    blob = repr(c.status())

    assert "ref1" not in blob and "acc1" not in blob and "manual" not in blob
    assert c.status()["connected"] is True


def test_status_of_a_fresh_account_is_simply_not_connected():
    c, _ = client()
    assert c.status() == {"connected": False, "reason": None}


def test_disconnecting_removes_the_token():
    s = FakeSession()
    c, store = client(s)
    c.connect("manual")
    c.disconnect()
    assert store.record is None
    assert c.status()["connected"] is False
