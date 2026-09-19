"""
The keep-alive, and the guard on the unhandled-error handler.

Questrade's refresh token lives three days and is single-use; each exchange
mints the next. A connection refreshed only when somebody opens the page dies
over a long weekend, and from the outside that is indistinguishable from a
bug. These tests pin the two properties that make the scheduled refresh safe:
it is served BY the API (so the existing per-user lock applies), and one
broken connection does not stop the others.

    python -m pytest api/tests/test_questrade_keepalive.py -q
"""

from __future__ import annotations

import os
import sys
import types

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from api import main as api_main  # noqa: E402
from api import questrade_api as qa  # noqa: E402
import questrade as qt  # noqa: E402


class Fake:
    def __init__(self, uid, fail=None):
        self.uid, self.fail = uid, fail
        self.exchanges = 0

    def _tokens(self):
        self.exchanges += 1
        if self.fail:
            raise self.fail
        return qt.Tokens(refresh="new")


def wire(monkeypatch, users, failures=None):
    """A questrade_tokens table with `users` in it, and a client per user."""
    failures = failures or {}
    made = {}

    db = types.SimpleNamespace(
        read_table=lambda t, columns=None, **kw: pd.DataFrame({"app_user_id": users}))
    monkeypatch.setitem(sys.modules, "db_manager", types.SimpleNamespace(db=db))
    monkeypatch.setattr(qa, "client",
                        lambda uid: made.setdefault(uid, Fake(uid, failures.get(uid))))
    return made


def test_every_connected_user_is_refreshed(monkeypatch):
    made = wire(monkeypatch, [1, 2, 3])
    out = qa.keepalive()

    assert out == {"checked": 3, "refreshed": 3,
                   "results": [{"user": u, "ok": True} for u in (1, 2, 3)]}
    assert all(c.exchanges == 1 for c in made.values())


def test_one_broken_connection_does_not_stop_the_others(monkeypatch):
    """
    A dead chain is that user's to reconnect. Letting it abort the run would
    let one stale row take every other connection down with it.
    """
    made = wire(monkeypatch, [1, 2, 3],
                failures={2: qt.NeedsReconnect("token rejected")})
    out = qa.keepalive()

    assert out["checked"] == 3 and out["refreshed"] == 2
    assert made[1].exchanges == 1 and made[3].exchanges == 1
    bad = [r for r in out["results"] if not r["ok"]]
    assert len(bad) == 1 and bad[0]["user"] == 2 and "NeedsReconnect" in bad[0]["error"]


def test_each_user_is_exchanged_exactly_once(monkeypatch):
    """The token is single-use — refreshing a user twice per run wastes one."""
    made = wire(monkeypatch, [1, 1, 2, 2, 2])       # duplicate rows
    qa.keepalive()
    assert [c.exchanges for c in made.values()] == [1, 1]


def test_nobody_connected_is_not_an_error(monkeypatch):
    db = types.SimpleNamespace(read_table=lambda t, columns=None, **kw: pd.DataFrame())
    monkeypatch.setitem(sys.modules, "db_manager", types.SimpleNamespace(db=db))
    assert qa.keepalive() == {"checked": 0, "refreshed": 0, "results": []}


def test_a_missing_table_says_which_migration_to_run(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("PGRST205 questrade_tokens does not exist")

    monkeypatch.setitem(sys.modules, "db_manager",
                        types.SimpleNamespace(db=types.SimpleNamespace(read_table=boom)))
    with pytest.raises(qt.QuestradeError, match="20260918_questrade.sql"):
        qa.keepalive()


# ── the endpoint's guard ─────────────────────────────────────────────────────
def test_an_unconfigured_secret_disables_the_route(monkeypatch):
    """
    An unset secret must not mean an open endpoint.

    Checked on the MESSAGE as well as the status: reaching keepalive() with no
    database configured also fails, and also as a 503, so the status alone
    cannot tell "refused" apart from "tried and broke".
    """
    monkeypatch.delenv("QUESTRADE_KEEPALIVE_SECRET", raising=False)
    monkeypatch.setattr(qa, "keepalive",
                        lambda: pytest.fail("route ran with no secret configured"))
    with pytest.raises(Exception) as got:
        api_main.questrade_keepalive(x_keepalive_secret="anything")

    assert got.value.status_code == 503
    assert "QUESTRADE_KEEPALIVE_SECRET" in got.value.detail


def test_a_wrong_secret_is_refused(monkeypatch):
    monkeypatch.setenv("QUESTRADE_KEEPALIVE_SECRET", "right")
    with pytest.raises(Exception) as got:
        api_main.questrade_keepalive(x_keepalive_secret="wrong")
    assert got.value.status_code == 403


def test_an_empty_header_never_matches(monkeypatch):
    monkeypatch.setenv("QUESTRADE_KEEPALIVE_SECRET", "right")
    with pytest.raises(Exception) as got:
        api_main.questrade_keepalive(x_keepalive_secret="")
    assert got.value.status_code == 403


# ── the unhandled-error handler ──────────────────────────────────────────────
@pytest.mark.parametrize("text, must_not_contain", [
    ("token=abcdef123456 failed", "abcdef123456"),
    ('refresh_token: "S0meL0ngSecretValue"', "S0meL0ngSecretValue"),
    ("Authorization: Bearer eyJhbGciOiJIUzI1", "eyJhbGciOiJIUzI1"),
])
def test_a_500_never_echoes_something_credential_shaped(text, must_not_contain):
    """
    The handler exists to make a bare 500 diagnosable. It must not make it
    diagnosable to somebody reading a screenshot.
    """
    assert must_not_contain not in api_main._SECRETISH.sub(r"\1=<redacted>", text)


def test_the_scrubber_leaves_ordinary_messages_alone():
    msg = "KeyError: 'currentMarketValue' while reading positions for A1"
    assert api_main._SECRETISH.sub(r"\1=<redacted>", msg) == msg
