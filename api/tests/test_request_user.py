"""
auth_manager.request_user: the API hook must not change Streamlit, and two
concurrent API requests must never see each other's user.

    python -m pytest api/tests/test_request_user.py -q
"""

from __future__ import annotations

import os
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


# supabase-py parses the URL at client construction, so the placeholder has to
# look like a URL. Nothing is ever sent to it — these tests are offline.
DUMMY = {"SUPABASE_URL": "https://offline.supabase.co", "SUPABASE_KEY": "test",
         "TUSHARE_TOKEN": "test", "DEEPSEEK_API_KEY": "test"}


def _am(monkeypatch):
    for k, v in DUMMY.items():
        monkeypatch.setenv(k, os.environ.get(k) or v)
    import auth_manager
    return auth_manager


def test_streamlit_path_unchanged_when_hook_unset(monkeypatch):
    am = _am(monkeypatch)
    fake_session = {"current_user": {"id": 7, "role": "user"}}
    monkeypatch.setattr(am.st, "session_state", fake_session, raising=False)
    assert am.get_current_user_id() == 7          # falls through to session_state


def test_hook_wins_inside_block_and_resets_after(monkeypatch):
    am = _am(monkeypatch)
    monkeypatch.setattr(am.st, "session_state", {"current_user": {"id": 7}}, raising=False)
    with am.request_user({"id": 42, "role": "admin"}):
        assert am.get_current_user_id() == 42
        assert am.is_admin()
    assert am.get_current_user_id() == 7          # restored, not left behind


def test_reset_even_when_the_block_raises(monkeypatch):
    am = _am(monkeypatch)
    monkeypatch.setattr(am.st, "session_state", {}, raising=False)
    try:
        with am.request_user({"id": 1}):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert am.get_current_user_id() is None


def test_concurrent_requests_are_isolated(monkeypatch):
    am = _am(monkeypatch)
    monkeypatch.setattr(am.st, "session_state", {}, raising=False)
    seen: dict[int, list[int]] = {}
    barrier = threading.Barrier(8)

    def worker(uid: int):
        with am.request_user({"id": uid}):
            barrier.wait()                 # all eight inside their blocks at once
            ids = []
            for _ in range(50):
                ids.append(am.get_current_user_id())
                time.sleep(0.0005)
            seen[uid] = ids

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for uid, ids in seen.items():
        assert set(ids) == {uid}, f"request {uid} saw {set(ids)}"
