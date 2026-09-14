"""
api/main.py — FastAPI service behind the new Technical Analysis frontend.

Runs the SAME Python the Streamlit app runs: analysis_engine, data_manager,
box_detection, ta_payload. It is imported from the repository root rather than
copied, so the two apps cannot drift. Deployed to Cloud Run from api/Dockerfile.

Every route except /health requires a Supabase access token. Per-user data
(search history today; watchlists and portfolios later) goes through the
existing data_manager functions inside `auth_manager.request_user(...)`, which
is what makes those functions — written for a Streamlit session — work here.

Local run, from the repository root:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
import warnings
from collections import OrderedDict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import Depends, FastAPI, HTTPException, Path  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402

from api.auth import AppUser, current_user  # noqa: E402

TICKER = Path(..., pattern=r"^\d{6}$", description="6-digit A-share code")

app = FastAPI(title="ASSRS API", version="0.1.0",
              docs_url="/docs", openapi_url="/openapi.json")

# The payload is ~25 series × ~470 bars of floats; gzip takes it from roughly
# 300 KB to under 80 KB on the wire.
app.add_middleware(GZipMiddleware, minimum_size=1024)

_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGIN", "").split(",") if o.strip()]
_origins += ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    # Cloudflare Pages preview deployments get a fresh subdomain per build;
    # allow those for this project only, never any *.pages.dev.
    allow_origin_regex=os.environ.get("ALLOWED_ORIGIN_REGEX") or None,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── small TTL caches ─────────────────────────────────────────────────────────
class TTLCache:
    """In-process LRU with expiry. One Cloud Run instance, so memory is enough."""

    def __init__(self, maxsize: int, ttl_s: float):
        self.maxsize, self.ttl = maxsize, ttl_s
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._inflight: dict = {}

    def get_or_compute(self, key, fn):
        with self._lock:
            hit = self._d.get(key)
            if hit and time.time() - hit[0] < self.ttl:
                self._d.move_to_end(key)
                return hit[1]
            ev = self._inflight.get(key)
            if ev is None:
                ev = self._inflight[key] = threading.Event()
                owner = True
            else:
                owner = False
        if not owner:
            # Someone else is already computing this key — wait for them
            # rather than running the ~7s analysis twice in parallel.
            ev.wait(timeout=300)
            with self._lock:
                hit = self._d.get(key)
            if hit:
                return hit[1]
        try:
            val = fn()
            with self._lock:
                self._d[key] = (time.time(), val)
                self._d.move_to_end(key)
                while len(self._d) > self.maxsize:
                    self._d.popitem(last=False)
            return val
        finally:
            if owner:
                with self._lock:
                    self._inflight.pop(key, None)
                ev.set()


_analysis_cache = TTLCache(maxsize=40, ttl_s=20 * 60)
_stocks_cache = TTLCache(maxsize=1, ttl_s=6 * 3600)


def _as_user(user: AppUser):
    import auth_manager
    return auth_manager.request_user(user.as_session_user())


# ── routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/me")
def me(user: AppUser = Depends(current_user)):
    return {"app_user_id": user.id, "username": user.username,
            "email": user.email, "role": user.role}


@app.get("/stocks")
def stocks(user: AppUser = Depends(current_user)):
    """
    Every A-share as [{t, n}]. Sent whole (≈5,600 rows, ~40 KB gzipped) so the
    combobox filters in the browser as you type, with no request per keystroke.
    """
    import data_manager

    def load():
        return [{"t": s["ticker"], "n": s["name"]} for s in data_manager.get_all_stock_basic()]

    return _stocks_cache.get_or_compute("all", load)


@app.get("/history")
def history(user: AppUser = Depends(current_user)):
    import data_manager
    with _as_user(user):
        rows = data_manager.get_search_history(limit=15)
    return [{"t": r["ticker"], "n": r.get("name") or r["ticker"],
             "at": r.get("timestamp")} for r in rows]


@app.post("/history/{ticker}")
def add_history(ticker: str = TICKER, user: AppUser = Depends(current_user)):
    import data_manager
    with _as_user(user):
        name = data_manager.update_search_history(ticker)
    return {"t": ticker, "n": name or ticker}


@app.get("/analysis/{ticker}")
def analysis(ticker: str = TICKER, user: AppUser = Depends(current_user)):
    """
    Everything the Technical Analysis page draws for one stock.

    Cached per ticker for 20 minutes: the walk-forward HMM makes a cold
    analysis ~7s, and the page is often reopened on the same stock.
    """
    import ta_payload
    try:
        return _analysis_cache.get_or_compute(ticker, lambda: ta_payload.build_payload(ticker))
    except LookupError as exc:
        raise HTTPException(404, str(exc))
