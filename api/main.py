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

from fastapi import Depends, FastAPI, HTTPException, Path, Query  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402

import pair_compare  # noqa: E402
import sector_affinity  # noqa: E402
import ta_payload  # noqa: E402
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
# The sector indices are rebuilt nightly and shared by every stock, so one
# 25-table read serves every ticker and every window until tomorrow.
_sector_ret_cache = TTLCache(maxsize=1, ttl_s=6 * 3600)
_sector_cache = TTLCache(maxsize=60, ttl_s=20 * 60)


def _as_user(user: AppUser):
    import auth_manager
    return auth_manager.request_user(user.as_session_user())


_frames_cache = TTLCache(maxsize=20, ttl_s=20 * 60)


def _frames(ticker: str):
    """
    (analysis_df, fundamentals_df) for a ticker, cached.

    The What-If simulator, the AI read and the comparison overlay all need the
    analysed frame, and re-deriving it costs a 3-year fetch plus a walk-forward
    HMM (~20s). Holding it for 20 minutes is what makes those features feel
    interactive instead of each one restarting the analysis.
    """
    def load():
        import data_manager
        import watchlist_scan
        from analysis_engine import run_single_stock_analysis
        df = watchlist_scan.fetch_frame(ticker)
        if df is None:
            raise RuntimeError(f"price data for {ticker} could not be fetched — try again")
        if len(df) < 60:
            raise LookupError(f"not enough price history for {ticker}")
        adf = run_single_stock_analysis(df)
        fund = data_manager.get_stock_fundamentals_live(
            ticker, df.index.min().strftime("%Y%m%d"), df.index.max().strftime("%Y%m%d"))
        return adf, fund
    return _frames_cache.get_or_compute(ticker, load)


_pricefund_cache = TTLCache(maxsize=40, ttl_s=20 * 60)
_bench_cache = TTLCache(maxsize=1, ttl_s=6 * 3600)
_pair_cache = TTLCache(maxsize=80, ttl_s=20 * 60)


def _price_fund(ticker: str):
    """
    (close, fundamentals) for a stock, WITHOUT running the analysis.

    The comparison stock needs prices and PE history, not indicators, and the
    walk-forward HMM in the full path costs ~20s. Comparing against a stock
    must not be slower than analysing one.
    """
    def load():
        import data_manager
        import watchlist_scan
        df = watchlist_scan.fetch_frame(ticker)
        if df is None:
            raise RuntimeError(f"price data for {ticker} could not be fetched — try again")
        if len(df) < 60:
            raise LookupError(f"not enough price history for {ticker}")
        fund = data_manager.get_stock_fundamentals_live(
            ticker, df.index.min().strftime("%Y%m%d"), df.index.max().strftime("%Y%m%d"))
        return df["Close"], fund
    return _pricefund_cache.get_or_compute(ticker, load)


def _benchmark():
    """沪深300 closes — shared by every comparison, so fetched once."""
    def load():
        import data_manager
        idx = data_manager.get_index_data_live("000300.SH", lookback_days=1200)
        if idx is None or idx.empty:
            raise RuntimeError("沪深300 指数数据暂时读取不到 — 请稍后重试")
        return idx["Close"]
    return _bench_cache.get_or_compute("csi300", load)


class SimReq(BaseModel):
    """Tomorrow's hypothetical bar. Omit o/h/l and the engine estimates them."""
    pct: float = Field(0.0, ge=-30, le=30, description="close change in %")
    volume: float = Field(..., gt=0)
    open: float | None = Field(None, gt=0)
    high: float | None = Field(None, gt=0)
    low: float | None = Field(None, gt=0)


class AiReq(SimReq):
    mode: str = Field("ghost", pattern="^(ghost|actual)$")
    volume: float | None = Field(None, gt=0)
    window: int = Field(20, ge=10, le=60)


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
        rows = [{"t": s["ticker"], "n": s["name"]} for s in data_manager.get_all_stock_basic()]
        if not rows:
            # get_all_stock_basic() swallows its errors and returns [], so a
            # momentary database hiccup looks identical to "no stocks exist".
            # Caching that for six hours leaves every search box silently
            # empty with nothing to show why — which is exactly what happened.
            # Raising keeps it OUT of the cache and tells the client to retry.
            raise RuntimeError("stock list unavailable — try again")
        return rows

    try:
        return _stocks_cache.get_or_compute("all", load)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


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
        adf, fund = _frames(ticker)
        return _analysis_cache.get_or_compute(
            ticker, lambda: ta_payload.build_payload(ticker, adf, fund))
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        # Upstream data unavailable after retries — a 503 says "try again",
        # which is true, where a 500 would read as a bug in this service.
        raise HTTPException(503, str(exc))


@app.post("/simulate/{ticker}")
def simulate(req: SimReq, ticker: str = TICKER, user: AppUser = Depends(current_user)):
    """
    What every indicator would read if tomorrow printed this bar.

    Cheap (no fetch, no HMM — the analysed frame is cached), so the frontend
    can call it on every edit and redraw the ghost immediately.
    """
    from api import extras
    adf, _ = _frames(ticker)
    try:
        return extras.simulate(adf, pct=req.pct, volume=req.volume,
                               open_=req.open, high=req.high, low=req.low)
    except LookupError as exc:
        raise HTTPException(404, str(exc))


@app.post("/whatif-ai/{ticker}")
def whatif_ai(req: AiReq, ticker: str = TICKER, user: AppUser = Depends(current_user)):
    """
    尾盘推演: the AI read of the ghost bar, or of the last real session when
    no ghost is drawn. One DeepSeek call, 20-60s.
    """
    import data_manager
    from api import extras
    adf, _ = _frames(ticker)
    name = data_manager.get_stock_name_from_db(ticker) or ticker
    try:
        return extras.whatif_ai(adf, ticker, name, mode=req.mode, pct=req.pct,
                                volume=req.volume, open_=req.open, high=req.high,
                                low=req.low, window=req.window)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, f"AI call failed: {exc}")


@app.get("/compare-stats/{ticker}")
def compare_stats(ticker: str = TICKER,
                  with_: str = Query(..., alias="with", pattern=r"^\d{6}$"),
                  window: str = Query(pair_compare.DEFAULT_WINDOW),
                  user: AppUser = Depends(current_user)):
    """
    量化对比: return, risk, beta/alpha, valuation and the gap attribution.

    Separate from /compare so that changing the statistics window does not
    refetch and redraw the chart overlay, and vice versa.
    """
    if window not in pair_compare.WINDOWS:
        raise HTTPException(422, f"window must be one of {sorted(pair_compare.WINDOWS)}")
    if with_ == ticker:
        raise HTTPException(422, "cannot compare a stock with itself")

    import data_manager
    try:
        adf, a_fund = _frames(ticker)
        b_close, b_fund = _price_fund(with_)
        try:
            bench = _benchmark()
        except RuntimeError:
            # Beta, alpha and capture need the market; everything else does
            # not. Losing the index should cost those fields, not the panel.
            bench = None
        return _pair_cache.get_or_compute(
            (ticker, with_, window),
            lambda: pair_compare.compare(
                adf["Close"], b_close, bench_close=bench,
                a_fund=a_fund, b_fund=b_fund,
                a_label=data_manager.get_stock_name_from_db(ticker) or ticker,
                b_label=data_manager.get_stock_name_from_db(with_) or with_,
                window=window))
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/sectors/{ticker}")
def sectors(ticker: str = TICKER,
            window: int = Query(sector_affinity.DEFAULT_WINDOW,
                                description="rolling window in trading days"),
            user: AppUser = Depends(current_user)):
    """
    板块相关性 + 板块轮动 for one stock at one window length.

    The sector returns are loaded once for the whole service; only the
    correlation is per stock, so changing the window is fast.
    """
    if window not in sector_affinity.WINDOWS:
        raise HTTPException(422, f"window must be one of {list(sector_affinity.WINDOWS)}")

    def load_rets():
        rets = sector_affinity.load_sector_returns()
        if not rets:
            # Same reasoning as /stocks: an empty read is a failure, and
            # caching it would keep the panel broken long after the fix.
            raise RuntimeError("板块指数（PPI_*）暂时读取不到 — 请稍后重试")
        return rets, sector_affinity.sector_members()

    try:
        adf, _ = _frames(ticker)
        rets, members = _sector_ret_cache.get_or_compute("all", load_rets)
        return _sector_cache.get_or_compute(
            (ticker, window),
            lambda: sector_affinity.analyse(adf["Close"], rets, window,
                                            ticker=ticker, members=members))
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/compare/{ticker}")
def compare(ticker: str = TICKER,
            with_: str = Query(..., alias="with", pattern=r"^\d{6}$"),
            user: AppUser = Depends(current_user)):
    """A second stock aligned to this one's bars, in both scalings."""
    from api import extras
    adf, _ = _frames(ticker)
    dates = [d.strftime("%Y-%m-%d") for d in ta_payload.chart_window(adf).index]
    try:
        return extras.compare(adf, with_, dates)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
