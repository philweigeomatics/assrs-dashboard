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

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402

import alerts_feed  # noqa: E402
import markets  # noqa: E402
import pair_compare  # noqa: E402
import sector_affinity  # noqa: E402
import ta_payload  # noqa: E402
from api.auth import AppUser, current_user  # noqa: E402

# A bare six-digit code is an A-share, for every ticker already stored; other
# markets carry a prefix (US:AAPL, CA:SHOP.TO). markets.split() is the one
# place that mapping lives.
#
# Validated as a DEPENDENCY rather than a path pattern so that every route
# receives the canonical form and none of them has to canonicalise again — and
# so a symbol that is shaped like a ticker but belongs to no market is refused
# here, before any fetch, cache lookup or HMM. "Not a symbol" is a malformed
# request (422), not a missing stock (404).
def _symbol(ticker: str = Path(..., pattern=markets.SYMBOL_RE.pattern,
                               description="600519, US:AAPL or CA:SHOP.TO")) -> str:
    try:
        return markets.canonical(ticker)
    except LookupError as exc:
        raise HTTPException(422, str(exc))


def _other(with_: str = Query(..., alias="with",
                              pattern=markets.SYMBOL_RE.pattern)) -> str:
    try:
        return markets.canonical(with_)
    except LookupError as exc:
        raise HTTPException(422, str(exc))


TICKER = Depends(_symbol)
OTHER = Depends(_other)

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
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


#: Anything that looks like a credential, scrubbed before an error message is
#: sent anywhere. The Questrade token is the one that matters: it is posted in
#: a form body, so it is not in any URL, but a library that echoes its input
#: into an exception would put it in a response without this.
#
#: The optional scheme word matters: "Authorization: Bearer eyJhbG…" otherwise
#: matches with "Bearer" as the value, redacts THAT, and leaves the token.
_SECRETISH = re.compile(
    r"""(?ix) (token|secret|key|password|authorization)   # the label
        ["'\s:=]+ (?: (?:bearer|basic|token)\s+ )?       # optional scheme
        ["']? ([A-Za-z0-9._\-]{6,})                       # the value""")


@app.exception_handler(Exception)
async def unhandled(request, exc: Exception):
    """
    Turn an unhandled exception into something diagnosable.

    Every route already maps its own failures onto a status with a message.
    Anything reaching here escaped that, and the default response is a bare
    500 with an empty body — which tells the user nothing, tells the client
    nothing to display, and leaves the only record in a log they have to go
    and find. Reporting the exception type and message costs nothing and is
    the difference between "it always returns 500" and a fix.

    The traceback still goes to stdout for Cloud Run; only the summary line
    goes to the client, with anything credential-shaped scrubbed out.
    """
    import traceback
    from fastapi.responses import JSONResponse

    traceback.print_exception(type(exc), exc, exc.__traceback__)
    detail = _SECRETISH.sub(r"\1=<redacted>", f"{type(exc).__name__}: {exc}")[:300]
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误 —— {detail}",
                 "path": str(request.url.path)})


# ── small TTL caches ─────────────────────────────────────────────────────────
class TTLCache:
    """In-process LRU with expiry. One Cloud Run instance, so memory is enough."""

    def __init__(self, maxsize: int, ttl_s: float):
        self.maxsize, self.ttl = maxsize, ttl_s
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._inflight: dict = {}

    def peek(self, key):
        """The value if it is present and unexpired, else None. Never computes."""
        with self._lock:
            hit = self._d.get(key)
            if hit and time.time() - hit[0] < self.ttl:
                self._d.move_to_end(key)
                return hit[1]
        return None

    def put(self, key, value) -> None:
        """Store a value computed elsewhere — a scan the caller already ran."""
        with self._lock:
            self._d[key] = (time.time(), value)
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

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
# The nightly snapshot does not change during the day, so this can be held
# far longer than the live-analysis caches. Keyed per user: the feed is
# that user's watchlist.
_alerts_cache = TTLCache(maxsize=20, ttl_s=30 * 60)


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
        from analysis_engine import run_single_stock_analysis
        market, code = markets.parse(ticker)
        df = market.fetch_ohlcv(code)
        if df is None:
            raise RuntimeError(f"price data for {ticker} could not be fetched — try again")
        if len(df) < 60:
            raise LookupError(f"not enough price history for {ticker}")
        adf = run_single_stock_analysis(df, regime_anchor=market.conv.regime_anchor)
        fund = market.fetch_fundamentals(
            code, df.index.min().strftime("%Y%m%d"), df.index.max().strftime("%Y%m%d"))
        return adf, fund
    return _frames_cache.get_or_compute(markets.canonical(ticker), load)


_pricefund_cache = TTLCache(maxsize=40, ttl_s=20 * 60)
_bench_cache = TTLCache(maxsize=4, ttl_s=6 * 3600)
_pair_cache = TTLCache(maxsize=80, ttl_s=20 * 60)


def _price_fund(ticker: str):
    """
    (close, fundamentals) for a stock, WITHOUT running the analysis.

    The comparison stock needs prices and PE history, not indicators, and the
    walk-forward HMM in the full path costs ~20s. Comparing against a stock
    must not be slower than analysing one.
    """
    def load():
        market, code = markets.parse(ticker)
        df = market.fetch_ohlcv(code)
        if df is None:
            raise RuntimeError(f"price data for {ticker} could not be fetched — try again")
        if len(df) < 60:
            raise LookupError(f"not enough price history for {ticker}")
        fund = market.fetch_fundamentals(
            code, df.index.min().strftime("%Y%m%d"), df.index.max().strftime("%Y%m%d"))
        return df["Close"], fund
    return _pricefund_cache.get_or_compute(markets.canonical(ticker), load)


def _name(symbol: str) -> str:
    market, code = markets.parse(symbol)
    ref = market.resolve(code)
    return (ref.name if ref else None) or symbol


def _benchmark(market_code: str = "CN"):
    """
    The market's own index, cached — one fetch serves every comparison.

    Beta against the wrong index is not a worse number, it is a meaningless
    one, so this is keyed by market rather than defaulting to 沪深300.
    """
    def load():
        mk = markets.get(market_code)
        series = mk.fetch_benchmark(years=4)
        if series is None or series.empty:
            raise RuntimeError(f"{mk.conv.benchmark_name} 指数数据暂时读取不到 — 请稍后重试")
        return series
    return _bench_cache.get_or_compute(market_code, load)


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


@app.get("/search")
def search(q: str = Query(..., min_length=1, max_length=40),
           market: str = Query("US", pattern="^(US|CA)$"),
           user: AppUser = Depends(current_user)):
    """
    Instrument search for the North American markets.

    A-shares are not served here: the frontend already holds all 5,600 of them
    from /stocks and filters locally, with no request per keystroke. There is
    no equivalent downloadable universe for US/CA listings, so those go to the
    source — which is why this route exists at all.
    """
    try:
        hits = markets.get(market).search(q.strip(), limit=8)
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(503, f"搜索暂时不可用：{exc}"[:200])
    return [{"t": f"{market}:{h.symbol}", "n": h.name, "ex": h.exchange} for h in hits]


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
    market, code = markets.parse(ticker)
    # Resolved here rather than inside data_manager: only the market knows how
    # to name a US ticker, and stock_basic never will.
    ref = market.resolve(code) if market.conv.code != "CN" else None
    with _as_user(user):
        name = data_manager.update_search_history(ticker, name=ref.name if ref else None)
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
    from api import extras
    adf, _ = _frames(ticker)
    market, code = markets.parse(ticker)
    ref = market.resolve(code)
    name = (ref.name if ref else None) or ticker
    try:
        return extras.whatif_ai(adf, ticker, name, mode=req.mode, pct=req.pct,
                                volume=req.volume, open_=req.open, high=req.high,
                                low=req.low, window=req.window)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, f"AI call failed: {exc}")


@app.get("/alerts")
def alerts(date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
           user: AppUser = Depends(current_user)):
    """
    今日提醒 — the nightly watchlist scan, reshaped into a filterable feed.

    Reads only. Everything here was computed by scan_watchlists.py in GitHub
    Actions at 20:00 Beijing; the whole point of that job is that this page
    never runs an HMM. Cheap enough to serve per user without a long cache.
    """
    import datetime as _dt
    import data_manager

    def load():
        with _as_user(user):
            day = date or data_manager.get_latest_signal_snapshot_date()
            if not day:
                raise LookupError("还没有扫描快照 — 夜间任务尚未运行过")
            rows = data_manager.get_cached_signals(day)
        if rows is None or rows.empty:
            raise LookupError(f"{day} 没有属于你的快照（自选股为空，或当天没有信号）")

        try:
            chips = data_manager.get_chip_scan(day, 1.0)
        except Exception:
            chips = None      # the feed is still worth showing without 筹码
        members = _sector_ret_cache.get_or_compute(
            "members", sector_affinity.sector_members)

        age = (_dt.date.today() - _dt.date.fromisoformat(str(day)[:10])).days
        return alerts_feed.build(rows, chips, members, str(day)[:10], age_days=age)

    try:
        return _alerts_cache.get_or_compute((user.id, date), load)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/compare-stats/{ticker}")
def compare_stats(ticker: str = TICKER,
                  with_: str = OTHER,
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

    market_code = markets.split(ticker)[0]
    if markets.split(with_)[0] != market_code:
        # Beta and alpha are measured against ONE index, and the two stocks
        # would be priced in different currencies. Neither is fixable without
        # an FX series, and a number produced anyway would look authoritative.
        raise HTTPException(422, "暂不支持跨市场对比 — 两只股票需在同一市场")

    try:
        adf, a_fund = _frames(ticker)
        b_close, b_fund = _price_fund(with_)
        try:
            bench = _benchmark(market_code)
        except RuntimeError:
            # Beta, alpha and capture need the market; everything else does
            # not. Losing the index should cost those fields, not the panel.
            bench = None
        return _pair_cache.get_or_compute(
            (ticker, with_, window),
            lambda: pair_compare.compare(
                adf["Close"], b_close, bench_close=bench,
                a_fund=a_fund, b_fund=b_fund,
                a_label=_name(ticker), b_label=_name(with_),
                benchmark_label=markets.get(market_code).conv.benchmark_name,
                market=market_code,
                window=window))
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


_basket_cache = TTLCache(maxsize=40, ttl_s=20 * 60)


@app.get("/basket")
def basket(symbols: str = Query(..., description="comma-separated, 2-8, one market"),
           window: str = Query(pair_compare.DEFAULT_WINDOW),
           user: AppUser = Depends(current_user)):
    """
    多股对比: rank a basket on return, risk, beta and alpha, and say whether
    "sell the one moving least" is actually supported for it.

    No chart — this is the question you ask about a group, where overlaying
    six lines answers nothing.
    """
    if window not in pair_compare.WINDOWS:
        raise HTTPException(422, f"window must be one of {sorted(pair_compare.WINDOWS)}")

    try:
        wanted = [markets.canonical(s) for s in symbols.split(",") if s.strip()]
    except LookupError as exc:
        raise HTTPException(422, str(exc))

    seen = list(dict.fromkeys(wanted))          # de-duplicated, order kept
    if not 2 <= len(seen) <= pair_compare.MAX_BASKET:
        raise HTTPException(422, f"需要 2–{pair_compare.MAX_BASKET} 只股票")

    codes = {markets.split(s)[0] for s in seen}
    if len(codes) > 1:
        # Same reason /compare-stats refuses it: one beta needs one index, and
        # two currencies cannot share a ranking.
        raise HTTPException(422, "暂不支持跨市场对比 — 所有股票需在同一市场")
    market_code = codes.pop()

    def load():
        closes, labels = {}, {}
        for sym in seen:
            closes[sym] = _price_fund(sym)[0]
            labels[sym] = _name(sym)
        try:
            bench = _benchmark(market_code)
        except RuntimeError:
            bench = None
        return pair_compare.basket(
            closes, bench_close=bench, labels=labels,
            benchmark_label=markets.get(market_code).conv.benchmark_name,
            market=market_code, window=window)

    try:
        return _basket_cache.get_or_compute((tuple(seen), window), load)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


_strategy_cache = TTLCache(maxsize=8, ttl_s=30 * 60)
# Cointegration over ten codes is 45 pairs of rolling OLS — seconds, not
# milliseconds, and the same basket gets re-examined as you read it.
_pairtrade_cache = TTLCache(maxsize=20, ttl_s=20 * 60)
# The data half is a handful of Tushare calls; the AI half is cached in
# the database and survives a restart, so this only has to stop the page
# re-fetching statements while you read it.
_equity_cache = TTLCache(maxsize=30, ttl_s=20 * 60)


@app.get("/strategies/{name}")
def strategy(name: str = Path(..., pattern="^(t-trading|mean-reversion)$"),
             user: AppUser = Depends(current_user)):
    """
    The last result for one watchlist screen. Never scans — see the POST.

    做T comes from the same t_trading_scans table the Streamlit page writes, so
    a scan run in either app is visible in both. 反转 has no such table and is
    held in memory, which is why it answers 404 until something has run.
    """
    from api import strategies_api as sa

    if name == "t-trading":
        with _as_user(user):
            saved = sa.t_trading_saved(user.id)
        if saved is None:
            raise HTTPException(404, "还没有扫描过 — 点击“开始扫描”")
        return saved

    hit = _strategy_cache.peek(("mean-reversion", user.id))
    if hit is None:
        raise HTTPException(404, "还没有扫描过 — 点击“开始扫描”")
    return hit


@app.post("/strategies/{name}/scan")
def strategy_scan(name: str = Path(..., pattern="^(t-trading|mean-reversion)$"),
                  user: AppUser = Depends(current_user)):
    """
    Run a screen across the whole watchlist. Slow on purpose — two or three
    Tushare calls per holding, serially, because bursts lose stocks and a
    screen that quietly omits one of your positions is worse than a slow one.
    """
    from api import strategies_api as sa

    with _as_user(user):
        import data_manager
        if not (data_manager.get_watchlist() or []):
            raise HTTPException(404, "自选股为空 — 先在自选股页面添加股票")
        out = (sa.t_trading_scan(user.id) if name == "t-trading"
               else sa.mean_reversion_scan(user.id))

    if name == "mean-reversion":
        _strategy_cache.put(("mean-reversion", user.id), out)
    return out


class PairReq(BaseModel):
    """2-10 A-share codes. No defaults anywhere — the caller picks every one."""
    symbols: list[str] = Field(..., min_length=2, max_length=10)
    z_window: int = Field(60, ge=20, le=120)
    ols_window: int = Field(252, ge=60, le=504)


@app.post("/strategies/pair-trade")
def pair_trade(req: PairReq, user: AppUser = Depends(current_user)):
    """
    配对交易: test every unique pair among the supplied codes.

    A-shares only. The engine's whole reading of the signal is buy-only
    because shorting A-shares is restricted, and the cointegration statistics
    are computed on a walk-forward spread whose hedge ratio never saw the day
    it is used on — neither of which transfers to another market without
    rethinking it, so mixing markets is refused rather than approximated.
    """
    import pandas as pd
    from strategies import pair_trade as pt

    try:
        codes = [markets.canonical(sym) for sym in req.symbols]
    except LookupError as exc:
        raise HTTPException(422, str(exc))
    codes = list(dict.fromkeys(codes))
    if any(markets.split(c)[0] != "CN" for c in codes):
        raise HTTPException(422, "配对交易目前仅支持 A 股")
    if len(codes) < 2:
        raise HTTPException(422, "至少需要两只不同的股票")

    def load():
        frames = {}
        for code in codes:
            df = markets.get("CN").fetch_ohlcv(code)
            if df is None or df.empty:
                raise RuntimeError(f"{code} 行情读取失败 — 请稍后重试")
            frames[code] = df["Close"].rename(code)
        prices = pd.concat(frames.values(), axis=1).dropna()

        results, skipped = pt.rank_pairs(prices, req.z_window, req.ols_window)
        out = []
        for r in results:
            signal, buy, reduce_ = pt.signal_for_pair(r)
            trades = pt.detect_trades(r["z_series"], r["dates"], prices,
                                      r["code_a"], r["code_b"])
            closed = [t for t in trades if not t["open"]]
            wins = [t for t in closed if (t["pnl_pct"] or 0) > 0]
            out.append({
                **{k: v for k, v in r.items()
                   if k not in ("dates", "spread", "z_series", "beta_series")},
                "name_a": _name(r["code_a"]), "name_b": _name(r["code_b"]),
                "signal": signal, "signal_cn": pt.SIGNAL_CN[signal],
                "buy": buy, "reduce": reduce_,
                "dates": [d.strftime("%Y-%m-%d") for d in r["dates"]],
                "spread": [round(float(v), 5) for v in r["spread"]],
                "z_series": [None if v != v else round(float(v), 3) for v in r["z_series"]],
                "trades": trades,
                "closed": len(closed),
                "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else None,
                "avg_pnl_pct": (round(sum(t["pnl_pct"] or 0 for t in closed) / len(closed), 2)
                                if closed else None),
            })
        return {
            "from": str(prices.index.min().date()), "to": str(prices.index.max().date()),
            "bars": len(prices), "z_window": req.z_window, "ols_window": req.ols_window,
            "codes": [{"t": c, "n": _name(c)} for c in codes],
            "pairs": out, "skipped": skipped,
        }

    try:
        return _pairtrade_cache.get_or_compute(
            (tuple(codes), req.z_window, req.ols_window), load)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


#: Which markets each watchlist tab covers. One table holds them all — the
#: canonical symbol already says which market a row belongs to — so this is
#: only about what a request asked to see.
WATCHLIST_MARKETS = {"CN": ("CN",), "NA": ("US", "CA")}


@app.get("/watchlist")
def watchlist(market: str | None = Query(None, pattern="^(CN|NA)$"),
              user: AppUser = Depends(current_user)):
    """
    The user's watchlist, optionally for one market group.

    No `market` returns everything, which is what the nightly jobs and any
    older client expect. The page asks for one group at a time, because
    A-shares and North America are two different lists to a person even though
    they are one table to the database.
    """
    import data_manager
    want = WATCHLIST_MARKETS.get(market or "")
    with _as_user(user):
        rows = data_manager.get_watchlist() or []

    out = []
    for r in rows:
        symbol = str(r.get("ticker") or "")
        try:
            code = markets.split(symbol)[0]
        except LookupError:
            # A row nothing can parse still belongs to the user; show it under
            # its own market group so it can be deleted rather than being
            # invisible and unfixable.
            code = "??"
        if want and code not in want:
            continue
        out.append({"t": symbol, "n": str(r.get("stock_name") or symbol),
                    "market": code, "at": r.get("added_date")})
    return out


@app.post("/watchlist/{ticker}")
def watchlist_add(ticker: str = TICKER, user: AppUser = Depends(current_user)):
    """
    Add a symbol from any supported market.

    A-shares still feed the nightly Tushare scan; US and Canadian symbols are
    picked up by the separate North American run, which is why scan_watchlists
    takes a --market and each job only takes its own.
    """
    import data_manager
    with _as_user(user):
        ok, msg = data_manager.add_to_watchlist(ticker)
    if not ok:
        raise HTTPException(400, str(msg))
    return {"t": ticker, "message": str(msg)}


@app.delete("/watchlist/{ticker}")
def watchlist_remove(ticker: str = TICKER, user: AppUser = Depends(current_user)):
    import data_manager
    with _as_user(user):
        result = data_manager.remove_from_watchlist(ticker)
    ok, msg = result if isinstance(result, tuple) else (bool(result), "")
    if not ok:
        raise HTTPException(400, str(msg) or "移除失败")
    return {"t": ticker, "message": str(msg)}


def _require_admin(user: AppUser) -> None:
    """
    Admin gate, enforced on the SERVER.

    The frontend also hides these controls, but that is a convenience — hiding
    a button does not stop a request. Peer sets and sector membership feed the
    nightly job and every user's screens, so they are checked here.
    """
    if (user.role or "").lower() != "admin":
        raise HTTPException(403, "仅管理员可执行此操作")


def _industry_of(ticker: str) -> str:
    import data_manager
    try:
        ts = data_manager.get_tushare_ticker(ticker)
        df = data_manager.db.read_table("stock_basic", filters={"ts_code": ts},
                                        columns="industry", limit=1)
        if df is not None and not df.empty:
            return str(df.iloc[0]["industry"] or "")
    except Exception:
        pass
    return ""


class NoteReq(BaseModel):
    """A call on the next session, made before it happens."""
    ticker: str = Field(..., pattern=r"^\d{6}$")
    scan_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    note: str = Field("", max_length=2000)
    predictions: list[dict] = Field(default_factory=list, max_length=8)


@app.get("/alerts/notes")
def notes_list(ticker: str | None = Query(None, pattern=r"^\d{6}$"),
               user: AppUser = Depends(current_user)):
    """Every note, newest session first, with a scorecard over the resolved ones."""
    import alert_notes

    with _as_user(user):
        notes = alert_notes.list_notes(user.id, ticker)
    return {"notes": notes, "scorecard": alert_notes.scorecard(notes)}


@app.post("/alerts/notes")
def notes_create(req: NoteReq, user: AppUser = Depends(current_user)):
    """
    Write a call. The claims are validated NOW, not at resolution time — a
    prediction nothing could ever settle is refused while you can still fix it.
    """
    import alert_notes

    try:
        with _as_user(user):
            return alert_notes.create(user.id, req.ticker, req.scan_date,
                                      req.note, req.predictions)
    except LookupError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:                                    # noqa: BLE001
        # Supabase is migrated by hand in this project, so a missing table is
        # a setup step rather than a bug — say which step.
        if "alert_notes" in str(exc):
            raise HTTPException(
                503, "alert_notes 表尚未创建 — 请在 Supabase SQL 编辑器运行 "
                     "supabase/migrations/20260917_alert_notes.sql")
        raise HTTPException(503, f"保存失败：{exc}"[:200])


@app.post("/alerts/notes/resolve")
def notes_resolve(user: AppUser = Depends(current_user)):
    """
    Mark every pending note that now has a later session.

    Deliberately idempotent and safe to call on page load: a note with no newer
    bar is left alone, so a weekend simply resolves nothing.
    """
    import alert_notes

    with _as_user(user):
        notes = alert_notes.list_notes(user.id)
        pending = [n for n in notes if not n.get("outcome")]

        resolved = 0
        frames: dict = {}
        for n in pending:
            t = n["ticker"]
            if t not in frames:
                frames[t] = markets.get("CN").fetch_ohlcv(t)
            out = alert_notes.resolve_note(n, frames[t])
            if out is None:
                continue
            alert_notes.save_resolution(n["id"], out)
            resolved += 1

        notes = alert_notes.list_notes(user.id)

    return {"resolved": resolved, "pending": len(pending) - resolved,
            "notes": notes, "scorecard": alert_notes.scorecard(notes)}


@app.delete("/alerts/notes/{note_id}")
def notes_delete(note_id: int, user: AppUser = Depends(current_user)):
    import alert_notes
    with _as_user(user):
        alert_notes.delete(user.id, note_id)
    return {"deleted": note_id}


@app.get("/equity/{ticker}")
def equity_brief(ticker: str = TICKER, user: AppUser = Depends(current_user)):
    """
    The Equity Brief for one stock: the data half immediately, plus whichever
    AI sections are already cached. Never generates — see the POST.
    """
    from api import equity_api

    if markets.split(ticker)[0] != "CN":
        raise HTTPException(404, "个股研报目前仅支持 A 股")
    try:
        return _equity_cache.get_or_compute(
            ticker,
            lambda: equity_api.build(ticker, _name(ticker), _industry_of(ticker)))
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/equity/{ticker}/generate/{section}")
def equity_generate(ticker: str = TICKER,
                    section: str = Path(..., pattern="^(overview|porters|pestel|competitors|supply-chain)$"),
                    force: bool = Query(False),
                    user: AppUser = Depends(current_user)):
    """
    Generate one AI section. One DeepSeek call, 20-60s, cached afterwards.

    Per section rather than all at once: each is a separate cost, and asking
    for Porter's should not also bill PESTEL.
    """
    from api import equity_api

    market = markets.split(ticker)[0]
    if section != "supply-chain" and market != "CN":
        # The rest of the brief is built from Tushare fundamentals; the supply
        # chain is not — it is a model call about the company, and works
        # wherever the company is listed, with a market-appropriate prompt.
        raise HTTPException(404, "个股研报目前仅支持 A 股")
    try:
        if section == "supply-chain":
            out = {"payload": equity_api.generate_supply_chain(
                ticker, _name(ticker), market)}
        else:
            out = equity_api.generate(ticker, _name(ticker), _industry_of(ticker),
                                      section, force)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(502, f"AI 生成失败：{exc}"[:200])

    _equity_cache._d.pop(ticker, None)       # the brief now has a new section
    return {"section": section, **out}


class PeersReq(BaseModel):
    competitors: list[dict] = Field(..., max_length=20)


@app.post("/equity/{ticker}/peers")
def equity_save_peers(req: PeersReq, ticker: str = TICKER,
                      user: AppUser = Depends(current_user)):
    """Admin: replace the curated peer set for this stock."""
    import equity_brief as eb
    _require_admin(user)

    cleaned = []
    for p in req.competitors:
        t = str(p.get("ticker", "")).strip().zfill(6)
        if not (len(t) == 6 and t.isdigit() and t != ticker):
            continue
        cleaned.append({"ticker": t,
                        "name": str(p.get("name") or "").strip(),
                        "why": str(p.get("why") or "").strip()})
    eb.ensure_equity_brief_cache_table()
    eb.save_competitors_curated(ticker, cleaned)
    _equity_cache._d.pop(ticker, None)
    return {"saved": len(cleaned)}


class SectorReq(BaseModel):
    sector: str = Field(..., min_length=1, max_length=40)
    tickers: list[str] = Field(..., min_length=1, max_length=40)


@app.post("/equity/{ticker}/sector")
def equity_save_sector(req: SectorReq, ticker: str = TICKER,
                       user: AppUser = Depends(current_user)):
    """
    Admin: file this stock and its peers into a sector.

    Creates the sector when it does not exist. Sector membership drives the
    PPI_* indices and the 板块相关性 panel, so this is how a peer group you
    curated here becomes something the rest of the app can measure against.
    """
    import data_manager
    _require_admin(user)

    codes = []
    for t in dict.fromkeys(req.tickers):
        t = str(t).strip()
        if len(t) == 6 and t.isdigit():
            codes.append(t)
    if not codes:
        raise HTTPException(422, "没有有效的股票代码")

    existing = set(data_manager.get_sector_stock_map() or {})
    if req.sector not in existing:
        data_manager.add_new_sector(req.sector, codes)
    else:
        for t in codes:
            data_manager.add_stock_to_sector(req.sector, t)
    return {"sector": req.sector, "added": len(codes),
            "created": req.sector not in existing}


@app.get("/sectors")
def sector_list(user: AppUser = Depends(current_user)):
    """Sector names, for the admin picker."""
    import data_manager
    try:
        return sorted(data_manager.get_sector_stock_map() or {})
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(503, f"板块列表读取失败：{exc}"[:160])


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
    if markets.split(ticker)[0] != "CN":
        # The PPI_* sector indices are A-share only. US/CA sectors will come
        # from sector ETFs; until then this says so rather than correlating
        # a US stock against Chinese baskets, which would return numbers.
        raise HTTPException(404, "该市场暂无板块指数数据")

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
            with_: str = OTHER,
            user: AppUser = Depends(current_user)):
    """A second stock aligned to this one's bars, in both scalings."""
    from api import extras
    adf, _ = _frames(ticker)
    anchor = markets.parse(ticker)[0].conv.regime_anchor
    dates = [d.strftime("%Y-%m-%d") for d in ta_payload.chart_window(adf, anchor).index]
    try:
        return extras.compare(adf, with_, dates)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


# ── 市场看板 ─────────────────────────────────────────────────────────────────
# Six panels, six caches, all read-only and all shared by every user — none of
# this is per-account, so one instance computes each at most once per TTL.
#
# The TTLs follow how often the underlying number can actually change: the
# heatmap moves with the tape, 两融 is published once a day after the close,
# 龙虎榜 once an evening, and breadth / rotation come from the nightly rebuild.
_heatmap_cache = TTLCache(maxsize=1, ttl_s=30 * 60)
_breadth_cache = TTLCache(maxsize=4, ttl_s=6 * 3600)
_leverage_cache = TTLCache(maxsize=1, ttl_s=30 * 60)
_toplist_cache = TTLCache(maxsize=1, ttl_s=4 * 3600)
_wyckoff_cache = TTLCache(maxsize=4, ttl_s=60 * 60)
_rotation_cache = TTLCache(maxsize=4, ttl_s=6 * 3600)

#: Indices the Wyckoff panel will run on. An allow-list rather than a free
#: parameter: the phases are only meaningful on a broad index, and an open
#: parameter is an open Tushare call with someone else's quota.
WYCKOFF_INDICES = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "399006.SZ": "创业板指",
    "000001.SH": "上证指数",
}


def _market_panel(cache, key, build):
    """Shared error contract for the dashboard panels."""
    try:
        return cache.get_or_compute(key, build)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(503, f"{type(exc).__name__}: {exc}"[:200])


@app.get("/market/heatmap")
def market_heatmap(user: AppUser = Depends(current_user)):
    """Sector → stock tree, sized by 流通市值 and coloured by today's move."""
    from api import dashboard_api
    return _market_panel(_heatmap_cache, "all", dashboard_api.heatmap)


@app.get("/market/breadth")
def market_breadth(days: int = Query(60, ge=10, le=250),
                   user: AppUser = Depends(current_user)):
    """
    The market_breadth table, by session.

    Not a member count despite the name — each cell is the sector index's own
    distance from its MA20, mapped from ±5% onto 0–1 and clipped. The client
    says so; see sector_rotation.load_trend for the full story.
    """
    from api import dashboard_api
    return _market_panel(_breadth_cache, days, lambda: dashboard_api.breadth(days))


@app.get("/market/leverage")
def market_leverage(user: AppUser = Depends(current_user)):
    """两融余额 and US margin debt."""
    from api import dashboard_api
    return _market_panel(_leverage_cache, "all", dashboard_api.leverage)


@app.get("/market/toplist")
def market_toplist(user: AppUser = Depends(current_user)):
    """龙虎榜 — the most recent session's abnormal-trading list."""
    from api import dashboard_api
    return _market_panel(_toplist_cache, "all", dashboard_api.top_list)


@app.get("/market/wyckoff")
def market_wyckoff(index: str = Query("000300.SH"),
                   user: AppUser = Depends(current_user)):
    """Statistical Wyckoff phase for a broad index, with what each phase paid."""
    if index not in WYCKOFF_INDICES:
        raise HTTPException(422, f"index must be one of {sorted(WYCKOFF_INDICES)}")

    def build():
        import data_manager
        import wyckoff
        # Four years: the chart shows 180 bars, but the forward-return table
        # needs every phase to appear enough times to be worth printing.
        df = data_manager.get_index_data_live(index, lookback_days=1500, freq="daily")
        if df is None or df.empty:
            raise RuntimeError(f"{WYCKOFF_INDICES[index]} 指数数据暂时读取不到 — 请稍后重试")
        return wyckoff.analyse(df, name=WYCKOFF_INDICES[index])

    return _market_panel(_wyckoff_cache, index, build)


@app.get("/market/rotation")
def market_rotation(freq: str = Query("w", pattern="^(w|d)$"),
                    user: AppUser = Depends(current_user)):
    """
    The relative-rotation map: which sector money is leaving, which it is entering.

    Replaces the pairwise-correlation panel, which could report that rotation
    was happening but never which direction — see sector_rotation's docstring.
    """
    def build():
        import sector_rotation
        closes, bench, label = sector_rotation.load_sector_closes()
        if not closes:
            raise RuntimeError("板块指数（PPI_*）暂时读取不到 — 请稍后重试")
        return sector_rotation.analyse(closes, bench, freq=freq,
                                       benchmark_label=label,
                                       trend=sector_rotation.load_trend())

    return _market_panel(_rotation_cache, freq, build)


# ── MyQuestrade ──────────────────────────────────────────────────────────────
# Per-user and never shared: the cache key is the app_user_id, because these
# are somebody's actual holdings.
#
# Positions move with the tape, so the book is held only a few minutes; the
# risk report is built from three years of daily history and changes far more
# slowly than that, so it is held for half an hour.
_qt_book_cache = TTLCache(maxsize=8, ttl_s=3 * 60)
_qt_risk_cache = TTLCache(maxsize=24, ttl_s=30 * 60)
# Exposure costs a Yahoo profile read per holding, so it is held longest —
# a company does not change sector between page loads.
_qt_exposure_cache = TTLCache(maxsize=8, ttl_s=6 * 3600)
_qt_opt_cache = TTLCache(maxsize=32, ttl_s=30 * 60)


class QuestradeConnectReq(BaseModel):
    """
    The manual refresh token from apphub.questrade.com.

    It goes straight to Questrade to be exchanged and is never echoed back,
    never logged, and never sent to any client — see api/questrade_api.py and
    the migration for where the rotated replacement lives.
    """
    refresh_token: str = Field(..., min_length=8, max_length=512)


def _questrade(exc: Exception):
    """
    Questrade's failures, mapped onto statuses the frontend can act on.

    409 rather than 401 for a broken token chain, deliberately: the client
    signs the user out of Supabase on a 401, and "your brokerage link expired"
    must not log you out of the app.
    """
    import questrade as qtm
    if isinstance(exc, qtm.NeedsReconnect):
        raise HTTPException(409, str(exc))
    if isinstance(exc, qtm.RateLimited):
        raise HTTPException(429, str(exc))
    if isinstance(exc, LookupError):
        raise HTTPException(404, str(exc))
    if isinstance(exc, qtm.QuestradeError):
        raise HTTPException(502, str(exc))
    raise HTTPException(503, f"{type(exc).__name__}: {exc}"[:200])


@app.get("/questrade/status")
def questrade_status(user: AppUser = Depends(current_user)):
    """Whether this user has a live connection. Never any part of the token."""
    from api import questrade_api
    try:
        return questrade_api.client(user.id).status()
    except Exception as exc:                                    # noqa: BLE001
        _questrade(exc)


@app.post("/questrade/connect")
def questrade_connect(req: QuestradeConnectReq, user: AppUser = Depends(current_user)):
    """
    Adopt a manually generated refresh token.

    Exchanged immediately. A token that does not work has to fail here, while
    the user is still looking at the box they pasted it into.
    """
    from api import questrade_api
    try:
        client = questrade_api.client(user.id)
        client.connect(req.refresh_token)
        accounts = client.accounts()
    except Exception as exc:                                    # noqa: BLE001
        _questrade(exc)
    _forget_questrade(user.id)
    return {**client.status(), "accounts": accounts}


def _forget_questrade(user_id: int) -> None:
    """
    Drop this user's cached book and risk report.

    The TTL caches have no delete, so a None is stored instead — every read
    site treats a falsy value as a miss. Called on connect AND on disconnect:
    reconnecting to a different Questrade login must not show the previous
    one's positions for the next three minutes.
    """
    from api import questrade_api
    questrade_api.forget(user_id)
    for base in ("CAD", "USD"):
        _qt_book_cache.put(("book", user_id, base), None)
        _qt_exposure_cache.put(("exposure", user_id, base), None)
        for scope in ("all", "stock", "etf"):
            for bench in ("^GSPC", "^IXIC", "^GSPTSE"):
                _qt_risk_cache.put(("risk", user_id, bench, base, scope), None)


@app.delete("/questrade/connect")
def questrade_disconnect(user: AppUser = Depends(current_user)):
    from api import questrade_api
    questrade_api.client(user.id).disconnect()
    _forget_questrade(user.id)
    return {"connected": False}


@app.get("/questrade/portfolio")
def questrade_portfolio(base: str = Query("CAD", pattern="^(CAD|USD)$"),
                        user: AppUser = Depends(current_user)):
    """Every account consolidated into one book, in `base` currency."""
    from api import questrade_api
    key = ("book", user.id, base)
    cached = _qt_book_cache.peek(key)
    if cached:
        return cached
    try:
        book = questrade_api.snapshot(user.id, base=base)
    except Exception as exc:                                    # noqa: BLE001
        _questrade(exc)
    _qt_book_cache.put(key, book)
    return book


@app.get("/questrade/risk")
def questrade_risk(benchmark: str = Query("^GSPC"),
                   base: str = Query("CAD", pattern="^(CAD|USD)$"),
                   scope: str = Query("all", pattern="^(all|stock|etf)$"),
                   user: AppUser = Depends(current_user)):
    """Today's holdings replayed against an index — both in `base` currency."""
    import portfolio
    from api import questrade_api
    if benchmark not in portfolio.BENCHMARKS:
        raise HTTPException(422, f"benchmark must be one of {sorted(portfolio.BENCHMARKS)}")

    key = ("risk", user.id, benchmark, base, scope)
    cached = _qt_risk_cache.peek(key)
    if cached:
        return cached
    try:
        report = questrade_api.risk_report(user.id, benchmark=benchmark,
                                          base=base, scope=scope)
    except Exception as exc:                                    # noqa: BLE001
        _questrade(exc)
    _qt_risk_cache.put(key, report)
    return report


@app.get("/questrade/exposure")
def questrade_exposure(base: str = Query("CAD", pattern="^(CAD|USD)$"),
                       user: AppUser = Depends(current_user)):
    """
    Sector exposure with the ETFs looked through.

    Slow on a first call — it reads a profile per holding from Yahoo — and
    then cached, which is why it is its own route rather than part of the
    portfolio payload.
    """
    from api import questrade_api
    key = ("exposure", user.id, base)
    cached = _qt_exposure_cache.peek(key)
    if cached:
        return cached
    try:
        report = questrade_api.exposure_report(user.id, base=base)
    except Exception as exc:                                    # noqa: BLE001
        _questrade(exc)
    _qt_exposure_cache.put(key, report)
    return report


@app.get("/questrade/optimise")
def questrade_optimise(base: str = Query("CAD", pattern="^(CAD|USD)$"),
                       scope: str = Query("all", pattern="^(all|stock|etf)$"),
                       method: str = Query("min_var"),
                       cap: float = Query(0.25, ge=0.05, le=1.0),
                       user: AppUser = Depends(current_user)):
    """A target allocation over the current holdings, with an out-of-sample check."""
    import optimise
    from api import questrade_api
    if method not in optimise.METHODS:
        raise HTTPException(422, f"method must be one of {sorted(optimise.METHODS)}")

    key = ("opt", user.id, base, scope, method, round(cap, 3))
    cached = _qt_opt_cache.peek(key)
    if cached:
        return cached
    try:
        report = questrade_api.optimise_report(user.id, base=base, scope=scope,
                                               method=method, cap=cap)
    except Exception as exc:                                    # noqa: BLE001
        _questrade(exc)
    _qt_opt_cache.put(key, report)
    return report


@app.get("/questrade/benchmarks")
def questrade_benchmarks(user: AppUser = Depends(current_user)):
    import portfolio
    return [{"id": k, "name": v} for k, v in portfolio.BENCHMARKS.items()]


@app.get("/supply-chain/{ticker}")
def supply_chain_graph(ticker: str = TICKER, user: AppUser = Depends(current_user)):
    """
    The saved supply-chain graph for any market, without the equity brief.

    Separate from /equity/{ticker} because that route is A-share only — it is
    built on Tushare fundamentals — while a supply-chain graph is a model's
    account of what a company makes and who buys it, which is answerable for a
    US or Canadian listing just as well. The watchlist reads this one.

    An absent graph is {} with generated=false, not a 404: "nobody has
    generated this yet" is the normal state of a stock you just added, and a
    404 would have the client render it as an error.
    """
    from api import equity_api
    graph = equity_api._supply_chain(ticker)
    return {"ticker": ticker, "generated": bool(graph.get("products")), **graph}


@app.post("/questrade/keepalive")
def questrade_keepalive(x_keepalive_secret: str = Header(default="")):
    """
    Refresh every connected Questrade token. For a scheduler, not a browser.

    A Questrade refresh token lives three days and is single-use; each exchange
    mints the next one. Refreshing only when somebody opens the page means the
    chain dies over any long absence — which is indistinguishable from a bug
    and is why this exists.

    Guarded by a shared secret rather than a user login, because the caller is
    a cron job with no user. If QUESTRADE_KEEPALIVE_SECRET is unset the route
    refuses outright: an unconfigured secret must not mean an open endpoint.
    """
    import hmac
    from api import questrade_api

    want = os.environ.get("QUESTRADE_KEEPALIVE_SECRET", "")
    if not want:
        raise HTTPException(503, "QUESTRADE_KEEPALIVE_SECRET 未配置，保活接口已禁用")
    if not hmac.compare_digest(x_keepalive_secret, want):
        raise HTTPException(403, "forbidden")

    try:
        return questrade_api.keepalive()
    except Exception as exc:                                    # noqa: BLE001
        _questrade(exc)
