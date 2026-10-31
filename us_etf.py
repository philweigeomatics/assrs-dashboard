"""
us_etf.py — where every US industry stands, read through its ETF.

One stock against its own 200-day average says something about that stock.
An industry ETF against ITS 200-day says something about the industry —
whether the whole group is in a structural uptrend, stretched, or broken —
which is the backdrop a single name trades against.

Per ETF, three readings:

  * Level and structure: distance from the 50- and 200-day averages, whether
    the 200-day is itself rising, the 50/200 cross and how long ago, where
    price sits in its 52-week range, and the drawdown from the high.
  * Risk: how stretched it is against ITS OWN history — +15% above the
    200-day is ordinary for semiconductors and extreme for utilities, so the
    distance is ranked within the ETF's ten years — and whether volatility is
    high for it. Then the part a level alone cannot say: on the past days it
    was at least this stretched, what did the next three months bring?
  * Relative strength against SPY over three months.

Breadth across the sector and industry funds sums it up: how many are above
their 200-day, how many above the 50-day, how many in golden cross.

Prices are Yahoo's, split- and dividend-adjusted. During US trading hours
the last bar is today's, 15 minutes delayed and still moving.
"""

from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BENCH = "SPY"

#: (ticker, 中文名, group). Liquid, long-lived funds, so the ten-year stretch
#: ranking has real history behind it.
UNIVERSE: list[tuple[str, str, str]] = [
    ("SPY", "标普500", "大盘"), ("QQQ", "纳指100", "大盘"), ("RSP", "标普500等权", "大盘"),
    ("IWM", "罗素2000小盘", "大盘"), ("DIA", "道琼斯", "大盘"),
    ("XLK", "科技", "板块"), ("XLC", "通信服务", "板块"), ("XLY", "可选消费", "板块"),
    ("XLF", "金融", "板块"), ("XLV", "医疗保健", "板块"), ("XLI", "工业", "板块"),
    ("XLE", "能源", "板块"), ("XLB", "材料", "板块"), ("XLP", "必需消费", "板块"),
    ("XLU", "公用事业", "板块"), ("XLRE", "房地产", "板块"),
    ("SMH", "半导体", "科技细分"), ("SOXX", "半导体(iShares)", "科技细分"),
    ("IGV", "软件", "科技细分"), ("CIBR", "网络安全", "科技细分"), ("SKYY", "云计算", "科技细分"),
    ("FDN", "互联网", "科技细分"), ("AIQ", "人工智能", "科技细分"), ("BOTZ", "机器人与AI", "科技细分"),
    ("ARKK", "颠覆式创新", "科技细分"),
    ("XBI", "生物科技", "医疗与金融"), ("IHI", "医疗器械", "医疗与金融"),
    ("KRE", "地区银行", "医疗与金融"), ("KBE", "银行", "医疗与金融"), ("IAI", "券商", "医疗与金融"),
    ("ITA", "航空航天与国防", "工业与消费"), ("IYT", "交通运输", "工业与消费"),
    ("PAVE", "基础设施", "工业与消费"), ("XHB", "住宅建筑", "工业与消费"),
    ("JETS", "航空公司", "工业与消费"), ("XRT", "零售", "工业与消费"),
    ("XOP", "油气开采", "能源与资源"), ("OIH", "油服", "能源与资源"), ("TAN", "太阳能", "能源与资源"),
    ("ICLN", "清洁能源", "能源与资源"), ("LIT", "锂电池", "能源与资源"), ("URA", "铀", "能源与资源"),
    ("GDX", "金矿", "能源与资源"), ("COPX", "铜矿", "能源与资源"), ("XME", "金属与采矿", "能源与资源"),
]
GROUPS = ["大盘", "板块", "科技细分", "医疗与金融", "工业与消费", "能源与资源"]
NAMES = {t: n for t, n, _ in UNIVERSE}
GROUP_OF = {t: g for t, _, g in UNIVERSE}

FWD_DAYS = 63          # "the next three months"
EPISODE_GAP = 21       # stretched days this close are one episode
SPARK_WEEKS = 52
MIN_BARS = 220         # enough for a 200-day average and some slope


def _f(v, nd=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(v) else round(v, nd)


def _pct(a, b):
    return (a / b - 1) * 100 if b else float("nan")


# ── data ─────────────────────────────────────────────────────────────────────
def fetch(period: str = "10y") -> pd.DataFrame:
    """Adjusted closes for the whole universe, one bulk download."""
    import yfinance as yf

    tickers = [t for t, _, _ in UNIVERSE]
    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                      progress=False, threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo 没有返回美股 ETF 行情 — 请稍后重试")
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    idx = pd.DatetimeIndex(closes.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    closes.index = idx.normalize()
    return closes.sort_index().dropna(how="all")


def session_state(last_bar: pd.Timestamp, now: datetime | None = None) -> str:
    """'live' when the last bar is today's and New York is trading; 'closed' otherwise."""
    ny = (now or datetime.now(ZoneInfo("America/New_York"))).astimezone(ZoneInfo("America/New_York"))
    open_ = ny.weekday() < 5 and (9, 30) <= (ny.hour, ny.minute) < (16, 0)
    return "live" if open_ and last_bar.date() == ny.date() else "closed"


# ── one ETF ──────────────────────────────────────────────────────────────────
def stretch_history(close: pd.Series) -> pd.Series:
    """Distance from the 200-day average, in %, every day it is defined."""
    return (close / close.rolling(200, min_periods=200).mean() - 1) * 100


def forward_after(close: pd.Series, dist: pd.Series, *, above: bool, level: float) -> dict | None:
    """
    What the next FWD_DAYS brought, starting from each past episode that was
    at least this stretched (above) or this washed out (below).

    Episodes, not days: three hundred consecutive stretched days are one
    stretch, and counting them as three hundred outcomes would claim a
    certainty the history does not hold. Each episode starts on its first
    qualifying day, and the next cannot start within EPISODE_GAP sessions.
    The latest FWD_DAYS are left out — their future has not happened yet.
    """
    c = close.to_numpy()
    d = dist.to_numpy()
    n = len(c)
    hit = (d >= level) if above else (d <= level)
    hit = np.nan_to_num(hit, nan=False).astype(bool)
    starts, last = [], -10 ** 9
    for i in np.flatnonzero(hit):
        if i + FWD_DAYS >= n:
            break
        if i - last > EPISODE_GAP:
            starts.append(i)
        last = i
    if len(starts) < 3:
        return None
    fwd = np.array([(c[i + FWD_DAYS] / c[i] - 1) * 100 for i in starts])
    worst = np.array([(c[i:i + FWD_DAYS + 1].min() / c[i] - 1) * 100 for i in starts])
    every = (c[FWD_DAYS:] / c[:-FWD_DAYS] - 1) * 100
    ok = ~np.isnan(d[:-FWD_DAYS])
    base = every[ok] if ok.any() else every
    return {
        "episodes": len(starts),
        "median": _f(np.median(fwd), 1), "down_share": _f(np.mean(fwd < 0), 2),
        "worst_median": _f(np.median(worst), 1),
        "base_median": _f(np.median(base), 1), "base_down_share": _f(np.mean(base < 0), 2),
        "dates": [str(close.index[i].date()) for i in starts[-6:]],
    }


def trend_state(above50: bool, above200: bool) -> str:
    return {(True, True): "多头", (False, True): "回调", (True, False): "反弹",
            (False, False): "空头"}[(above50, above200)]


def row(t: str, close: pd.Series, bench: pd.Series | None) -> dict | None:
    close = close.dropna()
    if len(close) < MIN_BARS:
        return None
    last = float(close.iloc[-1])
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    dist = stretch_history(close)
    d200 = float(dist.iloc[-1])
    d50 = _pct(last, float(ma50.iloc[-1]))
    slope200 = _pct(float(ma200.iloc[-1]), float(ma200.iloc[-21]))
    golden = (ma50 > ma200)
    g = golden.to_numpy()
    flips = np.flatnonzero(g[1:] != g[:-1]) + 1
    cross_days = int(len(g) - flips[-1]) if len(flips) else None
    yr = close.tail(252)
    hi, lo = float(yr.max()), float(yr.min())
    pct_rank = float((dist.dropna() < d200).mean() * 100)

    r = close.pct_change()
    vol20 = float(r.tail(20).std() * math.sqrt(252) * 100)
    vol_hist = (r.rolling(20).std() * math.sqrt(252) * 100).dropna().tail(756)
    vol_rank = float((vol_hist < vol20).mean() * 100) if len(vol_hist) > 60 else None

    def ret(k):
        return _f(_pct(last, float(close.iloc[-1 - k])), 1) if len(close) > k else None

    prev_year = close[close.index.year == close.index[-1].year - 1]
    ytd_ret = _f(_pct(last, float(prev_year.iloc[-1])), 1) if len(prev_year) else None

    rs = None
    if bench is not None and t != BENCH:
        b = bench.reindex(close.index).ffill()
        if b.notna().sum() > 70:
            rs = _f(((last / float(close.iloc[-64])) / (float(b.iloc[-1]) / float(b.iloc[-64])) - 1) * 100, 1)

    # The "risk" read: what followed past stretches at least this big. Only
    # asked at the extremes of the ETF's own range — in the middle, "what
    # happened next" is just the base rate.
    after = None
    if pct_rank >= 80:
        after = forward_after(close, dist, above=True, level=d200)
    elif pct_rank <= 20:
        after = forward_after(close, dist, above=False, level=d200)

    weekly = close.tail(SPARK_WEEKS * 5)[::-5][::-1]
    weekly_ma = ma200.reindex(weekly.index)
    return {
        "ticker": t, "name": NAMES.get(t, t), "group": GROUP_OF.get(t, ""),
        "price": _f(last, 2), "date": str(close.index[-1].date()),
        "ret": {"1d": ret(1), "5d": ret(5), "1m": ret(21), "3m": ret(63), "1y": ret(252), "ytd": ytd_ret},
        "d50": _f(d50, 1), "d200": _f(d200, 1), "slope200": _f(slope200, 2),
        "golden": bool(g[-1]), "cross_days": cross_days,
        "state": trend_state(d50 > 0, d200 > 0),
        "hi52": _f(hi, 2), "lo52": _f(lo, 2),
        "from_high": _f(_pct(last, hi), 1), "range_pos": _f((last - lo) / (hi - lo), 3) if hi > lo else None,
        "stretch_rank": _f(pct_rank, 0), "history_years": _f(len(close) / 252, 1),
        "vol20": _f(vol20, 1), "vol_rank": _f(vol_rank, 0),
        "rs_3m": rs, "after": after,
        "spark": [_f(v, 2) for v in weekly.to_numpy()],
        "spark_ma200": [_f(v, 2) for v in weekly_ma.to_numpy()],
    }


# ── the board ────────────────────────────────────────────────────────────────
def breadth(rows: list[dict]) -> dict:
    """Across the sector and industry funds — the broad-market ones are what is being explained."""
    pool = [r for r in rows if r["group"] != "大盘"]
    if not pool:
        return {}
    n = len(pool)
    by = sorted(pool, key=lambda r: -(r["stretch_rank"] or 0))
    return {
        "n": n,
        "above200": sum(r["d200"] > 0 for r in pool), "above50": sum(r["d50"] > 0 for r in pool),
        "golden": sum(r["golden"] for r in pool),
        "near_high": sum((r["from_high"] or -99) > -3 for r in pool),
        "median_d200": _f(float(np.median([r["d200"] for r in pool])), 1),
        "stretched": [r["ticker"] for r in by[:3]],
        "washed_out": [r["ticker"] for r in by[-3:][::-1]],
    }


def conclusions(rows: list[dict], br: dict) -> list[str]:
    if not br:
        return []
    out = []
    n = br["n"]
    share = br["above200"] / n
    tone = ("普遍处在长期上升趋势中" if share >= 0.7 else
            "多数仍在长期均线上方，但并不普遍" if share >= 0.5 else
            "多数已跌破长期均线 —— 结构性走弱" if share >= 0.3 else
            "绝大多数在长期均线下方 —— 普遍的下降趋势")
    out.append(f"{n} 个行业与主题 ETF 中，{br['above200']} 个在 200 日均线上方、{br['above50']} 个在 50 日均线上方："
               f"{tone}。")
    if br["above50"] < br["above200"] - 0.2 * n:
        out.append("50 日均线上方的明显少于 200 日上方：长期趋势还在，但短期在普遍回调。")
    elif br["above50"] > br["above200"] + 0.2 * n:
        out.append("50 日均线上方的明显多于 200 日上方：短期在普遍反弹，长期趋势还没修复。")
    by_t = {r["ticker"]: r for r in rows}
    hot = [by_t[t] for t in br["stretched"]
           if by_t[t]["stretch_rank"] is not None and by_t[t]["stretch_rank"] >= 90]
    if hot:
        out.append("相对自身历史最拉伸的：" + "、".join(
            f"{r['name']} {r['ticker']}（高于 200 日线 {r['d200']:+.0f}%，处于自身 {r['stretch_rank']:.0f} 分位）"
            for r in hot) + "。")
    for r in hot[:2]:
        a = r.get("after")
        if not a:
            out.append(f"{r['ticker']} 现在的拉伸程度在它 {r['history_years']:.0f} 年的历史里几乎没有先例"
                       f"（不足 3 次可比），历史给不出参照 —— 这本身就是风险。")
        if a:
            out.append(f"{r['ticker']} 过去 {a['episodes']} 次拉伸到这个程度，之后 3 个月中位 {a['median']:+.1f}%"
                       f"（{a['down_share'] * 100:.0f}% 的情况下跌，期间最大回撤中位 {a['worst_median']:.1f}%）；"
                       f"所有日子的 3 个月中位是 {a['base_median']:+.1f}%。")
    # `is not None`, not `or`: the most washed-out ETF of all has rank 0.
    cold = [by_t[t] for t in br["washed_out"]
            if by_t[t]["stretch_rank"] is not None and by_t[t]["stretch_rank"] <= 10]
    if cold:
        out.append("相对自身历史最超跌的：" + "、".join(
            f"{r['name']} {r['ticker']}（{r['d200']:+.0f}%，{r['stretch_rank']:.0f} 分位）" for r in cold) + "。")
        for r in cold[:2]:
            a = r.get("after")
            if a:
                out.append(f"{r['ticker']} 过去 {a['episodes']} 次跌到这个程度，之后 3 个月中位 {a['median']:+.1f}%"
                           f"（{(1 - a['down_share']) * 100:.0f}% 的情况上涨）；所有日子的 3 个月中位是 "
                           f"{a['base_median']:+.1f}%。次数很少，是参照不是规律。")
            else:
                out.append(f"{r['ticker']} 已低于它历史上任何可比的位置 —— 没有先例可参照。")
    return out


def analyse(closes: pd.DataFrame, now: datetime | None = None) -> dict:
    bench = closes[BENCH].dropna() if BENCH in closes.columns else None
    rows = [x for x in (row(t, closes[t], bench) for t, _, _ in UNIVERSE if t in closes.columns) if x]
    if not rows:
        raise LookupError("没有足够历史的美股 ETF 数据")
    br = breadth(rows)
    last = closes.index.max()
    return {
        "as_of": str(last.date()), "session": session_state(last, now),
        "groups": GROUPS, "rows": rows, "breadth": br,
        "conclusions": conclusions(rows, br),
        "missing": [t for t, _, _ in UNIVERSE if t not in {r["ticker"] for r in rows}],
    }


def detail(closes: pd.DataFrame, ticker: str, years: int = 5) -> dict:
    """One ETF over `years`: price with 50/200-day lines and its stretch with its own bands."""
    if ticker not in closes.columns:
        raise LookupError(f"没有 {ticker}")
    close = closes[ticker].dropna()
    dist = stretch_history(close)
    bands = {k: _f(np.nanpercentile(dist.dropna(), k), 1) for k in (10, 50, 90)} if dist.notna().any() else {}
    cut = close.index >= close.index[-1] - pd.DateOffset(years=years)
    ma50 = close.rolling(50).mean()[cut]
    ma200 = close.rolling(200).mean()[cut]
    return {
        "ticker": ticker, "name": NAMES.get(ticker, ticker),
        "dates": [str(d.date()) for d in close.index[cut]],
        "close": [_f(v, 2) for v in close[cut].to_numpy()],
        "ma50": [_f(v, 2) for v in ma50.to_numpy()],
        "ma200": [_f(v, 2) for v in ma200.to_numpy()],
        "stretch": [_f(v, 2) for v in dist[cut].to_numpy()],
        "bands": bands,
    }
