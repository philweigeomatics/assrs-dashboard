"""
market_risk.py — 🚦 市场风险: five readings of how fragile the market is today.

    成交额        is there money in the market at all?
    两融占成交     how much of today's buying was done with borrowed money?
    涨停情绪       is the speculative end euphoric, split, or panicking?
    估值          how much good news is already in the price?
    波动          how violently is the index moving?

A level means nothing without something to compare it against, so every tile
carries its PERCENTILE within the new-market regime — every session since
2024-10-14, the first clean session after the 9·24 policy turn. A 3–5 year
lookback would grade today against a different market: turnover of 1万亿
was a busy day in 2023 and is a dead one now.

Each tile is graded from its percentile into a state and a level:

    level 0   nothing to flag
    level 1   worth watching
    level 2   a warning

The grade is about RISK, not direction, which is why the client draws it in
its own neutral-to-amber scale and never in red/green: under 红涨绿跌 a red
tile would read "rising", and "turnover rising" is the opposite of a warning.

Two things this module would have used and cannot, on the current Tushare
tier: `limit_list_d` (the exchange's own limit-up list) and `yc_cb` (the
ChinaBond curve, for an equity risk premium). Limit-up counts are therefore
derived from `daily` against `stk_limit`, and valuation is the PE percentile
alone.

Limit-up stats cost two whole-market calls per session, ~500 sessions since
the anchor, so they are kept per day in `market_limit_stats` and filled in
newest-first. A past session's count never changes; there is nothing to
expire. Without the table the store is in-process only — the tile still
works, its percentile just rests on whatever has been fetched since start-up.
Run supabase/migrations/20261027_market_limit_stats.sql to persist it, and
`python market_risk.py` once to backfill.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CHINA_TZ = ZoneInfo("Asia/Shanghai")

#: First session of the regime every percentile is measured against.
ANCHOR = "20241014"
#: Fetch from here so the 20-day windows already have a full window by the
#: anchor rather than starting it with five sessions of history.
WARMUP_START = "20240801"
#: Sessions of shape on each tile's sparkline.
HISTORY = 60
#: Fewer regime sessions than this and a percentile is noise, not a grade.
MIN_SAMPLE = 20
#: The index valuation and volatility are read from.
INDEX = "000300.SH"
#: daily_info codes whose sum is the whole A/B-share market. SH_MARKET is
#: 主板 + 科创板 + B 股; SZ_MARKET is 主板 + 创业板. B shares are a rounding
#: error; 北交所 is not in daily_info at all.
TURNOVER_CODES = ("SH_MARKET", "SZ_MARKET")
#: Margin on the same two exchanges, so the ratio has the same basket top and
#: bottom.
MARGIN_EXCHANGES = ("SSE", "SZSE")

TABLE = "market_limit_stats"
#: Sessions fetched inside a request before the rest is left to the
#: background fill — enough to keep the latest few current without making
#: the panel wait for a two-year backfill.
SYNC_BUDGET = 5
#: Parallel calls. Low on purpose: the heatmap reads `daily` too, and the
#: per-minute quota is shared.
WORKERS = 3

#: (lower, upper, state, level) by percentile. Upper bounds are exclusive,
#: and the top band ends above 100 so the regime high itself is graded.
GRADES = {
    "turnover": [(0, 10, "枯竭", 2), (10, 25, "偏冷", 1), (25, 75, "正常", 0),
                 (75, 90, "活跃", 0), (90, 101, "过热", 1)],
    "margin": [(0, 25, "低迷", 0), (25, 75, "正常", 0),
               (75, 90, "偏高", 1), (90, 101, "过热", 2)],
    "limits": [(0, 25, "冷清", 0), (25, 75, "正常", 0),
               (75, 90, "活跃", 0), (90, 101, "亢奋", 1)],
    "valuation": [(0, 10, "低估", 0), (10, 75, "合理", 0),
                  (75, 90, "偏高", 1), (90, 101, "偏贵", 2)],
    "vol": [(0, 25, "平静", 0), (25, 75, "正常", 0),
            (75, 90, "偏高", 1), (90, 101, "剧烈", 2)],
}


def _api():
    import macro
    return macro._api()


def _today() -> str:
    return datetime.now(CHINA_TZ).strftime("%Y%m%d")


def _iso(d: str) -> str:
    d = str(d)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d


def _num(v, nd: int):
    if v is None or not np.isfinite(v):
        return None
    return round(float(v), nd)


# ── grading ──────────────────────────────────────────────────────────────────
def percentile(s: pd.Series) -> tuple[float | None, int]:
    """
    Where the latest reading sits among every regime session, 0–100.

    The share of regime sessions at or below today — so the regime high reads
    100 and the low reads 100/n, not 0. `s` is indexed by trade date
    (YYYYMMDD) and sorted ascending.
    """
    r = s[s.index >= ANCHOR].dropna()
    if len(r) < MIN_SAMPLE:
        return None, len(r)
    return round(float((r <= r.iloc[-1]).mean() * 100), 1), len(r)


def grade(key: str, pct: float | None) -> tuple[str, int]:
    if pct is None:
        return "样本不足", 0
    for lo, hi, state, level in GRADES[key]:
        if lo <= pct < hi:
            return state, level
    return "正常", 0


def tile(key: str, label: str, unit: str, s: pd.Series, *, nd: int,
         note: str, extra: list[dict] | None = None) -> dict | None:
    """One tile: the latest reading, its regime percentile, and its shape."""
    s = s.dropna().sort_index()
    if s.empty:
        return None
    pct, n = percentile(s)
    state, level = grade(key, pct)
    reg = s[s.index >= ANCHOR]
    tail = s.tail(HISTORY)
    return {
        "key": key, "label": label, "unit": unit,
        "value": _num(s.iloc[-1], nd),
        "prev": _num(s.iloc[-2], nd) if len(s) >= 2 else None,
        "asof": _iso(s.index[-1]),
        "pct": pct, "n": n,
        "median": _num(reg.median(), nd) if len(reg) else None,
        "state": state, "level": level, "note": note,
        "history": [_num(v, nd) for v in tail],
        "dates": [_iso(d) for d in tail.index],
        "extra": extra or [],
    }


# ── series, from frames ──────────────────────────────────────────────────────
def turnover_series(info: pd.DataFrame | None) -> pd.Series:
    """
    Whole-market 成交额 in 亿元, by session.

    Only sessions where BOTH exchanges reported: a day with one side missing
    would read as liquidity halving, and a percentile would grade it 枯竭.
    """
    if info is None or info.empty:
        return pd.Series(dtype=float)
    d = info[info["ts_code"].isin(TURNOVER_CODES)].copy()
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    piv = d.pivot_table(index="trade_date", columns="ts_code", values="amount",
                        aggfunc="sum")
    piv = piv.reindex(columns=list(TURNOVER_CODES)).dropna()
    return piv.sum(axis=1).sort_index().astype(float)


def margin_buy_series(margin: pd.DataFrame | None) -> pd.Series:
    """融资买入额 (rzmre) in 亿元, SSE + SZSE, complete sessions only."""
    if margin is None or margin.empty or "rzmre" not in margin.columns:
        return pd.Series(dtype=float)
    d = margin[margin["exchange_id"].isin(MARGIN_EXCHANGES)].copy()
    d["rzmre"] = pd.to_numeric(d["rzmre"], errors="coerce")
    piv = d.pivot_table(index="trade_date", columns="exchange_id",
                        values="rzmre", aggfunc="sum")
    piv = piv.reindex(columns=list(MARGIN_EXCHANGES)).dropna()
    return (piv.sum(axis=1) / 1e8).sort_index().astype(float)


def margin_share(buy: pd.Series, turnover: pd.Series) -> pd.Series:
    """融资买入额 as a % of 成交额, on the sessions both exist."""
    both = pd.concat([buy.rename("buy"), turnover.rename("amt")], axis=1).dropna()
    both = both[both["amt"] > 0]
    return (both["buy"] / both["amt"] * 100).sort_index()


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualised close-to-close volatility, in %."""
    r = np.log(close.sort_index().astype(float)).diff()
    return (r.rolling(window).std() * np.sqrt(252) * 100).dropna()


def limit_stats(daily: pd.DataFrame | None, limits: pd.DataFrame | None) -> dict | None:
    """
    One session's 涨停 / 跌停 / 炸板 from prices and that day's limit prices.

    Compared at two decimals because that is what both are quoted to — a raw
    float comparison misses a stock sealed at exactly its limit whenever the
    limit was computed as 10.89000000001. 炸板 = touched the limit intraday
    but did not close there. Stocks without a limit (the first days of a new
    listing) have no row in `stk_limit` and drop out of the merge.
    """
    if daily is None or daily.empty or limits is None or limits.empty:
        return None
    m = daily[["ts_code", "close", "high"]].merge(
        limits[["ts_code", "up_limit", "down_limit"]], on="ts_code")
    for c in ("close", "high", "up_limit", "down_limit"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna()
    if m.empty:
        return None
    close, high = m["close"].round(2), m["high"].round(2)
    up, down = m["up_limit"].round(2), m["down_limit"].round(2)
    return {"up_count": int((close >= up).sum()),
            "down_count": int((close <= down).sum()),
            "touched": int((high >= up).sum()),
            "total": int(len(m))}


def limit_frame(rows: dict[str, dict]) -> pd.DataFrame:
    """The stored sessions as a frame with 炸板率 alongside."""
    if not rows:
        return pd.DataFrame(columns=["up_count", "down_count", "touched",
                                     "total", "broken_rate"])
    f = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    f = f[["up_count", "down_count", "touched", "total"]].astype(float)
    # A session where nothing touched the limit has no 炸板率, not a zero one.
    touched = f["touched"].where(f["touched"] > 0)
    f["broken_rate"] = (touched - f["up_count"]) / touched * 100
    return f


# ── the limit-up store ───────────────────────────────────────────────────────
_mem: dict[str, dict] = {}
_lock = threading.Lock()
_filling = False
_warned = False


def _db():
    import data_manager
    return data_manager.db


def load_limits(db=None) -> dict[str, dict]:
    """Every stored session, merged with whatever this process fetched."""
    global _warned
    out: dict[str, dict] = {}
    try:
        rows = (db or _db()).read_table(TABLE)
        if rows is not None and not rows.empty:
            for r in rows.to_dict("records"):
                out[str(r["trade_date"])] = {k: r[k] for k in
                                             ("up_count", "down_count", "touched", "total")}
    except Exception as exc:                                       # noqa: BLE001
        if not _warned:
            _warned = True
            print(f"[market_risk] limit stats in memory only — {type(exc).__name__}: "
                  f"{exc}. Run supabase/migrations/20261027_market_limit_stats.sql.")
    with _lock:
        out.update(_mem)
    return out


def save_limits(rows: dict[str, dict], db=None) -> None:
    if not rows:
        return
    with _lock:
        _mem.update(rows)
    try:
        (db or _db()).insert_records(
            TABLE, [{"trade_date": d, **v} for d, v in rows.items()], upsert=True)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[market_risk] limit stats not stored — {type(exc).__name__}: {exc}")


def fetch_limit_day(pro, trade_date: str) -> dict | None:
    """One session, or None if it is not published yet or the call failed."""
    try:
        daily = pro.daily(trade_date=trade_date, fields="ts_code,close,high")
        limits = pro.stk_limit(trade_date=trade_date)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[market_risk] {trade_date}: {type(exc).__name__}: {exc}")
        return None
    return limit_stats(daily, limits)


def fill(pro, dates: list[str], db=None) -> dict[str, dict]:
    """Fetch these sessions and store the ones that came back."""
    if not dates:
        return {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        got = dict(zip(dates, pool.map(lambda d: fetch_limit_day(pro, d), dates)))
    # An unpublished session is NOT stored as empty: it would never be
    # fetched again, and the tile would read zero 涨停 for it forever.
    ok = {d: v for d, v in got.items() if v is not None}
    save_limits(ok, db)
    return ok


def _background(pro, dates: list[str], db) -> None:
    global _filling
    try:
        for i in range(0, len(dates), 20):
            fill(pro, dates[i:i + 20], db)
    finally:
        _filling = False


def limit_sessions(pro, open_dates: list[str], db=None, *,
                   sync_budget: int = SYNC_BUDGET,
                   background: bool = True) -> tuple[dict[str, dict], int]:
    """
    Stored sessions after topping up the newest, and how many are still missing.

    The newest `sync_budget` gaps are fetched now so the tile is current; the
    rest go to one background thread, newest first, so the percentile's
    sample grows from today backwards.
    """
    global _filling
    stored = load_limits(db)
    missing = [d for d in sorted(open_dates, reverse=True) if d not in stored]
    stored.update(fill(pro, missing[:sync_budget], db))
    rest = [d for d in missing[sync_budget:] if d not in stored]
    if rest and background:
        with _lock:
            start = not _filling
            _filling = True
        if start:
            threading.Thread(target=_background, args=(pro, rest, db),
                             daemon=True, name="limit-backfill").start()
    return stored, len(rest)


# ── the board ────────────────────────────────────────────────────────────────
def _extra(label: str, value, unit: str = "") -> dict:
    return {"label": label, "value": value, "unit": unit}


def turnover_tile(amt: pd.Series) -> dict | None:
    ma20 = amt.rolling(20).mean()
    last, avg = amt.iloc[-1], ma20.iloc[-1]
    vs = (last / avg - 1) * 100 if np.isfinite(avg) and avg else None
    return tile("turnover", "成交额", "亿", amt, nd=0,
                note="沪深两市股票成交额（daily_info · SH_MARKET + SZ_MARKET）。"
                     "流动性是一切风险的底：成交萎缩时，同样的卖盘砸出更大的跌幅。",
                extra=[_extra("20 日均", _num(avg, 0), "亿"),
                       _extra("较均值", _num(vs, 1), "%")])


def margin_tile(share: pd.Series, buy: pd.Series) -> dict | None:
    return tile("margin", "两融占成交", "%", share, nd=2,
                note="融资买入额 ÷ 成交额（沪深两所）。比两融余额更直接：余额是存量，"
                     "这一比例说明今天的买盘里有多少是借来的钱。",
                extra=[_extra("融资买入", _num(buy.iloc[-1], 0) if len(buy) else None, "亿")])


def limits_tile(f: pd.DataFrame, missing: int) -> dict | None:
    t = tile("limits", "涨停家数", "家", f["up_count"], nd=0,
             note="收盘封住涨停的个股数，由 daily 对照 stk_limit 计算。"
                  "炸板率 = 盘中触及涨停但没封住的比例；跌停家数高于常态是恐慌信号。",
             extra=[_extra("跌停", _num(f["down_count"].iloc[-1], 0), "家"),
                    _extra("炸板率", _num(f["broken_rate"].iloc[-1], 1), "%")])
    if t is None:
        return None
    # Down-limits and failed seals override the headline grade: a day of 80
    # 涨停 and 60 跌停 is not "正常".
    down_pct, _ = percentile(f["down_count"])
    broken_pct, _ = percentile(f["broken_rate"].dropna())
    if down_pct is not None and down_pct >= 90:
        t["state"], t["level"] = "恐慌", 2
    elif broken_pct is not None and broken_pct >= 90:
        t["state"], t["level"] = "分歧", max(t["level"], 1)
    t["backfill"] = missing
    return t


def valuation_tile(ib: pd.DataFrame) -> dict | None:
    d = ib.set_index("trade_date").sort_index()
    pe = pd.to_numeric(d["pe_ttm"], errors="coerce")
    pb = pd.to_numeric(d["pb"], errors="coerce")
    ey = 100 / pe.iloc[-1] if pe.iloc[-1] else None
    return tile("valuation", "沪深300 PE", "倍", pe, nd=2,
                note="沪深300 滚动市盈率（TTM）。股债性价比（ERP）需要中债国债收益率"
                     "（yc_cb），当前 Tushare 权限不含，暂用估值分位代替。",
                extra=[_extra("PB", _num(pb.iloc[-1], 2), "倍"),
                       _extra("盈利收益率", _num(ey, 2), "%")])


def vol_tile(close: pd.Series) -> dict | None:
    v20 = realized_vol(close, 20)
    v60 = realized_vol(close, 60)
    return tile("vol", "沪深300 波动", "%", v20, nd=1,
                note="20 日已实现波动率（年化）。波动放大时，同样的仓位承担更大的风险，"
                     "而且波动倾向于聚集。",
                extra=[_extra("60 日", _num(v60.iloc[-1], 1) if len(v60) else None, "%")])


def board(pro=None, db=None) -> dict:
    """Every tile that could be built, and the ones that could not, named."""
    pro = pro or _api()
    today = _today()
    tiles: list[dict] = []
    missing: list[str] = []

    def attempt(label: str, build):
        try:
            t = build()
        except Exception as exc:                                   # noqa: BLE001
            print(f"[market_risk] {label}: {type(exc).__name__}: {exc}")
            t = None
        if t is None:
            missing.append(label)
        else:
            tiles.append(t)

    # 成交额 feeds two tiles; read it once.
    try:
        info = pd.concat([pro.daily_info(ts_code=c, start_date=WARMUP_START, end_date=today,
                                         fields="trade_date,ts_code,amount")
                          for c in TURNOVER_CODES])
    except Exception as exc:                                       # noqa: BLE001
        print(f"[market_risk] daily_info: {type(exc).__name__}: {exc}")
        info = None
    amt = turnover_series(info)

    attempt("成交额", lambda: turnover_tile(amt) if len(amt) else None)

    def margin():
        buy = margin_buy_series(pro.margin(start_date=WARMUP_START, end_date=today))
        share = margin_share(buy, amt)
        return margin_tile(share, buy) if len(share) else None
    attempt("两融占成交", margin)

    backfill = {"have": 0, "need": 0}

    def limits():
        cal = pro.trade_cal(exchange="SSE", start_date=ANCHOR, end_date=today, is_open="1")
        dates = sorted(str(d) for d in cal["cal_date"])
        rows, left = limit_sessions(pro, dates, db)
        rows = {d: v for d, v in rows.items() if d >= ANCHOR}
        backfill.update(have=len(rows), need=len(rows) + left)
        f = limit_frame(rows)
        return limits_tile(f, left) if len(f) else None
    attempt("涨停家数", limits)

    attempt("估值", lambda: valuation_tile(
        pro.index_dailybasic(ts_code=INDEX, start_date=ANCHOR, end_date=today,
                             fields="trade_date,pe_ttm,pb")))

    def vol():
        px = pro.index_daily(ts_code=INDEX, start_date=WARMUP_START, end_date=today,
                             fields="trade_date,close")
        return vol_tile(px.set_index("trade_date")["close"]) if px is not None and len(px) else None
    attempt("波动", vol)

    order = list(GRADES)
    tiles.sort(key=lambda t: order.index(t["key"]))
    return {
        "tiles": tiles, "missing": missing,
        "anchor": _iso(ANCHOR), "backfill": backfill,
        "flags": sum(1 for t in tiles if t["level"] > 0),
        "as_of": datetime.now(CHINA_TZ).strftime("%Y-%m-%d %H:%M"),
    }


if __name__ == "__main__":
    # One-off backfill of every session since the anchor, synchronously.
    p = _api()
    cal = p.trade_cal(exchange="SSE", start_date=ANCHOR, end_date=_today(), is_open="1")
    have = load_limits()
    todo = [d for d in sorted(map(str, cal["cal_date"]), reverse=True) if d not in have]
    print(f"[market_risk] {len(have)} stored, {len(todo)} to fetch")
    for i in range(0, len(todo), 20):
        got = fill(p, todo[i:i + 20])
        print(f"[market_risk] {min(i + 20, len(todo))}/{len(todo)} · stored {len(got)}")
