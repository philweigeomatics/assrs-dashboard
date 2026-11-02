"""
leverage_cycle.py — 💳 杠杆周期: where the A-share leverage cycle stands.

Not a buy or sell call. Four layers of fixed, numeric rules, each one either
holding today or not, and a stage that is nothing more than which of them
hold:

  ① 杠杆存量下降   is borrowed money leaving?
  ② 杠杆浓度低位   is what is left small — against the market, and as a
                   share of the buying?
  ③ 底背离         has the index made a new low that the debt did not?
  ④ 配置资金承接   is anyone buying the bottom — broad ETFs, the worst
                   sectors refusing new lows, volatility calming?

  杠杆出清   ① holds (two of its three rules)
  底部区域   ② holds and ③ has begun (a higher low, or the debt flat while
             the index still falls) — within 120 sessions of 杠杆出清
  右侧确认   the debt turns up after the index stops falling, with ④ —
             within 120 sessions of 底部区域, so a bull market's rising
             debt is not read as a bottom being confirmed

WHY THESE PAIRINGS

融资余额 is a debt, not the market value of the positions: falling prices do
not shrink it, only repayment and forced selling do. Against the index that
is what makes a divergence mean something — a new index low without a new
debt low says the forced sellers are already out. Against 流通市值 it is
concentration, with a trap: in a crash the ratio RISES while people
deleverage, because the denominator falls faster — so ① reads the balance
itself as well as the ratio. Against 成交额, 融资买入额 is how much of each
day's buying is borrowed: the "冰点" reading.

WHICH PAST

Ratios mean the same in any regime, and "历史低位" needs whole cycles — the
market since 9·24 has had one — so both ratios and volatility are ranked
against every session since 2014 (margin was a pilot before), using only
what was known on each day: the stage history is computed without
hindsight. Readings are
whole-market: 中证全指 (000985) for the index, SSE + SZSE for 两融, and
Shanghai + Shenzhen for 流通市值 and 成交额.

STORAGE

The raw daily series are kept in market_cycle_day (one row per session);
the rules are computed from them on read. Each stage's past entries and
what followed them are shown beside it — the rules are fixed, not fitted
to that record, so it is a test rather than an advertisement. The daily job appends new
sessions and re-reads the last few, since margin data can publish a day
late. Run supabase/migrations/20261102_market_cycle_day.sql, then
`python leverage_cycle.py` once to backfill from 2010.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CHINA_TZ = ZoneInfo("Asia/Shanghai")
TABLE = "market_cycle_day"
MIGRATION = "supabase/migrations/20261102_market_cycle_day.sql"
START = "20100101"
INDEX = "000985.CSI"                 # 中证全指: the whole market, like 两融
EXCHANGES = ("SSE", "SZSE")
#: Shanghai's whole-market row starts in 2019; before it SH_A is the same
#: market (0.36% apart on the first common day — B shares) with no 科创板 yet.
SH_SPLICE = "20190102"
#: Broad-index ETFs: where allocation money — the "national team" in
#: 2024–25 above all — goes when it buys the market rather than a story.
ETFS = {
    "510050.SH": "上证50", "510300.SH": "沪深300", "510310.SH": "沪深300", "510330.SH": "沪深300",
    "159919.SZ": "沪深300", "510500.SH": "中证500", "512510.SH": "中证500", "512100.SH": "中证1000",
    "159845.SZ": "中证1000", "563300.SH": "中证2000", "159915.SZ": "创业板", "588000.SH": "科创50",
    "588080.SH": "科创50", "159338.SZ": "中证A500", "563800.SH": "中证A500",
}
#: Sessions re-read on every sync: margin data can arrive a day late, and an
#: ETF's flow needs the share count of the session before it.
REFETCH_DAYS = 20
MIN_RANK = 500                       # sessions of history before a percentile means anything
#: Margin trading was a pilot until the 2013–14 expansions — 0.0–1.7% of the
#: market against ~2% since. Percentiles start here, so "历史低位" is not
#: measured against a market that barely had margin accounts.
RANK_FROM = "2014-01-01"
REVISIT = 20                         # sessions apart before a stage counts as entered again
RIGHT_WINDOW = 120                   # each stage needs the one before it this recently


# ── fetching ─────────────────────────────────────────────────────────────────
def _api():
    import watchlist_board
    return watchlist_board._api()


def _stack(parts) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["trade_date"])


def _years(start: str, end: str, step: int):
    y0, y1 = int(start[:4]), int(end[:4])
    for y in range(y0, y1 + 1, step):
        a = max(start, f"{y}0101")
        b = min(end, f"{y + step - 1}1231")
        if a <= b:
            yield a, b


def _call(fn, **kw):
    for attempt in range(3):
        try:
            return fn(**kw)
        except Exception as exc:                                   # noqa: BLE001
            if attempt == 2:
                raise
            print(f"[cycle] retry {getattr(fn, '__name__', fn)}: {type(exc).__name__}"[:120])
            time.sleep(2.0 * (attempt + 1))


def fetch_margin(pro, start: str, end: str) -> pd.DataFrame:
    """融资余额 / 买入 / 偿还 and 融券余额, 亿元, on sessions where both exchanges reported."""
    parts = [_call(pro.margin, start_date=a, end_date=b,
                   fields="trade_date,exchange_id,rzye,rzmre,rzche,rqye")
             for a, b in _years(start, end, 4)]          # ≤ 4,000 rows a call
    df = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True) if parts else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=["rzye", "rzmre", "rzche", "rqye"])
    df = df[df["exchange_id"].isin(EXCHANGES)]
    out = {}
    for c in ("rzye", "rzmre", "rzche", "rqye"):
        piv = df.pivot_table(index="trade_date", columns="exchange_id",
                             values=c, aggfunc="sum").reindex(columns=list(EXCHANGES))
        out[c] = piv.dropna().sum(axis=1) / 1e8 if c != "rqye" else piv.fillna(0).sum(axis=1) / 1e8
    f = pd.DataFrame(out).dropna(subset=["rzye", "rzmre", "rzche"])
    return f.sort_index()


def fetch_market(pro, start: str, end: str) -> pd.DataFrame:
    """Whole-market 流通市值 and 成交额 (亿元), Shanghai + Shenzhen, sessions both reported."""
    def get(code, a, b):
        parts = [_call(pro.daily_info, ts_code=code, start_date=x, end_date=y,
                       fields="trade_date,float_mv,amount") for x, y in _years(a, b, 6)]
        df = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True) if parts else pd.DataFrame()
        if df.empty:
            return pd.DataFrame(columns=["float_mv", "amount"])
        df = df.drop_duplicates("trade_date").set_index("trade_date").sort_index()
        return df[["float_mv", "amount"]].apply(pd.to_numeric, errors="coerce")
    sh = []
    if start < SH_SPLICE:
        sh.append(get("SH_A", start, min(end, "20181231")))
    if end >= SH_SPLICE:
        sh.append(get("SH_MARKET", max(start, SH_SPLICE), end))
    sh = pd.concat(sh).sort_index()
    sz = get("SZ_MARKET", start, end)
    both = sh.join(sz, lsuffix="_sh", rsuffix="_sz", how="inner").dropna()
    return pd.DataFrame({"float_mv": both["float_mv_sh"] + both["float_mv_sz"],
                         "amount": both["amount_sh"] + both["amount_sz"]})


def fetch_index(pro, start: str, end: str) -> pd.Series:
    parts = [_call(pro.index_daily, ts_code=INDEX, start_date=a, end_date=b, fields="trade_date,close")
             for a, b in _years(start, end, 8)]
    df = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True)
    return df.drop_duplicates("trade_date").set_index("trade_date")["close"].astype(float).sort_index()


def etf_flows(shares: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """
    Daily net creation per ETF in 亿元: the change in units (万份) × that
    day's price. A share split or merge changes units without money moving —
    units jump while units × price stays put — and is not a flow.
    """
    s = shares.sort_index()
    p = closes.reindex(index=s.index, columns=s.columns).ffill()
    rs = s / s.shift(1)
    rp = p / p.shift(1)
    split = ((rs - 1).abs() > 0.2) & ((rs * rp - 1).abs() < 0.05)
    flow = (s - s.shift(1)) * p / 1e4                # 万份 × 元 = 万元 → 亿元
    return flow.mask(split, 0.0)


def fetch_etfs(pro, start: str, end: str, dates: list[str] | None = None) -> pd.DataFrame:
    """
    Units outstanding and closing prices for ETFS. A long span is read per
    fund (two calls each); a short one per day, where one call returns every
    fund — cheaper for the daily top-up.
    """
    codes = list(ETFS)
    if dates is not None and len(dates) <= 40:
        sh, px = {}, {}
        for d in dates:
            a = _call(pro.fund_share, trade_date=d, fields="ts_code,trade_date,fd_share")
            b = _call(pro.fund_daily, trade_date=d, fields="ts_code,trade_date,close")
            if a is not None and len(a):
                sh[d] = a[a["ts_code"].isin(codes)].set_index("ts_code")["fd_share"]
            if b is not None and len(b):
                px[d] = b[b["ts_code"].isin(codes)].set_index("ts_code")["close"]
        shares, closes = pd.DataFrame(sh).T, pd.DataFrame(px).T
    else:
        sh, px = {}, {}
        for c in codes:
            a = _stack(_call(pro.fund_share, ts_code=c, start_date=x, end_date=y, fields="trade_date,fd_share")
                       for x, y in _years(start, end, 6))
            b = _stack(_call(pro.fund_daily, ts_code=c, start_date=x, end_date=y, fields="trade_date,close")
                       for x, y in _years(start, end, 8))
            if len(a):
                sh[c] = a.drop_duplicates("trade_date").set_index("trade_date")["fd_share"]
            if len(b):
                px[c] = b.drop_duplicates("trade_date").set_index("trade_date")["close"]
            time.sleep(0.2)
        shares, closes = pd.DataFrame(sh), pd.DataFrame(px)
    if shares.empty:
        return pd.DataFrame()
    shares = shares.apply(pd.to_numeric, errors="coerce").sort_index()
    closes = closes.apply(pd.to_numeric, errors="coerce").sort_index()
    return etf_flows(shares, closes)


def fetch(pro, start: str, end: str) -> list[dict]:
    """Every session from start to end with all core series, as rows for the table."""
    m = fetch_margin(pro, start, end)
    k = fetch_market(pro, start, end)
    p = fetch_index(pro, start, end)
    core = m.join(k, how="inner").join(p.rename("idx_close"), how="inner").dropna(subset=["rzye", "float_mv", "idx_close"])
    flows = fetch_etfs(pro, start, end, dates=list(core.index) if len(core) <= 40 else None)
    rows = []
    for d, r in core.iterrows():
        f = flows.loc[d].dropna() if len(flows) and d in flows.index else pd.Series(dtype=float)
        rows.append({
            "d": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            **{c: _num(r[c], 4) for c in ("rzye", "rzmre", "rzche", "rqye", "float_mv", "amount", "idx_close")},
            "etf_flow": _num(f.sum(), 4) if len(f) else None,
            "etf_detail": {c: _num(v, 4) for c, v in f.items()} if len(f) else None,
        })
    # A flow needs the session before it: the first fetched day has none.
    if rows and start > START:
        rows = rows[1:]
    return rows


def _num(v, nd: int):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


# ── storage ──────────────────────────────────────────────────────────────────
def _db():
    import data_manager
    return data_manager.db


def load(db=None) -> pd.DataFrame:
    db = db or _db()
    try:
        df = db.read_all(TABLE, order_by="d")
    except Exception as exc:                                       # noqa: BLE001
        if TABLE in str(exc):
            raise RuntimeError(f"杠杆周期的表尚未创建 — 请在 Supabase SQL 编辑器运行 {MIGRATION}") from exc
        raise
    if df is None or df.empty:
        return pd.DataFrame()
    df["d"] = pd.to_datetime(df["d"].astype(str).str[:10])
    return df.set_index("d").sort_index()


def save(rows: list[dict], db=None) -> None:
    db = db or _db()
    for i in range(0, len(rows), 500):
        db.insert_records(TABLE, rows[i:i + 500], upsert=True)


def sync(pro=None, db=None) -> dict:
    """Backfill from 2010 when empty; otherwise re-read the last REFETCH_DAYS sessions and add the new ones."""
    pro = pro or _api()
    db = db or _db()
    last = db.read_table(TABLE, columns="d", order_by="-d", limit=1)
    end = datetime.now(CHINA_TZ).strftime("%Y%m%d")
    if last is None or last.empty:
        start = START
    else:
        d0 = pd.Timestamp(str(last.iloc[0]["d"])[:10])
        start = (d0 - timedelta(days=REFETCH_DAYS * 7 // 5 + 7)).strftime("%Y%m%d")
    rows = fetch(pro, start, end)
    save(rows, db)
    return {"from": start, "rows": len(rows), "last": rows[-1]["d"] if rows else None}


# ── the rules ────────────────────────────────────────────────────────────────
def _rank(s: pd.Series) -> pd.Series:
    """Each day's percentile (0–1) among every day up to and including it — no hindsight."""
    return s.loc[RANK_FROM:].expanding(min_periods=MIN_RANK).rank(pct=True).reindex(s.index)


def _since_low(p: pd.Series, w: int) -> pd.Series:
    """Sessions since the price was at its w-session low."""
    out = np.full(len(p), np.nan)
    v = p.to_numpy()
    for i in range(w - 1, len(v)):
        win = v[i - w + 1:i + 1]
        out[i] = w - 1 - int(np.nanargmin(win))
    return pd.Series(out, p.index)


def _themes_hold(themes: pd.DataFrame | None, index: pd.DatetimeIndex) -> pd.Series:
    """
    ④b on every day: the five theme indices that fell most over the 60
    sessions ending 20 sessions ago — the earlier leaders down — and whether
    at least four of them have stopped making new 60-session lows in the
    last 10 sessions.
    """
    out = pd.Series(np.nan, index)
    if themes is None or themes.empty:
        return out
    t = themes.reindex(index).ffill()
    ret = t.shift(20) / t.shift(80) - 1
    low = t.rolling(60, min_periods=40).min()
    fresh = (t <= low).rolling(10).max()                    # a new 60-day low in the last 10
    for i, d in enumerate(index):
        r = ret.loc[d].dropna()
        if len(r) < 10:
            continue
        worst = r.nsmallest(5).index
        f = fresh.loc[d, worst]
        if f.isna().any():
            continue
        out.iloc[i] = float((f == 0).sum() >= 4)
    return out


def signals(df: pd.DataFrame, themes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every rule as a 1/0 series (NaN where its data does not reach), plus the readings behind them."""
    B, P = df["rzye"], df["idx_close"]
    R1 = df["rzye"] / df["float_mv"] * 100
    R2 = (df["rzmre"] / df["amount"] * 100).rolling(5).mean()
    E = df["etf_flow"] if "etf_flow" in df else pd.Series(np.nan, df.index)
    vol = np.log(P).diff().rolling(20).std() * np.sqrt(252) * 100
    s = pd.DataFrame(index=df.index)
    s["R1"], s["R2"], s["vol"], s["E"] = R1, R2, vol, E
    s["R1_rank"], s["R2_rank"], s["vol_rank"] = _rank(R1), _rank(R2), _rank(vol)

    def rule(cond, *need):
        ok = pd.concat(need, axis=1).notna().all(axis=1)
        return cond.astype(float).where(ok)

    # ① 杠杆存量下降
    s["1a"] = rule((B < B.shift(20)) & (B < B.shift(5)), B.shift(20))
    hi = B.rolling(250, min_periods=120).max()
    s["1b"] = rule(B <= 0.9 * hi, hi)
    s["1c"] = rule(R1 < R1.shift(60), R1.shift(60))
    # ② 杠杆浓度低位
    s["2a"] = rule(s["R1_rank"] <= 0.2, s["R1_rank"])
    s["2b"] = rule(s["R2_rank"] <= 0.2, s["R2_rank"])
    peak = R2.rolling(250, min_periods=120).max()
    s["2c"] = rule(R2 <= 0.7 * peak, peak)
    # ③ 底背离
    # Judged on the index's most recent 60-day low (within 20 sessions): on
    # that day, was the debt above its own 60-day low? An earlier divergence
    # undone by a later joint low does not count.
    low60 = P.rolling(60).min()
    at_low = (P <= low60) & low60.notna()
    verdict = (B > B.rolling(60).min() * 1.001).astype(float).where(at_low)
    s["3a"] = rule(verdict.ffill(limit=19) == 1, low60.shift(20))
    s["3b"] = rule((P < P.shift(10)) & (B / B.shift(10) - 1 > -0.005), B.shift(10))
    calm = (_since_low(P, 20) >= 5) & (P >= 1.02 * P.rolling(20).min())
    s["3c"] = rule(calm & (B > B.shift(10)) & (B.shift(10) < B.shift(60)), B.shift(60))
    # ④ 配置资金承接
    s["4a"] = rule((E.rolling(20).sum() > 0) & ((E > 0).rolling(20).sum() >= 12), E.rolling(20, min_periods=20).sum())
    s["4b"] = _themes_hold(themes, df.index)
    q80 = vol.loc[RANK_FROM:].expanding(min_periods=MIN_RANK).quantile(0.8).reindex(vol.index)
    vpeak = vol.rolling(60).max()
    s["4c"] = rule((vpeak >= q80) & (vol <= 0.75 * vpeak), q80)

    def count(keys):
        sub = s[list(keys)]
        return sub.sum(axis=1, min_count=1), sub.notna().sum(axis=1)

    s["L1"], s["n1"] = count(("1a", "1b", "1c"))
    s["L2"], s["n2"] = count(("2a", "2b", "2c"))
    s["L3"], s["n3"] = count(("3a", "3b", "3c"))
    s["L4"], s["n4"] = count(("4a", "4b", "4c"))
    # The stages are a sequence: a bottom is only a bottom after a clear-out,
    # and a confirmation only confirms a bottom. Each needs the one before it
    # within RIGHT_WINDOW sessions (or holding today).
    def lately(x):
        return x.astype(float).rolling(RIGHT_WINDOW, min_periods=1).max() > 0
    clearing = s["L1"] >= 2
    bottom = (s["L2"] >= 2) & ((s["3a"] == 1) | (s["3b"] == 1)) & lately(clearing)
    # ④ holds on two thirds of whatever of it the data reaches (≥ 1).
    l4 = s["L4"] >= np.maximum(1, np.ceil(s["n4"] * 2 / 3))
    right = (s["3c"] == 1) & l4 & (s["n4"] > 0) & lately(bottom)
    # Each condition is kept; the label shows the furthest one that holds
    # (a bottom can form while the clear-out is still going on).
    s["is_clearing"], s["is_bottom"], s["is_right"] = clearing, bottom, right
    s["stage"] = np.select([right, bottom, clearing], ["right", "bottom", "clearing"], default="none")
    return s


STAGES = {"clearing": "杠杆出清", "bottom": "底部区域", "right": "右侧确认", "none": "未进入出清"}

RULES = [
    ("1", "杠杆存量下降", [
        ("1a", "融资余额持续下降", "比 20 个交易日前低，也比 5 个交易日前低"),
        ("1b", "从高位明显回落", "比一年内最高点低 10% 以上"),
        ("1c", "融资余额/流通市值下降", "比 60 个交易日前低（暴跌时这个比例会因市值缩水而上升，所以和余额本身一起看）"),
    ]),
    ("2", "杠杆浓度低位", [
        ("2a", "融资余额/流通市值接近历史低位", "在 2014 年以来的分位 ≤ 20%"),
        ("2b", "融资买入/成交额进入冰点", "5 日均值在 2014 年以来的分位 ≤ 20%"),
        ("2c", "两融买入占比远低于前期高点", "5 日均值不到一年内高点的 70%"),
    ]),
    ("3", "底背离", [
        ("3a", "指数新低，融资余额不新低", "看近 20 个交易日内最近一次指数 60 日新低：那一天融资余额高于它自己的 60 日低点"),
        ("3b", "指数续跌，融资余额走平", "指数 10 日下跌，融资余额 10 日变化不低于 −0.5%"),
        ("3c", "指数止跌，融资余额由降转升", "指数离开 20 日低点 ≥ 5 天且高出 2%，融资余额 10 日上升而此前 50 日下降"),
    ]),
    ("4", "配置资金承接", [
        ("4a", "宽基 ETF 持续净流入", "20 日累计净申购为正，且其中至少 12 天为正"),
        ("4b", "前期领跌板块不再新低", "此前跌得最多的 5 个主题指数中，至少 4 个近 10 日没有再创 60 日新低"),
        ("4c", "波动率从极端高位回落", "60 日内的波动高点达到历史 80 分位，现在已回落到它的 75% 以下"),
    ]),
]


def _leaders_text(themes: pd.DataFrame | None, index: pd.DatetimeIndex, i: int) -> str | None:
    """Which theme indices led the fall, and which of them are still making new lows."""
    if themes is None or themes.empty:
        return "主题指数数据暂缺"
    import cn_index
    names = {c: n for c, n, _ in cn_index.UNIVERSE}
    t = themes.reindex(index).ffill()
    d = index[i]
    r = (t.shift(20) / t.shift(80) - 1).loc[d].dropna()
    if len(r) < 10:
        return None
    low = t.rolling(60, min_periods=40).min()
    fresh = (t <= low).rolling(10).max().loc[d]
    worst = r.nsmallest(5).index
    parts = [f"{names.get(c, c)}{'↓' if fresh[c] == 1 else '✓'}" for c in worst]
    return f"{int((fresh[worst] == 0).sum())}/5 不再新低：" + " ".join(parts)


def _value(key: str, df: pd.DataFrame, s: pd.DataFrame, i: int, themes=None) -> str | None:
    """The reading behind a rule on day i, as text."""
    B, P = df["rzye"], df["idx_close"]

    def at(x, k=0):
        j = i - k
        return x.iloc[j] if 0 <= j < len(x) and pd.notna(x.iloc[j]) else None

    def pct(v):
        return f"{v * 100:.0f}%" if v is not None else "—"
    b, b20, b5, b10 = at(B), at(B, 20), at(B, 5), at(B, 10)
    if key == "1a" and None not in (b, b20, b5):
        return f"20 日 {b - b20:+,.0f} 亿 · 5 日 {b - b5:+,.0f} 亿"
    if key == "1b":
        hi = at(B.rolling(250, min_periods=120).max())
        return f"较一年高点 {(b / hi - 1) * 100:+.1f}%" if b and hi else None
    if key == "1c" and at(s["R1"]) is not None and at(s["R1"], 60) is not None:
        return f"{at(s['R1']):.2f}%（60 日前 {at(s['R1'], 60):.2f}%）"
    if key == "2a" and at(s["R1"]) is not None:
        return f"{at(s['R1']):.2f}% · 2014 年以来分位 {pct(at(s['R1_rank']))}"
    if key == "2b" and at(s["R2"]) is not None:
        return f"{at(s['R2']):.2f}% · 2014 年以来分位 {pct(at(s['R2_rank']))}"
    if key == "2c":
        peak = at(s["R2"].rolling(250, min_periods=120).max())
        return f"{at(s['R2']):.2f}% / 一年高点 {peak:.2f}%" if peak and at(s["R2"]) is not None else None
    if key == "3a":
        low60 = P.rolling(60).min()
        hit = [j for j in range(max(0, i - 19), i + 1) if P.iloc[j] <= low60.iloc[j]]
        if not hit:
            return "近 20 日指数未创 60 日新低"
        j = hit[-1]                                  # the latest low is the one judged
        bl = B.rolling(60).min().iloc[j]
        return f"指数 {P.index[j].date()} 创 60 日新低；当日融资余额高于自身 60 日低点 {(B.iloc[j] / bl - 1) * 100:+.1f}%"
    if key == "4b":
        return _leaders_text(themes, df.index, i)
    if key == "3b" and None not in (b, b10):
        p, p10 = at(P), at(P, 10)
        return f"指数 10 日 {(p / p10 - 1) * 100:+.1f}% · 融资 10 日 {(b / b10 - 1) * 100:+.2f}%"
    if key == "3c" and None not in (b, b10):
        low = at(P.rolling(20).min())
        return f"离 20 日低点 {(at(P) / low - 1) * 100:+.1f}% · 融资 10 日 {(b / b10 - 1) * 100:+.2f}%"
    if key == "4a":
        e = s["E"].iloc[max(0, i - 19):i + 1]
        return f"20 日 {e.sum():+,.0f} 亿 · {int((e > 0).sum())} 天流入" if e.notna().sum() >= 20 else "近 20 日份额数据不全"
    if key == "4c" and at(s["vol"]) is not None:
        vp = at(s["vol"].rolling(60).max())
        return f"波动 {at(s['vol']):.0f}%（60 日高点 {vp:.0f}% · 分位 {pct(at(s['vol_rank']))}）"
    return None


def _since(col: pd.Series) -> str | None:
    """The first day of the current unbroken run of 1s."""
    v = col.to_numpy()
    if not len(v) or v[-1] != 1:
        return None
    j = len(v) - 1
    while j > 0 and v[j - 1] == 1:
        j -= 1
    return str(col.index[j].date())


def episodes(s: pd.DataFrame, P: pd.Series) -> dict:
    """
    Each time a stage was entered (after REVISIT sessions without it), and
    what the index did over the next 63 and 126 sessions — next to the same
    for every day, so a stage's record can be judged against doing nothing.
    """
    p = P.to_numpy()
    n = len(p)

    def fwd(i, k):
        return (p[i + k] / p[i] - 1) * 100 if i + k < n else None
    out = {}
    stage = s["stage"].to_numpy()
    for key in ("clearing", "bottom", "right"):
        hits, last = [], -10 ** 9
        for i in np.flatnonzero(stage == key):
            if i - last > REVISIT:
                hits.append(i)
            last = i
        rows = [{"date": str(s.index[i].date()), "r63": _num(fwd(i, 63), 1), "r126": _num(fwd(i, 126), 1)}
                for i in hits]
        done = [r for r in rows if r["r63"] is not None]
        out[key] = {
            "n": len(rows), "list": rows[-12:][::-1],
            "median63": _num(np.median([r["r63"] for r in done]), 1) if done else None,
            "up63": _num(np.mean([r["r63"] > 0 for r in done]), 2) if done else None,
            "median126": _num(np.median([r["r126"] for r in done if r["r126"] is not None]), 1)
            if any(r["r126"] is not None for r in done) else None,
        }
    every63 = [(p[i + 63] / p[i] - 1) * 100 for i in range(n - 63)]
    every126 = [(p[i + 126] / p[i] - 1) * 100 for i in range(n - 126)]
    out["base"] = {"median63": _num(np.median(every63), 1) if every63 else None,
                   "up63": _num(np.mean(np.array(every63) > 0), 2) if every63 else None,
                   "median126": _num(np.median(every126), 1) if every126 else None}
    return out


def board(df: pd.DataFrame, themes: pd.DataFrame | None = None) -> dict:
    """The panel: today's stage and every rule behind it, the series to draw, and the stage's track record."""
    if df.empty:
        raise LookupError("杠杆周期的数据还没有 — 请运行 python leverage_cycle.py 回填")
    if themes is not None and len(themes) and not isinstance(themes.index, pd.DatetimeIndex):
        themes = themes.copy()
        themes.index = pd.to_datetime(themes.index.astype(str))
    s = signals(df, themes)
    i = len(df) - 1
    layers = []
    for lk, title, rules in RULES:
        items = []
        for key, label, rule in rules:
            v = s[key].iloc[i]
            items.append({"key": key, "label": label, "rule": rule,
                          "ok": None if pd.isna(v) else bool(v == 1),
                          "value": _value(key, df, s, i, themes), "since": _since(s[key].dropna())})
        known = [x for x in items if x["ok"] is not None]
        layers.append({"key": lk, "title": title, "items": items,
                       "hold": sum(x["ok"] for x in known), "of": len(known)})
    stage = s["stage"].iloc[i]
    st = s["stage"]
    run_start = i
    while run_start > 0 and st.iloc[run_start - 1] == stage:
        run_start -= 1
    def nan(x, nd):
        return [None if pd.isna(v) else round(float(v), nd) for v in x.loc[RANK_FROM:]]
    shown = df.loc[RANK_FROM:].index
    return {
        "as_of": str(df.index[-1].date()),
        "stage": stage, "stage_label": STAGES[stage], "stage_since": str(df.index[run_start].date()),
        "layers": layers,
        # From 2014, where the rankings start; the stage codes are one letter each.
        "series": {
            "dates": [str(d.date()) for d in shown],
            "index": nan(df["idx_close"], 1), "balance": nan(df["rzye"], 0),
            "ratio": nan(s["R1"], 3), "buy_share": nan(s["R2"], 2), "etf": nan(s["E"], 1),
            "stage": "".join({"none": "-", "clearing": "c", "bottom": "b", "right": "r"}[v]
                             for v in st.loc[RANK_FROM:]),
        },
        "bands": {k: {str(q): _num(s[col].loc[RANK_FROM:].dropna().quantile(q / 100), 3) for q in (10, 20, 50, 80, 90)}
                  for k, col in (("ratio", "R1"), ("buy_share", "R2"))},
        "episodes": episodes(s, df["idx_close"]),
        "latest": {"balance": _num(df["rzye"].iloc[-1], 0), "ratio": _num(s["R1"].iloc[-1], 3),
                   "buy_share": _num(s["R2"].iloc[-1], 2), "index": _num(df["idx_close"].iloc[-1], 2),
                   "repay": _num(df["rzche"].iloc[-1], 0), "buy": _num(df["rzmre"].iloc[-1], 0)},
    }


if __name__ == "__main__":
    t = time.time()
    print("[cycle]", sync(), f"{time.time() - t:.0f}s")
