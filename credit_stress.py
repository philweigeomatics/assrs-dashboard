"""
credit_stress.py — 🏦 信用与债务: is the debt behind the AI build-out turning
into credit stress?

US first, because that is where the borrowing is: hyperscalers, their
data-centre vehicles and the private credit behind them. Six fixed rules,
each holding or not, and a reading that is nothing more than how many hold:

  c1  信用利差快速走阔    Baa − 10y Treasury up ≥ 0.5 pp from its 3-month low
  c2  信用利差处于高位    Baa − 10y at or above its 80th percentile since 1986
  c3  私募信贷承压        BIZD (listed BDCs, the visible end of private
                          credit) ≥ 20% below its one-year high
  c4  杠杆贷款承压        BKLN (leveraged loans) ≥ 5% below its one-year high
  c5  金融条件收紧        Chicago Fed NFCI above 0, or up ≥ 0.25 in 13 weeks
  c6  银行收紧企业贷款    Fed loan-officer survey: net ≥ 20% of banks
                          tightening C&I standards

  平稳 at most one holds · 走弱 two · 紧张 three or more, and at least
  half of those the data reaches

WHY THESE SERIES

The rules use only series with long histories, so each reading carries a
record: every past entry into 紧张 and what the S&P 500 did next, beside
what any day brought. ICE's high-yield / investment-grade / BBB spreads are
better gauges but FRED now serves only their last three years — they are
drawn and read, not ruled on. Every series enters on the day it was
published, not the day it describes (the loan-officer survey dated a
quarter's first day comes out about five weeks later), so the record is
free of hindsight. Percentiles use only the history up to each day.

China is context, not a stress gauge: the corporate-bond curve is not on
this Tushare tier. Private-sector credit to GDP (BIS, via FRED), 社融存量
growth and M1/M2 show the credit cycle's direction.
"""

from __future__ import annotations

import io
import math
import urllib.request
from datetime import timedelta

import numpy as np
import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
#: series id → (label, days after its date that it is published)
FRED = {
    "BAA10Y": ("Baa − 10 年期国债", 1),
    "BAMLH0A0HYM2": ("高收益债利差", 1),
    "BAMLC0A0CM": ("投资级利差", 1),
    "BAMLC0A4CBBB": ("BBB 利差", 1),
    "NFCI": ("芝加哥联储金融条件指数", 5),
    "DRTSCILM": ("收紧企业贷款标准的银行净占比", 40),
    "BCNSDODNS": ("美国非金融企业债务", 165),
    "QCNPAM770A": ("中国私人部门信贷/GDP", 200),
    "QUSPAM770A": ("美国私人部门信贷/GDP", 200),
}
TICKERS = ["SPY", "BIZD", "BKLN", "HYG", "LQD", "ORCL", "CRWV"]
MIN_RANK = 500
REVISIT = 20
LEVELS = {"calm": "平稳", "weak": "走弱", "stress": "紧张"}

RULES = [
    ("c1", "信用利差快速走阔", "Baa 公司债与 10 年期国债利差，比 3 个月内低点高出 ≥ 0.5 个百分点"),
    ("c2", "信用利差处于高位", "Baa − 10 年期国债利差，在 1986 年以来的分位 ≥ 80%（只用当天之前的数据）"),
    ("c3", "私募信贷承压", "BIZD（上市 BDC，私募信贷看得见的一端）比一年内高点低 20% 以上"),
    ("c4", "杠杆贷款承压", "BKLN（杠杆贷款 ETF）比一年内高点低 5% 以上"),
    ("c5", "金融条件收紧", "芝加哥联储 NFCI 高于 0（比历史平均紧），或 13 周内上升 ≥ 0.25"),
    ("c6", "银行收紧企业贷款", "美联储贷款官员调查：收紧大中型企业贷款标准的银行净占比 ≥ 20%"),
]


# ── fetching ─────────────────────────────────────────────────────────────────
def fred(series_id: str) -> pd.Series:
    raw = urllib.request.urlopen(FRED_URL.format(series_id), timeout=30).read().decode()
    df = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True)
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().rename(series_id)


def published(s: pd.Series, lag_days: int) -> pd.Series:
    """Re-date a series to when it was public: a value is known `lag_days` after the date it describes."""
    out = s.copy()
    out.index = out.index + timedelta(days=lag_days)
    return out


def fetch_us() -> dict:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:
        got = dict(zip(FRED, pool.map(lambda k: _try(fred, k), FRED)))
    import yfinance as yf
    px = yf.download(" ".join(TICKERS), period="max", progress=False, auto_adjust=True, threads=True)["Close"]
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    return {"fred": got, "px": px}


def _try(fn, *a):
    try:
        return fn(*a)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[credit] {a}: {type(exc).__name__}: {exc}"[:160])
        return None


def fetch_china() -> dict:
    """社融存量 (万亿) and M1/M2 growth from Tushare — context for the credit cycle."""
    out = {}
    try:
        import watchlist_board
        pro = watchlist_board._api()
        sf = pro.sf_month(start_m="201501", end_m="209912", fields="month,stk_endval")
        if sf is not None and len(sf):
            s = sf.set_index("month")["stk_endval"].astype(float).sort_index()
            s.index = pd.to_datetime(s.index, format="%Y%m")
            out["tsf_yoy"] = (s / s.shift(12) - 1) * 100
        m = pro.cn_m(start_m="201501", end_m="209912", fields="month,m1_yoy,m2_yoy")
        if m is not None and len(m):
            m = m.set_index("month").sort_index()
            m.index = pd.to_datetime(m.index, format="%Y%m")
            out["m1_yoy"], out["m2_yoy"] = m["m1_yoy"].astype(float), m["m2_yoy"].astype(float)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[credit] china: {type(exc).__name__}: {exc}"[:160])
    return out


# ── the rules ────────────────────────────────────────────────────────────────
def frame(us: dict) -> pd.DataFrame:
    """One row per US trading day: every series as it was known that day."""
    px = us["px"]
    days = px["SPY"].dropna().index
    f = pd.DataFrame(index=days)
    for k in ("SPY", "BIZD", "BKLN", "HYG", "LQD", "ORCL", "CRWV"):
        if k in px:
            f[k] = px[k].reindex(days)
    for k, (_, lag) in FRED.items():
        s = us["fred"].get(k)
        if s is None or s.empty:
            f[k] = np.nan
            continue
        p = published(s, lag)
        f[k] = p.reindex(days.union(p.index)).ffill().reindex(days)
        # A series stops being "known" only when it ends; before it starts it is unknown.
        f.loc[f.index < p.index[0], k] = np.nan
    return f


def _off_high(s: pd.Series, days: int = 252) -> pd.Series:
    hi = s.rolling(days, min_periods=days // 2).max()
    return s / hi - 1


def signals(f: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=f.index)
    baa = f["BAA10Y"]

    def rule(cond, *need):
        ok = pd.concat(need, axis=1).notna().all(axis=1)
        return cond.astype(float).where(ok)

    low3m = baa.rolling(63, min_periods=40).min()
    s["baa_rank"] = baa.expanding(min_periods=MIN_RANK).rank(pct=True)
    s["bizd_off"], s["bkln_off"] = _off_high(f["BIZD"]), _off_high(f["BKLN"])
    s["c1"] = rule(baa - low3m >= 0.5, low3m)
    s["c2"] = rule(s["baa_rank"] >= 0.8, s["baa_rank"])
    s["c3"] = rule(s["bizd_off"] <= -0.20, s["bizd_off"])
    s["c4"] = rule(s["bkln_off"] <= -0.05, s["bkln_off"])
    nf = f["NFCI"]
    nf13 = nf.shift(63)
    s["c5"] = rule((nf > 0) | (nf - nf13 >= 0.25), nf13)
    s["c6"] = rule(f["DRTSCILM"] >= 20, f["DRTSCILM"])
    keys = [k for k, _, _ in RULES]
    s["hold"] = s[keys].sum(axis=1, min_count=1)
    s["of"] = s[keys].notna().sum(axis=1)
    stress = (s["hold"] >= 3) & (s["hold"] >= s["of"] / 2)
    weak = s["hold"] >= 2
    s["level"] = np.select([stress, weak], ["stress", "weak"], default="calm")
    return s


def _num(v, nd=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _value(key: str, f: pd.DataFrame, s: pd.DataFrame) -> str | None:
    last = lambda c: f[c].dropna().iloc[-1] if c in f and f[c].notna().any() else None
    baa = f["BAA10Y"]
    if key == "c1":
        low = baa.rolling(63, min_periods=40).min().iloc[-1]
        return f"{baa.iloc[-1]:.2f}%（3 个月低点 {low:.2f}%，{(baa.iloc[-1] - low) * 100:+.0f} 个基点）"
    if key == "c2":
        return f"{baa.iloc[-1]:.2f}% · 1986 年以来分位 {s['baa_rank'].iloc[-1] * 100:.0f}%"
    if key == "c3":
        return f"BIZD 较一年高点 {s['bizd_off'].iloc[-1] * 100:+.1f}%"
    if key == "c4":
        return f"BKLN 较一年高点 {s['bkln_off'].iloc[-1] * 100:+.1f}%"
    if key == "c5":
        nf = f["NFCI"].dropna()
        return f"NFCI {nf.iloc[-1]:+.2f}（13 周前 {nf.iloc[-64] if len(nf) > 64 else float('nan'):+.2f}）"
    if key == "c6":
        v = last("DRTSCILM")
        return f"净 {v:+.1f}% 的银行在收紧" if v is not None else None
    return None


def _since(col: pd.Series) -> str | None:
    v = col.dropna()
    if not len(v) or v.iloc[-1] != 1:
        return None
    j = len(v) - 1
    while j > 0 and v.iloc[j - 1] == 1:
        j -= 1
    return str(v.index[j].date())


def record(level: pd.Series, spy: pd.Series) -> dict:
    """Each entry into 紧张 / 走弱 and the S&P 500 over the next 63 sessions: return and worst drawdown."""
    p = spy.to_numpy()
    n = len(p)
    out = {}
    lv = level.to_numpy()
    for key in ("weak", "stress"):
        hit = (lv == "stress") if key == "stress" else np.isin(lv, ["weak", "stress"])
        starts, last = [], -10 ** 9
        for i in np.flatnonzero(hit):
            if i - last > REVISIT:
                starts.append(i)
            last = i
        rows = []
        for i in starts:
            done = i + 63 < n
            rows.append({"date": str(level.index[i].date()),
                         "r63": _num((p[i + 63] / p[i] - 1) * 100, 1) if done else None,
                         "dd63": _num((p[i:i + 64].min() / p[i] - 1) * 100, 1) if done else None})
        fin = [r for r in rows if r["r63"] is not None]
        out[key] = {"n": len(rows), "list": rows[-12:][::-1],
                    "median63": _num(np.median([r["r63"] for r in fin]), 1) if fin else None,
                    "up63": _num(np.mean([r["r63"] > 0 for r in fin]), 2) if fin else None,
                    "dd63": _num(np.median([r["dd63"] for r in fin]), 1) if fin else None}
    r63 = p[63:] / p[:-63] - 1
    dd = np.array([p[i:i + 64].min() / p[i] - 1 for i in range(n - 63)])
    out["base"] = {"median63": _num(np.median(r63) * 100, 1), "up63": _num(np.mean(r63 > 0), 2),
                   "dd63": _num(np.median(dd) * 100, 1)}
    return out


def _weekly(s: pd.Series, since: str | None = None, nd: int = 3) -> dict:
    s = s.dropna()
    if since:
        s = s.loc[since:]
    w = s.resample("W-FRI").last().dropna()
    return {"dates": [str(d.date()) for d in w.index], "v": [_num(v, nd) for v in w.to_numpy()]}


def board(us: dict, china: dict | None = None) -> dict:
    f = frame(us)
    if f["BAA10Y"].dropna().empty:
        raise RuntimeError("FRED 信用利差暂时读取不到 — 请稍后重试")
    s = signals(f)
    rules = []
    for key, label, rule in RULES:
        v = s[key].iloc[-1]
        rules.append({"key": key, "label": label, "rule": rule, "ok": None if pd.isna(v) else bool(v == 1),
                      "value": _value(key, f, s), "since": _since(s[key])})
    lvl = s["level"].iloc[-1]
    j = len(s) - 1
    while j > 0 and s["level"].iloc[j - 1] == lvl:
        j -= 1
    fr = us["fred"]
    spreads = {k: {"label": FRED[k][0], **_weekly(fr[k], nd=2),
                   "now": _num(fr[k].iloc[-1], 2), "m6": _num(fr[k].iloc[-127], 2) if len(fr[k]) > 127 else None,
                   "lo": _num(fr[k].min(), 2), "hi": _num(fr[k].max(), 2)}
               for k in ("BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLC0A4CBBB") if fr.get(k) is not None}
    baa = fr["BAA10Y"]
    px = f[["BIZD", "BKLN", "HYG", "ORCL", "CRWV"]]
    market = []
    for k, name in (("BIZD", "上市 BDC（私募信贷）"), ("BKLN", "杠杆贷款"), ("HYG", "高收益债"),
                    ("ORCL", "甲骨文（AI 举债最多的大厂之一）"), ("CRWV", "CoreWeave（AI 云，高杠杆）")):
        x = px[k].dropna()
        if len(x) < 30:
            continue
        hi = x.tail(252).max()
        market.append({"key": k, "name": name, "price": _num(x.iloc[-1]), "off_high": _num((x.iloc[-1] / hi - 1) * 100, 1),
                       "m3": _num((x.iloc[-1] / x.iloc[-64] - 1) * 100, 1) if len(x) > 64 else None,
                       **_weekly(x, since=str((x.index[-1] - pd.Timedelta(days=3 * 365)).date()))})
    debt = fr.get("BCNSDODNS")
    cn = china or {}

    def tail(s, nd=1):
        return {"dates": [str(d.date()) for d in s.dropna().index], "v": [_num(v, nd) for v in s.dropna()]} if s is not None else None

    return {
        "as_of": str(f.index[-1].date()),
        "level": lvl, "level_label": LEVELS[lvl], "level_since": str(s.index[j].date()),
        "hold": int(s["hold"].iloc[-1]), "of": int(s["of"].iloc[-1]),
        "rules": rules,
        "spreads": spreads,
        "baa": {**_weekly(baa, nd=2), "bands": {str(q): _num(baa.quantile(q / 100), 2) for q in (20, 50, 80)},
                "rank": _num(s["baa_rank"].iloc[-1], 2)},
        "level_hist": _weekly(s["level"].map({"calm": 0, "weak": 1, "stress": 2}).astype(float), since="2007-01-01", nd=0),
        "nfci": _weekly(fr["NFCI"].loc["2000":], nd=3) if fr.get("NFCI") is not None else None,
        "sloos": tail(fr.get("DRTSCILM").loc["2000":] if fr.get("DRTSCILM") is not None else None),
        "market": market,
        "record": record(s["level"], f["SPY"]),
        "us_debt": {"now": _num(debt.iloc[-1] / 1e6, 2), "yoy": _num((debt.iloc[-1] / debt.iloc[-5] - 1) * 100, 1),
                    "asof": str(debt.index[-1].date())} if debt is not None and len(debt) > 5 else None,
        "credit_gdp": {k: tail(fr.get(c).loc["2000":] if fr.get(c) is not None else None)
                       for k, c in (("CN", "QCNPAM770A"), ("US", "QUSPAM770A"))},
        "china": {k: tail(v.loc["2016":]) for k, v in cn.items()},
    }


if __name__ == "__main__":
    import json
    b = board(fetch_us(), fetch_china())
    print(b["level_label"], b["hold"], "/", b["of"], [(r["key"], r["ok"], r["value"]) for r in b["rules"]])
    print(json.dumps(b["record"], ensure_ascii=False))
