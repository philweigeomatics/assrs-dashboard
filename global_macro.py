"""
global_macro.py — US yield curve + overseas equity board.

Sources were chosen for one reason above all: they must work from a datacenter
IP. The Market Leverage panel already learned this the hard way — exchange and
association sites geo-block cloud hosts, so a scraper that works on a laptop
shows nothing in production. Everything here is a documented JSON API built for
programmatic access.

  FRED (api.stlouisfed.org)  — US Treasury constant-maturity yields. Free key,
                               120 req/min, updated ~18:00 ET.
  Tushare index_global       — real Nikkei / KOSPI / S&P index levels. Tried
                               first because the app is already authenticated
                               against Tushare; sits behind a points threshold,
                               so it may not be available on every account.
  Twelve Data                — fallback. NOTE its free tier does NOT include
                               indices ("SPX" returns "available starting with
                               the Grow plan"); US-listed ETF proxies do work,
                               so that is what the fallback uses. Verified.

Stooq was evaluated and rejected: it served a bot-block HTML page rather than
CSV from a datacenter IP.

All fetchers return the same shape and never raise — a dead source yields a
card with `ok=False` and a reason, matching market_leverage.py.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
TD_URL = "https://api.twelvedata.com/time_series"

# label, FRED series id, years-to-maturity (x-axis position)
TENORS: list[tuple[str, str, float]] = [
    ("1M",  "DGS1MO", 1 / 12),
    ("3M",  "DGS3MO", 0.25),
    ("6M",  "DGS6MO", 0.5),
    ("1Y",  "DGS1",   1.0),
    ("2Y",  "DGS2",   2.0),
    ("3Y",  "DGS3",   3.0),
    ("5Y",  "DGS5",   5.0),
    ("7Y",  "DGS7",   7.0),
    ("10Y", "DGS10",  10.0),
    ("20Y", "DGS20",  20.0),
    ("30Y", "DGS30",  30.0),
]

# Overseas board. Tushare code first, Twelve Data ETF proxy second.
# The proxies are US-listed and USD-denominated, so they carry FX and trade on
# US hours — see fetch_global_board() for why that matters.
#
# tz / open / close are the market's OWN session, and they are here because a
# bare trade_date lies by omission. Every row can read "2026-08-18" while
# referring to moments 14 hours apart: Tokyo's close is 14:00 Beijing the same
# day, New York's is 04:00 Beijing the NEXT day. At an A-share open, the US
# number is hours old and the Tokyo number is a session behind with Tokyo
# already trading again. session_context() turns the date into an actual moment.
BOARD: list[dict] = [
    {"label": "S&P 500",    "cn": "标普500",  "ts": "SPX",  "etf": "SPY",
     "tz": "America/New_York", "open": (9, 30), "close": (16, 0)},
    {"label": "Nasdaq",     "cn": "纳斯达克",  "ts": "IXIC", "etf": "QQQ",
     "tz": "America/New_York", "open": (9, 30), "close": (16, 0)},
    {"label": "Nikkei 225", "cn": "日经225",  "ts": "N225", "etf": "EWJ",
     "tz": "Asia/Tokyo",       "open": (9, 0),  "close": (15, 0)},
    {"label": "KOSPI",      "cn": "韩国综合",  "ts": "KS11", "etf": "EWY",
     "tz": "Asia/Seoul",       "open": (9, 0),  "close": (15, 30)},
    {"label": "Hang Seng",  "cn": "恒生",     "ts": "HSI",  "etf": "EWH",
     "tz": "Asia/Hong_Kong",   "open": (9, 30), "close": (16, 0)},
]

BEIJING = "Asia/Shanghai"


def session_context(trade_date: str, tz_name: str,
                    open_hm: tuple, close_hm: tuple, now=None) -> dict:
    """
    Turn a bare trade_date into a moment, expressed in Beijing time.

    Returns when that session actually closed (Beijing), how many hours ago
    that was, and whether the market is trading right now — which is the case
    that matters: a Tokyo close from "today" is already superseded by the time
    an A-share session opens the following morning.

    Holidays are not modelled; `is_open` is a freshness hint, not a trading
    calendar. Returns {"ok": False} on anything unparseable rather than raising.
    """
    from datetime import datetime, time as _time
    from zoneinfo import ZoneInfo

    try:
        d = pd.to_datetime(str(trade_date)).date()
        mtz = ZoneInfo(tz_name)
        btz = ZoneInfo(BEIJING)
        closed_local = datetime.combine(d, _time(*close_hm), tzinfo=mtz)
        closed_bj = closed_local.astimezone(btz)

        now_bj = (now or datetime.now(btz)).astimezone(btz)
        hours_ago = (now_bj - closed_bj).total_seconds() / 3600.0

        now_local = now_bj.astimezone(mtz)
        is_open = (
            now_local.weekday() < 5
            and _time(*open_hm) <= now_local.time() <= _time(*close_hm)
        )
        return {
            "ok": True,
            "closed_bj": closed_bj,
            "closed_bj_str": closed_bj.strftime("%m-%d %H:%M"),
            "hours_ago": hours_ago,
            "is_open": is_open,
            "local_close_str": closed_local.strftime("%m-%d %H:%M"),
        }
    except Exception:
        return {"ok": False}


def _secret(name: str) -> str:
    """
    Streamlit secrets → environment variable, same chain as api_config.

    Deliberately NOT importing api_config._get_secret: that module resolves
    every token at import time, so `from api_config import _get_secret` raises
    if any unrelated token is missing. Importing it here would make this panel
    fail for a reason that has nothing to do with its own keys.
    """
    import os
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    val = os.environ.get(name)
    if val:
        return val
    raise ValueError(
        f"Secret '{name}' not found. Add it to .streamlit/secrets.toml locally, "
        f"or to app Settings → Secrets on Streamlit Cloud."
    )


def _fred_key() -> str:
    return _secret("FRED_API_KEY")


def _td_key() -> str:
    return _secret("TWELVEDATA_API_KEY")


# ── US Treasury yield curve ───────────────────────────────────────────────────

def fetch_yield_curve(lookback_days: int = 400, timeout: int = 25) -> dict:
    """
    Daily constant-maturity Treasury yields for every tenor in TENORS.

    Returns {"ok": bool, "data": DataFrame indexed by date with one column per
    tenor label (percent), "as_of": str, "reason": str|None}.

    FRED has no multi-series endpoint on the free API, so this is one request
    per tenor — 11 in total, well inside the 120/min limit. Cache it.
    """
    try:
        key = _fred_key()
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "data": None}

    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    cols: dict[str, pd.Series] = {}
    failed: list[str] = []

    with requests.Session() as sess:
        for label, sid, _ in TENORS:
            try:
                r = sess.get(FRED_URL, timeout=timeout, params={
                    "series_id": sid, "api_key": key, "file_type": "json",
                    "observation_start": start,
                })
                r.raise_for_status()
                obs = r.json().get("observations", [])
                if not obs:
                    failed.append(label)
                    continue
                s = pd.Series(
                    {o["date"]: o["value"] for o in obs}, dtype="object")
                # FRED writes "." for a non-publication day (holidays).
                cols[label] = pd.to_numeric(s.replace(".", None), errors="coerce")
            except Exception:
                failed.append(label)

    if not cols:
        return {"ok": False, "reason": "no tenors returned", "data": None}

    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna(how="all")
    if df.empty:
        return {"ok": False, "reason": "all observations empty", "data": None}

    return {
        "ok": True,
        "data": df,
        "as_of": df.dropna(how="all").index[-1].strftime("%Y-%m-%d"),
        "missing": failed,
        "reason": None,
    }


def curve_snapshots(df: pd.DataFrame,
                    offsets: "dict[str, int] | None" = None) -> pd.DataFrame:
    """
    The curve as of today and at several points in the past, for overlay.

    `offsets` maps a label to calendar days back. Each snapshot uses the last
    published row on or before that date, so holidays don't blank a column.
    Returns a DataFrame indexed by tenor label with one column per snapshot.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    offsets = offsets or {"1W ago": 7, "1M ago": 30, "3M ago": 91, "1Y ago": 365}

    valid = df.dropna(how="all")
    if valid.empty:
        return pd.DataFrame()
    last = valid.index[-1]

    out: dict[str, pd.Series] = {"Today": df.loc[last]}
    for label, days in offsets.items():
        target = last - pd.Timedelta(days=days)
        prior = df.loc[df.index <= target]
        prior = prior.dropna(how="all")
        if not prior.empty:
            out[label] = prior.iloc[-1]

    order = [t[0] for t in TENORS if t[0] in df.columns]
    return pd.DataFrame(out).reindex(order)


def classify_curve_move(df: pd.DataFrame, days: int = 30,
                        short: str = "2Y", long: str = "10Y") -> dict:
    """
    Name the move over `days` using the standard four-way taxonomy.

    Direction comes from the long end (bull = yields fell = prices rose), shape
    from whether the long-short spread widened or narrowed. The four names are
    genuinely different macro stories, which is why "rates went up" alone is not
    a useful summary:

      bear steepening  — long end selling off faster; growth/inflation/supply
      bull steepening  — front end rallying faster; cuts being priced
      bear flattening  — front end selling off faster; hikes being priced
      bull flattening  — long end rallying faster; growth scare / duration bid
    """
    if df is None or df.empty or short not in df.columns or long not in df.columns:
        return {"ok": False}

    sub = df[[short, long]].dropna(how="all")
    if sub.empty:
        return {"ok": False}

    last = sub.index[-1]
    prior = sub.loc[sub.index <= last - pd.Timedelta(days=days)].dropna(how="all")
    if prior.empty:
        return {"ok": False}

    now, then = sub.loc[last], prior.iloc[-1]
    d_short = (now[short] - then[short]) * 100.0      # basis points
    d_long = (now[long] - then[long]) * 100.0
    d_spread = d_long - d_short

    if pd.isna(d_short) or pd.isna(d_long):
        return {"ok": False}

    # Direction from the LEVEL factor — the average of the two ends — rather
    # than the long end alone. Textbooks say "long end sets it", but that
    # degenerates when the long end is flat: a week with 2Y −6bp and 10Y exactly
    # 0bp got labelled "bull" purely on the sign of a zero. Averaging is the
    # level component of the usual level/slope decomposition, and the slope
    # component is the spread term already computed below.
    d_level = (d_short + d_long) / 2.0
    direction = "bear" if d_level > 0 else "bull"
    # A near-equal move at both ends is a parallel shift, not a shape change.
    if abs(d_spread) < 10:
        shape, label = "parallel", f"{direction} parallel shift 平行移动"
    elif d_spread > 0:
        shape, label = "steepening", f"{direction} steepening 陡峭化"
    else:
        shape, label = "flattening", f"{direction} flattening 平坦化"

    return {
        "ok": True, "label": label, "direction": direction, "shape": shape,
        "d_short_bp": float(d_short), "d_long_bp": float(d_long),
        "d_spread_bp": float(d_spread), "days": days,
        "short": short, "long": long,
        "from_date": prior.index[-1].strftime("%Y-%m-%d"),
        "to_date": last.strftime("%Y-%m-%d"),
    }


def fetch_spread(series_id: str = "T10Y2Y", lookback_days: int = 800,
                 timeout: int = 25) -> dict:
    """One FRED spread series, pre-computed by FRED (T10Y2Y, T10Y3M)."""
    try:
        key = _fred_key()
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "data": None}
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    try:
        r = requests.get(FRED_URL, timeout=timeout, params={
            "series_id": series_id, "api_key": key, "file_type": "json",
            "observation_start": start})
        r.raise_for_status()
        obs = r.json().get("observations", [])
        s = pd.Series({o["date"]: o["value"] for o in obs}, dtype="object")
        s = pd.to_numeric(s.replace(".", None), errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        if s.empty:
            return {"ok": False, "reason": "empty series", "data": None}
        return {"ok": True, "data": s.sort_index(), "series_id": series_id,
                "latest": float(s.iloc[-1]),
                "as_of": s.index[-1].strftime("%Y-%m-%d"), "reason": None}
    except Exception as exc:
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}", "data": None}


# ── Overseas equity board ─────────────────────────────────────────────────────

def _board_from_tushare(pro_api) -> list[dict] | None:
    """Real index levels via Tushare index_global. None if unavailable."""
    if pro_api is None:
        return None
    end = date.today().strftime("%Y%m%d")
    start = (date.today() - timedelta(days=30)).strftime("%Y%m%d")
    rows: list[dict] = []
    for item in BOARD:
        try:
            df = pro_api.index_global(ts_code=item["ts"],
                                      start_date=start, end_date=end)
            if df is None or df.empty:
                return None            # tier/points issue — fall back wholesale
            df = df.sort_values("trade_date")
            last = df.iloc[-1]
            pct = last.get("pct_chg")
            if pct is None or pd.isna(pct):
                prev = df.iloc[-2]["close"] if len(df) > 1 else None
                pct = ((last["close"] / prev) - 1) * 100 if prev else float("nan")
            _td = str(last["trade_date"])
            rows.append({"label": item["label"], "cn": item["cn"],
                         "value": float(last["close"]), "pct": float(pct),
                         "date": _td, "proxy": False,
                         "session": session_context(
                             _td, item["tz"], item["open"], item["close"])})
        except Exception:
            return None
    return rows or None


def _board_from_twelvedata(timeout: int = 25) -> list[dict]:
    """
    Fallback via US-listed ETF proxies.

    Twelve Data's free tier excludes index symbols outright, so this tracks
    ETFs instead. Two consequences the UI must state rather than hide: the
    proxies are USD-denominated, so a move can come from the currency rather
    than the market, and they trade US hours, so an "EWJ close" is the US
    session's read on Japan, not the Tokyo close.
    """
    try:
        key = _td_key()
    except Exception as exc:
        return [{"label": i["label"], "cn": i["cn"], "ok": False,
                 "reason": str(exc)} for i in BOARD]

    rows: list[dict] = []
    with requests.Session() as sess:
        for item in BOARD:
            try:
                j = sess.get(TD_URL, timeout=timeout, params={
                    "symbol": item["etf"], "interval": "1day",
                    "outputsize": "2", "apikey": key}).json()
                if j.get("status") == "error" or "values" not in j:
                    rows.append({"label": item["label"], "cn": item["cn"],
                                 "ok": False,
                                 "reason": str(j.get("message", "no data"))[:120]})
                    continue
                vals = j["values"]
                close = float(vals[0]["close"])
                prev = float(vals[1]["close"]) if len(vals) > 1 else None
                pct = ((close / prev) - 1) * 100 if prev else float("nan")
                # The proxy trades in New York regardless of which market it
                # tracks, so its session is the US one — not Tokyo's or Seoul's.
                rows.append({"label": item["label"], "cn": item["cn"], "ok": True,
                             "value": close, "pct": pct,
                             "date": vals[0]["datetime"], "proxy": True,
                             "symbol": item["etf"],
                             "session": session_context(
                                 vals[0]["datetime"], "America/New_York",
                                 (9, 30), (16, 0))})
            except Exception as exc:
                rows.append({"label": item["label"], "cn": item["cn"], "ok": False,
                             "reason": f"{type(exc).__name__}"})
    return rows


def fetch_global_board(pro_api=None) -> dict:
    """
    Overseas index board: real index levels when Tushare allows, ETF proxies
    otherwise. Returns {"ok", "rows", "source", "proxy"}.
    """
    rows = _board_from_tushare(pro_api)
    if rows:
        return {"ok": True, "rows": [dict(r, ok=True) for r in rows],
                "source": "Tushare index_global", "proxy": False}
    rows = _board_from_twelvedata()
    return {"ok": any(r.get("ok") for r in rows), "rows": rows,
            "source": "Twelve Data (ETF proxies)", "proxy": True}
