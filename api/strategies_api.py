"""
api/strategies_api.py — the watchlist screens, behind HTTP.

Both screens walk the user's whole watchlist and hit Tushare two or three times
per stock, so a scan is a deliberate action, not something a page load triggers:
GET returns the last result, POST .../scan recomputes it. That is the same
shape the Streamlit pages have, and for 做T it is literally the same rows —
saved through data_manager's existing t_trading_scans table, so a scan run in
either app shows up in the other.

Serial, not parallel. The nightly job measured this: four workers against
Tushare at once lost stocks to transient failures that a serial pass did not
hit, and a screen that silently omits a holding is worse than a slow one.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from strategies import mean_reversion as mr
from strategies import t_trading as tt

#: How long a saved scan stays useful. Longer than a session, shorter than a
#: market regime — the client shows the age either way.
STALE_AFTER_HOURS = 20


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prices(ticker: str, years: int = 1):
    import data_manager
    try:
        return data_manager.get_single_stock_data_live(ticker, lookback_years=years)
    except Exception:
        return None


def _ts_code(ticker: str) -> str:
    import data_manager
    try:
        return data_manager.get_tushare_ticker(ticker)
    except Exception:
        return f"{ticker}.SH" if ticker.startswith(("6", "9")) else f"{ticker}.SZ"


def _market_turnover_20d(trade_dates: list[str]) -> dict:
    """
    {ts_code: 20-day mean turnover %} for the WHOLE market, in 20 calls.

    daily_basic accepts a trade_date and returns every stock for that day, so
    one call per date covers the entire watchlist. Per-stock it was one call
    each — for 78 holdings that is 78 calls replaced by 20, and the saving
    grows with the watchlist instead of the reverse.
    """
    import data_manager
    from collections import defaultdict

    sums: dict = defaultdict(list)
    try:
        data_manager.init_tushare()
        api = data_manager.TUSHARE_API
        if api is None:
            return {}
    except Exception:
        return {}

    for day in trade_dates[-20:]:
        try:
            df = api.daily_basic(trade_date=day, fields="ts_code,turnover_rate")
        except Exception:
            continue
        if df is None or df.empty:
            continue
        for code, rate in zip(df["ts_code"], df["turnover_rate"]):
            if rate == rate:                       # not NaN
                sums[str(code)].append(float(rate))
    return {k: sum(v) / len(v) for k, v in sums.items() if v}


def _market_limits(trade_dates: list[str]) -> dict:
    """
    {ts_code: (hit any limit, limit-down count)} across the last five sessions,
    for the whole market, in ten calls rather than two per stock.
    """
    import data_manager
    from collections import defaultdict

    any_hit: dict = defaultdict(bool)
    downs: dict = defaultdict(int)
    try:
        data_manager.init_tushare()
        api = data_manager.TUSHARE_API
        if api is None:
            return {}
    except Exception:
        return {}

    for day in trade_dates[-5:]:
        try:
            limits = api.stk_limit(trade_date=day)
            daily = api.daily(trade_date=day)
        except Exception:
            continue
        if limits is None or limits.empty or daily is None or daily.empty:
            continue
        m = limits.merge(daily[["ts_code", "close"]], on="ts_code", how="inner")
        for code, close, up, dn in zip(m["ts_code"], m["close"], m["up_limit"], m["down_limit"]):
            try:
                close, up, dn = float(close), float(up), float(dn)
            except (TypeError, ValueError):
                continue
            if close >= up - 1e-4:
                any_hit[str(code)] = True
            if close <= dn + 1e-4:
                any_hit[str(code)] = True
                downs[str(code)] += 1
    return {c: (any_hit.get(c, False), downs.get(c, 0))
            for c in set(any_hit) | set(downs)}


def _turnover_20d(ts_code: str) -> float | None:
    """20-day average turnover from daily_basic, in percent."""
    import data_manager
    try:
        data_manager.init_tushare()
        if data_manager.TUSHARE_API is None:
            return None
        end = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=45)).strftime("%Y%m%d")
        df = data_manager.TUSHARE_API.daily_basic(
            ts_code=ts_code, start_date=start, end_date=end,
            fields="ts_code,trade_date,turnover_rate")
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["turnover_rate"]).sort_values("trade_date").tail(20)
        return float(df["turnover_rate"].mean()) if not df.empty else None
    except Exception:
        return None


def _limit_days(ts_code: str, lookback: int = 5) -> tuple[bool, int]:
    """
    (hit any limit, count of limit-DOWN days) over the last `lookback` sessions.

    One call serves both screens: 做T rejects a stock that locked either way,
    while 反转 counts only the down side as capitulation.
    """
    import data_manager
    try:
        data_manager.init_tushare()
        api = data_manager.TUSHARE_API
        if api is None:
            return False, 0
        end = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=lookback * 2 + 5)).strftime("%Y%m%d")
        limits = api.stk_limit(ts_code=ts_code, start_date=start, end_date=end)
        daily = api.daily(ts_code=ts_code, start_date=start, end_date=end)
        if limits is None or limits.empty or daily is None or daily.empty:
            return False, 0
        m = (limits.merge(daily[["trade_date", "close"]], on="trade_date", how="inner")
             .sort_values("trade_date").tail(lookback))
        up = m["close"] >= m["up_limit"] - 1e-4
        down = m["close"] <= m["down_limit"] + 1e-4
        return bool((up | down).any()), int(down.sum())
    except Exception:
        return False, 0


def _trade_dates(frames: dict) -> list[str]:
    """
    Recent trading days as YYYYMMDD, taken from the price frames already
    fetched — their index IS the exchange calendar, so this needs no extra
    call and cannot disagree with the data being scored.
    """
    best = None
    for df in frames.values():
        if df is not None and not getattr(df, "empty", True):
            if best is None or len(df) > len(best):
                best = df
    if best is None:
        return []
    return [d.strftime("%Y%m%d") for d in best.index[-25:]]


def _watchlist() -> list[dict]:
    import data_manager
    out = []
    for item in data_manager.get_watchlist() or []:
        ticker = str(item.get("ticker"))
        name = (item.get("stock_name")
                or data_manager.get_stock_name_from_db(ticker) or ticker)
        out.append({"ticker": ticker, "name": name})
    return out


# ── 做T ──────────────────────────────────────────────────────────────────────
#: The saved table uses the Streamlit page's column names. Mapping rather than
#: renaming the table keeps both apps reading the same rows.
_SAVE_KEYS = {
    "score": "T-Score", "range_pct": "Range %", "turnover_pct": "Turnover %",
    "meanrev_bias": "MeanRev bias", "adx": "ADX", "range_pos": "Range pos",
}


def _to_saved(row: dict) -> dict:
    out = {"Ticker": row["ticker"], "Name": row["name"],
           "Verdict": tt.VERDICT_CN.get(row["verdict"], row["verdict"]),
           "Limit event?": "⚠️ Yes" if row.get("limit_event") else "—",
           "Why": row.get("why") or ""}
    for k, col in _SAVE_KEYS.items():
        out[col] = row.get(k)
    return out


def _from_saved(df) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "ticker": str(r.get("Ticker")), "name": str(r.get("Name") or ""),
            "score": _num(r.get("T-Score")),
            "verdict_cn": str(r.get("Verdict") or ""),
            "range_pct": _num(r.get("Range %")),
            "turnover_pct": _num(r.get("Turnover %")),
            "meanrev_bias": _num(r.get("MeanRev bias")),
            "adx": _num(r.get("ADX")),
            "range_pos": _num(r.get("Range pos")),
            "limit_event": "Yes" in str(r.get("Limit event?") or ""),
            "why": str(r.get("Why") or ""),
        })
    rows.sort(key=lambda x: -(x["score"] if x["score"] is not None else -1))
    return rows


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def t_trading_saved(user_id: int) -> dict | None:
    """The last saved 做T scan, or None if this user has never run one."""
    import data_manager
    try:
        data_manager.ensure_t_trading_scan_table()
        # Always a 2-tuple, (None, None) when there is nothing saved — so the
        # emptiness test has to look inside it, not at the tuple.
        df, scanned_at = data_manager.load_t_trading_scan(user_id)
    except Exception:
        return None
    if df is None or getattr(df, "empty", True):
        return None
    return _wrap(_from_saved(df), scanned_at)


def t_trading_scan(user_id: int) -> dict:
    """Walk the watchlist and score every holding, then save."""
    import data_manager

    frames = {i["ticker"]: _prices(i["ticker"], 1) for i in _watchlist()}
    dates = _trade_dates(frames)
    turnover = _market_turnover_20d(dates)
    limits = _market_limits(dates)

    rows = []
    for item in _watchlist():
        ts = _ts_code(item["ticker"])
        limit_any, _ = limits.get(ts, (False, 0))
        out = tt.score(frames.get(item["ticker"]),
                       turnover_pct=turnover.get(ts), limit_event=limit_any)
        rows.append({**item, **out,
                     "verdict_cn": tt.VERDICT_CN.get(out["verdict"], out["verdict"])})

    rows.sort(key=lambda r: (r["rank"], -(r["score"] or 0)))
    try:
        data_manager.ensure_t_trading_scan_table()
        data_manager.save_t_trading_scan(user_id, [_to_saved(r) for r in rows])
    except Exception as exc:                                   # noqa: BLE001
        print(f"[strategies] saving 做T scan failed: {exc}"[:200])
    return _wrap(rows, _now_iso())


# ── 反转 ─────────────────────────────────────────────────────────────────────
def _sector_5d() -> dict:
    """{ticker: sector 5-day return %} — one read per sector, not per stock."""
    import data_manager
    out: dict[str, float] = {}
    try:
        smap = data_manager.get_sector_stock_map() or {}
    except Exception:
        return out
    for sector, members in smap.items():
        try:
            df = data_manager.db.read_table(f"PPI_{sector}", columns="Date, Close",
                                            order_by="-Date", limit=10)
            if df is None or len(df) < 6:
                continue
            df = df.sort_values("Date")
            ret = float(df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100
        except Exception:
            continue
        for t in members or []:
            out.setdefault(str(t), ret)
    return out


def mean_reversion_scan(user_id: int) -> dict:
    sector_ret = _sector_5d()
    frames = {i["ticker"]: _prices(i["ticker"], 1) for i in _watchlist()}
    limits = _market_limits(_trade_dates(frames))

    rows = []
    for item in _watchlist():
        df = frames.get(item["ticker"])
        close = df["Close"] if df is not None and not df.empty else None
        volume = df["Volume"] if df is not None and "Volume" in getattr(df, "columns", []) else None

        _, down_streak = limits.get(_ts_code(item["ticker"]), (False, 0))

        vs_sector = None
        if close is not None and len(close) >= 6 and item["ticker"] in sector_ret:
            stock_5d = float(close.iloc[-1] / close.iloc[-6] - 1) * 100
            vs_sector = stock_5d - sector_ret[item["ticker"]]

        out = mr.evaluate(close, volume, name=item["name"],
                          limit_down_streak=down_streak, vs_sector_pp=vs_sector)
        rows.append({**item, **out,
                     "verdict_cn": mr.VERDICT_CN.get(out["verdict"], out["verdict"])})

    rows.sort(key=lambda r: (r["rank"], -(r["passed"] or 0)))
    return _wrap(rows, _now_iso())


def _wrap(rows: list[dict], scanned_at) -> dict:
    at = str(scanned_at) if scanned_at else None
    hours = None
    if at:
        try:
            ts = pd.Timestamp(at)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            hours = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600
        except Exception:
            hours = None
    counts: dict[str, int] = {}
    for r in rows:
        key = r.get("verdict") or r.get("verdict_cn") or "?"
        counts[key] = counts.get(key, 0) + 1
    return {
        "scanned_at": at,
        "age_hours": round(hours, 1) if hours is not None else None,
        "stale": bool(hours is not None and hours > STALE_AFTER_HOURS),
        "count": len(rows),
        "counts": counts,
        "rows": rows,
    }
