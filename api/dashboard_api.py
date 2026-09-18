"""
api/dashboard_api.py — the market dashboard's data, behind HTTP.

Five panels that answer different questions about the same day:

    heatmap    where the money went today, by sector and by name
    breadth    how broad that was, over the last three months
    leverage   how much of it was borrowed
    toplist    the 龙虎榜 — where the abnormal flow was
    (wyckoff and rotation live in their own root modules)

Everything here is shaping, not computing: the arithmetic belongs to
data_manager, market_leverage, wyckoff and sector_rotation. What this module
owns is the JSON — in particular, turning DataFrames into plain lists BEFORE
they reach FastAPI, so a NaN never leaves as a bare `NaN` token that
JSON.parse() refuses.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

SHANGHAI = ZoneInfo("Asia/Shanghai")

#: How far back the 龙虎榜 walk looks for the most recent session with data.
TOPLIST_LOOKBACK_DAYS = 10
TOPLIST_ROWS = 40

#: Sessions of breadth history sent to the client.
BREADTH_DAYS = 60


def _num(v) -> float | None:
    """float(v), or None for NaN/None/'' — never a bare NaN in the payload."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def _iso(trade_date) -> str:
    """'20260917' or '2026-09-17' → '2026-09-17'."""
    s = str(trade_date)
    digits = re.sub(r"\D", "", s)
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}" if len(digits) == 8 else s


# ── 市场热力图 ───────────────────────────────────────────────────────────────
def heatmap() -> dict:
    """
    Every mapped stock sized by 流通市值 and coloured by today's move.

    Sector and market percentages are CAP-WEIGHTED means of the same per-stock
    numbers the leaves carry, not a separately sourced index — so a sector box
    can never disagree with the boxes inside it.

    The layout is not computed here. A treemap's rectangles depend on the pixel
    size of the container, which the server does not know; sending the tree and
    letting the client square it means the same payload serves a phone and a
    wide monitor, and resizing costs no request.
    """
    import data_manager as dm

    sector_map = dm.get_sector_stock_map()
    tickers = sorted({t for lst in sector_map.values() for t in lst})
    if not tickers:
        raise LookupError("板块成分股映射为空")

    mcap = dm.get_daily_basic_for_tickers(tickers)
    if mcap is None or mcap.empty:
        raise LookupError("没有 daily_basic 市值数据 — 夜间任务可能未运行")

    trade_date = str(mcap["trade_date"].iloc[0])
    caps = dict(zip(mcap["ticker"], pd.to_numeric(mcap["circ_mv_yi"], errors="coerce")))
    pcts = _pct_change_for(trade_date)
    names = {s["ticker"]: s["name"] for s in dm.get_all_stock_basic()}

    sectors, total_cap, total_w = [], 0.0, 0.0
    for sector, members in sector_map.items():
        stocks, cap, weighted = [], 0.0, 0.0
        for t in members:
            mc = _num(caps.get(t))
            if mc is None or mc <= 0:
                continue
            pct = _num(pcts.get(t)) or 0.0
            stocks.append({"t": t, "n": names.get(t, t),
                           "mcap": round(mc, 2), "pct": round(pct, 2)})
            cap += mc
            weighted += pct * mc
        if not stocks:
            continue
        stocks.sort(key=lambda s: -s["mcap"])
        sectors.append({"name": sector, "mcap": round(cap, 2),
                        "pct": round(weighted / cap, 2) if cap else 0.0,
                        "stocks": stocks})
        total_cap += cap
        total_w += weighted

    if not sectors:
        raise LookupError("没有一只成分股拿到市值数据")
    sectors.sort(key=lambda s: -s["mcap"])
    return {
        "trade_date": _iso(trade_date),
        "mcap": round(total_cap, 2),
        "pct": round(total_w / total_cap, 2) if total_cap else 0.0,
        # An empty pct map means the daily() call failed; every box would be
        # grey and look like a flat market rather than like missing data.
        "has_moves": bool(pcts),
        "sectors": sectors,
    }


def _pct_change_for(trade_date: str) -> dict:
    """
    {ticker: pct_chg} for the whole market on one date, in a single call.

    daily(trade_date=…) returns every A-share for that day, so this costs one
    request whether the map holds 200 stocks or 5,000. Per-stock daily
    percentage change is not stored in the DB, which is why it is fetched.
    """
    import data_manager as dm
    compact = re.sub(r"\D", "", str(trade_date))
    try:
        dm.init_tushare()
        if dm.TUSHARE_API is None:
            return {}
        d = dm.TUSHARE_API.daily(trade_date=compact, fields="ts_code,pct_chg")
    except Exception as exc:                                       # noqa: BLE001
        print(f"[heatmap] pct_chg fetch failed: {exc}")
        return {}
    if d is None or d.empty:
        return {}
    return dict(zip(d["ts_code"].str[:6], pd.to_numeric(d["pct_chg"], errors="coerce")))


# ── 市场宽度历史 ─────────────────────────────────────────────────────────────
def breadth(days: int = BREADTH_DAYS) -> dict:
    """
    Share of each sector's members trading above their MA20, by day.

    Ordered oldest → newest so the client can render left-to-right without
    reversing, and sectors are ordered by their LATEST reading, which is the
    column a person actually scans.
    """
    import data_manager as dm

    df = dm.load_market_breadth_from_db()
    if df is None or df.empty:
        raise LookupError("数据库中没有市场宽度数据（market_breadth 表为空）")

    df = df.sort_index().tail(days)
    rows = []
    for name in df.columns:
        values = [_num(v) for v in df[name]]
        if all(v is None for v in values):
            continue
        latest = next((v for v in reversed(values) if v is not None), None)
        rows.append({"name": str(name), "latest": latest,
                     "values": [None if v is None else round(v, 4) for v in values]})
    if not rows:
        raise LookupError("市场宽度表里没有任何有效数据")
    rows.sort(key=lambda r: -(r["latest"] or 0))

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "sectors": rows,
        # The one number that makes the grid readable at a glance: how many
        # sectors have most of their members above MA20 today.
        "hot": sum(1 for r in rows if (r["latest"] or 0) >= 0.5),
        "total": len(rows),
    }


# ── 市场杠杆 ─────────────────────────────────────────────────────────────────
def leverage() -> list[dict]:
    """
    Margin borrowings for China and the US, as JSON-safe dicts.

    The DataFrames market_leverage returns (`series`, `cn_detail`) are replaced
    with lists here. A failed market is returned with ok=False and its error
    rather than omitted — "FINRA is down" and "the US has no margin debt" must
    not look the same on the page.
    """
    import data_manager as dm
    import market_leverage as mlev

    try:
        dm.init_tushare()
        api = dm.TUSHARE_API
    except Exception:                                              # noqa: BLE001
        api = None

    out = []
    for r in mlev.fetch_all(api):
        s = r.get("series")
        detail = r.get("cn_detail")
        out.append({
            "market": r.get("market"),
            "label": r.get("label"),
            "ok": bool(r.get("ok")),
            "unit": (r.get("unit") or "").strip(),
            "freq": r.get("freq"),
            "latest": _num(r.get("latest")),
            "prev": _num(r.get("prev")),
            "asof": r.get("asof"),
            "note": r.get("note"),
            "error": r.get("error"),
            "series": ([] if s is None or s.empty else
                       [{"period": str(p), "value": _num(v)}
                        for p, v in zip(s["period"], s["value"])]),
            "detail": _cn_detail(detail),
        })
    return out


def _cn_detail(detail) -> list[dict]:
    """The balances and daily flows behind the A-share headline."""
    if detail is None or getattr(detail, "empty", True):
        return []
    keep = [c for c in ("rzye", "rqye", "rzmre", "rzche", "net_fin") if c in detail.columns]
    return [
        {"period": str(row["period"]), **{c: _num(row[c]) for c in keep}}
        for _, row in detail.tail(120).iterrows()
    ]


# ── 龙虎榜 ───────────────────────────────────────────────────────────────────
def top_list(lookback_days: int = TOPLIST_LOOKBACK_DAYS) -> dict:
    """
    The most recent session's abnormal-trading list.

    Walks back a day at a time because weekends and holidays return an empty
    frame, and because Tushare publishes it only after the close — asking for
    "today" at 10:00 Beijing legitimately has no answer yet.
    """
    import data_manager as dm

    try:
        dm.init_tushare()
    except Exception as exc:                                       # noqa: BLE001
        raise LookupError(f"Tushare 初始化失败：{exc}")
    if dm.TUSHARE_API is None:
        raise LookupError("Tushare 未配置")

    now = datetime.now(SHANGHAI)
    for delta in range(lookback_days):
        day = (now - timedelta(days=delta)).strftime("%Y%m%d")
        try:
            df = dm.TUSHARE_API.top_list(trade_date=day)
        except Exception:                                          # noqa: BLE001
            continue
        if df is None or df.empty:
            continue
        return {"trade_date": _iso(day), "rows": _top_rows(df)}

    raise LookupError(f"最近 {lookback_days} 天没有龙虎榜数据")


def _top_rows(df) -> list[dict]:
    sort_col = "net_amount" if "net_amount" in df.columns else "amount"
    df = df.sort_values(sort_col, ascending=False).head(TOPLIST_ROWS)
    rows = []
    for _, r in df.iterrows():
        code = str(r.get("ts_code", ""))
        rows.append({
            "t": code[:6],
            "n": r.get("name") or "",
            "close": _num(r.get("close")),
            "pct": _num(r.get("pct_change")),
            # 元 → 万元, which is how the number is quoted in Chinese media.
            "net": (lambda v: None if v is None else round(v / 1e4, 1))(_num(r.get("net_amount"))),
            "net_rate": _num(r.get("net_rate")),
            "reason": r.get("reason") or "",
        })
    return rows
