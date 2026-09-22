"""
world_indices.py — the major indices, and an honest answer to "as of when".

A strip of index moves looks like the simplest panel on a dashboard and is
the easiest one to make quietly wrong, because THE MARKETS ARE NOT ON THE
SAME CLOCK. At ten in the morning in Shanghai, the S&P's most recent close is
from yesterday New York time and today's Shanghai bar does not exist yet. Put
those two numbers side by side with one date over the top and the reader will
compare a session that has happened with one that has not.

So every index carries its OWN close date, and any index whose close is older
than the newest one in the set is marked behind and says by how much. That is
not a caveat in a tooltip; it is the difference between "Europe is down while
we are up" and "Europe's number is from before we opened".

TWO SOURCES, BECAUSE THERE IS NO ONE SOURCE

A-share indices come from Tushare, which is where the rest of this app's
A-share data comes from and is the only one that has them. Everything else
comes from Yahoo in a single batched request. A failure in either leaves the
other's rows intact: a dashboard strip that vanishes because Frankfurt was
unreachable is worse than one that shows Shanghai and says Frankfurt is
missing.
"""

from __future__ import annotations

import threading
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

#: Sessions fetched per index. Enough for a 10-point sparkline plus slack for
#: holidays, and small enough that the whole strip is one cheap request.
LOOKBACK = 20

#: Points in the sparkline sent to the client.
SPARK = 10

#: Yahoo shares a session and a cookie across calls; concurrent first-calls
#: race to establish it. Same reason markets/na.py serialises.
_yahoo_lock = threading.Lock()

#: The strip, in the order a reader in Shanghai would want it: home first,
#: then the region, then the markets that set the tone overnight.
GROUPS: list[dict] = [
    {"name": "A 股", "source": "tushare", "tz": "Asia/Shanghai", "members": [
        ("000001.SH", "上证指数"),
        ("399001.SZ", "深证成指"),
        ("399006.SZ", "创业板指"),
        ("000300.SH", "沪深300"),
        ("000905.SH", "中证500"),
        ("000688.SH", "科创50"),
    ]},
    {"name": "亚太", "source": "yahoo", "tz": "Asia/Tokyo", "members": [
        ("^HSI", "恒生指数"),
        ("^N225", "日经225"),
        ("^KS11", "韩国综合"),
        ("^AXJO", "澳洲200"),
    ]},
    {"name": "欧美", "source": "yahoo", "tz": "America/New_York", "members": [
        ("^GSPC", "标普500"),
        ("^IXIC", "纳斯达克"),
        ("^DJI", "道琼斯"),
        ("^FTSE", "英国富时100"),
        ("^GDAXI", "德国DAX"),
        ("^VIX", "VIX 恐慌指数"),
    ]},
]


def _row(code: str, name: str, closes: pd.Series) -> dict | None:
    """One index's last close, its move, and the date that close belongs to."""
    s = pd.to_numeric(closes, errors="coerce").dropna()
    if len(s) < 2:
        return None
    last, prev = float(s.iloc[-1]), float(s.iloc[-2])
    if prev == 0:
        return None

    tail = [round(float(v), 4) for v in s.iloc[-SPARK:]]
    return {
        "code": code,
        "name": name,
        "close": round(last, 2),
        "change_pct": round((last / prev - 1) * 100, 2),
        "date": str(pd.Timestamp(s.index[-1]).date()),
        "spark": tail,
    }


def _tushare(members: list[tuple[str, str]]) -> list[dict]:
    import data_manager

    out = []
    for code, name in members:
        try:
            df = data_manager.get_index_data_live(code, lookback_days=LOOKBACK * 2)
        except Exception as exc:                                   # noqa: BLE001
            print(f"[world_indices] {code}: {type(exc).__name__}: {exc}")
            continue
        if df is None or df.empty or "Close" not in df.columns:
            continue
        row = _row(code, name, df["Close"])
        if row:
            out.append(row)
    return out


def _yahoo(members: list[tuple[str, str]]) -> list[dict]:
    """
    One batched download for every Yahoo symbol in the group.

    Batched because fifteen sequential Ticker().history() calls is fifteen
    round trips for a strip that has to render before the heatmap does.
    """
    import yfinance as yf

    codes = [c for c, _ in members]
    try:
        with _yahoo_lock:
            raw = yf.download(codes, period=f"{LOOKBACK}d", interval="1d",
                              auto_adjust=True, progress=False,
                              group_by="ticker", threads=False)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[world_indices] yahoo batch failed: {type(exc).__name__}: {exc}")
        return []
    if raw is None or raw.empty:
        return []

    out = []
    for code, name in members:
        try:
            # A single-symbol download comes back without the ticker level.
            closes = (raw[code]["Close"] if isinstance(raw.columns, pd.MultiIndex)
                      else raw["Close"])
        except Exception:                                          # noqa: BLE001
            continue
        row = _row(code, name, closes)
        if row:
            out.append(row)
    return out


def snapshot() -> dict:
    """
    Every index, grouped, with each one's own close date and how far behind.

    `behind_days` is against the newest date in the whole set rather than
    against today: on a Monday morning in Shanghai every Western index is
    one session behind and that is normal, not an error.

    `today` says the bar belongs to the current date in that market's own
    timezone — so the move is the session so far, not a settled close.
    """
    groups = []
    for spec in GROUPS:
        fetch = _tushare if spec["source"] == "tushare" else _yahoo
        try:
            rows = fetch(spec["members"])
        except Exception as exc:                                   # noqa: BLE001
            print(f"[world_indices] {spec['name']}: {type(exc).__name__}: {exc}")
            rows = []
        # A bar dated today is not necessarily a CLOSE — the session may
        # still be running. Comparing against today in that market's own
        # timezone is enough to say which, without modelling opening hours
        # for six exchanges.
        try:
            there = datetime.now(ZoneInfo(spec["tz"])).strftime("%Y-%m-%d")
        except Exception:                                          # noqa: BLE001
            there = None
        for r in rows:
            r["today"] = bool(there and r["date"] == there)

        groups.append({"name": spec["name"], "indices": rows,
                       "missing": len(spec["members"]) - len(rows),
                       "tz": spec["tz"]})

    dates = [r["date"] for g in groups for r in g["indices"]]
    newest = max(dates) if dates else None
    if newest:
        for g in groups:
            for r in g["indices"]:
                r["behind_days"] = _sessions_between(r["date"], newest)

    return {
        "groups": groups,
        "as_of": newest,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total": len(dates),
    }


def _sessions_between(older: str, newer: str) -> int:
    """
    Weekdays between two dates — a rough session count, not a calendar one.

    Friday to Monday is one session behind, not three days behind, and
    calling it three would make every Monday morning look like an outage.
    Holidays are not modelled: this decides whether to dim a row, and being
    a day out on Christmas is not worth a trading calendar.
    """
    try:
        a, b = pd.Timestamp(older), pd.Timestamp(newer)
    except Exception:                                              # noqa: BLE001
        return 0
    # No guard for a >= b: bdate_range of a backwards range is empty, so the
    # expression already yields 0. A separate check would be unreachable
    # code wearing the costume of a safeguard.
    return max(0, len(pd.bdate_range(a, b)) - 1)
