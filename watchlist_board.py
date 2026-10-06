"""
watchlist_board.py — the watchlist as something you can actually watch.

The page stored names. Its columns were 代码 / 名称 / 加入日期 / 操作, which
tells you only what you already knew: that you added a stock. Everything that
would let you watch anything lived on other pages.

Worse, the app already computed the interesting part and dropped it here. The
nightly scan writes price, RSI, ADX, MACD and parsed signals for every
A-share on the list — 今日提醒 reads that table, the watchlist never did.

So this assembles one row per holding: last price, today's move, five and
twenty-day moves, sixty sessions of shape, and whatever the scan found.

Why it is fast
--------------
Both sources take a batch, which was the thing worth checking before
designing anything:

  · Tushare `daily` and `adj_factor` accept a comma-separated list. Eighty-one
    A-shares over three months came back in 1.4s as 5,265 rows — one call
    rather than eighty-one.
  · `yf.download` takes a space-separated list. Twenty-three North American
    names over three months took 1.6s.

A whole watchlist is therefore a handful of calls, not a hundred, and the
sparklines are close to free.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

#: Calendar days fetched. ~60 trading sessions, enough for a 20-day move and
#: a line with a recognisable shape.
WINDOW_DAYS = 95
#: Points kept in the sparkline. More is noise at this size.
SPARK_POINTS = 60
#: Codes per Tushare call. 81 codes × 65 sessions came to 5,265 rows against a
#: ~6,000 cap, which is close enough to chunk rather than hope.
CHUNK = 40
#: Lookbacks reported next to today's move.
SPANS = (5, 20)


def _api():
    import data_manager
    if not data_manager.init_tushare():
        raise RuntimeError("Tushare 未初始化")
    return data_manager.TUSHARE_API


def _chunks(items: list, size: int = CHUNK):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ── prices ───────────────────────────────────────────────────────────────────
def _cn_bars(codes: list[str], start: str, end: str) -> dict[str, pd.Series]:
    """
    Forward-adjusted closes per ts_code, from two batched calls per chunk.

    Adjusted rather than raw: `daily` is 不复权, so a 转增 inside the window
    would draw a cliff in the sparkline that never happened to anyone holding
    the stock. `adj_factor` batches the same way, so correctness here costs
    one more call rather than one call per name.
    """
    if not codes:
        return {}
    api = _api()
    out: dict[str, pd.Series] = {}

    for chunk in _chunks(codes):
        joined = ",".join(chunk)
        try:
            px = api.daily(ts_code=joined, start_date=start, end_date=end)
        except Exception as exc:                                   # noqa: BLE001
            print(f"[watchlist] daily: {type(exc).__name__}: {exc}"[:160])
            continue
        if px is None or px.empty:
            continue

        try:
            adj = api.adj_factor(ts_code=joined, start_date=start, end_date=end)
        except Exception:                                          # noqa: BLE001
            adj = None

        px = px[["ts_code", "trade_date", "close"]].copy()
        if adj is not None and not adj.empty:
            px = px.merge(adj[["ts_code", "trade_date", "adj_factor"]],
                          on=["ts_code", "trade_date"], how="left")
        else:
            px["adj_factor"] = 1.0

        for code, grp in px.groupby("ts_code"):
            g = grp.sort_values("trade_date")
            factor = pd.to_numeric(g["adj_factor"], errors="coerce").ffill()
            close = pd.to_numeric(g["close"], errors="coerce")
            latest = factor.iloc[-1] if len(factor) and factor.iloc[-1] else 1.0
            series = (close * factor / latest) if latest else close
            series.index = g["trade_date"].astype(str).values
            out[str(code)] = series.dropna()
    return out


def _na_bars(codes: list[str], days: int) -> dict[str, pd.Series]:
    """Closes per ticker, from one bulk Yahoo download."""
    if not codes:
        return {}
    try:
        import yfinance as yf
        frame = yf.download(" ".join(codes), period=f"{days}d", interval="1d",
                            progress=False, auto_adjust=True, threads=True)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[watchlist] yfinance: {type(exc).__name__}: {exc}"[:160])
        return {}
    if frame is None or frame.empty:
        return {}

    if isinstance(frame.columns, pd.MultiIndex):
        if "Close" not in frame.columns.get_level_values(0):
            return {}
        closes = frame["Close"]
    else:
        # One ticker comes back flat rather than as a MultiIndex.
        closes = frame[["Close"]].rename(columns={"Close": codes[0]})

    out: dict[str, pd.Series] = {}
    for code in closes.columns:
        s = pd.to_numeric(closes[code], errors="coerce").dropna()
        if s.empty:
            continue
        s.index = [str(i)[:10] for i in s.index]
        out[str(code)] = s
    return out


# ── the scan the watchlist never read ────────────────────────────────────────
def _signals(user_id: int) -> dict[str, dict]:
    """
    The newest nightly scan row per A-share ticker.

    Already written every night with price, RSI, ADX, MACD and parsed
    signals — the watchlist simply never looked at it.
    """
    try:
        from db_manager import db
        rows = db.read_table("daily_signals", filters={"user_id": int(user_id)},
                             order_by="-scan_date", limit=2000)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[watchlist] daily_signals: {type(exc).__name__}: {exc}"[:160])
        return {}
    if rows is None or rows.empty:
        return {}

    newest = str(rows["scan_date"].max())
    rows = rows[rows["scan_date"].astype(str) == newest]

    import alerts_feed

    out: dict[str, dict] = {}
    for _, r in rows.iterrows():
        ticker = str(r.get("ticker") or "")
        if not ticker:
            continue
        try:
            parsed = alerts_feed.parse_signals(str(r.get("signals") or ""))
        except Exception:                                          # noqa: BLE001
            parsed = []
        out[ticker] = {
            "scan_date": newest,
            "rsi": _num(r.get("rsi")),
            # `cn` is the human label alerts_feed produces; `id` is what a
            # filter would key on, so both travel.
            "signals": [{"id": p.get("id"), "label": p.get("cn") or p.get("en"),
                         "dir": p.get("dir"), "group": p.get("group")}
                        for p in parsed][:4],
            "signal_count": int(r.get("signal_count") or 0),
        }
    return out


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, 4)


def _move(series: pd.Series, back: int) -> float | None:
    """Percent change against the close `back` sessions ago."""
    if len(series) <= back:
        return None
    now = float(series.iloc[-1])
    then = float(series.iloc[-1 - back])
    return None if not then else round((now / then - 1) * 100, 2)


# ── the board ────────────────────────────────────────────────────────────────
def board(watchlist: list[dict], user_id: int,
          ref: date | None = None) -> dict:
    """One row per watched name, with enough on it to be worth watching."""
    import markets
    import session_state

    ref = ref or date.today()
    start = (ref - timedelta(days=WINDOW_DAYS)).strftime("%Y%m%d")
    end = ref.strftime("%Y%m%d")

    cn, na = [], []
    for row in watchlist:
        symbol = str(row.get("t") or "")
        name = str(row.get("n") or symbol)
        try:
            market = markets.split(symbol)[0]
        except Exception:                                          # noqa: BLE001
            continue
        entry = {"t": symbol, "n": name, "market": market,
                 "added": row.get("at")}
        (cn if market == "CN" else na).append(entry)

    import data_manager
    cn_codes = [data_manager.get_tushare_ticker(e["t"]) for e in cn]
    na_codes = [markets.split(e["t"])[1] for e in na]

    cn_bars = _cn_bars(cn_codes, start, end)
    na_bars = _na_bars(na_codes, WINDOW_DAYS)
    scan = _signals(user_id)

    rows = []
    for entry, code in list(zip(cn, cn_codes)) + list(zip(na, na_codes)):
        series = (cn_bars if entry["market"] == "CN" else na_bars).get(code)
        hit = scan.get(entry["t"]) if entry["market"] == "CN" else None
        rows.append(_row(entry, code, series, hit, session_state))

    rows.sort(key=lambda r: (r["market"] != "CN", r["n"]))
    got = sum(1 for r in rows if r["price"] is not None)
    return {
        "rows": rows,
        "as_of": ref.isoformat(),
        "counts": {"total": len(rows), "priced": got,
                   "cn": len(cn), "na": len(na)},
        "scan_date": next((v["scan_date"] for v in scan.values()), None),
        "spark_points": SPARK_POINTS,
    }


def _row(entry: dict, code: str, series, hit, session_state) -> dict:
    out = {
        **entry, "code": code,
        "price": None, "chg_pct": None, "last_bar": None,
        "spark": [], "session": None,
        "rsi": None, "signals": [], "signal_count": 0, "scan_date": None,
    }
    for span in SPANS:
        out[f"chg_{span}d_pct"] = None

    if series is not None and len(series):
        out["price"] = round(float(series.iloc[-1]), 4)
        out["chg_pct"] = _move(series, 1)
        for span in SPANS:
            out[f"chg_{span}d_pct"] = _move(series, span)
        tail = series.tail(SPARK_POINTS)
        out["spark"] = [round(float(v), 4) for v in tail]
        last = str(series.index[-1])
        out["last_bar"] = (f"{last[:4]}-{last[4:6]}-{last[6:8]}"
                           if len(last) == 8 and last.isdigit() else last)

        if entry["market"] in ("US", "CA"):
            # Yahoo serves a running bar while the session is open, so the
            # last point may be a part-finished day — the move is real, the
            # volume behind it is not yet. Marked rather than hidden.
            state = session_state.state(entry["market"], out["last_bar"])
            out["session"] = {"complete": state.complete,
                              "elapsed_pct": state.elapsed_pct,
                              "local_time": state.local_time}

    if hit:
        out["rsi"] = hit["rsi"]
        out["signals"] = hit["signals"]
        out["signal_count"] = hit["signal_count"]
        out["scan_date"] = hit["scan_date"]
    return out
