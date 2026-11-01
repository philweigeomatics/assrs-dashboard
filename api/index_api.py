"""
api/index_api.py — 自建指数: storage, data, and the decision that a stock is gone.

The ledger itself is custom_index.py and pure. This file owns the three
things that are not:

PERSISTENCE, OWNER-SCOPED

Two tables (supabase/migrations/20261031_custom_index.sql). Every read and
write carries `app_user_id`; a member row is reached only through an index
the caller owns.

STORED, DAY BY DAY

Every index is kept in custom_index_day: one row per session with the open,
the close, its members' turnover, and the state at that close. The history
from the start date is computed once, on first read; after that the index
only moves forward — the daily job (and any read after a session settles)
continues from the last stored close with a few months of fresh prices, and
a stored day is never recomputed. Pages read the record; nothing on a page
downloads a price.

THE LIVE BOUNDARY

An index built today and started in 2024 has a history made with hindsight:
its members were picked knowing how they did. `rules_updated_at` records the
last time the definition changed, and everything before it is drawn as
回溯（假设）— only what came after was watched, not chosen.

NOTHING WATCHED IS REWRITTEN

The rules are fixed at creation; a member joins or leaves at the next close
not yet stored; deleting a member from the whole history is for mistakes,
so only before its first stored close. The creation day is when the
backfill is still being assembled, so on that day alone a member can be
back-dated to any day from the start — and the record is computed again.

EXITS ARE WRITTEN DOWN

Yahoo drops an acquired company's history some weeks after the deal closes.
Recomputed from live data alone, a member that left would later vanish from
its own past and the index history would silently rewrite itself. So the
first time a member is seen to have stopped trading, its last day, exit price
and price history are stored on its row, and every later computation uses
them. US: no bar for 10 sessions. A-shares: the exchange's own delisted list
must agree — a 停牌 can last months, and a halted stock comes back.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta

import pandas as pd

import custom_index as C

NAME_MAX = 40
MAX_MEMBERS = 60
MIN_MEMBERS = 2
EARLIEST = date(2015, 1, 1)
US_EXIT_GAP = 10
CN_STALE_GAP = 20
FIRST_MIGRATION = "supabase/migrations/20261031_custom_index.sql"
MIGRATION = "supabase/migrations/20261101_custom_index_day.sql"
#: Prices fetched before the base date: inverse-volatility needs three months
#: of returns at the first rebalance. The level still starts on the base date.
WARMUP = timedelta(days=120)

#: Set by api.main to (market, code) → that board's closes, read from the
#: dashboard's own 30-minute cache, so a comparison line costs no request.
#: Unset (the daily job), every comparison is the broad market.
BENCH_SOURCE = None
_A = re.compile(r"^\d{6}$")
_US = re.compile(r"^US:[A-Z][A-Z0-9.\-]{0,9}$")


class MissingTables(RuntimeError):
    pass


def _db():
    import data_manager
    return data_manager.db


def _guard(fn):
    """A missing table is a setup step, not a bug — say which step."""
    import functools

    @functools.wraps(fn)
    def wrapped(*a, **kw):
        try:
            return fn(*a, **kw)
        except (ValueError, LookupError):
            raise
        except Exception as exc:                                   # noqa: BLE001
            if "custom_index" in str(exc) and ("does not exist" in str(exc) or "PGRST" in str(exc)
                                               or "relation" in str(exc)):
                raise MissingTables(f"自建指数的表尚未创建或不是最新 — 请在 Supabase SQL 编辑器运行 {MIGRATION}"
                                    f"（从未建过表的，先运行 {FIRST_MIGRATION}）") from exc
            raise
    return wrapped


def _rows(df) -> list[dict]:
    if df is None or df.empty:
        return []
    # object first: on a float column `where(..., None)` puts the NaN straight
    # back, and one capped index beside an uncapped one makes cap_pct float.
    out = df.astype(object).where(pd.notna(df), None).to_dict("records")
    for r in out:
        for k in ("added_on", "removed_on", "base_date"):
            if r.get(k) is not None:
                r[k] = str(r[k])[:10]
    return out


def _today(market: str) -> date:
    import pytz
    tz = "Asia/Shanghai" if market == "CN" else "America/New_York"
    return datetime.now(pytz.timezone(tz)).date()


def _settling(idx: dict) -> bool:
    """Still the day the index was made, in its market's calendar: the backfill can still be edited."""
    import pytz
    made = str(idx.get("created_at") or "")[:19]
    if not made:
        return False
    tz = pytz.timezone("Asia/Shanghai" if idx["market"] == "CN" else "America/New_York")
    at = pytz.utc.localize(datetime.fromisoformat(made)).astimezone(tz).date()
    return at == _today(idx["market"])


# ── validation ───────────────────────────────────────────────────────────────
def clean_ticker(market: str, t: str) -> str:
    t = (t or "").strip().upper()
    if market == "CN":
        t = t.split(".")[0].removeprefix("CN:")
        if not _A.match(t):
            raise ValueError(f"{t} 不是 A 股代码")
        return t
    if not t.startswith("US:"):
        t = "US:" + t
    if not _US.match(t):
        raise ValueError(f"{t} 不是美股代码")
    return t


def clean_rules(market: str, method: str, rebalance: str, cap_pct, base_date) -> dict:
    if market not in ("CN", "US"):
        raise ValueError("市场只能是 A 股或美股")
    if method not in C.METHODS:
        raise ValueError(f"未知的加权方式：{method}")
    if rebalance not in C.REBALANCE:
        raise ValueError(f"未知的调仓频率：{rebalance}")
    cap = None
    if method == "cap_capped":
        cap = float(cap_pct if cap_pct is not None else C.DEFAULT_CAP * 100)
        if not 2 <= cap <= 50:
            raise ValueError("单只上限应在 2%–50% 之间")
    try:
        bd = date.fromisoformat(str(base_date)[:10])
    except ValueError:
        raise ValueError("起始日格式应为 YYYY-MM-DD") from None
    if bd < EARLIEST or bd > _today(market):
        raise ValueError(f"起始日应在 {EARLIEST} 与今天之间")
    return {"market": market, "method": method, "rebalance": rebalance, "cap_pct": cap,
            "base_date": str(bd)}


def bench_choices() -> dict[str, list[dict]]:
    """What an index can be compared with: the dashboard board's own lists."""
    import cn_index
    import us_etf
    return {"US": [{"code": t, "name": n, "group": g} for t, n, g in us_etf.UNIVERSE],
            "CN": [{"code": t, "name": n, "group": g} for t, n, g in cn_index.UNIVERSE]}


def clean_bench(market: str, bench) -> str | None:
    if bench in (None, ""):
        return None
    if bench not in {c["code"] for c in bench_choices()[market]}:
        raise ValueError(f"{bench} 不在可选的对比指数里")
    return bench


def bench_label(market: str, code: str | None) -> str:
    if not code:
        return "沪深300" if market == "CN" else "SPY（含分红）"
    name = next((c["name"] for c in bench_choices()[market] if c["code"] == code), code)
    return f"{code} {name}" if market == "US" else name


def _comparison(market: str, code: str | None) -> pd.Series | None:
    """The chosen industry index's closes, or None to fall back to the market."""
    if not code or BENCH_SOURCE is None:
        return None
    try:
        s = BENCH_SOURCE(market, code)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[index] comparison {code}: {type(exc).__name__}: {exc}"[:160])
        return None
    return s.dropna() if s is not None and not s.dropna().empty else None


def _clean_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("指数名不能为空")
    if len(name) > NAME_MAX:
        raise ValueError(f"指数名最多 {NAME_MAX} 个字符")
    return name


# ── storage ──────────────────────────────────────────────────────────────────
#: Everything but the backtest, which is tens of kilobytes and read only by the board.
_IDX_COLS = ("id,app_user_id,name,market,method,rebalance,cap_pct,base_date,base_value,bench,"
             "created_at,rules_updated_at")


def _own(user_id: int, index_id: int) -> dict:
    rows = _rows(_db().read_table("custom_index", filters={"id": int(index_id), "app_user_id": int(user_id)},
                                  columns=_IDX_COLS))
    if not rows:
        raise LookupError("没有这个指数")
    return rows[0]


def _members(index_id: int) -> list[dict]:
    rows = _rows(_db().read_table("custom_index_member", filters={"index_id": int(index_id)}))
    return sorted(rows, key=lambda r: (r["added_on"], r["ticker"]))


@_guard
def list_indexes(user_id: int) -> dict:
    rows = _rows(_db().read_table("custom_index", filters={"app_user_id": int(user_id)}, columns=_IDX_COLS))
    out = []
    for r in sorted(rows, key=lambda r: str(r.get("created_at") or ""), reverse=True):
        mem = _members(r["id"])
        out.append({**_public(r), "members": sum(1 for m in mem if not m.get("removed_on"))})
    return {"indexes": out, "methods": {k: {"label": v[0], "hint": v[1]} for k, v in C.METHODS.items()},
            "rebalance": C.REBALANCE, "benches": bench_choices()}


def _public(r: dict) -> dict:
    return {"id": int(r["id"]), "name": r["name"], "market": r["market"], "method": r["method"],
            "rebalance": r["rebalance"], "cap_pct": r.get("cap_pct"), "base_date": r["base_date"],
            "bench": r.get("bench"), "bench_label": bench_label(r["market"], r.get("bench")),
            "created_at": str(r.get("created_at") or "")[:19],
            "rules_updated_at": str(r.get("rules_updated_at") or "")[:19],
            "settling": _settling(r)}


@_guard
def create_index(user_id: int, *, name: str, market: str, method: str, rebalance: str,
                 cap_pct, base_date, members: list[dict], bench: str | None = None) -> dict:
    name = _clean_name(name)
    rules = clean_rules(market, method, rebalance, cap_pct, base_date)
    bench = clean_bench(market, bench)
    picked: dict[str, dict] = {}
    for m in members or []:
        t = clean_ticker(market, m.get("ticker", ""))
        picked[t] = {"ticker": t, "name": (m.get("name") or "")[:60] or None}
    if len(picked) < MIN_MEMBERS:
        raise ValueError(f"至少需要 {MIN_MEMBERS} 只成分股")
    if len(picked) > MAX_MEMBERS:
        raise ValueError(f"最多 {MAX_MEMBERS} 只成分股")

    db = _db()
    if _rows(db.read_table("custom_index", filters={"app_user_id": int(user_id), "name": name})):
        raise ValueError(f"已有同名指数「{name}」")
    now = datetime.utcnow().isoformat()
    # `bench` only when set: an index without one must still save on a
    # database migrated before the column existed.
    db.insert_records("custom_index", [{"app_user_id": int(user_id), "name": name, **rules,
                                        "base_value": C.BASE_VALUE, "created_at": now,
                                        "rules_updated_at": now, **({"bench": bench} if bench else {})}])
    back = _rows(db.read_table("custom_index", filters={"app_user_id": int(user_id), "name": name}))
    if not back:
        raise RuntimeError("指数已写入但读不回来 — 请刷新后确认")
    iid = int(back[0]["id"])
    db.insert_records("custom_index_member", [{"index_id": iid, **m, "added_on": rules["base_date"]}
                                              for m in picked.values()])
    return {"id": iid}


@_guard
def update_index(user_id: int, index_id: int, *, name: str, bench: str | None = None) -> dict:
    """
    Rename, or change what the index is compared with — nothing that changes
    the index itself.

    The rules (method, schedule, cap, start date) are fixed at creation, the
    way an index provider's are: a rule changed after the fact would rewrite
    every value already watched. To see the same members under another rule,
    copy the index (`copy_index`) and the two run side by side.
    """
    cur = _own(user_id, index_id)
    name = _clean_name(name)
    bench = clean_bench(cur["market"], bench)
    db = _db()
    if name != cur["name"] and _rows(db.read_table("custom_index", filters={"app_user_id": int(user_id), "name": name})):
        raise ValueError(f"已有同名指数「{name}」")
    vals: dict = {"name": name}
    if bench is not None or cur.get("bench") is not None:
        vals["bench"] = bench
    db.update_records("custom_index", vals, {"id": int(index_id), "app_user_id": int(user_id)})
    return {"id": int(index_id)}


@_guard
def copy_index(user_id: int, index_id: int, *, name: str, method: str, rebalance: str,
               cap_pct, bench: str | None = None) -> dict:
    """
    A new index with the same members over the same dates — every addition,
    removal and recorded exit — under a different rule. Same market, same
    start date, so the two differ in weighting and nothing else.

    The copy's own history is hindsight until today: its rule was chosen
    after seeing how the original did. Its live boundary is now.
    """
    src = _own(user_id, index_id)
    name = _clean_name(name)
    rules = clean_rules(src["market"], method, rebalance, cap_pct, src["base_date"])
    bench = clean_bench(src["market"], bench)
    members = _members(index_id)
    db = _db()
    if _rows(db.read_table("custom_index", filters={"app_user_id": int(user_id), "name": name})):
        raise ValueError(f"已有同名指数「{name}」")
    now = datetime.utcnow().isoformat()
    db.insert_records("custom_index", [{"app_user_id": int(user_id), "name": name, **rules,
                                        "base_value": float(src.get("base_value") or C.BASE_VALUE),
                                        "created_at": now, "rules_updated_at": now,
                                        **({"bench": bench} if bench else {})}])
    back = _rows(db.read_table("custom_index", filters={"app_user_id": int(user_id), "name": name}))
    if not back:
        raise RuntimeError("指数已写入但读不回来 — 请刷新后确认")
    iid = int(back[0]["id"])
    keep = ("ticker", "name", "added_on", "removed_on", "removal", "exit_price", "frozen_closes")
    db.insert_records("custom_index_member", [
        {"index_id": iid, **{k: m.get(k) for k in keep if m.get(k) is not None}} for m in members])
    return {"id": iid, "members": len(members)}


@_guard
def delete_index(user_id: int, index_id: int) -> dict:
    _own(user_id, index_id)
    _db().delete_records(DAY, {"index_id": int(index_id)})
    _db().delete_records("custom_index_member", {"index_id": int(index_id)})
    _db().delete_records("custom_index", {"id": int(index_id), "app_user_id": int(user_id)})
    return {"deleted": int(index_id)}


@_guard
def add_member(user_id: int, index_id: int, *, ticker: str, name: str | None,
               added_on: str | None = None) -> dict:
    """
    Joins at the next close not yet stored — today's, or tomorrow's once
    today's is written down. On the day the index was made, from any day
    since its start instead, and the record is computed again.
    """
    cur = _own(user_id, index_id)
    t = clean_ticker(cur["market"], ticker)
    today = str(_today(cur["market"]))
    settling = _settling(cur)
    if settling and added_on:
        when = str(added_on)[:10]
        if when > today:
            raise ValueError("加入日不能晚于今天")
        if when < cur["base_date"]:
            raise ValueError("加入日不能早于指数起始日")
    else:
        if added_on and str(added_on)[:10] < today:
            raise ValueError("成分变化从下一个收盘生效 — 往回改会改写已经走过的指数（只有建立当天可以补加到更早的日期）")
        when = _effective(cur)
    mem = _members(index_id)
    if any(m["ticker"] == t and not m.get("removed_on") for m in mem):
        raise ValueError(f"{t} 已在指数中")
    if sum(1 for m in mem if not m.get("removed_on")) >= MAX_MEMBERS:
        raise ValueError(f"最多 {MAX_MEMBERS} 只成分股")
    _db().insert_records("custom_index_member", [{
        "index_id": int(index_id), "ticker": t, "name": (name or "")[:60] or None, "added_on": when}])
    if settling:
        _forget(int(index_id))
    return {"ticker": t, "added_on": when}


@_guard
def remove_member(user_id: int, index_id: int, member_id: int, *, mode: str, on: str | None) -> dict:
    """
    `end` leaves at the next close not yet stored and keeps the history.
    `purge` deletes a mistake outright, which rewrites every day it was in —
    so only while nothing it was in has been watched: on the index's first
    day, or before the member's first stored close.
    """
    cur = _own(user_id, index_id)
    rows = [m for m in _members(index_id) if int(m["id"]) == int(member_id)]
    if not rows:
        raise LookupError("没有这个成分股")
    m = rows[0]
    today = str(_today(cur["market"]))
    settling = _settling(cur)
    if mode == "purge":
        last = _last_day(int(index_id))
        watched = last["d"] if last else str(date.fromisoformat(today) - timedelta(days=1))
        if not settling and m["added_on"] <= watched:
            raise ValueError("它已经在指数里走过了，删除会改写那段历史 — 请用「移出」，从下一个收盘起不再计入")
        _db().delete_records("custom_index_member", {"id": int(member_id), "index_id": int(index_id)})
    else:
        if settling and on:
            when = str(on)[:10]
            if when > today:
                raise ValueError("移出日不能晚于今天")
        else:
            if on and str(on)[:10] < today:
                raise ValueError("成分变化从下一个收盘生效 — 往回改会改写已经走过的指数（只有建立当天可以改到更早的日期）")
            when = _effective(cur)
        if when <= m["added_on"]:
            raise ValueError("移出日应晚于加入日 — 加错了的请用「删除」")
        _db().update_records("custom_index_member", {"removed_on": when, "removal": "user"},
                             {"id": int(member_id)})
    if settling:
        _forget(int(index_id))
    return {"ok": True}


# ── data ─────────────────────────────────────────────────────────────────────
#: A session is stored only once it is over and published: Tushare's daily
#: bars are out by ~16:00 Beijing, Yahoo's closing prints settle shortly
#: after 16:00 New York. An hour's margin on both.
SETTLED_AT = 17


def _now(market: str) -> datetime:
    import pytz
    return datetime.now(pytz.timezone("Asia/Shanghai" if market == "CN" else "America/New_York"))


def _settled(cal: pd.DatetimeIndex, market: str) -> pd.DatetimeIndex:
    """The sessions in `cal` that are over: before today, or today after SETTLED_AT."""
    now = _now(market)
    today = pd.Timestamp(now.date())
    return cal[(cal < today) | ((cal == today) & (now.hour >= SETTLED_AT))]


def _cn_closes(codes: list[str], start: date) -> dict:
    import data_manager
    import watchlist_board
    end = _today("CN")
    sessions = max(30, int((end - start).days * 0.68))
    ts = {data_manager.get_tushare_ticker(c): c for c in codes}
    per_call = max(1, 5500 // (sessions + 40))
    # Ten years is ~2 stocks a call: 28 members are 14 chunks, four at a time.
    # Strict: a stored index must not keep a day computed from a missing chunk.
    bars = watchlist_board.cn_daily(list(ts), start.strftime("%Y%m%d"), end.strftime("%Y%m%d"),
                                    chunk=per_call, workers=4, strict=True)
    frame = {f: pd.DataFrame({ts[k]: v[f] for k, v in bars.items()}) for f in ("close", "open", "amount")}
    for f in frame.values():
        if not f.empty:
            f.index = pd.to_datetime(f.index, format="%Y%m%d")
    idx = data_manager.get_index_data_live("000300.SH", start_date=start.strftime("%Y%m%d"),
                                           end_date=end.strftime("%Y%m%d"))
    if idx is None or idx.empty:
        raise RuntimeError("沪深300 指数数据暂时读取不到 — 请稍后重试")
    return {"close": frame["close"].sort_index(), "open": frame["open"].sort_index(),
            "amount": frame["amount"].sort_index(), "bench": idx["Close"]}


def _us_closes(tickers: list[str], start: date) -> dict:
    """
    Adjusted closes and opens, and turnover (volume × close, in dollars),
    from one bulk Yahoo download — plus a second try for any ticker the bulk
    call came back empty for, which Yahoo does now and then under load.
    """
    import yfinance as yf

    def grab(codes):
        raw = yf.download(" ".join(codes), start=str(start), interval="1d", auto_adjust=True,
                          progress=False, threads=True)
        if raw is None or raw.empty:
            return {}
        out = {}
        for f in ("Close", "Open", "Volume"):
            df = raw[f] if isinstance(raw.columns, pd.MultiIndex) else raw[[f]].rename(columns={f: codes[0]})
            df = df.copy()
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
            out[f] = df
        return out

    codes = [t.split(":", 1)[1] for t in tickers]
    want = list(dict.fromkeys(codes + ["SPY"]))
    got = grab(want)
    if not got:
        raise RuntimeError("美股行情暂时读取不到 — 请稍后重试")
    empty = [c for c in want if c not in got["Close"].columns or got["Close"][c].dropna().empty]
    if empty:
        again = grab(empty)
        for f in got:
            if again and f in again:
                got[f] = got[f].drop(columns=[c for c in empty if c in got[f].columns]).join(again[f], how="outer")
    closes = got["Close"]
    if "SPY" not in closes or closes["SPY"].dropna().empty:
        raise RuntimeError("SPY 数据暂时读取不到 — 请稍后重试")
    have = [c for c in codes if c in closes.columns and closes[c].notna().any()]
    pick = lambda df: pd.DataFrame({f"US:{c}": df[c] for c in have if c in df.columns}).sort_index()
    return {"close": pick(closes), "open": pick(got["Open"]),
            "amount": pick(got["Volume"] * closes), "bench": closes["SPY"].dropna()}


def _fetch(market: str, tickers: list[str], start: date) -> dict:
    got = (_cn_closes if market == "CN" else _us_closes)(tickers, start)
    if isinstance(got, tuple):                     # closes and calendar only, no opens or turnover
        got = {"close": got[0], "bench": got[1], "open": None, "amount": None}
    return got


def _prices(market: str, tickers: list[str], start: date) -> dict:
    return _price_cache.get_or_compute((market, tuple(sorted(tickers)), str(start)),
                                       lambda: _fetch(market, sorted(tickers), start))


def _cn_delisted() -> set[str]:
    def load():
        import watchlist_board
        df = watchlist_board._api().stock_basic(list_status="D", fields="ts_code")
        return {str(c).split(".")[0] for c in (df["ts_code"] if df is not None and not df.empty else [])}
    return _aux_cache.get_or_compute("cn-delisted", load)


def _caps(market: str, tickers: list[str], closes: pd.DataFrame, start: date) -> pd.DataFrame:
    """
    A-shares: 流通市值 history from Tushare, exact. US: today's share count ×
    each day's adjusted price — Yahoo keeps no reliable share-count history,
    so buybacks and issuance between rebalances are not reflected.
    """
    out = {}
    if market == "CN":
        import data_manager
        import watchlist_board
        api = watchlist_board._api()
        for t in tickers:
            def load(t=t):
                df = api.daily_basic(ts_code=data_manager.get_tushare_ticker(t), start_date=start.strftime("%Y%m%d"),
                                     end_date=_today("CN").strftime("%Y%m%d"), fields="trade_date,circ_mv")
                if df is None or df.empty:
                    return None
                s = pd.Series(pd.to_numeric(df["circ_mv"], errors="coerce").to_numpy(),
                              pd.to_datetime(df["trade_date"], format="%Y%m%d"))
                return s.sort_index()
            s = _aux_cache.get_or_compute(("cap", t, str(start)), load)
            if s is not None:
                out[t] = s
    else:
        import yfinance as yf
        for t in tickers:
            def shares(t=t):
                try:
                    return float(yf.Ticker(t.split(":", 1)[1]).fast_info["shares"])
                except Exception:                                  # noqa: BLE001
                    return None
            n = _aux_cache.get_or_compute(("shares", t), shares)
            if n and t in closes.columns:
                out[t] = closes[t] * n
    return pd.DataFrame(out).reindex(closes.index).ffill() if out else None


def _record_exits(index_id: int, members: list[dict], closes: pd.DataFrame, market: str) -> list[dict]:
    """Write down members that stopped trading, once; return the updated rows."""
    open_rows = {m["ticker"]: m for m in members if not m.get("removed_on")}
    if not open_rows or closes.empty:
        return members
    sub = closes[[t for t in open_rows if t in closes.columns]]
    if market == "US":
        exits = C.detect_exits(sub, US_EXIT_GAP)
    else:
        stale = C.detect_exits(sub, CN_STALE_GAP)
        exits = {t: d for t, d in stale.items() if t in _cn_delisted()} if stale else {}
    for t, last in exits.items():
        m = open_rows[t]
        s = closes[t].dropna()
        frozen = json.dumps({str(d.date()): round(float(v), 6) for d, v in s.items()})
        vals = {"removed_on": str(last.date()), "removal": "delisted",
                "exit_price": round(float(s.iloc[-1]), 6), "frozen_closes": frozen}
        _db().update_records("custom_index_member", vals, {"id": int(m["id"])})
        m.update(vals)
    return members


def _with_frozen(closes: pd.DataFrame, members: list[dict], cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Prices Yahoo no longer has, from the copy kept when the member left."""
    for m in members:
        if not m.get("frozen_closes"):
            continue
        t = m["ticker"]
        if t in closes.columns and closes[t].notna().any():
            continue
        try:
            fz = pd.Series(json.loads(m["frozen_closes"]))
        except (TypeError, ValueError):
            continue
        fz.index = pd.to_datetime(fz.index)
        closes[t] = fz.reindex(cal)
    return closes


# ── compute ──────────────────────────────────────────────────────────────────
class _Cache:
    """Expiring memo. The result key is the index's whole definition, so an
    edit is simply a different key — nothing has to be invalidated."""

    def __init__(self, maxsize: int, ttl_s: float):
        self.maxsize, self.ttl, self._d = maxsize, ttl_s, {}

    def get_or_compute(self, key, fn):
        import time
        hit = self._d.get(key)
        if hit and time.time() - hit[0] < self.ttl:
            return hit[1]
        val = fn()
        self._d[key] = (time.time(), val)
        while len(self._d) > self.maxsize:
            self._d.pop(next(iter(self._d)))
        return val


_price_cache = _Cache(maxsize=30, ttl_s=15 * 60)
_aux_cache = _Cache(maxsize=400, ttl_s=12 * 3600)


# ── the stored record ────────────────────────────────────────────────────────
DAY = "custom_index_day"
#: Rows per upsert: each carries a state of every holding, so a thousand
#: rows of a 60-name index would be a request of several megabytes.
WRITE_BATCH = 200
_series_cache = _Cache(maxsize=200, ttl_s=10 * 60)


def _num(v, nd: int = 6):
    """A float JSON can carry — NaN and infinity become null."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _days(index_id: int) -> list[dict]:
    """The stored index, oldest first — without the per-day states, which only a few dates need."""
    df = _db().read_all(DAY, filters={"index_id": int(index_id)}, columns="d,open,close,amount,events",
                        order_by="d")
    rows = df.astype(object).where(pd.notna(df), None).to_dict("records") if df is not None and not df.empty else []
    for r in rows:
        r["d"] = str(r["d"])[:10]
        if isinstance(r.get("events"), str):
            r["events"] = json.loads(r["events"])
    return rows


def _states(index_id: int, dates: list[str]) -> dict[str, dict]:
    df = _db().read_all(DAY, filters={"index_id": int(index_id)}, columns="d,state", order_by="d",
                        in_=("d", sorted(set(dates))))
    out = {}
    for r in (df.to_dict("records") if df is not None and not df.empty else []):
        s = r["state"]
        out[str(r["d"])[:10]] = json.loads(s) if isinstance(s, str) else (s or {})
    return out


def _last_day(index_id: int) -> dict | None:
    df = _db().read_table(DAY, filters={"index_id": int(index_id)}, columns="d,state", order_by="-d", limit=1)
    if df is None or df.empty:
        return None
    r = df.to_dict("records")[0]
    s = r["state"]
    return {"d": str(r["d"])[:10], "state": json.loads(s) if isinstance(s, str) else (s or {})}


def _write_days(rows: list[dict]) -> None:
    for i in range(0, len(rows), WRITE_BATCH):
        _db().insert_records(DAY, rows[i:i + WRITE_BATCH], upsert=True)


def _forget(index_id: int) -> None:
    """Drop the stored record, to be computed again on the next read (creation-day edits only)."""
    _db().delete_records(DAY, {"index_id": int(index_id)})
    try:
        _db().update_records("custom_index", {"backtest": None}, {"id": int(index_id)})
    except Exception as exc:                                       # noqa: BLE001
        print(f"[index] backtest reset {index_id}: {type(exc).__name__}"[:120])
    _series_cache._d.pop(int(index_id), None)


def _effective(idx: dict) -> str:
    """
    The day a membership change takes effect: today's close — or, when
    today's close is already stored, the next session's. A change never
    lands on a day already written down.
    """
    today = _today(idx["market"])
    last = _last_day(int(idx["id"]))
    when = today if last is None or last["d"] < str(today) else today + timedelta(days=1)
    while when.weekday() >= 5:                     # a weekend has no close; holidays the ledger skips itself
        when += timedelta(days=1)
    return str(when)


def _state_json(s: dict) -> dict:
    out = {"v": {t: _num(v) for t, v in s["v"].items()}, "p": {t: _num(v) for t, v in s["p"].items()},
           "c": {t: _num(v) for t, v in s["c"].items() if v}, "cash": _num(s["cash"])}
    for k in ("h", "g", "nd"):
        if s.get(k):
            out[k] = list(s[k])
    if s.get("t"):
        out["t"] = {t: _num(v, 6) for t, v in s["t"].items()}
    return out


def _to_rows(index_id: int, run: dict) -> list[dict]:
    by_day: dict[str, list] = {}
    for e in run["events"]:
        by_day.setdefault(e["date"], []).append(e)
    rows = []
    for k, d in enumerate(run["level"].index):
        day = str(d.date())
        rows.append({"index_id": int(index_id), "d": day, "open": _num(run["open"].iloc[k], 4),
                     "close": _num(run["level"].iloc[k], 4), "amount": _num(run["amount"].iloc[k], 0),
                     "state": _state_json(run["states"][k]), "events": by_day.get(day)})
    return rows


# ── computing what is stored ─────────────────────────────────────────────────
#: How far back the dashboard reads an index, whatever its start date.
EXTEND_YEARS = 10


def _extend_from(market: str) -> date:
    return _today(market) - timedelta(days=int(EXTEND_YEARS * 365.25))


def extendable(idx: dict) -> bool:
    """
    Whether a longer history of this index is the same index.

    A rebalancing method resets every weight to the same targets at each
    rebalance whatever date the history began on, so a run started years
    earlier coincides with the real one from its first rebalance on — the
    extra years are only extra past. Buy-and-hold never resets: started
    earlier, its weights are a different portfolio, so it keeps its own start.
    """
    return idx["method"] not in C.HOLD_METHODS and idx["rebalance"] != "never"


def _cap(idx: dict) -> float:
    return float(idx.get("cap_pct") or C.DEFAULT_CAP * 100) / 100


def _inputs(idx: dict, members: list[dict], bars: dict, start: date):
    """Prices on the market's settled calendar from `start`, with exits written down."""
    market = idx["market"]
    cal = bars["bench"].index[bars["bench"].index >= pd.Timestamp(start)]
    cal = _settled(cal, market)
    tickers = sorted({m["ticker"] for m in members})
    pick = lambda df: (df.reindex(index=cal, columns=[t for t in tickers if t in df.columns])
                       if df is not None else None)
    closes = pick(bars["close"])
    # A last session where most names that traded the one before are missing
    # is a download caught mid-publication, not a market-wide halt.
    if len(closes) > 1:
        had = closes.iloc[-2].notna()
        if had.sum() >= 5 and (closes.iloc[-1].notna() & had).sum() < 0.5 * had.sum():
            closes = closes.iloc[:-1]
            cal = closes.index
    mem = _record_exits(int(idx["id"]), [dict(m) for m in members], closes, market)
    closes = _with_frozen(closes.copy(), mem, cal)
    caps = (_caps(market, tickers, closes, start)
            if idx["method"] in ("cap", "cap_capped") else None)
    return closes, pick(bars.get("open")), pick(bars.get("amount")), mem, caps


def _targets_now(idx: dict, run: dict, closes: pd.DataFrame, caps) -> dict:
    """Where the next rebalance would reset the weights to — stored with the latest day."""
    if run["hold"] or not run["held"]:
        return {}
    last = run["level"].index[-1]
    traded = run["traded"].iloc[-1]
    names = sorted(t for t in run["held"] if traded.get(t, False))
    px = closes.ffill().loc[last]
    cap_now = caps.reindex(closes.index).ffill().loc[last] if caps is not None else None
    vol = (C.vol_frame(closes[[t for t in names if t in closes.columns]]).loc[last]
           if idx["method"] == "inv_vol" else None)
    tgt, _ = C.targets(idx["method"], names, price=px, mcap=cap_now, cap=_cap(idx), vol=vol)
    return tgt


def _finish(idx: dict, run: dict, rows: list[dict], closes: pd.DataFrame, caps, mem: list[dict]) -> None:
    """The latest day also carries the next rebalance's targets and which members have no prices at all."""
    if not rows:
        return
    rows[-1]["state"]["t"] = {t: _num(v) for t, v in _targets_now(idx, run, closes, caps).items()}
    nd = sorted({m["ticker"] for m in mem if not m.get("removed_on")}
                - {t for t in closes.columns if closes[t].notna().any()})
    if nd:
        rows[-1]["state"]["nd"] = nd


import threading  # noqa: E402

_locks: dict[int, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(index_id: int) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(int(index_id), threading.Lock())


def rebuild(idx: dict) -> int:
    """
    Compute an index from its start date and store every day of it, and —
    for a rebalancing index that starts recently — the backtest the
    dashboard reads it over. Once, when the index is first read; after that
    it only moves forward (`_advance`). Returns the days stored.
    """
    iid = int(idx["id"])
    with _lock(iid):
        if _last_day(iid) is not None:
            return 0                                   # another request got here first
        market = idx["market"]
        base = date.fromisoformat(idx["base_date"])
        members = _members(iid)
        if not members:
            raise LookupError("指数里还没有成分股")
        ext = _extend_from(market)
        extend = extendable(idx) and base > ext + timedelta(days=30)
        start = (ext if extend else base) - WARMUP
        bars = _prices(market, [m["ticker"] for m in members], start)
        closes, opens, amounts, mem, caps = _inputs(idx, members, bars, start)
        if len(closes) > 6:
            _check_moves(closes, closes.index[-6], market)     # the newest days, whose adjustments may lag
        run = C.build(closes, mem, method=idx["method"], rebalance=idx["rebalance"], base_date=base,
                      base_value=float(idx.get("base_value") or C.BASE_VALUE), cap=_cap(idx), caps=caps,
                      opens=opens, amounts=amounts, states=True)
        rows = _to_rows(iid, run)
        _finish(idx, run, rows, closes, caps, mem)
        backtest = None
        if extend:
            # Founders moved back to the earlier start; later additions and
            # removals keep their dates. Joined to the index on its start date.
            early = [dict(m, added_on=str(ext)) if m["added_on"] == idx["base_date"] else dict(m) for m in mem]
            lv = C.build(closes, early, method=idx["method"], rebalance=idx["rebalance"], base_date=ext,
                         cap=_cap(idx), caps=caps)["level"]
            first = run["level"].index[0]
            if first in lv.index and lv.index[0] < first:
                pre = lv[lv.index < first] * (float(run["level"].iloc[0]) / float(lv.loc[first]))
                backtest = {"from": str(pre.index[0].date()), "dates": [str(d.date()) for d in pre.index],
                            "close": [_num(v, 4) for v in pre.to_numpy()]}
        _write_days(rows)
        if backtest is not None:
            _db().update_records("custom_index", {"backtest": backtest}, {"id": iid})
        _series_cache._d.pop(iid, None)
        return len(rows)


def _advance(market: str, idxs: list[dict]) -> list[dict]:
    """
    Carry indexes forward from their last stored close: one download for all
    of them, from a little before the earliest last day (inverse-volatility
    needs three months of returns), and only the sessions after each one's
    last day are computed and stored. Nothing already stored is touched.
    """
    report = []
    lasts, todo = {}, []
    for idx in idxs:
        last = _last_day(int(idx["id"]))
        if last is None:
            try:
                n = rebuild(idx)
                report.append({"id": int(idx["id"]), "name": idx["name"], "ok": True, "rebuilt": True, "added": n})
            except Exception as exc:                               # noqa: BLE001
                report.append({"id": int(idx["id"]), "name": idx["name"], "ok": False, "error": str(exc)[:160]})
            continue
        lasts[int(idx["id"])] = last
        todo.append(idx)
    if not todo:
        return report
    members = {int(i["id"]): _members(int(i["id"])) for i in todo}
    start = min(date.fromisoformat(l["d"]) for l in lasts.values()) - WARMUP
    tickers = sorted({m["ticker"] for ms in members.values() for m in ms}
                     | {t for l in lasts.values() for t in l["state"].get("v", {})})
    try:
        bars = _prices(market, tickers, start)
    except Exception as exc:                                       # noqa: BLE001
        return report + [{"id": int(i["id"]), "name": i["name"], "ok": False, "error": f"行情: {exc}"[:160]}
                         for i in todo]
    for idx in todo:
        iid = int(idx["id"])
        last = lasts[iid]
        try:
            with _lock(iid):
                mem = members[iid]
                before = {m["ticker"] for m in mem if m.get("removal") == "delisted"}
                closes, opens, amounts, mem, caps = _inputs(idx, mem, bars, start)
                d0 = pd.Timestamp(last["d"])
                if closes.index[-1] <= d0:
                    report.append({"id": iid, "name": idx["name"], "ok": True, "added": 0, "as_of": last["d"],
                                   "level": _num(sum(last["state"]["v"].values()) + last["state"]["cash"], 2),
                                   "new_exits": []})
                    continue
                _check_fresh(last, closes, mem)
                _check_moves(closes, d0, market)
                run = C.build(closes, mem, method=idx["method"], rebalance=idx["rebalance"],
                              base_date=date.fromisoformat(idx["base_date"]), cap=_cap(idx), caps=caps,
                              opens=opens, amounts=amounts, resume={**last["state"], "date": last["d"]},
                              states=True)
                rows = _to_rows(iid, run)
                _finish(idx, run, rows, closes, caps, mem)
                _write_days(rows)
                _series_cache._d.pop(iid, None)
                gone = sorted(m["ticker"] for m in mem if m.get("removal") == "delisted" and m["ticker"] not in before)
                report.append({"id": iid, "name": idx["name"], "ok": True, "added": len(rows),
                               "as_of": rows[-1]["d"] if rows else last["d"],
                               "level": rows[-1]["close"] if rows else None, "new_exits": gone})
        except Exception as exc:                                   # noqa: BLE001
            report.append({"id": iid, "name": idx["name"], "ok": False, "error": str(exc)[:160]})
    return report


#: A-share daily limits are ±10% (main board), ±20% (创业板, 科创板), ±30%
#: (北交所). An adjusted move beyond that is an adjustment not yet applied.
CN_LIMIT = 0.31
#: US moves this large are checked against Yahoo's split record.
US_SUSPECT = (-0.45, 0.8)


def _check_moves(closes: pd.DataFrame, since: pd.Timestamp, market: str) -> None:
    """
    The days about to be stored, checked for a price adjustment that has not
    reached the data yet — a 送转 without its factor, a split Yahoo has not
    applied. Stored once, such a day would be wrong for good; held back, the
    next run stores it correctly. A stock's first five sessions are exempt
    (A-share listings have no limit then).
    """
    r = closes.pct_change(fill_method=None)
    seen = closes.notna().cumsum()
    new = r.loc[r.index > since]
    for t in new.columns:
        for d, v in new[t].dropna().items():
            if seen.at[d, t] <= 5:
                continue
            if market == "CN" and abs(v) > CN_LIMIT:
                raise RuntimeError(f"{t} {d.date()} 复权后 {v * 100:+.0f}% 超过涨跌停 — 复权数据可能尚未更新，这一天暂不存")
            if market == "US" and not US_SUSPECT[0] < v < US_SUSPECT[1] and _split_on(t, d):
                raise RuntimeError(f"{t} {d.date()} {v * 100:+.0f}% 恰逢拆股 — 雅虎尚未复权，这一天暂不存")


def _split_on(ticker: str, d: pd.Timestamp) -> bool:
    try:
        import yfinance as yf
        s = yf.Ticker(ticker.split(":", 1)[1]).splits
        idx = s.index.tz_localize(None).normalize() if s.index.tz is not None else s.index.normalize()
        return bool((idx == d.normalize()).any())
    except Exception:                                              # noqa: BLE001
        return True                                                # can't tell: hold the day back


def _check_fresh(last: dict, closes: pd.DataFrame, members: list[dict]) -> None:
    """
    A holding that traded at the last stored close and is still a member must
    have a price in the fresh download. Without one the day would be stored
    with it frozen flat — a download failure written down for good. Better
    not to store the day; the next run tries again.
    """
    d0 = pd.Timestamp(last["d"])
    halted = set(last["state"].get("h", []))
    nxt = closes.index[closes.index > d0]
    if not len(nxt):
        return
    by_t: dict[str, list] = {}
    for m in members:
        by_t.setdefault(m["ticker"], []).append(m)
    for t in last["state"].get("v", {}):
        if t in halted or not C._member_on(by_t.get(t, []), nxt[0]):
            continue
        if t not in closes.columns or closes[t].loc[closes.index >= d0].dropna().empty:
            raise RuntimeError(f"{t} 没有任何新行情 — 下载失败，或已被收购退市（若是后者，请把它移出指数）")


# ── reading the record ───────────────────────────────────────────────────────
#: A failed catch-up is not retried on every page view.
_catch_up_tried: dict[str, float] = {}


def _latest_session(market: str) -> pd.Timestamp | None:
    """The market's latest finished session, off the dashboard's own cached board — no download."""
    if BENCH_SOURCE is None:
        return None
    try:
        s = BENCH_SOURCE(market, "SPY" if market == "US" else "000300.SH")
    except Exception:                                              # noqa: BLE001
        return None
    if s is None or s.dropna().empty:
        return None
    done = _settled(s.dropna().index, market)
    return done[-1] if len(done) else None


def _catch_up(market: str, idxs: list[dict], last: dict[int, str | None]) -> bool:
    """
    Indexes behind the market's latest session are carried forward now
    rather than waiting for the nightly job — a US close is stored the same
    evening. Returns whether anything was written.
    """
    import time
    latest = _latest_session(market)
    if latest is None:
        return False
    stale = [i for i in idxs if last.get(int(i["id"])) and pd.Timestamp(last[int(i["id"])]) < latest]
    if not stale or time.time() - _catch_up_tried.get(market, 0) < 600:
        return False
    _catch_up_tried[market] = time.time()
    try:
        return any(r.get("ok") and r.get("added") for r in _advance(market, stale))
    except Exception as exc:                                       # noqa: BLE001
        print(f"[index] catch-up {market}: {type(exc).__name__}: {exc}"[:160])
        return False


def _record(idx: dict) -> list[dict]:
    """The stored days, computing them first if this index has none yet."""
    iid = int(idx["id"])
    days = _days(iid)
    if not days:
        rebuild(idx)
        days = _days(iid)
    if not days:
        raise LookupError("起始日之后没有任何成分股有行情")
    if _catch_up(idx["market"], [idx], {iid: days[-1]["d"]}):
        days = _days(iid)
    return days


def _backtest(index_id: int) -> dict | None:
    df = _db().read_table("custom_index", filters={"id": int(index_id)}, columns="backtest")
    if df is None or df.empty:
        return None
    b = df.iloc[0].get("backtest")
    b = json.loads(b) if isinstance(b, str) else b
    return b if isinstance(b, dict) and b.get("dates") else None


def _series(idx: dict, days: list[dict]) -> tuple[pd.Series, str | None]:
    """The index's closes for the dashboard: its backtest, if it has one, then its own record."""
    iid = int(idx["id"])
    key = (days[-1]["d"], len(days))
    hit = _series_cache._d.get(iid)
    import time
    if hit and hit[1][0] == key and time.time() - hit[0] < _series_cache.ttl:
        return hit[1][1], hit[1][2]
    own = pd.Series([r["close"] for r in days], pd.to_datetime([r["d"] for r in days]), dtype=float)
    bt = _backtest(iid)
    if bt:
        pre = pd.Series(bt["close"], pd.to_datetime(bt["dates"]), dtype=float)
        lv, since = pd.concat([pre[pre.index < own.index[0]], own]), bt["from"]
    else:
        lv, since = own, None
    _series_cache._d[iid] = (time.time(), (key, lv, since))
    return lv, since


def _split(level: pd.Series, opens: pd.Series, days: int) -> dict | None:
    """Over the last `days` sessions: the move made overnight (open vs the close before) and in the session."""
    if len(level) <= days:
        return None
    lv, op = level.iloc[-days - 1:], opens.iloc[-days - 1:]
    gap = (op / lv.shift(1)).iloc[1:]
    sess = (lv / op).iloc[1:]
    ok = gap.notna() & sess.notna()
    if ok.sum() < max(1, 0.8 * days):
        return None
    return {"gap": _num((gap[ok].prod() - 1) * 100, 2), "session": _num((sess[ok].prod() - 1) * 100, 2)}


def _payload(idx: dict, members: list[dict], days: list[dict], bench: pd.Series | None) -> dict:
    """The detail view, from the stored record alone."""
    iid = int(idx["id"])
    level = pd.Series([r["close"] for r in days], pd.to_datetime([r["d"] for r in days]), dtype=float)
    opens = pd.Series([r["open"] for r in days], level.index, dtype=float)
    amount = pd.Series([r["amount"] for r in days], level.index, dtype=float)
    n = len(days)
    win = {"1d": 1, "5d": 5, "1m": 21, "3m": 63, "all": None}
    at = {k: (0 if v is None or v >= n else n - 1 - v) for k, v in win.items()}
    st = _states(iid, [days[i]["d"] for i in set(at.values()) | {n - 1}])
    last = st.get(days[-1]["d"], {})
    lv_last = float(level.iloc[-1])
    held = set(last.get("v", {}))
    halted = set(last.get("h", []))
    tgt = last.get("t") or {}
    prev = st.get(days[at["1d"]]["d"], {}) if n > 1 else {}

    def contrib(k):
        s0 = st.get(days[at[k]]["d"], {}) if at[k] > 0 else {}
        v0 = float(level.iloc[at[k]])
        c1, c0 = last.get("c", {}), s0.get("c", {})
        return {t: (c1.get(t, 0) - c0.get(t, 0)) / v0 * 100 for t in set(c1) | set(c0)} if v0 > 0 else {}

    cons = {k: contrib(k) for k in win}
    names = {}
    if idx["market"] == "CN":
        from api.portfolio_api import _cn_names
        names = _cn_names()
    as_of = level.index[-1]
    rows = []
    for m in members:
        t = m["ticker"]
        removed = m.get("removed_on")
        if t in held:
            status = "halted" if t in halted else "active"
        elif removed and pd.Timestamp(removed) <= as_of:
            status = "delisted" if m.get("removal") == "delisted" else "removed"
        elif t in last.get("nd", []):
            status = "nodata"
        else:
            status = "pending"
        v_prev = prev.get("v", {}).get(t)
        dv = last.get("c", {}).get(t, 0) - prev.get("c", {}).get(t, 0)
        rows.append({
            "id": m.get("id"), "ticker": t, "name": names.get(t) or m.get("name") or t, "status": status,
            "weight": C._f(last["v"][t] / lv_last * 100, 2) if t in held and lv_last > 0 else None,
            "target": C._f(tgt[t] * 100, 2) if t in tgt else None,
            "price": C._f(last.get("p", {}).get(t), 2) if t in held else C._f(m.get("exit_price"), 2),
            "ret_1d": C._f(dv / v_prev * 100, 2) if t in held and v_prev else None,
            "contrib": {k: C._f(c.get(t), 2) for k, c in cons.items()},
            "added_on": str(m["added_on"])[:10], "removed_on": str(removed)[:10] if removed else None,
            "exit_price": C._f(m.get("exit_price"), 2),
        })
    order = {"active": 0, "halted": 1, "pending": 2, "nodata": 3, "removed": 4, "delisted": 5}
    rows.sort(key=lambda r: (order[r["status"]], -(r["weight"] or 0), r["ticker"]))

    label = {m["ticker"]: (names.get(m["ticker"]) or m.get("name") or m["ticker"]) for m in members}
    events = [dict(e, name=label.get(e["ticker"], e["ticker"])) if e.get("ticker") else dict(e)
              for r in days for e in (r.get("events") or [])][::-1]

    b = None
    if bench is not None and not bench.dropna().empty:
        bb = bench.reindex(level.index).ffill().dropna()
        if len(bb):
            reb = bb / bb.iloc[0] * float(level.loc[bb.index[0]])
            b = {"series": [C._f(v, 2) for v in reb.reindex(level.index).to_numpy()],
                 "ret": {**{k: C._f(C._ret(bb, v), 2) for k, v in C.WINDOWS.items()},
                         "ytd": C._f(C._ytd(bb), 2), "all": C._f((bb.iloc[-1] / bb.iloc[0] - 1) * 100, 2)}}
    ma = level.rolling(200).mean()
    live = str(idx.get("rules_updated_at") or "")[:10] or None
    hold = idx["method"] in C.HOLD_METHODS or idx["rebalance"] == "never"
    a20 = amount.dropna().tail(21)
    return {
        "as_of": str(as_of.date()), "start": days[0]["d"],
        "live_from": live if live and pd.Timestamp(live) > level.index[0] else None,
        "stats": {**C.stats(level),
                  "split": {k: _split(level, opens, v) for k, v in (("1d", 1), ("1m", 21), ("3m", 63), ("1y", 252))},
                  "amount": C._f(a20.iloc[-1], 0) if len(a20) else None,
                  "amount_x20": C._f(a20.iloc[-1] / a20.iloc[:-1].mean(), 2) if len(a20) > 10 and a20.iloc[:-1].mean() > 0 else None},
        "chart": {"dates": [r["d"] for r in days], "level": [C._f(v, 2) for v in level.to_numpy()],
                  "ma200": [C._f(v, 2) for v in ma.to_numpy()], "bench": b["series"] if b else None},
        "bench_ret": b["ret"] if b else None,
        "members": rows, "events": events,
        "guessed_caps": list(last.get("g", [])),
        "next_rebalance": None if hold else C.next_rebalance(as_of, idx["rebalance"]),
        "hold": hold,
    }


@_guard
def compute(user_id: int, index_id: int) -> dict:
    """An index's detail view, read from its stored record."""
    idx = _own(user_id, index_id)
    if not _members(index_id):
        raise LookupError("指数里还没有成分股")
    days = _record(idx)
    members = _members(index_id)                   # after: computing may have written an exit down
    market = idx["market"]
    cmp = _comparison(market, idx.get("bench"))
    bench = cmp if cmp is not None else _comparison(market, "SPY" if market == "US" else "000300.SH")
    out = _payload(idx, members, days, bench)
    out["index"] = _public(idx)
    out["today"] = str(_today(market))
    out["effective"] = _effective(idx)
    out["bench_label"] = bench_label(market, idx.get("bench") if cmp is not None else None)
    out["bench_note"] = (None if cmp is not None or not idx.get("bench")
                         else f"{bench_label(market, idx['bench'])} 暂时读取不到，改用大盘对比")
    out["method_label"] = C.METHODS[idx["method"]][0]
    out["rebalance_label"] = C.REBALANCE[idx["rebalance"]]
    return out


@_guard
def board_rows(user_id: int, market: str) -> dict:
    """
    The caller's indexes in one market, as rows of the dashboard board —
    read with the same arithmetic as every ETF and theme index beside them,
    from what is stored: no price is downloaded here. An index whose record
    is still too short for a 200-day line is listed by name instead.
    """
    import us_etf
    from concurrent.futures import ThreadPoolExecutor
    idxs = [r for r in _rows(_db().read_table("custom_index", filters={"app_user_id": int(user_id)},
                                              columns=_IDX_COLS))
            if r["market"] == market]
    if not idxs:
        return {"rows": [], "young": [], "failed": []}

    def load(r):
        try:
            return int(r["id"]), _days(int(r["id"])), None
        except Exception as exc:                                   # noqa: BLE001
            return int(r["id"]), None, exc

    with ThreadPoolExecutor(max_workers=8) as pool:
        got = {i: (d, e) for i, d, e in pool.map(load, idxs)}
    if _catch_up(market, idxs, {i: (d[-1]["d"] if d else None) for i, (d, _) in got.items()}):
        with ThreadPoolExecutor(max_workers=8) as pool:
            got = {i: (d, e) for i, d, e in pool.map(load, idxs)}
    bench_t = "SPY" if market == "US" else "000300.SH"
    base = _comparison(market, bench_t)
    rows, young, failed = [], [], []
    for r in idxs:
        iid = int(r["id"])
        days, err = got[iid]
        try:
            if err is not None:
                raise err
            if not days:
                if not _members(iid):
                    continue
                rebuild(r)
                days = _days(iid)
            if not days:
                raise LookupError("起始日之后没有任何成分股有行情")
            level, since = _series(r, days)
        except Exception as exc:                                   # noqa: BLE001
            failed.append({"id": iid, "name": r["name"], "error": str(exc)[:120]})
            continue
        cmp = _comparison(market, r.get("bench"))
        x = us_etf.row(f"IDX{iid}", level, cmp if cmp is not None else base, name=r["name"], group=us_etf.MINE,
                       bench_ticker=r.get("bench") if cmp is not None else bench_t)
        if x is None:
            young.append({"id": iid, "name": r["name"], "days": len(level), "hold": not extendable(r)})
            continue
        x["index_id"] = iid
        x["method"] = C.METHODS[r["method"]][0]
        x["extended_from"] = since
        x["rs_vs"] = (r["bench"] if market == "US" else bench_label(market, r["bench"])) if cmp is not None else None
        rows.append(x)
    return {"rows": rows, "young": young, "failed": failed}


@_guard
def stretch_detail(user_id: int, index_id: int) -> dict:
    """The board's detail view (price, 50/200-day, stretch bands) for one of the caller's indexes."""
    import us_etf
    idx = _own(user_id, index_id)
    days = _record(idx)
    level, since = _series(idx, days)
    t = f"IDX{index_id}"
    d = us_etf.detail(pd.DataFrame({t: level}), t, names={t: idx["name"]})
    d["extended_from"] = since
    d["start"] = days[0]["d"]
    return d


def refresh_all() -> list[dict]:
    """
    The daily job: every user's every index carried forward from its last
    stored close — one download per market — and any index without a
    record yet computed from its start. Writing an exit down happens here
    too, so an index nobody opens still records a member's acquisition
    before Yahoo forgets it.
    """
    report = []
    rows = _rows(_db().read_table("custom_index", columns=_IDX_COLS))
    for market in ("US", "CN"):
        mine = [r for r in rows if r["market"] == market]
        mine = [r for r in mine if _members(int(r["id"]))]
        if mine:
            report += _advance(market, mine)
    return report
