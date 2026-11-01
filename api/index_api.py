"""
api/index_api.py — 自建指数: storage, data, and the decision that a stock is gone.

The ledger itself is custom_index.py and pure. This file owns the three
things that are not:

PERSISTENCE, OWNER-SCOPED

Two tables (supabase/migrations/20261031_custom_index.sql). Every read and
write carries `app_user_id`; a member row is reached only through an index
the caller owns.

THE LIVE BOUNDARY

An index built today and started in 2024 has a history made with hindsight:
its members were picked knowing how they did. `rules_updated_at` records the
last time the definition changed, and everything before it is drawn as
回溯（假设）— only what came after was watched, not chosen.

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
MIGRATION = "supabase/migrations/20261031_custom_index.sql"
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
    def wrapped(*a, **kw):
        try:
            return fn(*a, **kw)
        except (ValueError, LookupError):
            raise
        except Exception as exc:                                   # noqa: BLE001
            if "custom_index" in str(exc) and ("does not exist" in str(exc) or "PGRST" in str(exc)
                                               or "relation" in str(exc)):
                raise MissingTables(f"自建指数的表尚未创建或不是最新 — 请在 Supabase SQL 编辑器运行 {MIGRATION}") from exc
            raise
    wrapped.__name__ = fn.__name__
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
def _own(user_id: int, index_id: int) -> dict:
    rows = _rows(_db().read_table("custom_index", filters={"id": int(index_id), "app_user_id": int(user_id)}))
    if not rows:
        raise LookupError("没有这个指数")
    return rows[0]


def _members(index_id: int) -> list[dict]:
    rows = _rows(_db().read_table("custom_index_member", filters={"index_id": int(index_id)}))
    return sorted(rows, key=lambda r: (r["added_on"], r["ticker"]))


@_guard
def list_indexes(user_id: int) -> dict:
    rows = _rows(_db().read_table("custom_index", filters={"app_user_id": int(user_id)}))
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
            "rules_updated_at": str(r.get("rules_updated_at") or "")[:19]}


@_guard
def create_index(user_id: int, *, name: str, market: str, method: str, rebalance: str,
                 cap_pct, base_date, members: list[dict], bench: str | None = None) -> dict:
    name = _clean_name(name)
    rules = clean_rules(market, method, rebalance, cap_pct, base_date)
    bench = clean_bench(market, bench)
    picked: dict[str, dict] = {}
    for m in members or []:
        t = clean_ticker(market, m.get("ticker", ""))
        picked[t] = {"ticker": t, "name": (m.get("name") or "")[:60] or None,
                     "weight_pct": float(m["weight_pct"]) if m.get("weight_pct") not in (None, "") else None}
    if len(picked) < MIN_MEMBERS:
        raise ValueError(f"至少需要 {MIN_MEMBERS} 只成分股")
    if len(picked) > MAX_MEMBERS:
        raise ValueError(f"最多 {MAX_MEMBERS} 只成分股")
    if method == "custom" and not any((m["weight_pct"] or 0) > 0 for m in picked.values()):
        raise ValueError("自定权重需要至少一只设定了权重")

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
def update_rules(user_id: int, index_id: int, *, name: str, method: str, rebalance: str,
                 cap_pct, base_date, bench: str | None = None) -> dict:
    """
    A new rule recomputes the whole history, so the live boundary moves to
    now. The comparison index is not a rule — it changes nothing about the
    index itself — so changing it leaves the boundary where it was.
    """
    cur = _own(user_id, index_id)
    name = _clean_name(name)
    rules = clean_rules(cur["market"], method, rebalance, cap_pct, base_date)
    bench = clean_bench(cur["market"], bench)
    db = _db()
    if name != cur["name"] and _rows(db.read_table("custom_index", filters={"app_user_id": int(user_id), "name": name})):
        raise ValueError(f"已有同名指数「{name}」")
    changed = any(str(cur.get(k)) != str(rules[k]) for k in ("method", "rebalance", "cap_pct", "base_date"))
    vals = {"name": name, **{k: v for k, v in rules.items() if k != "market"}}
    if bench is not None or cur.get("bench") is not None:
        vals["bench"] = bench
    if changed:
        vals["rules_updated_at"] = datetime.utcnow().isoformat()
    db.update_records("custom_index", vals, {"id": int(index_id), "app_user_id": int(user_id)})
    # Founding members move with the start date; later additions keep theirs.
    if rules["base_date"] != cur["base_date"]:
        for m in _members(index_id):
            if m["added_on"] == cur["base_date"] or m["added_on"] < rules["base_date"]:
                db.update_records("custom_index_member", {"added_on": rules["base_date"]}, {"id": int(m["id"])})
    return {"id": int(index_id), "recomputed": changed}


@_guard
def delete_index(user_id: int, index_id: int) -> dict:
    _own(user_id, index_id)
    _db().delete_records("custom_index_member", {"index_id": int(index_id)})
    _db().delete_records("custom_index", {"id": int(index_id), "app_user_id": int(user_id)})
    return {"deleted": int(index_id)}


@_guard
def add_member(user_id: int, index_id: int, *, ticker: str, name: str | None,
               added_on: str | None, weight_pct) -> dict:
    cur = _own(user_id, index_id)
    t = clean_ticker(cur["market"], ticker)
    when = str(added_on or _today(cur["market"]))[:10]
    if when < cur["base_date"]:
        raise ValueError("加入日不能早于指数起始日")
    if when > str(_today(cur["market"])):
        raise ValueError("加入日不能晚于今天")
    mem = _members(index_id)
    if any(m["ticker"] == t and not m.get("removed_on") for m in mem):
        raise ValueError(f"{t} 已在指数中")
    if sum(1 for m in mem if not m.get("removed_on")) >= MAX_MEMBERS:
        raise ValueError(f"最多 {MAX_MEMBERS} 只成分股")
    _db().insert_records("custom_index_member", [{
        "index_id": int(index_id), "ticker": t, "name": (name or "")[:60] or None, "added_on": when,
        "weight_pct": float(weight_pct) if weight_pct not in (None, "") else None}])
    return {"ticker": t, "added_on": when}


@_guard
def remove_member(user_id: int, index_id: int, member_id: int, *, mode: str, on: str | None) -> dict:
    """`end` keeps the history (removed from a date); `purge` deletes a mistake outright."""
    cur = _own(user_id, index_id)
    rows = [m for m in _members(index_id) if int(m["id"]) == int(member_id)]
    if not rows:
        raise LookupError("没有这个成分股")
    m = rows[0]
    if mode == "purge":
        _db().delete_records("custom_index_member", {"id": int(member_id), "index_id": int(index_id)})
    else:
        when = str(on or _today(cur["market"]))[:10]
        if when <= m["added_on"]:
            raise ValueError("移出日应晚于加入日 — 加错了的请用「删除」")
        _db().update_records("custom_index_member", {"removed_on": when, "removal": "user"},
                             {"id": int(member_id)})
    return {"ok": True}


@_guard
def set_weight(user_id: int, index_id: int, member_id: int, weight_pct) -> dict:
    _own(user_id, index_id)
    w = float(weight_pct)
    if not 0 <= w <= 100:
        raise ValueError("权重应在 0–100 之间")
    _db().update_records("custom_index_member", {"weight_pct": w},
                         {"id": int(member_id), "index_id": int(index_id)})
    return {"ok": True}


# ── data ─────────────────────────────────────────────────────────────────────
def _cn_closes(codes: list[str], start: date) -> tuple[pd.DataFrame, pd.Series]:
    import data_manager
    import watchlist_board
    end = _today("CN")
    sessions = max(30, int((end - start).days * 0.68))
    ts = {data_manager.get_tushare_ticker(c): c for c in codes}
    per_call = max(1, 5500 // (sessions + 40))
    bars = watchlist_board._cn_bars(list(ts), start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), chunk=per_call)
    closes = pd.DataFrame({ts[k]: v for k, v in bars.items()})
    if not closes.empty:
        closes.index = pd.to_datetime(closes.index, format="%Y%m%d")
    idx = data_manager.get_index_data_live("000300.SH", start_date=start.strftime("%Y%m%d"),
                                           end_date=end.strftime("%Y%m%d"))
    if idx is None or idx.empty:
        raise RuntimeError("沪深300 指数数据暂时读取不到 — 请稍后重试")
    return closes.sort_index(), idx["Close"]


def _us_closes(tickers: list[str], start: date) -> tuple[pd.DataFrame, pd.Series]:
    import yfinance as yf
    codes = [t.split(":", 1)[1] for t in tickers]
    want = list(dict.fromkeys(codes + ["SPY"]))
    raw = yf.download(" ".join(want), start=str(start), interval="1d", auto_adjust=True,
                      progress=False, threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("美股行情暂时读取不到 — 请稍后重试")
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": want[0]})
    closes = closes.copy()
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
    closes.index = closes.index.normalize()
    if "SPY" not in closes or closes["SPY"].dropna().empty:
        raise RuntimeError("SPY 数据暂时读取不到 — 请稍后重试")
    bench = closes["SPY"].dropna()
    out = pd.DataFrame({f"US:{c}": closes[c] for c in codes if c in closes.columns and closes[c].notna().any()})
    return out.sort_index(), bench


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
_result_cache = _Cache(maxsize=30, ttl_s=15 * 60)


@_guard
def compute(user_id: int, index_id: int) -> dict:
    idx = _own(user_id, index_id)
    members = _members(index_id)
    if not members:
        raise LookupError("指数里还没有成分股")
    key = (int(index_id), json.dumps([idx, members], sort_keys=True, default=str))

    def run():
        market = idx["market"]
        start = date.fromisoformat(idx["base_date"])
        tickers = sorted({m["ticker"] for m in members})
        closes, bench = _prices(market, tickers, start - WARMUP)
        cal = bench.index[bench.index >= pd.Timestamp(start - WARMUP)]
        closes = closes.reindex(cal)
        mem = _record_exits(index_id, [dict(m) for m in members], closes, market)
        closes = _with_frozen(closes.copy(), mem, cal)
        cap = float(idx.get("cap_pct") or C.DEFAULT_CAP * 100) / 100
        caps = (_caps(market, tickers, closes, start)
                if idx["method"] in ("cap", "cap_capped") else None)
        ledger = C.build(closes, mem, method=idx["method"], rebalance=idx["rebalance"],
                         base_date=start, base_value=float(idx.get("base_value") or C.BASE_VALUE),
                         cap=cap, caps=caps)
        names = {}
        if market == "CN":
            from api.portfolio_api import _cn_names
            names = _cn_names()
        cmp = _comparison(market, idx.get("bench"))
        out = C.summarise(ledger, members=mem, names=names, method=idx["method"],
                          rebalance=idx["rebalance"], cap=cap, bench=cmp if cmp is not None else bench,
                          live_from=str(idx.get("rules_updated_at") or "")[:10] or None,
                          closes=closes, caps=caps)
        label = {m["ticker"]: (names.get(m["ticker"]) or m.get("name") or m["ticker"]) for m in mem}
        for e in out["events"]:
            if e.get("ticker"):
                e["name"] = label.get(e["ticker"], e["ticker"])
        out["index"] = _public(idx)
        out["bench_label"] = bench_label(market, idx.get("bench") if cmp is not None else None)
        out["bench_note"] = (None if cmp is not None or not idx.get("bench")
                             else f"{bench_label(market, idx['bench'])} 暂时读取不到，改用大盘对比")
        out["method_label"] = C.METHODS[idx["method"]][0]
        out["rebalance_label"] = C.REBALANCE[idx["rebalance"]]
        return out

    return _result_cache.get_or_compute(key, run)


# ── one download per market ──────────────────────────────────────────────────
#: The last bulk download per market: (time, start, tickers, closes, bench).
#: The dashboard shows every index at once; fetching each index's members
#: separately would be one Yahoo request per index per refresh. One request
#: for the union, sliced per index, keeps the request count flat however
#: many indexes there are.
_union: dict[str, tuple] = {}
UNION_TTL = 15 * 60


def _fetch(market: str, tickers: list[str], start: date):
    return (_cn_closes if market == "CN" else _us_closes)(tickers, start)


def _prices(market: str, tickers: list[str], start: date):
    import time
    hit = _union.get(market)
    if hit and time.time() - hit[0] < UNION_TTL and hit[1] <= start and set(tickers) <= hit[2]:
        closes, bench = hit[3], hit[4]
        return closes[[t for t in tickers if t in closes.columns]], bench
    return _price_cache.get_or_compute((market, tuple(tickers), str(start)),
                                       lambda: _fetch(market, tickers, start))


def _prefetch(market: str, idxs: list[dict], members: dict[int, list[dict]],
              start: date | None = None) -> None:
    import time
    tickers = sorted({m["ticker"] for i in idxs for m in members[i["id"]]})
    if not tickers:
        return
    start = start or min(date.fromisoformat(i["base_date"]) for i in idxs)
    hit = _union.get(market)
    if hit and time.time() - hit[0] < UNION_TTL and hit[1] <= start and set(tickers) <= hit[2]:
        return
    closes, bench = _fetch(market, tickers, start)
    _union[market] = (time.time(), start, set(tickers), closes, bench)


def _level(out: dict) -> pd.Series:
    c = out["chart"]
    return pd.Series(c["level"], pd.to_datetime(c["dates"]), dtype=float)


# ── reading an index further back than it starts ─────────────────────────────
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


def extended_level(user_id: int, index_id: int) -> tuple[pd.Series, str | None]:
    """
    The index's level for the dashboard: the real one, or — when the start
    date is recent and the method rebalances — the same rules replayed from
    up to EXTEND_YEARS earlier, scaled to end on the real level. Founding
    members are moved back to the earlier start; later additions and removals
    keep their dates; a member that listed later joins when it first trades.
    Returns (level, the date the extension starts from, or None).
    """
    idx = _own(user_id, index_id)
    out = compute(user_id, index_id)
    real = _level(out)
    market = idx["market"]
    ext = _extend_from(market)
    base = date.fromisoformat(idx["base_date"])
    if not extendable(idx) or base <= ext + timedelta(days=30):
        return real, None
    members = _members(index_id)
    key = ("ext", int(index_id), json.dumps([idx, members], sort_keys=True, default=str))

    def run():
        mem = [dict(m, added_on=str(ext)) if m["added_on"] == idx["base_date"] else dict(m) for m in members]
        tickers = sorted({m["ticker"] for m in mem})
        closes, bench = _prices(market, tickers, ext - WARMUP)
        cal = bench.index[bench.index >= pd.Timestamp(ext - WARMUP)]
        closes = _with_frozen(closes.reindex(cal).copy(), mem, cal)
        cap = float(idx.get("cap_pct") or C.DEFAULT_CAP * 100) / 100
        caps = _caps(market, tickers, closes, ext) if idx["method"] in ("cap", "cap_capped") else None
        lv = C.build(closes, mem, method=idx["method"], rebalance=idx["rebalance"], base_date=ext,
                     cap=cap, caps=caps)["level"]
        return lv

    lv = _result_cache.get_or_compute(key, run)
    if lv.index[0] >= real.index[0]:
        return real, None
    return lv * (float(real.iloc[-1]) / float(lv.iloc[-1])), str(lv.index[0].date())


@_guard
def board_rows(user_id: int, market: str) -> dict:
    """
    The caller's indexes in one market, as rows of the dashboard board —
    read with the same arithmetic as every ETF and theme index beside them.
    An index younger than the 200-day line it is measured against is listed
    by name rather than given readings it cannot have yet.
    """
    import us_etf
    idxs = [r for r in _rows(_db().read_table("custom_index", filters={"app_user_id": int(user_id)}))
            if r["market"] == market]
    members = {r["id"]: _members(r["id"]) for r in idxs}
    idxs = [r for r in idxs if members[r["id"]]]
    if not idxs:
        return {"rows": [], "young": [], "failed": []}
    reach = [_extend_from(market)] if any(extendable(r) for r in idxs) else []
    _prefetch(market, idxs, members,
              start=min([date.fromisoformat(r["base_date"]) for r in idxs] + reach) - WARMUP)
    bench_t = "SPY" if market == "US" else "000300.SH"
    rows, young, failed = [], [], []
    for r in idxs:
        try:
            out = compute(user_id, r["id"])
            level, since = extended_level(user_id, r["id"])
        except Exception as exc:                                   # noqa: BLE001
            failed.append({"id": int(r["id"]), "name": r["name"], "error": str(exc)[:120]})
            continue
        # Relative strength against the index's own comparison, when it has one.
        cmp = _comparison(market, r.get("bench"))
        bench = cmp if cmp is not None else _union.get(market, (None,) * 5)[4]
        x = us_etf.row(f"IDX{r['id']}", level, bench, name=r["name"], group=us_etf.MINE,
                       bench_ticker=r.get("bench") if cmp is not None else bench_t)
        if x is None:
            young.append({"id": int(r["id"]), "name": r["name"], "days": len(level),
                          "hold": not extendable(r)})
            continue
        x["index_id"] = int(r["id"])
        x["method"] = out["method_label"]
        x["extended_from"] = since
        x["rs_vs"] = (r["bench"] if market == "US" else bench_label(market, r["bench"])) if cmp is not None else None
        rows.append(x)
    return {"rows": rows, "young": young, "failed": failed}


@_guard
def stretch_detail(user_id: int, index_id: int) -> dict:
    """The board's detail view (price, 50/200-day, stretch bands) for one of the caller's indexes."""
    import us_etf
    out = compute(user_id, index_id)
    level, since = extended_level(user_id, index_id)
    t = f"IDX{index_id}"
    d = us_etf.detail(pd.DataFrame({t: level}), t, names={t: out["index"]["name"]})
    d["extended_from"] = since
    d["start"] = out["start"]
    return d


def refresh_all() -> list[dict]:
    """
    Every user's every index, once — the daily job. Computing is what writes
    an exit down, so an index nobody opens for weeks still records a member's
    acquisition before Yahoo forgets it.
    """
    report = []
    rows = _rows(_db().read_table("custom_index"))
    for market in ("US", "CN"):
        mine = [r for r in rows if r["market"] == market]
        members = {r["id"]: _members(r["id"]) for r in mine}
        mine = [r for r in mine if members[r["id"]]]
        if not mine:
            continue
        try:
            _prefetch(market, mine, members,
                      start=min(date.fromisoformat(r["base_date"]) for r in mine) - WARMUP)
        except Exception as exc:                                   # noqa: BLE001
            report += [{"id": r["id"], "name": r["name"], "ok": False, "error": f"prices: {exc}"[:160]} for r in mine]
            continue
        for r in mine:
            before = {m["ticker"] for m in members[r["id"]] if m.get("removal") == "delisted"}
            try:
                out = compute(int(r["app_user_id"]), int(r["id"]))
                gone = [m["ticker"] for m in out["members"] if m["status"] == "delisted" and m["ticker"] not in before]
                report.append({"id": r["id"], "name": r["name"], "ok": True,
                               "level": out["stats"]["level"], "as_of": out["as_of"], "new_exits": gone})
            except Exception as exc:                               # noqa: BLE001
                report.append({"id": r["id"], "name": r["name"], "ok": False, "error": str(exc)[:160]})
    return report
