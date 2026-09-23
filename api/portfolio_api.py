"""
api/portfolio_api.py — saved portfolios, on top of the Streamlit tables.

The optimiser produces an allocation; this is what keeps one. It writes the
same `funds` and `fund_positions` rows the Streamlit page wrote, so a
portfolio created in either app is visible in both and neither needs a
migration.

WEIGHTS ARE HISTORY, NOT A SETTING

`fund_positions` is temporal: a rebalance does not overwrite the old row, it
closes it with an `end_date` and inserts a new one. So "what was this
portfolio holding in March" stays answerable, which is the whole reason to
save a portfolio rather than re-run the optimiser and hope it lands in the
same place.

EVERY READ IS SCOPED TO THE OWNER

`funds.user_id` is on every query rather than relied upon from a session.
These rows carry someone's allocation; a listing that forgets the filter
shows one user's book to another.
"""

from __future__ import annotations

from datetime import datetime

#: A portfolio has to be nameable and findable.
NAME_MAX = 60

#: Same floor as the builder: one name is not a portfolio.
MIN_HOLDINGS = 2


def _db():
    import data_manager
    return data_manager.db


def _ensure():
    """The tables, created on demand exactly as the Streamlit page does."""
    import data_manager
    try:
        data_manager.init_portfolio_tables()
    except Exception as exc:                                       # noqa: BLE001
        print(f"[portfolio_api] init tables: {type(exc).__name__}: {exc}")


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def list_funds(user_id: int) -> dict:
    """Every portfolio this user owns, newest first."""
    _ensure()
    df = _db().read_table("funds", filters={"user_id": int(user_id)})
    if df is None or df.empty:
        return {"funds": []}

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "id": int(r["id"]),
            "name": str(r.get("fund_name") or ""),
            "benchmark": bench_label(str(r.get("benchmark") or "") or None),
            "inception": str(r.get("inception_date") or "")[:10] or None,
            "created_at": str(r.get("created_at") or "")[:19],
        })
    rows.sort(key=lambda x: x["created_at"], reverse=True)

    # Holdings count per fund, so the list says something without opening one.
    for row in rows:
        row["holdings"] = len(_active_positions(row["id"]))
    return {"funds": rows}


def _active_positions(fund_id: int) -> list[dict]:
    """
    The rows with no end_date — what the portfolio holds now.

    An end_date is how a rebalance retires a weight without deleting it, so
    "current" means open-ended rather than most-recent.
    """
    df = _db().read_table("fund_positions", filters={"fund_id": int(fund_id)})
    if df is None or df.empty:
        return []
    out = []
    for _, r in df.iterrows():
        end = r.get("end_date")
        if end is not None and str(end).strip() not in ("", "None", "NaT", "nan"):
            continue
        out.append({"t": str(r["ts_code"]),
                    "weight_pct": round(float(r["weight"]) * 100, 2),
                    "since": str(r.get("effective_date") or "")[:10]})
    return out


def create_fund(user_id: int, name: str, holdings: list[dict],
                benchmark: str | None = None) -> dict:
    """
    Save an allocation under a name.

    `holdings` is [{t, weight_pct}]. The weights are stored as fractions
    because that is what fund_positions holds and what the Streamlit page
    reads; converting on the way in keeps one convention in the database and
    one in the UI, rather than two in both.
    """
    _ensure()
    name = (name or "").strip()
    if not name:
        raise ValueError("组合名不能为空")
    if len(name) > NAME_MAX:
        raise ValueError(f"组合名最多 {NAME_MAX} 个字符")

    clean = []
    for h in holdings or []:
        t = str(h.get("t") or "").strip()
        pct = float(h.get("weight_pct") or 0)
        if t and pct > 0:
            clean.append((t, pct / 100.0))
    if len(clean) < MIN_HOLDINGS:
        raise ValueError(f"至少需要 {MIN_HOLDINGS} 只有权重的股票")

    total = sum(w for _, w in clean)
    if abs(total - 1.0) > 0.02:
        raise ValueError(f"权重合计 {total * 100:.1f}%，应当接近 100%")

    db = _db()
    existing = db.read_table("funds", filters={"user_id": int(user_id),
                                               "fund_name": name})
    if existing is not None and not existing.empty:
        raise ValueError(f"已有同名组合「{name}」")

    today = _today()
    db.insert_records("funds", [{
        "user_id": int(user_id), "fund_name": name,
        "benchmark": benchmark or None, "inception_date": today,
    }])

    back = db.read_table("funds", filters={"user_id": int(user_id),
                                           "fund_name": name}, limit=1)
    if back is None or back.empty:
        raise RuntimeError("组合已写入但读不回来 — 请刷新后确认")
    fund_id = int(back.iloc[0]["id"])

    db.insert_records("fund_positions", [{
        "fund_id": fund_id, "ts_code": t, "weight": round(w, 6),
        "effective_date": today, "end_date": None,
    } for t, w in clean])

    return {"id": fund_id, "name": name, "inception": today,
            "holdings": len(clean)}


#: The notional the Streamlit manager starts every mandate at, and the base
#: its return is measured from. Kept identical so a fund's reported return
#: does not change depending on which app you open it in.
INCEPTION_AUM = 10_000_000.0


def _track(fund_id: int, benchmark: str | None) -> dict:
    """
    The NAV history, and the index it is judged against.

    A portfolio's own equity curve says nothing on its own — every A-share
    book was up in a bull quarter. What is worth reading is the distance
    between it and the index over the same days, so the benchmark is fetched
    across exactly the fund's own dates and alpha is the difference.

    The history comes from fund_daily_metrics, which the nightly rollup
    writes. An empty table is a fund that has not been valued yet, not an
    error — it is what a fund created this morning looks like.
    """
    import pandas as pd

    df = _db().read_table("fund_daily_metrics",
                          filters={"fund_id": int(fund_id)},
                          order_by="trade_date")
    if df is None or df.empty:
        return {"valued": False, "dates": [], "curve": [], "benchmark": None,
                "aum": None, "total_return_pct": None, "alpha_pct": None,
                "daily_return_pct": None}

    dates = [str(d)[:10] for d in df["trade_date"].tolist()]
    aum = [float(v) for v in df["total_aum"].tolist()]
    curve = [round((v - INCEPTION_AUM) / INCEPTION_AUM * 100, 3) for v in aum]

    total = curve[-1] if curve else None
    daily = None
    if len(aum) > 1 and aum[-2] > 0:
        flow = float(df.iloc[-1].get("net_flow") or 0)
        daily = round((aum[-1] - aum[-2] - flow) / aum[-2] * 100, 3)
    elif total is not None:
        daily = total

    bench = _benchmark_curve(benchmark, dates)
    alpha = (round(total - bench["curve"][-1], 3)
             if total is not None and bench and bench["curve"] else None)

    return {
        "valued": True,
        "dates": dates,
        "curve": curve,
        "aum": round(aum[-1], 2),
        "total_return_pct": total,
        "daily_return_pct": daily,
        "benchmark": bench,
        "alpha_pct": alpha,
        "inception_aum": INCEPTION_AUM,
    }


#: Label → Tushare index code, for funds whose benchmark was stored as a name.
_BENCH_CODES = {"沪深300": "000300.SH", "上证指数": "000001.SH",
                "中证500": "000905.SH", "创业板指": "399006.SZ"}
#: …and back, for the funds that stored the code instead of the name.
_BENCH_NAMES = {v: k for k, v in _BENCH_CODES.items()}


def bench_label(benchmark: str | None) -> str | None:
    """`000905.SH` reads as nothing; `中证500` reads as an index."""
    if not benchmark:
        return None
    return _BENCH_NAMES.get(benchmark, benchmark)


def _benchmark_curve(benchmark: str | None, dates: list[str]) -> dict | None:
    """The index rebased to the fund's first valuation date, on its dates."""
    if not benchmark or not dates:
        return None
    code = benchmark if "." in benchmark else _BENCH_CODES.get(benchmark)
    if not code:
        return None

    import pandas as pd
    try:
        import data_manager
        df = data_manager.get_index_data_live(
            code, start_date=dates[0].replace("-", ""),
            end_date=dates[-1].replace("-", ""))
    except Exception as exc:                                       # noqa: BLE001
        print(f"[portfolio_api] benchmark {code}: {type(exc).__name__}: {exc}")
        return None
    if df is None or df.empty or "Close" not in df:
        return None

    s = df["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    want = pd.to_datetime(pd.Index(dates))
    on = s.reindex(want).ffill().bfill()
    if on.isna().all():
        return None
    base = float(on.iloc[0])
    if base == 0:
        return None
    return {
        "label": bench_label(benchmark),
        "curve": [None if v != v else round((float(v) / base - 1) * 100, 3)
                  for v in on],
    }


def _drift(fund_id: int, positions: list[dict]) -> list[dict]:
    """
    Target weight against what the market has made it.

    A portfolio is only its target weights on the day it is built; after that
    the winners grow into a bigger share than intended. fund_daily_weights
    holds the actual, written by the same rollup.
    """
    df = _db().read_table("fund_daily_weights", filters={"fund_id": int(fund_id)})
    if df is None or df.empty:
        return []

    latest = str(df["trade_date"].max())[:10]
    rows = df[df["trade_date"].astype(str).str[:10] == latest]
    actual = {str(r["ts_code"]): float(r["actual_weight"])
              for _, r in rows.iterrows()}
    if not actual:
        return []

    out = []
    for p in positions:
        act = actual.get(p["t"])
        if act is None:
            continue
        act_pct = act * 100 if act <= 1.5 else act
        out.append({"t": p["t"], "target_pct": p["weight_pct"],
                    "actual_pct": round(act_pct, 2),
                    "drift_pct": round(act_pct - p["weight_pct"], 2),
                    "as_of": latest})
    return out


def fund_detail(user_id: int, fund_id: int) -> dict:
    """
    One portfolio: what it targets, what it now holds, and how it is doing.
    """
    _ensure()
    db = _db()
    df = db.read_table("funds", filters={"id": int(fund_id),
                                         "user_id": int(user_id)}, limit=1)
    if df is None or df.empty:
        # Same answer whether it does not exist or belongs to someone else:
        # "not yours" and "not there" should not be distinguishable.
        raise LookupError("找不到这个组合")

    row = df.iloc[0].to_dict()
    positions = _active_positions(fund_id)
    benchmark = str(row.get("benchmark") or "") or None
    return {
        "id": int(fund_id),
        "name": str(row.get("fund_name") or ""),
        "benchmark": bench_label(benchmark),
        "inception": str(row.get("inception_date") or "")[:10] or None,
        "holdings": positions,
        "tracking": _track(fund_id, benchmark),
        "drift": _drift(fund_id, positions),
    }


def revalue(fund_id: int) -> dict:
    """
    Run the NAV rollup now, rather than waiting for the nightly job.

    The Streamlit page called this 强制 NAV 计算. It walks every fund, so the
    fund_id is only used to report what happened afterwards.
    """
    import data_manager
    try:
        data_manager.execute_daily_portfolio_rollup()
    except Exception as exc:                                       # noqa: BLE001
        raise RuntimeError(f"NAV 计算失败：{type(exc).__name__}: {exc}"[:200])
    return {"fund_id": int(fund_id), "ran": True}


def delete_fund(user_id: int, fund_id: int) -> dict:
    """Remove a portfolio and its positions. Scoped to the owner."""
    _ensure()
    db = _db()
    owned = db.read_table("funds", filters={"id": int(fund_id),
                                            "user_id": int(user_id)}, limit=1)
    if owned is None or owned.empty:
        raise LookupError("找不到这个组合")

    db.delete_records("fund_positions", {"fund_id": int(fund_id)})
    db.delete_records("funds", {"id": int(fund_id), "user_id": int(user_id)})
    return {"id": int(fund_id), "deleted": True}
