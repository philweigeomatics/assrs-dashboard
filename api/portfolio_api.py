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
            "benchmark": str(r.get("benchmark") or "") or None,
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


def fund_detail(user_id: int, fund_id: int) -> dict:
    """
    One portfolio, its current weights, and how it has done since inception.
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
    return {
        "id": int(fund_id),
        "name": str(row.get("fund_name") or ""),
        "benchmark": str(row.get("benchmark") or "") or None,
        "inception": str(row.get("inception_date") or "")[:10] or None,
        "holdings": positions,
    }


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
