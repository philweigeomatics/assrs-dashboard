"""
api/admin_api.py — sector membership, which is the input to almost everything.

A sector's stock list decides its PPI, and the PPI decides the heatmap, the
breadth grid, the rotation map and the regime score. So an edit here is not a
personal setting: it changes what every user sees on the market dashboard,
and it changes it for good, because the nightly job rebuilds from whatever is
in this table.

Three consequences shape this module.

IT IS ADMIN-ONLY, AND THAT IS CHECKED ON THE SERVER. The frontend hides the
page; hiding a page does not stop a request.

REMOVALS ARE SOFT. `is_active` goes to 0 and the row stays, so a sector's
history is recoverable and an accidental removal is one click to undo rather
than a re-typed ticker list.

A SECTOR NEEDS TWO STOCKS. The PPI is a cap-weighted index; with one
constituent it is that stock's price wearing a sector's name, and every
downstream screen would go on reporting it as a sector. The floor is enforced
here rather than left to the caller.
"""

from __future__ import annotations

#: Below this a sector's PPI is not an index of anything.
MIN_STOCKS = 2

#: Sector names become part of a table name (PPI_<sector>) and a column in
#: market_breadth, so they cannot contain anything that would have to be
#: quoted or escaped differently in the two places.
NAME_MAX = 40


def _dm():
    import data_manager
    return data_manager


def _names() -> dict[str, str]:
    """{ticker: company name} for everything in stock_basic."""
    try:
        return {s["ticker"]: s["name"] for s in _dm().get_all_stock_basic()}
    except Exception:                                              # noqa: BLE001
        return {}


def overview() -> dict:
    """
    Every sector, its members, and what has been removed from it.

    Removed rows come back too. An admin screen that only shows the current
    state cannot answer "what did I take out of 半导体 last month", which is
    the question asked right after a number moves.
    """
    dm = _dm()
    active = dm.get_sector_stock_map()
    names = _names()

    raw = dm.get_all_sector_stock_map_raw()
    removed: dict[str, list[dict]] = {}
    if raw is not None and not raw.empty and "is_active" in raw.columns:
        gone = raw[raw["is_active"] == 0]
        for _, row in gone.iterrows():
            sector = str(row["sector"])
            ticker = str(row["ticker"])
            removed.setdefault(sector, []).append({
                "t": ticker, "n": names.get(ticker, ""),
                "removed_at": str(row.get("removed_at") or "")[:19],
            })

    return {
        "sectors": [{
            "name": sector,
            "stocks": [{"t": t, "n": names.get(t, "")} for t in tickers],
            "removed": removed.get(sector, []),
        } for sector, tickers in sorted(active.items())],
        "min_stocks": MIN_STOCKS,
        "total_stocks": sum(len(v) for v in active.values()),
    }


def add_stock(sector: str, ticker: str) -> dict:
    """Put a stock into a sector, or bring a removed one back."""
    dm = _dm()
    sector, ticker = sector.strip(), ticker.strip()
    current = dm.get_sector_stock_map()
    if sector not in current:
        raise LookupError(f"板块「{sector}」不存在")
    if ticker in current[sector]:
        raise ValueError(f"{ticker} 已经在「{sector}」里了")
    known = _names()
    if known and ticker not in known:
        # Only reject when the list was actually readable. An empty
        # stock_basic means we cannot check, and refusing every edit because
        # a lookup table is unavailable would be worse than allowing a typo.
        raise ValueError(f"{ticker} 不在股票列表里 — 请确认代码")

    dm.add_stock_to_sector(sector, ticker)
    return {"sector": sector, "ticker": ticker,
            "name": known.get(ticker, ""), "action": "added"}


def remove_stock(sector: str, ticker: str) -> dict:
    """
    Take a stock out of a sector, keeping the row.

    Refused below MIN_STOCKS: the caller would otherwise be left with a
    "sector" whose index is one company, and nothing downstream would notice.
    """
    dm = _dm()
    sector, ticker = sector.strip(), ticker.strip()
    current = dm.get_sector_stock_map()
    if sector not in current:
        raise LookupError(f"板块「{sector}」不存在")
    if ticker not in current[sector]:
        raise ValueError(f"{ticker} 不在「{sector}」里")
    if len(current[sector]) <= MIN_STOCKS:
        raise ValueError(
            f"「{sector}」只剩 {len(current[sector])} 只股票，"
            f"至少要保留 {MIN_STOCKS} 只 —— 少于这个数，板块指数就只是单只股票")

    dm.remove_stock_from_sector(sector, ticker)
    return {"sector": sector, "ticker": ticker,
            "name": _names().get(ticker, ""), "action": "removed"}


def validate_name(name: str, existing: dict) -> str:
    """The sector name, or a reason it cannot be used."""
    name = (name or "").strip()
    if not name:
        raise ValueError("板块名不能为空")
    if len(name) > NAME_MAX:
        raise ValueError(f"板块名最多 {NAME_MAX} 个字符")
    # It becomes PPI_<name> as a table and a market_breadth column, so
    # anything needing quoting in one place but not the other is out.
    bad = set(name) & set(' "\'`;\\/()[]{}%,.-+*')
    if bad:
        raise ValueError(f"板块名不能包含 {' '.join(sorted(bad))} "
                         f"—— 它会变成数据表名的一部分")
    if name in existing:
        raise ValueError(f"板块「{name}」已存在 —— 请到编辑页修改")
    return name


def create_sector(name: str, tickers: list[str]) -> dict:
    """
    A new sector, and the manual step it needs on Supabase.

    Supabase cannot create a table or add a column from the client key, so
    this does NOT pretend to have finished. When the PPI table or the breadth
    column is missing it returns the exact SQL to run, and writes nothing —
    a half-created sector that every rebuild then fails on is worse than a
    refusal with instructions.
    """
    dm = _dm()
    existing = dm.get_sector_stock_map()
    name = validate_name(name, existing)

    seen, clean = set(), []
    for t in tickers:
        t = str(t).strip()
        if t and t not in seen:
            seen.add(t)
            clean.append(t)
    if len(clean) < MIN_STOCKS:
        raise ValueError(f"至少需要 {MIN_STOCKS} 只股票")

    sql = pending_sql(name)
    if sql:
        return {"created": False, "sector": name, "stocks": clean,
                "sql": sql, "job_id": None, "job_error": None}

    dm.add_new_sector(name, clean)

    # A new sector has no PPI until one is built, so the rebuild is part of
    # creating it rather than a step to remember. If another rebuild is
    # already running the sector still exists — the reason is returned so the
    # screen can say "created, rebuild it when the current job finishes"
    # instead of implying the creation failed.
    job_id, job_error = None, None
    try:
        job_id = start_rebuild([name])["job_id"]
    except Exception as exc:                                       # noqa: BLE001
        job_error = str(exc)

    return {"created": True, "sector": name, "stocks": clean, "sql": [],
            "job_id": job_id, "job_error": job_error}


def pending_sql(name: str) -> list[str]:
    """SQL an admin must run in Supabase before this sector can exist."""
    dm = _dm()
    import db_config

    if not getattr(db_config, "USE_SUPABASE", False):
        return []                                  # SQLite creates on demand

    out = []
    try:
        if dm.get_missing_ppi_tables([name]):
            out.append(
                f'CREATE TABLE IF NOT EXISTS "PPI_{name}" ('
                f'"Date" TEXT PRIMARY KEY, "Open" REAL, "High" REAL, '
                f'"Low" REAL, "Close" REAL, "Norm_Vol_Metric" REAL);')
        if dm.get_missing_breadth_columns([name]):
            out.append(f'ALTER TABLE market_breadth '
                       f'ADD COLUMN IF NOT EXISTS "{name}" REAL;')
    except Exception:                                              # noqa: BLE001
        # Asking whether a table exists failed. Returning "nothing pending"
        # here would write a sector whose rebuild can never succeed.
        return [f"-- 无法确认 PPI_{name} 是否存在，请先在 Supabase 中检查"]
    return out


# ── rebuild jobs ─────────────────────────────────────────────────────────────
# A rebuild recomputes PPI and market breadth from the sector map. It takes
# 20-60 minutes, so it cannot happen inside the request: rebuild_runner starts
# a thread that writes its progress to the rebuild_jobs table, and the screen
# polls that table. Nothing ever waits on the thread itself, which is what
# makes the progress bar survive a page reload.

import json
from datetime import datetime

#: A job still 'running' after this long has almost certainly lost its thread.
#: A full rebuild is 20-60 minutes, so this is generous rather than tight —
#: the cost of calling a live job dead is worse than showing a dead one late.
STALE_MINUTES = 120

#: Statuses that mean the job has not finished one way or the other.
OPEN = ("pending", "running")


def _age_minutes(stamp: str) -> float | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return (datetime.now() - datetime.strptime(str(stamp)[:19], fmt)
                    ).total_seconds() / 60
        except (ValueError, TypeError):
            continue
    return None


def _job(row: dict) -> dict:
    raw = row.get("sectors") or "[]"
    if raw == "__all__":
        sectors, scope = [], "all"
    else:
        try:
            sectors, scope = json.loads(raw), "some"
        except Exception:                                          # noqa: BLE001
            sectors, scope = [str(raw)], "some"

    status = str(row.get("status") or "pending")
    age = _age_minutes(row.get("created_at"))
    # A thread dies with its instance. The row then says "running" forever,
    # and a progress bar that never moves is worse than an honest "unknown".
    stalled = bool(status in OPEN and age is not None and age > STALE_MINUTES)

    return {
        "job_id": str(row.get("job_id") or ""),
        "job_type": str(row.get("job_type") or ""),
        "scope": scope,
        "sectors": sectors,
        "status": "stalled" if stalled else status,
        "progress": int(row.get("progress") or 0),
        "message": str(row.get("progress_message") or ""),
        "error": str(row.get("error_message") or "") or None,
        "created_at": str(row.get("created_at") or "")[:19],
        "started_at": str(row.get("started_at") or "")[:19] or None,
        "completed_at": str(row.get("completed_at") or "")[:19] or None,
        "age_minutes": round(age, 1) if age is not None else None,
    }


def jobs(limit: int = 15) -> dict:
    """Recent rebuilds, newest first, with dead ones called dead."""
    df = _dm().get_recent_rebuild_jobs(limit=limit)
    rows = [] if df is None or df.empty else [_job(r) for _, r in df.iterrows()]
    return {
        "jobs": rows,
        "running": any(j["status"] in OPEN for j in rows),
        "stale_minutes": STALE_MINUTES,
    }


def active_job(rows: list[dict]) -> dict | None:
    """The one job that is genuinely still going, if any."""
    return next((j for j in rows if j["status"] in OPEN), None)


def start_rebuild(sectors: list[str] | str) -> dict:
    """
    Kick off a rebuild and return its job id.

    Refuses while another is live. Two rebuilds at once both wipe and rewrite
    the same PPI tables, and the loser silently corrupts the winner — the
    Streamlit page allowed this and it is the one thing worth adding here.
    """
    dm = _dm()
    from rebuild_runner import start_rebuild_thread

    live = active_job(jobs(limit=10)["jobs"])
    if live:
        raise ValueError(
            f"任务 {live['job_id']} 还在进行（{live['progress']}%）—— "
            f"同时跑两个重建会互相覆盖 PPI 表，请等它结束")

    if sectors != "__all__":
        known = dm.get_sector_stock_map()
        sectors = [s for s in sectors if s in known]
        if not sectors:
            raise ValueError("没有可重建的板块")

    job_type = "full_rebuild" if sectors == "__all__" else "sector_rebuild"
    job_id = dm.create_rebuild_job(job_type, sectors)
    start_rebuild_thread(job_id, sectors)
    return {"job_id": job_id, "job_type": job_type,
            "sectors": [] if sectors == "__all__" else sectors}
