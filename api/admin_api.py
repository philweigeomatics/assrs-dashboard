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
        return {"created": False, "sector": name, "stocks": clean, "sql": sql}

    for t in clean:
        dm.add_stock_to_sector(name, t)
    return {"created": True, "sector": name, "stocks": clean, "sql": []}


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
