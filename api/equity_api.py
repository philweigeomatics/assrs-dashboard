"""
api/equity_api.py — the Equity Brief, assembled for one stock.

Two speeds, deliberately separated.

The FAST half is already in Tushare or the database: fundamentals with
year-on-year growth, the revenue split by product and by region, the latest
forecast and express filings, and whatever supply-chain graph has been saved.
That is what a GET returns, and it returns it whether or not anyone has ever
generated the rest.

The SLOW half is four DeepSeek calls — overview, Porter's five forces, PESTEL
and the peer set. Those are cached per ticker per section in
equity_brief_cache, so they cost a minute once and nothing thereafter. A GET
never triggers them; it reports which are missing and the client asks for them
explicitly. A section you did not ask for is never silently billed.

SWOT exists in equity_brief.py and is deliberately not surfaced here.
"""

from __future__ import annotations

import pandas as pd

#: Sections that cost an AI call. Order is the order they are generated in.
AI_SECTIONS = ("overview", "porters", "pestel", "competitors")

#: Periods of financial history to return.
PERIODS = 8


def _num(v, nd=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else round(v, nd)


def _rows(df: pd.DataFrame | None) -> list[dict]:
    if df is None or getattr(df, "empty", True):
        return []
    return df.where(pd.notna(df), None).to_dict("records")


# ── revenue segmentation ─────────────────────────────────────────────────────
def _drop_rollups(items: list[dict]) -> list[dict]:
    """
    Remove the subtotal rows Tushare mixes in with the real ones.

    fina_mainbz returns "产品" alongside the individual products and "地区"
    alongside 境内/境外 — each one the SUM of its siblings. Charted raw, the
    rollup is the entire pie and every real segment is a sliver.

    Detected by arithmetic rather than by label: an item whose sales equal the
    sum of all the others (within 1%) is a total, whatever it is called. That
    survives a company inventing its own subtotal name, which a list of known
    Chinese labels would not.
    """
    if len(items) < 2:
        return items
    out = list(items)
    changed = True
    while changed and len(out) > 1:
        changed = False
        for i, it in enumerate(out):
            rest = sum(abs(x["sales"] or 0) for j, x in enumerate(out) if j != i)
            mine = abs(it["sales"] or 0)
            if rest > 0 and abs(mine - rest) / rest < 0.01:
                out.pop(i)
                changed = True
                break
    return out


def _segments(ticker: str, bz_type: str) -> dict:
    """One period's split, plus the periods available."""
    import data_manager

    df = data_manager.fetch_fina_mainbz(ticker, bz_type)
    if df is None or df.empty:
        return {"period": None, "items": [], "periods": []}

    periods = sorted({str(d) for d in df["end_date"]}, reverse=True)

    def items_for(period: str) -> list[dict]:
        rows = df[df["end_date"].astype(str) == period]
        return _drop_rollups([{
            "item": str(r["bz_item"]),
            "sales": _num(r.get("bz_sales"), 0),
            "profit": _num(r.get("bz_profit"), 0),
            "cost": _num(r.get("bz_cost"), 0),
        } for _, r in rows.iterrows()])

    # Interim reports often carry only 主营业务 / 其他业务 while the annual
    # report breaks out the real segments. Walk back to the most recent period
    # that actually says something, rather than showing a two-row "breakdown"
    # because it happens to be the newest.
    latest, items = periods[0], items_for(periods[0])
    if len(items) < 3:
        for p in periods[:6]:
            candidate = items_for(p)
            if len(candidate) > len(items):
                latest, items = p, candidate
            if len(items) >= 3:
                break

    total = sum(abs(i["sales"] or 0) for i in items) or 1
    for i in items:
        i["share_pct"] = _num((i["sales"] or 0) / total * 100, 1)
        # Segment margin where the company reports segment profit; many do not.
        i["margin_pct"] = (_num((i["profit"] or 0) / i["sales"] * 100, 1)
                           if i["sales"] else None)
    items.sort(key=lambda x: -(x["sales"] or 0))
    return {"period": latest, "items": items, "periods": periods[:8]}


# ── fundamentals ─────────────────────────────────────────────────────────────
def _fundamentals(ticker: str) -> dict:
    """
    Revenue, profit and the ratios, newest period first, with growth.

    Year-on-year is taken from Tushare's own or_yoy / netprofit_yoy where
    present. A-share interim reports are CUMULATIVE within the year, so
    quarter-on-quarter differencing of the raw statement would compare a
    half-year against a quarter — the supplied YoY fields avoid that trap.
    """
    import data_manager

    income = data_manager.fetch_income_statement(ticker, PERIODS)
    ind = data_manager.fetch_full_fina_indicator(ticker, PERIODS)

    by_period: dict[str, dict] = {}
    for r in _rows(income):
        p = str(r.get("end_date"))
        by_period.setdefault(p, {"period": p})
        by_period[p].update({
            "revenue": _num(r.get("revenue") or r.get("total_revenue"), 0),
            "operate_profit": _num(r.get("operate_profit"), 0),
            "net_profit": _num(r.get("n_income_attr_p") or r.get("n_income"), 0),
        })
    for r in _rows(ind):
        p = str(r.get("end_date"))
        by_period.setdefault(p, {"period": p})
        by_period[p].update({
            "roe": _num(r.get("roe")), "roa": _num(r.get("roa")),
            "gross_margin": _num(r.get("grossprofit_margin")),
            "net_margin": _num(r.get("netprofit_margin")),
            "debt_to_assets": _num(r.get("debt_to_assets")),
            "current_ratio": _num(r.get("current_ratio")),
            "revenue_yoy": _num(r.get("or_yoy")),
            "profit_yoy": _num(r.get("netprofit_yoy")),
            "eps_yoy": _num(r.get("basic_eps_yoy")),
            "ocf_to_revenue": _num(r.get("ocf_to_or")),
        })

    periods = sorted(by_period.values(), key=lambda x: x["period"], reverse=True)
    return {"periods": periods[:PERIODS]}


def _filings(ticker: str) -> dict:
    """业绩预告 and 业绩快报 — what the company has said before the full report."""
    import data_manager

    def clean(rows, keys):
        out = []
        for r in rows:
            out.append({k: (_num(r.get(k), 2) if k not in ("end_date", "ann_date", "type", "summary", "change_reason")
                            else (str(r.get(k)) if r.get(k) is not None else None))
                        for k in keys})
        return out

    forecast = clean(_rows(data_manager.fetch_forecast(ticker, 8)),
                     ["end_date", "ann_date", "type", "p_change_min", "p_change_max",
                      "net_profit_min", "net_profit_max", "change_reason", "summary"])
    express = clean(_rows(data_manager.fetch_express(ticker, 4)),
                    ["end_date", "ann_date", "revenue", "operate_profit", "n_income",
                     "yoy_net_profit", "yoy_sales", "diluted_roe"])
    return {"forecast": forecast, "express": express}


# ── peers ────────────────────────────────────────────────────────────────────
def _peer_metrics(ticker: str) -> dict:
    """The handful of numbers a peer table compares on. Two calls per stock."""
    import data_manager

    out = {"ticker": ticker, "pe_ttm": None, "pb": None, "mv_yi": None,
           "roe": None, "gross_margin": None, "net_margin": None,
           "revenue_yoy": None, "profit_yoy": None, "debt_to_assets": None,
           "period": None}
    try:
        db = data_manager.get_latest_daily_basic(ticker)
        if db:
            out["pe_ttm"] = _num(db.get("pe_ttm"))
            out["pb"] = _num(db.get("pb"))
            # get_latest_daily_basic ALREADY converts 万元 → 亿元. Dividing
            # again reported a 1,200亿 company as 0.1亿.
            out["mv_yi"] = _num(db.get("total_mv"), 1)
    except Exception:
        pass
    try:
        ind = data_manager.fetch_full_fina_indicator(ticker, 1)
        if ind is not None and not ind.empty:
            r = ind.iloc[0]
            out.update({
                "period": str(r.get("end_date")),
                "roe": _num(r.get("roe")),
                "gross_margin": _num(r.get("grossprofit_margin")),
                "net_margin": _num(r.get("netprofit_margin")),
                "revenue_yoy": _num(r.get("or_yoy")),
                "profit_yoy": _num(r.get("netprofit_yoy")),
                "debt_to_assets": _num(r.get("debt_to_assets")),
            })
    except Exception:
        pass
    return out


def peer_table(ticker: str, name: str, peers: list[dict]) -> dict:
    """
    The target and its peers on the same rows, with the target marked.

    Ranked within each column so "cheapest", "highest margin" and "fastest
    growing" are readable at a glance rather than by scanning numbers — which
    is the whole reason to put them in one table.
    """
    import data_manager

    rows = [{**_peer_metrics(ticker), "name": name, "is_target": True, "why": ""}]
    for p in peers:
        t = str(p.get("ticker"))
        rows.append({
            **_peer_metrics(t),
            "name": p.get("name") or data_manager.get_stock_name_from_db(t) or t,
            "is_target": False,
            "why": p.get("why") or "",
        })

    # Rank each column. Cheap is good for the multiples, high is good for the
    # rest, and debt is its own direction — sorted the wrong way these would
    # quietly reward the worst company in the group.
    LOWER_BETTER = {"pe_ttm", "pb", "debt_to_assets"}
    for col in ("pe_ttm", "pb", "roe", "gross_margin", "net_margin",
                "revenue_yoy", "profit_yoy", "debt_to_assets"):
        vals = [(r[col], r) for r in rows if r.get(col) is not None]
        # A negative PE is not "cheap", it is loss-making; excluded from that rank.
        if col in ("pe_ttm", "pb"):
            vals = [(v, r) for v, r in vals if v > 0]
        vals.sort(key=lambda x: x[0] if col in LOWER_BETTER else -x[0])
        for i, (_, r) in enumerate(vals):
            r.setdefault("ranks", {})[col] = i + 1
        for r in rows:
            r.setdefault("ranks", {})
    return {"rows": rows, "n": len(rows)}


# ── assembly ─────────────────────────────────────────────────────────────────
def cached_ai(ticker: str) -> dict:
    """Whatever AI sections already exist, without generating any."""
    import equity_brief as eb

    eb.ensure_equity_brief_cache_table()
    out = {}
    for section in AI_SECTIONS:
        hit = eb._read_cache(ticker, section)
        out[section] = hit if hit else None
    return out


def generate(ticker: str, name: str, industry: str, section: str,
             force: bool = False) -> dict:
    """Run one AI section. Raises LookupError for an unknown section name."""
    import equity_brief as eb

    eb.ensure_equity_brief_cache_table()
    if section == "overview":
        return eb.get_company_overview(ticker, name, industry, force)
    if section == "porters":
        return eb.get_porters(ticker, name, industry, force)
    if section == "pestel":
        return eb.get_pestel(ticker, name, industry, force)
    if section == "competitors":
        chain = _supply_chain(ticker)
        products = [n.get("label") for n in (chain.get("nodes") or [])
                    if n.get("kind") in ("product", "core")][:8]
        return eb.get_competitors(ticker, name, industry, force,
                                  core_products=[p for p in products if p])
    raise LookupError(f"未知的分析模块：{section}")


def _supply_chain(ticker: str) -> dict:
    import data_manager
    try:
        graph = data_manager.get_supply_chain_graph(ticker)
    except Exception:
        return {}
    if not graph:
        return {}
    return graph if isinstance(graph, dict) else {}


def build(ticker: str, name: str, industry: str) -> dict:
    """The fast half, plus whichever AI sections are already cached."""
    ai = cached_ai(ticker)
    peers = ((ai.get("competitors") or {}).get("payload") or {}).get("competitors") or []

    return {
        "ticker": ticker, "name": name, "industry": industry,
        "fundamentals": _fundamentals(ticker),
        "filings": _filings(ticker),
        "segments": {
            "product": _segments(ticker, "P"),
            # Tushare's type "D" is 地区 — verified against 中天科技, which
            # returns 境内 / 境外 — despite the field being described as
            # department elsewhere.
            "region": _segments(ticker, "D"),
        },
        "supply_chain": _supply_chain(ticker),
        "ai": {k: (v or {}).get("payload") if v else None for k, v in ai.items()},
        "generated_at": {k: (v or {}).get("generated_at") if v else None
                         for k, v in ai.items()},
        "missing": [k for k, v in ai.items() if not v],
        "peers": peer_table(ticker, name, peers) if peers else None,
    }
