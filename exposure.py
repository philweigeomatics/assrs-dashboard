"""
exposure.py — which industries the book is actually in.

The naive version counts the sector of each stock and calls it exposure. For a
Canadian account that is usually wrong by a wide margin, because a large part
of the book is index ETFs, and an ETF has no sector of its own — it has
eleven. A portfolio that reads as "60% ETF, 40% technology" is really more like
70% technology once you look through VFV, and that is the number that decides
whether one more semiconductor name is diversification or doubling down.

So exposure is computed with LOOK-THROUGH: every ETF's weight is distributed
across its own published sector weightings, and stocks contribute their whole
weight to one sector. The two sources are then added, and reported separately
as well as combined, because "I chose this" and "my index fund chose this" are
different facts about the same number.

Industry (the level below sector) is stocks only. An ETF publishes sector
weights, not industry weights, so a combined industry table would silently be a
stocks-only table wearing a portfolio-wide label.

Pure: the Yahoo lookups are passed in, not fetched here.
"""

from __future__ import annotations

UNKNOWN = "未知"

#: Canonical sector keys, with the Chinese/English label shown.
#:
#: The two Yahoo sources disagree about spelling — a stock's `sector` is Title
#: Case ("Real Estate") and an ETF's weightings are snake_case with one
#: irregular member ("realestate") — so everything is normalised through
#: `canon` before it is counted. Without that, real-estate exposure splits
#: across two rows that never add up.
SECTORS = {
    "technology":             "科技 Technology",
    "financial_services":     "金融 Financials",
    "healthcare":             "医疗 Healthcare",
    "consumer_cyclical":      "可选消费 Consumer Cyclical",
    "consumer_defensive":     "必需消费 Consumer Defensive",
    "communication_services": "通信 Communication",
    "industrials":            "工业 Industrials",
    "energy":                 "能源 Energy",
    "basic_materials":        "原材料 Materials",
    "real_estate":            "房地产 Real Estate",
    "utilities":              "公用事业 Utilities",
}

_ALIASES = {
    "realestate": "real_estate",
    "financial": "financial_services",
    "financialservices": "financial_services",
    "communicationservices": "communication_services",
    "consumercyclical": "consumer_cyclical",
    "consumerdefensive": "consumer_defensive",
    "basicmaterials": "basic_materials",
    "health_care": "healthcare",
}


def canon(raw: str | None) -> str | None:
    """'Real Estate', 'realestate', 'real_estate' → 'real_estate'."""
    if not raw:
        return None
    key = str(raw).strip().lower().replace("-", " ").replace("&", " ")
    key = "_".join(key.split())
    key = _ALIASES.get(key, _ALIASES.get(key.replace("_", ""), key))
    return key if key in SECTORS else None


def label(key: str | None) -> str:
    return SECTORS.get(key or "", UNKNOWN)


def analyse(holdings: list[dict], lookups: dict[str, dict]) -> dict:
    """
    Sector and industry exposure for a consolidated book.

    `holdings` is what consolidate() produced. `lookups[symbol]` carries
    {"sector": str|None, "industry": str|None, "weights": {key: frac}|None};
    `weights` is present for ETFs and is what makes look-through possible.

    Weights are shares of the PRICED book, so they sum to 100 including the
    unknown bucket. A holding whose sector could not be determined is counted
    as 未知 rather than dropped — dropping it would quietly inflate every other
    sector, which is the one error nobody would notice.
    """
    total = sum(abs(h.get("market_value_base") or 0.0) for h in holdings)
    if total <= 0:
        raise LookupError("持仓市值为零，无法计算行业暴露")

    sectors: dict[str, dict] = {}
    industries: dict[str, float] = {}
    etf_value = stock_value = 0.0

    for h in holdings:
        value = abs(h.get("market_value_base") or 0.0)
        if value <= 0:
            continue
        look = lookups.get(h["symbol"]) or {}
        weights = look.get("weights") or None
        is_etf = h.get("group") == "etf"

        if is_etf and weights:
            etf_value += value
            spread = _normalise(weights)
            for key, share in spread.items():
                _add(sectors, key, value * share, h["symbol"], etf=True)
            missing = 1.0 - sum(spread.values())
            if missing > 1e-6:
                _add(sectors, None, value * missing, h["symbol"], etf=True)
            continue

        key = canon(look.get("sector"))
        if is_etf:
            # An ETF we could not look through. Counted as unknown rather than
            # dumped into whatever sector its own listing says it belongs to.
            etf_value += value
            _add(sectors, None, value, h["symbol"], etf=True)
            continue

        stock_value += value
        _add(sectors, key, value, h["symbol"], etf=False)
        if key:
            name = str(look.get("industry") or "").strip() or UNKNOWN
            industries[name] = industries.get(name, 0.0) + value

    rows = [{
        "sector": key or "",
        "label": label(key),
        "value": round(v["value"], 2),
        "pct": round(v["value"] / total * 100, 2),
        "from_stocks_pct": round(v["stock"] / total * 100, 2),
        "from_etfs_pct": round(v["etf"] / total * 100, 2),
        "holdings": sorted(v["symbols"]),
    } for key, v in sectors.items()]
    rows.sort(key=lambda r: -r["pct"])

    industry_rows = [{"name": k, "value": round(v, 2),
                      "pct": round(v / stock_value * 100, 2) if stock_value else None}
                     for k, v in sorted(industries.items(), key=lambda kv: -kv[1])]

    unknown = next((r["pct"] for r in rows if not r["sector"]), 0.0)
    return {
        "total": round(total, 2),
        "rows": rows,
        # Stocks only, and labelled that way: an ETF publishes sector weights,
        # never industry weights, so a portfolio-wide industry table cannot
        # exist however much one would like it to.
        "industries": industry_rows,
        "industry_basis": round(stock_value, 2),
        "split": {"stock_pct": round(stock_value / total * 100, 2),
                  "etf_pct": round(etf_value / total * 100, 2)},
        "unknown_pct": unknown,
        "concentration": _top(rows),
    }


def _normalise(weights: dict) -> dict:
    """
    Canonical keys, and a sanity check on the total.

    Yahoo's weightings occasionally sum slightly over or under 1. Rescaling a
    sum of 0.98 up to 1 is right; rescaling 0.4 up to 1 would invent exposure,
    so a badly incomplete set is left short and the remainder becomes unknown.
    """
    out: dict[str, float] = {}
    for raw, share in (weights or {}).items():
        key = canon(raw)
        try:
            value = float(share)
        except (TypeError, ValueError):
            continue
        if key and value > 0:
            out[key] = out.get(key, 0.0) + value

    total = sum(out.values())
    if 0.8 <= total <= 1.2 and total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def _add(bucket: dict, key: str | None, value: float, symbol: str, *, etf: bool):
    row = bucket.setdefault(key, {"value": 0.0, "stock": 0.0, "etf": 0.0,
                                  "symbols": set()})
    row["value"] += value
    row["etf" if etf else "stock"] += value
    row["symbols"].add(symbol)


def _top(rows: list[dict]) -> dict:
    known = [r for r in rows if r["sector"]]
    return {
        "sectors": len(known),
        "top_sector": known[0]["label"] if known else None,
        "top_pct": known[0]["pct"] if known else None,
        "top3_pct": round(sum(r["pct"] for r in known[:3]), 2),
    }
