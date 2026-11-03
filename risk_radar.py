"""
risk_radar.py — 🧭 风险雷达: one stock's risks, each read from two sides.

A fixed list of risks — systematic (宏观, shared with many companies) and
idiosyncratic (个股, this company's own). For each one, two questions kept
apart on purpose:

  敞口  does this risk MATTER for this stock? Measured where a price stand-in
        exists — weekly returns regressed on the market, US 10Y yields, the
        currency, oil, copper, gold, since the regime anchor — and otherwise
        the AI's reading of the business, labelled as such.
  证据  what is the evidence saying NOW? Every fact or story is filed under
        one risk with a direction (利空 / 利好), a strength 1–3, and whether it
        is a fact or an opinion. The rules-based sources (unlocks, insider
        trades, 8-K items, factor moves) are filed by rule; text is filed by
        the AI, once per item, and kept.

  压力(风险, 方向) = 敞口 × Σ 强度 × 可信度 × 事实系数 × 0.5^(天数 / 半衰期)

Downside and upside are summed separately and never netted inside a risk:
a lithium price fall that both writes down inventory and widens margins is
two facts, not zero. The sum is mapped onto 0–100 by a saturating curve, so a
pile of weak items cannot outrank a few strong ones by count alone.

WHAT THE NUMBER IS NOT

It is evidence density, not a probability. And no news is not low risk: a
risk with fewer than MIN_EVIDENCE weighted items in the window is shown as
证据不足, never as 低 — a thinly covered stock would otherwise look safe.

NO HINDSIGHT

score() takes an as-of date and ignores anything dated after it, so the
7-day change and the 30-day trend are the same function evaluated on earlier
days, and the record can later be tested: did a 高 come before the stock
doing worse?

V1 LIMITS, stated where they apply: the 0–100 scale is absolute (the design
ranks against the stock's own history, which needs months of stored
evidence first), and most feeds serve only their newest items, so the
record deepens with each refresh rather than arriving whole.
"""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

EVIDENCE_TABLE = "risk_radar_evidence"
CACHE_TABLE = "risk_radar_cache"
MIGRATION = "supabase/migrations/20261103_risk_radar.sql"

# ── the risk list ────────────────────────────────────────────────────────────
#: `hint` is what the tagger is told the risk covers. Order is display order.
TAXONOMY: list[dict] = [
    {"id": "rates", "kind": "macro", "name": "利率与流动性", "en": "Rates & liquidity",
     "hint": "interest rates, central-bank policy, bond yields, credit conditions, liquidity"},
    {"id": "fx", "kind": "macro", "name": "汇率", "en": "Currency",
     "hint": "exchange-rate moves affecting revenue, costs or translation"},
    {"id": "commodities", "kind": "macro", "name": "大宗商品与原材料", "en": "Commodities & inputs",
     "hint": "prices of raw materials, energy and commodities the company buys or sells"},
    {"id": "policy", "kind": "macro", "name": "产业政策与监管导向", "en": "Industry policy",
     "hint": "government industrial policy, subsidies, sector regulation, state support"},
    {"id": "trade", "kind": "macro", "name": "贸易、关税与地缘", "en": "Trade, tariffs & geopolitics",
     "hint": "tariffs, export controls, sanctions, geopolitical conflict, trade disputes"},
    {"id": "demand", "kind": "macro", "name": "宏观需求与经济周期", "en": "Macro demand & cycle",
     "hint": "GDP, consumption, property, employment, end-market demand across the economy"},
    {"id": "sector", "kind": "macro", "name": "行业景气与供需", "en": "Sector cycle & capacity",
     "hint": "industry-wide prices, capacity, inventory cycle, sector-level demand"},
    {"id": "market", "kind": "macro", "name": "市场情绪与杠杆", "en": "Market sentiment & leverage",
     "hint": "broad market direction, leverage, risk appetite, fund flows into equities"},
    {"id": "earnings", "kind": "micro", "name": "盈利与业绩指引", "en": "Earnings & guidance",
     "hint": "this company's results, margins, guidance, analyst estimates and ratings"},
    {"id": "balance", "kind": "micro", "name": "现金流与负债", "en": "Cash flow & balance sheet",
     "hint": "cash flow, debt, capex funding, impairments, working capital"},
    {"id": "holders", "kind": "micro", "name": "股东行为", "en": "Holder actions",
     "hint": "insider buying or selling, share pledges, lock-up expiries, buybacks, major holder changes"},
    {"id": "capital", "kind": "micro", "name": "融资与股本变动", "en": "Capital raising & dilution",
     "hint": "new share issuance, placements, convertibles, spin-offs, M&A financed with stock"},
    {"id": "competition", "kind": "micro", "name": "竞争与市场份额", "en": "Competition & share",
     "hint": "competitors, pricing pressure, market share gains or losses"},
    {"id": "customers", "kind": "micro", "name": "客户与供应链", "en": "Customers & supply chain",
     "hint": "key customers, orders, contracts, suppliers, supply-chain concentration"},
    {"id": "products", "kind": "micro", "name": "产品、技术与项目", "en": "Products, tech & projects",
     "hint": "product launches, R&D, technology, capacity projects, approvals"},
    {"id": "governance", "kind": "micro", "name": "治理与管理层", "en": "Governance & management",
     "hint": "management changes, related-party dealings, accounting quality, disclosure"},
    {"id": "legal", "kind": "micro", "name": "监管处罚与诉讼", "en": "Regulatory action & litigation",
     "hint": "investigations, penalties, exchange inquiries, lawsuits, delisting risk"},
]
RISK_IDS = {r["id"] for r in TAXONOMY}
BY_ID = {r["id"]: r for r in TAXONOMY}

# ── weights ──────────────────────────────────────────────────────────────────
#: How much an item of each kind is trusted. A company's own filing and an
#: exchange's data are what happened; a news flash is someone's account of it.
CREDIBILITY = {"filing": 1.0, "data": 1.0, "qa": 0.8, "macro": 0.8,
               "research": 0.6, "news": 0.6}
#: Days for an item's weight to halve. A filing stays relevant for a quarter;
#: a news flash is mostly priced within two weeks.
HALF_LIFE = {"filing": 60, "data": 30, "qa": 30, "macro": 21,
             "research": 30, "news": 10}
OPINION_WEIGHT = 0.4
#: How strongly measured or judged 敞口 scales the evidence.
EXPOSURE_MULT = {"高": 1.0, "中": 0.7, "低": 0.4}
DEFAULT_EXPOSURE = "中"
#: Evidence older than this carries nothing, whatever its half-life.
WINDOW_DAYS = 120
#: Fewer weighted items than this in the window → 证据不足, not 低.
MIN_EVIDENCE = 3
#: The saturating scale: K weighted units of pressure reads 63/100.
SCALE_K = 4.0
TREND_DAYS = 30


def evidence_id(source: str, title: str, d: str) -> str:
    """Stable per item, so a refresh that sees it again does not re-tag it."""
    return hashlib.sha1(f"{source}|{title.strip()}|{d}".encode("utf-8")).hexdigest()[:16]


def _day(v) -> date | None:
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    try:
        return pd.Timestamp(v).date()
    except Exception:
        return None


def weight(item: dict, as_of: date) -> float:
    """One item's contribution on `as_of`; 0 for untagged, future or stale items."""
    if item.get("risk") not in RISK_IDS or item.get("dir") not in ("down", "up"):
        return 0.0
    d = _day(item.get("d"))
    if d is None:
        return 0.0
    age = (as_of - d).days
    if age < 0 or age > WINDOW_DAYS:
        return 0.0
    kind = item.get("kind") or "news"
    strength = min(3, max(1, int(item.get("strength") or 1)))
    fact = 1.0 if item.get("fact", True) else OPINION_WEIGHT
    return (strength * CREDIBILITY.get(kind, 0.6) * fact
            * 0.5 ** (age / HALF_LIFE.get(kind, 30)))


def to_scale(raw: float, exposure: str | None) -> int:
    mult = EXPOSURE_MULT.get(exposure or DEFAULT_EXPOSURE, EXPOSURE_MULT[DEFAULT_EXPOSURE])
    return int(round(100 * (1 - math.exp(-mult * raw / SCALE_K))))


def score(items: list[dict], risk_id: str, exposure: str | None, as_of: date) -> dict:
    """Downside and upside pressure for one risk on one day, plus how many items count."""
    down = up = 0.0
    n = 0
    for it in items:
        if it.get("risk") != risk_id:
            continue
        w = weight(it, as_of)
        if w <= 0:
            continue
        n += 1
        if it["dir"] == "down":
            down += w
        else:
            up += w
    return {"down": to_scale(down, exposure), "up": to_scale(up, exposure), "n": n}


def tier(v: int, n: int) -> str:
    if n < MIN_EVIDENCE:
        return "证据不足"
    if v >= 70:
        return "高"
    if v >= 55:
        return "偏高"
    if v >= 40:
        return "中"
    if v >= 25:
        return "偏低"
    return "低"


def lean(down: int, up: int) -> str:
    """偏空 / 偏多 / 中性 — only a clear gap counts as a lean."""
    if up - down <= -15:
        return "偏空"
    if up - down >= 15:
        return "偏多"
    return "中性"


# ── 敞口: measured sensitivity ─────────────────────────────────────────────────
#: Price stand-ins, by market. `rate` factors are yields: their weekly CHANGE
#: in percentage points is the move, not a percent return.
FACTORS = {
    "CN": [
        {"key": "market", "ticker": "000300.SS", "label": "沪深300", "risk": "market"},
        {"key": "rate", "ticker": "^TNX", "label": "美债10年收益率", "risk": "rates", "rate": True},
        {"key": "fx", "ticker": "CNY=X", "label": "美元兑人民币", "risk": "fx"},
        {"key": "oil", "ticker": "CL=F", "label": "WTI原油", "risk": "commodities"},
        {"key": "copper", "ticker": "HG=F", "label": "铜", "risk": "commodities"},
        {"key": "gold", "ticker": "GC=F", "label": "黄金", "risk": "commodities"},
    ],
    "US": [
        {"key": "market", "ticker": "SPY", "label": "标普500", "risk": "market"},
        {"key": "rate", "ticker": "^TNX", "label": "美债10年收益率", "risk": "rates", "rate": True},
        {"key": "fx", "ticker": "DX-Y.NYB", "label": "美元指数", "risk": "fx"},
        {"key": "oil", "ticker": "CL=F", "label": "WTI原油", "risk": "commodities"},
        {"key": "copper", "ticker": "HG=F", "label": "铜", "risk": "commodities"},
        {"key": "gold", "ticker": "GC=F", "label": "黄金", "risk": "commodities"},
    ],
}
FACTORS["CA"] = FACTORS["US"]
#: Fewest weekly observations a regression is trusted on.
MIN_WEEKS = 52


def weekly_moves(closes: pd.Series, is_rate: bool = False) -> pd.Series:
    w = closes.dropna().resample("W-FRI").last().dropna()
    return (w.diff() if is_rate else w.pct_change()).dropna()


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Coefficients, t-stats and R² with an intercept prepended."""
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ coef
    dof = max(1, len(y) - X1.shape[1])
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(X1.T @ X1)
    t = coef / np.sqrt(np.maximum(np.diag(cov), 1e-18))
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / tss if tss > 0 else 0.0
    return coef[1:], t[1:], float(r2)


def exposure_level_t(t: float) -> str:
    a = abs(t)
    return "高" if a >= 3 else "中" if a >= 1.5 else "低"


def exposure_level_beta(beta: float) -> str:
    return "高" if beta >= 1.2 else "中" if beta >= 0.7 else "低"


def measure_exposure(stock: pd.Series, factors: dict[str, pd.Series], spec: list[dict],
                     start: pd.Timestamp) -> list[dict]:
    """
    One reading per factor. The market is a plain beta; every other factor is
    its coefficient ALONGSIDE the market, so copper is not credited with what
    is just the stock moving with the index.
    """
    y_all = weekly_moves(stock[stock.index >= start])
    mspec = next(s for s in spec if s["key"] == "market")
    if mspec["key"] not in factors:
        return []
    m = weekly_moves(factors["market"][factors["market"].index >= start])
    out = []
    base = pd.concat([y_all.rename("y"), m.rename("m")], axis=1).dropna()
    if len(base) >= MIN_WEEKS:
        coef, t, r2 = _ols(base["y"].to_numpy(), base[["m"]].to_numpy())
        out.append({**_spec_view(mspec), "beta": round(float(coef[0]), 2),
                    "t": round(float(t[0]), 1), "r2": round(r2, 2), "weeks": len(base),
                    "level": exposure_level_beta(float(coef[0]))})
    for s in spec:
        if s["key"] == "market" or s["key"] not in factors:
            continue
        f = weekly_moves(factors[s["key"]][factors[s["key"]].index >= start], s.get("rate", False))
        df = pd.concat([y_all.rename("y"), m.rename("m"), f.rename("f")], axis=1).dropna()
        if len(df) < MIN_WEEKS:
            continue
        coef, t, r2 = _ols(df["y"].to_numpy(), df[["m", "f"]].to_numpy())
        out.append({**_spec_view(s), "beta": round(float(coef[1]), 3),
                    "t": round(float(t[1]), 1), "r2": round(r2, 2), "weeks": len(df),
                    "level": exposure_level_t(float(t[1]))})
    return out


def _spec_view(s: dict) -> dict:
    return {"key": s["key"], "label": s["label"], "risk": s["risk"], "rate": bool(s.get("rate"))}


def exposure_by_risk(readings: list[dict]) -> dict[str, dict]:
    """The strongest reading per risk — three commodities feed one 大宗商品 row."""
    rank = {"高": 3, "中": 2, "低": 1}
    best: dict[str, dict] = {}
    for r in readings:
        cur = best.get(r["risk"])
        if cur is None or (rank[r["level"]], abs(r["t"])) > (rank[cur["level"]], abs(cur["t"])):
            best[r["risk"]] = r
    return best


def describe_reading(r: dict) -> str:
    if r["key"] == "market":
        return f"β {r['beta']:.2f} 对 {r['label']} · R² {r['r2']:.2f} · {r['weeks']} 周"
    # A return factor's coefficient is an elasticity (1% → β%); a yield's is
    # the stock's return per percentage point, a fraction until scaled.
    if r["rate"]:
        head = f"{r['label']} 每上行 1pct → 股价 {r['beta'] * 100:+.1f}%"
    else:
        head = f"{r['label']} 每涨 1% → 股价 {r['beta']:+.2f}%"
    return f"{head}（剔除大盘后 t={r['t']:+.1f}）· {r['weeks']} 周"


def factor_move_items(readings: list[dict], factors: dict[str, pd.Series],
                      as_of: date, lookback: int = 20) -> list[dict]:
    """
    Factor moves as evidence, filed by rule: a factor this stock is exposed to
    (中 or 高) that moved unusually over the last `lookback` sessions. The
    direction is the move times the measured sign — oil up is 利空 for an
    airline and 利好 for a driller, and the regression already knows which.
    """
    items = []
    for r in readings:
        if r["level"] == "低" or r["key"] not in factors:
            continue
        s = factors[r["key"]].dropna()
        s = s[s.index <= pd.Timestamp(as_of)]
        if len(s) < lookback * 6:
            continue
        chg = (s.diff(lookback) if r["rate"] else s.pct_change(lookback)).dropna()
        sd = float(chg.std())
        last = float(chg.iloc[-1])
        if sd <= 0:
            continue
        z = last / sd
        if abs(z) < 1:
            continue
        impact = last * (r["beta"] if r["key"] != "market" else 1.0)
        move = f"{last:+.2f}pct" if r["rate"] else f"{last * 100:+.1f}%"
        title = f"近{lookback}个交易日 {r['label']} {move}（{abs(z):.1f} 倍标准差）"
        d = s.index[-1].date().isoformat()
        items.append({
            "eid": evidence_id("factor", f"{r['key']}|{lookback}", d),
            "d": d, "source": "factor", "kind": "data", "title": title, "url": None,
            "risk": r["risk"], "dir": "up" if impact > 0 else "down",
            "strength": 3 if abs(z) >= 2.5 else 2 if abs(z) >= 1.75 else 1,
            "fact": True, "note": f"敞口 {r['level']}：{describe_reading(r)}",
            "tagged_by": "rule",
        })
    return items


def fetch_factors(market_code: str, years: int = 4) -> dict[str, pd.Series]:
    """
    The market from the app's own benchmark fetch (沪深300 via Tushare, S&P
    500 via Yahoo) — the same series every beta elsewhere in the app uses;
    Yahoo's 000300.SS came back empty inside a multi-ticker download, which
    left an A-share with no readings at all. The rest from one Yahoo download.
    """
    import markets
    import yfinance as yf
    spec = [s for s in FACTORS[market_code] if s["key"] != "market"]
    out = {}
    bench = markets.get(market_code).fetch_benchmark(years=years)
    if bench is not None and len(bench) > 60:
        bench.index = pd.to_datetime(bench.index).tz_localize(None)
        out["market"] = bench
    raw = yf.download([s["ticker"] for s in spec], period=f"{years}y",
                      progress=False, auto_adjust=True)["Close"]
    for s in spec:
        if s["ticker"] in raw.columns:
            col = raw[s["ticker"]].dropna()
            col.index = pd.to_datetime(col.index).tz_localize(None)
            if len(col) > 60:
                out[s["key"]] = col
    return out


def regression_start(market_code: str) -> pd.Timestamp:
    """A-shares from the regime anchor; elsewhere three years."""
    import markets
    anchor = markets.get(market_code).conv.regime_anchor
    if anchor == pd.Timestamp.min:
        return pd.Timestamp(date.today() - timedelta(days=3 * 365))
    from analysis_engine import REGIME_ANCHOR
    return max(anchor, REGIME_ANCHOR)


# ── storage ──────────────────────────────────────────────────────────────────
_EVIDENCE_COLS = ("eid", "d", "source", "kind", "title", "url", "risk", "dir",
                  "strength", "fact", "note", "tagged_by")


def _db():
    import data_manager
    return data_manager.db


def load_evidence(symbol: str, db=None) -> list[dict]:
    """Raises when the table is unreadable — the caller decides what that means."""
    db = db or _db()
    since = (date.today() - timedelta(days=WINDOW_DAYS + TREND_DAYS)).isoformat()
    df = db.read_all(EVIDENCE_TABLE, filters={"symbol": symbol}, order_by="-d,eid")
    if df is None or df.empty:
        return []
    df = df[df["d"].astype(str) >= since]
    rows = []
    for r in df.to_dict("records"):
        row = {k: r.get(k) for k in _EVIDENCE_COLS}
        row["d"] = str(row["d"])[:10]
        for k in ("risk", "dir", "url", "note"):
            if row[k] is not None and (row[k] != row[k]):        # NaN from pandas
                row[k] = None
        if row["strength"] is not None and row["strength"] == row["strength"]:
            row["strength"] = int(row["strength"])
        else:
            row["strength"] = None
        rows.append(row)
    return rows


def save_evidence(symbol: str, items: list[dict], db=None) -> None:
    if not items:
        return
    db = db or _db()
    now = datetime.now(timezone.utc).isoformat()
    db.insert_records(EVIDENCE_TABLE, [
        {"symbol": symbol, **{k: it.get(k) for k in _EVIDENCE_COLS}, "fetched_at": now}
        for it in items], upsert=True)


def read_cache(symbol: str, section: str, db=None) -> dict | None:
    try:
        df = (db or _db()).read_table(CACHE_TABLE, filters={"symbol": symbol, "section": section})
        if df is None or df.empty:
            return None
        row = df.iloc[0].to_dict()
        return {"payload": json.loads(row["payload"]), "generated_at": str(row.get("generated_at"))}
    except Exception:
        return None


def write_cache(symbol: str, section: str, payload, db=None) -> bool:
    try:
        (db or _db()).insert_records(CACHE_TABLE, [{
            "symbol": symbol, "section": section,
            "payload": json.dumps(payload, ensure_ascii=False),
            "generated_at": datetime.now(timezone.utc).isoformat()}], upsert=True)
        return True
    except Exception as exc:
        print(f"[risk_radar] ⚠️ cache write failed for {symbol}/{section}: {exc}")
        return False


# ── the board ────────────────────────────────────────────────────────────────
def assemble(symbol: str, name: str, market_code: str, items: list[dict], *,
             profile: dict | None, readings: list[dict], synthesis: dict | None,
             sources: list[dict], as_of: date | None = None,
             refreshed_at: str | None = None, synthesized_at: str | None = None,
             persisted: bool = True) -> dict:
    """Everything the page draws, from stored evidence. No fetching, no AI."""
    as_of = as_of or date.today()
    measured = exposure_by_risk(readings)
    judged = (profile or {}).get("exposure") or {}
    synth = (synthesis or {}).get("risks") or {}

    risks = []
    for r in TAXONOMY:
        rid = r["id"]
        if rid in measured:
            m = measured[rid]
            expo, expo_src, expo_note = m["level"], "measured", describe_reading(m)
        elif rid in judged:
            expo, expo_src = judged[rid].get("level", DEFAULT_EXPOSURE), "ai"
            expo_note = judged[rid].get("why") or ""
        else:
            expo, expo_src, expo_note = DEFAULT_EXPOSURE, "default", "未评估，按「中」计"

        now = score(items, rid, expo, as_of)
        week = score(items, rid, expo, as_of - timedelta(days=7))
        trend = [score(items, rid, expo, as_of - timedelta(days=k))
                 for k in range(TREND_DAYS - 1, -1, -1)]
        ev = sorted(
            ({**it, "weight": round(weight(it, as_of), 3)} for it in items
             if it.get("risk") == rid and weight(it, as_of) > 0),
            key=lambda e: (-e["weight"], e["d"]))
        s = synth.get(rid) or {}
        risks.append({
            "id": rid, "kind": r["kind"], "name": r["name"], "en": r["en"],
            "exposure": expo, "exposure_source": expo_src, "exposure_note": expo_note,
            "down": now["down"], "up": now["up"], "n": now["n"],
            "tier": tier(now["down"], now["n"]), "lean": lean(now["down"], now["up"]),
            "d7_down": now["down"] - week["down"], "d7_up": now["up"] - week["up"],
            "trend": {"down": [t["down"] for t in trend], "up": [t["up"] for t in trend]},
            "bear": s.get("bear"), "bull": s.get("bull"), "watch": s.get("watch") or [],
            "evidence": ev[:25],
        })

    scored = [x for x in risks if x["n"] >= MIN_EVIDENCE]
    wsum = sum(EXPOSURE_MULT[x["exposure"]] for x in scored) or 1.0
    down = round(sum(EXPOSURE_MULT[x["exposure"]] * x["down"] for x in scored) / wsum)
    up = round(sum(EXPOSURE_MULT[x["exposure"]] * x["up"] for x in scored) / wsum)
    top = max(scored, key=lambda x: EXPOSURE_MULT[x["exposure"]] * x["down"], default=None)
    counted = [it for it in items if weight(it, as_of) > 0]
    facts = sum(1 for it in counted if it.get("fact", True))

    return {
        "symbol": symbol, "name": name, "market": market_code,
        "as_of": as_of.isoformat(), "refreshed_at": refreshed_at,
        "synthesized_at": synthesized_at, "persisted": persisted,
        "summary": {
            "down": down if scored else None, "up": up if scored else None,
            "down_tier": tier(down, MIN_EVIDENCE) if scored else "证据不足",
            "up_tier": tier(up, MIN_EVIDENCE) if scored else "证据不足",
            "top": ({"id": top["id"], "name": top["name"], "down": top["down"],
                     "exposure": top["exposure"], "d7_down": top["d7_down"]} if top else None),
            "n": len(counted), "facts": facts, "opinions": len(counted) - facts,
            "stored": len(items),
            "untagged": sum(1 for it in items if not it.get("tagged_by")),
            "irrelevant": sum(1 for it in items if it.get("tagged_by") and not it.get("risk")),
        },
        "sources": sources, "readings": readings, "risks": risks,
        "method": {"window_days": WINDOW_DAYS, "min_evidence": MIN_EVIDENCE,
                   "half_life": HALF_LIFE, "credibility": CREDIBILITY,
                   "opinion_weight": OPINION_WEIGHT, "scale_k": SCALE_K},
    }


def board(symbol: str, name: str, market_code: str, db=None) -> dict:
    """The page as last refreshed. Raises RuntimeError when the tables are missing."""
    try:
        items = load_evidence(symbol, db)
    except Exception as exc:
        raise RuntimeError(f"风险雷达数据表不可用（请先运行 {MIGRATION}）：{exc}"[:300])
    state = read_cache(symbol, "state", db) or {}
    synth = read_cache(symbol, "synthesis", db) or {}
    profile = read_cache(symbol, "profile", db) or {}
    st = state.get("payload") or {}
    return assemble(symbol, name, market_code, items,
                    profile=profile.get("payload"), readings=st.get("readings") or [],
                    synthesis=synth.get("payload"), sources=st.get("sources") or [],
                    refreshed_at=state.get("generated_at"),
                    synthesized_at=synth.get("generated_at"))


# ── refresh ──────────────────────────────────────────────────────────────────
#: Newest untagged items sent to the tagger per refresh, and per call.
MAX_TAG = 160
#: 40 per call lost one batch in four to a truncated reply; 25 has not.
TAG_BATCH = 25


def refresh(symbol: str, name: str, market_code: str, industry: str = "",
            db=None, synthesize: bool = True) -> dict:
    """
    Collect, file, store, summarise — then the board. Slow: one to three
    minutes, most of it the AI. Each stage that fails is reported in
    `sources` and the rest still runs; a board with one broken feed is more
    use than an error page.
    """
    import markets
    import risk_ai
    import risk_sources

    as_of = date.today()
    persisted = True
    try:
        stored = load_evidence(symbol, db)
    except Exception as exc:
        print(f"[risk_radar] ⚠️ evidence table unreadable for {symbol}: {exc}")
        stored, persisted = [], False

    cached = read_cache(symbol, "profile", db)
    profile = (cached or {}).get("payload")
    if not profile:
        profile = risk_ai.profile(name, symbol, market_code, industry)
        write_cache(symbol, "profile", profile, db)

    def prices():
        # One thread for both, in order: the A-share fetchers each re-run
        # data_manager.init_tushare(), and two of those at once read a
        # half-written token file — the benchmark then came back None and
        # the stock silently got no readings at all.
        market, code = markets.parse(symbol)
        factors = fetch_factors(market_code)
        df = market.fetch_ohlcv(code, years=4)
        return factors, (None if df is None or df.empty else df["Close"])

    if market_code == "CN":
        try:
            risk_sources._pro()          # made once, before the feeds share it
        except Exception as exc:
            print(f"[risk_radar] ⚠️ Tushare unavailable: {exc}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_collect = pool.submit(risk_sources.collect, symbol, market_code, name, profile)
        f_prices = pool.submit(prices)
        collected, sources = f_collect.result()
        readings, factor_items = [], []
        label = "因子敞口（价格回归）"
        try:
            factors, stock = f_prices.result()
            if stock is None:
                raise RuntimeError("股价读取失败")
            if "market" not in factors:
                raise RuntimeError("大盘指数读取失败，无法剔除大盘")
            readings = measure_exposure(stock, factors, FACTORS[market_code],
                                        regression_start(market_code))
            factor_items = factor_move_items(readings, factors, as_of)
            sources.append({"source": "factor", "label": label, "ok": bool(readings),
                            "n": len(factor_items),
                            "reason": None if readings else f"不足 {MIN_WEEKS} 周的共同数据"})
        except Exception as exc:
            sources.append({"source": "factor", "label": label, "ok": False,
                            "n": 0, "reason": f"{type(exc).__name__}: {exc}"[:160]})

    known = {it["eid"]: it for it in stored}
    fresh = []
    for it in collected + factor_items:
        old = known.get(it["eid"])
        if old is None or (it.get("tagged_by") == "rule"):
            fresh.append(it)
        elif old.get("d") != it.get("d"):
            fresh.append({**old, "d": it["d"]})          # an upcoming unlock stays current

    todo = sorted((it for it in fresh if not it.get("tagged_by")),
                  key=lambda it: it["d"], reverse=True)[:MAX_TAG]
    todo += [it for it in stored if not it.get("tagged_by")][:max(0, MAX_TAG - len(todo))]
    if todo:
        batches = [todo[i:i + TAG_BATCH] for i in range(0, len(todo), TAG_BATCH)]
        with ThreadPoolExecutor(max_workers=7) as pool:
            results = list(pool.map(lambda b: risk_ai.tag(b, name, market_code), batches))
        tagged = {eid: t for res in results for eid, t in res.items()}
        failed = sum(1 for b, res in zip(batches, results) if not res)
        if failed:
            sources.append({"source": "tagger", "label": "AI 归类", "ok": False, "n": 0,
                            "reason": f"{failed}/{len(batches)} 批归类失败，下次刷新重试"})
        for it in todo:
            t = tagged.get(it["eid"])
            if t is not None:
                it.update(t, tagged_by="ai")
        retagged = [it for it in todo if it.get("tagged_by") and it["eid"] in known]
        fresh_ids = {it["eid"] for it in fresh}
        fresh += [it for it in retagged if it["eid"] not in fresh_ids]

    if persisted:
        try:
            save_evidence(symbol, fresh, db)
        except Exception as exc:
            print(f"[risk_radar] ⚠️ evidence write failed for {symbol}: {exc}")
            persisted = False

    merged = {it["eid"]: it for it in stored}
    merged.update({it["eid"]: it for it in fresh})
    items = list(merged.values())

    now_iso = datetime.now(timezone.utc).isoformat()
    write_cache(symbol, "state", {"readings": readings, "sources": sources}, db)

    synthesis, synthesized_at = None, None
    if synthesize:
        synthesis, synthesized_at = _synthesize(symbol, name, market_code, items, profile,
                                                readings, sources, as_of, db)
    if synthesis is None:
        old = read_cache(symbol, "synthesis", db) or {}
        synthesis, synthesized_at = old.get("payload"), old.get("generated_at")

    return assemble(symbol, name, market_code, items, profile=profile, readings=readings,
                    synthesis=synthesis, sources=sources, as_of=as_of,
                    refreshed_at=now_iso, synthesized_at=synthesized_at,
                    persisted=persisted)


def _synthesize(symbol, name, market_code, items, profile, readings, sources, as_of, db):
    import risk_ai
    pre = assemble(symbol, name, market_code, items, profile=profile, readings=readings,
                   synthesis=None, sources=sources, as_of=as_of)
    try:
        synthesis = risk_ai.synthesize(name, market_code, pre["risks"])
    except Exception as exc:
        sources.append({"source": "synthesis", "label": "AI 论点", "ok": False, "n": 0,
                        "reason": f"{type(exc).__name__}: {exc}"[:160]})
        return None, None
    write_cache(symbol, "synthesis", synthesis, db)
    return synthesis, datetime.now(timezone.utc).isoformat()


def synthesize(symbol: str, name: str, market_code: str, db=None) -> dict:
    """
    Rewrite the 利空/利好 cases from stored evidence, then the board.

    Its own step so the page can run it as a second request: a refresh of a
    well-covered A-share took 193s with the cases included, too near Cloud
    Run's 300s request limit to leave both in one.
    """
    try:
        items = load_evidence(symbol, db)
    except Exception as exc:
        raise RuntimeError(f"风险雷达数据表不可用（请先运行 {MIGRATION}）：{exc}"[:300])
    profile = (read_cache(symbol, "profile", db) or {}).get("payload")
    state = (read_cache(symbol, "state", db) or {})
    st = state.get("payload") or {}
    sources = list(st.get("sources") or [])
    synthesis, at = _synthesize(symbol, name, market_code, items, profile,
                                st.get("readings") or [], sources, date.today(), db)
    if synthesis is None:
        raise RuntimeError(next((s["reason"] for s in sources if s["source"] == "synthesis"),
                                "AI 论点生成失败"))
    return assemble(symbol, name, market_code, items, profile=profile,
                    readings=st.get("readings") or [], synthesis=synthesis,
                    sources=st.get("sources") or [], refreshed_at=state.get("generated_at"),
                    synthesized_at=at)
