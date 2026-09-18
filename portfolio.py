"""
portfolio.py — one book out of several accounts, and what its risk looks like.

Two jobs, both pure: nothing here opens a socket, so all of it is testable
without a brokerage connection.

Consolidation
-------------
Questrade reports positions per account, and the same name routinely sits in
two or three of them. A holding is therefore a symbol with a list of accounts
behind it, not a row per account — otherwise "how much NVDA do I own" needs
mental arithmetic and "what is my biggest position" is simply wrong.

Two things stay per-account rather than being summed away:

  * market value, because that is the number you check against the statement;
  * average cost, because it is not one number. ACB is tracked per account,
    a TFSA has no cost base for tax at all, and an RRSP loss is not
    harvestable. Summing them produces a figure that matches nothing.

Currency is the other trap. A Canadian book holding US names carries USD/CAD
exposure whether or not the owner thinks of it that way, so everything is
converted to one base currency and the rate used is reported alongside.

Risk
----
`risk()` takes returns, not prices from a broker: Questrade's candles are
unadjusted for splits and dividends, and a portfolio return series built on
unadjusted prices puts a -60% day in the record every time a holding splits.
The caller supplies adjusted history (Yahoo) converted to the base currency.

The portfolio series is TODAY'S weights applied backwards. That is a statement
about the book you hold now — what it would have done — not a performance
record of what you actually did, which would need the trade history. Every
number derived from it carries that caveat, and the payload says so rather
than leaving it to be assumed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252

#: Yahoo suffix per Questrade listing exchange. Questrade's own symbol is
#: usually already right for Yahoo; this is the authority when it is not.
EXCHANGE_SUFFIX = {
    "TSX": ".TO",
    "TSXV": ".V",
    "CNSX": ".CN",
    "NEO": ".NE",
}

#: Exchanges whose tickers Yahoo carries bare.
US_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "PINK", "OTCBB"}

#: What the two benchmark choices mean. Kept here so the API, the tests and
#: the UI cannot disagree about which index is which.
BENCHMARKS = {
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ 综合",
    "^GSPTSE": "S&P/TSX 综合",
}

#: Below this many overlapping sessions the statistics are not worth printing.
MIN_SESSIONS = 120


# ── symbols ──────────────────────────────────────────────────────────────────
def to_yahoo(symbol: str, exchange: str = "") -> str | None:
    """
    Questrade's ticker in Yahoo's spelling, or None if it cannot be mapped.

    None rather than a guess: a wrong mapping silently prices one holding off
    another company's chart, which is far worse than a row that says it could
    not find a price. Options and anything without a plain ticker land here.
    """
    s = (symbol or "").strip().upper()
    if not s or " " in s:
        return None
    ex = (exchange or "").strip().upper()

    if ex in US_EXCHANGES or (not ex and "." not in s):
        # Class shares: Questrade writes BRK.B, Yahoo writes BRK-B.
        head, _, tail = s.partition(".")
        return f"{head}-{tail}" if tail and len(tail) <= 2 else s

    suffix = EXCHANGE_SUFFIX.get(ex)
    if suffix:
        base = s.split(".")[0]
        return f"{base}{suffix}"

    # No exchange given: trust the suffix Questrade already wrote, fixing the
    # one spelling the two sources disagree on.
    if s.endswith(".VN"):
        return s[:-3] + ".V"
    return s


def kind_of(quote_type: str | None, security_type: str | None) -> str:
    """
    ETF or ordinary share.

    Questrade cannot answer this: its securityType enum is Stock / Option /
    Bond / Right / Gold / MutualFund / Index, and every ETF comes back as
    "Stock". Yahoo's quoteType does distinguish them, so that is the source,
    with Questrade's answer used only for the cases Yahoo has no opinion on.
    """
    q = (quote_type or "").upper()
    if q == "ETF":
        return "ETF"
    if q == "MUTUALFUND":
        return "基金"
    if q in ("EQUITY", "STOCK"):
        return "股票"
    s = (security_type or "").title()
    return {"Stock": "股票", "Option": "期权", "Bond": "债券",
            "MutualFund": "基金", "Index": "指数", "Gold": "黄金",
            "Right": "权证"}.get(s, s or "其他")


# ── consolidation ────────────────────────────────────────────────────────────
def _num(v, default=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else f


def consolidate(accounts: list[dict], positions: dict[str, list[dict]],
                balances: dict[str, dict], meta: dict[str, dict], *,
                base: str = "CAD", rates: dict[str, float] | None = None,
                rate_source: str = "") -> dict:
    """
    One book. `meta[symbol]` carries {name, currency, exchange, kind, yahoo}.

    `rates` maps a currency to its value in `base` — {"USD": 1.37} for a CAD
    book. A currency with no rate is NOT silently treated as 1:1; its holdings
    are still listed, with their base-currency value left null and a warning
    raised, because a USD position counted as CAD understates the book by a
    third and looks entirely plausible.
    """
    rates = dict(rates or {})
    rates.setdefault(base, 1.0)
    warnings: list[str] = []

    def to_base(value, currency):
        if value is None:
            return None
        rate = rates.get((currency or base).upper())
        return None if rate is None else value * rate

    by_symbol: dict[str, dict] = {}
    account_rows: list[dict] = []
    labels = {a["id"]: a for a in accounts}

    for account in accounts:
        aid = account["id"]
        rows = positions.get(aid) or []
        acc_mv = 0.0
        for p in rows:
            symbol = str(p.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            qty = _num(p.get("openQuantity"), 0.0) or 0.0
            if qty == 0:
                continue                      # closed today; nothing to hold
            m = meta.get(symbol, {})
            ccy = str(m.get("currency") or "").upper() or base
            mv = _num(p.get("currentMarketValue"))
            cost = _num(p.get("totalCost"))
            mv_base = to_base(mv, ccy)
            if mv is not None and mv_base is None:
                warnings.append(f"{symbol}：缺少 {ccy} 汇率，未计入总市值")
            if mv_base is not None:
                acc_mv += mv_base

            h = by_symbol.setdefault(symbol, {
                "symbol": symbol,
                "name": str(m.get("name") or symbol),
                "kind": str(m.get("kind") or "其他"),
                "currency": ccy,
                "exchange": str(m.get("exchange") or ""),
                "yahoo": m.get("yahoo"),
                "quantity": 0.0, "cost": 0.0, "market_value": 0.0,
                "market_value_base": 0.0, "price": _num(p.get("currentPrice")),
                "open_pnl": 0.0, "accounts": [],
            })
            h["quantity"] += qty
            h["cost"] += cost or 0.0
            h["market_value"] += mv or 0.0
            h["market_value_base"] += mv_base or 0.0
            h["open_pnl"] += _num(p.get("openPnl"), 0.0) or 0.0
            if h["price"] is None:
                h["price"] = _num(p.get("currentPrice"))
            h["accounts"].append({
                "id": aid,
                "label": labels[aid]["label"],
                "type": labels[aid]["type"],
                "quantity": qty,
                # Per account, because ACB is per account — and in a TFSA it
                # is not a tax figure at all.
                "avg_cost": _num(p.get("averageEntryPrice")),
                "market_value": mv,
                "market_value_base": mv_base,
                "open_pnl": _num(p.get("openPnl")),
            })

        account_rows.append({
            **account,
            "positions": len(rows),
            "market_value_base": round(acc_mv, 2),
            **_balances(balances.get(aid) or {}, base, to_base),
        })

    holdings = list(by_symbol.values())
    total_mv = sum(h["market_value_base"] for h in holdings)
    for h in holdings:
        h["avg_cost"] = (h["cost"] / h["quantity"]) if h["quantity"] else None
        h["open_pnl_pct"] = (h["open_pnl"] / h["cost"] * 100) if h["cost"] else None
        h["weight_pct"] = (h["market_value_base"] / total_mv * 100) if total_mv else None
        h["split"] = len(h["accounts"]) > 1
        for k in ("quantity", "cost", "market_value", "market_value_base", "open_pnl"):
            h[k] = round(h[k], 4)
    holdings.sort(key=lambda h: -(h["market_value_base"] or 0))

    cash = sum(a.get("cash_base") or 0.0 for a in account_rows)
    cost = sum(h["cost"] for h in holdings)
    pnl = sum(h["open_pnl"] for h in holdings)
    unmapped = [h["symbol"] for h in holdings if not h["yahoo"]]
    if unmapped:
        warnings.append("以下持仓无法匹配行情源，未纳入风险分析："
                        + "、".join(unmapped[:8]))

    return {
        "base": base,
        "fx": {"rates": {k: round(v, 6) for k, v in rates.items()},
               "source": rate_source or "未知"},
        "accounts": account_rows,
        "holdings": holdings,
        "totals": {
            "market_value": round(total_mv, 2),
            "cash": round(cash, 2),
            "equity": round(total_mv + cash, 2),
            "cost": round(cost, 2),
            "open_pnl": round(pnl, 2),
            "open_pnl_pct": round(pnl / cost * 100, 2) if cost else None,
            "positions": len(holdings),
            "accounts": len(account_rows),
        },
        "mix": _mix(holdings, total_mv),
        "warnings": warnings,
    }


def _balances(raw: dict, base: str, to_base) -> dict:
    """Cash and equity per currency, plus the base-currency roll-up."""
    per = []
    cash_base = 0.0
    for b in raw.get("perCurrencyBalances") or []:
        ccy = str(b.get("currency") or "").upper()
        cash = _num(b.get("cash"), 0.0) or 0.0
        per.append({
            "currency": ccy,
            "cash": round(cash, 2),
            "market_value": _num(b.get("marketValue")),
            "total_equity": _num(b.get("totalEquity")),
        })
        cash_base += to_base(cash, ccy) or 0.0
    return {"per_currency": per, "cash_base": round(cash_base, 2)}


def _mix(holdings: list[dict], total: float) -> dict:
    """Weight by currency and by instrument kind — the two cuts worth a chart."""
    def bucket(key):
        out: dict[str, float] = {}
        for h in holdings:
            out[h[key]] = out.get(h[key], 0.0) + (h["market_value_base"] or 0.0)
        return [{"name": k, "value": round(v, 2),
                 "pct": round(v / total * 100, 2) if total else None}
                for k, v in sorted(out.items(), key=lambda kv: -kv[1])]

    return {"currency": bucket("currency"), "kind": bucket("kind")}


def implied_fx(balances: list[dict], base: str, quote: str) -> float | None:
    """
    The broker's own USD/CAD, read out of its combined balances.

    Questrade reports `combinedBalances` as the WHOLE account expressed in each
    currency, so the ratio of the two is the rate it used — which keeps the
    page's totals consistent with the statement instead of a few tenths of a
    percent away from it on a different mid-market quote.
    """
    got = {str(b.get("currency") or "").upper(): _num(b.get("totalEquity"))
           for b in balances or []}
    top, bottom = got.get(base.upper()), got.get(quote.upper())
    if not top or not bottom or bottom == 0:
        return None
    rate = top / bottom
    # A plausibility band. Anything outside it means the two entries were not
    # the same book — an empty sleeve, say — and a bad rate silently rescales
    # every holding.
    return rate if 0.2 < rate < 5 else None


# ── risk ─────────────────────────────────────────────────────────────────────
def risk(weights: dict[str, float], prices: pd.DataFrame, benchmark: str, *,
         window: int = 3 * TRADING_DAYS, rf_annual: float = 0.0) -> dict:
    """
    What the CURRENT book looks like against an index.

    `weights` are fractions summing to ~1 over the symbols that could be
    priced; `prices` is an adjusted, base-currency close panel including the
    benchmark column. Raises LookupError when there is not enough overlap —
    a beta from forty sessions is a number, not an estimate.
    """
    if benchmark not in prices.columns:
        raise LookupError(f"缺少基准 {benchmark} 的行情")
    names = [s for s in weights if s in prices.columns and s != benchmark]
    if not names:
        raise LookupError("没有任何持仓可以取得行情，无法计算风险指标")

    px = prices[names + [benchmark]].dropna(how="all").tail(window)
    rets = px.ffill().pct_change().dropna(how="any")
    if len(rets) < MIN_SESSIONS:
        raise LookupError(f"可用历史仅 {len(rets)} 个交易日，少于 {MIN_SESSIONS} 天，"
                          "统计量不可靠")

    w = pd.Series({n: weights[n] for n in names}, dtype=float)
    w = w / w.sum()
    port = (rets[names] * w).sum(axis=1)
    bench = rets[benchmark]

    stats = _series_stats(port, bench, rf_annual)
    stats.update({
        "benchmark": benchmark,
        "benchmark_name": BENCHMARKS.get(benchmark, benchmark),
        "sessions": int(len(rets)),
        "from": rets.index[0].strftime("%Y-%m-%d"),
        "to": rets.index[-1].strftime("%Y-%m-%d"),
        # How much of the book these statistics actually speak for: a
        # portfolio that is 30% in something unpriceable has a beta for
        # the other 70%, and saying so is the difference between a
        # measurement and a claim.
        "covered_pct": round(sum(weights[n] for n in names)
                             / (sum(weights.values()) or 1) * 100, 1),
        "benchmark_stats": _series_stats(bench, bench, rf_annual, is_benchmark=True),
        "holdings": _holding_risk(rets[names], port, bench, w),
        "concentration": _concentration(w),
        # Said in the payload, not only in a comment: every number above is
        # today's holdings replayed over history, not the account's own record.
        "basis": "以当前持仓权重回溯，非账户实际历史收益",
    })
    return stats


def _series_stats(r: pd.Series, bench: pd.Series, rf_annual: float, *,
                  is_benchmark: bool = False) -> dict:
    """
    `is_benchmark` is passed in rather than inferred from `r.equals(bench)`.
    A portfolio that tracks its index perfectly legitimately has beta 1.00 and
    100% capture, and inferring would erase exactly that finding — the one an
    index-ETF holder most wants to see.
    """
    ann_ret = float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)
    ann_vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    downside = r[r < 0]
    dd_vol = float(downside.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(downside) > 2 else None

    beta = alpha = r2 = None
    if not is_benchmark and bench.std(ddof=1) > 0:
        cov = float(np.cov(r, bench, ddof=1)[0, 1])
        beta = cov / float(bench.var(ddof=1))
        alpha = float((r.mean() - beta * bench.mean()) * TRADING_DAYS)
        corr = float(r.corr(bench))
        r2 = corr ** 2

    curve = (1 + r).cumprod()
    drawdown = float((curve / curve.cummax() - 1).min())
    excess = r - bench
    te = float(excess.std(ddof=1) * math.sqrt(TRADING_DAYS))

    return {
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "downside_vol_pct": round(dd_vol * 100, 2) if dd_vol else None,
        "sharpe": round((ann_ret - rf_annual) / ann_vol, 2) if ann_vol else None,
        "sortino": round((ann_ret - rf_annual) / dd_vol, 2) if dd_vol else None,
        "max_drawdown_pct": round(drawdown * 100, 2),
        "beta": round(beta, 3) if beta is not None else None,
        "alpha_pct": round(alpha * 100, 2) if alpha is not None else None,
        "r2": round(r2, 3) if r2 is not None else None,
        # Zero is an answer here, not a missing value: a portfolio that tracks
        # its index perfectly has exactly no tracking error, and reporting that
        # as "—" hides the finding an index holder is looking for.
        "tracking_error_pct": round(te * 100, 2),
        "info_ratio": round(float(excess.mean() * TRADING_DAYS) / te, 2) if te > 0 else None,
        # Historical 1-day 95% VaR: the 5th percentile of what actually
        # happened, not a normal approximation, which understates fat tails.
        "var95_pct": round(float(r.quantile(0.05)) * 100, 2),
        "worst_day_pct": round(float(r.min()) * 100, 2),
        "up_capture_pct": None if is_benchmark else _capture(r, bench, up=True),
        "down_capture_pct": None if is_benchmark else _capture(r, bench, up=False),
    }


def _capture(r: pd.Series, bench: pd.Series, *, up: bool) -> float | None:
    mask = bench > 0 if up else bench < 0
    if mask.sum() < 10:
        return None
    b = float((1 + bench[mask]).prod() - 1)
    p = float((1 + r[mask]).prod() - 1)
    return round(p / b * 100, 1) if b else None


def _holding_risk(rets: pd.DataFrame, port: pd.Series, bench: pd.Series,
                  w: pd.Series) -> list[dict]:
    """
    Per holding: its own volatility, its beta, and — the useful one — how much
    of the PORTFOLIO's risk it is responsible for.

    Marginal contribution to risk, w_i·cov(r_i, r_p)/σ_p, sums exactly to the
    portfolio volatility. A 4% position in something wild can contribute more
    risk than a 20% position in a utility, and a list of individual vols will
    never show that.
    """
    port_vol = float(port.std(ddof=1))
    bench_var = float(bench.var(ddof=1))
    rows = []
    for name in rets.columns:
        r = rets[name]
        ctr = (float(w[name]) * float(np.cov(r, port, ddof=1)[0, 1]) / port_vol
               if port_vol else 0.0)
        rows.append({
            "symbol": name,
            "weight_pct": round(float(w[name]) * 100, 2),
            "ann_vol_pct": round(float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) * 100, 2),
            "beta": round(float(np.cov(r, bench, ddof=1)[0, 1]) / bench_var, 3)
                    if bench_var else None,
            "corr_bench": round(float(r.corr(bench)), 2),
            "risk_pct": round(ctr / port_vol * 100, 2) if port_vol else None,
            "ann_return_pct": round(float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1) * 100, 2),
        })
    rows.sort(key=lambda x: -(x["risk_pct"] or 0))
    return rows


def _concentration(w: pd.Series) -> dict:
    """
    How few names this book really is.

    Effective N = 1/Σw² — the number of equally-weighted positions that would
    be this concentrated. Twenty holdings with one at 40% is not twenty
    holdings, and the count alone never says so.
    """
    hhi = float((w ** 2).sum())
    top = w.sort_values(ascending=False)
    return {
        "positions": int(len(w)),
        "effective_n": round(1 / hhi, 1) if hhi else None,
        "hhi": round(hhi, 4),
        "top1_pct": round(float(top.iloc[0]) * 100, 1),
        "top5_pct": round(float(top.head(5).sum()) * 100, 1),
    }
