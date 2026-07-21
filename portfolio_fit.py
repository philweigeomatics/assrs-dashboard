"""
portfolio_fit.py — does the stock currently on the Single Stock Analysis page
belong in one of the user's saved portfolios?

Answers one question: if I add this candidate at weight w to a saved mandate,
does it diversify the book, duplicate what is already there, or deliberately
concentrate an exposure I already hold?

Design notes
------------
* Pure functions — no Streamlit import here, so the module is unit-testable on
  its own. The page wraps the loaders in @st.cache_data (same convention as
  market_leverage.py).
* Analysis window is anchored at REGIME_START (2024-10-14). The A-share market
  that began after 2024-09-24 behaves differently enough from what preceded it
  that blending the two produces correlations describing a market that no
  longer exists. The window is max(REGIME_START, today - 3y), so the 3-year
  arm only starts binding in Oct 2027.
* Prices are qfq via get_single_stock_data_live() — the same basis the
  Portfolio Optimizer and the fund backfill use, so numbers here are directly
  comparable to what those pages report.
* Weight accounting assumes PRO-RATA DILUTION: existing holdings are scaled by
  (1 - w) so the book still sums to 1. This answers "should I make room for
  this", not "should I deploy new cash here".
* The quantitative half is deterministic and free. The qualitative half is a
  single DeepSeek call (analyse_business_overlap) and is the caller's choice to
  invoke — it is model knowledge, not database fact, and is labelled as such in
  the UI.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import ai_client
import data_manager
from db_manager import db

# ── Constants ─────────────────────────────────────────────────────────────────

# Start of the post-2024-09-24 A-share regime. Analysis never reaches behind
# this date, no matter how much history a stock has.
REGIME_START = "2024-10-14"

MAX_LOOKBACK_YEARS = 3
TRADING_DAYS = 242
MIN_OVERLAP_DAYS = 120          # below this the panel flags low confidence
RECENT_WINDOW = 60              # trading days for the "recent" correlation
VAR_CONFIDENCE = 0.95


# ── Window ────────────────────────────────────────────────────────────────────

def analysis_window(today: date | None = None) -> tuple[str, str]:
    """
    Return (start_yyyymmdd, end_yyyymmdd) for every series in this module.

    The later of REGIME_START and today - MAX_LOOKBACK_YEARS wins, so the
    window is regime-safe now and self-limiting once the regime is old enough.
    """
    today = today or date.today()
    regime = datetime.strptime(REGIME_START, "%Y-%m-%d").date()
    rolling = today - timedelta(days=365 * MAX_LOOKBACK_YEARS)
    start = max(regime, rolling)
    return start.strftime("%Y%m%d"), today.strftime("%Y%m%d")


# ── Portfolio loading ─────────────────────────────────────────────────────────

def get_user_portfolios(user_id) -> list[dict]:
    """Return [{id, fund_name, benchmark}] for a user, newest first. [] on any failure."""
    if not user_id:
        return []
    try:
        df = db.read_table("funds", filters={"user_id": user_id})
        if df is None or df.empty:
            return []
        df = df.sort_values("id", ascending=False)
        return [
            {
                "id": int(r["id"]),
                "fund_name": str(r["fund_name"]),
                "benchmark": str(r.get("benchmark") or ""),
            }
            for _, r in df.iterrows()
        ]
    except Exception as exc:
        print(f"[portfolio_fit] get_user_portfolios: {exc}")
        return []


def get_portfolio_positions(fund_id: int) -> pd.DataFrame:
    """
    Current holdings of a mandate as DataFrame[ticker, ts_code, weight].

    Only live positions count — rows with an end_date have been closed out, and
    including them would analyse a book the user no longer owns. Weights are
    stored as 0-1 fractions; they are renormalised here so the arithmetic
    downstream is exact even if the saved mandate drifted off 100%.
    """
    try:
        df = db.read_table("fund_positions", filters={"fund_id": int(fund_id)})
        if df is None or df.empty:
            return pd.DataFrame(columns=["ticker", "ts_code", "weight"])

        if "end_date" in df.columns:
            df = df[df["end_date"].isna() | (df["end_date"].astype(str).str.strip() == "")]
        if df.empty:
            return pd.DataFrame(columns=["ticker", "ts_code", "weight"])

        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
        df = df[df["weight"] > 0].copy()
        if df.empty:
            return pd.DataFrame(columns=["ticker", "ts_code", "weight"])

        # A mandate can hold the same code across two effective_dates; keep the sum.
        df["ticker"] = df["ts_code"].astype(str).str.split(".").str[0]
        out = df.groupby(["ticker", "ts_code"], as_index=False)["weight"].sum()
        total = out["weight"].sum()
        if total > 0:
            out["weight"] = out["weight"] / total
        return out.sort_values("weight", ascending=False).reset_index(drop=True)
    except Exception as exc:
        print(f"[portfolio_fit] get_portfolio_positions({fund_id}): {exc}")
        return pd.DataFrame(columns=["ticker", "ts_code", "weight"])


# ── Price panel ───────────────────────────────────────────────────────────────

def load_price_panel(tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Fetch qfq closes for every ticker over the analysis window.

    Returns (panel, failed) where panel is DataFrame[date x ticker] of closes
    (union of dates, not yet aligned) and failed lists tickers with no data.
    """
    start, end = analysis_window()
    series: dict[str, pd.Series] = {}
    failed: list[str] = []

    for t in dict.fromkeys(str(x).strip() for x in tickers if str(x).strip()):
        try:
            df = data_manager.get_single_stock_data_live(t, start_date=start, end_date=end)
            if df is None or df.empty or "Close" not in df.columns:
                failed.append(t)
                continue
            s = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if s.empty:
                failed.append(t)
                continue
            s.index = pd.to_datetime(s.index)
            series[t] = s
        except Exception as exc:
            print(f"[portfolio_fit] price fetch {t}: {exc}")
            failed.append(t)

    panel = pd.DataFrame(series).sort_index() if series else pd.DataFrame()
    return panel, failed


# ── Risk primitives ───────────────────────────────────────────────────────────

def _ann_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _hist_var(returns: pd.Series, confidence: float = VAR_CONFIDENCE) -> float:
    """Historical one-day VaR as a positive loss number."""
    if returns.empty:
        return float("nan")
    return float(-np.percentile(returns, (1 - confidence) * 100))


def _hist_cvar(returns: pd.Series, confidence: float = VAR_CONFIDENCE) -> float:
    """Mean loss in the tail beyond VaR, as a positive number."""
    if returns.empty:
        return float("nan")
    cutoff = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= cutoff]
    return float(-tail.mean()) if len(tail) else float("nan")


def _max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the compounded series, as a negative number."""
    if returns.empty:
        return float("nan")
    curve = (1 + returns).cumprod()
    return float((curve / curve.cummax() - 1).min())


def _effective_bets(weights: np.ndarray) -> float:
    """Inverse Herfindahl — how many equally-sized positions this book behaves like."""
    w = np.asarray(weights, dtype=float)
    denom = float((w ** 2).sum())
    return float(1.0 / denom) if denom > 0 else float("nan")


def _diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """Weighted average vol / portfolio vol. 1.0 means no diversification at all."""
    w = np.asarray(weights, dtype=float)
    port_vol = float(np.sqrt(w @ cov @ w))
    if port_vol <= 0:
        return float("nan")
    return float((w @ np.sqrt(np.diag(cov))) / port_vol)


def _herfindahl(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    return float((w ** 2).sum())


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or a.std(ddof=1) == 0 or b.std(ddof=1) == 0:
        return float("nan")
    return float(a.corr(b))


# ── Descriptive metadata ──────────────────────────────────────────────────────

def get_stock_meta(tickers: list[str]) -> pd.DataFrame:
    """
    DataFrame[ticker, name, industry, area, market] from stock_basic.

    Missing rows fall back to the ticker itself with '未分类' industry so the
    concentration maths never silently drops a holding.
    """
    rows = []
    for t in tickers:
        rec = {"ticker": t, "name": t, "industry": "未分类", "area": "", "market": ""}
        try:
            ts_code = data_manager.get_tushare_ticker(t)
            df = db.read_table(
                "stock_basic",
                filters={"ts_code": ts_code},
                columns="name,industry,area,market",
            )
            if df is not None and not df.empty:
                r = df.iloc[0]
                rec["name"] = str(r.get("name") or t)
                ind = r.get("industry")
                rec["industry"] = str(ind) if pd.notna(ind) and str(ind).strip() else "未分类"
                rec["area"] = str(r.get("area") or "")
                rec["market"] = str(r.get("market") or "")
        except Exception:
            pass
        rows.append(rec)
    return pd.DataFrame(rows)


def get_factor_snapshot(tickers: list[str], panel: pd.DataFrame) -> pd.DataFrame:
    """
    Style factors per ticker: size, value, liquidity from daily_basic, momentum
    from the price panel.

    Deliberately sourced from the bulk daily_basic read rather than per-ticker
    fundamentals calls — this runs for every holding, so a fan-out of N Tushare
    calls would make the free panel expensive. ROE/quality is therefore not
    included here.
    """
    try:
        db_df = data_manager.get_daily_basic_latest(list(tickers))
    except Exception as exc:
        print(f"[portfolio_fit] daily_basic: {exc}")
        db_df = pd.DataFrame()

    rows = []
    for t in tickers:
        rec = {"ticker": t, "circ_mv_yi": np.nan, "pe_ttm": np.nan,
               "pb": np.nan, "turnover_rate": np.nan, "momentum_12_1": np.nan}
        if not db_df.empty and "ticker" in db_df.columns:
            hit = db_df[db_df["ticker"] == t]
            if not hit.empty:
                r = hit.iloc[0]
                rec["circ_mv_yi"] = pd.to_numeric(r.get("circ_mv_yi"), errors="coerce")
                rec["pe_ttm"] = pd.to_numeric(r.get("pe_ttm"), errors="coerce")
                rec["pb"] = pd.to_numeric(r.get("pb"), errors="coerce")
                rec["turnover_rate"] = pd.to_numeric(r.get("turnover_rate"), errors="coerce")

        # 12-1 momentum: skip the most recent month to avoid short-term reversal.
        if t in panel.columns:
            s = panel[t].dropna()
            if len(s) > 40:
                skip = min(21, len(s) - 2)
                lookback = min(TRADING_DAYS, len(s) - 1)
                start_px, end_px = s.iloc[-lookback], s.iloc[-1 - skip]
                if start_px > 0:
                    rec["momentum_12_1"] = float(end_px / start_px - 1)
        rows.append(rec)

    df = pd.DataFrame(rows)
    # Earnings yield / book yield invert the ratios so "high = cheap" everywhere.
    df["earnings_yield"] = np.where(df["pe_ttm"] > 0, 1.0 / df["pe_ttm"], np.nan)
    df["book_yield"] = np.where(df["pb"] > 0, 1.0 / df["pb"], np.nan)
    df["log_size"] = np.where(df["circ_mv_yi"] > 0, np.log(df["circ_mv_yi"]), np.nan)
    return df


def _factor_tilts(factors: pd.DataFrame, candidate: str,
                  holdings: list[str], weights: np.ndarray) -> list[dict]:
    """
    How far the candidate sits from the portfolio's weighted average on each
    style factor, expressed in cross-sectional standard deviations of the
    existing book.
    """
    labels = {
        "log_size": "规模 Size (log 流通市值)",
        "earnings_yield": "价值 Value (1/PE)",
        "book_yield": "价值 Value (1/PB)",
        "turnover_rate": "流动性 Liquidity (换手率)",
        "momentum_12_1": "动量 Momentum (12-1)",
    }
    idx = factors.set_index("ticker")
    out = []
    for col, label in labels.items():
        if col not in idx.columns:
            continue
        try:
            cand_val = float(idx.at[candidate, col])
        except Exception:
            continue
        held = idx.loc[[h for h in holdings if h in idx.index], col].astype(float)
        aligned_w = np.array([w for h, w in zip(holdings, weights) if h in idx.index])
        mask = held.notna().values
        if mask.sum() < 2 or not np.isfinite(cand_val):
            continue
        vals, ws = held.values[mask], aligned_w[mask]
        if ws.sum() <= 0:
            continue
        port_val = float(np.average(vals, weights=ws))
        sd = float(np.std(vals, ddof=1))
        if sd <= 0:
            continue  # book has no spread on this factor — a z-score is meaningless
        z = float((cand_val - port_val) / sd)
        if not np.isfinite(z):
            continue
        out.append({
            "factor": label,
            "candidate": cand_val,
            "portfolio": port_val,
            "z": z,
        })
    return out


# ── Sector affinity ───────────────────────────────────────────────────────────

def sector_affinity_vectors(returns: pd.DataFrame) -> pd.DataFrame | None:
    """
    Correlation of each column in `returns` to every PPI sector index.

    This is the cross-check on the industry label: two stocks tagged to
    different industries but driven by the same thing will still show near
    identical affinity vectors. Returns DataFrame[ticker x sector], or None if
    the PPI tables are not built.
    """
    sector_returns: dict[str, pd.Series] = {}
    try:
        sector_map = data_manager.get_sector_stock_map()
    except Exception:
        return None

    for sector_name in sector_map.keys():
        table = f"PPI_{sector_name}"
        try:
            if not db.table_exists(table):
                continue
            ppi = db.read_table(table, columns="Date,Close", order_by="Date")
            if ppi is None or ppi.empty:
                continue
            ppi["Date"] = pd.to_datetime(ppi["Date"])
            s = ppi.set_index("Date")["Close"].sort_index().pct_change().dropna()
            if len(s) > 10:
                sector_returns[sector_name] = s
        except Exception:
            continue

    if not sector_returns:
        return None

    rows = {}
    for ticker in returns.columns:
        vec = {}
        for sector, sec_ret in sector_returns.items():
            aligned = pd.concat(
                [returns[ticker].rename("s"), sec_ret.rename("x")], axis=1
            ).dropna()
            if len(aligned) >= 30:
                vec[sector] = _safe_corr(aligned["s"], aligned["x"])
        if vec:
            rows[ticker] = vec
    return pd.DataFrame(rows).T if rows else None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a, b = a[mask], b[mask]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyse_fit(candidate: str, new_weight: float, positions: pd.DataFrame) -> dict:
    """
    Full quantitative fit of `candidate` at `new_weight` (0-1) against the book
    described by `positions` (DataFrame[ticker, weight], weights summing to 1).

    Returns a dict with an 'ok' flag; on failure 'error' explains why. Every
    number is computed on the same regime-anchored aligned window, reported
    back in 'window'.
    """
    holdings = [t for t in positions["ticker"].tolist() if t != candidate]
    already_held = candidate in set(positions["ticker"])

    if not holdings:
        return {"ok": False, "error": "This mandate has no other holdings to compare against."}
    if not (0 < new_weight < 1):
        return {"ok": False, "error": "Proposed weight must be between 0% and 100%."}

    panel, failed = load_price_panel(holdings + [candidate])
    if panel.empty or candidate not in panel.columns:
        return {"ok": False, "error": f"No price history for the candidate {candidate} in this window."}

    usable = [t for t in holdings if t in panel.columns]
    if not usable:
        return {"ok": False, "error": "No price history for any holding in this window."}

    # Reweight over the holdings that actually have data, then align on the
    # intersection of trading days so every statistic uses the same sample.
    w0 = positions.set_index("ticker").loc[usable, "weight"].astype(float).values
    w0 = w0 / w0.sum()

    aligned = panel[usable + [candidate]].dropna()
    returns = aligned.pct_change().dropna()
    if len(returns) < 30:
        return {"ok": False, "error": f"Only {len(returns)} overlapping sessions — too few to analyse."}

    # The shortest history in the book sets the window for everyone.
    first_valid = {t: panel[t].dropna().index.min() for t in usable + [candidate]}
    binding = max(first_valid, key=lambda k: first_valid[k])

    port_ret = pd.Series(returns[usable].values @ w0, index=returns.index)
    cand_ret = returns[candidate]

    # --- Relationship to the existing book -----------------------------------
    corr = _safe_corr(cand_ret, port_ret)
    recent = returns.tail(RECENT_WINDOW)
    corr_recent = (
        _safe_corr(recent[candidate], pd.Series(recent[usable].values @ w0, index=recent.index))
        if len(recent) >= 20 else float("nan")
    )

    # Downside correlation is the number that matters: diversification that
    # only holds on up days is not diversification.
    down_mask = port_ret < 0
    down_corr = (
        _safe_corr(cand_ret[down_mask], port_ret[down_mask])
        if down_mask.sum() >= 20 else float("nan")
    )

    var_p = float(port_ret.var(ddof=1))
    beta = float(np.cov(cand_ret, port_ret, ddof=1)[0, 1] / var_p) if var_p > 0 else float("nan")
    residual_share = float(1 - corr ** 2) if np.isfinite(corr) else float("nan")

    # --- Before / after, under pro-rata dilution ------------------------------
    w_after = np.append(w0 * (1 - new_weight), new_weight)
    cols_after = usable + [candidate]
    ret_after = pd.Series(returns[cols_after].values @ w_after, index=returns.index)

    cov_before = returns[usable].cov().values
    cov_after = returns[cols_after].cov().values

    # Candidate's share of total portfolio variance vs its share of capital —
    # the headline "is this paying its way in risk terms" number.
    var_after = float(w_after @ cov_after @ w_after)
    mctr = float((cov_after @ w_after)[-1])
    risk_share = float(new_weight * mctr / var_after) if var_after > 0 else float("nan")

    before = {
        "vol": _ann_vol(port_ret),
        "var95": _hist_var(port_ret),
        "cvar95": _hist_cvar(port_ret),
        "max_dd": _max_drawdown(port_ret),
        "enb": _effective_bets(w0),
        "div_ratio": _diversification_ratio(w0, cov_before),
        "hhi": _herfindahl(w0),
    }
    after = {
        "vol": _ann_vol(ret_after),
        "var95": _hist_var(ret_after),
        "cvar95": _hist_cvar(ret_after),
        "max_dd": _max_drawdown(ret_after),
        "enb": _effective_bets(w_after),
        "div_ratio": _diversification_ratio(w_after, cov_after),
        "hhi": _herfindahl(w_after),
    }

    # --- Nearest economic twin ------------------------------------------------
    meta = get_stock_meta(usable + [candidate])
    meta_idx = meta.set_index("ticker")
    cand_industry = str(meta_idx.at[candidate, "industry"])

    pair_rows = []
    for i, t in enumerate(usable):
        c = _safe_corr(cand_ret, returns[t])
        same_ind = str(meta_idx.at[t, "industry"]) == cand_industry and cand_industry != "未分类"
        pair_rows.append({
            "ticker": t,
            "name": str(meta_idx.at[t, "name"]),
            "industry": str(meta_idx.at[t, "industry"]),
            "weight": float(w0[i]),
            "corr": c,
            "same_industry": bool(same_ind),
            # Same-industry pairs get a nudge so an industry peer outranks an
            # unrelated name at equal correlation.
            "twin_score": (c if np.isfinite(c) else 0.0) + (0.10 if same_ind else 0.0),
        })
    pairs = pd.DataFrame(pair_rows).sort_values("twin_score", ascending=False).reset_index(drop=True)
    max_pair_corr = float(pairs["corr"].max()) if not pairs.empty else float("nan")

    # --- Industry concentration ----------------------------------------------
    ind_before: dict[str, float] = {}
    for i, t in enumerate(usable):
        ind = str(meta_idx.at[t, "industry"])
        ind_before[ind] = ind_before.get(ind, 0.0) + float(w0[i])
    ind_after = {k: v * (1 - new_weight) for k, v in ind_before.items()}
    ind_after[cand_industry] = ind_after.get(cand_industry, 0.0) + new_weight

    cand_ind_before = ind_before.get(cand_industry, 0.0)
    cand_ind_after = ind_after.get(cand_industry, 0.0)

    # --- Sector affinity overlap ---------------------------------------------
    sector_cos = float("nan")
    affinity = sector_affinity_vectors(returns[cols_after])
    if affinity is not None and candidate in affinity.index:
        held_rows = affinity.loc[[t for t in usable if t in affinity.index]]
        if not held_rows.empty:
            aligned_w = np.array([w for t, w in zip(usable, w0) if t in affinity.index])
            port_vec = np.average(held_rows.values, axis=0, weights=aligned_w)
            sector_cos = _cosine(affinity.loc[candidate].values.astype(float), port_vec)

    # --- Factor tilts ---------------------------------------------------------
    factors = get_factor_snapshot(usable + [candidate], panel)
    tilts = _factor_tilts(factors, candidate, usable, w0)

    metrics = {
        "corr": corr,
        "corr_recent": corr_recent,
        "down_corr": down_corr,
        "beta": beta,
        "residual_share": residual_share,
        "risk_share": risk_share,
        "weight_share": new_weight,
        "max_pair_corr": max_pair_corr,
        "sector_cosine": sector_cos,
        "cand_industry": cand_industry,
        "cand_industry_before": cand_ind_before,
        "cand_industry_after": cand_ind_after,
    }
    verdict = classify_fit(metrics)

    return {
        "ok": True,
        "candidate": candidate,
        "candidate_name": str(meta_idx.at[candidate, "name"]),
        "already_held": already_held,
        "new_weight": new_weight,
        "metrics": metrics,
        "before": before,
        "after": after,
        "pairs": pairs,
        "industry_before": ind_before,
        "industry_after": ind_after,
        "factor_tilts": tilts,
        "meta": meta,
        "window": {
            "start": returns.index.min().strftime("%Y-%m-%d"),
            "end": returns.index.max().strftime("%Y-%m-%d"),
            "sessions": int(len(returns)),
            "low_confidence": len(returns) < MIN_OVERLAP_DAYS,
            "binding_ticker": binding,
            "binding_start": first_valid[binding].strftime("%Y-%m-%d"),
        },
        "holdings_used": usable,
        "holdings_failed": failed,
        "verdict": verdict,
    }


# ── Verdict ───────────────────────────────────────────────────────────────────

VERDICT_STYLES = {
    "twin": ("🔴 经济双胞胎 Economic Twin", "error"),
    "concentration": ("🟠 主动加仓集中度 Deliberate Concentration", "warning"),
    "partial": ("🟡 部分重叠 Partial Overlap", "warning"),
    "diversifier": ("🟢 分散化 Diversifier", "success"),
}


def classify_fit(m: dict) -> dict:
    """
    Deterministic bucket + the reasons that put it there.

    The label is rule-based on purpose: the same inputs must always give the
    same call, so the AI narrative explains a verdict rather than inventing it.
    Order matters — twin beats concentration beats partial.
    """
    reasons: list[str] = []

    def num(key, default=0.0):
        v = m.get(key, default)
        return v if isinstance(v, (int, float)) and np.isfinite(v) else default

    corr = num("corr")
    down_corr = num("down_corr")
    resid = num("residual_share", 1.0)
    pair = num("max_pair_corr")
    cos = num("sector_cosine")
    ind_after = num("cand_industry_after")
    ind_before = num("cand_industry_before")
    risk_ratio = (num("risk_share") / m["weight_share"]) if m.get("weight_share") else 0.0

    # 1. Economic twin — it moves like something already owned, and offers
    #    little of its own.
    if pair >= 0.80:
        reasons.append(f"Correlation to a single existing holding is {pair:.2f}.")
    if (down_corr >= 0.70 or corr >= 0.75) and resid < 0.40:
        reasons.append(
            f"Correlation to the book is {corr:.2f} "
            f"(downside {down_corr:.2f}) with only {resid:.0%} idiosyncratic variance."
        )
    if reasons:
        return {"bucket": "twin", "reasons": reasons}

    # 2. Deliberate concentration — not a clone, but it doubles down on an
    #    industry the book already leans on.
    if ind_after >= 0.30 and ind_before >= 0.15:
        reasons.append(
            f"Lifts {m.get('cand_industry')} from {ind_before:.0%} to {ind_after:.0%} of the book."
        )
        return {"bucket": "concentration", "reasons": reasons}

    # 3. Partial overlap.
    if corr >= 0.45:
        reasons.append(f"Correlation to the book is {corr:.2f}.")
    if risk_ratio >= 1.25:
        reasons.append(
            f"Takes {num('risk_share'):.0%} of portfolio risk for "
            f"{m['weight_share']:.0%} of capital."
        )
    if cos >= 0.80:
        reasons.append(f"Sector-driver profile matches the book (cosine {cos:.2f}).")
    if ind_before >= 0.15:
        reasons.append(f"{m.get('cand_industry')} is already {ind_before:.0%} of the book.")
    if reasons:
        return {"bucket": "partial", "reasons": reasons}

    # 4. Diversifier.
    return {
        "bucket": "diversifier",
        "reasons": [
            f"Correlation to the book is {corr:.2f} (downside {down_corr:.2f}) "
            f"with {resid:.0%} idiosyncratic variance.",
            f"Takes {num('risk_share'):.0%} of risk for {m['weight_share']:.0%} of capital.",
        ],
    }


# ── Candidate revenue mix (the one hard fact behind the AI section) ───────────

def get_revenue_mix(ticker: str) -> dict:
    """
    Latest reported revenue split for the candidate, by product and by
    region/segment, from Tushare fina_mainbz.

    This is fetched for the CANDIDATE ONLY — one call, not a fan-out across the
    whole book — and is passed to the model as ground truth so the qualitative
    read is anchored on reported numbers rather than recollection.
    """
    out = {"period": None, "by_product": [], "by_region": []}
    for bz_type, key in (("P", "by_product"), ("D", "by_region")):
        try:
            df = data_manager.fetch_fina_mainbz(ticker, bz_type)
            if df is None or df.empty:
                continue
            latest = df["end_date"].iloc[0]
            recent = df[df["end_date"] == latest].copy()
            total = float(recent["bz_sales"].sum())
            if total <= 0:
                continue
            recent["share"] = recent["bz_sales"] / total
            recent = recent.sort_values("share", ascending=False).head(8)
            out["period"] = out["period"] or str(latest)
            out[key] = [
                {"item": str(r["bz_item"]).strip(), "share": round(float(r["share"]), 4)}
                for _, r in recent.iterrows()
            ]
        except Exception as exc:
            print(f"[portfolio_fit] fina_mainbz {ticker}/{bz_type}: {exc}")
    return out


# ── Qualitative layer (single DeepSeek call) ──────────────────────────────────

_OVERLAP_PROMPT = """\
You are a senior Chinese A-share portfolio risk analyst.

You are given a CANDIDATE stock, the CURRENT HOLDINGS of a portfolio, and
quantitative statistics already computed from price data. Your job is the
qualitative half only: explain the BUSINESS overlap that the correlation
numbers cannot see.

Assess:
1. Shared themes — policy cycles, end-demand cycles, or narratives that drive
   the candidate and specific holdings together.
2. Supply-chain adjacency — is the candidate an upstream supplier or a
   downstream customer of any holding, or at the same layer of the same chain?
   Same chain at a different layer still means one demand shock hits both.
3. Revenue-mix overlap — shared products, shared end-markets, shared export
   exposure.
4. Shared risk drivers — a single input cost, customer, regulator, or macro
   variable that would hit several names at once.

RULES:
- Only name holdings that were actually given to you. Never invent tickers.
- If you are unsure about a company, say so and set confidence to "low" rather
  than guessing. A wrong supply-chain claim is worse than an absent one.
- Explain the quantitative numbers you were given; do not contradict them, and
  do not compute your own.
- Prose fields must be written in Chinese. Ticker and item names stay as given.
- Return ONLY raw JSON (start { end }). No markdown fences.

Schema:
{
  "shared_themes": [
    {"theme": "...", "holdings": ["600xxx"], "strength": "high|medium|low", "why": "..."}
  ],
  "supply_chain_links": [
    {"holding": "600xxx", "relation": "upstream_supplier|downstream_customer|same_layer|none", "why": "..."}
  ],
  "revenue_overlap": [
    {"holding": "600xxx", "overlap": "high|medium|low", "shared_end_markets": ["..."], "why": "..."}
  ],
  "shared_risk_drivers": [
    {"driver": "...", "affected": ["600xxx"], "why": "..."}
  ],
  "single_event_risk": "One concrete event that would hurt the candidate and the book together.",
  "verdict_comment": "2-4 sentences on whether this diversifies, duplicates, or concentrates, in business terms.",
  "confidence": "high|medium|low"
}
"""


def analyse_business_overlap(result: dict, positions: pd.DataFrame,
                             revenue_mix: dict | None = None) -> dict:
    """
    One DeepSeek call covering themes, supply-chain adjacency and revenue-mix
    overlap. Raises RuntimeError on failure (caller shows the message).

    Costs one API call per invocation, which is why the UI puts it behind its
    own button rather than running it with the free quantitative panel.
    """
    m = result["metrics"]
    meta = result["meta"].set_index("ticker")
    wmap = positions.set_index("ticker")["weight"].to_dict()

    holding_lines = []
    for t in result["holdings_used"]:
        holding_lines.append(
            f"- {t} {meta.at[t, 'name']} | 行业 {meta.at[t, 'industry']} "
            f"| 权重 {wmap.get(t, 0):.1%}"
        )

    mix_text = ""
    if revenue_mix:
        if revenue_mix.get("by_product"):
            items = "; ".join(
                f"{d['item']} {d['share']:.0%}" for d in revenue_mix["by_product"]
            )
            mix_text += f"\n按产品收入构成 ({revenue_mix.get('period')}): {items}"
        if revenue_mix.get("by_region"):
            items = "; ".join(
                f"{d['item']} {d['share']:.0%}" for d in revenue_mix["by_region"]
            )
            mix_text += f"\n按地区/分部收入构成: {items}"

    top_pairs = result["pairs"].head(5)
    pair_text = "\n".join(
        f"- {r['ticker']} {r['name']} ({r['industry']}): 相关系数 {r['corr']:.2f}"
        for _, r in top_pairs.iterrows()
    )

    user_msg = (
        f"CANDIDATE: {result['candidate']} {result['candidate_name']} "
        f"| 行业 {m['cand_industry']} | 拟配置权重 {result['new_weight']:.1%}"
        f"{mix_text}\n\n"
        f"CURRENT HOLDINGS ({len(holding_lines)}):\n" + "\n".join(holding_lines) + "\n\n"
        f"QUANTITATIVE RESULTS (window {result['window']['start']} to "
        f"{result['window']['end']}, {result['window']['sessions']} sessions):\n"
        f"- 与组合相关性 {m['corr']:.2f} (近{RECENT_WINDOW}日 {m['corr_recent']:.2f}, "
        f"下跌日 {m['down_corr']:.2f})\n"
        f"- Beta {m['beta']:.2f}, 特质方差占比 {m['residual_share']:.0%}\n"
        f"- 占组合风险 {m['risk_share']:.0%} vs 占资金 {m['weight_share']:.0%}\n"
        f"- {m['cand_industry']} 行业权重 {m['cand_industry_before']:.0%} → "
        f"{m['cand_industry_after']:.0%}\n"
        f"- 规则判定: {result['verdict']['bucket']}\n\n"
        f"MOST CORRELATED HOLDINGS:\n{pair_text}"
    )

    return ai_client.call_json(
        _OVERLAP_PROMPT, user_msg,
        max_tokens=3500,
        temperature=0.2,
    )
