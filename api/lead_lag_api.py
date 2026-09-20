"""
api/lead_lag_api.py — 领先滞后, behind HTTP.

The statistics come from lead_lag_stats.py unchanged, so this screen and the
Streamlit page answer with the same numbers. Three questions per pair, and
they are deliberately different questions:

    Granger causality   does one stock's PAST help predict the other's
                        future, beyond the other's own past? Direction.
    Cross-correlation   at which lag is the co-movement strongest, and how
                        strong? Shape — and the sanity check on the above.
    Cointegration + OU  do the two prices actually travel together, and how
                        long does a gap take to close? Exploitability.

A pair can pass the first and fail the third, and that combination is the
common one: a statistically real lead of half a percent that closes over
eleven days is not a trade after costs.

What this module adds
---------------------
Multiple testing. The Streamlit page runs Granger in both directions for
every peer and calls anything under p < 0.05 a relationship. With ten peers
that is twenty tests, and under the null a p-value is uniform — so twenty of
them contain one pass at 5% ON AVERAGE, with two or three not unusual. The
original
guards against this with an economic filter (|peak correlation| > 0.15), which
helps but is not a correction.

So every p-value in a run goes through Benjamini-Hochberg together, and each
row carries the resulting q-value and whether it survives. The original labels
are left exactly as they were — changing what 🔥 Strong means would silently
alter a screen the user already reads — and the correction is reported
alongside, including how many passes to expect from noise.

A-shares only: fetch_qfq_returns goes through Tushare.
"""

from __future__ import annotations

import math

import pandas as pd

#: Tests below this q-value are worth looking at. Same number as the original
#: alpha, but applied to a FALSE DISCOVERY RATE — "at most 5% of the
#: relationships shown here are noise" rather than "each test had a 5% chance".
FDR_Q = 0.05
ALPHA = 0.05

MIN_PEERS, MAX_PEERS = 1, 15
LOOKBACK_CHOICES = (90, 180, 252, 504)


def _n(v, nd=4):
    """A JSON-safe float. NaN and inf are 'no answer', not numbers."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else round(f, nd)


def lag_labels(max_lag: int) -> list[str]:
    """
    Column headings for the lag grid, written from T's point of view.

    Signed lags are unreadable in a table — every reader has to re-derive
    which way round a negative one goes. "S 领先 3 天" cannot be misread.
    """
    out = []
    for k in range(-max_lag, max_lag + 1):
        if k < 0:
            out.append(f"S 领先 {-k} 天")
        elif k == 0:
            out.append("同日")
        else:
            out.append(f"T 领先 {k} 天")
    return out


def analyse(ticker: str, peers: list[str], *, lookback_days: int = 180,
            max_lag: int = 5) -> dict:
    """
    Lead-lag for one stock against a chosen set of peers.

    `peers` is whatever the caller picked — a watchlist, a sector, or names
    typed in by hand. No discovery step: which stocks are worth testing is a
    judgement, and an AI guessing at it produces a peer list nobody can
    defend and twenty more tests to correct for.
    """
    import lead_lag_stats as lls

    peers = [p for p in dict.fromkeys(peers) if p and p != ticker]
    if not peers:
        raise LookupError("请至少选择一只对比股票")
    if len(peers) > MAX_PEERS:
        raise LookupError(f"一次最多比较 {MAX_PEERS} 只")

    every = [ticker, *peers]
    returns_df, prices_df = lls.fetch_qfq_returns(every, lookback_days=lookback_days)
    if returns_df.empty or ticker not in returns_df.columns:
        raise RuntimeError(f"{ticker} 的行情暂时读取不到 — 请稍后重试")

    names = _names(every)
    peer_data = [{"ticker": p, "name": names.get(p, p), "layer_name": "", "layer_idx": 0}
                 for p in peers if p in returns_df.columns]
    missing = [p for p in peers if p not in returns_df.columns]
    if not peer_data:
        raise LookupError("所选股票都没有可用的重叠行情")

    df = lls.compute_lead_lag(ticker, peer_data, returns_df, prices_df, max_lag=max_lag)
    if df.empty:
        raise LookupError("重叠的交易日不足 30 天，无法做统计检验")

    rows = _rows(df, max_lag)
    q_by_index, tests = _fdr(rows)
    for row in rows:
        row.update(q_by_index[row["ticker"]])

    rows.sort(key=lambda r: (not r["survives_fdr"],
                             r["q_best"] if r["q_best"] is not None else 1.0))
    return {
        "ticker": ticker,
        "name": names.get(ticker, ticker),
        "lookback_days": lookback_days,
        "max_lag": max_lag,
        "sessions": int(len(returns_df)),
        "from": returns_df.index[0].strftime("%Y-%m-%d"),
        "to": returns_df.index[-1].strftime("%Y-%m-%d"),
        "lags": list(range(-max_lag, max_lag + 1)),
        "lag_labels": lag_labels(max_lag),
        "rows": rows,
        "tests": tests,
        "missing": [{"ticker": m, "name": names.get(m, m)} for m in missing],
    }


def _names(tickers: list[str]) -> dict:
    import data_manager
    out = {}
    for t in tickers:
        try:
            out[t] = data_manager.get_stock_name_from_db(t) or t
        except Exception:                                          # noqa: BLE001
            out[t] = t
    return out


def _rows(df: pd.DataFrame, max_lag: int) -> list[dict]:
    lags = list(range(-max_lag, max_lag + 1))
    rows = []
    for _, r in df.iterrows():
        xc = r.get("_xcorrs") or {}
        rows.append({
            "ticker": str(r["ticker"]),
            "name": str(r["name"]),
            "n_obs": int(r["n_obs"]),
            "beta": _n(r["beta"], 3),
            "relationship": str(r["relationship"]),
            "signal": str(r["signal"]),
            "peak_corr": _n(r["peak_corr"], 3),
            "peak_lag": int(r["peak_lag"]),
            "p_t_leads_s": _n(r["p_T_leads_S"], 5),
            "lag_t_leads_s": int(r["lag_T_leads_S"] or 0),
            "p_s_leads_t": _n(r["p_S_leads_T"], 5),
            "lag_s_leads_t": int(r["lag_S_leads_T"] or 0),
            "cointegrated": bool(r["cointegrated"]),
            "half_life": _n(r["half_life"], 1),
            "xcorr": [_n(xc.get(k), 3) for k in lags],
        })
    return rows


def _fdr(rows: list[dict]) -> tuple[dict, dict]:
    """
    Benjamini-Hochberg across every Granger test in the run.

    Both directions for every peer go into one family, because they were all
    run in one sweep looking for whichever came out significant — which is
    exactly the situation the correction exists for. Reported per row as a
    q-value: the share of results at least this extreme that would be false.
    """
    flat = [(row["ticker"], key, row[key])
            for row in rows for key in ("p_t_leads_s", "p_s_leads_t")
            if row[key] is not None]
    n = len(flat)
    raw_hits = sum(1 for _, _, p in flat if p < ALPHA)

    per_row = {row["ticker"]: {"q_t_leads_s": None, "q_s_leads_t": None,
                               "q_best": None, "survives_fdr": False}
               for row in rows}
    if not n:
        return per_row, {"n": 0, "alpha": ALPHA, "q": FDR_Q, "raw_hits": 0,
                         "expected_false": 0.0, "survivors": 0, "method": "BH"}

    qs = _bh([p for _, _, p in flat])
    for (tk, key, _), q in zip(flat, qs):
        per_row[tk]["q_" + key[2:]] = round(q, 5)
    for tk, cell in per_row.items():
        got = [v for v in (cell["q_t_leads_s"], cell["q_s_leads_t"]) if v is not None]
        cell["q_best"] = min(got) if got else None
        cell["survives_fdr"] = bool(got and min(got) < FDR_Q)

    return per_row, {
        "n": n,
        "alpha": ALPHA,
        "q": FDR_Q,
        "raw_hits": raw_hits,
        # The number to hold the raw count against: n × alpha is how many
        # passes to expect from data with no structure in it at all.
        "expected_false": round(n * ALPHA, 1),
        "survivors": sum(1 for c in per_row.values() if c["survives_fdr"]),
        "method": "Benjamini-Hochberg",
    }


def _bh(pvals: list[float]) -> list[float]:
    """
    Benjamini-Hochberg q-values, monotone-adjusted.

    Written out rather than imported so the module has no hard dependency on
    statsmodels' multitest — lead_lag_stats already degrades when statsmodels
    is absent, and the correction should not be the thing that breaks first.
    """
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    prev = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        prev = min(prev, pvals[idx] * n / rank)
        q[idx] = min(prev, 1.0)
    return q


# ── discovery ────────────────────────────────────────────────────────────────
#: Two years, so each half of the split is a year. Cointegration over six
#: months is a coin toss dressed as a test.
DISCOVER_DAYS = 504


def sector_of() -> dict:
    """{ticker: sector} from the sector map, first membership wins."""
    import data_manager
    out = {}
    try:
        for sector, members in (data_manager.get_sector_stock_map() or {}).items():
            for m in members or []:
                out.setdefault(str(m).split(".")[0], sector)
    except Exception:                                              # noqa: BLE001
        return {}
    return out


def discover(tickers: list[str], kind: str = "pair-trade", *,
             lookback_days: int = DISCOVER_DAYS, min_corr: float | None = None,
             within_sector: bool = False, target: str | None = None) -> dict:
    """
    Search a whole list for pairs, instead of being told which to test.

    The search is a funnel with an out-of-sample confirmation at the end —
    see pair_scan for why that shape and not a straight sweep. This function
    only fetches, names and annotates; the statistics and the honesty are
    over there.
    """
    import lead_lag_stats as lls
    import pair_scan

    codes = sorted({t for t in tickers if t and t.isdigit() and len(t) == 6})
    if len(codes) < 2:
        raise LookupError("至少需要两只 A 股才能配对搜索")

    _, prices = lls.fetch_qfq_returns(codes, lookback_days=lookback_days)
    if prices.empty:
        raise RuntimeError("行情暂时读取不到 — 请稍后重试")

    if target and target not in codes:
        raise LookupError(f"{target} 不在自选股里 — 先把它加进自选股")

    # A target replaces the shortlist, so the sector restriction and the
    # correlation floor have nothing left to do: every peer is tested either
    # way, and saying otherwise on screen would be a lie about the funnel.
    groups = sector_of() if (within_sector and not target) else None
    out = pair_scan.scan(prices, kind,
                         min_corr=pair_scan.MIN_CORR if min_corr is None else min_corr,
                         within=groups, target=target or None)

    names = _names(sorted({c for r in out["rows"] for c in (r["a"], r["b"])}))
    sectors = groups if groups is not None else sector_of()
    for row in out["rows"]:
        row["name_a"] = names.get(row["a"], row["a"])
        row["name_b"] = names.get(row["b"], row["b"])
        row["sector_a"] = sectors.get(row["a"], "")
        row["sector_b"] = sectors.get(row["b"], "")
    out["requested"] = len(codes)
    out["within_sector"] = bool(within_sector and not target)
    out["lookback_days"] = lookback_days
    out["target"] = target or None
    if target:
        out["target_name"] = _names([target]).get(target, target)
    return out
