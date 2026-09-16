"""
watchlist_scan.py — the per-stock Today's-Alerts scan, with no Streamlit in it.

Why this lives outside the page
-------------------------------
The same scan now runs in two places: interactively on the Today's Alerts page,
and nightly in GitHub Actions to pre-fill the cache. If each carried its own
copy of the signal rules they would drift, and a cached row would stop meaning
what the chart marker means — the invariant the page's comments exist to
protect. So the rules live here once and both callers import them.

Nothing in this module may touch st.* or a logged-in session: the nightly job
has neither.

Why the scan is slow, and why that is not fixed here
----------------------------------------------------
Measured: ~7s per stock end to end on a fast desktop, the bulk of it in
run_single_stock_analysis (fetch ~1.2s, box sweep ~1.1s). A profiler reported
~15s for the analysis alone, but cProfile inflates code made of many small
calls; the 7s is wall-clock. The cost is the walk-forward Gaussian HMM regime
detector — it
refits a 3-state model every 5 bars on a trailing 90-bar window (~106 fits,
each KMeans-initialised) and Viterbi-decodes every bar (~528 decodes). It is
slow ON PURPOSE: refitting only on past data is what keeps it free of
lookahead. And its output selects the MACD gear, so it feeds MACD, crossovers,
bottoming and the reversal signals. Speeding it up by changing its numerics
would silently change the alerts. The fix is to run it somewhere that is
allowed to be slow, not to make it less correct.
"""

from __future__ import annotations

import pandas as pd

# ── Signal criteria ──────────────────────────────────────────────────────────
# These mirror EXACTLY the discrete markers drawn on the Technical Analysis
# chart, so an alert means the same thing as the chart marker. The ADX-based
# buy/sell signals come via Entry_Candidate / Exit_Candidate, which the engine
# direction-gates by +DI vs −DI (same gating as the chart) — NOT the old
# 5-day-EMA price-trend gate, which produced contradictory calls.

BULL_BOOL = {
    'Squeeze_Fired_Bullish':  'Bullish Squeeze Breakout 🚀',
    'Signal_Accumulation':    'Phase 1: Accumulation',
    'MACD_Bottoming':         'MACD Bottoming',
    'MACD_ClassicCrossover':  'MACD Bullish Crossover',
    'RSI_Bottoming':          'RSI Bottoming',
    'Downtrend_Reversal':     'Downtrend Reversal 🔄',
}
BULL_ENTRY = {  # Entry_Candidate value → label
    'Strength Returning':     'Strength Returning (ADX)',
    'Trend Accelerating':     'Trend Accelerating (ADX)',
    'Screaming Buy':          'DI Screaming Buy 🚀',
}
BEAR_BOOL = {
    'Squeeze_Fired_Bearish':  'Bearish Squeeze Drop 🩸',
    'Exit_MACD_Lead':         'MACD Exit Signal',
    'MACD_Peaking':           'MACD Peaking',
    'MACD_BearishCrossover':  'MACD Bearish Crossover',
    'RSI_Peaking':            'RSI Peaking',
    'Uptrend_Reversal':       'Uptrend Reversal 🔄',
}
BEAR_EXIT = {  # Exit_Candidate value → label
    'Trend Topping':          'Trend Topping (ADX)',
    'Trend Collapsing':       'Trend Collapsing (ADX)',
    'Screaming Sell':         'DI Screaming Sell 🛑',
}

BIAS_RANK = {'🚀 Bullish': 0, '⚖️ Mixed': 1, '⚠️ Bearish': 2}

# MUST match the Technical Analysis page's window (lookback_years=3). The
# squeeze uses a rolling BB-width percentile whose window is ADAPTIVE to data
# length: ~243 bars → 120-day window, ≥250 bars → 250-day window. Feeding
# different history made the same bar's squeeze fire here but not on the chart.
LOOKBACK_YEARS = 3
MIN_BARS = 100
FETCH_ATTEMPTS = 3
FETCH_BACKOFF_S = 2.0


def extract_signals(latest) -> tuple[list[str], list[str]]:
    """(bullish_labels, bearish_labels) for one stock's latest bar."""
    bull, bear = [], []
    for col, label in BULL_BOOL.items():
        if col in latest.index and bool(latest[col]):
            bull.append(label)
    for col, label in BEAR_BOOL.items():
        if col in latest.index and bool(latest[col]):
            bear.append(label)
    ec = str(latest.get('Entry_Candidate', '') or '')
    if ec in BULL_ENTRY:
        bull.append(BULL_ENTRY[ec])
    xc = str(latest.get('Exit_Candidate', '') or '')
    if xc in BEAR_EXIT:
        bear.append(BEAR_EXIT[xc])
    return bull, bear


def box_signal(analysis_df):
    """
    (marker, label, box) for a stock sitting at the edge of a live 箱体.

    Encoded into the Signals string rather than a column, because the cache
    stores a fixed 10-column schema. The bracketed form is machine-readable so
    the page's 箱体 section can pull the numbers back out of a cached row.
    """
    import box_detection as bxd
    try:
        info = bxd.box_alert(analysis_df)
    except Exception:
        return None, None, None
    if info is None:
        return None, None, None
    b = info["box"]
    tag = (f"[{b['bot']:.2f}-{b['top']:.2f}] 位置{info['position']*100:.0f}%"
           f" 触{b['touches_top']}/{b['touches_bot']} 质{b['quality']:.2f}")
    st_ = info["status"]
    if st_ == "AT_SUPPORT":
        return "bull", f"箱体下沿 {tag}", b
    if st_ == "AT_RESISTANCE":
        return "bear", f"箱体上沿 {tag}", b
    if st_ == "BREAKOUT":
        return "bull", f"箱体突破 {tag}", b
    if st_ == "BREAKDOWN":
        return "bear", f"箱体跌破 {tag}", b
    return None, None, b


def fetch_frame(ticker: str):
    """
    Price history for one stock, retried. None means the CALL failed.

    None and "too short" are different failures and must not share a status.
    ts.pro_bar returns None when the call fails — a timeout, a rate-limit
    hiccup — while a genuinely young stock returns a short frame. Folding None
    into no_data made a transient network blip look like a legitimate "not
    enough history", so a stock could vanish from a nightly snapshot with
    nothing to show that anything went wrong.
    """
    import time
    import data_manager
    for attempt in range(FETCH_ATTEMPTS):
        stock_df = data_manager.get_single_stock_data_live(
            ticker, lookback_years=LOOKBACK_YEARS)
        if stock_df is not None:
            return stock_df
        time.sleep(FETCH_BACKOFF_S * (attempt + 1))
    return None


def scan_ticker(ticker: str, stock_df=None) -> dict:
    """
    Scan one stock. Always returns a dict with a `status`:

      'ok'       — row is in `row`, signals fired
      'quiet'    — analysed fine, nothing fired (not a table row)
      'no_data'  — too little history
      'error'    — `error` says why

    Returning the reason instead of None matters in the nightly job, where
    nobody is watching: "80 scanned, 12 rows" is ambiguous between a quiet
    market and a broken fetch unless the non-rows say which they are.

    Pass `stock_df` to reuse a frame already fetched (the nightly job hands the
    same frame to the chip scan); leave it None to fetch here.
    """
    import data_manager
    from analysis_engine import run_single_stock_analysis

    try:
        if stock_df is None:
            stock_df = fetch_frame(ticker)
        if stock_df is None:
            return {"ticker": ticker, "status": "error",
                    "error": f"fetch returned nothing after {FETCH_ATTEMPTS} attempts"}
        if len(stock_df) < MIN_BARS:
            return {"ticker": ticker, "status": "no_data"}

        analysis_df = run_single_stock_analysis(stock_df)
        if analysis_df is None or analysis_df.empty:
            return {"ticker": ticker, "status": "no_data"}

        latest = analysis_df.iloc[-1]
        data_date = str(analysis_df.index[-1].date())
        bull, bear = extract_signals(latest)

        # 箱体 edge test — checked BEFORE the empty-signal skip, because
        # "sitting on a support that has held four times" is the whole alert
        # even on a day when nothing else fires.
        _bmark, _blabel, _box = box_signal(analysis_df)
        if _bmark == "bull":
            bull.append(_blabel)
        elif _bmark == "bear":
            bear.append(_blabel)

        if not bull and not bear:
            return {"ticker": ticker, "status": "quiet", "data_date": data_date}

        if bull and not bear:
            bias = '🚀 Bullish'
        elif bear and not bull:
            bias = '⚠️ Bearish'
        else:
            bias = '⚖️ Mixed'

        parts = [f"▲ {s}" for s in bull] + [f"▼ {s}" for s in bear]
        name = data_manager.get_stock_name_from_db(ticker) or ticker
        return {
            "ticker": ticker, "status": "ok", "data_date": data_date,
            "row": {
                'Type':         bias,
                'Ticker':       ticker,
                'Name':         name,
                'Signals':      "  ·  ".join(parts),
                'Signal_Count': len(bull) + len(bear),
                'Price':        float(latest.get('Close', 0)),
                'RSI':          float(latest.get('RSI_14', 0)),
                'ADX':          float(latest.get('ADX', 0)),
                'MACD':         float(latest.get('MACD', 0)),
                'Volume':       float(latest.get('Volume', 0)),
            },
        }
    except Exception as e:
        return {"ticker": ticker, "status": "error", "error": str(e)[:300]}


# ── 筹码结构 ─────────────────────────────────────────────────────────────────
# Every decay the page's slider offers. The model is ~90ms per decay; the two
# API calls behind it are the cost, and they are shared, so the nightly job
# computes all three and the slider never has to recompute.
CHIP_DECAYS = (0.8, 1.0, 1.2)
CHIP_MIN_BARS = 120


def _finite(x, nd=None):
    """JSON (PostgREST) rejects NaN — store unknowns as NULL."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return round(x, nd) if nd is not None else x


def scan_chips(ticker: str, stock_df=None, decays=CHIP_DECAYS) -> dict:
    """
    Chip structure for one stock, one record per decay. Same status contract
    as scan_ticker: 'ok' (records in `rows`), 'no_data', or 'error'.

    Records use the chip_scan table's column names, so the nightly job writes
    them as they are and the page renames for display.
    """
    import time
    import data_manager
    import chip_distribution as cdist

    try:
        if stock_df is None:
            stock_df = fetch_frame(ticker)
        if stock_df is None:
            return {"ticker": ticker, "status": "error",
                    "error": f"price fetch returned nothing after {FETCH_ATTEMPTS} attempts"}
        if len(stock_df) < CHIP_MIN_BARS:
            return {"ticker": ticker, "status": "no_data"}

        start = stock_df.index.min().strftime("%Y%m%d")
        end = stock_df.index.max().strftime("%Y%m%d")
        # get_stock_fundamentals_live returns None for a failed call AND for an
        # empty answer, so it cannot tell a blip from a stock with no
        # daily_basic. Retry, then call it an error: a stock with 120+ bars
        # that genuinely has no turnover rate is not a thing Tushare produces.
        fund = None
        for attempt in range(FETCH_ATTEMPTS):
            fund = data_manager.get_stock_fundamentals_live(ticker, start, end)
            if fund is not None:
                break
            time.sleep(FETCH_BACKOFF_S * (attempt + 1))
        if fund is None or "Turnover_Rate" not in fund.columns:
            return {"ticker": ticker, "status": "error",
                    "error": "no turnover data (daily_basic)"}
        turnover = fund["Turnover_Rate"].reindex(stock_df.index)

        name = data_manager.get_stock_name_from_db(ticker) or ticker
        data_date = str(stock_df.index[-1].date())
        rows = []
        for decay in decays:
            m = cdist.analyse(stock_df, turnover, decay=decay)
            if not m.get("ok"):
                continue
            px, peak = m.get("price"), m.get("peak_price")
            rows.append({
                "ticker": ticker,
                "decay": float(decay),
                "name": name,
                "setup_score": _finite(m.get("setup_score"), 3),
                "setup_label": m.get("setup_label"),
                "price": _finite(px, 2),
                "peak_price": _finite(peak, 2),
                "n_peaks": int(m.get("n_peaks") or 0),
                "winner_rate": _finite(m.get("winner_rate"), 4),
                "concentration": _finite(m.get("concentration"), 4),
                "weight_avg": _finite(m.get("weight_avg"), 2),
                "pct_from_peak": (_finite((px / peak - 1) * 100, 1)
                                  if px and peak else None),
                "converged": bool(m.get("converged")),
            })
        if not rows:
            return {"ticker": ticker, "status": "no_data", "data_date": data_date}
        return {"ticker": ticker, "status": "ok", "data_date": data_date, "rows": rows}
    except Exception as e:
        return {"ticker": ticker, "status": "error", "error": str(e)[:300]}


def rank_rows(rows: list[dict]) -> pd.DataFrame:
    """Bullish first, then by signal count — the page's display order."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['_rank'] = df['Type'].map(BIAS_RANK).fillna(3)
    return (df.sort_values(['_rank', 'Signal_Count'], ascending=[True, False])
              .drop(columns=['_rank']).reset_index(drop=True))
