"""
Parity: the SPA payload must plot the same numbers as the Streamlit chart.

Builds the REAL Plotly figure with create_single_stock_chart_analysis from the
Streamlit page, feeds ta_payload.build_series the identical frames, and
compares every trace the two share, by name. Lines are compared bar by bar;
marker traces are compared as the set of dates they mark.

Integration test: needs live Tushare + Supabase credentials in the
environment (or .streamlit/secrets.toml). Run manually:

    python -m pytest api/tests/test_ta_payload_parity.py -q
    python api/tests/test_ta_payload_parity.py 600519 002594
"""

from __future__ import annotations

import io
import math
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

PAGE = os.path.join(ROOT, "pages", "2_Single_Stock_Analysis_个股分析.py")

LINES = {           # Plotly trace name → payload series key
    "MA5": "MA5", "MA10": "MA10", "MA20": "MA20", "MA50": "MA50",
    "MA60": "MA60", "MA200": "MA200", "EMA5": "EMA5",
    "BB Upper": "BB_Upper", "BB Lower": "BB_Lower",
    "Vol-Scaled OBV": "Vol_Scaled_OBV", "OBV动能(20日)": "OBV_Mom",
    "MACD": "MACD", "MACD Signal": "MACD_Signal",
    "MACD Histogram (2.5x)": "MACD_Hist",
    "RSI(14)": "RSI",
    "ADX (Raw)": "ADX", "ADX LOWESS": "ADX_LOWESS",
    "ADX BB Upper": "ADX_BB_Upper", "ADX BB Lower": "ADX_BB_Lower",
    "+DI": "DI_Plus", "-DI": "DI_Minus",
    "Price Z": "Price_Z", "Volume Z": "Volume_Z",
    "P/E (TTM)": "PE_TTM",
    "主力净流入/日": "MF_Daily", "近20日主力净流入": "MF_Rolling",
}

# The Streamlit chart min-max RESCALES these two onto the volume axis so they
# fit one Plotly axis, and keeps the true values in customdata for the hover.
# The payload sends the true values (the SPA gives each its own price scale),
# so these are compared against customdata, not y.
USE_CUSTOMDATA = {"Vol-Scaled OBV", "OBV动能(20日)"}

# Decimal places ta_payload rounds each series to; the comparison tolerance is
# half a rounding step, so rounding can never read as a mismatch.
DECIMALS = {"RSI": 2, "RSI_P10": 2, "RSI_P90": 2, "ADX": 2, "ADX_LOWESS": 2,
            "ADX_BB_Upper": 2, "ADX_BB_Lower": 2, "DI_Plus": 2, "DI_Minus": 2,
            "PE_TTM": 2, "MF_Daily": 1, "MF_Rolling": 1,
            "MACD": 4, "MACD_Signal": 4, "MACD_Hist": 4}

MARKERS = {         # Plotly trace name → (payload group, key)
    "Phase 1: Accumulation": ("price", "accumulation"),
    "Phase 2: Squeeze": ("price", "squeeze"),
    "🚀 Bullish Squeeze Breakout": ("price", "squeeze_bull"),
    "🩸 Bearish Squeeze Drop": ("price", "squeeze_bear"),
    "🔄 Downtrend Reversal": ("price", "downtrend_reversal"),
    "🔄 Uptrend Reversal": ("price", "uptrend_reversal"),
    "Exit Signal": ("price", "exit_macd"),
    "🚀 Entry · Screaming Buy": ("price", "screaming_buy"),
    "🛑 Exit · Screaming Sell": ("price", "screaming_sell"),
    "MACD Triggers": ("macd", "trigger"),
    "MACD Peaking": ("macd", "peaking"),
    "Bearish Cross": ("macd", "bearish_cross"),
    "🔵 RSI 极低 (P10)": ("rsi", "bottoming"),
    "🔴 RSI 极高 (P90)": ("rsi", "peaking"),
    "DI Screaming Breakout": ("adx", "di_screaming_buy"),
    "🛑 DI Screaming Sell": ("adx", "di_screaming_sell"),
    "🔄 Bottoming (DI+ dom.)": ("adx", "bottoming"),
    "🔺 Reversing Up (DI+ dom.)": ("adx", "reversing_up"),
    "🔴 Peaking (uptrend top)": ("adx", "peaking"),
    "🔻 Reversing Down (uptrend fading)": ("adx", "reversing_down"),
    "Oversold (Z ≤ -2.5)": ("z", "oversold"),
    "Overbought (Z ≥ +2.0)": ("z", "overbought"),
}


def _load_secrets():
    path = os.path.join(ROOT, ".streamlit", "secrets.toml")
    for cand in (path, os.path.join(ROOT, "..", "..", "..", ".streamlit", "secrets.toml")):
        if os.path.exists(cand):
            for line in open(cand, encoding="utf-8"):
                m = re.match(r'\s*([A-Z_]+)\s*=\s*"([^"]+)"', line)
                if m:
                    os.environ.setdefault(m.group(1), m.group(2))
            return


def _page_chart_fn():
    """Pull create_single_stock_chart_analysis and its helpers out of the page."""
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st
    from plotly.subplots import make_subplots
    import chart_utils
    from analysis_engine import REGIME_ANCHOR

    src = io.open(PAGE, encoding="utf-8").read()

    def grab(name):
        # From "def name" to the next TOP-LEVEL def/class/decorator, so a
        # decorated function that follows is never half-included.
        i = src.index("\ndef " + name + "(") + 1
        m = re.compile(r"^(def |class |@)", re.M).search(src, i + 4)
        return src[i:m.start() if m else len(src)]

    ns = {"go": go, "np": np, "pd": pd, "st": st, "make_subplots": make_subplots,
          "split_legends_by_panel": chart_utils.split_legends_by_panel,
          "REGIME_ANCHOR": REGIME_ANCHOR, "INITIAL_VISIBLE_BARS": 250}
    for name in ("chart_window", "_add_ghost_traces", "create_single_stock_chart_analysis"):
        exec(grab(name), ns)
    return ns["create_single_stock_chart_analysis"]


def compare(ticker: str) -> list[str]:
    import data_manager
    import ta_payload
    from analysis_engine import run_single_stock_analysis

    stock_df = data_manager.get_single_stock_data_live(ticker, lookback_years=3)
    adf = run_single_stock_analysis(stock_df)
    fund = data_manager.get_stock_fundamentals_live(
        ticker, stock_df.index.min().strftime('%Y%m%d'), stock_df.index.max().strftime('%Y%m%d'))
    mf = ta_payload.load_moneyflow(ticker)

    fig = _page_chart_fn()(adf, fundamentals_df=fund, blocks=[], moneyflow_df=mf,
                           mf_cum_days=20)
    body = ta_payload.build_series(adf, fund, mf)
    dates = body["dates"]
    pos = {d: i for i, d in enumerate(dates)}

    problems, checked = [], {"lines": 0, "markers": 0}
    seen = set()
    for tr in fig.data:
        name = tr.name
        if name in seen or (name not in LINES and name not in MARKERS):
            continue
        xs = [str(x) for x in (tr.x if tr.x is not None else [])]
        ys = list(getattr(tr, "y", None) if getattr(tr, "y", None) is not None else [])

        if name in LINES:
            seen.add(name)
            key = LINES[name]
            ours = body["series"][key]
            if name in USE_CUSTOMDATA:
                ys = list(tr.customdata)
            nd = DECIMALS.get(key, 3)
            half_step = 0.5 * 10 ** (-nd) + 1e-9
            for x, y in zip(xs, ys):
                if x not in pos:
                    problems.append(f"{name}: plotted date {x} not in payload dates")
                    break
                a = ours[pos[x]]
                b = None if (y is None or (isinstance(y, float) and math.isnan(y))) else float(y)
                if a is None and b is None:
                    continue
                if a is None or b is None or abs(a - b) > half_step:
                    problems.append(f"{name} @ {x}: payload {a} vs chart {b}")
                    break
            checked["lines"] += 1
        elif name in MARKERS:
            seen.add(name)
            grp, key = MARKERS[name]
            ours = {dates[i] for i in body["markers"][grp][key]}
            theirs = set(xs)
            if ours != theirs:
                problems.append(f"{name}: payload-only {sorted(ours - theirs)[:4]} "
                                f"chart-only {sorted(theirs - ours)[:4]}")
            checked["markers"] += 1

    missing_lines = [n for n in LINES if n not in seen and n != "近20日主力净流入" or (
        n == "近20日主力净流入" and body["has_moneyflow"] and n not in seen)]
    missing_lines = [n for n in missing_lines if not (n.startswith("主力") and not body["has_moneyflow"])]
    print(f"  {ticker}: compared {checked['lines']} line traces, "
          f"{checked['markers']} marker traces over {len(dates)} bars; "
          f"line traces absent from this chart: {missing_lines or 'none'}")
    return problems


def test_parity():
    _load_secrets()
    problems = []
    for t in ("600519", "002594", "300750"):
        problems += compare(t)
    assert not problems, "\n".join(problems)


if __name__ == "__main__":
    _load_secrets()
    import warnings
    warnings.filterwarnings("ignore")
    allp = []
    for t in (sys.argv[1:] or ["600519", "002594", "300750"]):
        p = compare(t)
        allp += p
        for x in p:
            print("   MISMATCH", x)
    print("PARITY OK" if not allp else f"{len(allp)} MISMATCH(ES)")
    sys.exit(1 if allp else 0)
