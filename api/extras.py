"""
api/extras.py — the What-If simulation, the 尾盘推演 AI read, and comparison.

All three reuse the modules the Streamlit app uses (analysis_engine,
whatif_advisor, watchlist_scan), so a ghost bar here means what a ghost bar
means there. The frames are cached alongside the analysis payload: a What-If
recompute must not pay for another 3-year fetch or another HMM fit, which is
what makes the simulator feel instant instead of costing 20 seconds a keystroke.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import ta_payload


def _n(v, nd=4):
    return ta_payload._num(v, nd)


def simulate(analysis_df: pd.DataFrame, *, pct: float, volume: float,
             open_: float | None, high: float | None, low: float | None) -> dict:
    """
    Tomorrow's bar and what every indicator would read if it printed.

    Same call the Streamlit simulator makes, so ±DI and ADX are exact when
    O/H/L are supplied rather than estimated from the close alone.
    """
    from analysis_engine import simulate_next_day_indicators

    sim = simulate_next_day_indicators(
        analysis_df, pct, volume,
        open_tomorrow=open_, high_tomorrow=high, low_tomorrow=low)
    if not sim:
        raise LookupError("not enough history to simulate")

    close_t = float(sim["close_tomorrow"])
    ma = sim.get("ma_tomorrow") or {}
    bb = sim.get("bb_tomorrow") or {}

    # Volume Z on the same 100-day baseline the chart's Z pane uses.
    last = analysis_df.iloc[-1]
    mu, sd = last.get("Vol_Mean_100d"), last.get("Vol_Std_100d")
    vol_z = (volume - float(mu)) / float(sd) if (pd.notna(mu) and pd.notna(sd) and sd) else None

    # Price Z: 20-day, population std, on returns — as in ta_payload.
    rets = analysis_df["Close"].pct_change().tail(ta_payload.Z_WINDOW - 1).tolist()
    r_new = close_t / float(last["Close"]) - 1
    series = pd.Series(rets + [r_new])
    price_z = ((r_new - series.mean()) / series.std(ddof=0)) if series.std(ddof=0) else None

    # OBV momentum, extended by the simulated bar.
    obv_mom = None
    if "OBV" in analysis_df.columns and len(analysis_df) > ta_payload.OBV_MOM_WINDOW:
        sign = 1 if close_t > float(last["Close"]) else (-1 if close_t < float(last["Close"]) else 0)
        obv_new = float(last["OBV"]) + sign * volume
        avg = float(pd.concat([analysis_df["Volume"].tail(19),
                               pd.Series([volume])]).mean())
        back = float(analysis_df["OBV"].iloc[-ta_payload.OBV_MOM_WINDOW])
        obv_mom = (obv_new - back) / avg if avg else None

    return {
        "date": str((analysis_df.index[-1] + pd.Timedelta(days=1)).date()),
        "ohlcv": {
            "o": _n(sim.get("open_tomorrow"), 3), "h": _n(sim.get("high_tomorrow"), 3),
            "l": _n(sim.get("low_tomorrow"), 3), "c": _n(close_t, 3),
            "v": _n(volume, 0),
        },
        "ohl_supplied": bool(sim.get("ohl_supplied")),
        "series": {
            "MA5": _n(ma.get("MA5"), 3), "MA10": _n(ma.get("MA10"), 3),
            "MA20": _n(ma.get("MA20"), 3), "MA50": _n(ma.get("MA50"), 3),
            "MA60": _n(ma.get("MA60"), 3), "MA200": _n(ma.get("MA200"), 3),
            "EMA5": _n(sim.get("ema5_tomorrow"), 3),
            "BB_Upper": _n(bb.get("upper"), 3), "BB_Lower": _n(bb.get("lower"), 3),
            "MACD": _n(sim.get("macd_tomorrow")),
            "MACD_Signal": _n(sim.get("macd_signal_tomorrow")),
            "MACD_Hist": _n((sim.get("macd_hist_tomorrow") or 0) * 2.5),
            "RSI": _n(sim.get("rsi_tomorrow"), 2),
            "ADX": _n(sim.get("adx_tomorrow"), 2),
            "DI_Plus": _n(sim.get("di_plus_tomorrow"), 2),
            "DI_Minus": _n(sim.get("di_minus_tomorrow"), 2),
            "OBV_Mom": _n(obv_mom, 3),
            "Price_Z": _n(price_z, 3), "Volume_Z": _n(vol_z, 3),
        },
        "signals": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                    for k, v in (sim.get("signals") or {}).items()},
        "adx_pattern": sim.get("adx_pattern"),
        "conditions_met": int(sim.get("conditions_met", 0)),
    }


def whatif_ai(analysis_df: pd.DataFrame, ticker: str, name: str, *,
              mode: str, pct: float = 0.0, volume: float | None = None,
              open_: float | None = None, high: float | None = None,
              low: float | None = None, window: int = 20) -> dict:
    """
    尾盘推演 — the AI read, on either the simulated bar or the last real one.

    Mirrors the Streamlit section: the deterministic crossings are computed
    here and handed over as fact, and 吸筹/出货 is re-detected on the bar being
    read using the same detector path.
    """
    import accumulation_signals as acsig
    import whatif_advisor as wadv
    from analysis_engine import run_single_stock_analysis, simulate_next_day_indicators

    if mode == "actual":
        hist = analysis_df.iloc[:-1]
        sim = wadv.sim_from_real_bar(analysis_df)
        if sim is None:
            raise LookupError("not enough history")
        ad_today = acsig.summarise(acsig.detect(hist, None, window=window))
        ad_bar = acsig.summarise(acsig.detect(analysis_df, None, window=window))
        brief = wadv.build_brief(hist, sim, ticker=ticker, name=name,
                                 ad_today=ad_today, ad_tomorrow=ad_bar,
                                 ad_window=window, mode="actual",
                                 bar_date=str(analysis_df.index[-1].date()))
    else:
        if volume is None:
            volume = float(analysis_df["Volume"].rolling(10).mean().iloc[-1])
        sim = simulate_next_day_indicators(analysis_df, pct, volume,
                                           open_tomorrow=open_, high_tomorrow=high,
                                           low_tomorrow=low)
        if not sim:
            raise LookupError("not enough history")
        c0 = float(analysis_df["Close"].iloc[-1])
        ct = c0 * (1 + pct / 100.0)
        o = open_ if open_ is not None else c0
        h = max(high if high is not None else max(o, ct), o, ct)
        l = min(low if low is not None else min(o, ct), o, ct)
        nb = analysis_df.index[-1] + pd.Timedelta(days=1)
        ext = pd.concat([
            analysis_df[["Open", "High", "Low", "Close", "Volume"]],
            pd.DataFrame({"Open": [o], "High": [h], "Low": [l],
                          "Close": [ct], "Volume": [volume]}, index=[nb])])
        ad_today = acsig.summarise(acsig.detect(analysis_df, None, window=window))
        ad_bar = acsig.summarise(acsig.detect(run_single_stock_analysis(ext.copy()),
                                              None, window=window))
        brief = wadv.build_brief(analysis_df, sim, ticker=ticker, name=name,
                                 ad_today=ad_today, ad_tomorrow=ad_bar,
                                 ad_window=window, mode="simulated")

    out = wadv.explain(brief)
    return {"mode": mode, "read": out, "crossings": brief.get("crossings", []),
            "bar_date": brief["asof"]["sim_session"]}


def compare(main_df: pd.DataFrame, other: str, dates: list[str]) -> dict:
    """
    A second stock on the price pane, aligned bar-for-bar with the main one.

    Returns BOTH scalings so the toggle is instant and needs no refetch:
      * `rebased` — the comparison restated so it starts at the main stock's
        first close. Both then share one ¥ axis and the gap between the lines
        is pure relative performance (TradingView's "same % scale").
      * `price`   — the comparison's own closes, for its own axis.
    """
    import data_manager
    import watchlist_scan

    df = watchlist_scan.fetch_frame(other)
    if df is None:
        raise RuntimeError(f"price data for {other} could not be fetched — try again")
    if len(df) < 30:
        raise LookupError(f"not enough price history for {other}")

    idx = pd.to_datetime(dates)
    aligned = df["Close"].reindex(idx, method="ffill")
    first = next((v for v in aligned.tolist() if v and not math.isnan(v)), None)
    base = float(main_df["Close"].reindex(idx, method="ffill").dropna().iloc[0])
    rebased = (aligned / first * base) if first else aligned

    return {
        "ticker": other,
        "name": data_manager.get_stock_name_from_db(other) or other,
        "price": [_n(v, 3) for v in aligned.tolist()],
        "rebased": [_n(v, 3) for v in rebased.tolist()],
        # Total return over the window, for the legend.
        "change_pct": _n((aligned.dropna().iloc[-1] / first - 1) * 100, 2) if first else None,
    }
