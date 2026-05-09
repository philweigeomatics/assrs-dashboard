"""
trading_strategy.py — Composite signal score, stop/target ladder, entry zones.

Reuses the existing analysis_engine columns wherever possible so we don't
duplicate indicator math. Adds only:
  - ATR (14-day Wilder smoothing)
  - Volume-weighted block detection (lifted from Single Stock Analysis page)
  - Composite signal score weighted across diversified sources
  - Stop/target calculator using MAX(ATR_stop, support_stop) — NOT a blend
  - Entry-zone classifier tied to ATR distance from S/R cluster

All consumers (Equity Brief, future pages) should call:
    summary = build_strategy_summary(raw_df, analysis_df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── ATR (14-day, Wilder) ──────────────────────────────────────────────────────

def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = Wilder-smoothed TR. Returns a series aligned to df.index.
    """
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder smoothing == EMA with alpha = 1/period
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ── Volume-weighted block detection (lifted, parameterized) ───────────────────

def detect_blocks(df: pd.DataFrame, lookback: int = 120) -> list[dict]:
    """
    Volume-weighted box detection. Each block's top/bot are the price band
    that contains 70% of traded volume during that segment.

    Segments are split by big-volume / big-move days (>3% move and >1.5x avg
    volume). Returns list of {start, end, top, bot, status, is_active}.

    Lookback widened to 120 (vs 60 in Single Stock) for a more stable S/R map
    on the brief page — short windows produce 1-2 blocks which is too few.
    """
    if df is None or len(df) < 20:
        return []

    subset = df.tail(lookback).copy()
    all_dates = subset.index.tolist()

    subset["PctChange"] = subset["Close"].pct_change().abs()
    subset["VolRatio"]  = subset["Volume"] / subset["Volume"].rolling(20).mean().shift(1)

    breakout_mask  = (subset["PctChange"] > 0.03) & (subset["VolRatio"] > 1.5)
    breakout_dates = subset.index[breakout_mask].tolist()

    boundary = [0]
    for d in breakout_dates:
        if d in all_dates:
            boundary.append(all_dates.index(d))
    boundary.append(len(all_dates))
    boundary = sorted(set(boundary))

    blocks = []
    cur_price = float(df["Close"].iloc[-1])

    for i in range(len(boundary) - 1):
        a, b = boundary[i], boundary[i + 1]
        is_active = (i == len(boundary) - 2)
        seg = subset.iloc[a:] if is_active else subset.iloc[a:b]
        if len(seg) < 3:
            continue

        pmin, pmax = float(seg["Low"].min()), float(seg["High"].max())
        if pmin == pmax:
            pmax = pmin + 0.01

        bins    = np.linspace(pmin, pmax, 21)
        indices = np.digitize(seg["Close"], bins)
        bin_vol: dict = {}
        for j, vol in zip(indices, seg["Volume"]):
            bin_vol[j] = bin_vol.get(j, 0) + vol

        sorted_bins = sorted(bin_vol.items(), key=lambda x: x[1], reverse=True)
        total = sum(bin_vol.values())
        cum, value_bins = 0, []
        for bin_idx, v in sorted_bins:
            cum += v
            value_bins.append(bin_idx)
            if cum >= 0.7 * total:
                break

        valid = [v for v in value_bins if 1 <= v < len(bins)]
        if not valid:
            continue

        top, bot = float(bins[max(valid)]), float(bins[min(valid) - 1])
        if top <= bot:
            top = bot + 0.01

        if is_active:
            status = ("BREAKOUT" if cur_price > top
                      else "BREAKDOWN" if cur_price < bot
                      else "INSIDE")
        else:
            status = ("SUPPORT_BELOW" if cur_price > top
                      else "RESISTANCE_ABOVE" if cur_price < bot
                      else "INSIDE_OLD_RANGE")

        blocks.append({
            "start": seg.index[0], "end": seg.index[-1],
            "top": top, "bot": bot, "status": status, "is_active": is_active,
        })
    return blocks


# ── Convert blocks into ranked S/R levels ─────────────────────────────────────

def extract_sr_levels(blocks: list[dict], current_price: float
                      ) -> tuple[list[dict], list[dict]]:
    """
    From the block list, pull every top/bot edge and bucket them into
    resistances (above current) and supports (below current). Levels within
    1% of each other are merged (averaged), and "strength" = number of
    blocks contributing to that cluster.

    Returns (resistances, supports), each sorted by distance from current.
    """
    raw_levels = []
    for b in blocks:
        raw_levels.append((b["top"], b))
        raw_levels.append((b["bot"], b))

    # Cluster within 1% of each other
    raw_levels.sort(key=lambda x: x[0])
    clusters: list[list[float]] = []
    for price, _ in raw_levels:
        if not clusters:
            clusters.append([price])
            continue
        last_avg = sum(clusters[-1]) / len(clusters[-1])
        if abs(price - last_avg) / last_avg < 0.01:
            clusters[-1].append(price)
        else:
            clusters.append([price])

    levels = [{"price": sum(c) / len(c), "strength": len(c)} for c in clusters]

    resistances = sorted(
        [l for l in levels if l["price"] > current_price],
        key=lambda l: l["price"],
    )
    supports = sorted(
        [l for l in levels if l["price"] < current_price],
        key=lambda l: l["price"], reverse=True,
    )
    return resistances, supports


# ── Composite Signal Score (weighted, diversified sources) ────────────────────

def compute_signal_score(analysis_df: pd.DataFrame) -> dict:
    """
    Returns {score: int -100..100, label: str, components: dict}.

    Five buckets, each contributing a signed value:
      - Trend (MA stack)         25 pts
      - Momentum (MACD)          20 pts
      - RSI mean-reversion bias  15 pts
      - Volatility regime (BB)   15 pts
      - Custom signals           25 pts
    """
    if analysis_df is None or analysis_df.empty:
        return {"score": 0, "label": "—", "components": {}}

    last = analysis_df.iloc[-1]

    # ── 1. Trend (MA stack) — 25 pts max
    ma5   = last.get("MA5")
    ma20  = last.get("MA20")
    ma50  = last.get("MA50")
    ma200 = last.get("MA200")
    trend_score = 0
    trend_label = "—"
    if all(pd.notna(x) for x in (ma5, ma20, ma50, ma200)):
        if   ma5 > ma20 > ma50 > ma200: trend_score, trend_label = +25, "Strong Bullish"
        elif ma5 > ma20 > ma50:         trend_score, trend_label = +18, "Bullish"
        elif ma5 < ma20 < ma50 < ma200: trend_score, trend_label = -25, "Strong Bearish"
        elif ma5 < ma20 < ma50:         trend_score, trend_label = -18, "Bearish"
        else:                            trend_score, trend_label =   0, "Neutral"

    # ── 2. Momentum (MACD) — 20 pts max
    macd      = last.get("MACD")
    macd_sig  = last.get("MACD_Signal")
    macd_hist = last.get("MACD_Hist")
    macd_score = 0
    if all(pd.notna(x) for x in (macd, macd_sig, macd_hist)):
        if macd > macd_sig and macd_hist > 0: macd_score = +20
        elif macd > macd_sig:                  macd_score = +10
        elif macd < macd_sig and macd_hist < 0: macd_score = -20
        elif macd < macd_sig:                  macd_score = -10

    # ── 3. RSI — 15 pts max (mean-reversion bias: extremes favour reversal)
    rsi = last.get("RSI_14")
    rsi_score = 0
    if pd.notna(rsi):
        if   rsi >= 75: rsi_score = -15  # severely overbought
        elif rsi >= 65: rsi_score = -5   # warming up
        elif rsi <= 25: rsi_score = +15  # severely oversold
        elif rsi <= 35: rsi_score = +5   # cooling off
        elif 45 <= rsi <= 55: rsi_score =  0
        elif rsi > 55:  rsi_score = +3
        else:           rsi_score = -3

    # ── 4. Volatility regime / BB position — 15 pts max
    bb_u = last.get("BB_Upper")
    bb_l = last.get("BB_Lower")
    close = last.get("Close")
    bb_score = 0
    if all(pd.notna(x) for x in (bb_u, bb_l, close)) and bb_u > bb_l:
        pos = (close - bb_l) / (bb_u - bb_l)
        if   pos >= 0.95: bb_score = -15  # riding upper band — stretched
        elif pos >= 0.80: bb_score = -5
        elif pos <= 0.05: bb_score = +15  # riding lower — coiled
        elif pos <= 0.20: bb_score = +5
        else:             bb_score =  0

    # ── 5. Custom signals — 25 pts max (Accumulation, Squeeze, Squeeze Fired)
    custom = 0
    if last.get("Squeeze_Fired_Bullish", False):  custom += 25
    elif last.get("Signal_Accumulation", False):   custom += 18
    elif last.get("Signal_Squeeze", False):        custom +=  8  # waiting → mildly positive
    if last.get("Squeeze_Fired_Bearish", False):  custom -= 25
    if last.get("Exit_MACD_Lead", False):          custom -= 12
    custom = max(-25, min(25, custom))

    total = trend_score + macd_score + rsi_score + bb_score + custom
    total = max(-100, min(100, total))

    if   total >=  70: label = "Strong Buy"
    elif total >=  40: label = "Buy"
    elif total >=  10: label = "Accumulate"
    elif total >  -10: label = "Neutral"
    elif total >  -40: label = "Weak Sell"
    elif total >  -70: label = "Sell"
    else:              label = "Strong Sell"

    return {
        "score": int(total),
        "label": label,
        "components": {
            "trend":      {"value": trend_score, "max": 25, "label": trend_label},
            "momentum":   {"value": macd_score,  "max": 20, "label": "MACD"},
            "rsi":        {"value": rsi_score,   "max": 15, "label": f"RSI {rsi:.0f}" if pd.notna(rsi) else "RSI —"},
            "volatility": {"value": bb_score,    "max": 15, "label": "BB position"},
            "custom":     {"value": custom,      "max": 25, "label": "Custom signals"},
        },
    }


# ── Stop & Target ladder ──────────────────────────────────────────────────────

def compute_stop_targets(price: float, atr: float,
                         resistances: list[dict], supports: list[dict]) -> dict:
    """
    Stop = MAX(price - 2*ATR, nearest_support * 0.97).
    Taking the MAX (i.e. tighter stop) is correct because:
      - if support is far below: ATR stop protects against noise
      - if support is right below: the support stop is more meaningful (a
        decisive break of support invalidates the setup)

    Targets are an ATR-multiple ladder, T3 capped at the nearest resistance
    if one exists below T3.
    """
    atr_stop = price - 2 * atr
    nearest_support = supports[0]["price"] if supports else None
    sup_stop = nearest_support * 0.97 if nearest_support else atr_stop

    stop = max(atr_stop, sup_stop)
    risk = max(price - stop, 1e-6)

    t1 = price + 2 * atr
    t2 = price + 3 * atr
    t3 = price + 4 * atr
    if resistances:
        nearest_r = resistances[0]["price"]
        if nearest_r < t3:
            t3 = nearest_r

    rr = (t2 - price) / risk  # use T2 as the realistic-target R:R

    return {
        "stop":          round(stop, 2),
        "stop_atr":      round(atr_stop, 2),
        "stop_support":  round(sup_stop, 2) if nearest_support else None,
        "t1":            round(t1, 2),
        "t2":            round(t2, 2),
        "t3":            round(t3, 2),
        "risk_pct":      round((price - stop) / price * 100, 2),
        "reward_t1_pct": round((t1 - price) / price * 100, 2),
        "reward_t2_pct": round((t2 - price) / price * 100, 2),
        "reward_t3_pct": round((t3 - price) / price * 100, 2),
        "rr":            round(rr, 2),
    }


# ── Entry-zone classifier (6 bands, ATR-distance from S/R cluster) ────────────

def classify_entry_zone(price: float, atr: float,
                        resistances: list[dict], supports: list[dict]) -> dict:
    """
    Six unambiguous zones based on ATR distance from the nearest S/R cluster.
    No overlaps possible.

    Returns {zone, label, action, all_zones}.
      all_zones is a list of {label, low, high, color} for the bar visual.
    """
    s1 = supports[0]["price"]   if len(supports)    >= 1 else price - 2 * atr
    s2 = supports[1]["price"]   if len(supports)    >= 2 else s1 - 2 * atr
    r1 = resistances[0]["price"] if len(resistances) >= 1 else price + 2 * atr
    r2 = resistances[1]["price"] if len(resistances) >= 2 else r1 + 2 * atr

    half = 0.5 * atr

    # Build the 6 bands as contiguous price intervals, low → high
    bands = [
        {"key": "deep",        "label": "Deep Value",   "low": s2 - 2 * atr, "high": s2 - half,
         "action": "Heavy load — only if thesis intact",         "color": "#27ae60"},
        {"key": "buy",         "label": "Buy",          "low": s2 - half,    "high": s1 + half,
         "action": "Build position around support",              "color": "#2ecc71"},
        {"key": "accumulate",  "label": "Accumulate",   "low": s1 + half,    "high": price - half,
         "action": "Scale in on dips",                           "color": "#82e0aa"},
        {"key": "fair",        "label": "Fair Value",   "low": price - half, "high": price + half,
         "action": "Pilot position only",                        "color": "#f4d03f"},
        {"key": "stretched",   "label": "Stretched",    "low": price + half, "high": r1 - half,
         "action": "Chase only with small size",                 "color": "#e67e22"},
        {"key": "overextended","label": "Overextended", "low": r1 - half,    "high": r2 + 2 * atr,
         "action": "Avoid / take profits on existing",           "color": "#e74c3c"},
    ]

    # Resolve order issues if S/R are tightly bunched (e.g. s1 ≈ s2)
    # by enforcing low < high; collapse degenerate bands to a thin slice.
    cleaned = []
    last_high = bands[0]["low"]
    for b in bands:
        if b["high"] <= b["low"]:
            b["high"] = b["low"] + atr * 0.1
        if b["low"] < last_high:
            b["low"] = last_high
        if b["high"] <= b["low"]:
            b["high"] = b["low"] + atr * 0.1
        last_high = b["high"]
        cleaned.append(b)

    # Find the band containing the current price
    current = next((b for b in cleaned if b["low"] <= price < b["high"]),
                   cleaned[3])  # default to "Fair Value"

    return {
        "zone":      current["key"],
        "label":     current["label"],
        "action":    current["action"],
        "all_zones": cleaned,
    }


# ── Public assembly ───────────────────────────────────────────────────────────

def build_strategy_summary(raw_df: pd.DataFrame, analysis_df: pd.DataFrame) -> dict:
    """
    One call → everything needed to render the strategy block.
    Returns a dict with:
      price, atr, score (dict), stop_targets, entry_zone, supports, resistances.
    """
    if raw_df is None or raw_df.empty or analysis_df is None or analysis_df.empty:
        return {}

    price = float(analysis_df["Close"].iloc[-1])
    atr = float(atr_series(raw_df, 14).iloc[-1])
    if pd.isna(atr) or atr <= 0:
        atr = price * 0.02  # 2% fallback if ATR unavailable

    blocks = detect_blocks(raw_df, lookback=120)
    resistances, supports = extract_sr_levels(blocks, price)

    score = compute_signal_score(analysis_df)
    st_tg = compute_stop_targets(price, atr, resistances, supports)
    zone  = classify_entry_zone(price, atr, resistances, supports)

    return {
        "price":        price,
        "atr":          atr,
        "score":        score,
        "stop_targets": st_tg,
        "entry_zone":   zone,
        "resistances":  resistances[:3],
        "supports":     supports[:3],
    }
