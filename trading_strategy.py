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

    # Guard: rolling mean can be 0 or NaN at the start of the window → Inf ratio.
    # Replace Inf/NaN with 0 so those rows never trigger a breakout boundary.
    raw_ratio = subset["Volume"] / subset["Volume"].rolling(20).mean().shift(1)
    subset["VolRatio"] = (raw_ratio
                         .replace([np.inf, -np.inf], np.nan)
                         .fillna(0))

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
        if pmin >= pmax:
            pmax = pmin + 0.01

        bins = np.linspace(pmin, pmax, 21)   # 21 edges → 20 bins

        # np.digitize with right=False assigns index len(bins) to values equal
        # to bins[-1] (i.e. closes at the segment high).  Clip to [1, 20] so
        # the highest-price closes are counted in the top bin, not discarded.
        raw_idx = np.digitize(seg["Close"].values, bins)
        indices = np.clip(raw_idx, 1, len(bins) - 1)

        bin_vol: dict = {}
        for j, vol in zip(indices, seg["Volume"]):
            bin_vol[j] = bin_vol.get(j, 0) + vol

        sorted_bins = sorted(bin_vol.items(), key=lambda x: x[1], reverse=True)
        total = sum(bin_vol.values())
        if total == 0:
            continue

        cum, value_bins = 0, []
        for bin_idx, v in sorted_bins:
            cum += v
            value_bins.append(bin_idx)
            if cum >= 0.7 * total:
                break

        # All indices are already clipped to [1, 20] — no further filter needed
        if not value_bins:
            continue

        top = float(bins[max(value_bins)])        # upper edge of highest-volume bin
        bot = float(bins[min(value_bins) - 1])    # lower edge of lowest-volume bin
        if top <= bot:
            top = bot + 0.01

        if is_active:
            status = ("BREAKOUT"  if cur_price > top
                      else "BREAKDOWN" if cur_price < bot
                      else "INSIDE")
        else:
            status = ("SUPPORT_BELOW"    if cur_price > top
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
    From the block list, pull every top/bot edge and cluster them into
    resistances (above current) and supports (below current).

    Clustering rule: levels within 1% of the FIRST member of a cluster
    (anchor-based, not running-average-based) are merged. This avoids
    "cluster drift" where three levels at 100 / 100.8 / 101.5 would all
    merge under a running-average comparison (avg drifts to 100.4, and
    101.5 is 1.09% from 100.4 — borderline), but 101.5 is 1.5% from the
    anchor 100 and correctly starts a new cluster.

    strength = number of block edges that fell into the same cluster
    (proxy for how many distinct consolidation periods touched that level).

    Levels more than 25% away from current price are dropped — they are too
    distant to be relevant for the immediate trade setup.

    Returns (resistances, supports), each sorted nearest-first.
    """
    raw_levels = []
    for b in blocks:
        raw_levels.append(b["top"])
        raw_levels.append(b["bot"])

    # Remove levels that are unrealistically far from current price
    max_dist = 0.25  # 25% either side
    raw_levels = [p for p in raw_levels
                  if abs(p - current_price) / current_price <= max_dist]

    raw_levels.sort()

    # Anchor-based clustering: compare each incoming level to clusters[i][0]
    clusters: list[list[float]] = []
    for price in raw_levels:
        if not clusters:
            clusters.append([price])
            continue
        anchor = clusters[-1][0]                     # ← always the first member
        if abs(price - anchor) / anchor < 0.01:      # within 1% of anchor
            clusters[-1].append(price)
        else:
            clusters.append([price])

    levels = [{"price": sum(c) / len(c), "strength": len(c)} for c in clusters]

    resistances = sorted(
        [lv for lv in levels if lv["price"] > current_price],
        key=lambda lv: lv["price"],          # nearest first (ascending)
    )
    supports = sorted(
        [lv for lv in levels if lv["price"] < current_price],
        key=lambda lv: lv["price"], reverse=True,   # nearest first (descending)
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

    # China context: in a confirmed long-term uptrend (price > MA50 > MA200),
    # bearish MACD is typically a healthy pullback inside a bull cycle, not a
    # trend reversal.  Policy/SOE/sector-driven A-share names rarely collapse
    # indefinitely — a very low MACD is often a bottoming signal.
    # Halve the bearish penalty when the structural trend is clearly bullish.
    if macd_score < 0:
        close_v = last.get("Close")
        ma50_v  = last.get("MA50")
        ma200_v = last.get("MA200")
        if all(pd.notna(x) for x in (close_v, ma50_v, ma200_v)):
            if close_v > ma50_v and ma50_v > ma200_v:
                macd_score = macd_score // 2  # -20 → -10, -10 → -5

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

    # MAX = tighter stop (closer to current price = smaller loss).
    # ATR stop wins when support is far away; sup stop wins when support is
    # right below (breaking it clearly invalidates the setup).
    # Floor at price * 0.01 so stop never goes negative on very cheap stocks.
    stop = max(atr_stop, sup_stop, price * 0.01)
    risk = max(price - stop, 1e-6)

    t1 = price + 2 * atr   # partial exit  (~1R reward)
    t2 = price + 3 * atr   # main target   (~1.5R reward)
    t3 = price + 4 * atr   # full run      (~2R reward)

    # Cap targets at the nearest resistance, working top-down so the ordering
    # T1 ≤ T2 ≤ T3 is always preserved.
    if resistances:
        r1 = resistances[0]["price"]
        if r1 <= t1:       # resistance is at or below T1 — all targets get capped
            t1 = t2 = t3 = r1
        elif r1 <= t2:     # resistance between T1 and T2 — cap T2 and T3
            t2 = t3 = r1
        elif r1 < t3:      # resistance between T2 and T3 — cap only T3
            t3 = r1

    rr = (t2 - price) / risk   # R:R measured to T2 (the realistic exit)

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
