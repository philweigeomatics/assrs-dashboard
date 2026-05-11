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
    trend_reason = "insufficient MA data"
    if all(pd.notna(x) for x in (ma5, ma20, ma50, ma200)):
        if   ma5 > ma20 > ma50 > ma200: trend_score, trend_label, trend_reason = +25, "Strong Bullish", "MA5 > MA20 > MA50 > MA200"
        elif ma5 > ma20 > ma50:         trend_score, trend_label, trend_reason = +18, "Bullish",        "MA5 > MA20 > MA50"
        elif ma5 < ma20 < ma50 < ma200: trend_score, trend_label, trend_reason = -25, "Strong Bearish", "MA5 < MA20 < MA50 < MA200"
        elif ma5 < ma20 < ma50:         trend_score, trend_label, trend_reason = -18, "Bearish",        "MA5 < MA20 < MA50"
        else:                            trend_score, trend_label, trend_reason =   0, "Neutral",        "MAs mixed / no clean stack"

    # ── 2. Momentum (MACD scenarios) — 20 pts max ───────────────────────────────
    # Scenario-based scoring: we use the richer MACD signals computed by
    # analysis_engine (Bottoming, ClassicCrossover, Approaching, MomentumBuilding,
    # Peaking, BearishCrossover) rather than the raw MACD position.
    #
    # Key insight: MACD_Bottoming (negative MACD turning up with volume) is a
    # *buy* signal — the stock is cheap and early.  The old position-based system
    # would penalise it as -20, which is the opposite of correct.
    # MACD_Peaking (positive but decelerating) is a *sell* signal — the old
    # system would score it +10 to +20, also wrong.
    macd_score = 0
    macd_label = "MACD"
    macd_reason = "no active scenario"

    if last.get("MACD_Bottoming", False):
        macd_score  = +15
        macd_label  = "MACD Bottoming ↑"
        macd_reason = "neg. MACD curling up + OBV rising"
    elif last.get("MACD_ClassicCrossover", False):
        macd_score  = +20
        macd_label  = "MACD Crossover ✓"
        macd_reason = "MACD crossed above signal line"
    elif last.get("MACD_Approaching", False):
        macd_score  = +10
        macd_label  = "MACD Approaching"
        macd_reason = "below signal, gap closing"
    elif last.get("MACD_MomentumBuilding", False):
        macd_score  = +15
        macd_label  = "MACD Momentum"
        macd_reason = "above signal, gap expanding 3d"
    elif last.get("MACD_Peaking", False):
        macd_score  = -15
        macd_label  = "MACD Peaking ↓"
        macd_reason = "positive but momentum decelerating"
    elif last.get("MACD_BearishCrossover", False):
        macd_score  = -20
        macd_label  = "MACD Bear Cross"
        macd_reason = "MACD crossed below signal line"
    else:
        # No scenario active — fall back to raw position (half weight vs. a scenario)
        macd      = last.get("MACD")
        macd_sig  = last.get("MACD_Signal")
        macd_hist = last.get("MACD_Hist")
        if all(pd.notna(x) for x in (macd, macd_sig, macd_hist)):
            if   macd > macd_sig and macd_hist > 0:
                macd_score = +10; macd_reason = "above signal, hist positive"
            elif macd > macd_sig:
                macd_score =  +5; macd_reason = "above signal, hist negative"
            elif macd < macd_sig and macd_hist < 0:
                macd_score = -10; macd_reason = "below signal, hist negative"
            elif macd < macd_sig:
                macd_score =  -5; macd_reason = "below signal, hist positive"
        macd_label = "MACD"

    # China uptrend dampening — halve bearish penalty when price > MA50 > MA200
    # (structural bull trend = healthy pullback, not collapse).
    # Intentionally excluded for MACD_Peaking: a peaking divergence is a real
    # warning even inside a bull trend and should NOT be softened.
    if macd_score < 0 and not last.get("MACD_Peaking", False):
        close_v = last.get("Close")
        ma50_v  = last.get("MA50")
        ma200_v = last.get("MA200")
        if all(pd.notna(x) for x in (close_v, ma50_v, ma200_v)):
            if close_v > ma50_v and ma50_v > ma200_v:
                macd_score = macd_score // 2  # e.g. -20 → -10

    # ── 3. RSI — 15 pts max (mean-reversion bias: extremes favour reversal)
    rsi = last.get("RSI_14")
    rsi_score = 0
    rsi_reason = "no RSI data"
    if pd.notna(rsi):
        if   rsi >= 75: rsi_score, rsi_reason = -15, f"RSI {rsi:.0f} — severely overbought"
        elif rsi >= 65: rsi_score, rsi_reason =  -5, f"RSI {rsi:.0f} — warming up"
        elif rsi <= 25: rsi_score, rsi_reason = +15, f"RSI {rsi:.0f} — severely oversold"
        elif rsi <= 35: rsi_score, rsi_reason =  +5, f"RSI {rsi:.0f} — cooling off"
        elif 45 <= rsi <= 55: rsi_score, rsi_reason = 0, f"RSI {rsi:.0f} — neutral range"
        elif rsi > 55:  rsi_score, rsi_reason =  +3, f"RSI {rsi:.0f} — slightly extended"
        else:           rsi_score, rsi_reason =  -3, f"RSI {rsi:.0f} — slightly elevated"

    # ── 4. Volatility regime / BB position — 15 pts max
    bb_u = last.get("BB_Upper")
    bb_l = last.get("BB_Lower")
    close = last.get("Close")
    bb_score = 0
    bb_reason = "no BB data"
    if all(pd.notna(x) for x in (bb_u, bb_l, close)) and bb_u > bb_l:
        pos = (close - bb_l) / (bb_u - bb_l)
        bb_pct = f"{pos * 100:.0f}% of band"
        if   pos >= 0.95: bb_score, bb_reason = -15, f"riding upper band ({bb_pct})"
        elif pos >= 0.80: bb_score, bb_reason =  -5, f"near upper band ({bb_pct})"
        elif pos <= 0.05: bb_score, bb_reason = +15, f"riding lower band ({bb_pct}) — coiled"
        elif pos <= 0.20: bb_score, bb_reason =  +5, f"near lower band ({bb_pct})"
        else:             bb_score, bb_reason =   0, f"mid-band ({bb_pct})"

    # ── 5. Custom signals — 25 pts max (Accumulation, Squeeze, Squeeze Fired)
    custom = 0
    _csigs: list[str] = []
    if last.get("Squeeze_Fired_Bullish", False):  custom += 25; _csigs.append("Squeeze Fired Bullish")
    elif last.get("Signal_Accumulation", False):   custom += 18; _csigs.append("Accumulation")
    elif last.get("Signal_Squeeze", False):        custom +=  8; _csigs.append("In Squeeze")
    if last.get("Squeeze_Fired_Bearish", False):  custom -= 25; _csigs.append("Squeeze Fired Bearish")
    if last.get("Exit_MACD_Lead", False):          custom -= 12; _csigs.append("Exit MACD Lead")
    custom = max(-25, min(25, custom))
    custom_reason = " · ".join(_csigs) if _csigs else "no active signal"

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
            "trend":      {"value": trend_score, "max": 25, "label": trend_label,  "reason": trend_reason},
            "momentum":   {"value": macd_score,  "max": 20, "label": macd_label,   "reason": macd_reason},
            "rsi":        {"value": rsi_score,   "max": 15, "label": f"RSI {rsi:.0f}" if pd.notna(rsi) else "RSI —", "reason": rsi_reason},
            "volatility": {"value": bb_score,    "max": 15, "label": "BB",         "reason": bb_reason},
            "custom":     {"value": custom,      "max": 25, "label": "Custom",     "reason": custom_reason},
        },
    }


# ── Stop & Target ladder ──────────────────────────────────────────────────────

def compute_stop_targets(price: float, atr: float,
                         resistances: list[dict], supports: list[dict]) -> dict:
    """
    Stop = MAX(price - 2*ATR, nearest_support * 0.97, price * 0.01).
    Taking the MAX (i.e. tighter stop) is correct because:
      - if support is far below: ATR stop protects against noise
      - if support is right below: the support stop is more meaningful (a
        decisive break of support invalidates the setup)

    Targets ladder — T1 (partial exit) → T2 (main) → T3 (full run):

    Base ATR-multiples are 2 / 3 / 4 ATR above price.  Each target is capped
    at the FIRST resistance above the PREVIOUS target — so each target
    consumes a distinct resistance level.

    Why this matters: if R1 is very close to entry (e.g., 0.7% above price),
    a "cap-everything-at-R1" rule would collapse T1 = T2 = T3 = R1, giving
    three identical targets and a useless R:R ratio.  Instead, T1 = R1, then
    T2 looks for the next resistance above T1 (or falls back to its ATR
    target), and T3 looks for the next one above T2.  Targets stay
    meaningfully separated.
    """
    atr_stop = price - 2 * atr
    nearest_support = supports[0]["price"] if supports else None
    sup_stop = nearest_support * 0.97 if nearest_support else atr_stop

    stop = max(atr_stop, sup_stop, price * 0.01)
    risk = max(price - stop, 1e-6)

    t1_base = price + 2 * atr   # partial exit
    t2_base = price + 3 * atr   # main target
    t3_base = price + 4 * atr   # full run

    # Walk resistances above current price and assign one to each target.
    res_pool = sorted([r["price"] for r in resistances if r["price"] > price])

    def _resolve(base, prev, pool):
        """
        Find the first resistance r in pool with prev < r <= base.
        Returns (target, pool_after_consuming_used_resistances).
        Falls back to `base` if no qualifying resistance exists.
        """
        # Drop any resistances at or below the previous target — already used
        # or no longer in the way.
        i = 0
        while i < len(pool) and pool[i] <= prev:
            i += 1
        if i < len(pool) and pool[i] <= base:
            return pool[i], pool[i + 1:]
        return base, pool[i:]

    t1, res_pool = _resolve(t1_base, price, res_pool)
    t2, res_pool = _resolve(t2_base, t1,    res_pool)
    t3, _        = _resolve(t3_base, t2,    res_pool)

    # Minimum separation so targets stay visually & operationally distinct
    # when resistances are tightly clustered.
    min_sep = atr * 0.3
    t2 = max(t2, t1 + min_sep)
    t3 = max(t3, t2 + min_sep)

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
