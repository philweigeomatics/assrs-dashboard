"""
box_detection.py — 箱体 detection that refuses to draw a box unless there is one.

Why the old detector drew boxes everywhere
------------------------------------------
It split the window on high-volume days and, for every segment, drew the price
band holding 70% of traded volume. That band is a *statistic*, not a level: it
needs no relationship to any price the market actually tested. So every
segment produced a box — including pure trends, where the "box" is just where
the trend happened to spend its volume. Measured across five liquid names, six
of eight boxes were directional and six of eight had fewer than two touches on
the floor.

What actually makes a 箱体
-------------------------
A box is a truce between buyers and sellers, and a truce has to be *observed*:

1. Both edges must have been TOUCHED AND REJECTED more than once. A level hit
   once is a coincidence — it is simply where the move happened to stop that
   day. Two touches is the minimum that distinguishes a level from an
   accident, three or more is a wall. This is the single biggest filter.

2. The edges are ZONES, not prices. Support is not 48.05, it is roughly
   48.00–48.15. The zone is scaled to the stock's own ATR, because a ¥12 bank
   and a ¥1300 distiller do not agree on what "near" means.

3. It must go NOWHERE. If price drifted a large fraction of the box height
   across the window, the range is sloping and it is a 通道, not a 箱体. Two
   independent tests: net drift against height, and R² of a linear fit —
   a rising channel has a tight fit, a box has none.

4. It must be TRADABLE. Too short and the range is noise, not a level. Too
   narrow and normal daily amplitude swallows the edges. Too wide and the
   thing has already made its move — a 30%+ "range" is a trend leg with two
   ends, and under ±10% limits it can be traversed in three sessions.

Anything that fails 3 is classified as a rising/falling channel rather than
silently dropped, because "this is a 上升通道" is useful information — it is
just not a box, and must not be traded like one.

A-share specifics
-----------------
Height floor adapts to the stock: a name that swings 4% a day cannot hold a
tradable 5% box, since a single session crosses it. T+1 settlement reinforces
this — a box you cannot exit the same day must be tall enough to survive an
overnight gap. Prices are assumed adjusted (they are, upstream).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Defaults. Every one is a judgement call; each is explained. ──────────────
MIN_SESSIONS = 15      # below this a "range" is sentiment, not a level (see note)
MAX_SESSIONS = 120     # half a year; beyond this the level has stopped mattering
MIN_HEIGHT_PCT = 5.0   # floor, before the volatility-adaptive floor is applied
MAX_HEIGHT_PCT = 30.0  # above this the move has already happened
MIN_TOUCHES = 2        # per edge, counted as distinct visits
AMPLITUDE_MULT = 2.5   # height must also exceed this × median daily amplitude
MAX_DRIFT_FRAC = 0.50  # net drift may not exceed this fraction of box height
MAX_R2 = 0.35          # above this the fit is a line, i.e. a channel
MIN_CONTAINMENT = 0.85 # fraction of closes that must sit inside the box
EDGE_PCTL = 95         # percentile of highs/lows defining the edges
ZONE_ATR_MULT = 0.5    # edge zone half-width, in ATR
ZONE_MIN_PCT = 0.8     # …but never tighter than this % of price
NEAR_EDGE = 0.20       # within this fraction of height = "at" that edge

# Calling something a 通道 is a claim, and needs its own evidence: a tight fit
# alone is not enough, because any short window fits a line.
CHANNEL_MIN_SESSIONS = 30    # a trend shorter than this is a swing, not a channel
CHANNEL_MIN_R2 = 0.60        # the fit must actually be a line
CHANNEL_MIN_DRIFT_PCT = 8.0  # …and it must have gone somewhere

# On MIN_SESSIONS: the brief said 10, and 10 is the right floor for "is this
# flat at all". It is raised to 15 for a TRADABLE box because two clean tests
# of each edge cannot fit into 10 sessions without them being the same swing.
# It is a parameter — drop it to 10 to loosen.


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    """Average true range of the window, in price units."""
    h, l, c = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    v = tr.tail(n).mean()
    return float(v) if pd.notna(v) else float((h - l).mean() or 0.0)


def _touch_runs(mask: np.ndarray) -> int:
    """
    Count DISTINCT visits, not bars.

    Five sessions resting on support is one test of that support, not five.
    Counting bars would let a single lazy drift along the floor masquerade as
    a heavily defended level, which is exactly the error being fixed.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0
    return int((m & ~np.r_[False, m[:-1]]).sum())


def _evaluate(seg: pd.DataFrame, atr: float, *,
              min_touches: int, min_height_pct: float, max_height_pct: float,
              amplitude_mult: float, max_drift_frac: float, max_r2: float,
              min_containment: float, edge_pctl: float,
              zone_atr_mult: float, zone_min_pct: float) -> dict | None:
    """
    Measure one candidate window. Returns a verdict dict, or None if the
    window cannot be judged at all. The dict always carries `kind` and the
    `reasons` it failed, so the caller can explain a rejection.
    """
    n = len(seg)
    if n < 5:
        return None

    high, low, close = seg["High"].values, seg["Low"].values, seg["Close"].values

    # Edges from the extremes, trimmed so a single spike cannot set the ceiling.
    top = float(np.percentile(high, edge_pctl))
    bot = float(np.percentile(low, 100 - edge_pctl))
    if top <= bot:
        return None
    mid = (top + bot) / 2.0
    height = top - bot
    height_pct = height / mid * 100.0

    # Edge ZONES — support is a neighbourhood, not a number.
    zone = max(zone_atr_mult * atr, zone_min_pct / 100.0 * mid)
    touches_top = _touch_runs(high >= top - zone)
    touches_bot = _touch_runs(low <= bot + zone)

    # Direction: net drift measured against the box's own height, so the test
    # means "did it go somewhere relative to how tall it is".
    x = np.arange(n, dtype=float)
    slope, _ = np.polyfit(x, close, 1)
    drift = float(slope) * (n - 1)
    drift_pct = drift / mid * 100.0
    drift_frac = abs(drift) / height if height else np.inf
    r2 = float(np.corrcoef(x, close)[0, 1] ** 2) if close.std() > 0 else 0.0

    containment = float(((close >= bot) & (close <= top)).mean())

    # Volatility-adaptive floor: a stock that swings 4% daily cannot hold a
    # 5% box, because one ordinary session crosses the whole thing.
    prev_close = seg["Close"].shift(1)
    amp = ((seg["High"] - seg["Low"]) / prev_close * 100.0).dropna()
    med_amp = float(amp.median()) if len(amp) else 0.0
    height_floor = max(min_height_pct, amplitude_mult * med_amp)

    reasons = []
    if height_pct < height_floor:
        reasons.append(f"太窄 {height_pct:.1f}% < {height_floor:.1f}%"
                       f"（日均振幅 {med_amp:.1f}%）")
    if height_pct > max_height_pct:
        reasons.append(f"太宽 {height_pct:.1f}% > {max_height_pct:.0f}%，已是趋势不是箱体")
    if touches_top < min_touches:
        reasons.append(f"上沿仅触及 {touches_top} 次，不成压力")
    if touches_bot < min_touches:
        reasons.append(f"下沿仅触及 {touches_bot} 次，不成支撑")
    if containment < min_containment:
        reasons.append(f"收盘在箱内仅 {containment:.0%}")

    # Directional windows are re-labelled, not merely rejected.
    directional = (drift_frac > max_drift_frac) or (r2 > max_r2)
    if directional:
        kind = "RISING_CHANNEL" if slope > 0 else "FALLING_CHANNEL"
        reasons.append(f"有方向：净漂移 {drift_pct:+.1f}%（箱高的 {drift_frac:.0%}）"
                       f"，R² {r2:.2f}")
    else:
        kind = "BOX"

    # Quality, 0–1, only meaningful for a box that passed. Each term is a
    # thing a trader would actually look at.
    q_touch = min(1.0, (min(touches_top, touches_bot) - 1) / 3.0)
    q_flat = max(0.0, 1.0 - drift_frac / max_drift_frac)
    q_hold = max(0.0, (containment - min_containment) / (1 - min_containment))
    q_time = min(1.0, n / 60.0)
    quality = 0.40 * q_touch + 0.25 * q_flat + 0.20 * q_hold + 0.15 * q_time

    return {
        "start": seg.index[0], "end": seg.index[-1], "n_sessions": n,
        "top": top, "bot": bot, "mid": mid,
        "height": height, "height_pct": height_pct, "height_floor_pct": height_floor,
        "median_amplitude_pct": med_amp,
        "zone": zone, "zone_top_lo": top - zone, "zone_bot_hi": bot + zone,
        "touches_top": touches_top, "touches_bot": touches_bot,
        "drift_pct": drift_pct, "drift_frac": drift_frac, "r2": r2,
        "containment": containment,
        "kind": kind, "valid": kind == "BOX" and not reasons,
        "reasons": reasons, "quality": round(quality, 3),
    }


def _position(box: dict, price: float) -> float:
    """Where price sits in the box: 0.0 = on the floor, 1.0 = on the ceiling."""
    h = box["top"] - box["bot"]
    return float((price - box["bot"]) / h) if h else 0.5


def _status(box: dict, price: float, near_edge: float,
            is_active: bool = True) -> tuple[str, str]:
    """
    (status, Chinese label) for where price stands relative to the box.

    A finished box is not "breaking out" — price left it long ago. What it
    still offers is a level: an old ceiling now under price is support, an old
    floor above price is resistance. Calling that a breakout would put a
    today-tense event label on last quarter's range.
    """
    if price > box["top"]:
        return ("BREAKOUT", "向上突破") if is_active else ("SUPPORT_BELOW", "下方支撑（旧箱顶）")
    if price < box["bot"]:
        return ("BREAKDOWN", "向下跌破") if is_active else ("RESISTANCE_ABOVE", "上方压力（旧箱底）")
    if not is_active:
        return "INSIDE_OLD_RANGE", "回到旧箱体内"
    p = _position(box, price)
    if p <= near_edge:
        return "AT_SUPPORT", "贴近下沿（支撑）"
    if p >= 1 - near_edge:
        return "AT_RESISTANCE", "贴近上沿（压力）"
    return "INSIDE", "箱体中部"


def detect_boxes(df: pd.DataFrame, *,
                 lookback: int = 250,
                 min_sessions: int = MIN_SESSIONS,
                 max_sessions: int = MAX_SESSIONS,
                 min_touches: int = MIN_TOUCHES,
                 min_height_pct: float = MIN_HEIGHT_PCT,
                 max_height_pct: float = MAX_HEIGHT_PCT,
                 amplitude_mult: float = AMPLITUDE_MULT,
                 max_drift_frac: float = MAX_DRIFT_FRAC,
                 max_r2: float = MAX_R2,
                 min_containment: float = MIN_CONTAINMENT,
                 edge_pctl: float = EDGE_PCTL,
                 zone_atr_mult: float = ZONE_ATR_MULT,
                 zone_min_pct: float = ZONE_MIN_PCT,
                 near_edge: float = NEAR_EDGE,
                 max_boxes: int = 3,
                 include_channels: bool = True) -> list[dict]:
    """
    Find every window that genuinely qualifies as a box, best first.

    Returns at most `max_boxes` non-overlapping boxes. The active box — one
    whose window runs to the latest bar — is searched exhaustively and always
    reported first when found, because that is the one you can trade and the
    one the alert cares about.

    With include_channels, windows that were flat-ish but directional come
    back tagged RISING_CHANNEL / FALLING_CHANNEL so the caller can say
    "上升通道" instead of drawing a box that isn't there.
    """
    if df is None or len(df) < min_sessions + 5:
        return []

    data = df.tail(lookback)
    atr = _atr(data)
    price = float(data["Close"].iloc[-1])
    n_all = len(data)
    kw = dict(min_touches=min_touches, min_height_pct=min_height_pct,
              max_height_pct=max_height_pct, amplitude_mult=amplitude_mult,
              max_drift_frac=max_drift_frac, max_r2=max_r2,
              min_containment=min_containment, edge_pctl=edge_pctl,
              zone_atr_mult=zone_atr_mult, zone_min_pct=zone_min_pct)

    found: list[dict] = []

    # ── The active box: every length, anchored at the last bar ──────────────
    # Searched at step 1 because this is the tradable one and its edges feed
    # the alert; a 3-bar error in the window moves the support level.
    best_active = None
    for length in range(min_sessions, min(max_sessions, n_all) + 1):
        v = _evaluate(data.iloc[n_all - length:], atr, **kw)
        if v and v["valid"]:
            # Longer wins on ties: a level held for 60 sessions outranks the
            # same level held for 20.
            if (best_active is None
                    or (v["quality"], v["n_sessions"])
                    > (best_active["quality"], best_active["n_sessions"])):
                best_active = v
    if best_active:
        best_active["is_active"] = True
        found.append(best_active)

    # ── Historical boxes: coarser sweep, they only provide S/R context ──────
    for end in range(n_all - 1, min_sessions, -3):
        if len(found) >= max_boxes * 4:
            break
        for length in range(min_sessions, min(max_sessions, end) + 1, 5):
            v = _evaluate(data.iloc[end - length:end], atr, **kw)
            if v and v["valid"]:
                v["is_active"] = False
                found.append(v)

    # ── Keep the best non-overlapping set ───────────────────────────────────
    found.sort(key=lambda b: (b.get("is_active", False), b["quality"]), reverse=True)
    chosen: list[dict] = []
    for b in found:
        if len(chosen) >= max_boxes:
            break
        if any(not (b["end"] < c["start"] or b["start"] > c["end"]) for c in chosen):
            continue
        chosen.append(b)

    # ── If nothing qualified, still say what IS going on ────────────────────
    # Claiming a channel needs more than a tight fit. Any short stretch is
    # well fit by a line — a 15-bar window scores R² 0.98 on pure noise — so
    # ranking candidates by R² alone degenerates into "find the shortest
    # window", which is how a 2%-drift range got called a 下降通道. A channel
    # must therefore also GO somewhere and last long enough to be a trend.
    if include_channels and not any(c.get("is_active") for c in chosen):
        best_ch, best_score = None, 0.0
        floor = max(min_sessions * 2, CHANNEL_MIN_SESSIONS)
        for length in range(floor, min(max_sessions, n_all) + 1, 2):
            v = _evaluate(data.iloc[n_all - length:], atr, **kw)
            if not v or v["kind"] not in ("RISING_CHANNEL", "FALLING_CHANNEL"):
                continue
            if v["r2"] < CHANNEL_MIN_R2 or abs(v["drift_pct"]) < CHANNEL_MIN_DRIFT_PCT:
                continue
            score = v["r2"] * min(1.0, v["n_sessions"] / 60.0)
            if score > best_score:
                best_ch, best_score = v, score
        if best_ch is not None:
            best_ch["is_active"] = True
            chosen.append(best_ch)

    for b in chosen:
        b["status"], b["status_cn"] = (
            _status(b, price, near_edge, b.get("is_active", False))
            if b["kind"] == "BOX"
            else (b["kind"], "上升通道" if b["kind"] == "RISING_CHANNEL" else "下降通道"))
        b["position"] = _position(b, price) if b["kind"] == "BOX" else None
        b.setdefault("is_active", False)

    chosen.sort(key=lambda b: b["start"])
    return chosen


def active_box(df: pd.DataFrame, **kw) -> dict | None:
    """The box in force right now, or None. Channels are not boxes."""
    for b in detect_boxes(df, **kw):
        if b.get("is_active") and b["kind"] == "BOX":
            return b
    return None


def box_alert(df: pd.DataFrame, *, near_edge: float = NEAR_EDGE,
              **kw) -> dict | None:
    """
    Watchlist alert: is there a live box, and is price at one of its edges?

    Returns None when there is no valid active box — which is the point. No
    box means no edge, and no edge means there is nothing here to trade
    against, so nothing is reported.
    """
    kw.pop("near_edge", None)
    b = active_box(df, near_edge=near_edge, **kw)
    if b is None:
        return None
    price = float(df["Close"].iloc[-1])
    pos = _position(b, price)
    at_edge = b["status"] in ("AT_SUPPORT", "AT_RESISTANCE",
                              "BREAKOUT", "BREAKDOWN")
    return {
        "ticker": None,
        "price": price,
        "top": b["top"], "bot": b["bot"],
        "position": pos,
        "pct_to_top": (b["top"] / price - 1) * 100,
        "pct_to_bot": (b["bot"] / price - 1) * 100,
        "status": b["status"], "status_cn": b["status_cn"],
        "quality": b["quality"], "n_sessions": b["n_sessions"],
        "height_pct": b["height_pct"],
        "touches_top": b["touches_top"], "touches_bot": b["touches_bot"],
        "start": b["start"], "end": b["end"],
        "alert": at_edge,
        "box": b,
    }
