"""
price_zones.py — 价值区间: the price ranges the market kept coming back to.

The idea
--------
A stock does not spend its time evenly across its price history. Some ranges
it passes through in a week; others it sits in for months, leaves, and comes
back to. Those are the prices buyers and sellers kept AGREEING on — the
market's consensus value for the company, in the sense Market Profile calls
"accepted" price.

That is not intrinsic value, and the module is careful not to pretend it is:

* It is the consensus of a particular period. When earnings change, the same
  price means a different valuation — a range that was ¥40 at 30× earnings is
  ¥40 at 15× after profits double. So each zone carries the median PE the
  market paid while it was there, and what that PE is worth on today's
  earnings. The PE-adjusted price is often the more honest "value range".
* A stock that trended for five years has no consensus — it was never at any
  price long. The profile says so (no zone holds much of the time) instead of
  inventing ranges.

How
---
1. Time at price (TPO). Each session counts as one day spread evenly over the
   price bins its high–low range touched, on a log scale so a ¥10 and a ¥1000
   stock bin alike. Volume at price is built the same way, weighted by the
   day's volume, and drawn alongside for reference.
2. Zones are the peaks of the smoothed time profile: each peak with real
   prominence, bounded where the density falls to half the peak's prominence
   (scipy's peak width). A plateau splits into peaks only where there is a
   genuine valley between them. A peak must also stand ABOVE_MEAN over the
   density an even spread would give, and be narrower than MAX_WIDTH of the
   range — otherwise a steady trend's flat profile reads as one zone the
   size of the whole range.
3. Periods. For each zone, the runs of sessions that CLOSED inside it, with
   short excursions (≤ GAP_SESSIONS) bridged so one stay with a two-day spike
   out is one stay, and only stays of ≥ MIN_STAY sessions kept.

Window: the last five years, or from listing when the stock is younger. The
first IPO_SKIP sessions after a listing are price discovery — A-share IPOs
run without limits for five days and then often limit-up for weeks — so they
are drawn but left out of the profile.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

LOOKBACK_YEARS = 5
MIN_BARS = 120          # half a year — less than this has no "usual" price yet
IPO_SKIP = 20           # sessions after listing left out of the profile
LISTED_SLACK_DAYS = 30  # first bar this long after the asked start = listed in-window
MAX_BINS = 150
MIN_STEP = 0.01         # 1% bins at the finest
SMOOTH_SIGMA = 1.5      # bins
PROMINENCE = 0.15       # of the tallest peak — below this a bump is noise
ABOVE_MEAN = 1.3        # a zone's peak must stand this far above the even spread
MAX_WIDTH = 0.6         # of the whole log range — wider is the range, not a zone
MIN_SHARE = 0.05        # a zone must hold at least 5% of the sessions
MAX_ZONES = 5
GAP_SESSIONS = 5        # an excursion this short does not end a stay
MIN_STAY = 10           # sessions; shorter is passing through, not staying
VALUE_AREA = 0.70       # the classic Market Profile value area
RECENT_SESSIONS = 242   # "the last year"


def start_date(today: date, years: int = LOOKBACK_YEARS) -> date:
    return today - timedelta(days=round(365.25 * years))


def _num(v, nd: int = 4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def _smooth(x: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    r = int(math.ceil(4 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="constant"), k, mode="valid")


def profile(df: pd.DataFrame) -> dict:
    """
    Time and volume at price over `df`'s sessions, on log-spaced bins.

    Returns base (log price of the lowest edge), step (log width), and the
    per-bin arrays `time` (sessions) and `volume` (share units), plus their
    centres in price.
    """
    lo = np.log(df["Low"].to_numpy(dtype=float))
    hi = np.log(df["High"].to_numpy(dtype=float))
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    span = float(hi.max() - lo.min())
    amp = float(np.median(hi - lo)) if len(hi) else 0.0
    step = max(MIN_STEP, 0.5 * amp, span / MAX_BINS if span > 0 else MIN_STEP)
    base = float(lo.min())
    n = max(1, int(math.floor(span / step)) + 1)

    a = np.clip(np.floor((lo - base) / step).astype(int), 0, n - 1)
    b = np.clip(np.floor((hi - base) / step).astype(int), 0, n - 1)
    k = (b - a + 1).astype(float)
    vol = df["Volume"].to_numpy(dtype=float) if "Volume" in df.columns else np.ones(len(df))
    vol = np.nan_to_num(vol, nan=0.0)

    # Difference arrays: +w at a, -w after b, then a running sum spreads each
    # session's weight evenly over every bin it touched.
    t = np.zeros(n + 1)
    v = np.zeros(n + 1)
    np.add.at(t, a, 1.0 / k)
    np.add.at(t, b + 1, -1.0 / k)
    np.add.at(v, a, vol / k)
    np.add.at(v, b + 1, -vol / k)
    t = np.cumsum(t)[:n]
    v = np.cumsum(v)[:n]
    centres = np.exp(base + (np.arange(n) + 0.5) * step)
    return {"base": base, "step": step, "time": t, "volume": v, "centres": centres}


def _price_at(base: float, step: float, x: float) -> float:
    """Price at fractional bin position x (bin i's centre is x = i)."""
    return float(math.exp(base + (x + 0.5) * step))


def value_area(time: np.ndarray, share: float = VALUE_AREA) -> tuple[int, int, int]:
    """
    (lo_bin, hi_bin, poc_bin): grow from the most-visited bin, each step
    taking whichever neighbour holds more time, until `share` is covered.
    """
    poc = int(np.argmax(time))
    total = float(time.sum())
    lo = hi = poc
    got = float(time[poc])
    while got < share * total and (lo > 0 or hi < len(time) - 1):
        left = time[lo - 1] if lo > 0 else -1.0
        right = time[hi + 1] if hi < len(time) - 1 else -1.0
        if right >= left:
            hi += 1
            got += float(time[hi])
        else:
            lo -= 1
            got += float(time[lo])
    return lo, hi, poc


def find_zones(time: np.ndarray, base: float, step: float) -> list[dict]:
    """Price bands around the prominent peaks of the smoothed time profile."""
    from scipy.signal import find_peaks, peak_widths

    s = _smooth(time)
    if s.max() <= 0:
        return []
    # Zero padding so a peak at the very top or bottom of the range counts —
    # the all-time high or low is often exactly where a stock sat longest.
    padded = np.concatenate([[0.0], s, [0.0]])
    peaks, props = find_peaks(padded, prominence=PROMINENCE * float(s.max()))
    if len(peaks) == 0:
        return []
    _, _, left, right = peak_widths(padded, peaks, rel_height=0.5,
                                    prominence_data=(props["prominences"],
                                                     props["left_bases"],
                                                     props["right_bases"]))
    # A trend spends about the same time at every price, so its profile is
    # flat and its one "peak" is the whole range at half height. A zone has to
    # stand out from what an even spread over the range would give, and be
    # narrower than most of the range.
    even = float(s.sum()) / len(s)
    zones = []
    for p, l, r in zip(peaks, left, right):
        l, r = max(l - 1, 0.0), min(r - 1, len(s) - 1.0)     # undo the pad
        if s[p - 1] < ABOVE_MEAN * even or (r - l + 1) > MAX_WIDTH * len(s):
            continue
        zones.append({"peak": _price_at(base, step, p - 1),
                      "lo": _price_at(base, step, l - 0.5),
                      "hi": _price_at(base, step, r + 0.5),
                      "height": float(s[p - 1])})
    # Overlaps: the taller keeps its band; the other is dropped.
    zones.sort(key=lambda z: -z["height"])
    kept: list[dict] = []
    for z in zones:
        if all(z["hi"] <= k["lo"] or z["lo"] >= k["hi"] for k in kept):
            kept.append(z)
    return kept


def stays(inside: np.ndarray, gap: int = GAP_SESSIONS, min_len: int = MIN_STAY) -> list[tuple[int, int]]:
    """
    (first, last) bar positions of each stay: runs of True, with gaps of at
    most `gap` sessions bridged, kept when they span ≥ min_len sessions.
    """
    idx = np.flatnonzero(inside)
    if len(idx) == 0:
        return []
    runs, start, prev = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev - 1 > gap:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return [(a, b) for a, b in runs if b - a + 1 >= min_len]


def build(df: pd.DataFrame, *, asked_from: date, pe: pd.Series | None = None) -> dict:
    """
    The 价值区间 payload for one stock's adjusted daily bars `df`
    (Open/High/Low/Close/Volume, date-indexed, oldest first).
    """
    d = df.sort_index()
    d = d[["High", "Low", "Close"] + (["Volume"] if "Volume" in d.columns else [])]
    d = d.dropna(subset=["High", "Low", "Close"])
    d = d[(d["Low"] > 0) & (d["High"] > 0) & (d["Close"] > 0)]
    if len(d) < 2:
        raise LookupError("no price history")

    first = d.index[0].date()
    listed_late = first > asked_from + timedelta(days=LISTED_SLACK_DAYS)
    skip = IPO_SKIP if listed_late else 0
    body = d.iloc[skip:]
    if len(body) < MIN_BARS:
        raise LookupError(
            f"只有 {len(body)} 个交易日可用 — 至少要 {MIN_BARS} 个才谈得上“常待的价格”")

    prof = profile(body)
    base, step, time = prof["base"], prof["step"], prof["time"]
    total = len(body)
    lo_b, hi_b, poc_b = value_area(time)
    va = {"lo": _price_at(base, step, lo_b - 0.5), "hi": _price_at(base, step, hi_b + 0.5),
          "poc": float(prof["centres"][poc_b]),
          "share": float(time[lo_b:hi_b + 1].sum() / time.sum())}

    close = body["Close"].to_numpy(dtype=float)
    pe_al = None
    if pe is not None and not pe.empty:
        p = pd.to_numeric(pe, errors="coerce")
        p = p[p > 0]                                  # a loss has no P/E
        if not p.empty:
            pe_al = p.reindex(body.index).to_numpy(dtype=float)

    last_close = float(close[-1])
    pe_now = None
    if pe_al is not None:
        tail = pd.Series(pe_al).dropna()
        # Only "now" if the last reading is recent — a stock that went
        # loss-making has no current P/E, and a stale one would mislead.
        if not tail.empty and tail.index[-1] >= len(pe_al) - 5:
            pe_now = float(tail.iloc[-1])

    zones = []
    for z in find_zones(time, base, step):
        inside = (close >= z["lo"]) & (close <= z["hi"])
        days = int(inside.sum())
        if days < MIN_SHARE * total:
            continue
        runs = stays(inside)
        recent = inside[-RECENT_SESSIONS:]
        zpe = None
        if pe_al is not None:
            vals = pe_al[inside]
            vals = vals[~np.isnan(vals)]
            if len(vals) >= 0.5 * days:
                zpe = float(np.median(vals))
        periods = [{
            "from": body.index[a].strftime("%Y-%m-%d"),
            "to": body.index[b].strftime("%Y-%m-%d"),
            "sessions": int(b - a + 1),
            "days": int(inside[a:b + 1].sum()),
            "avg": _num(close[a:b + 1].mean(), 3),
        } for a, b in runs]
        zones.append({
            "lo": _num(z["lo"], 3), "hi": _num(z["hi"], 3), "peak": _num(z["peak"], 3),
            "days": days, "share": _num(days / total),
            "recent_days": int(recent.sum()), "recent_share": _num(recent.mean()),
            "visits": len(periods),
            "longest": max((p["sessions"] for p in periods), default=0),
            "first": periods[0]["from"] if periods else None,
            "last": periods[-1]["to"] if periods else None,
            "pe": _num(zpe, 2),
            # The zone's PE on today's earnings: what the stock would cost if
            # the market paid what it paid then. Price × (PE_then / PE_now)
            # because price / PE is earnings per share.
            "pe_price": _num(last_close * zpe / pe_now, 3) if zpe and pe_now else None,
            "periods": periods,
        })
        if len(zones) == MAX_ZONES:
            break

    # Highest price first: the table reads like the chart's y-axis.
    zones.sort(key=lambda z: -(z["peak"] or 0))
    for i, z in enumerate(zones):
        z["id"] = i
    rank = sorted(range(len(zones)), key=lambda i: -zones[i]["days"])
    for r, i in enumerate(rank):
        zones[i]["rank"] = r + 1

    where = {"zone": None, "above": None, "below": None}
    for z in zones:
        if z["lo"] <= last_close <= z["hi"]:
            where["zone"] = z["id"]
    if where["zone"] is None:
        under = [z for z in zones if z["hi"] < last_close]
        over = [z for z in zones if z["lo"] > last_close]
        if under:
            where["above"] = max(under, key=lambda z: z["hi"])["id"]
        if over:
            where["below"] = min(over, key=lambda z: z["lo"])["id"]

    tsum = float(time.sum()) or 1.0
    vsum = float(prof["volume"].sum()) or 1.0
    return {
        "asked_from": asked_from.isoformat(),
        "start": d.index[0].strftime("%Y-%m-%d"),
        "profile_start": body.index[0].strftime("%Y-%m-%d"),
        "end": d.index[-1].strftime("%Y-%m-%d"),
        "listed_late": bool(listed_late),
        "ipo_skipped": int(skip),
        "sessions": int(total),
        "dates": [t.strftime("%Y-%m-%d") for t in d.index],
        "close": [_num(v, 3) for v in d["Close"]],
        "profile": {
            "price": [_num(v, 3) for v in prof["centres"]],
            "time": [_num(v / tsum, 5) for v in time],
            "volume": [_num(v / vsum, 5) for v in prof["volume"]],
        },
        "value_area": {k: _num(v, 3 if k != "share" else 4) for k, v in va.items()},
        # Share of sessions in the single biggest zone. Low means the stock
        # trended through its range and never settled — the zones then say
        # little, and the page says so.
        "settled": _num(max((z["share"] or 0) for z in zones) if zones else 0.0),
        "zones": zones,
        "now": {"close": _num(last_close, 3), "date": d.index[-1].strftime("%Y-%m-%d"),
                "pe": _num(pe_now, 2), **where},
    }
