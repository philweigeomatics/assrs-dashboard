"""
rotation.py — when this one moves, does that one follow, and by how much?

This replaces a correlation-based reading of lead-lag, which was wrong in two
ways that matter for finding where money goes next.

WRONG 1: A CORRELATION HAS NO SIGN PREFERENCE. A negative correlation at lag
+3 was drawn as a relationship of equal standing — "A moved first, B went the
other way". That is not a lead. Money leaving A and arriving somewhere else is
a different phenomenon from a move propagating from A to B, and mixing the two
under one arrow makes the output unreadable. Here the measurement is
directional by construction: a follower that moves AGAINST the leader scores
below zero and is reported as "did not follow", never as an inverse lead.

WRONG 2: A CORRELATION HAS NO SCALE. If the leader drops 5% and the follower
drops 1% three days later, correlation counts that as a strong relationship
when the follower's ordinary day is ±3% — a 1% move is it doing nothing.
Everything here is measured in the FOLLOWER'S OWN units: how many of its own
standard deviations did it move, relative to what it normally does. A 1% move
for a 3% stock is 0.33σ, and 0.33σ is not a response.

SO THE QUESTION IS ASKED THE WAY A PERSON ASKS IT

    On the days A made a real move — big for A, not big in the abstract —
    how far did B move in the SAME direction over the next few days,
    measured against what a normal day for B looks like?

Each such day is an event with a date, and the events are the output. A number
averaged over two years tells you nothing about whether the behaviour is still
there; eleven dated instances let you see for yourself that four of them are
from last spring and none since.

CAUSAL BY CONSTRUCTION

The volatility each move is measured against is TRAILING — the 60 sessions
before the day, never including it. Normalising by full-sample volatility
would let a stock's later behaviour decide whether its earlier move counted
as large, which quietly turns hindsight into signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Sessions of trailing volatility each move is measured against.
VOL_WINDOW = 60

#: How large a move has to be, in the leader's own sigma, to count as an
#: event worth following. Below about 1.5 the "events" are ordinary days and
#: the question stops meaning anything.
EVENT_Z = 1.5

#: How far the follower must move, in its own sigma and in the same
#: direction, for that event to count as followed. Half a sigma is modest on
#: purpose: the point is to separate a response from nothing, not to find
#: large ones.
FOLLOW_Z = 0.5

#: Trading days after the event that the follower is given to respond.
MAX_LAG = 5

#: Fewer events than this and no average is worth printing.
MIN_EVENTS = 8

#: Rotations used for the null. Each destroys the alignment between the two
#: stocks while keeping each one's own behaviour intact.
ROTATIONS = 8


def zscores(returns: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """
    Daily returns in units of the stock's own recent volatility.

    Trailing and shifted: day t is scored against the `window` sessions
    BEFORE it. This is the whole answer to "a 1% move means different things
    for different stocks", and the shift is what keeps it honest — a
    full-sample sigma would let August decide whether March was a big day.
    """
    sd = returns.rolling(window).std().shift(1)
    return (returns / sd.replace(0.0, np.nan)).rename(returns.name)


def events(z_lead: pd.Series, threshold: float = EVENT_Z) -> pd.DatetimeIndex:
    """Days the leader made a move that was large for it."""
    hit = z_lead.abs() >= threshold
    return z_lead.index[hit.fillna(False)]


def follow_through(lead: pd.Series, lag: pd.Series, *,
                   window: int = VOL_WINDOW, threshold: float = EVENT_Z,
                   maxlag: int = MAX_LAG,
                   follow: float = FOLLOW_Z) -> dict:
    """
    Every event, and what the follower did in the same direction afterwards.

    `resp[k]` is the follower's move on day t+k in its own sigma, multiplied
    by the sign of the leader's move — so positive always means "went the
    same way" regardless of whether the pair was falling or rising together.
    A joint sell-off scores exactly like a joint rally, which is the point:
    both are the money moving, and only the direction of the pair relative to
    ITSELF carries information.
    """
    df = pd.concat([lead.rename("a"), lag.rename("b")], axis=1).dropna()
    if len(df) < window + maxlag + 10:
        raise LookupError(f"可用历史仅 {len(df)} 个交易日，不足以测量")

    za = zscores(df["a"], window)
    zb = zscores(df["b"], window)
    idx = list(df.index)
    pos = {d: i for i, d in enumerate(idx)}

    rows = []
    for day in events(za, threshold):
        i = pos[day]
        if i + maxlag >= len(idx):
            continue                      # no room left to respond
        direction = 1.0 if df["a"].iloc[i] > 0 else -1.0
        resp, cum = [], []
        running = 0.0
        for k in range(0, maxlag + 1):
            v = zb.iloc[i + k]
            aligned = float(v) * direction if v == v else float("nan")
            resp.append(aligned)
            if k >= 1 and aligned == aligned:
                running += aligned
            cum.append(running if k >= 1 else 0.0)
        rows.append({
            "date": str(day.date()),
            "a_ret": round(float(df["a"].iloc[i]) * 100, 2),
            "a_z": round(float(za.iloc[i]), 2),
            "dir": "up" if direction > 0 else "down",
            "resp": [None if v != v else round(v, 2) for v in resp],
            "cum": [None if v != v else round(v, 2) for v in cum],
            # The follower's own move on the best single day, as a percentage,
            # so the table can show something a person recognises.
            "b_ret": [round(float(df["b"].iloc[i + k]) * 100, 2)
                      if i + k < len(idx) else None
                      for k in range(0, maxlag + 1)],
        })

    return {"events": rows, "window": window, "threshold": threshold,
            "maxlag": maxlag, "follow": follow,
            "sessions": len(df),
            "from": str(df.index[0].date()), "to": str(df.index[-1].date())}


def summarise(fired: dict) -> list[dict]:
    """
    Per lag: how many events, how far the follower went, how often it went.

    `mean` is in the follower's sigma. `hit` is the share of events where it
    moved at least `follow` sigma the same way — a coin-flip follower sits
    near the share you would get from its own distribution, which is why the
    null matters more than the number.
    """
    rows = fired["events"]
    out = []
    for k in range(0, fired["maxlag"] + 1):
        vals = [r["resp"][k] for r in rows if r["resp"][k] is not None]
        cums = [r["cum"][k] for r in rows if r["cum"][k] is not None]
        n = len(vals)
        out.append({
            "lag": k,
            "n": n,
            "mean": round(float(np.mean(vals)), 3) if n else None,
            "median": round(float(np.median(vals)), 3) if n else None,
            "hit": round(sum(1 for v in vals if v >= fired["follow"]) / n, 3)
            if n else None,
            "cum": round(float(np.mean(cums)), 3) if cums and k >= 1 else None,
        })
    return out


def nulls(lead: pd.Series, lag: pd.Series, *, rotations: int = ROTATIONS,
          **kw) -> list[dict]:
    """
    The same measurement with the two series slid out of alignment.

    Rotation keeps each stock's volatility clustering and its own event days;
    only the correspondence between them is destroyed. So this is what the
    follower's average response looks like when it is not responding.
    """
    per_lag: dict[int, list[float]] = {}
    hits: dict[int, list[float]] = {}
    n = len(lag)
    for i in range(max(1, rotations)):
        by = int(n * (0.12 + 0.76 * (i / max(1, rotations - 1)))) or 1
        rolled = pd.Series(np.roll(lag.to_numpy(), by), index=lag.index,
                           name=lag.name)
        try:
            s = summarise(follow_through(lead, rolled, **kw))
        except LookupError:
            continue
        for row in s:
            if row["mean"] is not None:
                per_lag.setdefault(row["lag"], []).append(row["mean"])
                hits.setdefault(row["lag"], []).append(row["hit"])

    return [{
        "lag": k,
        "mean": round(float(np.mean(v)), 3),
        "mean_hi": round(float(np.percentile(v, 95)), 3),
        "hit": round(float(np.mean(hits[k])), 3),
        "hit_hi": round(float(np.percentile(hits[k], 95)), 3),
        "rotations": len(v),
    } for k, v in sorted(per_lag.items())]


def verdict(rows: list[dict], null: list[dict], *,
            min_events: int = MIN_EVENTS,
            min_response: float = FOLLOW_Z) -> dict:
    """
    The lag where the follower actually followed, or nothing.

    "Nothing" is the common answer and is returned as such. Three conditions,
    all required, because any one of them alone fires on noise:

      1. the average response is a REAL move — at least `min_response` of the
         follower's own sigma. Measured on random pairs the averages wander
         between -0.15 and +0.25 sigma, while a genuine echo sits above 0.5
         and same-day co-movement reaches 1.8. Without this, a rotation that
         happened to land low is enough to crown a lag.
      2. it beats the rotations at that lag.
      3. it happens OFTEN — a hit rate above both the rotations' and 40% —
         so one enormous day cannot carry the average on its own.
    """
    by_lag = {n["lag"]: n for n in null}
    best, margin = None, 0.0
    for r in rows:
        if r["lag"] == 0 or r["n"] < min_events or r["mean"] is None:
            continue
        ref = by_lag.get(r["lag"])
        if ref is None:
            continue
        if r["mean"] < min_response:
            continue
        if r["hit"] is None or r["hit"] <= max(ref["hit_hi"], 0.4):
            continue
        # Condition 2 lives here rather than as its own guard: `margin`
        # starts at zero, so requiring a bigger gap than the best so far
        # already requires a positive one, and a separate check above would
        # be unreachable code pretending to be a safeguard.
        gap = r["mean"] - ref["mean_hi"]
        if gap > margin:
            best, margin = r["lag"], gap

    same_day = next((r for r in rows if r["lag"] == 0), None)
    return {
        "best_lag": best,
        "margin": round(margin, 3) if best else None,
        # n_events, not events: the payload spreads this dict alongside the
        # event LIST, and a shared key silently replaced the list with a
        # count.
        "n_events": rows[0]["n"] if rows else 0,
        # Reported because it is usually the true answer: the whole response
        # lands on the day itself and there is no gap to act in.
        "same_day": same_day["mean"] if same_day else None,
        "enough": bool(rows and rows[0]["n"] >= min_events),
    }
