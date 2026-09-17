"""
alert_notes.py — write down what you think tomorrow's bar will do, and let the
next scan mark it.

WHY THIS SHAPE

The value is not the note, it is that the note is FALSIFIABLE and scored by
something other than your memory. "It looks like it's about to break out" can
never be wrong. "Closes above 68.06 on volume below the last three days'
average" is either right or it isn't, and a month later you cannot quietly
remember having said something else. Hindsight reliably rewrites a vague
prediction into whatever happened; a stored, timestamped, machine-checked one
is the only kind that teaches anything.

So a note carries free text AND a list of machine-checkable claims. Each claim
is resolved independently, because the interesting finding is usually that you
are good at one thing and poor at another — levels and volume are often
predictable when direction is not.

EVERY CLAIM IS SCORED AGAINST A BASELINE

A 55% hit rate means nothing on its own. If the stock closes green 54% of the
time, predicting green is worth almost exactly nothing, and a scorecard that
prints "55% correct" without saying so is flattering you. Each claim type has a
naive baseline computed from the stock's own recent history, and the scorecard
reports the EDGE over it. That number can be negative, and it should be allowed
to be.

RESOLUTION ONLY ON A GENUINELY NEW SESSION

A note written after Friday's close is not wrong on Saturday. Resolution needs
a bar dated strictly after the session the note was written against, so a note
written Friday resolves on Monday's bar and stays pending over the weekend and
through holidays, with no calendar logic needed — the absence of a bar is the
absence of a session.

Supabase migration lives in supabase/migrations/20260917_alert_notes.sql.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

#: Claim kinds, and what each needs. Everything here must be decidable from one
#: future bar plus the bars before it — a claim nothing can settle is a wish.
KINDS = {
    "direction_up": "收阳（收盘 > 开盘）",
    "direction_down": "收阴（收盘 < 开盘）",
    "close_above": "收盘价高于",
    "close_below": "收盘价低于",
    "close_between": "收盘价介于",
    "change_between": "涨跌幅介于",
    "volume_below_avg": "成交量低于前N日均量",
    "volume_above_avg": "成交量高于前N日均量",
}

DEFAULT_LOOKBACK = 3


def ensure_table() -> None:
    """Idempotent SQLite migration. Supabase is migrated by the SQL file."""
    from db_config import USE_SQLITE
    if not USE_SQLITE:
        return

    import sqlite3
    from db_config import DBNAME
    with sqlite3.connect(DBNAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alert_notes (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL,
                ticker        TEXT    NOT NULL,
                scan_date     TEXT    NOT NULL,
                note          TEXT,
                predictions   TEXT    NOT NULL,
                created_at    TEXT    NOT NULL,
                resolved_date TEXT,
                resolved_at   TEXT,
                outcome       TEXT
            )
        """)
        conn.commit()


# ── claims ───────────────────────────────────────────────────────────────────
def normalise(pred: dict) -> dict:
    """
    One claim, validated into the only shape the resolver understands.

    Raises LookupError on anything it could not later decide — better to refuse
    a claim at the moment of writing than to store one that can never resolve.
    """
    kind = str(pred.get("kind") or "")
    if kind not in KINDS:
        raise LookupError(f"未知的预测类型：{kind}")

    out = {"kind": kind, "label": str(pred.get("label") or "").strip()[:80]}

    def num(key, required=True):
        v = pred.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            if required:
                raise LookupError(f"{KINDS[kind]} 需要数值 {key}")
            return None

    if kind in ("close_above", "close_below"):
        out["value"] = num("value")
    elif kind in ("close_between", "change_between"):
        lo, hi = num("value"), num("value2")
        # Accepting them either way round costs nothing and removes a whole
        # class of "why did my prediction fail" that is really a typo.
        out["value"], out["value2"] = min(lo, hi), max(lo, hi)
    elif kind in ("volume_below_avg", "volume_above_avg"):
        lb = pred.get("lookback") or DEFAULT_LOOKBACK
        try:
            lb = int(lb)
        except (TypeError, ValueError):
            lb = DEFAULT_LOOKBACK
        out["lookback"] = max(1, min(lb, 20))
    return out


def resolve_claim(claim: dict, bar: dict, prior: pd.DataFrame) -> dict:
    """
    Decide one claim against the next bar. `prior` holds the bars before it.

    Returns the claim with `hit`, `actual` and `baseline` filled in. Baseline is
    how often this claim would have been TRUE over the prior window without any
    skill — the thing the hit has to beat to mean anything.
    """
    kind = claim["kind"]
    close, open_, vol = bar["close"], bar["open"], bar["volume"]
    out = dict(claim)

    if kind == "direction_up":
        out["hit"] = close > open_
        out["actual"] = f"收{'阳' if close > open_ else '阴'} {open_:.2f}→{close:.2f}"
        out["baseline"] = _rate(prior, lambda d: d["Close"] > d["Open"])
    elif kind == "direction_down":
        out["hit"] = close < open_
        out["actual"] = f"收{'阴' if close < open_ else '阳'} {open_:.2f}→{close:.2f}"
        out["baseline"] = _rate(prior, lambda d: d["Close"] < d["Open"])
    elif kind == "close_above":
        out["hit"] = close > claim["value"]
        out["actual"] = f"收盘 {close:.2f}"
        out["baseline"] = _rate(prior, lambda d: d["Close"] > claim["value"])
    elif kind == "close_below":
        out["hit"] = close < claim["value"]
        out["actual"] = f"收盘 {close:.2f}"
        out["baseline"] = _rate(prior, lambda d: d["Close"] < claim["value"])
    elif kind == "close_between":
        out["hit"] = claim["value"] <= close <= claim["value2"]
        out["actual"] = f"收盘 {close:.2f}"
        out["baseline"] = _rate(
            prior, lambda d: (d["Close"] >= claim["value"]) & (d["Close"] <= claim["value2"]))
    elif kind == "change_between":
        prev = bar.get("prev_close")
        chg = (close / prev - 1) * 100 if prev else None
        out["hit"] = chg is not None and claim["value"] <= chg <= claim["value2"]
        out["actual"] = "涨跌 —" if chg is None else f"涨跌 {chg:+.2f}%"
        pct = prior["Close"].pct_change() * 100
        out["baseline"] = _rate_series(
            (pct >= claim["value"]) & (pct <= claim["value2"]))
    elif kind in ("volume_below_avg", "volume_above_avg"):
        lb = claim.get("lookback", DEFAULT_LOOKBACK)
        avg = float(prior["Volume"].tail(lb).mean()) if len(prior) >= 1 else None
        if avg is None or avg <= 0:
            out["hit"] = None
            out["actual"] = "无成交量基准"
            out["baseline"] = None
        else:
            below = vol < avg
            out["hit"] = below if kind == "volume_below_avg" else not below
            out["actual"] = f"成交量 {vol/avg:.2f}× 前{lb}日均量"
            rolling = prior["Volume"].rolling(lb).mean().shift(1)
            cmp = prior["Volume"] < rolling
            out["baseline"] = _rate_series(cmp if kind == "volume_below_avg" else ~cmp)
    else:                                                     # pragma: no cover
        out["hit"] = None
        out["actual"] = ""
        out["baseline"] = None

    if isinstance(out.get("hit"), (bool,)):
        out["hit"] = bool(out["hit"])
    return out


def _rate(prior: pd.DataFrame, fn) -> float | None:
    if prior is None or prior.empty:
        return None
    try:
        return _rate_series(fn(prior))
    except Exception:
        return None


def _rate_series(mask) -> float | None:
    try:
        m = pd.Series(mask).dropna()
        return round(float(m.mean()) * 100, 1) if len(m) else None
    except Exception:
        return None


def resolve_note(note: dict, frame: pd.DataFrame) -> dict | None:
    """
    Settle a note if — and only if — a session later than it exists.

    `frame` is the stock's OHLCV. Returns the resolved note, or None when there
    is still no newer bar, which is exactly what a weekend or a holiday looks
    like from here: no calendar, just an absent row.
    """
    if frame is None or frame.empty:
        return None
    idx = pd.DatetimeIndex(frame.index)
    after = idx[idx > pd.Timestamp(note["scan_date"])]
    if len(after) == 0:
        return None

    day = after[0]
    row = frame.loc[day]
    pos = idx.get_loc(day)
    prior = frame.iloc[max(0, pos - 60):pos]
    prev_close = float(frame["Close"].iloc[pos - 1]) if pos > 0 else None

    bar = {"date": day.strftime("%Y-%m-%d"),
           "open": float(row["Open"]), "high": float(row["High"]),
           "low": float(row["Low"]), "close": float(row["Close"]),
           "volume": float(row["Volume"]), "prev_close": prev_close}

    claims = [resolve_claim(c, bar, prior) for c in note.get("predictions", [])]
    decided = [c for c in claims if c.get("hit") is not None]
    hits = sum(1 for c in decided if c["hit"])

    return {
        **note,
        "resolved_date": bar["date"],
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "outcome": {
            "bar": bar,
            "claims": claims,
            "hits": hits,
            "decided": len(decided),
            "score_pct": round(hits / len(decided) * 100, 1) if decided else None,
        },
    }


# ── scorecard ────────────────────────────────────────────────────────────────
def scorecard(notes: list[dict]) -> dict:
    """
    Hit rate per claim type, against the naive baseline for each.

    `edge_pp` is the number that matters: percentage points above what
    predicting the same thing blindly would have scored. `n` matters just as
    much — at ten resolved claims a 20-point edge is noise, and the UI says so
    rather than congratulating you.
    """
    by_kind: dict[str, dict] = {}
    total_hits = total = 0

    for n in notes:
        for c in ((n.get("outcome") or {}).get("claims") or []):
            if c.get("hit") is None:
                continue
            k = c["kind"]
            e = by_kind.setdefault(k, {"kind": k, "label": KINDS.get(k, k),
                                       "n": 0, "hits": 0, "baseline_sum": 0.0,
                                       "baseline_n": 0})
            e["n"] += 1
            e["hits"] += 1 if c["hit"] else 0
            if c.get("baseline") is not None:
                e["baseline_sum"] += float(c["baseline"])
                e["baseline_n"] += 1
            total += 1
            total_hits += 1 if c["hit"] else 0

    rows = []
    for e in by_kind.values():
        rate = e["hits"] / e["n"] * 100
        base = (e["baseline_sum"] / e["baseline_n"]) if e["baseline_n"] else None
        rows.append({
            "kind": e["kind"], "label": e["label"], "n": e["n"], "hits": e["hits"],
            "rate_pct": round(rate, 1),
            "baseline_pct": round(base, 1) if base is not None else None,
            "edge_pp": round(rate - base, 1) if base is not None else None,
        })
    rows.sort(key=lambda r: -r["n"])

    return {
        "rows": rows,
        "total": total,
        "hits": total_hits,
        "rate_pct": round(total_hits / total * 100, 1) if total else None,
        # Below this, the scorecard is describing coin flips. Stated as data so
        # the UI does not have to hardcode a judgement it might get wrong.
        "meaningful_at": 30,
    }


# ── storage ──────────────────────────────────────────────────────────────────
def _db():
    import data_manager
    return data_manager.db


def create(user_id: int, ticker: str, scan_date: str, note: str,
           predictions: list[dict]) -> dict:
    claims = [normalise(p) for p in predictions]
    row = {
        "user_id": int(user_id), "ticker": str(ticker),
        "scan_date": str(scan_date)[:10], "note": (note or "")[:2000],
        "predictions": json.dumps(claims, ensure_ascii=False),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resolved_date": None, "resolved_at": None, "outcome": None,
    }
    ensure_table()
    _db().insert_records("alert_notes", [row])
    return {**row, "predictions": claims}


def _hydrate(r: dict) -> dict:
    def load(v):
        if isinstance(v, (dict, list)) or v is None:
            return v
        try:
            return json.loads(v)
        except Exception:
            return None
    return {
        "id": r.get("id"), "ticker": str(r.get("ticker")),
        "scan_date": str(r.get("scan_date")), "note": r.get("note") or "",
        "predictions": load(r.get("predictions")) or [],
        "created_at": r.get("created_at"),
        "resolved_date": r.get("resolved_date"),
        "resolved_at": r.get("resolved_at"),
        "outcome": load(r.get("outcome")),
    }


def list_notes(user_id: int, ticker: str | None = None) -> list[dict]:
    ensure_table()
    filters = {"user_id": int(user_id)}
    if ticker:
        filters["ticker"] = str(ticker)
    try:
        df = _db().read_table("alert_notes", filters=filters, limit=500)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    notes = [_hydrate(r) for r in df.to_dict("records")]
    notes.sort(key=lambda n: (n["scan_date"], str(n.get("created_at"))), reverse=True)
    return notes


def save_resolution(note_id, resolved: dict) -> None:
    _db().update_records(
        "alert_notes", {"id": note_id},
        {"resolved_date": resolved["resolved_date"],
         "resolved_at": resolved["resolved_at"],
         "outcome": json.dumps(resolved["outcome"], ensure_ascii=False)})


def delete(user_id: int, note_id) -> None:
    _db().delete_records("alert_notes", {"id": note_id, "user_id": int(user_id)})
