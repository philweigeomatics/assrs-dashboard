"""
宏观数据 → 指数 — which session each release lands on, and how big that session was.

The rule
--------
A release can only move an index in the first session that CLOSES after it.
So the reaction to a release is that session's close-to-close return: from
the last close at or before the release to the first close after it.

That one rule handles every case the calendar throws at it, without a table
of special cases:

  · 美国CPI at 08:30 New York (05:30 Vancouver) on a Tuesday lands on that
    Tuesday's 标普500 session, and on WEDNESDAY's 沪深300 — Shanghai closed at
    15:00 Beijing, which was 23:00 Vancouver the night before.
  · 中国CPI at 09:30 Beijing on a Wednesday is 17:30 Vancouver on TUESDAY,
    and still lands on Wednesday for both: Shanghai closes six hours later,
    New York the following afternoon.
  · 社融 at 16:00 Beijing comes after Shanghai's close, so it is the NEXT
    A-share session's news.
  · A release on a weekend or holiday waits for the next session that
    actually trades. 春节 included, without knowing it is 春节: the session
    list is the index's own bars.

The viewer's time zone plays no part in this. It only decides which column of
the calendar the release is drawn in.

How big is big
--------------
A session's move contains everything else that happened that day, so a
return on its own says little. Each reaction is also given as a multiple of a
normal day: |return| over the mean |return| of the 60 sessions before it.
The mean rather than the median, so that an ordinary day averages 1.0× — with
the median an ordinary day averages about 1.18×, and every release would look
like it mattered. A release that averages well above 1.0× across years moves
the market; one near 1.0× is a day like any other.

Two things this cannot separate, and the page says so: releases at the same
instant (CPI and core CPI) share one session and show identical moves; and a
session with a release in it is still a session with everything else in it.

Storage
-------
History needs years of releases, and eco_cal cannot be asked for one event's
history — the `event` filter wants an exact name and names carry their period
("(九月)"). So US and China releases are kept in `eco_release`, filled once
by `python eco_impact.py backfill 2021-11-01` and topped up daily by
`python eco_impact.py`. Rows are stored as eco_cal gives them; a release
listed twice (once at its scheduled time with no value, once with the value,
sometimes a day apart) is resolved when read, not when written.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BJ = ZoneInfo("Asia/Shanghai")
NY = ZoneInfo("America/New_York")

TABLE = "eco_release"
MIGRATION = "supabase/migrations/20261111_eco_release.sql"

#: The countries whose releases are kept and mapped. eco_cal's country filter
#: is loose — asking for 中国 also returns 澳大利亚 and 英国 rows — so a row is
#: kept only when its event name starts with the country asked for.
COUNTRIES = ("美国", "中国")

#: Indexes a release can be read against: (label, market). An allow-list,
#: because each one is ten years of prices fetched on demand.
INDEXES = {
    "^GSPC": ("标普500", "US"),
    "^NDX": ("纳指100", "US"),
    "000300.SH": ("沪深300", "CN"),
    "000001.SH": ("上证指数", "CN"),
    "399006.SZ": ("创业板指", "CN"),
}
DEFAULT_CODES = ("^GSPC", "000300.SH")

#: When each market's session closes, in its own zone. New York's 16:00 moves
#: against UTC twice a year and zoneinfo moves it with it.
CLOSE = {"US": (NY, 16), "CN": (BJ, 15)}

#: Sessions in the "normal day" baseline, and the fewest it may rest on.
TYPICAL_DAYS = 60
TYPICAL_MIN = 20

#: A scheduled row with no value and a valued row for the same series within
#: this many days are the same release. Weekly series are seven days apart,
#: so three cannot merge two real releases.
TWIN_DAYS = 3

#: eco_cal returns at most PAGE rows per call. Ranges sized to stay under it
#: on a normal stretch (US ≈10–25 a day, China ≈3), and halved when one fills.
PAGE = 100
STEP_DAYS = {"美国": 4, "中国": 14}
#: 20 calls a minute.
CALL_GAP_S = 3.1

#: The daily top-up re-reads this far back (actuals arrive after the
#: scheduled row was first stored) and this far ahead (the coming schedule).
SYNC_BACK_DAYS = 14
SYNC_AHEAD_DAYS = 10

_PERIOD = re.compile(
    r"\s*\((?:[一二三四五六七八九十]+月|第[一二三四]季度|上半年|下半年"
    r"|[0-9]+月[0-9]*日?当周|[0-9]+/[0-9]+)\)$")
_NUM = re.compile(r"^([-+]?\d*\.?\d+)\s*([KMBT%]?)")
_SCALE = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


# ── naming and values ────────────────────────────────────────────────────────
def series_of(event: str) -> str:
    """
    "美国CPI月率(%)(九月)" → "美国CPI月率(%)".

    The period suffix is what makes every month a different event name; the
    rest is the series. Units in brackets ("(%)", "(人民币十亿元)") stay.
    """
    return _PERIOD.sub("", str(event or "")).strip()


def num(v) -> float | None:
    """ "2,490.0B" → 2.49e12, "4.9%" → 4.9, "" → None. """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    m = _NUM.match(str(v).replace(",", "").strip())
    if not m:
        return None
    return float(m.group(1)) * _SCALE.get(m.group(2), 1.0)


def _ns(at) -> int | None:
    """An ISO instant (any offset) → UTC nanoseconds. None for no instant."""
    if not at:
        return None
    try:
        t = pd.Timestamp(at)
    except (ValueError, TypeError):
        return None
    if t.tzinfo is None:
        t = t.tz_localize(BJ)
    return int(t.value)


# ── sessions ─────────────────────────────────────────────────────────────────
def close_instants(days: Iterable, market: str) -> np.ndarray:
    """Each session date → the UTC instant its exchange closed, in ns."""
    tz, hour = CLOSE[market]
    out = [pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hour, tz=tz).value
           for d in days]
    return np.asarray(out, dtype="int64")


def _easter(y: int) -> date:
    """Gregorian Easter Sunday (the anonymous algorithm)."""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    return date(y, month, (h + l - 7 * m + 114) % 31 + 1)


def _nth(y: int, month: int, weekday: int, n: int) -> date:
    """The nth (1-based; -1 = last) given weekday of a month."""
    if n > 0:
        d = date(y, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    d = date(y + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date | None:
    """Saturday holidays move to Friday, Sunday ones to Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nyse_holidays(y: int) -> set[date]:
    """
    NYSE full-day closures for a year, by rule.

    Only used for sessions that have not happened yet — the past comes from
    the index's own bars, which already know about one-off closures.
    """
    out = {
        _nth(y, 1, 0, 3), _nth(y, 2, 0, 3), _easter(y) - timedelta(days=2),
        _nth(y, 5, 0, -1), _nth(y, 9, 0, 1), _nth(y, 11, 3, 4),
        _observed(date(y, 7, 4)), _observed(date(y, 12, 25)),
    }
    # New Year on a Saturday is not moved back into December.
    ny = date(y, 1, 1)
    if ny.weekday() != 5:
        out.add(_observed(ny))
    if y >= 2022:
        out.add(_observed(date(y, 6, 19)))
    return out


_cn_cal: dict[str, tuple[float, list[date]]] = {}
_cn_lock = threading.Lock()


def _cn_open_days(start: date, end: date) -> list[date] | None:
    """SSE open days from Tushare's trade_cal, cached for half a day."""
    key = f"{start}:{end}"
    with _cn_lock:
        hit = _cn_cal.get(key)
        if hit and time.time() - hit[0] < 12 * 3600:
            return hit[1]
    try:
        cal = _api().trade_cal(exchange="SSE", start_date=start.strftime("%Y%m%d"),
                               end_date=end.strftime("%Y%m%d"), is_open="1")
        days = sorted(pd.Timestamp(str(d)).date() for d in cal["cal_date"])
    except Exception:                                              # noqa: BLE001
        return None
    with _cn_lock:
        _cn_cal[key] = (time.time(), days)
    return days


def upcoming_sessions(market: str, start: date, days: int = 21) -> list[date]:
    """
    Trading days from `start` on, for sessions the index has no bar for yet.

    A-shares read the exchange's calendar (holidays are announced, not
    derivable); the US uses the NYSE rules. If the A-share calendar cannot be
    read, weekdays — wrong only across a holiday, and only until it passes.
    """
    end = start + timedelta(days=days)
    if market == "CN":
        open_days = _cn_open_days(start, end)
        if open_days is not None:
            return open_days
    shut = nyse_holidays(start.year) | nyse_holidays(end.year) if market == "US" else set()
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in shut:
            out.append(d)
        d += timedelta(days=1)
    return out


# ── the mapping ──────────────────────────────────────────────────────────────
class Board:
    """
    One index, ready to answer "which session, and how big" for any instant.

    Built once per index per request: the closes, their close instants, the
    returns and the normal-day baseline are shared by every release.
    """

    def __init__(self, code: str, px: pd.Series, *, now_ns: int,
                 upcoming: Callable[[str, date], list[date]] = upcoming_sessions):
        self.code = code
        self.label, self.market = INDEXES[code]
        px = px.dropna().sort_index()
        self.days = pd.DatetimeIndex(px.index)
        closes = close_instants(self.days, self.market)
        # A bar whose close is still ahead is today's session in progress —
        # its "close" is the latest price, not a close.
        self.n_final = int(np.searchsorted(closes, now_ns, side="right"))
        self.closes = closes[:self.n_final]
        ret = px.pct_change()
        self.ret = ret.to_numpy()
        self.typical = (ret.abs().rolling(TYPICAL_DAYS, min_periods=TYPICAL_MIN)
                        .mean().shift(1).to_numpy())
        # Sessions with no final close yet: the live bar, then the calendar.
        pending = [d.date() for d in self.days[self.n_final:]]
        last = self.days[-1].date() if len(self.days) else None
        start = (last + timedelta(days=1)) if last else date.today()
        pending += [d for d in upcoming(self.market, start) if last is None or d > last]
        self.pending = pending
        self.pending_closes = close_instants(pending, self.market)

    def react(self, at_ns: int | None) -> dict | None:
        """The session a release at `at_ns` lands on, and its move if it has closed."""
        if at_ns is None:
            return None
        p = int(np.searchsorted(self.closes, at_ns, side="right"))
        if p < self.n_final:
            r = self.ret[p]
            if not np.isfinite(r):            # the very first bar: no prior close
                return None
            t = self.typical[p]
            return {
                "day": self.days[p].date().isoformat(),
                "ret": round(float(r) * 100, 2),
                "x": round(abs(float(r)) / float(t), 2) if np.isfinite(t) and t > 0 else None,
                "done": True,
            }
        q = int(np.searchsorted(self.pending_closes, at_ns, side="right"))
        if q < len(self.pending):
            return {"day": self.pending[q].isoformat(), "done": False}
        return None

    def meta(self) -> dict:
        return {"code": self.code, "label": self.label, "market": self.market}


def event_key(e: dict) -> str:
    """The client's key for an event: what identifies it on the server."""
    return f"{e.get('date')}|{e.get('time')}|{e.get('country')}|{e.get('event')}"


def mapped(e: dict) -> bool:
    """Only US and China releases are mapped, and only ones with a time."""
    c = e.get("country")
    return bool(e.get("at")) and c in COUNTRIES and str(e.get("event", "")).startswith(c)


def week(events: list[dict], boards: list[Board]) -> dict:
    """Each mapped release in a week → its session on each index."""
    out = {}
    for e in events:
        if not mapped(e):
            continue
        t = _ns(e["at"])
        out[event_key(e)] = {b.code: b.react(t) for b in boards}
    return {"indexes": [b.meta() for b in boards], "reactions": out}


# ── one series' history ──────────────────────────────────────────────────────
def dedupe(rows: list[dict]) -> list[dict]:
    """
    One row per real release.

    eco_cal lists some releases twice: once at the scheduled time with no
    value and once with it, sometimes on different days. The valueless row is
    dropped when a valued one sits within TWIN_DAYS; a valueless row with no
    twin is a release that has not happened yet and stays.
    """
    rows = sorted(rows, key=lambda r: _ns(r.get("at")) or 0)
    seen, uniq = set(), []
    for r in rows:
        k = (_ns(r.get("at")), r.get("event"))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    valued = [_ns(r["at"]) for r in uniq if r.get("value")]
    span = TWIN_DAYS * 86400 * 10**9
    keep = []
    for r in uniq:
        if not r.get("value"):
            t = _ns(r.get("at"))
            if t is not None and any(abs(t - v) <= span for v in valued if v is not None):
                continue
        keep.append(r)
    return keep


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return round(float(np.mean(xs)), 2) if xs else None


def history(rows: list[dict], boards: list[Board]) -> dict:
    """
    Every past release of one series, with each index's reaction session.

    "Above/below forecast" is direction only, not good or bad: an
    unemployment rate above forecast and retail sales above forecast are
    opposite news, and the page leaves that reading to the reader.
    """
    out = []
    for r in dedupe(rows):
        t = _ns(r.get("at"))
        actual, fore = num(r.get("value")), num(r.get("forecast"))
        surprise = (actual - fore) if actual is not None and fore is not None else None
        out.append({
            "at": r.get("at"),
            "event": r.get("event"),
            "value": r.get("value"),
            "forecast": r.get("forecast"),
            "prev": r.get("prev"),
            "surprise": surprise,
            # Rounding noise in the forecast is not a surprise.
            "dir": None if surprise is None else
                   (0 if abs(surprise) <= 1e-9 * max(1.0, abs(fore or 0)) else
                    (1 if surprise > 0 else -1)),
            "react": {b.code: b.react(t) for b in boards},
        })
    out.sort(key=lambda r: _ns(r["at"]) or 0, reverse=True)

    summary = {}
    for b in boards:
        done = [(r["react"][b.code], r["dir"]) for r in out
                if r["react"].get(b.code) and r["react"][b.code]["done"]]
        rets = [x["ret"] for x, _ in done]
        summary[b.code] = {
            "n": len(done),
            "mean_x": _mean([x["x"] for x, _ in done]),
            "mean_abs": _mean([abs(v) for v in rets]),
            "up": round(sum(v > 0 for v in rets) / len(rets), 2) if rets else None,
            "above": {"n": sum(d == 1 for _, d in done),
                      "ret": _mean([x["ret"] for x, d in done if d == 1])},
            "below": {"n": sum(d == -1 for _, d in done),
                      "ret": _mean([x["ret"] for x, d in done if d == -1])},
        }
    return {"indexes": [b.meta() for b in boards], "rows": out, "summary": summary}


# ── storage ──────────────────────────────────────────────────────────────────
def _api():
    import data_manager
    if not data_manager.init_tushare():
        raise RuntimeError("Tushare 未初始化")
    return data_manager.TUSHARE_API


def _db():
    import data_manager
    return data_manager.db


def _clean(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return s or None


def rows_from_frame(df: pd.DataFrame | None, country: str) -> list[dict]:
    """eco_cal rows → table rows, keeping only the country asked for."""
    if df is None or df.empty:
        return []
    out = {}
    for r in df.to_dict("records"):
        event = _clean(r.get("event")) or ""
        clock = _clean(r.get("time"))
        day = str(r.get("date") or "")
        if not event.startswith(country) or not clock or len(day) != 8:
            continue
        try:
            hh, mm = (int(x) for x in clock.split(":")[:2])
            at = datetime(int(day[:4]), int(day[4:6]), int(day[6:]), hh, mm, tzinfo=BJ)
        except (ValueError, TypeError):
            continue
        row = {
            "country": country, "event": event, "series": series_of(event),
            "at": at.isoformat(), "value": _clean(r.get("value")),
            "forecast": _clean(r.get("fore_value")), "prev": _clean(r.get("pre_value")),
        }
        k = (row["at"], event)
        # The same key twice in one batch would fail the upsert; keep the one
        # with a value.
        if k not in out or (row["value"] and not out[k]["value"]):
            out[k] = row
    return list(out.values())


def fetch_range(pro, country: str, a: date, b: date, *, gap_s: float = CALL_GAP_S,
                log=print) -> pd.DataFrame:
    """One country's releases from a to b, halving any range that hits the row cap."""
    for attempt in range(8):
        try:
            df = pro.eco_cal(country=country, start_date=a.strftime("%Y%m%d"),
                             end_date=b.strftime("%Y%m%d"))
            break
        except Exception as exc:                                   # noqa: BLE001
            log(f"[eco] retry {attempt + 1}: {str(exc)[:120]}")
            time.sleep(25)
    else:
        raise RuntimeError(f"eco_cal {country} {a}–{b} 读取失败")
    time.sleep(gap_s)
    df = df if df is not None else pd.DataFrame()
    if len(df) >= PAGE and a < b:
        mid = a + (b - a) // 2
        return pd.concat([fetch_range(pro, country, a, mid, gap_s=gap_s, log=log),
                          fetch_range(pro, country, mid + timedelta(days=1), b,
                                      gap_s=gap_s, log=log)], ignore_index=True)
    if len(df) >= PAGE:
        log(f"[eco] {country} {a}: one day filled a page; some rows may be missing")
    return df


def fetch(pro, start: date, end: date, *, log=print, gap_s: float = CALL_GAP_S) -> list[dict]:
    rows = []
    for country in COUNTRIES:
        a = start
        while a <= end:
            b = min(end, a + timedelta(days=STEP_DAYS[country] - 1))
            rows += rows_from_frame(fetch_range(pro, country, a, b, gap_s=gap_s, log=log),
                                    country)
            a = b + timedelta(days=1)
    return rows


def save(rows: list[dict], db=None) -> None:
    db = db or _db()
    for i in range(0, len(rows), 500):
        db.insert_records(TABLE, rows[i:i + 500], upsert=True)


def _missing(exc: Exception) -> bool:
    s = str(exc)
    return TABLE in s and any(m in s for m in ("does not exist", "PGRST205", "not find", "no such table"))


def load_series(country: str, series: str, db=None) -> list[dict]:
    """Every stored row of one series. LookupError when the table is not there yet."""
    db = db or _db()
    try:
        df = db.read_all(TABLE, filters={"country": country, "series": series},
                         order_by="at")
    except Exception as exc:                                       # noqa: BLE001
        if _missing(exc):
            raise LookupError(
                f"宏观数据历史的表还没建 — 请在 Supabase SQL 编辑器运行 {MIGRATION}") from exc
        raise
    if df is None or df.empty:
        return []
    df = df.astype(object).where(df.notna(), None)
    return df.to_dict("records")


def sync(pro=None, db=None, *, today: date | None = None, log=print) -> dict:
    """Re-read the last SYNC_BACK_DAYS and the next SYNC_AHEAD_DAYS."""
    pro = pro or _api()
    today = today or datetime.now(BJ).date()
    start, end = today - timedelta(days=SYNC_BACK_DAYS), today + timedelta(days=SYNC_AHEAD_DAYS)
    rows = fetch(pro, start, end, log=log)
    save(rows, db)
    return {"from": start.isoformat(), "to": end.isoformat(), "rows": len(rows)}


def backfill(start: date, end: date | None = None, *, pro=None, db=None, log=print) -> int:
    """Fill a span month by month, saving as it goes, so a failure keeps what it got."""
    pro = pro or _api()
    end = end or datetime.now(BJ).date()
    total, a = 0, start
    while a <= end:
        b = min(end, (pd.Timestamp(a) + pd.offsets.MonthEnd(0)).date())
        rows = fetch(pro, a, b, log=log)
        save(rows, db)
        total += len(rows)
        log(f"[eco] {a:%Y-%m}: {len(rows)} rows")
        a = b + timedelta(days=1)
    return total


if __name__ == "__main__":
    t0 = time.time()
    if len(sys.argv) >= 3 and sys.argv[1] == "backfill":
        n = backfill(date.fromisoformat(sys.argv[2]),
                     date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else None)
        print("[eco] backfill", n, "rows", f"{time.time() - t0:.0f}s")
    else:
        print("[eco] sync", sync(), f"{time.time() - t0:.0f}s")
