"""
日历 — what is scheduled to happen, from two sources that behave nothing alike.

  · 经济数据 — Tushare `eco_cal`, global macro releases.
  · 财报披露 — Tushare `disclosure_date`, A-share earnings disclosure dates.

The economic half is the awkward one, and the shape of this module is mostly
a response to how `eco_cal` behaves. Measured against the live API:

  · It returns at most 100 rows per call, whatever `end_date` says. A request
    for 29 Sep – 1 Oct came back with 100 rows when those three days actually
    hold 253 events, so a multi-day request silently drops the rest.
  · `offset` pages, but the underlying result is not stably ordered. Paging a
    31-day range returned 786 rows of which only 779 were distinct, and one
    day came back 147 instead of its true 149 — duplicates in, real events
    out.
  · It allows 20 calls a minute.

So: one request per calendar day, paged by offset within that day, which is
both complete and stable. A week is eight calls and about five seconds. A
month at that rate would breach the limit, which is why the economic view is
a week rather than a month.

Past days are cached forever. An economic release that has already happened
does not change, and that is what makes moving through the calendar cheap.

Times are Beijing, including the date
-------------------------------------
`eco_cal` reports every event in Asia/Shanghai, converted correctly from the
source market: US initial jobless claims, fixed at 08:30 New York, come back
at 21:30 before 8 March 2026 and 20:30 after it — the hour the United States
moved its clocks and Beijing did not.

That means the DATE is a Beijing date too, and for anyone outside UTC+8 a
chunk of the calendar belongs to a different day than Tushare labels it. On
one real week, 74 of 262 events — 28% — fell on a different calendar day in
Toronto. So each event carries an unambiguous instant and the client places
it in the viewer's own day, and the range is padded a day on each side so a
viewer anywhere between UTC-12 and UTC+14 gets a complete week.
"""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

#: Tushare reports eco_cal in Beijing time, and nothing in the payload says so.
SOURCE_TZ = "Asia/Shanghai"
BEIJING = ZoneInfo(SOURCE_TZ)
#: Beijing is UTC+8 and the world runs from UTC-12 to UTC+14, so an event can
#: move at most one calendar day in either direction. One day of padding each
#: side therefore covers every viewer, and two would be wasted calls.
PAD_DAYS = 1

#: Tushare's per-call row ceiling for eco_cal.
PAGE = 100
#: Refuse to page a single day past this — a guard against a changed API
#: turning one day into an unbounded loop.
MAX_PAGES = 6
#: Days in the economic view. Eight calls, five seconds, inside 20/minute.
WEEK = 7
#: A finished day never changes; today and the future do.
PAST_TTL_S = 30 * 24 * 3600
LIVE_TTL_S = 30 * 60

_cache: dict[str, tuple[float, list]] = {}
_lock = threading.Lock()


def _api():
    import data_manager
    if not data_manager.init_tushare():
        raise RuntimeError("Tushare 未初始化")
    return data_manager.TUSHARE_API


def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _pretty(ymd: str) -> str:
    t = str(ymd)
    return f"{t[:4]}-{t[4:6]}-{t[6:8]}" if len(t) == 8 and t.isdigit() else t


def _clean(v) -> str | None:
    s = str(v or "").strip()
    return s or None


def _instant(day: str, clock: str) -> str | None:
    """
    "2026-10-08" + "02:00" → "2026-10-08T02:00:00+08:00".

    None when Tushare gives no time. Every event in a sampled week had one,
    but an all-day entry should degrade to its Beijing date rather than being
    placed at midnight and shifted into the wrong day by the conversion.
    """
    if not clock:
        return None
    try:
        hh, mm = (int(x) for x in clock.split(":")[:2])
        return (datetime.fromisoformat(day)
                .replace(hour=hh, minute=mm, tzinfo=BEIJING).isoformat())
    except (ValueError, TypeError):
        return None


# ── economic calendar ────────────────────────────────────────────────────────
def eco_day(day: date) -> list[dict]:
    """
    One calendar day of economic releases, complete.

    Paged by offset within the day because a busy day exceeds the 100-row
    ceiling — 30 September 2026 holds 147 distinct events.
    """
    key = _ymd(day)
    ttl = PAST_TTL_S if day < date.today() else LIVE_TTL_S
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]

    api = _api()
    frames, offset = [], 0
    for _ in range(MAX_PAGES):
        kw = {"start_date": key, "end_date": key}
        if offset:
            kw["offset"] = offset
        try:
            page = api.eco_cal(**kw)
        except Exception as exc:                                   # noqa: BLE001
            # A rate-limit or credit failure mid-week must not empty the whole
            # view; the days already fetched are still worth showing.
            raise RuntimeError(f"经济日历读取失败：{exc}"[:200])
        n = 0 if page is None else len(page)
        if n:
            frames.append(page)
        if n < PAGE:
            break
        offset += n

    rows: list[dict] = []
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["date", "time", "event", "country"])
        df = df.sort_values(["time", "country"], kind="stable")
        for _, r in df.iterrows():
            day = _pretty(r.get("date"))
            clock = _clean(r.get("time")) or ""
            rows.append({
                # The instant, offset included, so the browser can place it in
                # the viewer's own day instead of trusting a Beijing label.
                "at": _instant(day, clock),
                "date": day,
                "time": clock,
                "country": _clean(r.get("country")) or "—",
                "currency": _clean(r.get("currency")) or "",
                "event": _clean(r.get("event")) or "",
                # Actual, previous, forecast — all strings from Tushare, with
                # their units baked in ("4.9%", "6.58%"), so they are passed
                # through rather than parsed into numbers that would lose them.
                "value": _clean(r.get("value")),
                "prev": _clean(r.get("pre_value")),
                "forecast": _clean(r.get("fore_value")),
            })

    with _lock:
        _cache[key] = (time.time(), rows)
    return rows


def economic(start: date, days: int = WEEK) -> dict:
    """
    A week of releases as a flat list, each with the instant it happens.

    Not grouped into days here, because which day an event belongs to depends
    on who is looking: 28% of one real week moved a day between Beijing and
    Toronto. The caller groups, and gets a padded range so its own first and
    last days are complete.
    """
    first = start - timedelta(days=PAD_DAYS)
    last = start + timedelta(days=days - 1 + PAD_DAYS)

    events = []
    cursor = first
    while cursor <= last:
        events.extend(eco_day(cursor))
        cursor += timedelta(days=1)

    return {
        "from": start.isoformat(),
        "to": (start + timedelta(days=days - 1)).isoformat(),
        # The padded span actually fetched, so the client knows which events
        # it may safely show and which are only there to fill the edges.
        "fetched_from": first.isoformat(),
        "fetched_to": last.isoformat(),
        "source_tz": SOURCE_TZ,
        "events": events,
        "total": len(events),
    }


# ── earnings disclosure ──────────────────────────────────────────────────────
#: How early a reporting period becomes worth offering, relative to its end.
#:
#: The Streamlit page waited until the month AFTER a quarter closed, which
#: hides the calendar during the fortnight it is most useful. Measured on
#: 29 September 2026 — one day before Q3 closed — Tushare already held 2,323
#: disclosure rows for the period and 36 of the watchlist had October dates
#: booked. Companies file their intended date before the quarter ends, so a
#: period is offered once it is close enough for those to exist.
OPENS_DAYS_BEFORE_END = 21


def periods(ref: date | None = None) -> list[dict]:
    """
    The reporting periods worth asking about, newest last.

    A period that is still far off is left out rather than offered empty:
    asking for Q3 in May returns nothing and reads as a failure rather than
    as a calendar that has not happened yet.
    """
    ref = ref or date.today()
    y = ref.year
    candidates = [
        (f"{y - 1} 年报", date(y - 1, 12, 31)),
        (f"{y} 一季报", date(y, 3, 31)),
        (f"{y} 中报", date(y, 6, 30)),
        (f"{y} 三季报", date(y, 9, 30)),
    ]
    out = []
    for label, end in candidates:
        if ref >= end - timedelta(days=OPENS_DAYS_BEFORE_END):
            out.append({"label": label, "end": end.strftime("%Y%m%d")})
    return out


def _disclosures(period_end: str) -> pd.DataFrame:
    """
    Every A-share's disclosure dates for one reporting period, in one call.

    Bulk rather than per-ticker: the whole market is 2,300–5,600 rows and one
    request, against one request per name for a watchlist of any size.
    """
    api = _api()
    df = api.disclosure_date(
        end_date=str(period_end),
        fields="ts_code,ann_date,end_date,pre_date,actual_date")
    if df is None or df.empty:
        return pd.DataFrame()
    return df.replace("", None)


def _status(actual: str | None, effective: str | None, today: date) -> str:
    """
    Reported, overdue, or still scheduled.

    Overdue is worth its own state: a date that has passed with nothing filed
    is the one row on the page that needs chasing, and folding it into
    "scheduled" hides exactly that.
    """
    if actual:
        return "reported"
    if not effective:
        return "unknown"
    try:
        when = datetime.strptime(str(effective), "%Y%m%d").date()
    except ValueError:
        return "unknown"
    return "overdue" if when < today else "scheduled"


def earnings(period_end: str, watchlist: list[dict],
             ref: date | None = None, with_na: bool = True) -> dict:
    """
    Disclosure dates for the watchlist, for one reporting period.

    `watchlist` is [{t, n}] in the app's own symbol form; A-shares are matched
    by their Tushare code, and anything else is left out rather than being
    reported as missing — a US holding has no A-share disclosure date and
    saying so on every load would be noise.
    """
    ref = ref or date.today()
    import data_manager

    wanted, names = {}, {}
    for row in watchlist:
        symbol = str(row.get("t") or "")
        if not symbol or not symbol[:6].isdigit():
            continue
        code = data_manager.get_tushare_ticker(symbol)
        wanted[code] = symbol
        names[code] = str(row.get("n") or symbol)

    if not wanted:
        na = (na_earnings(watchlist, ref) if with_na
              else {"rows": [], "watched": 0, "missing": []})
        by_day: dict[str, list] = {}
        for r in na["rows"]:
            by_day.setdefault(r["date"], []).append(r)
        counts: dict[str, int] = {}
        for r in na["rows"]:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        months = _months(by_day)
        return {
            "period": period_end,
            "rows": na["rows"], "cn_rows": [], "na_rows": na["rows"],
            "by_date": [{"date": d, "rows": v}
                        for d, v in sorted(by_day.items())],
            "months": months,
            "focus": months[0]["ym"] if months else ref.strftime("%Y-%m"),
            "missing": {"cn": [], "na": na["missing"]},
            "counts": counts,
            "watched": {"cn": 0, "na": na["watched"], "total": na["watched"]},
        }

    df = _disclosures(period_end)
    rows, seen = [], set()
    if not df.empty:
        hits = df[df["ts_code"].isin(list(wanted))]
        for _, r in hits.iterrows():
            code = str(r["ts_code"])
            actual = _clean(r.get("actual_date"))
            pre = _clean(r.get("pre_date"))
            effective = actual or pre
            seen.add(code)
            rows.append({
                "t": wanted[code], "code": code, "n": names[code],
                "market": "CN",
                "date": _pretty(effective) if effective else None,
                "pre_date": _pretty(pre) if pre else None,
                "actual_date": _pretty(actual) if actual else None,
                "ann_date": _pretty(_clean(r.get("ann_date")) or ""),
                "status": _status(actual, effective, ref),
                # An estimate that moved is worth seeing: the company pushed
                # its own date.
                "moved": bool(actual and pre and actual != pre),
            })

    rows.sort(key=lambda r: (r["date"] or "9999", r["n"]))

    by_date: dict[str, list] = {}
    for r in rows:
        if r["date"]:
            by_date.setdefault(r["date"], []).append(r)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    na = (na_earnings(watchlist, ref) if with_na
          else {"rows": [], "watched": 0, "missing": []})

    # One calendar, both markets. A-share disclosure dates belong to the
    # reporting period you picked; North American ones are simply whatever
    # falls in the month you are looking at, because US and Canadian filers
    # do not report against the A-share quarterly calendar.
    merged = dict(by_date)
    for r in na["rows"]:
        merged.setdefault(r["date"], []).append(r)
    for day in merged:
        merged[day] = sorted(merged[day], key=lambda r: (r["market"], r["n"]))

    for r in na["rows"]:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    months = _months(merged)
    return {
        "period": period_end,
        "rows": sorted(rows + na["rows"],
                       key=lambda r: (r["date"] or "9999", r["n"])),
        "cn_rows": rows,
        "na_rows": na["rows"],
        "by_date": [{"date": d, "rows": v} for d, v in sorted(merged.items())],
        "months": months,
        # The month the calendar should open on. A-share disclosure dates for
        # one quarter cluster into a few weeks, so landing on today's month
        # would usually show an empty grid a page away from everything.
        "focus": months[0]["ym"] if months else ref.strftime("%Y-%m"),
        "missing": {
            "cn": [names[c] for c in wanted if c not in seen],
            "na": na["missing"],
        },
        "counts": counts,
        "watched": {"cn": len(wanted), "na": na["watched"],
                    "total": len(wanted) + na["watched"]},
    }


def _months(by_date: dict[str, list]) -> list[dict]:
    """Every month holding a disclosure, busiest first."""
    tally: dict[str, int] = {}
    for day, rows in by_date.items():
        tally[day[:7]] = tally.get(day[:7], 0) + len(rows)
    return [{"ym": ym, "count": n}
            for ym, n in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))]

# ── US and Canadian earnings ─────────────────────────────────────────────────
#: Yahoo, not Tushare: `disclosure_date` is an A-share filing calendar and has
#: nothing for a North American name. Measured on a real watchlist, 23 names
#: took 4.2s across 8 threads, and 21 of them had dates — the two without were
#: ETFs, which do not report earnings at all.
NA_WORKERS = 8
NA_LIMIT = 16
#: How far either side of today a North American date is worth keeping.
NA_BACK_DAYS = 200
NA_FWD_DAYS = 200


def _when(stamp) -> str:
    """
    盘前 / 盘后 / 盘中, from the hour in the market's own time.

    The hour is the useful part of a US earnings date and the part a date
    alone throws away: 08:00 means the print lands before the open and the
    gap is tomorrow's problem, 16:00 means it lands after the close.
    """
    try:
        minute = stamp.hour * 60 + stamp.minute
    except AttributeError:
        return ""
    if minute < 9 * 60 + 30:
        return "盘前"
    if minute >= 16 * 60:
        return "盘后"
    return "盘中"


def _na_one(symbol: str, name: str) -> list[dict]:
    """Every dated earnings event Yahoo holds for one name."""
    import markets
    import pandas as pd

    try:
        _, code = markets.split(symbol)
    except Exception:                                              # noqa: BLE001
        return []

    try:
        import yfinance as yf
        frame = yf.Ticker(code).get_earnings_dates(limit=NA_LIMIT)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[calendar] {symbol}: {type(exc).__name__}: {exc}"[:160])
        return []
    if frame is None or frame.empty:
        return []      # an ETF, or a name Yahoo has no estimates for

    out = []
    for stamp, row in frame.iterrows():
        try:
            day = stamp.date().isoformat()
        except AttributeError:
            continue
        reported = row.get("Reported EPS")
        est = row.get("EPS Estimate")
        surprise = row.get("Surprise(%)")
        out.append({
            "t": symbol, "code": code, "n": name,
            "market": "NA",
            "date": day,
            "when": _when(stamp),
            "eps_estimate": None if pd.isna(est) else round(float(est), 4),
            "eps_reported": None if pd.isna(reported) else round(float(reported), 4),
            "surprise_pct": None if pd.isna(surprise) else round(float(surprise), 2),
            # Yahoo does not say whether a future date is confirmed or
            # estimated, so neither does this.
            "estimated": True,
        })
    return out


def na_earnings(watchlist: list[dict], ref: date | None = None) -> dict:
    """
    Earnings dates for the US and Canadian names on the watchlist.

    Scoped to a window around today rather than to a reporting period: North
    American companies do not file against the quarterly calendar A-shares
    use, so there is no period to select.
    """
    from concurrent.futures import ThreadPoolExecutor

    import markets

    ref = ref or date.today()
    wanted = []
    for row in watchlist:
        symbol = str(row.get("t") or "")
        try:
            if markets.split(symbol)[0] in ("US", "CA"):
                wanted.append((symbol, str(row.get("n") or symbol)))
        except Exception:                                          # noqa: BLE001
            continue

    if not wanted:
        return {"rows": [], "watched": 0, "missing": []}

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(NA_WORKERS, len(wanted))) as pool:
        for got in pool.map(lambda w: _na_one(*w), wanted):
            rows.extend(got)

    lo = (ref - timedelta(days=NA_BACK_DAYS)).isoformat()
    hi = (ref + timedelta(days=NA_FWD_DAYS)).isoformat()
    rows = [r for r in rows if lo <= r["date"] <= hi]
    for r in rows:
        r["status"] = ("reported" if r["eps_reported"] is not None
                       else _status(None, r["date"].replace("-", ""), ref))
    rows.sort(key=lambda r: (r["date"], r["n"]))

    seen = {r["t"] for r in rows}
    return {
        "rows": rows,
        "watched": len(wanted),
        "missing": [n for t, n in wanted if t not in seen],
    }

