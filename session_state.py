"""
session_state.py — is the newest bar a finished session, or still being written?

Everything downstream of a daily bar assumes the bar is finished. 尾盘推演
says so out loud: "the bar is essentially formed and the close is a minute
away". Volume Z-scores, 量比, OBV momentum and the 吸筹/出货 detector all
compare today's volume against an average of full sessions.

That assumption holds for A-shares and does not hold for US and Canadian
names, and the difference is in the data source rather than the market:

  · Tushare publishes a daily bar after the close. The newest bar is always
    a complete session.
  · Yahoo serves a running bar for the session in progress. At 11:52 in New
    York — 36% of the way through — AAPL's "daily" bar held 13.3M shares
    against a 20-day average of 42.3M. Read as a finished day that is 0.32x
    normal volume, which reads as a dead tape. Divided by the 36% of the
    session that had actually happened it is 0.89x: a slightly quiet but
    entirely ordinary morning.

So the volume regime was being read off a number that was a third of the way
to existing. This module answers how far through the session a bar is, so
callers can either project the partial figures or decline to use them.

Nothing here hard-codes "CN is complete". A bar is complete when its date is
behind today, or when today's session has closed — which is true per market
and stays true if Tushare ever starts publishing intraday.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

#: Regular trading hours, local to each market. Minutes from midnight, as
#: (open, close) pairs — A-shares break for lunch, so they have two.
#: Extended-hours trading is deliberately excluded: Yahoo's daily bar tracks
#: the regular session, so the fraction has to be measured against the same.
HOURS: dict[str, tuple[str, tuple[tuple[int, int], ...]]] = {
    "CN": ("Asia/Shanghai", ((9 * 60 + 30, 11 * 60 + 30),
                             (13 * 60, 15 * 60))),
    "US": ("America/New_York", ((9 * 60 + 30, 16 * 60),)),
    "CA": ("America/Toronto", ((9 * 60 + 30, 16 * 60),)),
}

#: Below this, a partial bar's volume is too small a sample to project from —
#: the first minutes of a session carry the overnight auction and scale to
#: nonsense.
MIN_ELAPSED = 0.08


@dataclass(frozen=True)
class BarState:
    """What is known about the newest bar."""

    market: str
    #: The bar's own date, as the data source labelled it.
    bar_date: str
    #: True when the bar covers a session that has finished.
    complete: bool
    #: Fraction of the regular session that had elapsed, 0–1.
    elapsed: float
    #: Local time in the market when this was determined.
    local_time: str
    #: Whether a partial bar's volume can be scaled to a day at all.
    projectable: bool

    @property
    def elapsed_pct(self) -> float:
        return round(self.elapsed * 100, 1)

    def project(self, value: float | None) -> float | None:
        """
        A partial session's volume, scaled to what a full day would hold.

        An estimate and labelled as one everywhere it surfaces: it assumes the
        rest of the session trades at the pace of the part already seen, which
        is wrong around the close in particular, when volume runs heaviest.
        """
        if value is None or self.complete:
            return value
        if not self.projectable or self.elapsed <= 0:
            return None
        return value / self.elapsed


def session_minutes(market: str) -> int:
    _, spans = HOURS.get(market, HOURS["US"])
    return sum(end - start for start, end in spans)


def elapsed_fraction(market: str, now: datetime | None = None) -> float:
    """
    How much of today's regular session has happened, 0–1.

    Counts only minutes the market is actually open, so an A-share bar at
    12:00 reads 0.5 rather than 0.625 — the lunch break is not trading time
    and counting it would overstate how formed the bar is.
    """
    tz_name, spans = HOURS.get(market, HOURS["US"])
    here = (now or datetime.now(ZoneInfo(tz_name))).astimezone(ZoneInfo(tz_name))
    if here.weekday() >= 5:
        return 1.0

    minute = here.hour * 60 + here.minute
    done = 0
    total = 0
    for start, end in spans:
        total += end - start
        done += max(0, min(minute, end) - start)
    return 0.0 if total <= 0 else max(0.0, min(1.0, done / total))


def state(market: str, bar_date: date | str,
          now: datetime | None = None) -> BarState:
    """
    Whether `bar_date` is a finished session in `market`.

    A bar dated before today is finished whatever the clock says. A bar dated
    today is finished only once the session has closed — and a weekend or a
    holiday reads as closed, which is correct: a bar carrying that date was
    written by a session that has ended.
    """
    tz_name, _ = HOURS.get(market, HOURS["US"])
    tz = ZoneInfo(tz_name)
    here = (now or datetime.now(tz)).astimezone(tz)
    today = here.date()

    if isinstance(bar_date, str):
        try:
            stamp = date.fromisoformat(bar_date[:10])
        except ValueError:
            stamp = today
    else:
        stamp = bar_date

    if stamp < today:
        return BarState(market, stamp.isoformat(), True, 1.0,
                        here.strftime("%H:%M"), True)

    # A bar dated in the future is a data problem, not a live session; treat
    # it as finished rather than inventing a fraction for it.
    if stamp > today:
        return BarState(market, stamp.isoformat(), True, 1.0,
                        here.strftime("%H:%M"), True)

    elapsed = elapsed_fraction(market, here)
    return BarState(
        market, stamp.isoformat(), elapsed >= 1.0, elapsed,
        here.strftime("%H:%M"), elapsed >= MIN_ELAPSED)


def of_frame(market: str, index_last, now: datetime | None = None) -> BarState:
    """`state` for the last row of an OHLCV frame."""
    try:
        stamp = index_last.date()
    except AttributeError:
        stamp = index_last
    return state(market, stamp, now)


def describe(bar: BarState) -> str:
    """One line for a prompt or a tooltip."""
    if bar.complete:
        return f"{bar.bar_date} 已收盘，这根是完整的一天。"
    return (f"{bar.bar_date} 仍在交易中 —— 当地时间 {bar.local_time}，"
            f"这根 K 线只走完了 {bar.elapsed_pct:.0f}%。"
            f"成交量、最高价、最低价都还会变。")
