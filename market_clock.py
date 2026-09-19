"""
market_clock.py — "has a new trading session actually appeared?"

Anything cached against market data needs to know when that data changed, and
the obvious answers are both wrong:

  * A clock. "Refresh after six hours" regenerates at 2pm on a day when
    nothing happened, and still serves yesterday an hour after today's close.
  * A calendar. Tushare's trade_cal will tell you the exchange was OPEN today,
    which is a different claim from "today's bar is published". Between the
    15:00 close and the evening publish, the calendar says yes and the data
    says nothing. A cache keyed on the calendar invalidates itself into an
    empty fetch every afternoon.

So this asks the only question that matters: what is the newest bar the data
source will actually give us? One small request against a broad index —
never suspended, never thinly traded, so its last bar IS the market's last
session — rather than 80 requests for the stocks we were going to cache.

Held in memory for a few minutes, because the answer cannot change faster
than that and a cache-validity check must not cost more than the thing it is
guarding.
"""

from __future__ import annotations

import threading
import time

#: 上证指数. A broad index rather than a stock: a stock can be suspended, and
#: a suspended stock's last bar is not the market's last session.
REFERENCE_INDEX = "000001.SH"

#: How long the answer is reused. The publish happens once an evening, so
#: minutes of staleness cost nothing and save a request per cache check.
TTL_S = 300

_cache: dict[str, tuple[float, str]] = {}
_lock = threading.Lock()


def latest_session(index: str = REFERENCE_INDEX, *, now=time.time) -> str | None:
    """
    The date of the newest published bar, as 'YYYY-MM-DD'. None if unknown.

    None is not "no new data" — it means the question could not be answered,
    and callers must treat it as a cache MISS rather than a hit. Serving a
    stale result because the freshness check itself failed is the one outcome
    worse than recomputing.
    """
    with _lock:
        hit = _cache.get(index)
        if hit and now() - hit[0] < TTL_S:
            return hit[1]

    value = _fetch(index)
    if value:
        with _lock:
            _cache[index] = (now(), value)
    return value


def _fetch(index: str) -> str | None:
    import data_manager
    try:
        # A short window: we want the last row, not a history, and asking for
        # three years to read one date would make this as expensive as the
        # scan it protects.
        df = data_manager.get_index_data_live(index, lookback_days=15)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[market_clock] could not read {index}: {type(exc).__name__}: {exc}")
        return None
    if df is None or df.empty:
        return None
    try:
        return str(df.index[-1].date())
    except Exception:                                              # noqa: BLE001
        return None


def forget(index: str = REFERENCE_INDEX) -> None:
    """Drop the memoised answer — for tests, and for a forced refresh."""
    with _lock:
        _cache.pop(index, None)
