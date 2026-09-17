"""
markets/na.py — US and Canadian equities, via Yahoo.

Prices come back split- and dividend-adjusted (`auto_adjust=True`), which is
the same basis as the A-share 前复权 series the indicators are calibrated on.
Mixing the two bases would put a fake gap in every chart on an ex-dividend day
and would make a multi-year return wrong, so this is not a detail.

Fundamentals return None. Yahoo publishes a stock's CURRENT market cap, PE and
float — not a daily history of them — and the two features that want that
history both degrade correctly without it: build_chips() already returns None
with no turnover column, and the PE pane simply has no series to draw. Deriving
a turnover history from today's float would silently misstate every past
session, which is worse than an empty panel. The current snapshot is exposed
separately, through profile(), for the header.

Yahoo is an unofficial source and does break from time to time. Everything
specific to it lives in this file, so replacing it with Tiingo or EODHD is one
adapter, not a rewrite.
"""

from __future__ import annotations

import threading
import time

import pandas as pd

from .base import NO_REGIME_BREAK, Conventions, Market, StockRef

FETCH_ATTEMPTS = 3
FETCH_BACKOFF_S = 1.5

#: Yahoo exchange labels we accept per market, for search filtering. Anything
#: else — LSE, Frankfurt, crypto — is not something this app can analyse
#: alongside a Questrade book, so it is left out of the picker rather than
#: offered and then failing on fetch.
US_EXCHANGES = {"NMS", "NYQ", "NGM", "ASE", "PCX", "BTS", "NCM", "NYS"}
CA_EXCHANGES = {"TOR", "VAN", "CNQ", "NEO"}

#: Yahoo suffixes that identify a Canadian listing.
CA_SUFFIXES = (".TO", ".V", ".CN", ".NE")


class YahooMarket(Market):
    def __init__(self, conv: Conventions, exchanges: set[str]):
        self.conv = conv
        self._exchanges = exchanges
        self._lock = threading.Lock()

    # ── prices ───────────────────────────────────────────────────────────
    def fetch_ohlcv(self, symbol: str, years: int = 3) -> pd.DataFrame | None:
        import yfinance as yf

        last_err = None
        for attempt in range(FETCH_ATTEMPTS):
            try:
                # Serialised: yfinance shares a session and a cookie/crumb, and
                # concurrent first-calls race to establish it.
                with self._lock:
                    raw = yf.Ticker(symbol).history(
                        period=f"{years}y", auto_adjust=True, raise_errors=False)
                if raw is not None and not raw.empty:
                    return _normalise(raw)
                last_err = "empty response"
            except Exception as exc:                      # noqa: BLE001
                last_err = repr(exc)[:200]
            if attempt < FETCH_ATTEMPTS - 1:
                time.sleep(FETCH_BACKOFF_S * (attempt + 1))

        print(f"[markets.na] {symbol}: no data after {FETCH_ATTEMPTS} attempts ({last_err})")
        return None

    # ── identity ─────────────────────────────────────────────────────────
    def resolve(self, symbol: str) -> StockRef | None:
        info = self._info(symbol)
        if not info:
            return None
        name = info.get("longName") or info.get("shortName") or symbol
        return StockRef(symbol=symbol, name=str(name),
                        exchange=str(info.get("exchange") or ""))

    def profile(self, symbol: str) -> dict | None:
        """
        Today's snapshot for the header — market cap, PE, PB, float, sector.

        A point-in-time reading, NOT a history, and the caller must not treat
        it as one: it is right for the last bar and wrong for every earlier one.
        """
        info = self._info(symbol)
        if not info:
            return None

        def num(key):
            v = info.get(key)
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return None if v != v else v

        return {
            "name": str(info.get("longName") or info.get("shortName") or symbol),
            "currency": str(info.get("currency") or self.conv.currency),
            "market_cap": num("marketCap"),
            "pe_ttm": num("trailingPE"),
            "pb": num("priceToBook"),
            "shares_outstanding": num("sharesOutstanding"),
            "float_shares": num("floatShares"),
            "sector": info.get("sector") or None,
            "industry": info.get("industry") or None,
            "exchange": str(info.get("exchange") or ""),
        }

    def search(self, query: str, limit: int = 8) -> list[StockRef]:
        import yfinance as yf
        try:
            with self._lock:
                quotes = yf.Search(query, max_results=max(limit * 3, 10)).quotes or []
        except Exception as exc:                          # noqa: BLE001
            print(f"[markets.na] search({query!r}) failed: {exc!r}"[:200])
            return []

        out: list[StockRef] = []
        for q in quotes:
            sym = str(q.get("symbol") or "")
            if not sym or q.get("quoteType") not in ("EQUITY", "ETF"):
                continue
            if not self._owns(sym, str(q.get("exchange") or "")):
                continue
            out.append(StockRef(
                symbol=sym,
                name=str(q.get("shortname") or q.get("longname") or sym),
                exchange=str(q.get("exchDisp") or ""),
            ))
            if len(out) >= limit:
                break
        return out

    def _owns(self, symbol: str, exchange: str) -> bool:
        """
        Which of the two North American markets a hit belongs to.

        The suffix decides, not the exchange code: Yahoo labels a TSX listing
        "TOR" in one field and "Toronto" in another depending on the endpoint,
        but SHOP.TO always ends in .TO.
        """
        is_ca = symbol.endswith(CA_SUFFIXES)
        if self.conv.code == "CA":
            return is_ca
        return not is_ca and (exchange in self._exchanges or not exchange)

    def _info(self, symbol: str) -> dict | None:
        import yfinance as yf
        try:
            with self._lock:
                info = yf.Ticker(symbol).info
            return info if info and info.get("quoteType") else None
        except Exception as exc:                          # noqa: BLE001
            print(f"[markets.na] info({symbol}) failed: {exc!r}"[:200])
            return None


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Yahoo's frame, in the shape the rest of the app expects.

    Two things have to change. The index is timezone-aware (exchange local),
    while every other frame in this app is tz-naive midnight — leaving it would
    make reindex-and-align silently drop every row when a Yahoo series meets a
    Tushare one. And the Dividends / Stock Splits / Capital Gains columns are
    not OHLCV; they ride along on some tickers and not others.
    """
    df = raw.copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"

    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # A halted session prints a zero-volume bar with no trade; the indicators
    # treat it as a real day and it is not one.
    return df.dropna(subset=["Close"])


US = YahooMarket(
    Conventions(
        code="US", name="美股", currency="USD", currency_symbol="$",
        # North America draws gains green — the opposite of the A-share page.
        up_is_red=False,
        # No structural break to work around, so no bar is ever excluded.
        regime_anchor=NO_REGIME_BREAK,
        benchmark="^GSPC", benchmark_name="标普500",
    ),
    US_EXCHANGES,
)

CA = YahooMarket(
    Conventions(
        code="CA", name="加股", currency="CAD", currency_symbol="C$",
        up_is_red=False,
        regime_anchor=NO_REGIME_BREAK,
        benchmark="^GSPTSE", benchmark_name="标普/多伦多综合",
    ),
    CA_EXCHANGES,
)
