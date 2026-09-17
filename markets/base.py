"""
markets/base.py — what a market is, as far as the rest of the app is concerned.

The analysis does not care which exchange a bar came from: run_single_stock_analysis
takes Open/High/Low/Close/Volume and nothing else. What actually differs between
A-shares and North America is a short list of conventions and one data source,
and this is that list.

Conventions worth spelling out, because getting one wrong is silently wrong
rather than loudly broken:

  * `up_is_red` — mainland China draws gains red and losses green, the reverse
    of North America. This is not a theme preference; a green candle means the
    opposite thing in the two markets.
  * `regime_anchor` — A-shares had a structural break at the 924 policy pivot,
    and percentile baselines (BB-width squeeze, RSI bands) must not span it.
    No such break applies to a US stock, so North America anchors at
    Timestamp.min, which means "no break" without needing a special case: no
    bar is ever before it, so every window uses the full history.
  * `benchmark` — beta, alpha and the capture ratios are meaningless against
    the wrong index. 沪深300 for A-shares, S&P 500 for the US, S&P/TSX
    Composite for Canada.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: An anchor no bar can precede — the honest way to say "this market has no
#: structural break to work around" without a sentinel the callers must test.
NO_REGIME_BREAK = pd.Timestamp.min


@dataclass(frozen=True)
class Conventions:
    code: str                    # CN | US | CA
    name: str                    # display name
    currency: str                # CNY | USD | CAD
    currency_symbol: str         # ¥ | $ | C$
    up_is_red: bool
    regime_anchor: pd.Timestamp
    benchmark: str               # symbol understood by this market's fetcher
    benchmark_name: str


@dataclass(frozen=True)
class StockRef:
    """A resolved instrument. `symbol` is always in canonical app form."""
    symbol: str
    name: str
    exchange: str = ""


class Market:
    """
    One market's data access. Subclasses supply the fetchers.

    Every method returns None / [] rather than raising on a missing instrument,
    and raises only when the SOURCE is broken — the distinction the API relies
    on to answer 404 versus 503.
    """

    conv: Conventions

    # `symbol` here is always the bare, source-native code — "600519", "AAPL",
    # "SHOP.TO" — never the namespaced app form. markets.parse() strips that
    # off before anything reaches an adapter.

    def fetch_ohlcv(self, symbol: str, years: int = 3) -> pd.DataFrame | None:
        """Adjusted daily bars indexed by tz-naive date, or None if unavailable."""
        raise NotImplementedError

    def fetch_fundamentals(self, symbol: str, start: str, end: str) -> pd.DataFrame | None:
        """
        Daily PE/PB/market-cap/turnover history, or None when this market has
        no such series. None is a normal answer: the chip model and the PE pane
        both already hide themselves without it, which is the right outcome —
        far better than a number derived from something that does not exist.
        """
        return None

    def resolve(self, symbol: str) -> StockRef | None:
        """Name lookup for one symbol."""
        raise NotImplementedError

    def search(self, query: str, limit: int = 8) -> list[StockRef]:
        """Free-text instrument search. [] when the market has no search."""
        return []

    def fetch_benchmark(self, years: int = 3) -> pd.Series | None:
        df = self.fetch_ohlcv(self.conv.benchmark, years=years)
        return None if df is None or df.empty else df["Close"]
