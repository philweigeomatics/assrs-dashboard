"""
markets/cn.py — A-shares, through the path the app already uses.

This adapter deliberately adds nothing. It calls watchlist_scan.fetch_frame and
data_manager.get_stock_fundamentals_live exactly as every existing caller does,
so routing A-shares through the market abstraction cannot change a single
number on the Streamlit page or in the API. New markets get a new adapter; this
one stays a wrapper.
"""

from __future__ import annotations

import pandas as pd

# Imported, never re-declared. analysis_engine owns the 924 anchor; this file
# holding its own copy is exactly how the codebase previously ended up with
# 2024-10-14 in one place and 2024-10-10 in another.
from analysis_engine import REGIME_ANCHOR

from .base import Conventions, Market, StockRef

CONV = Conventions(
    code="CN",
    name="A股",
    currency="CNY",
    currency_symbol="¥",
    # Mainland convention: 红涨绿跌.
    up_is_red=True,
    # The 924 policy pivot, settled a week after the National Day re-open.
    regime_anchor=REGIME_ANCHOR,
    benchmark="000300.SH",
    benchmark_name="沪深300",
)


class CNMarket(Market):
    conv = CONV

    def fetch_ohlcv(self, symbol: str, years: int = 3) -> pd.DataFrame | None:
        import watchlist_scan
        # fetch_frame owns the retry/backoff policy and the 3-year lookback the
        # squeeze percentile window is calibrated against; `years` is accepted
        # for interface parity but not overridden here, because changing the
        # history length changes which signals fire.
        return watchlist_scan.fetch_frame(symbol)

    def fetch_fundamentals(self, symbol: str, start: str, end: str) -> pd.DataFrame | None:
        import data_manager
        return data_manager.get_stock_fundamentals_live(symbol, start, end)

    def resolve(self, symbol: str) -> StockRef | None:
        import data_manager
        name = data_manager.get_stock_name_from_db(symbol)
        return StockRef(symbol=symbol, name=name or symbol, exchange="SSE/SZSE")

    def search(self, query: str, limit: int = 8) -> list[StockRef]:
        # A-share search is served from the full 5,600-row stock_basic list the
        # frontend already holds and filters locally, so there is nothing to do
        # here — see /stocks.
        return []

    def fetch_benchmark(self, years: int = 3) -> pd.Series | None:
        import data_manager
        idx = data_manager.get_index_data_live(
            self.conv.benchmark, lookback_days=int(years * 365) + 100)
        return None if idx is None or idx.empty else idx["Close"]


MARKET = CNMarket()
