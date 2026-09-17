"""
markets — which exchange a symbol belongs to, and how to get its data.

Symbol form
-----------
A-shares keep their bare six-digit code:

    600519          →  CN / 600519
    US:AAPL         →  US / AAPL
    CA:SHOP.TO      →  CA / SHOP.TO

That asymmetry is deliberate. Every A-share ticker already stored — search
history, watchlists, daily_signals rows, the Streamlit app's URLs — is bare and
unprefixed. Making "CN:600519" the canonical form would require migrating all
of it and would break the Streamlit app the same afternoon. Only the new
markets carry a prefix, so nothing that exists today has to change.

Adapters are imported lazily. markets.cn pulls in analysis_engine (hmmlearn,
scikit-learn) and markets.na pulls in yfinance; importing this package should
not cost either of those until a symbol actually needs one.
"""

from __future__ import annotations

import re

from .base import NO_REGIME_BREAK, Conventions, Market, StockRef

__all__ = ["CODES", "Conventions", "Market", "NO_REGIME_BREAK", "StockRef",
           "SYMBOL_RE", "canonical", "get", "parse", "split"]

CODES = ("CN", "US", "CA")

#: Accepts a bare A-share code or a prefixed symbol. Deliberately strict about
#: the characters allowed after the prefix — this value reaches a URL, a cache
#: key and an outbound HTTP request.
#:
#: The prefix is matched as any two letters rather than the three known codes,
#: so that "JP:7203" gets "unknown market JP" from split() instead of being
#: rejected by a route's path pattern as unparseable. Doubles as the FastAPI
#: path pattern, which is why it has to be permissive enough to let a
#: recognisable mistake reach the code that can explain it.
SYMBOL_RE = re.compile(r"^(?:([A-Za-z]{2}):)?([A-Za-z0-9][A-Za-z0-9.\-]{0,15})$")

_A_SHARE = re.compile(r"^\d{6}$")

_CACHE: dict[str, Market] = {}


def split(symbol: str) -> tuple[str, str]:
    """
    ("US", "AAPL") for "US:AAPL"; ("CN", "600519") for a bare six-digit code.

    Raises LookupError — not ValueError — on anything malformed, because every
    caller is an API route that turns LookupError into a 404.
    """
    m = SYMBOL_RE.match((symbol or "").strip())
    if not m:
        raise LookupError(f"无法识别的代码：{symbol!r}")
    prefix, code = m.group(1), m.group(2)

    if prefix:
        prefix = prefix.upper()
        if prefix not in CODES:
            raise LookupError(f"暂不支持的市场：{prefix}（支持 {', '.join(CODES)}）")
        # US tickers are upper case; A-share codes are digits either way.
        return prefix, code.upper() if prefix != "CN" else code
    if _A_SHARE.match(code):
        return "CN", code
    raise LookupError(
        f"{code} 需要带市场前缀，例如 US:{code.upper()} 或 CA:{code.upper()}.TO")


def canonical(symbol: str) -> str:
    """The form stored in the database and used in URLs."""
    market, code = split(symbol)
    return code if market == "CN" else f"{market}:{code}"


def get(market_or_symbol: str) -> Market:
    """The adapter for a market code, or for a symbol belonging to one."""
    code = market_or_symbol.upper()
    if code not in CODES:
        code = split(market_or_symbol)[0]

    if code not in _CACHE:
        if code == "CN":
            from .cn import MARKET
            _CACHE[code] = MARKET
        elif code == "US":
            from .na import US
            _CACHE[code] = US
        else:
            from .na import CA
            _CACHE[code] = CA
    return _CACHE[code]


def parse(symbol: str) -> tuple[Market, str]:
    """(adapter, source-native code) — what every data call needs."""
    market_code, code = split(symbol)
    return get(market_code), code
