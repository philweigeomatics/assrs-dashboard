"""
api/questrade_api.py — the MyQuestrade page's data.

Three sources, each used for the one thing it is actually good at:

    Questrade   what you own, in which account, at what cost — the only source
                that knows, and the only one that is authoritative
    Yahoo       adjusted price history, and the ETF/stock distinction, which
                Questrade's securityType enum cannot express (every ETF comes
                back as "Stock")
    Questrade   the USD/CAD rate, read out of its own combined balances, so the
                page's totals match the statement rather than a mid-market
                quote a few tenths of a percent away

Why the risk numbers do NOT come from Questrade's candles: those are
unadjusted. A holding that split 4:1 puts a -75% day into the return series,
and every volatility, beta and drawdown downstream of it is then wrong in a way
that looks like a bad week rather than like a bug.

Currency runs through all of it. A Canadian book holding US names carries
USD/CAD whether or not the owner thinks of it that way, so every series is
converted to the base currency before a single statistic is computed — the
benchmark included. Beta against an S&P 500 measured in USD, from a portfolio
measured in CAD, is beta plus an exchange rate.
"""

from __future__ import annotations

import time

import pandas as pd

import portfolio as pf
import questrade as qt

#: How long a Yahoo instrument classification is good for. A stock does not
#: become an ETF, so this is only re-read to pick up new holdings.
KIND_TTL_S = 24 * 3600

#: Years of adjusted history pulled for the risk report.
HISTORY_YEARS = 3

#: Currency of each benchmark, so it can be put into the book's base currency
#: before anything is regressed against it.
BENCHMARK_CCY = {"^GSPC": "USD", "^IXIC": "USD", "^GSPTSE": "CAD"}

_kind_cache: dict[str, tuple[float, dict]] = {}


class SupabaseStore(qt.TokenStore):
    """
    The rotating token, in the one table the browser can never read.

    Written through service_role; `questrade_tokens` has RLS on with no policy,
    so anon and authenticated are denied by default (see the migration).
    """

    TABLE = "questrade_tokens"

    def __init__(self, app_user_id: int):
        self.app_user_id = int(app_user_id)
        self.key = f"questrade:{self.app_user_id}"

    def _db(self):
        from db_manager import db
        return db

    def load(self) -> dict | None:
        db = self._db()
        try:
            rows = db.read_table(self.TABLE, filters={"app_user_id": self.app_user_id},
                                 limit=1)
        except Exception as exc:                                   # noqa: BLE001
            raise qt.QuestradeError(_missing_table(exc))
        if rows is None or rows.empty:
            return None
        return rows.iloc[0].to_dict()

    def save(self, record: dict) -> None:
        db = self._db()
        row = {"app_user_id": self.app_user_id,
               "refresh_token": record.get("refresh_token") or "",
               "access_token": record.get("access_token") or "",
               "api_server": record.get("api_server") or "",
               "expires_at": float(record.get("expires_at") or 0),
               "connected_at": float(record.get("connected_at") or 0),
               "last_error": record.get("last_error") or ""}
        try:
            db.insert_records(self.TABLE, [row], upsert=True)
        except Exception as exc:                                   # noqa: BLE001
            raise qt.QuestradeError(_missing_table(exc))

    def clear(self) -> None:
        try:
            self._db().delete_records(self.TABLE, {"app_user_id": self.app_user_id})
        except Exception:                                          # noqa: BLE001
            pass


def _missing_table(exc: Exception) -> str:
    text = str(exc)
    if "questrade_tokens" in text or "PGRST205" in text or "does not exist" in text:
        return ("questrade_tokens 表尚未创建 —— 请在 Supabase SQL 编辑器运行 "
                "supabase/migrations/20260918_questrade.sql")
    return f"读取 Questrade 连接状态失败：{text}"[:200]


def client(app_user_id: int) -> qt.Questrade:
    return qt.Questrade(SupabaseStore(app_user_id))


# ── the portfolio ────────────────────────────────────────────────────────────
def snapshot(app_user_id: int, base: str = "CAD") -> dict:
    """Every account, consolidated into one book."""
    c = client(app_user_id)
    accounts = c.accounts()
    if not accounts:
        raise LookupError("这个 Questrade 登录下没有找到任何账户")

    positions: dict[str, list[dict]] = {}
    balances: dict[str, dict] = {}
    for a in accounts:
        positions[a["id"]] = c.positions(a["id"])
        balances[a["id"]] = c.balances(a["id"])

    meta = _meta(c, positions)
    rates, source = _rates(balances, base)

    book = pf.consolidate(accounts, positions, balances, meta,
                          base=base, rates=rates, rate_source=source)
    book["as_of"] = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    # Questrade marks quotes it could not stream. Worth surfacing: a delayed
    # price is fine for a weekly review and wrong for a same-day decision.
    book["delayed"] = any(p.get("isRealTime") is False
                          for rows in positions.values() for p in rows)
    return book


def _meta(c: qt.Questrade, positions: dict[str, list[dict]]) -> dict:
    """{symbol: {name, currency, exchange, kind, yahoo}} for everything held."""
    ids, by_symbol = {}, {}
    for rows in positions.values():
        for p in rows:
            s = str(p.get("symbol") or "").strip().upper()
            if s and p.get("symbolId"):
                ids[s] = int(p["symbolId"])

    detail = {}
    if ids:
        for row in c.symbols(list(ids.values())):
            detail[str(row.get("symbol") or "").strip().upper()] = row

    for symbol, symbol_id in ids.items():
        d = detail.get(symbol, {})
        exchange = str(d.get("listingExchange") or "")
        yahoo = pf.to_yahoo(symbol, exchange)
        by_symbol[symbol] = {
            "symbol_id": symbol_id,
            "name": str(d.get("description") or symbol),
            "currency": str(d.get("currency") or "").upper(),
            "exchange": exchange,
            "yahoo": yahoo,
            "kind": pf.kind_of(_quote_type(yahoo), d.get("securityType")),
        }
    return by_symbol


def _quote_type(yahoo: str | None) -> str | None:
    """
    Yahoo's ETF/EQUITY flag, cached for a day.

    `fast_info` rather than `info`: the same field, one cheap request instead
    of the full profile payload, which for twenty holdings is the difference
    between a page that loads and one that times out.
    """
    if not yahoo:
        return None
    hit = _kind_cache.get(yahoo)
    if hit and time.time() - hit[0] < KIND_TTL_S:
        return hit[1].get("quote_type")

    value = None
    try:
        import yfinance as yf
        value = getattr(yf.Ticker(yahoo).fast_info, "quote_type", None)
    except Exception:                                              # noqa: BLE001
        value = None
    _kind_cache[yahoo] = (time.time(), {"quote_type": value})
    return value


def _rates(balances: dict[str, dict], base: str) -> tuple[dict, str]:
    """
    Currency rates into `base`, preferring the broker's own.

    Questrade states each account's whole equity in both currencies, so the
    ratio is exactly the rate it used — which keeps this page's totals equal to
    the statement instead of a few tenths of a percent away on a different
    mid-market quote. Yahoo is the fallback, and which one was used is
    reported, because a total that silently changed source is a total nobody
    can reconcile.
    """
    quote = "USD" if base.upper() == "CAD" else "CAD"
    for raw in balances.values():
        rate = pf.implied_fx(raw.get("combinedBalances") or [], base, quote)
        if rate:
            return {base.upper(): 1.0, quote: rate}, "Questrade 合并余额"

    pair = f"{quote}{base.upper()}=X"
    try:
        import yfinance as yf
        last = getattr(yf.Ticker(pair).fast_info, "last_price", None)
        if last and 0.2 < float(last) < 5:
            return {base.upper(): 1.0, quote: float(last)}, f"Yahoo {pair}"
    except Exception:                                              # noqa: BLE001
        pass
    # No rate at all: consolidate() leaves those holdings out of the total and
    # says so, rather than counting US dollars as Canadian ones.
    return {base.upper(): 1.0}, "不可用"


# ── the risk report ──────────────────────────────────────────────────────────
def risk_report(app_user_id: int, benchmark: str = "^GSPC",
                base: str = "CAD") -> dict:
    """Today's book, replayed against an index — both in the base currency."""
    if benchmark not in pf.BENCHMARKS:
        raise LookupError(f"未知的基准：{benchmark}")

    book = snapshot(app_user_id, base=base)
    weights, currencies = {}, {}
    for h in book["holdings"]:
        if not h["yahoo"] or not h["market_value_base"]:
            continue
        weights[h["yahoo"]] = weights.get(h["yahoo"], 0.0) + h["market_value_base"]
        currencies[h["yahoo"]] = h["currency"]
    if not weights:
        raise LookupError("没有任何持仓能取得行情，无法计算风险指标")

    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    currencies[benchmark] = BENCHMARK_CCY.get(benchmark, base)

    prices = _history(list(weights) + [benchmark], currencies, base)
    report = pf.risk(weights, prices, benchmark)
    report["base"] = base
    report["as_of"] = book["as_of"]
    report["totals"] = book["totals"]
    # Names carried through so the table does not read as a list of tickers.
    labels = {h["yahoo"]: h["name"] for h in book["holdings"] if h["yahoo"]}
    for row in report["holdings"]:
        row["name"] = labels.get(row["symbol"], row["symbol"])
    report["warnings"] = book["warnings"]
    return report


def _history(symbols: list[str], currencies: dict[str, str], base: str) -> pd.DataFrame:
    """
    Adjusted closes for everything, expressed in `base`.

    auto_adjust=True, so splits and dividends are already in — the reason this
    goes to Yahoo rather than to Questrade's candles, which are neither.
    """
    import yfinance as yf

    base = base.upper()
    needs_fx = {c.upper() for c in currencies.values()} - {base, ""}
    fx_pairs = {c: f"{c}{base}=X" for c in needs_fx}
    tickers = sorted(set(symbols) | set(fx_pairs.values()))

    raw = yf.download(tickers, period=f"{HISTORY_YEARS}y", auto_adjust=True,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError("行情数据暂时读取不到 —— 请稍后重试")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(raw.columns, pd.MultiIndex):
        close.columns = tickers[:1]

    out = pd.DataFrame(index=close.index)
    for symbol in symbols:
        if symbol not in close.columns:
            continue
        series = close[symbol]
        ccy = (currencies.get(symbol) or base).upper()
        if ccy != base and ccy in fx_pairs:
            pair = fx_pairs[ccy]
            if pair not in close.columns:
                continue          # no rate → leave it out rather than mis-scale
            series = series * close[pair].reindex(series.index).ffill()
        out[symbol] = series
    return out.dropna(how="all")
