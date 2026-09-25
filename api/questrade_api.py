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

import threading
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

#: How long a fetched book is reused. Positions move with the tape, so this is
#: short — but it must be longer than one page load, because the page fires
#: five queries at once and every one of them needs the book.
BOOK_TTL_S = 150

_kind_cache: dict[str, tuple[float, dict]] = {}
_book_cache: dict[tuple, tuple[float, dict]] = {}
_book_lock = threading.Lock()


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
    """
    Every account, consolidated into one book — fetched at most once per
    BOOK_TTL_S per user.

    The caching is here rather than only in the route because the route is not
    the only caller: the risk, exposure and allocation reports each need the
    book, and the page asks for all of them at once. Without this, opening
    MyQuestrade made four independent passes over every account, four sets of
    Questrade calls, and four chances to be the request that finds the access
    token stale and tries to refresh it. Refreshing is the one operation that
    can permanently break the connection if two of them race, so doing it
    once instead of four times is not only about speed.
    """
    key = (int(app_user_id), base)
    with _book_lock:
        hit = _book_cache.get(key)
        if hit and time.time() - hit[0] < BOOK_TTL_S:
            return hit[1]

    book = _fetch_snapshot(app_user_id, base)
    with _book_lock:
        _book_cache[key] = (time.time(), book)
    return book


def forget(app_user_id: int) -> None:
    """Drop this user's cached book — on connect, disconnect, or a manual refresh."""
    with _book_lock:
        for key in [k for k in _book_cache if k[0] == int(app_user_id)]:
            _book_cache.pop(key, None)


def _fetch_snapshot(app_user_id: int, base: str) -> dict:
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


#: Which sleeve of the book a report covers. "仅股票" matters: a book that is
#: 60% index funds has a beta near 1 whatever its stocks do, and the stock
#: sleeve's own risk is invisible inside the whole-book number.
SCOPES = {"all": "全部持仓", "stock": "仅股票", "etf": "仅 ETF"}


# ── the risk report ──────────────────────────────────────────────────────────
def risk_report(app_user_id: int, benchmark: str = "^GSPC",
                base: str = "CAD", scope: str = "all") -> dict:
    """
    Today's book, replayed against an index — both in the base currency.

    `scope` narrows it to the stock sleeve or the ETF sleeve. That is not a
    cosmetic filter: a book that is 60% index funds has a beta near 1 whatever
    the stocks are doing, and "what is my stock picking actually costing me in
    volatility" is a question the whole-book number cannot answer.
    """
    if benchmark not in pf.BENCHMARKS:
        raise LookupError(f"未知的基准：{benchmark}")
    if scope not in SCOPES:
        raise LookupError(f"未知的范围：{scope}")

    book = snapshot(app_user_id, base=base)
    chosen = [h for h in book["holdings"]
              if h["yahoo"] and h["market_value_base"]
              and (scope == "all" or h["group"] == scope)]
    if not chosen:
        raise LookupError(f"{SCOPES[scope]}中没有可取得行情的持仓")

    weights, currencies = {}, {}
    for h in chosen:
        weights[h["yahoo"]] = weights.get(h["yahoo"], 0.0) + h["market_value_base"]
        currencies[h["yahoo"]] = h["currency"]
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    currencies[benchmark] = BENCHMARK_CCY.get(benchmark, base)

    prices = _history(list(weights) + [benchmark], currencies, base)
    report = pf.risk(weights, prices, benchmark)
    report["base"] = base
    report["scope"] = scope
    report["scope_label"] = SCOPES[scope]
    report["as_of"] = book["as_of"]
    report["totals"] = book["totals"]
    # What share of the WHOLE book this sleeve is, which is the context the
    # covered_pct inside the sleeve cannot give.
    whole = sum(h["market_value_base"] for h in book["holdings"]) or 1.0
    report["sleeve_pct"] = round(total / whole * 100, 1)
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


# ── industry exposure ────────────────────────────────────────────────────────
_sector_cache: dict[str, tuple[float, dict]] = {}


def _sector_lookup(yahoo: str, group: str) -> dict:
    """
    {sector, industry} for a stock; {weights} for an ETF. Cached for a day.

    Two different Yahoo calls, because they are two different questions. A
    stock has one sector. An ETF has eleven, published as `sector_weightings`,
    and that is what makes look-through possible — without it a book that is
    half VFV reports half its exposure as "ETF", which is not a sector and not
    an answer.
    """
    hit = _sector_cache.get(yahoo)
    if hit and time.time() - hit[0] < KIND_TTL_S:
        return hit[1]

    out: dict = {}
    try:
        import yfinance as yf
        ticker = yf.Ticker(yahoo)
        if group == "etf":
            weights = getattr(ticker.funds_data, "sector_weightings", None)
            out = {"weights": dict(weights)} if weights else {}
        else:
            info = ticker.info or {}
            out = {"sector": info.get("sector"), "industry": info.get("industry")}
    except Exception:                                              # noqa: BLE001
        out = {}
    _sector_cache[yahoo] = (time.time(), out)
    return out


def exposure_report(app_user_id: int, base: str = "CAD") -> dict:
    """Sector exposure with ETFs looked through, plus a stocks-only industry cut."""
    import exposure

    book = snapshot(app_user_id, base=base)
    lookups = {}
    for h in book["holdings"]:
        if h["yahoo"]:
            lookups[h["symbol"]] = _sector_lookup(h["yahoo"], h["group"])

    report = exposure.analyse(book["holdings"], lookups)
    report["base"] = base
    report["as_of"] = book["as_of"]
    names = {h["symbol"]: h["name"] for h in book["holdings"]}
    for row in report["rows"]:
        row["holdings"] = [{"symbol": s, "name": names.get(s, s)} for s in row["holdings"]]
    return report


# ── allocation ───────────────────────────────────────────────────────────────
def _panel(app_user_id: int, base: str, scope: str):
    """(book, selected holdings, return panel, current weights) for one scope."""
    import portfolio as pf

    if scope not in SCOPES:
        raise LookupError(f"未知的范围：{scope}")
    book = snapshot(app_user_id, base=base)
    chosen = [h for h in book["holdings"]
              if h["yahoo"] and h["market_value_base"]
              and (scope == "all" or h["group"] == scope)]
    if len(chosen) < 2:
        raise LookupError(f"{SCOPES[scope]}中可计价的标的少于两只，无法做配置分析")

    weights, currencies = {}, {}
    for h in chosen:
        weights[h["yahoo"]] = weights.get(h["yahoo"], 0.0) + h["market_value_base"]
        currencies[h["yahoo"]] = h["currency"]
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    # The benchmark rides along purely to define the trading calendar; the
    # optimiser never sees it.
    bench = "^GSPC"
    currencies[bench] = BENCHMARK_CCY[bench]
    prices = _history(list(weights) + [bench], currencies, base)
    rets, dropped = pf.returns(prices, list(weights), bench)
    return book, chosen, rets.drop(columns=[bench]), weights, dropped


def optimise_report(app_user_id: int, *, base: str = "CAD", scope: str = "all",
                    method: str = "min_var", cap: float = 0.25) -> dict:
    """A target allocation, and whether it has ever been worth switching to."""
    import optimise as opt
    import pandas as pd

    book, chosen, rets, weights, dropped = _panel(app_user_id, base, scope)
    names = {h["yahoo"]: h["name"] for h in chosen if h["yahoo"]}
    current = pd.Series({c: weights.get(c, 0.0) for c in rets.columns}, dtype=float)

    report = opt.suggest(rets, current, method, cap=cap, names=names)
    try:
        report["walk_forward"] = opt.walk_forward(
            rets, current, ["min_var", "risk_parity", "equal", "max_sharpe"], cap=cap)
    except LookupError as exc:
        # The evidence is the point, so its absence is reported, not hidden.
        report["walk_forward"] = None
        report["walk_forward_error"] = str(exc)

    report.update({
        "base": base, "scope": scope, "scope_label": SCOPES[scope],
        "as_of": book["as_of"], "sessions": int(len(rets)),
        "from": str(rets.index[0].date()), "to": str(rets.index[-1].date()),
        "methods": [{"id": k, **v} for k, v in opt.METHODS.items()],
        "excluded": [{"symbol": k, "reason": v} for k, v in sorted(dropped.items())],
    })
    return report


# ── keeping the chain alive ──────────────────────────────────────────────────
#: Questrade's refresh token is valid for THREE DAYS from the moment it is
#: issued, and every exchange issues a new one good for another three. (The
#: manual token from the App Hub is separate: it lasts seven days from
#: generation, but only until it is used once.)
#:
#: So the chain survives indefinitely while somebody keeps refreshing it — and
#: dies quietly if nobody does. A connection that only refreshes when the page
#: is opened therefore breaks after a long weekend, which looks exactly like
#: "the token expires every few days" and is indistinguishable, from the
#: outside, from a bug.
REFRESH_TTL_DAYS = 3


def keepalive() -> dict:
    """
    Refresh every connected user's token, so no chain lapses from disuse.

    Called on a schedule. Deliberately served by the API rather than by a job
    that talks to Questrade itself: the refresh token is single-use, and two
    processes exchanging the same one leaves one of them holding a dead token
    with no way to get another. Inside the API the per-user lock in
    questrade.py already serialises it against live page traffic.

    Returns a per-user result rather than raising, because one broken
    connection must not stop the others being refreshed.
    """
    from db_manager import db

    try:
        rows = db.read_table(SupabaseStore.TABLE, columns="app_user_id")
    except Exception as exc:                                       # noqa: BLE001
        raise qt.QuestradeError(_missing_table(exc))
    if rows is None or rows.empty:
        return {"checked": 0, "refreshed": 0, "results": []}

    results = []
    for uid in sorted({int(v) for v in rows["app_user_id"].tolist()}):
        try:
            # _tokens() exchanges whenever the ACCESS token is stale, which on
            # a daily ping it always is — and an exchange is what mints the
            # next refresh token. Nothing else needs to happen.
            client(uid)._tokens()
            results.append({"user": uid, "ok": True})
            forget(uid)
        except Exception as exc:                                   # noqa: BLE001
            results.append({"user": uid, "ok": False,
                            "error": f"{type(exc).__name__}: {exc}"[:200]})
    return {"checked": len(results),
            "refreshed": sum(1 for r in results if r["ok"]),
            "results": results}

# ── the activity ledger ──────────────────────────────────────────────────────
#: Questrade's own activity types, in the order a statement reads.
ACTIVITY_TYPES = [
    ("Trades", "买卖"),
    ("Dividends", "股息"),
    ("Interest", "利息"),
    ("Deposits", "存入"),
    ("Withdrawals", "转出"),
    ("Fees and rebates", "费用与返还"),
    ("Corporate actions", "公司行动"),
    ("Other", "其他"),
]
TYPE_LABEL = dict(ACTIVITY_TYPES)

#: Two legs of one internal transfer can settle a day or two apart.
TRANSFER_DAYS = 3


def _day(value) -> str:
    return str(value or "")[:10]


def _num(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _row(raw: dict) -> dict:
    """One activity, flattened to what a ledger actually shows."""
    return {
        # Trade date is the date of disposition, which is the one that
        # decides which tax year a gain falls in. Settlement is kept
        # alongside because cash movements are dated by it.
        "date": _day(raw.get("tradeDate")),
        "settled": _day(raw.get("settlementDate")),
        "type": str(raw.get("type") or "Other"),
        "type_label": TYPE_LABEL.get(str(raw.get("type") or ""), "其他"),
        "action": str(raw.get("action") or "").strip(),
        "symbol": str(raw.get("symbol") or "").strip(),
        "description": str(raw.get("description") or "").strip(),
        "quantity": _num(raw.get("quantity")),
        "price": _num(raw.get("price")),
        "gross": _num(raw.get("grossAmount")),
        "commission": _num(raw.get("commission")),
        "net": _num(raw.get("netAmount")),
        "currency": str(raw.get("currency") or "").strip() or "CAD",
        "internal": False,
    }


def _mark_internal(accounts: list[dict]) -> list[dict]:
    """
    Pair the two legs of a move between the user's own accounts.

    Money going from the margin account into the TFSA shows up twice: a
    Withdrawal in one and a Deposit in the other, same amount, same currency,
    a day or two apart. Adding the Deposits column across accounts therefore
    counts it as new money arriving, which it is not. On one real year the
    accounts showed USD 16,588.56 of deposits between them and USD 15,788.56
    of it had simply come from the margin account — 800 dollars actually
    arrived.

    Each leg is consumed once, so two genuinely separate 1,000-dollar moves
    on the same day pair up one-to-one rather than collapsing into one.
    """
    outs, ins = [], []
    for acct in accounts:
        for row in acct["rows"]:
            if row["type"] == "Withdrawals" and (row["net"] or 0) < 0:
                outs.append((acct, row))
            elif row["type"] == "Deposits" and (row["net"] or 0) > 0:
                ins.append((acct, row))

    pairs = []
    taken = set()
    for out_acct, out_row in outs:
        for i, (in_acct, in_row) in enumerate(ins):
            if i in taken or in_acct is out_acct:
                continue
            if in_row["currency"] != out_row["currency"]:
                continue
            if abs(abs(out_row["net"]) - in_row["net"]) > 0.005:
                continue
            if abs(_days_between(out_row["date"], in_row["date"])) > TRANSFER_DAYS:
                continue
            taken.add(i)
            out_row["internal"] = True
            in_row["internal"] = True
            pairs.append({
                "amount": round(in_row["net"], 2),
                "currency": in_row["currency"],
                "date": in_row["date"],
                "from": out_acct["label"], "to": in_acct["label"],
            })
            break
    return pairs


def _days_between(a: str, b: str) -> int:
    from datetime import date
    try:
        d1 = date.fromisoformat(a)
        d2 = date.fromisoformat(b)
    except ValueError:
        return 999
    return (d2 - d1).days


def _summarise(rows: list[dict]) -> dict:
    """
    Totals per type, per currency — never across them.

    CAD and USD are different money and one real year held both. A single
    summed column would be a number that does not exist.
    """
    by_type: dict[str, dict] = {}
    for r in rows:
        bucket = by_type.setdefault(r["type"], {
            "type": r["type"], "type_label": r["type_label"],
            "count": 0, "by_currency": {}})
        bucket["count"] += 1
        ccy = bucket["by_currency"].setdefault(
            r["currency"], {"currency": r["currency"], "net": 0.0,
                            "internal_net": 0.0, "count": 0})
        ccy["count"] += 1
        ccy["net"] += r["net"] or 0.0
        if r["internal"]:
            ccy["internal_net"] += r["net"] or 0.0

    order = [t for t, _ in ACTIVITY_TYPES]
    out = []
    for t in order:
        if t not in by_type:
            continue
        b = by_type.pop(t)
        for c in b["by_currency"].values():
            c["net"] = round(c["net"], 2)
            c["internal_net"] = round(c["internal_net"], 2)
            # What actually crossed the household boundary.
            c["external_net"] = round(c["net"] - c["internal_net"], 2)
        b["by_currency"] = sorted(b["by_currency"].values(),
                                  key=lambda c: c["currency"])
        out.append(b)
    out.extend(by_type.values())      # any type Questrade adds later
    return {"by_type": out, "count": len(rows)}


def transactions(app_user_id: int, year: int) -> dict:
    """
    One calendar year of activity, per account, for putting a return together.

    Deliberately NOT computed here: adjusted cost base, capital gains, or
    anything else that belongs on a tax form. Superficial-loss rules,
    identical property held across accounts, and the exchange rate that
    applies to each leg all change the answer, and a plausible wrong number
    on a tax return is worse than no number. This assembles the records; a
    person or their accountant does the rest.
    """
    from datetime import datetime, timezone

    c = client(app_user_id)
    start = datetime(int(year), 1, 1, tzinfo=timezone.utc)
    end = datetime(int(year) + 1, 1, 1, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if end > now:
        end = now

    accounts = []
    for a in c.accounts():
        raw = c.activities(a["id"], start, end)
        rows = sorted((_row(r) for r in raw),
                      key=lambda r: (r["date"], r["type"], r["symbol"]))
        accounts.append({
            "id": a["id"],
            # Only the tail reaches the browser: enough to tell two accounts
            # of the same type apart, not enough to be an account number.
            "tail": a["id"][-4:],
            "type": a["type"],
            "label": a["label"],
            "registered": a["type"] in qt.REGISTERED,
            "rows": rows,
            "summary": _summarise(rows),
        })

    transfers = _mark_internal(accounts)
    for acct in accounts:
        acct["summary"] = _summarise(acct["rows"])

    every = [r for a in accounts for r in a["rows"]]
    return {
        "year": int(year),
        "partial": end < datetime(int(year) + 1, 1, 1, tzinfo=timezone.utc),
        "through": end.date().isoformat(),
        "accounts": accounts,
        "internal_transfers": transfers,
        "summary": _summarise(every),
        "types": [{"type": t, "label": lab} for t, lab in ACTIVITY_TYPES],
    }

