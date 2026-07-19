"""
market_leverage.py — whole-market leverage (margin-debt / 融资融券) series for
the four markets shown on the Sector Dashboard.

Each fetcher returns a plain dict with a common shape so the UI can render them
uniformly and degrade gracefully when a source is unavailable:

    {
        "market": "CN",                    # short code
        "label":  "China A-share 两融余额", # display name
        "ok":     True/False,
        "unit":   " 亿元",                  # appended to values
        "freq":   "daily" | "weekly" | "monthly",
        "latest": float | None,            # most recent value
        "prev":   float | None,            # prior period value (for the delta)
        "asof":   "2026-07-16",            # label of the latest period
        "series": DataFrame[period(str), value(float)] ascending,  # or None
        "note":   "one-line provenance / caveat",
        "error":  None | str,              # populated when ok is False
    }

Design notes
------------
* Pure functions — no Streamlit import here, so the module is unit-testable on
  its own. The page wraps each call in @st.cache_data.
* China is sourced from Tushare (reliable). US from FINRA's public margin
  statistics table (verified parseable). Japan and Korea do not have a clean,
  datacenter-reachable aggregate we can trust yet, so their fetchers return a
  documented "unavailable" result rather than fabricated numbers — see each
  function's docstring for exactly what source is still required.
* Leverage here means the borrowed-money-to-buy-stock balance:
    - CN: 融资融券余额 (rzrqye), the number Chinese media quote as 两融余额.
    - US: "Debit Balances in Customers' Securities Margin Accounts" (margin debt).
  These are directly comparable in concept (customer margin borrowings).
"""
from __future__ import annotations

import io
import ssl
import urllib.request
from datetime import datetime, timedelta

import pandas as pd


def _blank(market: str, label: str, unit: str, freq: str, note: str,
           error: str | None = None) -> dict:
    return {
        "market": market, "label": label, "ok": False, "unit": unit,
        "freq": freq, "latest": None, "prev": None, "asof": None,
        "series": None, "note": note, "error": error,
    }


def _finalise(res: dict, series: pd.DataFrame) -> dict:
    """Populate latest/prev/asof/series/ok from an ascending [period, value] df."""
    s = series.dropna(subset=["value"]).sort_values("period").reset_index(drop=True)
    if s.empty:
        res["error"] = res.get("error") or "No rows after parsing."
        return res
    res["series"] = s
    res["latest"] = float(s["value"].iloc[-1])
    res["prev"] = float(s["value"].iloc[-2]) if len(s) >= 2 else None
    res["asof"] = str(s["period"].iloc[-1])
    res["ok"] = True
    return res


# ─────────────────────────────────────────────────────────────────────────────
# CHINA — Tushare `margin` (融资融券交易汇总)
# ─────────────────────────────────────────────────────────────────────────────
# Balances vs flows — the identity that makes the flows worth showing:
#     rzye(today) = rzye(yesterday) + rzmre − rzche
# i.e. today's 融资余额 (a balance) = yesterday's balance + 融资买入额 (new
# leveraged buying) − 融资偿还额 (repayments). A flat balance can hide huge gross
# churn; a falling balance driven by a spike in rzche = active de-leveraging /
# forced selling. So we surface rzye, rqye and the daily buy/repay/net flows,
# not just the headline total.
_CN_METRICS = ["rzye", "rqye", "rzrqye", "rzmre", "rzche"]


def fetch_china_margin(pro_api, lookback_days: int = 400) -> dict:
    """
    Whole-market A-share margin data, aggregated on a CONSISTENT exchange basket.

    `pro_api` is a Tushare pro_api() handle (the page passes data_manager's).
    Headline series = 融资融券余额 (rzrqye) in 亿元. Also returns `cn_detail`, a
    per-day DataFrame with 融资余额/融券余额 balances and the 融资买入/偿还/净额
    daily flows (all 亿元) so the UI can show the nuance behind the total.

    Consistent-basket rule (fixes the fake-cliff bug): naively summing rzrqye per
    day halves the total on any day an exchange fails to report. Instead we take
    the "core" exchanges that report on ≥80% of days (SSE+SZSE in practice; BSE
    is sparse/tiny and drops out, matching the conventional 两融 headline) and
    keep ONLY the days where every core exchange reported — incomplete days are
    dropped, never summed partially.
    """
    res = _blank("CN", "China A-share 两融余额", " 亿元", "daily",
                 "Tushare `margin` · 融资融券余额 (一致交易所口径), 亿元.")
    res["cn_detail"] = None
    res["core_exchanges"] = None
    if pro_api is None:
        res["error"] = "Tushare not initialised."
        return res
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
    try:
        df = pro_api.margin(start_date=start, end_date=end)
    except Exception as exc:                       # tier / credit / network
        res["error"] = str(exc)
        return res
    if df is None or df.empty or "rzrqye" not in df.columns:
        res["error"] = "margin endpoint returned no rzrqye column (data tier?)."
        return res
    df = df.copy()
    metric_cols = [c for c in _CN_METRICS if c in df.columns]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Core basket = exchanges present on ≥80% of trade dates; valid dates =
    # those where every core exchange reported rzrqye. This is the fix.
    piv = df.pivot_table(index="trade_date", columns="exchange_id",
                         values="rzrqye", aggfunc="sum").sort_index()
    coverage = piv.notna().mean()
    core = coverage[coverage >= 0.8].index.tolist() or piv.columns.tolist()
    valid_dates = piv[core].dropna(how="any").index
    n_dropped = int(len(piv) - len(valid_dates))
    sub = df[df["exchange_id"].isin(core) & df["trade_date"].isin(valid_dates)]
    agg = sub.groupby("trade_date")[metric_cols].sum().sort_index()
    if agg.empty:
        res["error"] = "No trade dates with a complete exchange basket."
        return res

    agg_yi = agg / 1e8   # 元 → 亿元
    periods = agg_yi.index.astype(str).str.replace(
        r"(\d{4})(\d{2})(\d{2})", r"\1-\2-\3", regex=True)

    # Detail frame: balances + daily financing flows, all in 亿元.
    detail = pd.DataFrame({"period": periods.values})
    for c in metric_cols:
        detail[c] = agg_yi[c].values
    if "rzmre" in detail.columns and "rzche" in detail.columns:
        detail["net_fin"] = detail["rzmre"] - detail["rzche"]   # 净融资流入
    res["cn_detail"] = detail
    res["core_exchanges"] = core
    res["note"] = (
        f"Tushare `margin` · 一致口径（{'+'.join(map(str, core))} 齐全的交易日）· "
        f"融资融券余额, 亿元"
        + (f"; 已剔除 {n_dropped} 个交易所数据不齐的交易日" if n_dropped else "")
    )

    total = pd.DataFrame({"period": periods.values, "value": agg_yi["rzrqye"].values})
    return _finalise(res, total)


# ─────────────────────────────────────────────────────────────────────────────
# US — FINRA margin statistics (monthly margin debt)
# ─────────────────────────────────────────────────────────────────────────────
_FINRA_URL = "https://www.finra.org/investors/margin-statistics"
_UA = {"User-Agent": "Mozilla/5.0 (compatible; ASSRS-dashboard/1.0)"}


def fetch_us_margin(timeout: int = 30) -> dict:
    """
    US customer margin debt from FINRA's public margin-statistics table.

    The page carries a single HTML table whose first data column is
    "Debit Balances in Customers' Securities Margin Accounts" (US$ millions,
    monthly). We convert to US$ bn. ~13 recent months are published inline; the
    full history lives in an .xlsx we deliberately do NOT pull (the inline table
    is smaller and more robust for a trend card).
    """
    res = _blank("US", "US Margin Debt (FINRA)", " $bn", "monthly",
                 "FINRA Rule 4521 · customer securities-margin debit balances, in US$ bn.")
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(_FINRA_URL, headers=_UA)
        html = urllib.request.urlopen(req, timeout=timeout, context=ctx).read()
        html = html.decode("utf-8", "ignore")
        tables = pd.read_html(io.StringIO(html))
    except Exception as exc:
        res["error"] = str(exc)
        return res
    # Find the table whose columns include the debit-balance measure.
    debit_col = None
    tbl = None
    for t in tables:
        for c in t.columns:
            if "debit balances" in str(c).lower():
                tbl, debit_col = t, c
                break
        if tbl is not None:
            break
    if tbl is None or debit_col is None:
        res["error"] = "FINRA page loaded but the margin-debt table was not found."
        return res
    period_col = tbl.columns[0]                     # "Month/Year", e.g. Jun-26
    d = tbl[[period_col, debit_col]].copy()
    d.columns = ["period", "value"]
    d["value"] = pd.to_numeric(d["value"], errors="coerce") / 1000.0   # $mn → $bn
    # FINRA lists newest-first; sort chronologically by parsing "Mon-YY".
    def _key(p):
        try:
            return datetime.strptime(str(p).strip(), "%b-%y")
        except Exception:
            return pd.NaT
    d["_k"] = d["period"].map(_key)
    d = d.dropna(subset=["_k"]).sort_values("_k")
    d["period"] = d["_k"].dt.strftime("%Y-%m")
    return _finalise(res, d[["period", "value"]])


# ─────────────────────────────────────────────────────────────────────────────
# JAPAN — pending a clean aggregate source
# ─────────────────────────────────────────────────────────────────────────────
def fetch_japan_margin() -> dict:
    """
    NOT YET WIRED to a trustworthy aggregate.

    JPX's readily-linked daily file
    (statistics-equities/margin/…/mtdaily…​.xls) is *per-issue* outstanding
    margin in SHARES — you cannot sum shares across different issues into a
    market total, so it is the wrong granularity for a leverage figure.

    The correct source is the market-wide 信用取引現在高 (margin transaction
    outstanding, in ¥) published *weekly*:
      • JSDA (日本証券業協会) aggregate margin balances, or
      • the TSE weekly market-total file (not the per-issue mtdaily file).
    Both need a dedicated parser and, from a non-JP datacenter IP, may be
    rate-limited or geo-restricted — hence deferred rather than faked.
    """
    return _blank(
        "JP", "Japan Margin Balance", " ¥tn", "weekly",
        "Source pending: needs JSDA/TSE weekly 信用取引現在高 (¥ aggregate), "
        "not the per-issue share-count file.",
        error="No trusted aggregate source wired yet.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# KOREA — pending the correct KRX statistic id
# ─────────────────────────────────────────────────────────────────────────────
def fetch_korea_margin() -> dict:
    """
    NOT YET WIRED to a confirmed endpoint.

    KRX's data portal (data.krx.co.kr) serves downloads via a two-step OTP flow
    (GenerateOTP → download.cmd) which is reachable, but the market-wide
    margin-loan balance (신용거래융자 잔고) needs the exact statistic id (`bld`,
    e.g. dbms/MDC/STAT/standard/MDCSTATxxxxx). That id has to be captured from
    the live KRX margin-statistics menu (network tab) — blind guesses return
    HTTP 400. KOFIA FreeSIS (freesis.kofia.or.kr) is an alternative aggregate.
    Deferred rather than faked until the id is confirmed against the live site.
    """
    return _blank(
        "KR", "Korea Margin Loans", " ₩tn", "daily",
        "Source pending: needs the confirmed KRX `bld` for 신용거래융자 잔고 "
        "(or KOFIA FreeSIS aggregate).",
        error="Correct KRX statistic id not confirmed yet.",
    )


def fetch_all(pro_api) -> list[dict]:
    """Convenience: all four markets in display order (CN, US, JP, KR)."""
    return [
        fetch_china_margin(pro_api),
        fetch_us_margin(),
        fetch_japan_margin(),
        fetch_korea_margin(),
    ]
