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
def fetch_china_margin(pro_api, lookback_days: int = 400) -> dict:
    """
    Whole-market A-share margin balance = Σ rzrqye across exchanges per day.

    `pro_api` is a Tushare pro_api() handle (the page passes data_manager's).
    rzrqye is 融资融券余额 in 元; we sum SSE + SZSE (+ BSE if present) and convert
    to 亿元. Returns a daily series over `lookback_days`.
    """
    res = _blank("CN", "China A-share 两融余额", " 亿元", "daily",
                 "Tushare `margin` · Σ 融资融券余额 across exchanges, in 亿元.")
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
    df["rzrqye"] = pd.to_numeric(df["rzrqye"], errors="coerce")
    g = (df.groupby("trade_date", as_index=False)["rzrqye"].sum()
           .rename(columns={"trade_date": "period", "rzrqye": "value"}))
    # 元 → 亿元, and normalise YYYYMMDD → YYYY-MM-DD for display
    g["value"] = g["value"] / 1e8
    g["period"] = g["period"].astype(str).str.replace(
        r"(\d{4})(\d{2})(\d{2})", r"\1-\2-\3", regex=True)
    return _finalise(res, g[["period", "value"]])


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
