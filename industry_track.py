"""
industry_track.py — which industry one stock has been trading with, over ten years.

A fibreglass maker can fall with property for a year and then rise with
semiconductors, because the same plant makes construction mesh and the cloth
inside circuit boards. This follows one stock against a handful of industry
indexes, through rolling correlation, so those switches show.

The market is taken out first. A stock and an industry index both mostly
follow the market, so their raw correlation sits between 0.4 and 0.8 almost
all the time, whichever industry it is, and a switch from one industry to
another barely moves it. What is left once the market's share of every move is
removed (the partial correlation, holding 中证全指 or 标普500 fixed) is the
industry link, and that is where a switch shows.

Weekly returns over 26 weeks is the default. Daily over 60 sessions reacts
faster and is noisier. Monthly is not offered: a year of months is twelve
observations, too few to tell a correlation of 0.5 from none.

Lives at the repository root, not under api/, so Streamlit pages can import it
too; the API only wraps it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

import portfolio_window as pw

#: Industry indexes, per market. An allow-list because an open parameter is an
#: open Tushare or Yahoo call. CN: 中证 indexes, every one with ten years of
#: history (申万's sw_daily needs Tushare points this account does not have).
#: US: sector ETFs, since Yahoo has no index for most of these.
INDUSTRIES = {
    "CN": {
        "931775.CSI": "房地产",
        "931009.CSI": "建筑材料",
        "399995.SZ": "基建工程",
        "930606.CSI": "钢铁",
        "000819.SH": "有色金属",
        "399998.SZ": "煤炭",
        "000813.CSI": "化工",
        "H30184.CSI": "半导体",
        "930652.CSI": "电子",
        "930651.CSI": "计算机",
        "931160.CSI": "通信设备",
        "399808.SZ": "新能源",
        "399976.SZ": "新能源车",
        "930607.CSI": "汽车",
        "000812.CSI": "机械",
        "399967.SZ": "军工",
        "H30199.CSI": "电力",
        "399986.SZ": "银行",
        "399975.SZ": "证券",
        "399997.SZ": "白酒",
        "000807.CSI": "食品饮料",
        "399989.SZ": "医疗",
        "930697.CSI": "家用电器",
        "399971.SZ": "传媒",
        "000949.CSI": "农业",
    },
    "US": {
        "XLK": "科技 XLK",
        "SMH": "半导体 SMH",
        "IGV": "软件 IGV",
        "XLC": "通信服务 XLC",
        "XLF": "金融 XLF",
        "KRE": "地区银行 KRE",
        "XLE": "能源 XLE",
        "XLB": "原材料 XLB",
        "XLI": "工业 XLI",
        "ITA": "航空军工 ITA",
        "XLY": "可选消费 XLY",
        "XLP": "必需消费 XLP",
        "XLV": "医疗 XLV",
        "XBI": "生物科技 XBI",
        "XLU": "公用事业 XLU",
        "XLRE": "房地产 XLRE",
        "ITB": "住宅建筑 ITB",
        "GDX": "金矿 GDX",
    },
}

#: What "the market" is when it is taken out: the broadest index in each.
MARKET_PROXY = {"CN": ("000985.CSI", "中证全指"), "US": ("^GSPC", "标普500")}

#: Rolling span per frequency: (span, unit). Half a year either way, roughly.
FREQS = {"W": (26, "周"), "D": (60, "日")}

MAX_INDUSTRIES = 8

LABELS = {code: label for m in INDUSTRIES.values() for code, label in m.items()}
LABELS.update({code: label for code, label in MARKET_PROXY.values()})
CODE_MARKET = {code: m for m, idx in INDUSTRIES.items() for code in idx}
CODE_MARKET.update({code: m for m, (code, _) in MARKET_PROXY.items()})

History = Callable[[str], pd.Series]


def industry_history(code: str) -> pd.Series:
    """Ten years of one industry index (or market proxy). Raises rather than returns nothing."""
    if code not in CODE_MARKET:
        raise LookupError(f"没有这个行业指数：{code}")
    if CODE_MARKET[code] == "US":
        import markets

        close = markets.get("US").fetch_closes_since(code, pw._history_start().date())
        if close is None or close.dropna().empty:
            raise RuntimeError(f"{LABELS[code]} 行情读取失败")
        return pw._clean(close)

    import data_manager

    df = data_manager.get_index_data_live(
        code, start_date=pw._history_start().strftime("%Y%m%d"))
    if df is None or df.empty or "Close" not in df:
        raise RuntimeError(f"{LABELS[code]} 行情读取失败")
    return pw._clean(df["Close"])


def members(code: str) -> dict[str, float]:
    """
    The latest constituents of a CN industry index, {six-digit code: weight %}.

    A large constituent correlates with its own index almost by construction,
    which is a different fact from an outsider trading like the basket. Empty
    for US ETFs and whenever Tushare will not say.
    """
    if CODE_MARKET.get(code) != "CN":
        return {}
    import data_manager

    if data_manager.TUSHARE_API is None and not data_manager.init_tushare():
        return {}
    end = pd.Timestamp.today()
    w = data_manager.TUSHARE_API.index_weight(
        index_code=code, start_date=(end - pd.Timedelta(days=62)).strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"))
    if w is None or w.empty:
        return {}
    w = w[w["trade_date"] == w["trade_date"].max()]
    return {str(c).split(".")[0]: round(float(v), 2) for c, v in zip(w["con_code"], w["weight"])}


# ── arithmetic ───────────────────────────────────────────────────────────────
def partial(r_sx, r_sm, r_xm):
    """
    Correlation of s and x with m held fixed. Works on numbers or aligned Series.

    The same number as correlating what is left of each after regressing it on
    m over the same span. Undefined where either moves one-for-one with m.
    """
    den = np.sqrt((1 - r_sm ** 2) * (1 - r_xm ** 2))
    out = (r_sx - r_sm * r_xm) / den.where(den > 1e-6) if isinstance(den, pd.Series) \
        else ((r_sx - r_sm * r_xm) / den if den > 1e-6 else np.nan)
    return out.clip(-1, 1) if isinstance(out, pd.Series) else float(np.clip(out, -1, 1))


def _returns(px: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Returns at the chosen frequency, dated on the last session of each period.

    Weeks are built from the sessions that exist, so a week the exchange was
    shut for (春节, 国庆) is not a week at all, rather than a gap that blanks
    the next half-year of rolling windows.
    """
    if freq == "W":
        px = px.groupby(px.index.to_period("W-FRI")).tail(1)
    return px.pct_change(fill_method=None).iloc[1:]


def rolling_track(rets: pd.DataFrame, stock: str, market: str, inds: list[str],
                  span: int) -> tuple[dict[str, pd.Series], dict[str, pd.Series], pd.Series]:
    """Rolling raw and ex-market correlation of `stock` with each industry."""
    s, m = rets[stock], rets[market]
    r_sm = s.rolling(span, min_periods=span).corr(m)
    raw, ex = {}, {}
    for c in inds:
        r_sx = s.rolling(span, min_periods=span).corr(rets[c])
        r_xm = rets[c].rolling(span, min_periods=span).corr(m)
        raw[c] = r_sx
        ex[c] = partial(r_sx, r_sm, r_xm)
    return raw, ex, r_sm


def window_corr(rets: pd.DataFrame, stock: str, market: str, inds: list[str]) -> dict:
    """Raw and ex-market correlation over one window of daily returns."""
    r = rets.dropna(subset=[stock, market])
    if len(r) < pw.MIN_SHARED_DAYS:
        return {"days": int(len(r)), "market_raw": None,
                "rows": {c: {"raw": None, "ex": None} for c in inds}}
    r_sm = float(r[stock].corr(r[market]))
    rows = {}
    for c in inds:
        both = r[[stock, market, c]].dropna()
        if len(both) < pw.MIN_SHARED_DAYS:
            rows[c] = {"raw": None, "ex": None}
            continue
        r_sx = float(both[stock].corr(both[c]))
        r_xm = float(both[c].corr(both[market]))
        rsm = float(both[stock].corr(both[market]))
        rows[c] = {"raw": _r(r_sx), "ex": _r(partial(r_sx, rsm, r_xm))}
    return {"days": int(len(r)), "market_raw": _r(r_sm), "rows": rows}


def _r(v) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), 3)


def _series(v: pd.Series, calendar: pd.DatetimeIndex) -> list:
    """A series on its own dates, stepped onto the calendar: each session shows the last value known."""
    s = v.reindex(v.index.union(calendar)).ffill().reindex(calendar)
    return [None if not np.isfinite(x) else round(float(x), 2) for x in s.to_numpy(dtype=float)]


# ── the whole view ───────────────────────────────────────────────────────────
def track(symbol: str, industries: list[str], *, calendar: pd.DatetimeIndex,
          freq: str = "W", start: str | None = None, end: str | None = None,
          series: bool = True, history: History = pw.fetch_history,
          industry: History = industry_history,
          membership: Callable[[str], dict[str, float]] | None = None) -> dict:
    """
    One stock against up to MAX_INDUSTRIES industry indexes.

    `calendar` is the window index's sessions, so every series lines up with
    the brush above it. With `series` the ten-year rolling tracks come back;
    with `start` and `end` the correlations inside that window, from daily
    returns (a 60-session window is only twelve weeks).
    """
    import markets
    import portfolio_build as pb

    sym = markets.canonical(symbol)
    market = pb.market_of([sym])
    if market not in INDUSTRIES:
        raise LookupError("行业轨迹目前只支持 A 股和美股")
    inds = list(dict.fromkeys(industries))
    if not inds:
        raise LookupError("至少选一个行业")
    if len(inds) > MAX_INDUSTRIES:
        raise LookupError(f"最多选 {MAX_INDUSTRIES} 个行业")
    wrong = [c for c in inds if CODE_MARKET.get(c) != market or c not in INDUSTRIES[market]]
    if wrong:
        raise LookupError(f"这些行业指数和 {sym} 不在同一个市场：{'、'.join(wrong)}")
    if freq not in FREQS:
        raise LookupError(f"频率只能是 {'/'.join(FREQS)}")
    proxy, proxy_label = MARKET_PROXY[market]

    frames, missing, why = pw._load([sym], history)
    if not frames:
        raise RuntimeError("行情读取失败 — " + "；".join(why[:3]))
    closes, failed, why = pw._load([proxy, *inds], industry)
    if proxy not in closes:
        raise RuntimeError(f"{proxy_label} 行情读取失败 — " + "；".join(why[:2]))
    inds = [c for c in inds if c in closes]
    if not inds:
        raise RuntimeError("行业指数行情读取失败 — " + "；".join(why[:3]))

    stock = frames[sym]
    first = stock.index[0]
    # Everything on the window index's sessions. A stock's suspended day keeps
    # its last price (a flat day, as in the optimiser); before it listed it
    # has no price at all.
    px = pd.DataFrame({c: pw._on_calendar(v, calendar) for c, v in
                       {sym: stock, **{c: closes[c] for c in [proxy, *inds]}}.items()})
    px.loc[px.index < first, sym] = np.nan

    span, unit = FREQS[freq]
    out: dict = {
        "symbol": sym, "market": market, "freq": freq, "span": span, "unit": unit,
        "market_proxy": {"code": proxy, "label": proxy_label},
        "industries": [{"code": c, "label": LABELS[c]} for c in inds],
        "failed": [c for c in failed if c != proxy],
        "listed": str(first.date()),
    }

    # Weight % if a constituent, False if known not to be, None if unknown
    # (US ETFs, or Tushare would not say): "no" must not stand in for "can't tell".
    look = membership or (lambda c: {})
    code6 = sym.split(":")[-1]
    held: dict[str, float | bool | None] = {}
    for c in inds:
        try:
            got = look(c) if market == "CN" else {}
        except Exception:                                          # noqa: BLE001
            got = {}
        held[c] = got.get(code6, False) if got else None
    out["member"] = held

    if series:
        rets = _returns(px, freq)
        raw, ex, r_sm = rolling_track(rets, sym, proxy, inds, span)
        out["dates"] = [str(d.date()) for d in calendar]
        out["close"] = [None if not np.isfinite(v) else round(float(v), 3)
                        for v in px[sym].to_numpy(dtype=float)]
        out["raw"] = {c: _series(raw[c], calendar) for c in inds}
        out["ex"] = {c: _series(ex[c], calendar) for c in inds}
        out["market_raw"] = _series(r_sm, calendar)

    if start and end:
        span_px = px.loc[pd.Timestamp(start):pd.Timestamp(end)]
        w = window_corr(span_px.pct_change(fill_method=None).iloc[1:], sym, proxy, inds)
        out["window"] = {"from": start, "to": end, **w}

    return out


__all__ = ["FREQS", "INDUSTRIES", "MARKET_PROXY", "MAX_INDUSTRIES", "industry_history",
           "members", "partial", "rolling_track", "track", "window_corr"]
