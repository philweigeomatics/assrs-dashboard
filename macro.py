"""
宏观与大宗 — China macro series and domestic commodity term structure.

Ported from pages/macro_commodities.py. Two halves that answer different
questions and share nothing but a data source:

  · Macro — inflation, growth, liquidity and rates, each a level plus the
    change from the prior period, plus enough history to draw the shape.
  · Commodities — the forward curve of a futures product on one trade date.
    A futures price alone says nothing; the SLOPE from near to far contracts
    is the signal. Upward means carry is priced in (contango); downward means
    spot is tight (backwardation).

Tushare is inconsistent about column case: `cn_cpi` returns `month`,
`cn_pmi` returns `MONTH` and `PMI010000`. The Streamlit page asked for
lowercase everywhere, so its PMI card has been blank the whole time — the
series is there, and manufacturing PMI last read 49.8. Column lookup here is
case-insensitive for that reason.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

CHINA_TZ = ZoneInfo("Asia/Shanghai")

#: How much history each card carries for its sparkline.
MONTHS_BACK = 36
DAYS_BACK = 120
#: Contracts with less than this share of the busiest month's open interest
#: are dead months whose settlements are administrative, not traded.
LIQUID_FRACTION = 0.10
#: A flat curve within this fraction of the front price is not a signal.
FLAT_BAND = 0.001


def _api():
    import data_manager
    data_manager.init_tushare()
    api = data_manager.TUSHARE_API
    if api is None:
        raise RuntimeError("Tushare 未初始化")
    return api


def _fetch(endpoint: str, **kwargs) -> pd.DataFrame | None:
    """One Tushare endpoint, or None with the reason printed.

    A dead endpoint must degrade to one missing card, never to a dead page:
    macro series come from seven separate calls and any of them can be out
    of tier or out of credits on a given day.
    """
    try:
        return getattr(_api(), endpoint)(**kwargs)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[macro] {endpoint}: {type(exc).__name__}: {exc}")
        return None


def _col(df: pd.DataFrame, name: str) -> str | None:
    """The real column behind a name, whatever case Tushare chose today."""
    if df is None or df.empty:
        return None
    if name in df.columns:
        return name
    lower = {str(c).lower(): c for c in df.columns}
    return lower.get(name.lower())


def _pretty(period: str) -> str:
    p = str(period)
    if len(p) == 6 and p.isdigit():
        return f"{p[:4]}-{p[4:6]}"
    if len(p) == 8 and p.isdigit():
        return f"{p[:4]}-{p[4:6]}-{p[6:8]}"
    return p


def series(df: pd.DataFrame | None, period_col: str, value_col: str,
           points: int = 24) -> dict | None:
    """Latest reading, the change from the prior period, and a short history."""
    pc = _col(df, period_col)
    vc = _col(df, value_col)
    if pc is None or vc is None:
        return None

    d = df[[pc, vc]].copy()
    d[vc] = pd.to_numeric(d[vc], errors="coerce")
    d = d.dropna(subset=[vc]).sort_values(pc)
    if d.empty:
        return None

    tail = d.tail(points)
    latest = float(d[vc].iloc[-1])
    prev = float(d[vc].iloc[-2]) if len(d) >= 2 else None
    return {
        "value": round(latest, 3),
        "prev": None if prev is None else round(prev, 3),
        "change": None if prev is None else round(latest - prev, 3),
        "period": _pretty(d[pc].iloc[-1]),
        "history": [round(float(v), 3) for v in tail[vc]],
        "periods": [_pretty(p) for p in tail[pc]],
    }


# ── macro ────────────────────────────────────────────────────────────────────
#: label, group, unit, endpoint, period column, value column, and the level
#: that separates expansion from contraction where one exists.
MACRO = [
    ("CPI 同比", "通胀", "%", "cn_cpi", "month", "nt_yoy", 0.0,
     "居民消费价格指数同比。通缩区间说明需求偏弱。"),
    ("PPI 同比", "通胀", "%", "cn_ppi", "month", "ppi_yoy", 0.0,
     "工业品出厂价格同比 —— 上游利润和工业需求最直接的温度计。"),
    ("制造业 PMI", "增长", "", "cn_pmi", "MONTH", "PMI010000", 50.0,
     "50 是荣枯线：以上扩张，以下收缩。"),
    ("非制造业 PMI", "增长", "", "cn_pmi", "MONTH", "PMI020100", 50.0,
     "服务业与建筑业的荣枯线，同样以 50 为界。"),
    ("GDP 同比", "增长", "%", "cn_gdp", "quarter", "gdp_yoy", None,
     "季度实际 GDP 同比。"),
    ("M2 同比", "流动性", "%", "cn_m", "month", "m2_yoy", None,
     "广义货币供应同比 —— 钱有多少。"),
    ("M1 同比", "流动性", "%", "cn_m", "month", "m1_yoy", None,
     "狭义货币供应同比 —— 钱有多活跃。M1 走高通常先于股市。"),
    ("SHIBOR 隔夜", "国内利率", "%", "shibor", "date", "on", None,
     "银行间隔夜拆借利率，资金面最敏感的一根。"),
    ("SHIBOR 3 个月", "国内利率", "%", "shibor", "date", "3m", None,
     "三个月期拆借利率，反映中期资金价格。"),
    ("美债 2 年", "美国利率", "%", "us_tycr", "date", "y2", None,
     "对美联储政策路径最敏感的一段。"),
    ("美债 10 年", "美国利率", "%", "us_tycr", "date", "y10", None,
     "全球资产定价的锚。"),
]


def macro() -> dict:
    """Every macro card, grouped, with whatever failed named rather than hidden."""
    now = datetime.now(CHINA_TZ)
    m0 = (now - timedelta(days=365 * 3)).strftime("%Y%m")
    m1 = now.strftime("%Y%m")
    d0 = (now - timedelta(days=DAYS_BACK)).strftime("%Y%m%d")
    d1 = now.strftime("%Y%m%d")

    frames = {
        "cn_cpi": _fetch("cn_cpi", start_m=m0, end_m=m1),
        "cn_ppi": _fetch("cn_ppi", start_m=m0, end_m=m1),
        "cn_pmi": _fetch("cn_pmi", start_m=m0, end_m=m1),
        "cn_m": _fetch("cn_m", start_m=m0, end_m=m1),
        "cn_gdp": _fetch("cn_gdp", start_q=f"{now.year - 3}Q1",
                         end_q=f"{now.year}Q4"),
        "shibor": _fetch("shibor", start_date=d0, end_date=d1),
        "us_tycr": _fetch("us_tycr", start_date=d0, end_date=d1),
    }

    cards, missing = [], []
    for label, group, unit, endpoint, pc, vc, threshold, note in MACRO:
        got = series(frames.get(endpoint), pc, vc)
        if got is None:
            missing.append(label)
            continue
        cards.append({"label": label, "group": group, "unit": unit,
                      "threshold": threshold, "note": note, **got})

    groups = []
    for _, g, *_ in MACRO:
        if g not in groups:
            groups.append(g)

    return {"cards": cards, "groups": [g for g in groups
                                       if any(c["group"] == g for c in cards)],
            "missing": missing,
            "as_of": now.strftime("%Y-%m-%d %H:%M")}


# ── commodities ──────────────────────────────────────────────────────────────
FUTURES = [
    {"code": "CU", "exchange": "SHFE", "label": "沪铜", "en": "Copper",
     "unit": "¥/吨", "group": "有色"},
    {"code": "AL", "exchange": "SHFE", "label": "沪铝", "en": "Aluminium",
     "unit": "¥/吨", "group": "有色"},
    {"code": "ZN", "exchange": "SHFE", "label": "沪锌", "en": "Zinc",
     "unit": "¥/吨", "group": "有色"},
    {"code": "AU", "exchange": "SHFE", "label": "沪金", "en": "Gold",
     "unit": "¥/克", "group": "贵金属"},
    {"code": "AG", "exchange": "SHFE", "label": "沪银", "en": "Silver",
     "unit": "¥/千克", "group": "贵金属"},
    {"code": "RB", "exchange": "SHFE", "label": "螺纹钢", "en": "Rebar",
     "unit": "¥/吨", "group": "黑色"},
    {"code": "I", "exchange": "DCE", "label": "铁矿石", "en": "Iron ore",
     "unit": "¥/吨", "group": "黑色"},
    {"code": "JM", "exchange": "DCE", "label": "焦煤", "en": "Coking coal",
     "unit": "¥/吨", "group": "黑色"},
    {"code": "SC", "exchange": "INE", "label": "原油", "en": "Crude",
     "unit": "¥/桶", "group": "能源"},
    {"code": "M", "exchange": "DCE", "label": "豆粕", "en": "Soybean meal",
     "unit": "¥/吨", "group": "农产品"},
    {"code": "P", "exchange": "DCE", "label": "棕榈油", "en": "Palm oil",
     "unit": "¥/吨", "group": "农产品"},
    {"code": "TA", "exchange": "CZCE", "label": "PTA", "en": "PTA",
     "unit": "¥/吨", "group": "化工"},
]
BY_CODE = {f["code"]: f for f in FUTURES}


def latest_trade_date() -> str | None:
    """Probed off a liquid contract rather than assumed from the calendar."""
    now = datetime.now(CHINA_TZ)
    df = _fetch("fut_daily", ts_code="CU.SHF",
                start_date=(now - timedelta(days=DAYS_BACK)).strftime("%Y%m%d"),
                end_date=now.strftime("%Y%m%d"))
    if df is None or df.empty or "trade_date" not in df.columns:
        return None
    return str(df["trade_date"].max())


class Frames:
    """
    The two big reads, fetched once and shared across every product.

    `fut_daily(trade_date)` is one call covering all 1,075 contracts, and
    `fut_basic` is one per exchange. Asking per product instead meant
    thirteen identical whole-day reads and took 25 seconds.
    """

    def __init__(self, trade_date: str):
        self.trade_date = trade_date
        self._day: pd.DataFrame | None = None
        self._loaded = False
        self._basic: dict[str, pd.DataFrame | None] = {}

    @property
    def day(self) -> pd.DataFrame | None:
        if not self._loaded:
            self._day = _fetch("fut_daily", trade_date=self.trade_date)
            self._loaded = True
        return self._day

    def basic(self, exchange: str) -> pd.DataFrame | None:
        if exchange not in self._basic:
            self._basic[exchange] = _fetch(
                "fut_basic", exchange=exchange, fut_type="1",
                fields="ts_code,symbol,fut_code,name,list_date,delist_date")
        return self._basic[exchange]


def forward_curve(code: str, exchange: str, trade_date: str,
                  frames: Frames | None = None) -> pd.DataFrame | None:
    """
    Every still-listed contract of one product, near to far, on one date.

    This is the object the whole tab is about: a single price is not a market
    view, and the shape across delivery months is.
    """
    frames = frames or Frames(trade_date)
    basic = frames.basic(exchange)
    day = frames.day
    if basic is None or basic.empty or day is None or day.empty:
        return None

    b = basic[basic["fut_code"].astype(str).str.upper() == code.upper()].copy()
    b = b[b["delist_date"].astype(str) >= str(trade_date)]
    if b.empty:
        return None

    keep = [c for c in ("ts_code", "settle", "close", "oi", "vol")
            if c in day.columns]
    m = b.merge(day[keep], on="ts_code", how="inner")
    if m.empty:
        return None

    m["price"] = pd.to_numeric(m.get("settle"), errors="coerce")
    if "close" in m.columns:
        m["price"] = m["price"].fillna(pd.to_numeric(m["close"], errors="coerce"))
    m = m.dropna(subset=["price"])
    m = m[m["price"] > 0]
    if m.empty:
        return None

    m = m.sort_values("delist_date").reset_index(drop=True)
    m["maturity"] = m["delist_date"].astype(str)
    cols = ["symbol", "maturity", "price"] + [c for c in ("oi", "vol")
                                              if c in m.columns]
    return m[cols]


def liquid(curve: pd.DataFrame, frac: float = LIQUID_FRACTION) -> pd.DataFrame:
    """
    The contracts that actually trade.

    Chinese commodity liquidity clusters in the 1/5/9 delivery months. The
    off-months barely change hands and their settlements are set
    administratively, so a full curve shows kinks that no one could trade.
    Filtering on open interest finds the real nodes without hard-coding a
    month rule that would be wrong for crude and several agriculturals.
    """
    if curve is None or curve.empty or "oi" not in curve.columns:
        return curve
    oi = pd.to_numeric(curve["oi"], errors="coerce").fillna(0)
    if oi.max() <= 0:
        return curve
    keep = curve[oi >= oi.max() * frac]
    return keep if len(keep) >= 2 else curve


def term_structure(curve: pd.DataFrame | None, unit: str = "") -> dict | None:
    """
    What the front two contracts say about the state of the market.

    Roll yield is the number that matters to anyone holding the front
    contract: it has to be rolled before delivery, and the curve decides
    whether that costs or pays.
    """
    if curve is None or len(curve) < 2:
        return None

    front, nxt = curve.iloc[0], curve.iloc[1]
    fp, np_ = float(front["price"]), float(nxt["price"])
    spread = fp - np_

    try:
        d0 = datetime.strptime(str(front["maturity"]), "%Y%m%d")
        d1 = datetime.strptime(str(nxt["maturity"]), "%Y%m%d")
        days = max((d1 - d0).days, 1)
    except ValueError:
        days = 30

    roll = (fp / np_ - 1.0) * (365.0 / days) if np_ else 0.0

    if abs(spread) <= fp * FLAT_BAND:
        state, tone = "flat", "flat"
        note = "近月与远月基本持平，曲线没有方向。"
    elif spread > 0:
        state, tone = "backwardation", "up"
        note = ("近月贵于远月 —— 现货紧张。持有近月往后滚是赚的，"
                "通常出现在供给受限或库存偏低的时候。")
    else:
        state, tone = "contango", "down"
        note = ("远月贵于近月 —— 仓储和资金成本被计入价格。"
                "持有近月往后滚要贴钱，通常意味着供给宽松。")

    return {
        "state": state, "tone": tone, "note": note,
        "front_symbol": str(front["symbol"]), "front_price": round(fp, 2),
        "next_symbol": str(nxt["symbol"]), "next_price": round(np_, 2),
        "spread": round(spread, 2),
        "roll_ann_pct": round(roll * 100, 2),
        "days_between": days,
        "unit": unit,
    }


def _rows(curve: pd.DataFrame) -> list[dict]:
    front = float(curve["price"].iloc[0])
    out = []
    for _, r in curve.iterrows():
        price = float(r["price"])
        spread = price - front
        out.append({
            "symbol": str(r["symbol"]),
            "maturity": _pretty(r["maturity"]),
            "price": round(price, 2),
            "spread": round(spread, 2),
            "spread_pct": round(spread / front * 100, 2) if front else None,
            "oi": int(pd.to_numeric(r.get("oi"), errors="coerce") or 0),
            "vol": int(pd.to_numeric(r.get("vol"), errors="coerce") or 0),
        })
    return out


def commodities(code: str | None = None, *, liquid_only: bool = True) -> dict:
    """
    The headline board, plus the full curve for one product.

    The board is four front-month prices with each market's state attached,
    because "copper is 78,000" is not information and "copper is 78,000 and
    backwardated" is.
    """
    trade_date = latest_trade_date()
    if not trade_date:
        raise RuntimeError("读不到最近的期货交易日")

    frames = Frames(trade_date)
    board = []
    for spec in FUTURES:
        curve = forward_curve(spec["code"], spec["exchange"], trade_date, frames)
        if curve is None or curve.empty:
            board.append({**_spec(spec), "price": None, "state": None})
            continue
        shown = liquid(curve) if liquid_only else curve
        ts = term_structure(shown, spec["unit"])
        board.append({
            **_spec(spec),
            "symbol": str(shown["symbol"].iloc[0]),
            "price": round(float(shown["price"].iloc[0]), 2),
            "state": None if ts is None else ts["state"],
            "tone": None if ts is None else ts["tone"],
            "roll_ann_pct": None if ts is None else ts["roll_ann_pct"],
            "spread": None if ts is None else ts["spread"],
        })

    picked = code if code in BY_CODE else FUTURES[0]["code"]
    spec = BY_CODE[picked]
    full = forward_curve(picked, spec["exchange"], trade_date, frames)
    detail = None
    if full is not None and not full.empty:
        shown = liquid(full) if liquid_only else full
        detail = {
            **_spec(spec),
            "term": term_structure(shown, spec["unit"]),
            "curve": _rows(shown),
            "listed": len(full),
            "shown": len(shown),
            "liquid_only": liquid_only,
        }

    return {"trade_date": _pretty(trade_date), "board": board,
            "detail": detail, "products": [_spec(f) for f in FUTURES]}


def _spec(f: dict) -> dict:
    return {"code": f["code"], "label": f["label"], "en": f["en"],
            "unit": f["unit"], "group": f["group"]}
