"""
cn_index.py — where every A-share theme stands, read through its index.

The A-share counterpart of us_etf.py, and read with exactly its arithmetic:
distance from the 50- and 200-day lines, the cross, the 52-week range, the
stretch ranked against the index's own ten years, and — at the extremes —
what followed past stretches as large. Relative strength is against 沪深300.

The universe is the indices the big theme ETFs track (中证全指半导体 for
半导体ETF, 中证人工智能主题 for 人工智能ETF, 中证白酒 for 白酒ETF …), so a
reading here is a reading of the fund you would actually buy.

Tushare's index bars are end-of-day and price-only (dividends not added
back), published after the close; there is no intraday bar to show.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd

import us_etf

BENCH = "000300.SH"

UNIVERSE: list[tuple[str, str, str]] = [
    ("000300.SH", "沪深300", "大盘"), ("000016.SH", "上证50", "大盘"), ("000905.SH", "中证500", "大盘"),
    ("000852.SH", "中证1000", "大盘"), ("932000.CSI", "中证2000", "大盘"), ("399006.SZ", "创业板指", "大盘"),
    ("000688.SH", "科创50", "大盘"),
    ("H30184.CSI", "半导体", "科技"), ("000685.SH", "科创芯片", "科技"), ("930713.CSI", "人工智能", "科技"),
    ("931160.CSI", "通信设备", "科技"), ("931079.CSI", "5G通信", "科技"), ("930851.CSI", "云计算", "科技"),
    ("H30202.CSI", "软件", "科技"), ("930651.CSI", "计算机", "科技"), ("931494.CSI", "消费电子", "科技"),
    ("H30590.CSI", "机器人", "科技"), ("930901.CSI", "动漫游戏", "科技"), ("399971.SZ", "传媒", "科技"),
    ("399976.SZ", "新能源车", "新能源与资源"), ("931719.CSI", "电池", "新能源与资源"),
    ("931151.CSI", "光伏产业", "新能源与资源"), ("931897.CSI", "绿色电力", "新能源与资源"),
    ("930708.CSI", "有色金属", "新能源与资源"), ("930598.CSI", "稀土产业", "新能源与资源"),
    ("931238.CSI", "黄金股", "新能源与资源"), ("399998.SZ", "煤炭", "新能源与资源"),
    ("399989.SZ", "医疗", "医药与消费"), ("931152.CSI", "创新药", "医药与消费"), ("399441.SZ", "生物医药", "医药与消费"),
    ("399997.SZ", "白酒", "医药与消费"), ("000932.SH", "主要消费", "医药与消费"), ("931021.CSI", "家电龙头", "医药与消费"),
    ("930707.CSI", "畜牧养殖", "医药与消费"),
    ("399975.SZ", "证券公司", "金融与制造"), ("399986.SZ", "银行", "金融与制造"), ("931775.CSI", "房地产", "金融与制造"),
    ("000922.CSI", "红利", "金融与制造"), ("399967.SZ", "军工", "金融与制造"), ("930607.CSI", "汽车", "金融与制造"),
    ("930608.CSI", "基建", "金融与制造"),
]
GROUPS = ["大盘", "科技", "新能源与资源", "医药与消费", "金融与制造"]
NAMES = {t: n for t, n, _ in UNIVERSE}
CODE_RE = r"^(\d{6}|H\d{5})\.(SH|SZ|CSI)$"


def _one(api, code: str, start: str, end: str) -> tuple[str, pd.Series | None]:
    for attempt in range(2):
        try:
            df = api.index_daily(ts_code=code, start_date=start, end_date=end, fields="trade_date,close")
            if df is None or df.empty:
                return code, None
            s = pd.Series(pd.to_numeric(df["close"], errors="coerce").to_numpy(),
                          pd.to_datetime(df["trade_date"], format="%Y%m%d"))
            return code, s.sort_index()
        except Exception as exc:                                   # noqa: BLE001
            if attempt:
                print(f"[cn_index] {code}: {type(exc).__name__}: {exc}"[:160])
    return code, None


def fetch(years: int = 10) -> pd.DataFrame:
    """Ten years of daily closes for the universe — one Tushare call per index, four at a time."""
    import watchlist_board

    api = watchlist_board._api()
    end = date.today()
    start = (end - timedelta(days=int(years * 365.25))).strftime("%Y%m%d")
    with ThreadPoolExecutor(max_workers=4) as pool:
        got = dict(pool.map(lambda c: _one(api, c, start, end.strftime("%Y%m%d")), [t for t, _, _ in UNIVERSE]))
    closes = pd.DataFrame({c: s for c, s in got.items() if s is not None})
    if closes.empty or BENCH not in closes:
        raise RuntimeError("Tushare 没有返回 A 股指数行情 — 请稍后重试")
    return closes.sort_index()


def analyse(closes: pd.DataFrame) -> dict:
    out = us_etf.analyse(closes, universe=UNIVERSE, groups=GROUPS, bench=BENCH, bench_name="沪深300",
                         noun="行业与主题指数", session="closed")
    out["market"] = "CN"
    return out


def detail(closes: pd.DataFrame, code: str, years: int = 5) -> dict:
    return us_etf.detail(closes, code, years, names=NAMES)
