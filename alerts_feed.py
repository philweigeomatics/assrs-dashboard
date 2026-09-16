"""
alerts_feed.py — today's watchlist alerts, as something you can filter.

The nightly GitHub Actions scan (scan_watchlists.py) already writes everything
this needs: one `daily_signals` row per stock that fired, and a `chip_scan` row
per stock per decay. Nothing new is computed here and the job is unchanged —
this module only reshapes the snapshot into a feed with facets.

Why reshaping is needed
-----------------------
The cache stores each stock's signals as ONE joined string, because the table
has a fixed ten-column schema:

    "▲ MACD Bottoming  ·  ▲ 箱体下沿 [12.30-15.60] 位置5% 触4/3 质0.82"

You cannot filter a list by "show me only RSI signals" when the signals are
punctuation inside a string. So the string is parsed back against the closed
vocabulary the scanner writes it from — watchlist_scan's own BULL_BOOL /
BULL_ENTRY / BEAR_BOOL / BEAR_EXIT dictionaries, imported rather than copied,
so a label renamed there cannot silently stop matching here.

The box signal carries its numbers in the bracketed tag, and those are pulled
out too: sitting on a support that has held four times is a different alert
from touching one that has held once, and the difference is in the tag.
"""

from __future__ import annotations

import re

import pandas as pd

#: Indicator family, used to group the filter UI. Nineteen checkboxes in a
#: flat list is not a filter; six groups of three is.
GROUP_MACD = "MACD"
GROUP_RSI = "RSI"
GROUP_ADX = "ADX / DI"
GROUP_SQUEEZE = "挤压"
GROUP_BOX = "箱体"
GROUP_REVERSAL = "反转"
GROUP_ACC = "吸筹"

#: English label as stored → (id, Chinese label, group). The English side must
#: match watchlist_scan exactly; _check_catalog() enforces that at import.
_LABELS: dict[str, tuple[str, str, str]] = {
    "Bullish Squeeze Breakout 🚀": ("squeeze_bull", "挤压突破", GROUP_SQUEEZE),
    "Bearish Squeeze Drop 🩸":     ("squeeze_bear", "挤压下破", GROUP_SQUEEZE),
    "Phase 1: Accumulation":       ("accumulation", "吸筹阶段", GROUP_ACC),
    "MACD Bottoming":              ("macd_bottom", "MACD 筑底", GROUP_MACD),
    "MACD ClassicCrossover":       ("macd_cross_up", "MACD 金叉", GROUP_MACD),
    "MACD Bullish Crossover":      ("macd_cross_up", "MACD 金叉", GROUP_MACD),
    "MACD Peaking":                ("macd_peak", "MACD 见顶", GROUP_MACD),
    "MACD Bearish Crossover":      ("macd_cross_down", "MACD 死叉", GROUP_MACD),
    "MACD Exit Signal":            ("macd_exit", "MACD 离场", GROUP_MACD),
    "RSI Bottoming":               ("rsi_bottom", "RSI 筑底", GROUP_RSI),
    "RSI Peaking":                 ("rsi_peak", "RSI 见顶", GROUP_RSI),
    "Downtrend Reversal 🔄":       ("downtrend_rev", "下跌反转", GROUP_REVERSAL),
    "Uptrend Reversal 🔄":         ("uptrend_rev", "上涨反转", GROUP_REVERSAL),
    "Strength Returning (ADX)":    ("adx_returning", "ADX 转强", GROUP_ADX),
    "Trend Accelerating (ADX)":    ("adx_accel", "ADX 加速", GROUP_ADX),
    "Trend Topping (ADX)":         ("adx_topping", "ADX 见顶", GROUP_ADX),
    "Trend Collapsing (ADX)":      ("adx_collapse", "ADX 转弱", GROUP_ADX),
    "DI Screaming Buy 🚀":         ("di_buy", "DI 强买", GROUP_ADX),
    "DI Screaming Sell 🛑":        ("di_sell", "DI 强卖", GROUP_ADX),
}

#: Box signals are written as a prefix plus a bracketed tag of numbers.
_BOX = {
    "箱体下沿": ("box_support", "箱体下沿", "bull"),
    "箱体上沿": ("box_resistance", "箱体上沿", "bear"),
    "箱体突破": ("box_breakout", "箱体突破", "bull"),
    "箱体跌破": ("box_breakdown", "箱体跌破", "bear"),
}

_BOX_RE = re.compile(
    r"^(箱体[下上]沿|箱体突破|箱体跌破)\s*"
    r"\[([\d.]+)-([\d.]+)\]\s*位置(-?[\d.]+)%\s*触(\d+)/(\d+)\s*质([\d.]+)")

#: Facet name for watchlist stocks that are in no sector index.
UNCLASSIFIED = "未归类"

SEPARATOR = "  ·  "
_DIR_MARK = {"▲": "bull", "▼": "bear"}


def _check_catalog() -> None:
    """
    Every label the scanner can write must be one this module can parse.

    Imported from watchlist_scan rather than duplicated, so renaming a signal
    there fails loudly here instead of quietly dropping it out of the filters.
    """
    import watchlist_scan as ws
    known = set(_LABELS)
    produced = set(ws.BULL_BOOL.values()) | set(ws.BEAR_BOOL.values()) \
        | set(ws.BULL_ENTRY.values()) | set(ws.BEAR_EXIT.values())
    missing = produced - known
    if missing:
        raise RuntimeError(
            f"alerts_feed._LABELS is missing signal label(s) the scanner writes: "
            f"{sorted(missing)}")


def parse_signals(text: str) -> list[dict]:
    """
    The joined signal string, back into structured items.

    An unrecognised fragment is kept as an "other" item rather than dropped:
    a signal nobody can filter by is still a signal you need to see.
    """
    out: list[dict] = []
    for part in (text or "").split(SEPARATOR):
        part = part.strip()
        if not part:
            continue
        direction = None
        if part[:1] in _DIR_MARK:
            direction = _DIR_MARK[part[0]]
            part = part[1:].strip()

        box = _BOX_RE.match(part)
        if box:
            kind, bot, top, pos, t_top, t_bot, quality = box.groups()
            sid, cn, box_dir = _BOX[kind]
            out.append({
                "id": sid, "cn": cn, "en": part, "group": GROUP_BOX,
                "dir": direction or box_dir,
                "detail": {
                    "bot": float(bot), "top": float(top),
                    "position_pct": float(pos),
                    "touches_top": int(t_top), "touches_bot": int(t_bot),
                    "quality": float(quality),
                    "height_pct": round((float(top) / float(bot) - 1) * 100, 1),
                },
            })
            continue

        hit = _LABELS.get(part)
        if hit:
            sid, cn, group = hit
            out.append({"id": sid, "cn": cn, "en": part, "group": group,
                        "dir": direction or "bull", "detail": None})
        else:
            out.append({"id": "other", "cn": part, "en": part, "group": "其他",
                        "dir": direction or "bull", "detail": None})
    return out


def _num(v, nd=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else round(v, nd)


def _chip_map(chips: pd.DataFrame | None) -> dict[str, dict]:
    """Chip structure per ticker, from the nightly scan's neutral decay."""
    if chips is None or chips.empty:
        return {}
    keep = ["setup_score", "setup_label", "winner_rate", "concentration",
            "pct_from_peak", "n_peaks", "peak_price", "converged"]
    out = {}
    for _, r in chips.iterrows():
        d = {k: (r[k] if k in chips.columns else None) for k in keep}
        out[str(r["ticker"])] = {
            "setup_score": _num(d["setup_score"], 3),
            "setup_label": (None if pd.isna(d["setup_label"]) else str(d["setup_label"])),
            "winner_rate": _num(d["winner_rate"], 4),
            "concentration": _num(d["concentration"], 4),
            "pct_from_peak": _num(d["pct_from_peak"], 1),
            "n_peaks": int(d["n_peaks"]) if pd.notna(d["n_peaks"]) else None,
            "peak_price": _num(d["peak_price"]),
            "converged": bool(d["converged"]) if pd.notna(d["converged"]) else None,
        }
    return out


#: 筹码 shapes worth filtering on. Derived from the nightly setup_label, so the
#: page and the scanner agree on what "a good distribution" means.
def _chip_shape(chip: dict | None) -> str | None:
    return chip.get("setup_label") if chip else None


#: A snapshot older than this many calendar days has missed a session that
#: was not just a weekend, and the page says so instead of presenting it as
#: today's.
STALE_AFTER_DAYS = 4


def build(rows: pd.DataFrame, chips: pd.DataFrame | None,
          sector_members: dict[str, set[str]], scan_date: str,
          age_days: int | None = None) -> dict:
    """
    The whole feed: one entry per stock that fired, plus the facet counts the
    filter UI needs.

    Facets are counted over ALL stocks, not the filtered subset, so a filter
    never hides the option that would bring results back.
    """
    _check_catalog()

    by_ticker: dict[str, list[str]] = {}
    for sector, members in (sector_members or {}).items():
        for t in members:
            by_ticker.setdefault(str(t), []).append(sector)

    chip_map = _chip_map(chips)

    stocks = []
    for _, r in rows.iterrows():
        ticker = str(r["Ticker"])
        signals = parse_signals(r.get("Signals", ""))
        chip = chip_map.get(ticker)
        stocks.append({
            "t": ticker,
            "n": str(r.get("Name") or ticker),
            "bias": str(r.get("Type") or ""),
            "price": _num(r.get("Price")),
            "rsi": _num(r.get("RSI"), 1),
            "adx": _num(r.get("ADX"), 1),
            "macd": _num(r.get("MACD"), 3),
            "volume": _num(r.get("Volume"), 0),
            "signal_count": int(r.get("Signal_Count") or len(signals)),
            "signals": signals,
            "sectors": sorted(by_ticker.get(ticker, [])),
            "chips": chip,
            "chip_shape": _chip_shape(chip),
        })

    # Bullish first, then most signals — the same order the scanner ranks in.
    stocks.sort(key=lambda s: ({"🚀 Bullish": 0, "⚖️ Mixed": 1, "⚠️ Bearish": 2}
                               .get(s["bias"], 3), -s["signal_count"]))

    return {
        "scan_date": scan_date,
        "age_days": age_days,
        # A weekend is not staleness; a missed nightly run is. The page shows
        # the date either way, and warns only when the data has actually
        # fallen behind.
        "stale": bool(age_days is not None and age_days > STALE_AFTER_DAYS),
        "stocks": stocks,
        "facets": _facets(stocks),
        "groups": [GROUP_MACD, GROUP_RSI, GROUP_ADX, GROUP_SQUEEZE,
                   GROUP_BOX, GROUP_REVERSAL, GROUP_ACC, "其他"],
    }


def _facets(stocks: list[dict]) -> dict:
    sig: dict[str, dict] = {}
    sector: dict[str, int] = {}
    bias: dict[str, int] = {}
    shape: dict[str, int] = {}

    for s in stocks:
        bias[s["bias"]] = bias.get(s["bias"], 0) + 1
        # A watchlist stock that belongs to no PPI sector would otherwise be
        # unreachable once any sector filter is on, with no hint it exists.
        for name in (s["sectors"] or [UNCLASSIFIED]):
            sector[name] = sector.get(name, 0) + 1
        if s["chip_shape"]:
            shape[s["chip_shape"]] = shape.get(s["chip_shape"], 0) + 1
        # A stock that fires the same signal twice still counts once, or the
        # facet number stops matching the number of rows the filter returns.
        for item in {x["id"]: x for x in s["signals"]}.values():
            e = sig.setdefault(item["id"], {"id": item["id"], "cn": item["cn"],
                                            "group": item["group"],
                                            "dir": item["dir"], "count": 0})
            e["count"] += 1

    return {
        "signals": sorted(sig.values(), key=lambda x: (-x["count"], x["id"])),
        "sectors": sorted(({"name": k, "count": v} for k, v in sector.items()),
                          key=lambda x: (-x["count"], x["name"])),
        "bias": sorted(({"id": k, "count": v} for k, v in bias.items()),
                       key=lambda x: -x["count"]),
        "shapes": sorted(({"name": k, "count": v} for k, v in shape.items()),
                         key=lambda x: (-x["count"], x["name"])),
    }
