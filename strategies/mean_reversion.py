"""
strategies/mean_reversion.py — 反转候选: stocks retail has sold too far, too
fast, on sentiment rather than news.

Extracted verbatim from pages/sentiment_mean_reversion.py, so the Streamlit
page and the new app reach the same verdict. What is new is that the rules are
a pure function of a price series and three supplied facts, testable offline.

THE FIVE RULES, in the order they stop being about price and start being about
whether anyone is still selling:

  1. 收益率 Z ≤ −2.5     — today's fall is abnormal against its own 20 days.
  2. 连续下跌 ≥ 4 天      — a cascade, not a single bad session.
  3. 缩量                 — average volume across recent down days is BELOW the
                            prior down days. Price still falling while volume
                            fades is sellers running out of inventory, which is
                            the actual signal; falling on rising volume is not
                            exhaustion, it is distribution.
  4. RSI(14) < 25        — deeply oversold on the classic measure.
  5. 弱于板块             — the sector is not down with it, so this is an
                            isolated panic rather than a sector-wide repricing.
                            A stock in no tracked sector scores half credit
                            rather than failing: missing data is not evidence.

ST / *ST names are rejected outright — delisting risk is not a mean-reversion
setup, and a stock can fall a long way for reasons that never revert.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MIN_BARS = 30


@dataclass(frozen=True)
class Params:
    """The page's defaults."""
    z_max: float = -2.5
    down_days_min: int = 4
    rsi_max: float = 25.0
    #: How many points weaker than its sector the stock must be, over 5 days.
    sector_div_pp: float = 5.0


DEFAULTS = Params()


# ── components ───────────────────────────────────────────────────────────────
def zscore_of_latest_return(close: pd.Series, window: int = 20) -> float | None:
    """
    Z-score of the latest daily return against the PRIOR window of returns.

    The latest bar is excluded from the mean and deviation it is measured
    against — including it would let a big move inflate the yardstick it is
    being judged by and understate exactly the day that matters.
    """
    rets = close.pct_change().dropna()
    if len(rets) < window + 1:
        return None
    sample = rets.iloc[-(window + 1):-1]
    mu, sd = sample.mean(), sample.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return None
    return float((rets.iloc[-1] - mu) / sd)


def rsi(close: pd.Series, window: int = 14) -> float | None:
    """Classic RSI on the close. 100 when there were no down days at all."""
    if len(close) < window + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    if loss.iloc[-1] == 0 or pd.isna(loss.iloc[-1]):
        return 100.0
    return float(100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])))


def consecutive_down_days(close: pd.Series) -> int:
    """Trailing run of negative days ending on the latest bar."""
    n = 0
    for r in reversed(close.pct_change().dropna().tolist()):
        if r < 0:
            n += 1
        else:
            break
    return n


def volume_exhaustion(close: pd.Series, volume: pd.Series, lookback: int = 5) -> bool:
    """
    True when recent down days trade LESS than earlier down days.

    Compared across down days only, never across all days: a stock falling
    through a quiet week would otherwise look like exhaustion simply because
    the whole market was quiet.
    """
    rets = close.pct_change()
    df = pd.concat([rets.rename("ret"), volume.rename("vol")], axis=1).dropna()
    down = df[df["ret"] < 0]
    if len(down) < 2 * lookback:
        # Not enough down days for two windows; fall back to comparing the
        # latest against the few before it.
        if len(down) < 3:
            return False
        return bool(down["vol"].iloc[-1] < down["vol"].iloc[-min(6, len(down)):-1].mean())
    return bool(down["vol"].iloc[-lookback:].mean() < down["vol"].iloc[-2 * lookback:-lookback].mean())


def is_st(name: str | None) -> bool:
    """ST / *ST / S*ST — under delisting supervision."""
    return bool(name) and "ST" in str(name).upper()


def _r(v, nd):
    return None if v is None else round(v, nd)


# ── the verdict ──────────────────────────────────────────────────────────────
def evaluate(close: pd.Series | None, volume: pd.Series | None, *,
             name: str | None = None,
             limit_down_streak: int = 0,
             vs_sector_pp: float | None = None,
             params: Params = DEFAULTS) -> dict:
    """
    Run the five rules. Pure — the caller supplies the limit-down count and the
    5-day gap to the sector, both of which need the database.

    `vs_sector_pp` of None means the stock is in no tracked sector. That rule
    then returns None rather than False and counts as half a pass: absence of
    data is not evidence against, and a watchlist name outside every PPI index
    should not be punished for it.
    """
    if is_st(name):
        return {"verdict": "skip", "rank": 3, "why": "ST / *ST — 退市风险",
                "rules": {}, "passed": 0, "z": None, "rsi": None,
                "down_days": None, "vol_exhausted": None,
                "limit_down_streak": limit_down_streak, "vs_sector_pp": None,
                "ret_5d_pct": None}

    if close is None or len(close) < MIN_BARS:
        return {"verdict": "no_data", "rank": 2, "why": "价格历史不足",
                "rules": {}, "passed": 0, "z": None, "rsi": None,
                "down_days": None, "vol_exhausted": None,
                "limit_down_streak": limit_down_streak, "vs_sector_pp": None,
                "ret_5d_pct": None}

    z = zscore_of_latest_return(close, 20)
    r = rsi(close, 14)
    run = consecutive_down_days(close)
    vol_ex = volume_exhaustion(close, volume, 5) if volume is not None else False
    ret5 = (float(close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else None

    rules = {
        "z": z is not None and z <= params.z_max,
        "down": run >= params.down_days_min,
        "vol": bool(vol_ex),
        "rsi": r is not None and r < params.rsi_max,
        # None, not False — see the docstring.
        "sector": None if vs_sector_pp is None else vs_sector_pp <= -params.sector_div_pp,
    }

    passed = sum(1 for v in rules.values() if v is True)
    soft = passed + (0.5 if rules["sector"] is None else 0)

    if all(v is True for v in rules.values() if v is not None) and passed >= 4:
        verdict, rank = "strong", 0
    elif soft >= 3:
        verdict, rank = "watch", 1
    else:
        verdict, rank = "not_now", 2

    return {
        "verdict": verdict, "rank": rank, "why": "",
        "rules": rules, "passed": passed,
        "z": _r(z, 2), "rsi": _r(r, 1), "down_days": run,
        "vol_exhausted": bool(vol_ex),
        "limit_down_streak": int(limit_down_streak),
        "vs_sector_pp": _r(vs_sector_pp, 1),
        "ret_5d_pct": _r(ret5, 1),
    }


VERDICT_CN = {
    "strong": "🟢 强反转候选",
    "watch": "🟡 观察",
    "not_now": "⚪ 暂不符合",
    "skip": "⛔ 排除",
    "no_data": "⚠️ 无数据",
}

RULE_CN = {"z": "收益率Z", "down": "连续下跌", "vol": "缩量", "rsi": "RSI", "sector": "弱于板块"}
