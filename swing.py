"""
swing.py — if the market is near a bottom, do the bigger swingers win back more?

The intuition is right as far as it goes: a stock that moves 2× the market
recovers 2× as fast — and fell 2× as far on the way down. Whether it is worth
holding depends on whether that is SYMMETRIC. So three things are measured,
each answering a narrower version of the question:

1. Asymmetric beta. Beta is fitted twice: once on days the index rose, once
   on days it fell. A stock with up-β 1.6 and down-β 1.1 amplifies the
   market's good days more than its bad ones; one with 1.1 / 1.6 does the
   reverse, and its "bigger swings" are mostly the wrong kind.

2. Real rebounds. Every time the index fell at least `threshold` from a peak,
   each stock is measured on the way down (peak → trough) and on the way back
   (trough → 20 and 60 sessions later). That is the question asked directly,
   on what actually happened rather than a model of it.

   Buying the exact trough is only possible in hindsight, so the rebound is
   also measured from a CONFIRMED entry: the first close 3% above the low.
   The gap between the two is what waiting for confirmation costs.

3. A scenario. "If the index moves X% from here" → up-β or down-β × X, with
   the stock's own (non-market) volatility over the horizon as the range
   around it. Linear and therefore approximate; it is there to make the
   asymmetry concrete, not to forecast.

With three to six years of history there are only a handful of drawdowns.
Every summary says how many it rests on.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
THRESHOLDS = (5, 8, 10, 15)          # % drawdown that counts as an episode
DEFAULT_THRESHOLD = 8
HORIZONS = (20, 60)                  # sessions after the trough
CONFIRM = 0.03                       # "the bounce is real": 3% off the low
HISTORY_YEARS = 6


def _f(v, nd=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(v) else round(v, nd)


# ── up-day and down-day multiples ───────────────────────────────────────────
def _slope(y: np.ndarray, x: np.ndarray) -> float | None:
    if len(x) < 10 or np.var(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def asym_beta(r: pd.Series, m: pd.Series) -> dict:
    """
    β overall, and how many times the index's move the stock made on the
    index's up days and on its down days.

    The up/down multiples are capture ratios — the stock's average move on
    index-up days over the index's average move on those days — not slopes
    fitted to each half. A slope fitted to up days alone sees only a narrow
    slice of the index's range and swings wildly (CRM came out at −0.68 one
    year, ±0.45 of noise); the ratio is stable, and it is the same number the
    ranking table shows as 上/下行捕获, so the page does not disagree with
    itself.
    """
    x, y = m.to_numpy(), r.to_numpy()
    up, dn = x > 0, x < 0
    beta = _slope(y, x)
    resid = y - (beta or 0) * x

    def mult(mask):
        return float(y[mask].mean() / x[mask].mean()) if mask.sum() >= 10 else None

    return {
        "beta": _f(beta),
        "beta_up": _f(mult(up)),
        "beta_down": _f(mult(dn)),
        "resid_vol_daily_pct": _f(float(np.std(resid, ddof=1)) * 100, 3),
        "n_up": int(up.sum()), "n_down": int(dn.sum()),
    }


def swing_stats(prices: pd.DataFrame, m: pd.Series | None) -> dict:
    """Per stock: how big the swings are, and which way they are lopsided."""
    r = prices.pct_change().dropna()
    mr = m.pct_change().reindex(r.index) if m is not None else None
    out = {}
    for s in prices.columns:
        x = r[s]
        row = {
            "vol_annual_pct": _f(float(x.std(ddof=1)) * math.sqrt(TRADING_DAYS) * 100, 1),
            "avg_abs_daily_pct": _f(float(x.abs().mean()) * 100, 2),
            "up_day_avg_pct": _f(float(x[x > 0].mean()) * 100, 2) if (x > 0).any() else None,
            "down_day_avg_pct": _f(float(x[x < 0].mean()) * 100, 2) if (x < 0).any() else None,
        }
        if mr is not None:
            keep = mr.notna()
            ab = asym_beta(x[keep], mr[keep])
            row.update(ab)
            row["asymmetry"] = (_f(ab["beta_up"] - ab["beta_down"])
                                if ab["beta_up"] is not None and ab["beta_down"] is not None else None)
        out[s] = row
    if mr is not None:
        out["__market__"] = {
            "vol_annual_pct": _f(float(mr.std(ddof=1)) * math.sqrt(TRADING_DAYS) * 100, 1),
            "avg_abs_daily_pct": _f(float(mr.abs().mean()) * 100, 2),
        }
    return out


# ── drawdown episodes ────────────────────────────────────────────────────────
def episodes(m: pd.Series, threshold_pct: float) -> list[dict]:
    """
    Every fall of at least `threshold_pct` from a swing high to a swing low.

    Swing pivots, not "until it regains the peak": 沪深300 has not seen its
    February 2021 high since, so the recover-to-peak definition makes six
    years of A-shares one episode. Here a high is fixed once the index has
    fallen `threshold` from it, and a low once it has bounced `threshold` off
    it — so each correction inside a long bear market counts on its own.

    The last episode may still be open: the index fell far enough to count
    but has not yet bounced far enough to fix the low. Its "trough" is the
    lowest close so far and is marked provisional.
    """
    thr = threshold_pct / 100
    v = m.to_numpy(dtype=float)
    idx = m.index
    n = len(v)
    out = []
    looking = "peak"
    ext = 0
    peak_i = None
    for i in range(1, n):
        if looking == "peak":
            if v[i] > v[ext]:
                ext = i
            elif v[i] <= v[ext] * (1 - thr):
                peak_i, looking, ext = ext, "trough", i
        else:
            if v[i] < v[ext]:
                ext = i
            elif v[i] >= v[ext] * (1 + thr):
                out.append({"peak_i": peak_i, "trough_i": ext, "turned_i": i, "open": False})
                looking, ext = "peak", i
    if looking == "trough" and peak_i is not None:
        out.append({"peak_i": peak_i, "trough_i": ext, "turned_i": None, "open": True})
    for e in out:
        e.update({
            "peak": str(idx[e["peak_i"]].date()), "trough": str(idx[e["trough_i"]].date()),
            "turned": str(idx[e["turned_i"]].date()) if e["turned_i"] is not None else None,
            "fall_pct": _f((v[e["trough_i"]] / v[e["peak_i"]] - 1) * 100, 1),
        })
    return out


def _confirm_i(v: np.ndarray, trough_i: int) -> int | None:
    """First close at least CONFIRM above the low — an entry you could have known."""
    hit = np.flatnonzero(v[trough_i:] >= v[trough_i] * (1 + CONFIRM))
    return trough_i + int(hit[0]) if len(hit) else None


def _move(v: np.ndarray, a: int, b: int) -> float | None:
    if a is None or b >= len(v) or not (np.isfinite(v[a]) and np.isfinite(v[b])):
        return None
    return (v[b] / v[a] - 1) * 100


def episode_moves(prices: pd.DataFrame, m: pd.Series, eps: list[dict]) -> list[dict]:
    """Each stock and the index through each episode: down, then back up."""
    mv = m.to_numpy(dtype=float)
    out = []
    for e in eps:
        p, t = e["peak_i"], e["trough_i"]
        c = _confirm_i(mv, t)
        row = {k: e[k] for k in ("peak", "trough", "turned", "open")}
        row["confirm"] = str(m.index[c].date()) if c is not None else None
        row["market"] = {
            "fall": e["fall_pct"],
            **{f"reb{h}": _f(_move(mv, t, t + h), 1) for h in HORIZONS},
            **{f"reb{h}_confirm": _f(_move(mv, c, c + h), 1) if c is not None else None for h in HORIZONS},
        }
        row["stocks"] = {}
        for s in prices.columns:
            sv = prices[s].to_numpy(dtype=float)
            row["stocks"][s] = {
                "fall": _f(_move(sv, p, t), 1),
                **{f"reb{h}": _f(_move(sv, t, t + h), 1) for h in HORIZONS},
                **{f"reb{h}_confirm": _f(_move(sv, c, c + h), 1) if c is not None else None
                   for h in HORIZONS},
            }
        out.append(row)
    return out


def summarise(moves: list[dict], symbols: list[str], key: str) -> dict:
    """Per stock across episodes: average fall, average rebound, and how often it beat the index."""
    out = {}
    for s in symbols:
        pts = [(e["stocks"][s]["fall"], e["stocks"][s][key], e["market"][key])
               for e in moves if e["stocks"][s][key] is not None and e["market"][key] is not None]
        if not pts:
            out[s] = None
            continue
        falls = [f for f, _, _ in pts if f is not None]
        rebs = [r for _, r, _ in pts]
        beat = sum(r > mk for _, r, mk in pts)
        avg_fall = float(np.mean(falls)) if falls else None
        avg_reb = float(np.mean(rebs))
        out[s] = {
            "avg_fall": _f(avg_fall, 1), "avg_rebound": _f(avg_reb, 1),
            # What a rebound earned per unit of what the fall cost. >1: the
            # swing paid for itself; <1: it gave back more than it got. Only
            # defined when it actually fell — a stock that rose through the
            # index's falls has no cost to divide by.
            "ratio": _f(avg_reb / abs(avg_fall)) if avg_fall is not None and avg_fall < -0.5 else None,
            "beat": beat, "n": len(pts),
        }
    return out


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    v = ra.corr(rb)
    return None if not math.isfinite(v) else float(v)


def market_now(m: pd.Series, threshold_pct: float) -> dict:
    """Where the index sits against its high of the last year — context, not a call."""
    last = m.tail(TRADING_DAYS)
    hi_i = int(np.argmax(last.to_numpy()))
    dd = (float(m.iloc[-1]) / float(last.iloc[hi_i]) - 1) * 100
    lo = float(last.iloc[hi_i:].min())
    off_low = (float(m.iloc[-1]) / lo - 1) * 100
    return {"dd_from_high_pct": _f(dd, 1), "high_date": str(last.index[hi_i].date()),
            "off_low_pct": _f(off_low, 1), "in_drawdown": bool(dd <= -threshold_pct)}


# ── words ────────────────────────────────────────────────────────────────────
def conclusions(stats: dict, moves: list[dict], summ: dict, now: dict | None,
                labels: dict, bench: str | None, threshold: int, entry: str) -> list[str]:
    name = lambda s: labels.get(s, s)                                    # noqa: E731
    syms = [s for s in stats if s != "__market__"]
    out = []
    by_vol = sorted(syms, key=lambda s: -(stats[s]["vol_annual_pct"] or 0))
    mk = stats.get("__market__")
    out.append(f"波动最大的是 {name(by_vol[0])}（年化 {stats[by_vol[0]]['vol_annual_pct']:.0f}%），"
               f"最小的是 {name(by_vol[-1])}（{stats[by_vol[-1]]['vol_annual_pct']:.0f}%）"
               + (f"；{bench} 本身 {mk['vol_annual_pct']:.0f}%。" if mk else "。"))

    asy = [s for s in syms if stats[s].get("asymmetry") is not None]
    if asy:
        good = max(asy, key=lambda s: stats[s]["asymmetry"])
        bad = min(asy, key=lambda s: stats[s]["asymmetry"])
        g, b = stats[good], stats[bad]
        if g["asymmetry"] > 0.15:
            out.append(f"涨跌不对称：{name(good)} 大盘涨时跟 {g['beta_up']:.2f} 倍、跌时只跟 "
                       f"{g['beta_down']:.2f} 倍 —— 它的大波动主要在上涨一侧。")
        if b["asymmetry"] < -0.15 and bad != good:
            out.append(f"反过来，{name(bad)} 涨时跟 {b['beta_up']:.2f} 倍、跌时却跟 "
                       f"{b['beta_down']:.2f} 倍 —— 它的波动偏向下跌一侧。")
        if g["asymmetry"] <= 0.15 and b["asymmetry"] >= -0.15:
            out.append("这些股票涨跌两侧跟随大盘的程度差不多 —— 波动大就是双向都大，没有免费的反弹。")

    when = "确认反弹（离低点 +3%）后买入" if entry == "confirm" else "在最低点买入（事后才知道）"
    key = "reb60_confirm" if entry == "confirm" else "reb60"
    if not moves:
        out.append(f"所选历史里，{bench or '大盘'} 没有出现过 ≥{threshold}% 的回撤 —— 换个更小的阈值看看。")
    else:
        mk_reb = [e["market"][key] for e in moves if e["market"][key] is not None]
        mk_fall = [e["market"]["fall"] for e in moves]
        n = sum(1 for e in moves if not e["open"])
        live = len(moves) - n
        line = (f"过去 {n} 次 {bench or '大盘'} 回撤 ≥{threshold}%"
                + (f"（另有 1 次仍在进行中）" if live else "")
                + f"，平均跌 {np.mean(mk_fall):.1f}%")
        if mk_reb:
            line += f"，{when}后 60 个交易日大盘平均 {np.mean(mk_reb):+.1f}%"
        out.append(line + "。")
        ranked = sorted([s for s in syms if summ.get(s)], key=lambda s: -summ[s]["avg_rebound"])
        if ranked:
            top = ranked[0]
            t = summ[top]
            fall = (f"但下跌段它平均 {t['avg_fall']:+.1f}%。" if (t["avg_fall"] or 0) < 0
                    else f"而且大盘下跌时它平均反而 {t['avg_fall']:+.1f}% —— 那几次它根本没跟着跌，"
                         f"反弹多不是靠「跌得深」换来的。")
            out.append(f"反弹最多的是 {name(top)}：平均 {t['avg_rebound']:+.1f}%，{t['n']} 次里 "
                       f"{t['beat']} 次跑赢大盘；{fall}"
                       + (f"反弹 / 下跌 = {t['ratio']:.2f}。" if t["ratio"] is not None else ""))
            # The question itself: did bigger swings mean bigger rebounds here?
            vols = [stats[s]["vol_annual_pct"] for s in ranked]
            rebs = [summ[s]["avg_rebound"] for s in ranked]
            ratios = [summ[s]["ratio"] for s in ranked if summ[s]["ratio"] is not None]
            rho = _spearman(vols, rebs)
            if rho is not None:
                verdict = ("是的 —— 波动越大反弹越多" if rho >= 0.6 else
                           "大体如此，但不绝对" if rho >= 0.3 else
                           "并不 —— 波动大小和反弹多少关系不大" if rho > -0.3 else
                           "恰恰相反 —— 波动小的反而反弹得多")
                out.append(f"在这几只里，「波动大 = 反弹多」吗？{verdict}（排序相关 {rho:.2f}）。")
            if ratios:
                best_ratio = max((s for s in ranked if summ[s]["ratio"] is not None),
                                 key=lambda s: summ[s]["ratio"])
                if best_ratio != top:
                    out.append(f"按「反弹 / 下跌」算性价比最高的是 {name(best_ratio)}"
                               f"（{summ[best_ratio]['ratio']:.2f}），不是反弹最多的那只。")
        if n < 4:
            out.append(f"只有 {n} 次回撤可比，样本很少：把它当作这几次的记录，而不是规律。")

    if now:
        state = "正处在回撤中" if now["in_drawdown"] else "不在 ≥{}% 的回撤中".format(threshold)
        out.append(f"现在 {bench or '大盘'} 距一年内高点（{now['high_date']}）{now['dd_from_high_pct']:+.1f}%，"
                   f"离之后的低点 {now['off_low_pct']:+.1f}% —— {state}。这只是位置，不是对底部的判断。")
    return out


# ── assembly ─────────────────────────────────────────────────────────────────
def analyse(window_prices: pd.DataFrame, window_bench: pd.Series | None,
            long_prices: pd.DataFrame, long_bench: pd.Series | None, *,
            labels: dict, bench_label: str | None,
            threshold: int = DEFAULT_THRESHOLD, entry: str = "trough") -> dict:
    """
    `window_*`: the page's window, for the swing statistics.
    `long_*`: as much history as there is, for drawdown episodes — a one-year
    window holds one or two at most.
    """
    if threshold not in THRESHOLDS:
        raise ValueError(f"threshold must be one of {THRESHOLDS}")
    if entry not in ("trough", "confirm"):
        raise ValueError("entry must be trough or confirm")
    symbols = list(window_prices.columns)

    wm = window_bench.reindex(window_prices.index).ffill() if window_bench is not None else None
    stats = swing_stats(window_prices, wm)

    moves, summ, now, eps_n = [], {}, None, 0
    if long_bench is not None:
        lb = long_bench.dropna().sort_index()
        lp = long_prices.reindex(lb.index).ffill(limit=5)
        lp = lp[lp.index >= long_prices.index.min()]
        lb = lb.reindex(lp.index)
        eps = episodes(lb, threshold)
        eps_n = len(eps)
        moves = episode_moves(lp, lb, eps)
        key = "reb60_confirm" if entry == "confirm" else "reb60"
        summ = summarise(moves, symbols, key)
        now = market_now(lb, threshold)

    return {
        "symbols": symbols, "labels": labels, "benchmark": bench_label,
        "threshold": threshold, "entry": entry, "horizons": list(HORIZONS),
        "stats": stats, "episodes": moves, "summary": summ, "now": now,
        "history_from": str(long_prices.index.min().date()),
        "episode_count": eps_n,
        "conclusions": conclusions(stats, moves, summ, now, labels, bench_label, threshold, entry),
    }
