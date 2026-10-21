"""
seesaw.py — do these stocks take turns?

The suspicion this answers: money rotating between two groups (software and
AI hardware, say), so one side rises while the other falls. The obvious test
— their correlation — usually says the opposite, and that is the first thing
worth understanding:

    Raw returns carry the market. On a day the index falls 2%, nearly
    everything falls, so two groups that rotate against each other still show
    a POSITIVE correlation — the shared market move swamps the rotation.

So every statistic here comes in two versions. `raw` is the returns as they
were. `resid` is each stock with its own market exposure taken out — its
return minus beta × the index (沪深300 / S&P 500 / TSX), beta estimated per
stock over the same window. The rotation, if there is one, lives in `resid`.

Two traps avoided on purpose:

  * Residuals are taken against the external INDEX, never against the
    basket's own average. Subtracting the group average forces the members
    to look negatively correlated — the deviations from a mean must sum to
    zero — and would manufacture exactly the seesaw being tested for.
  * When the stocks are split into two camps, the split that makes them look
    most opposed is chosen by searching, which flatters it. So the split is
    also chosen on the FIRST half of the window alone and measured on the
    second half, which it has not seen. A seesaw that is real survives that.

Daily and weekly are both computed because rotation often runs at a weekly
pace: day-to-day noise can hide a pattern that shows clearly week to week, or
a daily seesaw can wash out over a week.
"""

from __future__ import annotations

import itertools
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

ROLL = {"daily": 60, "weekly": 26}       # rolling-correlation windows
RECENT = {"daily": 60, "weekly": 13}     # "these days"
MIN_WEEKS = 20                           # below this, weekly numbers are a guess


def _f(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(v) else round(v, nd)


def weekly_closes(prices: pd.DataFrame) -> pd.DataFrame:
    """Last close of each Mon-Fri week; a week with no session (国庆) drops out."""
    return prices.resample("W-FRI").last().dropna(how="any")


def residuals(r: pd.DataFrame, m: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Each column minus its own beta times the index. Returns (resid, betas)."""
    var = float(m.var())
    betas = {}
    out = {}
    for c in r.columns:
        b = float(r[c].cov(m) / var) if var > 0 else 0.0
        betas[c] = b
        out[c] = r[c] - b * m
    return pd.DataFrame(out, index=r.index), betas


def opposite_share(a: pd.Series, b: pd.Series) -> float | None:
    """Share of periods where one rose and the other fell (flat periods excluded)."""
    sa, sb = np.sign(a.to_numpy()), np.sign(b.to_numpy())
    keep = (sa != 0) & (sb != 0)
    return float(np.mean(sa[keep] != sb[keep])) if keep.any() else None


def search_threshold(n: int, tries: int, alpha: float = 0.05) -> float:
    """
    How negative the MOST negative of `tries` correlations must be before it
    means anything. Picking the lowest of 28 pairs and testing it as if it
    were the only one finds a "seesaw" in pure noise most of the time.
    """
    z = NormalDist().inv_cdf(1 - alpha / (2 * max(1, tries)))
    return z / math.sqrt(max(n, 4))


def _pairs(k: int):
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


def _mean_corr(C: np.ndarray, idx_a, idx_b=None) -> float | None:
    """Mean correlation within one camp (idx_b None) or between two."""
    if idx_b is None:
        vals = [C[i, j] for i, j in itertools.combinations(idx_a, 2)]
    else:
        vals = [C[i, j] for i in idx_a for j in idx_b]
    return float(np.mean(vals)) if vals else None


def best_split(C: np.ndarray) -> tuple[list[int], list[int]] | None:
    """
    The two-camp split whose camps are most opposed: every bipartition is
    tried (at most 127 for eight stocks), scored by how much more the members
    of a camp agree with each other than with the other camp. None when even
    the best split leaves the camps positively correlated — no seesaw.
    """
    k = C.shape[0]
    if k < 3:
        return None
    best, best_score = None, -np.inf
    for mask in range(1, 2 ** (k - 1)):            # stock k-1 always in camp B
        a = [i for i in range(k) if mask >> i & 1]
        b = [i for i in range(k) if not mask >> i & 1]
        between = _mean_corr(C, a, b)
        within = [w for w in (_mean_corr(C, a), _mean_corr(C, b)) if w is not None]
        score = (np.mean(within) if within else 0.0) - between
        if score > best_score:
            best, best_score = (a, b), score
    if best is None or _mean_corr(C, *best) >= 0:
        return None
    return best


def _camp_series(r: pd.DataFrame, cols: list[str]) -> pd.Series:
    return r[cols].mean(axis=1)


def _resolution(prices: pd.DataFrame, m: pd.Series | None, name: str,
                symbols: list[str]) -> dict:
    """Everything for one resolution (daily or weekly)."""
    r = prices.pct_change().dropna()
    mr = m.pct_change().dropna().reindex(r.index) if m is not None else None
    if mr is not None:
        keep = mr.notna()
        r, mr = r[keep], mr[keep]
        resid, betas = residuals(r, mr)
    else:
        resid, betas = None, {}

    k = len(symbols)
    n = len(r)

    def matrices(x: pd.DataFrame | None):
        if x is None:
            return None, None
        C = x.corr().to_numpy()
        opp = [[None if i == j else _f(opposite_share(x.iloc[:, i], x.iloc[:, j]))
                for j in range(k)] for i in range(k)]
        return C, opp

    Craw, opp_raw = matrices(r)
    Cres, opp_res = matrices(resid)

    def lst(x: pd.DataFrame | None):
        if x is None:
            return None
        return {s: [_f(v * 100, 3) for v in x[s].to_numpy()] for s in symbols}

    out = {
        "n": n,
        "significant_at": _f(2 / math.sqrt(n)) if n > 4 else None,
        "significant_min": _f(search_threshold(n, len(_pairs(k)))) if n > 4 else None,
        "dates": [str(d.date()) for d in r.index],
        "returns": lst(r),
        "resid": lst(resid),
        "betas": {s: _f(b, 2) for s, b in betas.items()},
        "raw_corr": [[_f(v) for v in row] for row in Craw],
        "resid_corr": [[_f(v) for v in row] for row in Cres] if Cres is not None else None,
        "opposite_raw": opp_raw,
        "opposite_resid": opp_res,
        "avg_raw": _f(np.mean([Craw[i, j] for i, j in _pairs(k)])),
        "avg_resid": _f(np.mean([Cres[i, j] for i, j in _pairs(k)])) if Cres is not None else None,
        "roll": ROLL[name],
    }

    if Cres is not None:
        i, j = min(_pairs(k), key=lambda p: Cres[p[0], p[1]])
        recent = resid.tail(RECENT[name])
        out["strongest"] = {
            "a": symbols[i], "b": symbols[j], "corr": _f(Cres[i, j]),
            "opposite": _f(opposite_share(resid.iloc[:, i], resid.iloc[:, j])),
            "recent_corr": _f(recent.iloc[:, i].corr(recent.iloc[:, j])),
            "recent_n": len(recent),
        }
    return out | {"_resid_frame": resid, "_raw_frame": r}


def _camps(daily: dict, weekly: dict, symbols: list[str]) -> dict | None:
    """Two opposed camps, if the residuals split into them, with an out-of-sample check."""
    resid = daily["_resid_frame"]
    if resid is None or len(symbols) < 3:
        return None
    C = resid.corr().to_numpy()
    split = best_split(C)
    if split is None:
        return None
    a = [symbols[i] for i in split[0]]
    b = [symbols[i] for i in split[1]]

    # Chosen on the first half, judged on the second.
    half = len(resid) // 2
    first, second = resid.iloc[:half], resid.iloc[half:]
    early = best_split(first.corr().to_numpy())
    oos = None
    same = False
    if early is not None:
        ea = [symbols[i] for i in early[0]]
        eb = [symbols[i] for i in early[1]]
        same = set(map(frozenset, (ea, eb))) == set(map(frozenset, (a, b)))
        oos = _f(_camp_series(second, ea).corr(_camp_series(second, eb)))

    def stats(res: dict) -> dict:
        x = res["_resid_frame"]
        sa, sb = _camp_series(x, a), _camp_series(x, b)
        rec = RECENT["daily" if res is daily else "weekly"]
        return {
            "between": _f(sa.corr(sb)),
            "within_a": _f(_mean_corr(x.corr().to_numpy(), [symbols.index(s) for s in a])),
            "within_b": _f(_mean_corr(x.corr().to_numpy(), [symbols.index(s) for s in b])),
            "opposite": _f(opposite_share(sa, sb)),
            "recent": _f(sa.tail(rec).corr(sb.tail(rec))),
            "n": len(x),
        }

    raw = daily["_raw_frame"]
    spread = ((1 + _camp_series(raw, a)).cumprod() / (1 + _camp_series(raw, b)).cumprod() - 1) * 100
    n2 = max(len(second), 5)
    # Confirmed means the SAME camps come out of the first half alone and
    # stay opposed in the second. A different first-half split that happens
    # to overlap them validates a different grouping, not this one.
    confirmed = same and oos is not None and oos < -2 / math.sqrt(n2)
    rec = resid.tail(RECENT["daily"])
    recent = _camp_series(rec, a).corr(_camp_series(rec, b))
    emerging = (not confirmed and not same and math.isfinite(recent)
                and recent < -2 / math.sqrt(max(len(rec), 5)))
    return {
        "a": a, "b": b, "confirmed": bool(confirmed), "emerging": bool(emerging),
        "oos_threshold": _f(2 / math.sqrt(n2)),
        "daily": stats(daily),
        "weekly": stats(weekly) if weekly["_resid_frame"] is not None else None,
        "out_of_sample": {"corr": oos, "same_split": same, "half_from": str(second.index[0].date())},
        "spread": [_f(v, 2) for v in spread.to_numpy()],
        "spread_dates": [str(d.date()) for d in spread.index],
    }


def conclusions(d: dict, w: dict, camps: dict | None, labels: dict, bench: str | None) -> list[str]:
    name = lambda s: labels.get(s, s)                               # noqa: E731
    out = []
    if d["avg_raw"] is not None:
        tone = ("大盘把它们绑在一起，同涨同跌是主旋律" if d["avg_raw"] >= 0.3 else
                "本来就不太一起动" if d["avg_raw"] < 0.1 else "有一定联动")
        out.append(f"原始日收益两两平均相关 {d['avg_raw']:.2f}：{tone}。")
    if d.get("avg_resid") is None:
        out.append("指数数据不可用，无法扣除大盘 —— 只能看原始相关，跷跷板可能被大盘的共同涨跌掩盖。")
        return out
    out.append(f"按各自 β 扣除{bench or '大盘'}后，平均相关降到 {d['avg_resid']:.2f}"
               f"（周线 {w['avg_resid']:.2f}）。跷跷板只会出现在这一层。")

    s = d["strongest"]
    thr = d["significant_min"] or 1
    if s["corr"] is not None and s["corr"] < -thr:
        wc = None
        if w.get("resid_corr") is not None:
            ia, ib = list(d["betas"]).index(s["a"]), list(d["betas"]).index(s["b"])
            wc = w["resid_corr"][ia][ib]
        out.append(
            f"最明显的跷跷板：{name(s['a'])} 与 {name(s['b'])}，去大盘后日相关 {s['corr']:.2f}"
            + (f"、周相关 {wc:.2f}" if wc is not None else "")
            + f"；{s['opposite'] * 100:.0f}% 的交易日一涨一跌。")
        if s["recent_corr"] is not None:
            delta = s["recent_corr"] - s["corr"]
            trend = "在加剧" if delta < -0.1 else "在减弱" if delta > 0.1 else "大致稳定"
            out.append(f"最近 {s['recent_n']} 个交易日这对的相关是 {s['recent_corr']:.2f} —— {trend}。")
    else:
        out.append(f"去掉大盘后，没有哪两只显著负相关（显著线约 −{thr:.2f}）—— "
                   f"看到的此消彼长更可能是噪音，或者只在周线上才有。")
        wthr = w.get("significant_min") or 1
        wsx = w.get("strongest") or {}
        if wsx.get("corr") is not None and wsx["corr"] < -wthr:
            out.append(f"周线上倒是有：{name(wsx['a'])} 与 {name(wsx['b'])} 周相关 {wsx['corr']:.2f}"
                       f"（显著线约 −{wthr:.2f}）。轮动按周发生，日内噪音盖住了它。")

    if camps:
        cd = camps["daily"]
        a = "、".join(name(x) for x in camps["a"])
        b = "、".join(name(x) for x in camps["b"])
        wk = f" / 周 {camps['weekly']['between']:.2f}" if camps.get("weekly") else ""
        o = camps["out_of_sample"]
        same = "（与全段分组相同）" if o["same_split"] else "（与全段分组不同）"
        if camps["confirmed"]:
            out.append(f"两个阵营：【{a}】对【{b}】。组间去大盘相关 日 {cd['between']:.2f}{wk}。")
            out.append(f"检验：只用前半段数据分组{same}，在没见过的后半段（{o['half_from']} 起）"
                       f"组间相关 {o['corr']:.2f} —— 前后一致，跷跷板站得住。")
        elif camps["emerging"]:
            out.append(f"最近才出现：【{a}】对【{b}】。前半段的数据分不出这两组，"
                       f"但最近 {RECENT['daily']} 个交易日组间去大盘相关 {cd['recent']:.2f} —— "
                       f"跷跷板可能是新近形成的，还没有足够长的历史来检验。")
        elif o["corr"] is not None and o["corr"] < 0 and o["same_split"]:
            out.append(f"有迹象：【{a}】对【{b}】（组间 日 {cd['between']:.2f}{wk}）。"
                       f"只用前半段数据分组也得到同样的两组，后半段组间相关 {o['corr']:.2f}，"
                       f"方向一致但还不够显著（需低于 −{camps['oos_threshold']:.2f}）—— "
                       f"值得继续观察，证据还不够。")
        else:
            oc = "—" if o["corr"] is None else f"{o['corr']:.2f}"
            out.append(f"勉强能分成【{a}】对【{b}】（组间 日 {cd['between']:.2f}{wk}），"
                       f"但只用前半段数据分组{same}、拿到后半段检验时组间相关只有 {oc} —— "
                       f"这个分组更像是从噪音里挑出来的，不算跷跷板。")
    elif len(d["betas"]) >= 3:
        out.append("去大盘后，这些股票分不出两个互相对立的阵营。")

    if w["n"] < MIN_WEEKS:
        out.append(f"周线只有 {w['n']} 个数据点，周线结论仅供参考；把窗口拉长到 1 年或全部。")
    return out


def analyse(prices: pd.DataFrame, bench: pd.Series | None, *,
            labels: dict, bench_label: str | None) -> dict:
    """`prices`: one column per symbol, already cut to common sessions and the window."""
    symbols = list(prices.columns)
    m = bench.reindex(prices.index).ffill() if bench is not None else None
    if m is not None and m.isna().any():
        keep = m.notna()
        prices, m = prices[keep], m[keep]

    d = _resolution(prices, m, "daily", symbols)
    wp = weekly_closes(prices)
    wm = weekly_closes(m.to_frame("m"))["m"].reindex(wp.index) if m is not None else None
    w = _resolution(wp, wm, "weekly", symbols)
    camps = _camps(d, w, symbols)
    lines = conclusions(d, w, camps, labels, bench_label)
    for x in (d, w):
        x.pop("_resid_frame")
        x.pop("_raw_frame")
    return {"symbols": symbols, "labels": labels, "benchmark": bench_label,
            "daily": d, "weekly": w, "camps": camps, "conclusions": lines}
