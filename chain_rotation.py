"""
chain_rotation.py — 接力与背离 along a chain of groups.

A chain is an ordered list of legs, upstream to downstream (电子布 → 光纤 →
光模块 → HPC), each leg an equal-weight group of stocks. Two questions are
asked of it, both the way a trader notices them rather than the way a
textbook tests them.

接力 (relay)
------------
"When this leg has a big day, does that one move more than usual over the
next few days?"

  * A big day is big FOR THAT LEG: its market-adjusted move in units of its
    own recent volatility (trailing 60 sessions, not including the day). A 5%
    day in a leg that routinely swings 4% is not an event; 2% in a leg that
    moves 0.8% is. The market's part (beta × 沪深300) is removed first, so a
    day when everything jumped is not mistaken for this leg's day.
  * The response is measured the same way — the follower's market-adjusted
    move over the next 1/3/5 sessions in ITS own σ — and compared with what
    the follower does over any 1/3/5 sessions. "Moved more than its daily
    behaviour" is the excess over that norm.
  * Each cell is tested against random days (the same number of them, drawn
    2,000 times), corrected for the number of cells tested (Benjamini-
    Hochberg), and split in time: a relay that holds in the first half of
    the history and the second half is worth more than one that lives in a
    single burst.
  * Events closer than three sessions to the previous one are merged, so one
    violent week is one event rather than four.

背离 (divergence)
-----------------
"An upstream leg has rolled over while a downstream leg is still at its
highs" — and what the downstream leg did afterwards.

  * Weak: ≥5% below its own 20-session high, and lower than ten sessions ago.
  * Strong: within 3% of its own 60-session high.
  * Both use only data up to that day, so a past divergence is flagged on
    the day it could have been seen, not with hindsight.
  * Each episode is followed: how many sessions until the downstream leg
    peaked, how far it fell from that peak, and its next-20-session return —
    next to the same numbers for days the downstream leg was equally strong
    WITHOUT upstream weakness. That control is what makes "divergence
    preceded the top" a finding rather than a description of every top.

Also: breadth, the share of all the chain's stocks above their 20-day
average. Breadth falling while the downstream leg holds its high is the
narrowing that usually accompanies the last leg of a theme.

A-shares only: the 板块 these chains are built from are A-share lists.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

LOOKBACKS = (252, 504, 756)
DEFAULT_LOOKBACK = 504
THRESHOLDS = (1.5, 2.0, 2.5)
DEFAULT_THRESHOLD = 2.0
HORIZONS = (1, 3, 5)
SIGMA_WIN = 60
DECLUSTER = 3
PLACEBO_DRAWS = 2000
MAX_LEGS = 6
MAX_PER_LEG = 12

WEAK_DROP = 0.05          # below its 20-session high by this much
WEAK_LOOKBACK = 10        # and lower than this many sessions ago
STRONG_NEAR = 0.03        # within this of its 60-session high
EPISODE_GAP = 3           # divergence days this close are one episode
EPISODE_MIN = 2           # and it must last at least this long
PEAK_WINDOW = 30          # sessions to look for the downstream peak
AFTER_PEAK = 20           # sessions after the peak to measure the fall
WARMUP = 70               # sessions needed before anything is measured


def _f(v, nd=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(v) else round(v, nd)


# ── legs ─────────────────────────────────────────────────────────────────────
def leg_returns(closes: pd.DataFrame, legs: list[dict]) -> pd.DataFrame:
    """Equal-weight daily return of each leg, over whichever members traded that day."""
    r = closes.pct_change()
    out = {}
    for leg in legs:
        cols = [t for t in leg["tickers"] if t in r.columns]
        if cols:
            out[leg["name"]] = r[cols].mean(axis=1, skipna=True)
    return pd.DataFrame(out)


def abnormal(leg_r: pd.DataFrame, m: pd.Series):
    """
    Market-adjusted returns, and each in units of the leg's own trailing σ.

    Beta is fitted over the whole window — it moves slowly, and a rolling
    estimate would cost a year of warm-up on a two-year history. σ is
    trailing and excludes the day itself, so a big day cannot inflate the
    yardstick it is measured with.
    """
    a, betas = {}, {}
    mv = m.reindex(leg_r.index)
    var = float(mv.var())
    for c in leg_r.columns:
        b = float(leg_r[c].cov(mv) / var) if var > 0 else 1.0
        betas[c] = b
        a[c] = leg_r[c] - b * mv
    a = pd.DataFrame(a)
    sigma = a.rolling(SIGMA_WIN, min_periods=SIGMA_WIN // 2).std().shift(1)
    return a, sigma, a / sigma, betas


def events(z: pd.Series, k: float, direction: str) -> list[int]:
    """Positions of big days for one leg, merged when they come in a cluster."""
    v = z.to_numpy()
    hit = (v >= k) if direction == "up" else (v <= -k)
    out, last = [], -10 ** 9
    for i in np.flatnonzero(np.nan_to_num(hit, nan=False)):
        if i - last > DECLUSTER:
            out.append(int(i))
            last = i
    return out


def _paths(a: np.ndarray, sigma: np.ndarray, starts: np.ndarray, H: int) -> np.ndarray:
    """
    Cumulative market-adjusted move over days +1..+H after each start, in the
    follower's own daily σ (as known at the start). NaN where the future is
    not there yet.
    """
    n = len(a)
    out = np.full((len(starts), H), np.nan)
    for row, t in enumerate(starts):
        if t + H >= n:
            continue
        s = sigma[t + 1]
        if not np.isfinite(s) or s <= 0:
            continue
        out[row] = np.cumsum(a[t + 1:t + 1 + H]) / s
    return out


def bh_q(ps: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, in the input order."""
    p = np.asarray(ps, dtype=float)
    n = len(p)
    if not n:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out.tolist()


def signed_events(z: pd.Series, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Big days in either direction, merged when clustered: (positions, signs)."""
    v = z.to_numpy()
    hit = np.nan_to_num(np.abs(v) >= k, nan=False)
    pos, sgn, last = [], [], -10 ** 9
    for i in np.flatnonzero(hit):
        if i - last > DECLUSTER:
            pos.append(int(i))
            sgn.append(1.0 if v[i] > 0 else -1.0)
            last = i
    return np.asarray(pos, dtype=int), np.asarray(sgn)


def relay(a: pd.DataFrame, sigma: pd.DataFrame, z: pd.DataFrame, k: float,
          dates: pd.DatetimeIndex, seed: int = 7) -> dict:
    """
    Every leader -> follower cell at every horizon.

    Up and down days are tested together, as "did the follower move the
    leader's way": the follower's move is multiplied by the sign of the
    leader's day. Tested separately they split the events in two and
    double the number of tests, and a real two-day relay planted into
    synthetic data was found in 2 runs of 20; pooled, the same relay is
    found far more often while noise is flagged no more often. The up-only
    and down-only averages are still reported, for relays that only run
    one way.
    """
    rng = np.random.default_rng(seed)
    names = list(a.columns)
    n = len(a)
    mid = dates[WARMUP + (n - WARMUP) // 2] if n > WARMUP else dates[n // 2]
    all_starts = np.arange(WARMUP, n)
    out = {}
    for H in HORIZONS:
        base_of = {}
        for F in names:
            base = _paths(a[F].to_numpy(), sigma[F].to_numpy(), all_starts, H)
            end = base[:, -1] / math.sqrt(H)
            base_of[F] = (base, end[np.isfinite(end)])
        cells = []
        for L in names:
            ev, sg = signed_events(z[L], k)
            keep = ev >= WARMUP
            ev, sg = ev[keep], sg[keep]
            zl, rl = z[L].to_numpy(), a[L].to_numpy()
            for F in names:
                if F == L:
                    continue
                base, pool = base_of[F]
                mu = float(pool.mean()) if len(pool) else 0.0
                cell = {"leader": L, "follower": F, "n": 0, "excess": None,
                        "follow_rate": None, "p": None, "first": None, "second": None,
                        "up": None, "down": None, "n_up": 0, "n_down": 0,
                        "curve": None, "base_curve": None, "events": []}
                if len(ev) and len(pool) > 30:
                    paths = _paths(a[F].to_numpy(), sigma[F].to_numpy(), ev, H)
                    ends = paths[:, -1] / math.sqrt(H)
                    ok = np.isfinite(ends)
                    if ok.sum() >= 3:
                        e_ok, s_ok, p_ok, end_ok = ev[ok], sg[ok], paths[ok], ends[ok]
                        # Everything in the leader's direction: + means "followed".
                        signed = end_ok * s_ok
                        mu_s = float(np.mean(s_ok)) * mu
                        m_obs = float(signed.mean())
                        draws = (pool[rng.integers(0, len(pool), (PLACEBO_DRAWS, len(end_ok)))]
                                 * s_ok[None, :]).mean(axis=1)
                        p = float((1 + np.sum(np.abs(draws - mu_s) >= abs(m_obs - mu_s)))
                                  / (PLACEBO_DRAWS + 1))
                        early = dates[e_ok] < mid
                        upm, dnm = s_ok > 0, s_ok < 0
                        base_curve = np.nanmean(base, axis=0)
                        cell.update({
                            "n": int(ok.sum()),
                            "excess": _f(m_obs - mu_s),
                            "follow_rate": _f(float(np.mean(signed > 0))),
                            "p": _f(p, 4),
                            "first": _f(float(signed[early].mean()) - mu_s) if early.any() else None,
                            "second": _f(float(signed[~early].mean()) - mu_s) if (~early).any() else None,
                            "first_n": int(early.sum()), "second_n": int((~early).sum()),
                            "up": _f(float(end_ok[upm].mean()) - mu) if upm.any() else None,
                            "down": _f(float(end_ok[dnm].mean()) - mu) if dnm.any() else None,
                            "n_up": int(upm.sum()), "n_down": int(dnm.sum()),
                            # Paths in the leader's direction, against the follower's norm.
                            "curve": [_f(v) for v in (p_ok * s_ok[:, None]).mean(axis=0)],
                            "base_curve": [_f(v * float(np.mean(s_ok))) for v in base_curve],
                            "events": [{
                                "date": str(dates[t].date()),
                                "leader_z": _f(zl[t]), "leader_pct": _f(rl[t] * 100),
                                "path": [_f(v) for v in path],
                                "end": _f(e), "followed": bool(e * sgn > 0),
                            } for t, sgn, path, e in zip(e_ok, s_ok, p_ok, end_ok)],
                        })
                cells.append(cell)
        out[str(H)] = cells
    # One family for every cell on the page, every horizon: the reader sees
    # all of them, so all of them are what has to be corrected for.
    every = [c for cells in out.values() for c in cells]
    qs = bh_q([c["p"] if c["p"] is not None else 1.0 for c in every])
    for c, q in zip(every, qs):
        c["q"] = _f(q, 4) if c["p"] is not None else None
        c["stable"] = bool(c["first"] is not None and c["second"] is not None
                           and c["first"] * c["second"] > 0
                           and min(c.get("first_n", 0), c.get("second_n", 0)) >= 2)
    return out


def live(z: pd.DataFrame, a: pd.DataFrame, sigma: pd.DataFrame, cells: dict,
         k: float, dates: pd.DatetimeIndex, recent: int = 3) -> list[dict]:
    """Big days in the last `recent` sessions, with what has historically followed them."""
    out = []
    n = len(z)
    for L in z.columns:
        zl = z[L].to_numpy()
        for t in range(max(0, n - recent), n):
            if not np.isfinite(zl[t]) or abs(zl[t]) < k:
                continue
            direction = "up" if zl[t] > 0 else "down"
            followers = []
            for H in HORIZONS:
                for c in cells[str(H)]:
                    if c["leader"] != L or c["q"] is None or c["q"] > 0.2:
                        continue
                    F = c["follower"]
                    so_far = np.nansum(a[F].to_numpy()[t + 1:]) / sigma[F].to_numpy()[min(t + 1, n - 1)] \
                        if t + 1 < n else 0.0
                    followers.append({"follower": F, "horizon": H, "excess": c["excess"],
                                      "follow_rate": c["follow_rate"], "n": c["n"], "q": c["q"],
                                      "stable": c["stable"], "since": _f(so_far),
                                      "days_since": n - 1 - t})
            out.append({"leg": L, "date": str(dates[t].date()), "z": _f(zl[t]),
                        "pct": _f(a[L].to_numpy()[t] * 100), "direction": direction,
                        "followers": followers})
    return out


# ── divergence ───────────────────────────────────────────────────────────────
def leg_index(closes: pd.DataFrame, legs: list[dict]) -> pd.DataFrame:
    """
    Each leg as equal amounts bought at the start and held — what owning the
    leg was actually like. Compounding the daily equal-weight return instead
    rebalances every day, which harvests the members' swings against each
    other: 光模块中游 came to +731% that way over two years against +603% for
    the same six stocks simply held. Peaks, falls and "near its high" are
    read off this line, so it has to be the one a holder lived through.
    """
    out = {}
    for leg in legs:
        cols = [t for t in leg["tickers"] if t in closes.columns]
        if not cols:
            continue
        px = closes[cols].ffill()
        first = px.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
        out[leg["name"]] = (px / first).mean(axis=1, skipna=True)
    return pd.DataFrame(out)


def flags(idx: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(weak, strong) per day, each using only data up to that day."""
    hi20 = idx.rolling(20, min_periods=20).max()
    hi60 = idx.rolling(60, min_periods=60).max()
    weak = (idx <= hi20 * (1 - WEAK_DROP)) & (idx < idx.shift(WEAK_LOOKBACK))
    strong = idx >= hi60 * (1 - STRONG_NEAR)
    return weak.fillna(False), strong.fillna(False)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Episodes of a boolean series, gaps of up to EPISODE_GAP days bridged."""
    on = np.flatnonzero(mask)
    if not len(on):
        return []
    out, s, e = [], on[0], on[0]
    for i in on[1:]:
        if i - e <= EPISODE_GAP + 1:
            e = i
        else:
            out.append((s, e))
            s = e = i
    out.append((s, e))
    return [(s, e) for s, e in out if e - s + 1 >= EPISODE_MIN]


def _after(v: np.ndarray, s: int) -> dict:
    """What a leg did from day s: peak, the fall from it, and its 20-session return."""
    n = len(v)
    end = min(n, s + PEAK_WINDOW + 1)
    if end - s < 5:
        return {"complete": False}
    seg = v[s:end]
    pk = int(np.argmax(seg))
    peak_i = s + pk
    after = v[peak_i:min(n, peak_i + AFTER_PEAK + 1)]
    return {
        "complete": bool(s + PEAK_WINDOW < n),
        "days_to_peak": pk,
        "peak_gain_pct": _f((seg[pk] / v[s] - 1) * 100, 1),
        "fall_after_peak_pct": _f((after.min() / v[peak_i] - 1) * 100, 1) if len(after) > 1 else None,
        "fwd20_pct": _f((v[s + 20] / v[s] - 1) * 100, 1) if s + 20 < n else None,
    }


def divergence(idx: pd.DataFrame, order: list[str], dates: pd.DatetimeIndex) -> list[dict]:
    """Every upstream → downstream pair, its episodes, a control, and today's state."""
    fl = {c: flags(idx[c]) for c in order}
    out = []
    for i, up in enumerate(order):
        for down in order[i + 1:]:
            weak = fl[up][0].to_numpy()
            strong = fl[down][1].to_numpy()
            div = weak & strong
            div[:WARMUP] = False
            v = idx[down].to_numpy()
            eps = []
            for s, e in runs(div):
                eps.append({"start": str(dates[s].date()), "end": str(dates[e].date()),
                            "days": int(e - s + 1),
                            "open": bool(e >= len(v) - 1 - EPISODE_GAP),
                            **_after(v, s)})
            # The control: days the downstream leg was just as strong, with the
            # upstream leg NOT weak — what "strong downstream" leads to anyway.
            ctrl_days = np.flatnonzero(strong & ~weak & (np.arange(len(v)) >= WARMUP))
            ctrl = [(v[t + 20] / v[t] - 1) * 100 for t in ctrl_days if t + 20 < len(v)]
            done = [x["fwd20_pct"] for x in eps if x.get("fwd20_pct") is not None]
            p = None
            if ctrl and done:
                rng = np.random.default_rng(11)
                arr = np.asarray(ctrl)
                draws = arr[rng.integers(0, len(arr), (PLACEBO_DRAWS, len(done)))].mean(axis=1)
                p = float((1 + np.sum(draws <= np.mean(done))) / (PLACEBO_DRAWS + 1))
            run_now = 0
            for x in div[::-1]:
                if not x:
                    break
                run_now += 1
            out.append({
                "up": up, "down": down, "episodes": eps,
                "control": {"n_days": len(ctrl), "fwd20_mean": _f(np.mean(ctrl), 1) if ctrl else None,
                            "fwd20_neg_share": _f(float(np.mean(np.asarray(ctrl) < 0)), 2) if ctrl else None},
                "episodes_fwd20_mean": _f(np.mean(done), 1) if done else None,
                "p_lower": _f(p, 4),
                "now": {"diverging": bool(div[-1]), "days": run_now,
                        "up_weak": bool(weak[-1]), "down_strong": bool(strong[-1]),
                        "down_off_high_pct": _f((v[-1] / pd.Series(v).tail(60).max() - 1) * 100, 1)},
                "mask": [bool(x) for x in div],
            })
    return out


def breadth(closes: pd.DataFrame) -> pd.Series:
    """Share of the chain's stocks above their own 20-day average, 0-100."""
    ma = closes.rolling(20, min_periods=20).mean()
    above = (closes > ma).where(ma.notna())
    return above.mean(axis=1) * 100


# ── words ────────────────────────────────────────────────────────────────────
def conclusions(rel: dict, div: list[dict], lv: list[dict], k: float, br: pd.Series,
                idx: pd.DataFrame, order: list[str]) -> list[str]:
    out = []
    for item in lv:
        word = "大涨" if item["direction"] == "up" else "大跌"
        when = "今天" if item["date"] == str(idx.index[-1].date()) else item["date"]
        fs = sorted(item["followers"], key=lambda f: f["q"])
        if fs:
            f = fs[0]
            verb = "同向跟随" if (f["excess"] or 0) > 0 else "反向"
            out.append(f"{when} {item['leg']} {word} {item['z']:+.1f}σ（{item['pct']:+.1f}%，已扣除大盘）。"
                       f"历史上之后 {f['horizon']} 天 {f['follower']} 平均{verb} {abs(f['excess']):.2f}σ"
                       f"（{f['n']} 次，跟随率 {f['follow_rate'] * 100:.0f}%"
                       f"{'，前后两半都成立' if f['stable'] else ''}）"
                       + ("。" if f["days_since"] == 0
                          else f"；之后 {f['days_since']} 天它已动了 {f['since']:+.1f}σ。"))
        else:
            out.append(f"{when} {item['leg']} {word} {item['z']:+.1f}σ —— 历史上它的{word}之后，"
                       f"链上没有哪一环有超出随机的反应。")

    found = []
    for key, cells in rel.items():
        for c in cells:
            if c["q"] is not None and c["q"] <= 0.1 and c["n"] >= 5:
                found.append((c["q"], int(key), c))
    found.sort(key=lambda x: x[0])
    seen = set()
    shown = 0
    for q, H, c in found:
        pair = (c["leader"], c["follower"])
        if pair in seen or shown >= 4:
            continue
        seen.add(pair)
        shown += 1
        if (c["excess"] or 0) > 0:
            kind = f"接力：{c['follower']} 同向多走 {c['excess']:.2f}σ"
        else:
            kind = f"反向：{c['follower']} 反着走 {abs(c['excess']):.2f}σ（资金像是在两者间切换）"
        side = ""
        if c["up"] is not None and c["down"] is not None and c["n_up"] >= 3 and c["n_down"] >= 3:
            side = f"（大涨后 {c['up']:+.2f}σ / 大跌后 {c['down']:+.2f}σ）"
        out.append(f"{c['leader']} 出现 ≥{k:g}σ 的大涨或大跌后 {H} 天 —— {kind}{side}；"
                   f"{c['n']} 次，跟随率 {c['follow_rate'] * 100:.0f}%，随机日很难做到（q={q:.3f}）"
                   + ("，前后两半都成立。" if c["stable"] else "，但只在一半历史里明显。"))
    if not found:
        out.append(f"接力：在 ≥{k:g}σ 的大涨/大跌之后，链上各环的反应都与随机日无异（已按检验次数校正）。"
                   f"可以试试降低阈值，或换一组更接近的环节。")

    # Divergence: only what is live now or clearly mattered; the table has the rest.
    MIN_EPISODES = 3
    told = 0
    quiet = 0
    for d in div:
        closed = [e for e in d["episodes"] if e.get("complete")]
        if d["now"]["diverging"]:
            out.append(f"背离进行中：{d['up']} 走弱、{d['down']} 仍在高位，已持续 {d['now']['days']} 天"
                       f"（{d['down']} 距 60 日高点 {d['now']['down_off_high_pct']:+.1f}%）。")
        clear = (len(closed) >= MIN_EPISODES and d["p_lower"] is not None and d["p_lower"] < 0.05)
        if not closed:
            continue
        if not (clear or d["now"]["diverging"]):
            quiet += 1
            continue
        told += 1
        peaks = [e["days_to_peak"] for e in closed]
        falls = [e["fall_after_peak_pct"] for e in closed if e["fall_after_peak_pct"] is not None]
        line = (f"{d['up']} → {d['down']}：历史上 {len(closed)} 次背离，{d['down']} 之后中位 "
                f"{int(np.median(peaks))} 个交易日见顶"
                + (f"，见顶后平均回落 {np.mean(falls):.1f}%" if falls else "") + "。")
        if d["episodes_fwd20_mean"] is not None and d["control"]["fwd20_mean"] is not None:
            line += (f"背离后 20 天平均 {d['episodes_fwd20_mean']:+.1f}%，同样在高位但上游没走弱时 "
                     f"{d['control']['fwd20_mean']:+.1f}%")
            if len(closed) < MIN_EPISODES:
                line += f" —— 只有 {len(closed)} 次，不足以下结论。"
            elif clear:
                line += " —— 差别超出随机。"
            else:
                line += " —— 差别还不足以排除巧合。"
        out.append(line)
    if quiet:
        out.append(f"另有 {quiet} 对环节出现过背离，但次数太少或之后的表现与平常无异 —— 见背离页的表格。")
    if not any(d["episodes"] for d in div) and len(order) >= 2:
        out.append("背离：所选历史里没有出现过「上游走弱、下游仍在高位」的情形。")

    b = br.dropna()
    last_leg = order[-1] if order else None
    if len(b) > 10 and last_leg is not None:
        drop = float(b.iloc[-1] - b.tail(10).max())
        strong_now = bool(flags(idx[last_leg])[1].iloc[-1])
        if drop <= -25 and strong_now:
            out.append(f"宽度收窄：全链 20 日线上方的股票比例 10 天内下降 {abs(drop):.0f} 个百分点，"
                       f"而 {last_leg} 仍在高位 —— 上涨越来越集中在少数票上。")
    return out


# ── assembly ─────────────────────────────────────────────────────────────────
def analyse(closes: pd.DataFrame, bench: pd.Series, legs: list[dict], *,
            k: float = DEFAULT_THRESHOLD, lookback: int = DEFAULT_LOOKBACK) -> dict:
    """
    `closes`: adjusted closes, one column per ticker, with at least
    `lookback` + WARMUP sessions. `legs`: [{name, tickers}] in chain order.
    """
    if k not in THRESHOLDS:
        raise ValueError(f"k must be one of {THRESHOLDS}")
    if lookback not in LOOKBACKS:
        raise ValueError(f"lookback must be one of {LOOKBACKS}")
    if not 2 <= len(legs) <= MAX_LEGS:
        raise ValueError(f"需要 2–{MAX_LEGS} 个环节")
    names = [leg["name"] for leg in legs]
    if len(set(names)) != len(names):
        raise ValueError("环节名称不能重复")

    closes = closes.sort_index()
    m = bench.reindex(closes.index).ffill()
    keep = m.notna()
    closes, m = closes[keep], m[keep]
    closes = closes.tail(lookback + WARMUP)
    m = m.reindex(closes.index)
    mr = m.pct_change()

    lr = leg_returns(closes, legs).iloc[1:]
    mr = mr.reindex(lr.index)
    missing = [leg["name"] for leg in legs if leg["name"] not in lr.columns]
    if missing:
        raise LookupError(f"这些环节没有可用的行情：{'、'.join(missing)}")
    dates = lr.index
    a, sigma, z, betas = abnormal(lr, mr)
    rel = relay(a, sigma, z, k, dates)
    lv = live(z, a, sigma, rel, k, dates)
    idx = leg_index(closes, legs).reindex(dates)
    div = divergence(idx, names, dates)
    br = breadth(closes).reindex(dates)

    show = slice(WARMUP, None)
    bench_idx = (1 + mr.fillna(0)).cumprod()
    base = idx.iloc[WARMUP]
    out = {
        "legs": [{"name": leg["name"], "tickers": [t for t in leg["tickers"] if t in closes.columns],
                  "beta": _f(betas.get(leg["name"]))} for leg in legs],
        "k": k, "lookback": lookback,
        "dates": [str(d.date()) for d in dates[show]],
        "index": {c: [_f((v / base[c] - 1) * 100, 2) for v in idx[c].to_numpy()[show]] for c in names},
        "bench": [_f((v / bench_idx.iloc[WARMUP] - 1) * 100, 2) for v in bench_idx.to_numpy()[show]],
        "breadth": [_f(v, 1) for v in br.to_numpy()[show]],
        "z": {c: [_f(v) for v in z[c].to_numpy()[show]] for c in names},
        "relay": rel,
        "live": lv,
        "divergence": [{**d, "mask": d["mask"][WARMUP:]} for d in div],
        "conclusions": conclusions(rel, div, lv, k, br, idx, names),
    }
    return out
