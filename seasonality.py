"""
seasonality.py — does this stock tend to do the same thing at the same time
of year?

Every year is laid on one calendar (Jan 1 → Dec 31) and measured from the
previous year's last close, so 2023, 2024 and 2025 can be read against each
other day by day, week by week and month by month.

The hard part is not drawing it. It is not fooling yourself.
------------------------------------------------------------
With three years and 52 weeks there are about 330 windows of 2-8 weeks, and
any one of them has a 1-in-4 chance of having gone the same way three years
running by luck alone. Search them all and you will ALWAYS find a "reliable"
window. So every window found here is scored against the same search run on
shuffled calendars: each year's weekly returns are rotated by a random number
of weeks — keeping that year's volatility, trend and streakiness, destroying
only the date — and the best window is found again, 300 times. A real
seasonal pattern has to beat what the shuffled calendars produce. With three
years it usually does not, and the conclusions say so plainly.

Two more honest corrections are offered rather than imposed:

  * `excess` — the same analysis on the stock minus its index. A-share
    seasonality is mostly the market's (the spring rally, the April earnings
    gauntlet); what is left after subtracting 沪深300 is what belongs to the
    stock.
  * For A-shares, years are tagged by the 924 policy pivot (2024-09-24). The
    pattern in years before it describes a market that has since changed.

Calendar alignment uses month-day, so Feb 29 has its own slot and a non-leap
year simply has no bar there. Weeks are fixed 7-day blocks from Jan 1, not
ISO weeks, so week 1 is Jan 1-7 in every year; the last block runs to Dec 31.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

YEAR_CHOICES = (3, 5, 10)
WEEKS = 52
MIN_WINDOW, MAX_WINDOW = 2, 8          # weeks
SHUFFLES = 300
#: A window has to have moved at least this much in its WEAKEST qualifying
#: year to be reported. Below it, "all three years were up" means "flat".
MIN_MOVE = 0.01
HORIZONS = (30, 90)                    # calendar days, for "what comes next"
SPRING_DAYS = 10                       # trading days either side of 春节
REGIME_YEAR = 2024                     # the 924 pivot, A-shares only

#: 春节 (first day of the lunar year). Markets close for the week around it,
#: and the run-up and run-off are the best-known A-share seasonal effect.
SPRING_FESTIVAL = {
    2014: date(2014, 1, 31), 2015: date(2015, 2, 19), 2016: date(2016, 2, 8),
    2017: date(2017, 1, 28), 2018: date(2018, 2, 16), 2019: date(2019, 2, 5),
    2020: date(2020, 1, 25), 2021: date(2021, 2, 12), 2022: date(2022, 2, 1),
    2023: date(2023, 1, 22), 2024: date(2024, 2, 10), 2025: date(2025, 1, 29),
    2026: date(2026, 2, 17), 2027: date(2027, 2, 6), 2028: date(2028, 1, 26),
    2029: date(2029, 2, 13), 2030: date(2030, 2, 3),
}

_CAL0 = date(2000, 1, 1)               # a leap year, so every month-day has a slot


# ── calendar ─────────────────────────────────────────────────────────────────
def doy(d) -> int:
    """Day slot 0..365 on a leap calendar: the same month-day, the same slot."""
    return (date(2000, d.month, d.day) - _CAL0).days


def bucket(slot: int) -> int:
    """Week 0..51. The last block takes the leftover days to Dec 31."""
    return min(slot // 7, WEEKS - 1)


def week_labels() -> list[str]:
    out = []
    for k in range(WEEKS):
        a = _CAL0 + timedelta(days=7 * k)
        b = date(2000, 12, 31) if k == WEEKS - 1 else a + timedelta(days=6)
        out.append(f"{a:%m-%d}~{b:%m-%d}")
    return out


def start_date(years: int, today: date) -> date:
    """Fetch from here: the December before the first full year, for its base."""
    return date(today.year - years - 1, 12, 1)


# ── one year ─────────────────────────────────────────────────────────────────
def _year(close: pd.Series, y: int) -> pd.Series:
    return close[close.index.year == y]


def _base(close: pd.Series, y: int) -> float | None:
    prev = _year(close, y - 1)
    return float(prev.iloc[-1]) if len(prev) else None


def _complete(close: pd.Series, y: int) -> bool:
    """A full year has a base (last year's close) and trades into late December."""
    s = _year(close, y)
    return (_base(close, y) is not None and len(s) > 150
            and s.index[-1] >= pd.Timestamp(y, 12, 20))


def _weekly(close: pd.Series, y: int, upto: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Log return of each week block in year y, and whether the block traded.

    A block with no session (国庆, 春节 weeks) returns 0 — the price did not
    move because the market was shut — and is marked untraded so the grid
    can grey it rather than paint it flat. Blocks after `upto` are NaN.
    """
    s = _year(close, y)
    base = _base(close, y)
    ks = np.array([bucket(doy(t)) for t in s.index])
    last = pd.Series(s.values, index=ks).groupby(level=0).last()
    lvl = last.reindex(range(WEEKS)).ffill().fillna(base).to_numpy(dtype=float)
    prev = np.concatenate([[base], lvl[:-1]])
    w = np.log(lvl / prev)
    traded = np.isin(np.arange(WEEKS), last.index)
    if upto is not None:
        w[upto + 1:] = np.nan
    return w, traded


def _monthly(close: pd.Series, y: int, upto: int | None = None) -> np.ndarray:
    s = _year(close, y)
    base = _base(close, y)
    last = s.groupby(s.index.month).last()
    lvl = last.reindex(range(1, 13)).ffill().fillna(base).to_numpy(dtype=float)
    prev = np.concatenate([[base], lvl[:-1]])
    m = lvl / prev - 1
    if upto is not None:
        m[upto:] = np.nan
    return m


def _path(close: pd.Series, y: int, bench: pd.Series | None = None) -> list[list[float]]:
    """[[slot, cumulative %], …] for every session in year y."""
    s = _year(close, y)
    rel = s / _base(close, y)
    if bench is not None:
        b = bench.reindex(s.index, method="ffill")
        rel = rel / (b / _base(bench, y))
    return [[doy(t), round(float(v - 1) * 100, 2)] for t, v in rel.items()
            if np.isfinite(v)]


def _avg_path(paths: list[list[list[float]]]) -> list[float | None]:
    """The average year, slot by slot, each year held flat across its closed days."""
    if not paths:
        return []
    grid = []
    for p in paths:
        ser = pd.Series({int(a): b for a, b in p}).reindex(range(366)).ffill().fillna(0.0)
        grid.append(ser.to_numpy())
    return [round(float(v), 2) for v in np.mean(grid, axis=0)]


# ── windows ──────────────────────────────────────────────────────────────────
def allowed_misses(n: int) -> int:
    """
    How many years may go the other way. None with five years or fewer —
    "two out of three" is a coin with a bad day — and one in five beyond.
    """
    return 0 if n <= 5 else n // 5


def _scores(W: np.ndarray):
    """
    For each window length, every start's score in each direction.

    Two tests, in order. A window QUALIFIES only if the years agree — all of
    them, or all but `allowed_misses` — and even the weakest qualifying year
    moved at least MIN_MOVE. A qualifying window is then RANKED by a z-score:
    the average move over the years, divided by what this stock's own weekly
    volatility would produce over that many weeks by chance. Ranking by raw
    size instead favours the longest windows of the most volatile stocks,
    which is noise with a big number on it.

    Yields (L, R, z_up, z_down, weak_up, weak_down): R is the per-year log
    return of each start; z is 0 for a window that does not qualify; weak is
    the weakest qualifying year's log move (positive = in the direction).
    """
    n = W.shape[0]
    m = allowed_misses(n)
    sigma = float(np.std(W, ddof=1)) or 1e-9
    floor = math.log1p(MIN_MOVE)
    C = np.concatenate([np.zeros((n, 1)), np.cumsum(W, axis=1)], axis=1)
    for L in range(MIN_WINDOW, MAX_WINDOW + 1):
        R = C[:, L:] - C[:, :-L]                    # (n, WEEKS-L+1)
        srt = np.sort(R, axis=0)
        weak_up, weak_dn = srt[m], -srt[n - 1 - m]
        z = R.mean(axis=0) / (sigma * math.sqrt(L) / math.sqrt(n))
        z_up = np.where(weak_up >= floor, np.maximum(z, 0), 0.0)
        z_dn = np.where(weak_dn >= floor, np.maximum(-z, 0), 0.0)
        yield L, R, z_up, z_dn, weak_up, weak_dn


def best_score(W: np.ndarray) -> float:
    return max(max(up.max(), dn.max()) for _, _, up, dn, _, _ in _scores(W))


def null_best(W: np.ndarray, shuffles: int = SHUFFLES, seed: int = 924) -> np.ndarray:
    """The best score found on `shuffles` calendars with each year rotated at random."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    cols = np.arange(WEEKS)
    out = np.empty(shuffles)
    for i in range(shuffles):
        off = rng.integers(0, WEEKS, n)
        idx = (cols[None, :] - off[:, None]) % WEEKS
        out[i] = best_score(W[np.arange(n)[:, None], idx])
    return out


def find_windows(W: np.ndarray, per_side: int = 3) -> list[dict]:
    """The strongest non-overlapping qualifying up and down windows, best first."""
    cands = []
    for L, R, z_up, z_dn, weak_up, weak_dn in _scores(W):
        for a in range(R.shape[1]):
            if z_up[a] > 0:
                cands.append((float(z_up[a]), a, L, "up", R[:, a], float(weak_up[a])))
            if z_dn[a] > 0:
                cands.append((float(z_dn[a]), a, L, "down", R[:, a], -float(weak_dn[a])))
    cands.sort(key=lambda c: -c[0])
    taken = np.zeros(WEEKS, dtype=bool)
    picked = {"up": 0, "down": 0}
    out = []
    for score, a, L, d, r, weak in cands:
        if picked[d] >= per_side or taken[a:a + L].any():
            continue
        taken[a:a + L] = True
        picked[d] += 1
        out.append({"start": a, "weeks": L, "dir": d, "score": score, "r": r,
                    "weak": weak})
    return out


# ── forward look and this year ───────────────────────────────────────────────
def _on_or_before(s: pd.Series, d: date):
    hit = s[s.index <= pd.Timestamp(d)]
    return (hit.index[-1], float(hit.iloc[-1])) if len(hit) else (None, None)


def _same_day(y: int, today: date) -> date:
    try:
        return today.replace(year=y)
    except ValueError:                                   # Feb 29
        return date(y, 2, 28)


def ahead(close: pd.Series, full: list[int], today: date, days: int,
          bench: pd.Series | None = None) -> dict:
    """What the next `days` calendar days did in each past year, from this date."""
    last_bar = close.index[-1]
    per = {}
    for y in full:
        a = _same_day(y, today)
        b = a + timedelta(days=days)
        if pd.Timestamp(b) > last_bar:
            continue
        _, pa = _on_or_before(close, a)
        _, pb = _on_or_before(close, b)
        if pa is None or pb is None:
            continue
        r = pb / pa
        if bench is not None:
            _, ba = _on_or_before(bench, a)
            _, bb = _on_or_before(bench, b)
            if ba is None or bb is None:
                continue
            r = r / (bb / ba)
        per[y] = round((r - 1) * 100, 2)
    vals = list(per.values())
    end = today + timedelta(days=days)
    return {"days": days, "from": f"{today:%m-%d}", "to": f"{end:%m-%d}",
            "per_year": {str(k): v for k, v in per.items()},
            "mean": round(float(np.mean(vals)), 2) if vals else None,
            "ups": sum(v > 0 for v in vals), "n": len(vals)}


def ytd(close: pd.Series, full: list[int], cur: int, bench: pd.Series | None = None) -> dict | None:
    """This year so far against every past year at the same date."""
    if _base(close, cur) is None or not len(_year(close, cur)):
        return None
    last = close.index[-1].date()

    def at(y: int):
        d = _same_day(y, last)
        _, p = _on_or_before(_year(close, y), d)
        if p is None:
            return None
        r = p / _base(close, y)
        if bench is not None:
            _, bp = _on_or_before(_year(bench, y), d)
            bb = _base(bench, y)
            if bp is None or bb is None:
                return None
            r = r / (bp / bb)
        return round((r - 1) * 100, 2)

    now = at(cur)
    past = {str(y): at(y) for y in full}
    vals = [v for v in past.values() if v is not None]
    rank = 1 + sum(v > now for v in vals) if now is not None else None
    return {"as_of": f"{last:%m-%d}", "now": now, "past": past,
            "mean": round(float(np.mean(vals)), 2) if vals else None,
            "rank": rank, "of": len(vals) + 1}


def spring(close: pd.Series, yrs: list[int], bench: pd.Series | None = None) -> list[dict]:
    """10 trading days into 春节 and 10 out, per year."""
    out = []
    for y in yrs:
        sf = SPRING_FESTIVAL.get(y)
        if sf is None:
            continue
        pre = close[close.index < pd.Timestamp(sf)]
        post = close[close.index > pd.Timestamp(sf)]
        if len(pre) <= SPRING_DAYS:
            continue
        p0, p1 = float(pre.iloc[-SPRING_DAYS - 1]), float(pre.iloc[-1])
        p2 = float(post.iloc[SPRING_DAYS - 1]) if len(post) >= SPRING_DAYS else None
        before, after = p1 / p0, (p2 / p1 if p2 else None)
        if bench is not None:
            def lvl(ts):
                _, v = _on_or_before(bench, ts.date())
                return v
            b0, b1 = lvl(pre.index[-SPRING_DAYS - 1]), lvl(pre.index[-1])
            if not b0 or not b1:
                continue
            before /= b1 / b0
            if p2:
                b2 = lvl(post.index[SPRING_DAYS - 1])
                after = after / (b2 / b1) if b2 else None
        out.append({"year": y, "date": f"{sf:%m-%d}",
                    "before": round((before - 1) * 100, 2),
                    "after": round((after - 1) * 100, 2) if after else None})
    return out


# ── words ────────────────────────────────────────────────────────────────────
def _pct(v: float | None, nd: int = 1) -> str:
    return "—" if v is None else f"{v:+.{nd}f}%"


def verdict(p: float) -> tuple[str, str]:
    if p < 0.05:
        return "strong", "超出随机：打乱日历后极少能找到同样强的窗口"
    if p < 0.20:
        return "weak", "有一点证据，但不排除巧合"
    return "none", "和随机差不多：打乱日历后也经常能找到同样强的窗口"


def conclusions(*, n: int, full: list[int], windows: list[dict], sig: dict,
                nxt: list[dict], this_year: dict | None, spr: list[dict],
                market: str, basis: str, labels: list[str]) -> list[str]:
    who = "相对大盘的超额收益" if basis == "excess" else "股价"
    lines = [f"比较 {'、'.join(map(str, full))} 共 {n} 个完整年度的{who}。"]

    kind, words = verdict(sig["p_best"])
    if windows:
        w = windows[0]
        lines.append(
            f"最一致的时段是 {labels[w['start']].split('~')[0]}~"
            f"{labels[w['start'] + w['weeks'] - 1].split('~')[1]}（{w['weeks']} 周）："
            f"{w['hits']}/{n} 年{'上涨' if w['dir'] == 'up' else '下跌'}，"
            f"最弱的一年也{'涨' if w['dir'] == 'up' else '跌'}了 {abs(w['weakest']):.1f}%，"
            f"平均 {_pct(w['mean'])}。")
    else:
        lines.append("没有找到任何一个 2-8 周的时段在这些年里方向一致且幅度超过 1%。")
    share = ("不到 1%" if sig["p_best"] < 0.01 else f"{sig['p_best'] * 100:.0f}%")
    lines.append(
        f"统计检验：把每年的周收益随机错位 {sig['shuffles']} 次，"
        f"{share} 的情况下也能找到同样强的窗口 —— {words}。")
    if n <= 3 and kind != "strong":
        lines.append("只有 3 年数据时，季节性很难与运气区分；切换到 5 年或 10 年看是否还成立。")

    for h in nxt:
        if h["n"]:
            lines.append(
                f"接下来 {h['days']} 天（{h['from']}→{h['to']}），过去 {h['n']} 年里"
                f" {h['ups']} 年上涨，平均 {_pct(h['mean'])}。")
    if this_year and this_year["now"] is not None:
        lines.append(
            f"今年至今（截至 {this_year['as_of']}）{_pct(this_year['now'])}，"
            f"往年同期平均 {_pct(this_year['mean'])}，在 {this_year['of']} 年中排第 {this_year['rank']}。")
    if market == "CN" and spr:
        ups = sum(1 for s in spr if s["before"] > 0)
        mean = float(np.mean([s["before"] for s in spr]))
        lines.append(f"春节前 {SPRING_DAYS} 个交易日（{spr[0]['year']}–{spr[-1]['year']}）："
                     f"{len(spr)} 年里 {ups} 年上涨，平均 {_pct(mean)}。")
    if market == "CN" and any(y <= REGIME_YEAR for y in full):
        post = [y for y in full if y > REGIME_YEAR]
        if windows and post:
            w = windows[0]
            idx = [full.index(y) for y in post]
            agree = all((w["r_list"][i] > 0) == (w["dir"] == "up") for i in idx)
            lines.append(
                f"924 之后的年份（{'、'.join(map(str, post))}）在这个时段方向也一致。"
                if agree else
                f"924 之后的年份（{'、'.join(map(str, post))}）在这个时段方向相反 —— "
                f"这个时段的表现可能属于 924 前的旧市场。")
        elif not post:
            lines.append("这些年份都在 924 之前或当年，规律可能已不适用于现在的市场。")
    return lines


# ── one basis ────────────────────────────────────────────────────────────────
def _analyse(close: pd.Series, full: list[int], cur: int, today: date, market: str,
             basis: str, bench: pd.Series | None, labels: list[str],
             shuffles: int) -> dict:
    """Everything for one basis. For `excess`, `bench` divides the stock out."""
    have_cur = _base(close, cur) is not None and len(_year(close, cur)) > 0
    cur_week = bucket(doy(close.index[-1])) if have_cur else None
    cur_month = close.index[-1].month if have_cur else None

    def W(y, upto=None):
        w, t = _weekly(close, y, upto)
        if bench is not None:
            wb, _ = _weekly(bench, y, upto)
            w = w - wb
        return w, t

    weekly, traded, monthly, daily = {}, {}, {}, {}
    for y in full + ([cur] if have_cur else []):
        upto = cur_week if y == cur else None
        w, t = W(y, upto)
        weekly[str(y)] = [None if not np.isfinite(v) else round(math.expm1(v) * 100, 2) for v in w]
        traded[str(y)] = [bool(x) for x in t]
        m = _monthly(close, y, cur_month if y == cur else None)
        if bench is not None:
            m = (1 + m) / (1 + _monthly(bench, y, cur_month if y == cur else None)) - 1
        monthly[str(y)] = [None if not np.isfinite(v) else round(float(v) * 100, 2) for v in m]
        daily[str(y)] = _path(close, y, bench)

    M = np.array([W(y)[0] for y in full])                 # (n, 52) full years
    n = len(full)
    obs = best_score(M)
    null = null_best(M, shuffles)
    p_best = float((1 + np.sum(null >= obs)) / (shuffles + 1))

    windows = []
    for w in find_windows(M):
        r = np.expm1(w["r"]) * 100
        hits = int(np.sum(r > 0)) if w["dir"] == "up" else int(np.sum(r < 0))
        windows.append({
            "start": w["start"], "weeks": w["weeks"], "dir": w["dir"],
            "from": labels[w["start"]].split("~")[0],
            "to": labels[w["start"] + w["weeks"] - 1].split("~")[1],
            "per_year": {str(y): round(float(v), 2) for y, v in zip(full, r)},
            "r_list": [float(v) for v in r],
            "mean": round(float(np.mean(r)), 2),
            "weakest": round(math.expm1(w["weak"]) * 100, 2),
            "z": round(w["score"], 2),
            "hits": hits,
            "p": round(float((1 + np.sum(null >= w["score"])) / (shuffles + 1)), 3),
        })

    Wpct = np.expm1(M) * 100
    week_stats = [{"mean": round(float(np.mean(Wpct[:, k])), 2),
                   "median": round(float(np.median(Wpct[:, k])), 2),
                   "ups": int(np.sum(Wpct[:, k] > 0)), "n": n} for k in range(WEEKS)]
    Mm = np.array([monthly[str(y)] for y in full], dtype=float)
    month_stats = [{"mean": round(float(np.nanmean(Mm[:, i])), 2),
                    "ups": int(np.sum(Mm[:, i] > 0)), "n": n} for i in range(12)]

    nxt = [ahead(close, full, today, h, bench) for h in HORIZONS]
    this_year = ytd(close, full, cur, bench)
    spr = spring(close, full + ([cur] if have_cur else []), bench) if market == "CN" else []
    sig = {"p_best": round(p_best, 3), "shuffles": shuffles,
           "observed_z": round(obs, 2),
           "null_median_z": round(float(np.median(null)), 2),
           "verdict": verdict(p_best)[0]}

    lines = conclusions(n=n, full=full, windows=windows, sig=sig, nxt=nxt,
                        this_year=this_year, spr=spr, market=market,
                        basis=basis, labels=labels)
    for w in windows:
        w.pop("r_list")
    return {
        "daily": daily, "avg_daily": _avg_path([daily[str(y)] for y in full]),
        "weekly": weekly, "traded": traded, "week_stats": week_stats,
        "monthly": monthly, "month_stats": month_stats,
        "windows": windows, "significance": sig,
        "ahead": nxt, "ytd": this_year, "spring": spr,
        "conclusions": lines,
    }


def build(close: pd.Series, *, years: int, today: date, market: str,
          bench: pd.Series | None = None, shuffles: int = SHUFFLES) -> dict:
    """
    The whole seasonality payload. `close` is adjusted daily closes; `bench`
    the market index on the same basis, or None to skip the excess view.
    """
    if years not in YEAR_CHOICES:
        raise ValueError(f"years must be one of {YEAR_CHOICES}")
    close = close.dropna().sort_index()
    cur = today.year
    wanted = list(range(cur - years, cur))
    full = [y for y in wanted if _complete(close, y)]
    skipped = [y for y in wanted if y not in full]
    if len(full) < 2:
        raise LookupError(f"完整年度不足：只有 {len(full)} 年数据可比（上市时间太短？）")

    labels = week_labels()
    out = {
        "years": [{"year": y, "full": True,
                   "regime": (None if market != "CN" else
                              "pre" if y < REGIME_YEAR else
                              "pivot" if y == REGIME_YEAR else "post")} for y in full],
        "current_year": cur, "skipped": skipped,
        "as_of": f"{close.index[-1]:%Y-%m-%d}",
        "week_labels": labels,
        "spring_dates": ({str(y): f"{SPRING_FESTIVAL[y]:%m-%d}"
                          for y in full + [cur] if y in SPRING_FESTIVAL}
                         if market == "CN" else {}),
        "abs": _analyse(close, full, cur, today, market, "abs", None, labels, shuffles),
        "excess": None,
    }
    if bench is not None and not bench.dropna().empty:
        b = bench.dropna().sort_index()
        if all(_complete(b, y) for y in full):
            out["excess"] = _analyse(close, full, cur, today, market, "excess",
                                     b, labels, shuffles)
    return out
