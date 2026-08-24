"""
whatif_advisor.py — 尾盘推演: you are looking at the What-If bar as it prints.

The What-If Simulator already answers "what would the indicators read tomorrow
if the bar closed HERE". This module answers the question after that one: given
that reading, what is the trade — decided at 尾盘 of the simulated session,
with the bar essentially formed and the close a minute away.

Why that framing matters
------------------------
A decision taken at the close is the only one the simulated bar can actually
support. Anything earlier and the bar isn't formed yet, so its own indicators
don't exist; anything later and you are trading a session the simulator has
said nothing about. Pinning the moment also pins what is knowable: the
simulated close, the indicators it produces, and nothing whatsoever about the
session after it.

Division of labour (same as trade_review)
-----------------------------------------
Everything numeric here is computed deterministically — including, crucially,
the *state changes* between today and the simulated bar. Those crossings are
the whole point of the exercise, and a model asked to eyeball them from two
columns of numbers will get some of them wrong with total confidence. So the
crossings are computed and handed over as findings; the model's job is to
weigh them against each other and argue a stance, not to detect them.

This is a technical-scenario exercise, not investment advice, and the prompt
says so to the model as well.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIMELINE_BARS = 20          # recent real sessions given as context
RANGE_WINDOW = 120          # window for "where in its range" position


# ─────────────────────────── small formatters ────────────────────────────
def _n(v, fmt="{:.2f}"):
    """Format, tolerating None/NaN uniformly so the prompt never shows 'nan'."""
    if v is None:
        return "—"
    try:
        if pd.isna(v):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return fmt.format(float(v))


def _f(v):
    """Coerce to float or None — the sim dict mixes numpy scalars and None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _ma_stack(close, mas: dict) -> str:
    """Which averages price is above, in one token."""
    out = []
    for w in (5, 10, 20, 60):
        v = _f(mas.get(w))
        if v is not None and close is not None:
            out.append(f"{'>' if close >= v else '<'}MA{w}")
    return " ".join(out) if out else "—"


def _bar_line(r) -> str:
    """One real session, compressed to a single line of the context timeline."""
    def g(k, fmt="{:.1f}"):
        return _n(r.get(k), fmt)
    prev = _f(r.get("prev_close"))
    pct = ((r["Close"] / prev - 1) * 100) if prev else None
    amp = ((r["High"] - r["Low"]) / prev * 100) if prev else None
    pat = str(r.get("ADX_Pattern") or "")
    return " · ".join([
        str(r.name.date()),
        f"¥{r['Close']:.2f}",
        f"{_n(pct, '{:+.2f}')}%",
        f"振幅{_n(amp, '{:.2f}')}%",
        _ma_stack(_f(r["Close"]), {w: r.get(f"MA{w}") for w in (5, 10, 20, 60)}),
        f"MACD{g('MACD', '{:+.3f}')}/{g('MACD_Signal', '{:+.3f}')}",
        f"RSI{g('RSI_14')}",
        f"ADX{g('ADX')}" + (f"({pat})" if pat and pat != "Neutral" else ""),
        f"±DI {g('DI_Plus')}/{g('DI_Minus')}",
        f"量Z{g('Volume_ZScore', '{:+.1f}')}",
    ])


# ───────────────────────── crossing detection ────────────────────────────
def _cross(name, today_a, today_b, tmr_a, tmr_b, up_msg, down_msg):
    """A crosses B between today and the simulated bar, or it doesn't."""
    if None in (today_a, today_b, tmr_a, tmr_b):
        return None
    was, now = today_a - today_b, tmr_a - tmr_b
    if was <= 0 < now:
        return {"what": name, "dir": "up", "detail": up_msg}
    if was >= 0 > now:
        return {"what": name, "dir": "down", "detail": down_msg}
    return None


def _band(v, edges, labels):
    """Which named band a value falls in."""
    if v is None:
        return None
    for e, l in zip(edges, labels):
        if v < e:
            return l
    return labels[-1]


_ADX_EDGES = [20, 25, 40]
_ADX_LABELS = ["无趋势", "趋势萌芽", "趋势确立", "趋势过热"]
_RSI_EDGES = [30, 50, 70]
_RSI_LABELS = ["超卖", "偏弱", "偏强", "超买"]


def _crossings(df: pd.DataFrame, sim: dict) -> list[dict]:
    """
    Every discrete state change between today's close and the simulated close.

    Computed rather than inferred: these are the findings the model is asked
    to weigh, and eyeballing sign changes off a table is exactly where a
    language model quietly invents a golden cross that didn't happen.
    """
    last = df.iloc[-1]
    out = []

    c_t, c_m = _f(sim.get("close_today")), _f(sim.get("close_tomorrow"))
    ma_t = {w: _f((sim.get("ma_today") or {}).get(f"MA{w}")) for w in (5, 10, 20, 60)}
    ma_m = {w: _f((sim.get("ma_tomorrow") or {}).get(f"MA{w}")) for w in (5, 10, 20, 60)}

    # price vs each moving average
    for w in (5, 10, 20, 60):
        h = _cross(f"价格 vs MA{w}", c_t, ma_t.get(w), c_m, ma_m.get(w),
                   f"收盘站上 MA{w} ({_n(ma_m.get(w))})",
                   f"收盘跌破 MA{w} ({_n(ma_m.get(w))})")
        if h:
            out.append(h)

    # MA5 vs MA20 — the stack itself turning over
    h = _cross("MA5 vs MA20", ma_t.get(5), ma_t.get(20), ma_m.get(5), ma_m.get(20),
               "MA5 上穿 MA20（金叉）", "MA5 下穿 MA20（死叉）")
    if h:
        out.append(h)

    # MACD line vs signal, and histogram sign
    h = _cross("MACD", _f(sim.get("macd_today")), _f(sim.get("macd_signal_today")),
               _f(sim.get("macd_tomorrow")), _f(sim.get("macd_signal_tomorrow")),
               "MACD 上穿信号线（金叉）", "MACD 下穿信号线（死叉）")
    if h:
        out.append(h)
    ht, hm = _f(sim.get("macd_hist_today")), _f(sim.get("macd_hist_tomorrow"))
    if ht is not None and hm is not None and (ht < 0 <= hm or ht >= 0 > hm):
        out.append({"what": "MACD 柱", "dir": "up" if hm > 0 else "down",
                    "detail": f"柱状体由{'负转正' if hm > 0 else '正转负'}"
                              f"（{ht:+.3f} → {hm:+.3f}）"})

    # RSI zones and the adaptive percentile rails
    r_t, r_m = _f(sim.get("rsi_today")), _f(sim.get("rsi_tomorrow"))
    z_t, z_m = _band(r_t, _RSI_EDGES, _RSI_LABELS), _band(r_m, _RSI_EDGES, _RSI_LABELS)
    if z_t and z_m and z_t != z_m:
        out.append({"what": "RSI 区间", "dir": "up" if r_m > r_t else "down",
                    "detail": f"{z_t} → {z_m}（{r_t:.1f} → {r_m:.1f}）"})
    for rail, nm in ((sim.get("rsi_p10"), "底部10%线"), (sim.get("rsi_p90"), "顶部90%线")):
        h = _cross(f"RSI vs {nm}", r_t, _f(rail), r_m, _f(rail),
                   f"RSI 上穿{nm} ({_n(rail, '{:.1f}')})",
                   f"RSI 下穿{nm} ({_n(rail, '{:.1f}')})")
        if h:
            out.append(h)

    # ADX strength band and ±DI
    a_t, a_m = _f(sim.get("adx_today")), _f(sim.get("adx_tomorrow"))
    b_t, b_m = _band(a_t, _ADX_EDGES, _ADX_LABELS), _band(a_m, _ADX_EDGES, _ADX_LABELS)
    if b_t and b_m and b_t != b_m:
        out.append({"what": "ADX 强度", "dir": "up" if a_m > a_t else "down",
                    "detail": f"{b_t} → {b_m}（{a_t:.1f} → {a_m:.1f}）"})
    h = _cross("±DI", _f(sim.get("di_plus_today")), _f(sim.get("di_minus_today")),
               _f(sim.get("di_plus_tomorrow")), _f(sim.get("di_minus_tomorrow")),
               "DI+ 上穿 DI−（多头掌控）", "DI− 上穿 DI+（空头掌控）")
    if h:
        out.append(h)

    # Bollinger breaches
    bt, bm = (sim.get("bb_today") or {}), (sim.get("bb_tomorrow") or {})
    for edge, nm, up, dn in (("upper", "上轨", "收盘突破布林上轨", "收盘回落至上轨内"),
                             ("lower", "下轨", "收盘回到布林下轨上方", "收盘跌破布林下轨")):
        h = _cross(f"价格 vs 布林{nm}", c_t, _f(bt.get(edge)), c_m, _f(bm.get(edge)), up, dn)
        if h:
            out.append(h)

    # OBV momentum sign
    ot, om = _f(sim.get("obv_mom_today")), _f(sim.get("obv_mom_tomorrow"))
    if ot is not None and om is not None and (ot < 0 <= om or ot >= 0 > om):
        out.append({"what": "OBV 动能", "dir": "up" if om > 0 else "down",
                    "detail": f"由{'负转正' if om > 0 else '正转负'}（{ot:+.2f} → {om:+.2f}）"})

    # Volume regime, measured the same way the chart's Z-score is
    v_m = _f(sim.get("volume_tomorrow"))
    mu, sd = _f(last.get("Vol_Mean_100d")), _f(last.get("Vol_Std_100d"))
    vz_t, vz_m = _f(last.get("Volume_ZScore")), None
    if v_m is not None and mu is not None and sd:
        vz_m = (v_m - mu) / sd
    if vz_t is not None and vz_m is not None:
        for thr, nm in ((2.0, "放量(Z>2)"), (-1.0, "地量(Z<-1)")):
            if (vz_t < thr <= vz_m) or (vz_t >= thr > vz_m):
                out.append({"what": "量能", "dir": "up" if vz_m > vz_t else "down",
                            "detail": f"跨过{nm}阈值（Z {vz_t:+.1f} → {vz_m:+.1f}）"})
    return out


# ───────────────── the last REAL bar, shaped like a sim ──────────────────
def sim_from_real_bar(df: pd.DataFrame) -> dict | None:
    """
    Describe the last REAL session in the shape simulate_next_day_indicators()
    returns, so the whole pipeline downstream reads either kind of bar.

    "Today" becomes the second-to-last session and "tomorrow" the last one —
    i.e. the question shifts from "if this bar printed" to "this bar just
    printed". Values are read straight out of the computed columns rather than
    re-simulated: the simulator has to estimate ADX and the MA/BOLL recursions
    forward, but for a bar that actually exists the engine has already done
    the exact arithmetic, and re-deriving it would only introduce error.
    """
    if df is None or len(df) < 30:
        return None
    a, b = df.iloc[-2], df.iloc[-1]          # a = today, b = the bar to read

    def v(row, col):
        return _f(row.get(col)) if col in df.columns else None

    prev = _f(b.get("prev_close")) or _f(a.get("Close"))
    vol20_b = _f(df["Volume"].rolling(20, min_periods=1).mean().iloc[-1])
    vol20_a = _f(df["Volume"].iloc[:-1].rolling(20, min_periods=1).mean().iloc[-1])

    # OBV动能: net signed volume over 20 sessions / average daily volume —
    # the same ratio the simulator forms, just with both ends already real.
    def obv_mom(idx, avg):
        if "OBV" not in df.columns or len(df) < 21 + abs(idx) - 1 or not avg:
            return None
        o_now, o_then = _f(df["OBV"].iloc[idx]), _f(df["OBV"].iloc[idx - 20])
        return (o_now - o_then) / avg if (o_now is not None and o_then is not None) else None

    # Signals come from the engine's own detector columns, so the box agrees
    # with what the chart is drawing rather than re-deriving its own opinion.
    def flag(col, row=b):
        return bool(row.get(col)) if col in df.columns else False

    mac_a, sig_a = v(a, "MACD"), v(a, "MACD_Signal")
    mac_b, sig_b = v(b, "MACD"), v(b, "MACD_Signal")
    crossed_up = (mac_a is not None and sig_a is not None and mac_b is not None
                  and sig_b is not None and mac_a < sig_a and mac_b > sig_b)
    crossed_dn = (mac_a is not None and sig_a is not None and mac_b is not None
                  and sig_b is not None and mac_a > sig_a and mac_b < sig_b)

    return {
        "input_price_change_pct": ((_f(b["Close"]) / prev - 1) * 100) if prev else None,
        "input_volume": _f(b["Volume"]),
        "close_today": _f(a["Close"]), "close_tomorrow": _f(b["Close"]),

        "macd_today": mac_a, "macd_tomorrow": mac_b,
        "macd_signal_today": sig_a, "macd_signal_tomorrow": sig_b,
        "macd_hist_today": v(a, "MACD_Hist"), "macd_hist_tomorrow": v(b, "MACD_Hist"),

        "rsi_today": v(a, "RSI_14"), "rsi_tomorrow": v(b, "RSI_14"),
        "rsi_p10": v(b, "RSI_P10"), "rsi_p90": v(b, "RSI_P90"),

        "adx_today": v(a, "ADX"), "adx_tomorrow": v(b, "ADX"),
        "adx_pattern": str(b.get("ADX_Pattern") or ""),
        "di_plus_today": v(a, "DI_Plus"), "di_plus_tomorrow": v(b, "DI_Plus"),
        "di_minus_today": v(a, "DI_Minus"), "di_minus_tomorrow": v(b, "DI_Minus"),

        "open_tomorrow": _f(b["Open"]), "high_tomorrow": _f(b["High"]),
        "low_tomorrow": _f(b["Low"]), "ohl_supplied": True,

        "ma_today": {f"MA{n}": v(a, f"MA{n}") for n in (5, 10, 20, 50, 60, 200)},
        "ma_tomorrow": {f"MA{n}": v(b, f"MA{n}") for n in (5, 10, 20, 50, 60, 200)},
        "ema5_today": v(a, "EMA5"), "ema5_tomorrow": v(b, "EMA5"),
        "bb_today": {"upper": v(a, "BB_Upper"), "lower": v(a, "BB_Lower")},
        "bb_tomorrow": {"upper": v(b, "BB_Upper"), "lower": v(b, "BB_Lower")},

        "obv_mom_today": obv_mom(-2, vol20_a),
        "obv_mom_tomorrow": obv_mom(-1, vol20_b),

        "volume_today": _f(a["Volume"]), "volume_tomorrow": _f(b["Volume"]),
        # The 10-day average BEFORE this bar, so "volume vs 10d" measures the
        # bar against its own run-up rather than against a window it is in.
        "volume_10d_avg": _f(df["Volume"].iloc[:-1].rolling(10).mean().iloc[-1]),

        "signals": {
            "MACD_Bottoming": flag("MACD_Bottoming"),
            "MACD_Bullish_Cross": crossed_up,
            "MACD_Bearish_Cross": crossed_dn,
            "RSI_Bottoming": flag("RSI_Bottoming"),
            "RSI_Peaking": flag("RSI_Peaking"),
            "ADX_Pattern": str(b.get("ADX_Pattern") or ""),
            "DI_Screaming_Buy": flag("DI_Screaming_Buy"),
        },
    }


# ─────────────────────────────── the brief ───────────────────────────────
def build_brief(df: pd.DataFrame, sim: dict, *,
                ticker: str | None = None, name: str | None = None,
                ad_today: dict | None = None, ad_tomorrow: dict | None = None,
                ad_window: int = 20, mode: str = "simulated",
                bar_date: str | None = None,
                timeline_bars: int = TIMELINE_BARS) -> dict:
    """
    Package the tape leading in, the bar under the microscope, and every state
    change between them into one deterministic brief.

    df   : output of run_single_stock_analysis, ending at the bar BEFORE the
           one being read — that bar is the "today" everything is measured from
    sim  : output of simulate_next_day_indicators, or of sim_from_real_bar()
    mode : "simulated" for a What-If bar, "actual" for the last real session.
           Only changes how the bar is described; the maths is identical.
    ad_* : optional accumulation_signals.summarise() results for today and for
           the bar being read, so the 吸筹/出货 read travels with the rest.
    """
    if df is None or len(df) < 30 or not sim:
        return {"ok": False, "reason": "not enough history or no simulation"}

    last = df.iloc[-1]
    c_t, c_m = _f(sim.get("close_today")), _f(sim.get("close_tomorrow"))
    o, hi, lo = (_f(sim.get("open_tomorrow")), _f(sim.get("high_tomorrow")),
                 _f(sim.get("low_tomorrow")))

    # Shape of the simulated candle — where the close sits inside its own range
    # is often the whole message (a long upper shadow on a green close says
    # something that a bare +2% does not).
    rng = (hi - lo) if (hi is not None and lo is not None) else None
    close_pos = ((c_m - lo) / rng) if (rng and rng > 0) else None
    body = abs(c_m - o) if (c_m is not None and o is not None) else None
    up_shadow = (hi - max(c_m, o)) if None not in (hi, c_m, o) else None
    dn_shadow = (min(c_m, o) - lo) if None not in (lo, c_m, o) else None
    gap = ((o / c_t - 1) * 100) if (o and c_t) else None

    win = df.tail(RANGE_WINDOW)
    rhi, rlo = float(win["High"].max()), float(win["Low"].min())
    # The simulated bar can itself set a new extreme — include it, or the
    # position pins at a flat 100% when the bar has actually broken out.
    rhi_m, rlo_m = max(rhi, hi or rhi), min(rlo, lo or rlo)
    pos_t = (c_t - rlo) / (rhi - rlo) if rhi > rlo else None
    pos_m = (c_m - rlo_m) / (rhi_m - rlo_m) if rhi_m > rlo_m else None

    v_m = _f(sim.get("volume_tomorrow"))
    v10 = _f(sim.get("volume_10d_avg"))
    mu, sd = _f(last.get("Vol_Mean_100d")), _f(last.get("Vol_Std_100d"))

    tl = [_bar_line(r) for _, r in df.tail(timeline_bars).iterrows()]

    def _ad_pack(s):
        if not s:
            return None
        return {"verdict": s.get("verdict"), "n_acc": s.get("n_acc"),
                "n_dist": s.get("n_dist"),
                "acc": [x["cn"] for x in s.get("acc_live", [])],
                "dist": [x["cn"] for x in s.get("dist_live", [])]}

    ad = {"window": ad_window,
          "today": _ad_pack(ad_today), "tomorrow": _ad_pack(ad_tomorrow)}
    if ad_today and ad_tomorrow:
        now = {x["key"] for x in ad_today.get("acc_live", []) + ad_today.get("dist_live", [])}
        nxt = {x["key"] for x in ad_tomorrow.get("acc_live", []) + ad_tomorrow.get("dist_live", [])}
        lk = {x["key"]: x["cn"] for x in
              ad_tomorrow.get("acc_live", []) + ad_tomorrow.get("dist_live", [])}
        ok = {x["key"]: x["cn"] for x in
              ad_today.get("acc_live", []) + ad_today.get("dist_live", [])}
        ad["turned_on"] = [lk[k] for k in sorted(nxt - now)]
        ad["turned_off"] = [ok[k] for k in sorted(now - nxt)]

    return {
        "ok": True,
        "stock": {"ticker": ticker, "name": name},
        "mode": mode,
        "asof": {"last_real_session": str(df.index[-1].date()),
                 "sim_session": (bar_date or
                                 str((df.index[-1] + pd.Timedelta(days=1)).date())),
                 "bars_of_history": len(df)},
        "timeline": tl,
        "today": {
            "close": c_t,
            "ma": {w: _f((sim.get("ma_today") or {}).get(f"MA{w}")) for w in (5, 10, 20, 60)},
            "ma_stack": _ma_stack(c_t, {w: (sim.get("ma_today") or {}).get(f"MA{w}")
                                        for w in (5, 10, 20, 60)}),
            "ema5": _f(sim.get("ema5_today")),
            "macd": _f(sim.get("macd_today")),
            "macd_signal": _f(sim.get("macd_signal_today")),
            "macd_hist": _f(sim.get("macd_hist_today")),
            "rsi": _f(sim.get("rsi_today")),
            "adx": _f(sim.get("adx_today")), "adx_pattern": sim.get("adx_pattern"),
            "di_plus": _f(sim.get("di_plus_today")),
            "di_minus": _f(sim.get("di_minus_today")),
            "bb": {k: _f((sim.get("bb_today") or {}).get(k)) for k in ("upper", "lower")},
            "obv_mom": _f(sim.get("obv_mom_today")),
            "volume": _f(sim.get("volume_today")),
            "volume_z": _f(last.get("Volume_ZScore")),
            "range_pos": pos_t,
            "atr_pct": _f(last.get("NATR")),
            "regime": str(last.get("Market_Regime") or ""),
        },
        "simulated_bar": {
            "close": c_m, "open": o, "high": hi, "low": lo,
            "pct_change": _f(sim.get("input_price_change_pct")),
            "amplitude_pct": (rng / c_t * 100) if (rng and c_t) else None,
            "gap_pct": gap,
            "close_position_in_bar": close_pos,
            "body": body, "upper_shadow": up_shadow, "lower_shadow": dn_shadow,
            "ohl_supplied": bool(sim.get("ohl_supplied")),
            "volume": v_m,
            "volume_vs_10d": (v_m / v10) if (v_m and v10) else None,
            "volume_z": ((v_m - mu) / sd) if (v_m is not None and mu is not None and sd) else None,
            "ma": {w: _f((sim.get("ma_tomorrow") or {}).get(f"MA{w}")) for w in (5, 10, 20, 60)},
            "ma_stack": _ma_stack(c_m, {w: (sim.get("ma_tomorrow") or {}).get(f"MA{w}")
                                        for w in (5, 10, 20, 60)}),
            "ema5": _f(sim.get("ema5_tomorrow")),
            "macd": _f(sim.get("macd_tomorrow")),
            "macd_signal": _f(sim.get("macd_signal_tomorrow")),
            "macd_hist": _f(sim.get("macd_hist_tomorrow")),
            "rsi": _f(sim.get("rsi_tomorrow")),
            "adx": _f(sim.get("adx_tomorrow")),
            "di_plus": _f(sim.get("di_plus_tomorrow")),
            "di_minus": _f(sim.get("di_minus_tomorrow")),
            "bb": {k: _f((sim.get("bb_tomorrow") or {}).get(k)) for k in ("upper", "lower")},
            "obv_mom": _f(sim.get("obv_mom_tomorrow")),
            "range_pos": pos_m,
            "signals": sim.get("signals") or {},
        },
        "crossings": _crossings(df, sim),
        "accumulation_distribution": ad,
        "range": {"window": RANGE_WINDOW, "high": rhi, "low": rlo},
    }


_INTRO = {
    "simulated": (
        '你是一位资深的A股技术分析教练。学员在"明日指标模拟器"里假设了一根K线，\n'
        '现在的场景是：**这根K线已经基本走完，还有几分钟收盘（尾盘）**，\n'
        '盘面就是数据里那根K线的样子。'),
    "actual": (
        '你是一位资深的A股技术分析教练。下面是该股**最新一个真实交易日**的盘面，\n'
        '现在的场景是：**这根K线已经基本走完，还有几分钟收盘（尾盘）**。\n'
        '注意：这是真实成交出来的K线，不是假设——O/H/L/收盘/成交量都是实际数据。'),
}

_PROMPT = """\
{intro}

你的任务：把这根K线读懂，然后说清楚**在这个尾盘时点该怎么处理**，
以及**明天开盘后按什么条件行动**。

你会拿到三样东西：
1. 最近若干个真实交易日的逐日行情与指标；
2. 今日（最后一个真实交易日）与这根模拟K线的完整指标对照；
3. **crossings**——已经由程序精确算出的状态变化（金叉/死叉/突破/区间切换等）。

关于 crossings 的硬性要求：
- 这些是**算好的事实**，直接采信、直接引用。
- **不要自己去比大小推断有没有交叉**。如果 crossings 里没有列出某个交叉，
  就说明它没有发生，不许说它发生了。
- 反过来，crossings 里列出的每一条都必须在你的解读里被考虑到
  （可以判断它不重要，但不能装作没看见）。

解读要点：
- K线形态本身很重要：收盘位于当日振幅的百分位（0%=收在最低，100%=收在最高）
  以及上下影线的长短。同样的涨幅，收在最高点和留一根长上影线是完全不同的两件事。
- 量价配合：成交量相对10日均量的倍数、量能Z值。放量突破和缩量突破意义不同。
- 位置：区间位置是价格在近{rw}日区间中的百分位。同样的信号出现在区间底部
  和顶部，含义相反。
- 吸筹/出货信号的新增与消失，说明主力行为可能在切换。

严格要求：
- **只能使用给定的数据**。不许编造价格、指标值、基本面、消息面或资金流数据。
- 明确区分"这根K线说明了什么"和"如果它成立我会怎么做"。
- 不要预测明天的具体涨跌幅，也不要给出精确目标价。可以给**条件触发位**
  （例如"跌破X则……"），因为那是纪律，不是预测。
- 这是技术面推演训练，不是投资建议。caveats 里必须点明这一点。
- **全部用中文回答**。
- 只输出原始JSON（以 {{ 开始，以 }} 结束），不要markdown代码块。

JSON结构：
{{
  "headline": "一句话概括这根K线的性质",
  "bar_read": "对这根K线本身的解读：形态、量价、位置",
  "key_changes": ["最重要的状态变化，逐条，引用crossings里的事实"],
  "bull_case": "看多的理由，基于数据",
  "bear_case": "看空的理由，基于数据",
  "stance": {{
    "call": "买入 / 加仓 / 持有 / 减仓 / 清仓 / 观望 之一",
    "conviction": "高 / 中 / 低",
    "why": "为什么是这个结论，必须引用具体指标",
    "if_holding": "已经持仓的话，尾盘该怎么办",
    "if_flat": "空仓的话，尾盘该怎么办"
  }},
  "levels": {{
    "confirm": "什么价位/条件出现算是确认",
    "invalidate": "什么价位/条件出现说明这个判断错了",
    "note": "补充说明"
  }},
  "next_session_plan": "明天开盘后按什么条件行动，写成if-then",
  "what_would_change_my_mind": ["会推翻当前判断的信号，最多3条"],
  "caveats": ["局限与风险提示，必须包含'这是技术推演不是投资建议'"]
}}
"""


def explain(brief: dict) -> dict:
    """One DeepSeek call turning the brief into a Chinese scenario read."""
    import ai_client

    if not brief.get("ok"):
        raise RuntimeError(brief.get("reason", "brief unavailable"))

    t, s, a = brief["today"], brief["simulated_bar"], brief["asof"]
    st_ = brief.get("stock") or {}
    who = " ".join(x for x in (st_.get("name"), st_.get("ticker")) if x) or "该股"

    def row(label, key, fmt="{:.2f}"):
        vt, vm = t.get(key), s.get(key)
        d = f"（{vm - vt:+.2f}）" if (vt is not None and vm is not None) else ""
        return f"  {label:<10} {_n(vt, fmt):>10} → {_n(vm, fmt):>10}{d}"

    cross = brief.get("crossings") or []
    cross_txt = ("\n".join(f"  [{'↑' if c['dir'] == 'up' else '↓'}] {c['what']}：{c['detail']}"
                           for c in cross)
                 if cross else "  （无——这根K线不触发任何状态变化）")

    ad = brief.get("accumulation_distribution") or {}
    adt, adm = ad.get("today"), ad.get("tomorrow")
    if adt and adm:
        ad_txt = (f"  今日：{adt['verdict']}（吸筹{adt['n_acc']}/出货{adt['n_dist']}）"
                  f" → 这根K线之后：{adm['verdict']}"
                  f"（吸筹{adm['n_acc']}/出货{adm['n_dist']}）\n"
                  f"  新增触发：{'、'.join(ad.get('turned_on') or []) or '无'}\n"
                  f"  不再触发：{'、'.join(ad.get('turned_off') or []) or '无'}")
    elif adt:
        ad_txt = f"  今日：{adt['verdict']}（吸筹{adt['n_acc']}/出货{adt['n_dist']}）"
    else:
        ad_txt = "  （未计算）"

    sig_on = [k for k, v in (s.get("signals") or {}).items() if v is True]
    actual = brief.get("mode") == "actual"
    ohl_note = ("O/H/L 为真实成交数据" if actual
                else "O/H/L 由用户明确指定" if s["ohl_supplied"]
                else "O/H/L 为估算值（用户只给了收盘涨跌幅），形态解读需留余地")
    bar_label = "这根真实K线" if actual else "模拟的这根K线"
    when = (f"上一交易日：{a['last_real_session']} · 本次解读的交易日：{a['sim_session']}"
            if actual else
            f"最后一个真实交易日：{a['last_real_session']} · 模拟的这根K线：{a['sim_session']}")

    user = f"""\
标的：{who}（只做技术面推演）
{when}

【{bar_label} —— 尾盘时点，盘面已成型】
  开 {_n(s['open'])} 高 {_n(s['high'])} 低 {_n(s['low'])} 收 {_n(s['close'])}
  涨跌 {_n(s['pct_change'], '{:+.2f}')}% · 振幅 {_n(s['amplitude_pct'])}% · 跳空 {_n(s['gap_pct'], '{:+.2f}')}%
  收盘位于当日振幅的 {_n(s['close_position_in_bar'], '{:.0%}')} 处
  （实体 {_n(s['body'])} · 上影 {_n(s['upper_shadow'])} · 下影 {_n(s['lower_shadow'])}）
  成交量 {_n(s['volume'], '{:,.0f}')} · 10日均量的 {_n(s['volume_vs_10d'], '{:.2f}')} 倍 · 量能Z {_n(s['volume_z'], '{:+.2f}')}
  {ohl_note}

【指标对照：今日收盘 → 这根K线收盘】
{row('收盘价', 'close')}
{row('EMA5', 'ema5')}
  {'MA排列':<10} {t['ma_stack']:>10} → {s['ma_stack']}
{row('MACD', 'macd', '{:+.4f}')}
{row('MACD信号', 'macd_signal', '{:+.4f}')}
{row('MACD柱', 'macd_hist', '{:+.4f}')}
{row('RSI(14)', 'rsi', '{:.1f}')}
{row('ADX', 'adx', '{:.1f}')}
{row('DI+', 'di_plus', '{:.1f}')}
{row('DI-', 'di_minus', '{:.1f}')}
{row('OBV动能', 'obv_mom', '{:+.2f}')}
  {'区间位置':<10} {_n(t['range_pos'], '{:.0%}'):>10} → {_n(s['range_pos'], '{:.0%}')}（近{brief['range']['window']}日 {_n(brief['range']['low'])}~{_n(brief['range']['high'])}）
  今日ADX形态：{t['adx_pattern'] or '—'} · 波动状态：{t['regime'] or '—'} · NATR {_n(t['atr_pct'])}%
  布林上轨 {_n(t['bb']['upper'])} → {_n(s['bb']['upper'])} · 下轨 {_n(t['bb']['lower'])} → {_n(s['bb']['lower'])}

【程序算出的状态变化 crossings —— 事实，直接引用】
{cross_txt}

【模拟器触发的信号】
  {'、'.join(sig_on) if sig_on else '无'}

【吸筹 / 出货（{ad.get('window')}日窗口）】
{ad_txt}

【最近 {len(brief['timeline'])} 个真实交易日】
{chr(10).join('  ' + x for x in brief['timeline'])}
"""

    # Far more bounded than the 复盘: one bar to reason about instead of a
    # whole game, so a near-flat budget is honest here. The floor is still
    # generous because the reasoning trace shares this same allowance.
    rows = len(brief.get("timeline", []))
    budget = min(20000, max(12000, 9000 + rows * 120))
    timeout = min(240, max(120, 90 + rows * 2))

    return ai_client.call_json(
        _PROMPT.format(rw=RANGE_WINDOW,
                       intro=_INTRO.get(brief.get("mode", "simulated"),
                                        _INTRO["simulated"])), user,
        max_tokens=budget, temperature=0.4,
        reasoning_effort="low", timeout=timeout)
