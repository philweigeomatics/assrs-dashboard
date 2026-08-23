"""
trade_review.py — 复盘: what the chart was saying that you didn't read.

Scope of the review window, and why
-----------------------------------
It starts FIVE sessions before the first purchase, because the setup that
justified (or didn't justify) the entry was already forming before the entry,
and a critique that begins at the buy cannot see what invited it.

It ends at TODAY in game time — the session being played when the review is
asked for. That bound is what makes reviewing mid-game safe: everything in the
window has already been lived through, so nothing here reveals a bar the player
has not seen. It also means every "what happened after" figure is measured only
up to today, never over a fixed forward window that could run past it.

Everything in between is included, flat days and passes alike. A stretch of
doing nothing while the chart was shouting is itself a decision, and it is
usually where the lesson is.

Division of labour: the metrics below are computed deterministically and the
model is asked to interpret them. Left to invent its own numbers a model
produces fluent, confident, wrong critique.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRE_ENTRY_BARS = 5
MAX_DETAIL_ROWS = 70        # beyond this, quiet stretches are compressed


def _fmt_ma(r) -> str:
    """MA stack in one token: which averages price is above."""
    out = []
    for w in (5, 10, 20, 60):
        v = r.get(f"MA{w}")
        if pd.notna(v):
            out.append(f"{'>' if r['Close'] >= v else '<'}MA{w}")
    return " ".join(out) if out else "—"


def _day_line(r, day_no, action, pos) -> str:
    def g(k, f="{:.1f}"):
        v = r.get(k)
        return "—" if v is None or pd.isna(v) else f.format(v)
    scen = str(r.get("MACD_Scenario") or "")
    pat = str(r.get("ADX_Pattern") or "")
    bits = [
        f"D{day_no:>3}", str(r.name.date()), f"¥{r['Close']:.2f}",
        f"{r.get('PctChg', float('nan')):+.2f}%" if pd.notna(r.get("PctChg")) else "—",
        f"[{action}]", f"持{pos}",
        _fmt_ma(r),
        f"MACD{g('MACD','{:+.3f}')}" + (f"/{scen}" if scen else ""),
        f"RSI{g('RSI_14')}",
        f"ADX{g('ADX')}" + (f"/{pat}" if pat and pat != "Neutral" else ""),
        f"VolZ{g('Volume_ZScore','{:+.1f}')}",
        f"OBV动能{g('OBV_Momentum','{:+.2f}')}",
    ]
    return " | ".join(bits)


def build_review(game, df: pd.DataFrame, pre_bars: int = PRE_ENTRY_BARS) -> dict:
    """
    Assemble everything the critique needs, bounded at the current session.

    Returns {"ok", "timeline", "trades", "metrics", "window", ...}.
    """
    if not getattr(game, "trades", None):
        return {"ok": False, "reason": "还没有任何交易 · no trades yet"}

    cur = game.cur_idx
    today = pd.Timestamp(df.index[cur])

    first_dt = pd.Timestamp(min(t["date"] for t in game.trades))
    first_pos = int(df.index.get_indexer([first_dt], method="nearest")[0])
    lo = max(0, first_pos - pre_bars)
    win = df.iloc[lo:cur + 1].copy()
    win["PctChg"] = win["Close"].pct_change() * 100

    by_date: dict[str, list] = {}
    for t in game.trades:
        by_date.setdefault(t["date"], []).append(t)

    # ── Day-by-day, including the days nothing happened ──
    pos, lines, compressed = 0, [], 0
    quiet_run: list = []

    def _flush():
        nonlocal quiet_run, compressed
        if not quiet_run:
            return
        if len(quiet_run) >= 4 and len(win) > MAX_DETAIL_ROWS:
            a, b = quiet_run[0], quiet_run[-1]
            chg = (b[1]["Close"] / a[1]["Close"] - 1) * 100
            lines.append(
                f"D{a[0]:>3}-D{b[0]:<3} {a[1].name.date()}→{b[1].name.date()} "
                f"[无操作 {len(quiet_run)}日] 持{a[2]} | 价格{chg:+.1f}% | "
                f"RSI{a[1].get('RSI_14', float('nan')):.0f}→{b[1].get('RSI_14', float('nan')):.0f} | "
                f"ADX{a[1].get('ADX', float('nan')):.0f}→{b[1].get('ADX', float('nan')):.0f}")
            compressed += len(quiet_run)
        else:
            for dno, rr, pp in quiet_run:
                lines.append(_day_line(rr, dno, "—", pp))
        quiet_run = []

    for i, (idx, r) in enumerate(win.iterrows()):
        dno = lo + i - game.start_idx + 1
        ds = pd.Timestamp(idx).strftime("%Y-%m-%d")
        acts = by_date.get(ds, [])
        if acts:
            _flush()
            for t in acts:
                pos += t["shares"] if t["side"] == "buy" else -t["shares"]
            label = "/".join(
                f"{'买入' if t['side'] == 'buy' else '卖出'}{t['shares']}@{t['price']:.2f}"
                for t in acts)
            lines.append(_day_line(r, dno, label, pos))
        else:
            quiet_run.append((dno, r, pos))
    _flush()

    # ── Outcome of each trade, measured ONLY up to today ──
    closes = win["Close"]
    tdetail, forgone, mae = [], [], []
    for t in game.trades:
        d = pd.Timestamp(t["date"])
        after = closes[closes.index > d]
        row = dict(t)
        if len(after):
            hi, loo = float(after.max()), float(after.min())
            row["max_after"] = hi
            row["min_after"] = loo
            if t["side"] == "sell":
                # 卖飞 measured against what actually followed, up to today.
                f = (hi - t["price"]) / t["price"] * 100
                row["forgone_pct"] = f
                forgone.append(f)
            else:
                m = (loo - t["price"]) / t["price"] * 100
                row["mae_pct"] = m
                mae.append(m)
                row["mfe_pct"] = (hi - t["price"]) / t["price"] * 100
        tdetail.append(row)

    price = float(df.iloc[cur]["Close"])
    eq = game.equity(price)
    first_close = float(df.iloc[game.start_idx]["Close"])
    n_sell = sum(1 for t in game.trades if t["side"] == "sell")
    span = max(cur - game.start_idx + 1, 1)

    metrics = {
        "sessions_played": span,
        "n_trades": len(game.trades),
        "n_buys": len(game.trades) - n_sell,
        "n_sells": n_sell,
        "trades_per_20d": round(len(game.trades) / span * 20, 1),
        "fees_paid": round(sum(t["fees"] for t in game.trades), 2),
        "fees_pct_of_capital": round(sum(t["fees"] for t in game.trades)
                                     / game.initial_cash * 100, 3),
        "equity": round(eq, 2),
        "return_pct": round((eq / game.initial_cash - 1) * 100, 2),
        "buy_hold_pct": round((price / first_close - 1) * 100, 2),
        "vs_buy_hold": round((eq / game.initial_cash - 1) * 100
                             - (price / first_close - 1) * 100, 2),
        "shares_now": game.shares,
        # The 卖飞 number: how much the average sale left on the table by today.
        "avg_forgone_after_sell_pct": round(float(np.mean(forgone)), 2) if forgone else None,
        "worst_forgone_pct": round(float(np.max(forgone)), 2) if forgone else None,
        "avg_mae_after_buy_pct": round(float(np.mean(mae)), 2) if mae else None,
        "days_flat": int(sum(1 for ln in lines if "持0" in ln)),
        "compressed_days": compressed,
    }

    return {
        "ok": True,
        "timeline": lines,
        "trades": tdetail,
        "metrics": metrics,
        "window": {"from": str(win.index[0].date()), "to": str(today.date()),
                   "bars": len(win), "pre_entry_bars": pre_bars},
        "today": {
            "date": str(today.date()), "close": price,
            "rsi": float(df.iloc[cur].get("RSI_14", float("nan"))),
            "adx": float(df.iloc[cur].get("ADX", float("nan"))),
            "adx_pattern": str(df.iloc[cur].get("ADX_Pattern", "")),
            "macd_scenario": str(df.iloc[cur].get("MACD_Scenario", "")),
            "vol_z": float(df.iloc[cur].get("Volume_ZScore", float("nan"))),
        },
    }


_PROMPT = """\
你是一位资深的A股交易教练，正在为学员做一次复盘。

学员在一个"盲盘"训练里操作了一只**隐藏名称**的真实股票。你会看到：
从首次买入前5个交易日开始、一直到"今天"（他提问的这一天）的逐日行情与指标，
他的每一笔买卖，以及**没有操作的日子**。

你的任务：指出他**看漏了什么信号**，而不是复述他做了什么。

重点：
1. 找出最关键的一两个错误，不要罗列一堆。最常见的是"卖飞"——过早卖出、
   之后价格继续上涨。数据里的 avg_forgone_after_sell_pct 就是衡量这个的：
   卖出后到今天为止最高价比卖出价高多少。
2. 对每个错误，**明确指出当天盘面上有什么信号**（均线排列、MACD、RSI、
   ADX形态、量能Z值、OBV动能）本可以提示他不该那样做。必须引用给你的数据，
   不要编造。
3. 无操作的日子同样重要——趋势明确却空仓观望，和乱操作一样是错误。
4. 最后只给**一条**最该改的习惯，具体可执行。

严格要求：
- 只能使用提供的数据。不得假设任何"今天"之后的价格走势——你也看不到。
- 不要猜测这是哪只股票，也不要提任何公司名。
- 这是交易训练的复盘，不是投资建议。
- **全部用中文回答**。
- 只输出原始JSON（以 { 开始，以 } 结束），不要markdown代码块。

JSON结构：
{
  "headline": "一句话点出核心问题",
  "biggest_mistake": {
    "what": "做错了什么",
    "when": "哪一天/哪几天",
    "signals_missed": ["当天盘面上本应看到的信号，逐条列出"],
    "cost": "这个错误的代价，用数据说明"
  },
  "secondary_issues": ["次要问题，最多2条"],
  "what_you_did_well": ["做对的地方，最多2条，没有就留空"],
  "inaction_review": "对没有操作的那些日子的评价",
  "one_habit_to_change": "只给一条最该改的习惯"
}
"""


def explain(review: dict) -> dict:
    """One DeepSeek call turning the assembled review into a Chinese critique."""
    import ai_client

    m = review["metrics"]
    w = review["window"]
    t = review["today"]

    trades_txt = "\n".join(
        f"  D{x['day']} {x['date']} {'买入' if x['side']=='buy' else '卖出'} "
        f"{x['shares']}股 @¥{x['price']:.2f}"
        + (f" · 卖出后至今最高 ¥{x['max_after']:.2f}（少赚 {x['forgone_pct']:+.1f}%）"
           if x.get("forgone_pct") is not None else "")
        + (f" · 买入后至今最低 ¥{x['min_after']:.2f}（浮亏 {x['mae_pct']:+.1f}%）"
           if x.get("mae_pct") is not None else "")
        for x in review["trades"])

    user = f"""\
复盘区间：{w['from']} → {w['to']}（共{w['bars']}个交易日，其中首次买入前{w['pre_entry_bars']}日为铺垫）
提问当天：{t['date']} 收盘 ¥{t['close']:.2f} · RSI {t['rsi']:.1f} · ADX {t['adx']:.1f} \
{t['adx_pattern']} · MACD {t['macd_scenario']} · 量能Z {t['vol_z']:+.2f}

【逐日行情与操作】（[—]表示当天没有操作）
{chr(10).join(review['timeline'])}

【交易明细】
{trades_txt}

【统计】
- 已交易 {m['sessions_played']} 个交易日，共 {m['n_trades']} 笔（买{m['n_buys']}/卖{m['n_sells']}），
  折合每20日 {m['trades_per_20d']} 笔
- 手续费 ¥{m['fees_paid']}（占初始资金 {m['fees_pct_of_capital']}%）
- 当前收益 {m['return_pct']:+.2f}% vs 买入持有 {m['buy_hold_pct']:+.2f}%（差 {m['vs_buy_hold']:+.2f}%）
- 卖出后平均少赚（卖飞幅度）：{m['avg_forgone_after_sell_pct'] if m['avg_forgone_after_sell_pct'] is not None else '无卖出'}%
  最严重一次：{m['worst_forgone_pct'] if m['worst_forgone_pct'] is not None else '—'}%
- 买入后平均最大浮亏：{m['avg_mae_after_buy_pct'] if m['avg_mae_after_buy_pct'] is not None else '—'}%
- 当前持股 {m['shares_now']} 股
"""
    return ai_client.call_json(_PROMPT, user, max_tokens=9000,
                               temperature=0.4, reasoning_effort="low")
