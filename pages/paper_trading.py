"""
Blind Replay — A-share paper trading game.

A real stock over a real date range, with the identity withheld, played one
session at a time. The point is to build reflexes against genuine price action
and genuine noise, without the anchoring that comes from knowing the name.

The whole game lives in one @st.fragment, so pressing Pass/Buy/Sell reruns only
this block — the page does not reload on every simulated day.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import auth_manager
import data_manager
import game_news
import paper_trading as pt

auth_manager.require_login()

st.title("🎲 盲盘复盘 · Blind Replay")
st.caption(
    "A real A-share, a real stretch of history, identity hidden. Decide from "
    "the chart alone: buy, sell, or pass — one session at a time. Prices, "
    "T+1 settlement, lot sizes and trading costs are all real."
)

S = st.session_state


def _start_game(cash: float, play_bars: int):
    with st.spinner("Picking a stock…"):
        pick = pt.pick_random_stock(data_manager, play_bars=play_bars)
    if not pick.get("ok"):
        S["bp_error"] = pick.get("reason", "could not pick a stock")
        S.pop("bp_game", None)
        return
    S["bp_error"] = None
    S["bp_pick"] = pick
    S["bp_game"] = pt.new_game(pick, cash=cash, play_bars=play_bars)
    # Cleared so a new stock never shows the previous one's themes or articles.
    S.pop("bp_risk", None)
    S.pop("bp_news_cache", None)


# ── Setup ─────────────────────────────────────────────────────────────────────
if "bp_game" not in S:
    st.subheader("开始新局 · New game")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        _cash = st.number_input("起始资金 Starting cash (¥)", 10_000, 10_000_000,
                                100_000, 10_000, key="bp_cash")
    with c2:
        _bars = st.select_slider("交易日数 Sessions to play",
                                 options=[30, 60, 90, 120, 180], value=120,
                                 key="bp_bars")
    with c3:
        st.write("")
        st.write("")
        if st.button("🎲 抽取股票 Deal me a stock", type="primary",
                     use_container_width=True):
            _start_game(float(_cash), int(_bars))
            st.rerun()
    if S.get("bp_error"):
        st.error(S["bp_error"])
    st.stop()


@st.fragment
def game():
    g: pt.GameState = S["bp_game"]
    pick = S["bp_pick"]
    df: pd.DataFrame = pick["data"]

    today = pt.today_str(g, df)
    row = pt.today_row(g, df)
    price = float(row["Close"])
    perf = pt.performance(g, df)
    over = g.finished()

    # ── Status bar ────────────────────────────────────────────────────────
    m = st.columns(6)
    m[0].metric("交易日 Session", f"{perf['day']} / {perf['play_bars']}")
    m[1].metric("日期 Date", today)
    m[2].metric("收盘 Close", f"¥{price:,.2f}",
                delta=f"{(price / float(df.iloc[g.cur_idx - 1]['Close']) - 1) * 100:+.2f}%"
                if g.cur_idx > 0 else None, delta_color="inverse")
    m[3].metric("现金 Cash", f"¥{perf['cash']:,.0f}")
    m[4].metric("持股 Shares", f"{perf['shares']:,}",
                delta=f"成本 ¥{perf['avg_cost']:.2f}" if perf["shares"] else None,
                delta_color="off")
    m[5].metric("总资产 Equity", f"¥{perf['equity']:,.0f}",
                delta=f"{perf['pnl_pct']:+.2f}%", delta_color="inverse")

    # ── Chart ─────────────────────────────────────────────────────────────
    vis = pt.visible_frame(g, df)
    dates = [d.strftime("%Y-%m-%d") for d in vis.index]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.52, 0.13, 0.19, 0.16],
                        subplot_titles=("价格 Price", "成交量 Volume",
                                        "MACD", "RSI"))
    fig.add_trace(go.Candlestick(
        x=dates, open=vis["Open"], high=vis["High"], low=vis["Low"],
        close=vis["Close"], name="K线",
        increasing=dict(line=dict(color="#dc2626")),   # 红涨绿跌
        decreasing=dict(line=dict(color="#22c55e"))), row=1, col=1)
    for col, colour, width in (("MA5", "#ec4899", 1.2), ("MA10", "#14b8a6", 1.2),
                               ("MA20", "#fbbf24", 1.5), ("MA60", "#f97316", 2)):
        fig.add_trace(go.Scatter(x=dates, y=vis[col], name=col,
                                 line=dict(color=colour, width=width)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=vis["EMA5"], name="EMA5",
                             line=dict(color="#a855f7", width=1.5, dash="dash")),
                  row=1, col=1)
    for col in ("BB_Upper", "BB_Lower"):
        fig.add_trace(go.Scatter(x=dates, y=vis[col], name=col, showlegend=False,
                                 line=dict(color="rgba(148,163,184,.55)", width=1)),
                      row=1, col=1)

    # Your own fills, so the replay doubles as a review of your decisions.
    for side, sym, colour in (("buy", "triangle-up", "#dc2626"),
                              ("sell", "triangle-down", "#15803d")):
        xs = [t["date"] for t in g.trades if t["side"] == side and t["date"] in dates]
        ys = [t["price"] for t in g.trades if t["side"] == side and t["date"] in dates]
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", name=("买入" if side == "buy" else "卖出"),
                marker=dict(symbol=sym, size=13, color=colour,
                            line=dict(width=1, color="white"))), row=1, col=1)

    fig.add_trace(go.Bar(x=dates, y=vis["Volume"], name="Volume", showlegend=False,
                         marker_color=["#dc2626" if c >= o else "#22c55e"
                                       for c, o in zip(vis["Close"], vis["Open"])]),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=vis["Vol_MA20"], name="Vol MA20",
                             line=dict(color="#64748b", width=1)), row=2, col=1)

    fig.add_trace(go.Bar(x=dates, y=vis["MACD_Hist"], name="Hist", showlegend=False,
                         marker_color=["#dc2626" if v >= 0 else "#22c55e"
                                       for v in vis["MACD_Hist"].fillna(0)]),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=vis["MACD"], name="MACD",
                             line=dict(color="#2a78d6", width=1.4)), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=vis["MACD_Signal"], name="Signal",
                             line=dict(color="#eb6834", width=1.2)), row=3, col=1)

    fig.add_trace(go.Scatter(x=dates, y=vis["RSI"], name="RSI",
                             line=dict(color="#7c3aed", width=1.5)), row=4, col=1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line_dash="dot", line_width=1,
                      line_color="rgba(148,163,184,.8)", row=4, col=1)

    fig.update_layout(height=760, template="plotly_white",
                      xaxis_rangeslider_visible=False, hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                                  font=dict(size=10)))
    fig.update_yaxes(range=[0, 100], row=4, col=1)
    _step = max(1, len(dates) // 10)
    fig.update_xaxes(type="category", showticklabels=False)
    fig.update_xaxes(type="category", tickangle=-45, showticklabels=True,
                     tickmode="array", tickvals=dates[::_step], row=4, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── Actions ───────────────────────────────────────────────────────────
    if over:
        st.success(f"🏁 本局结束 · Game over after {perf['play_bars']} sessions.")
    else:
        st.markdown("#### 今日决策 · Today's decision")
        a1, a2, a3, a4 = st.columns([1.2, 1.2, 1, 1])

        max_buy = int((g.cash // (price * 100 * (1 + pt.COMMISSION_RATE))) * 100)
        with a1:
            qty_b = st.number_input("买入股数 Buy (×100)", 0, max(max_buy, 0),
                                    min(100, max_buy), 100, key="bp_qty_b")
            if st.button(f"🔴 买入 Buy @ ¥{price:.2f}", use_container_width=True,
                         disabled=max_buy < 100):
                r = pt.buy(g, int(qty_b), price, today)
                if r["ok"]:
                    pt.advance(g)
                    st.rerun(scope="fragment")
                else:
                    st.error(r["reason"])

        sellable = g.sellable(today)
        with a2:
            qty_s = st.number_input("卖出股数 Sell", 0, max(sellable, 0),
                                    min(100, sellable), 100, key="bp_qty_s")
            if st.button(f"🟢 卖出 Sell @ ¥{price:.2f}", use_container_width=True,
                         disabled=sellable < 1):
                r = pt.sell(g, int(qty_s), price, today)
                if r["ok"]:
                    pt.advance(g)
                    st.rerun(scope="fragment")
                else:
                    st.error(r["reason"])
            if g.shares and not sellable:
                st.caption("T+1：今日买入明日才可卖")

        with a3:
            st.write("")
            if st.button("⏭️ 跳过 Pass", use_container_width=True, type="primary"):
                pt.advance(g)
                st.rerun(scope="fragment")
        with a4:
            st.write("")
            if st.button("↩️ 撤销 Revert", use_container_width=True,
                         disabled=not g.undo):
                pt.revert(g)
                st.rerun(scope="fragment")
            st.caption(f"{len(g.undo)} step(s) back available")

    # ── News ──────────────────────────────────────────────────────────────
    with st.expander("📰 新闻 · Macro news for this date", expanded=False):
        st.caption(
            "Macro only, and dated — nothing published after the session you "
            "are playing. Company-specific stories are excluded by design, and "
            "the stock's name is filtered out of results, since either would "
            "give the answer away."
        )
        if "bp_risk" not in S:
            if st.button("🤖 Identify macro risk themes for this stock"):
                with st.spinner("Asking DeepSeek for macro themes…"):
                    S["bp_risk"] = game_news.derive_risk_factors(
                        pick["ticker"], pick["name"], pick.get("industry", ""))
                st.rerun(scope="fragment")
            st.caption("One DeepSeek call per game; themes are reused for every day.")
        else:
            risk = S["bp_risk"]
            if risk.get("ok"):
                st.markdown("**风险主题 Macro themes:** " +
                            " · ".join(f"`{t}`" for t in risk["themes"]))
            else:
                st.warning(f"Theme extraction failed: {risk.get('reason','')[:120]}")

            src = st.radio(
                "Feed",
                ["主题 Themes", "全球宏观 Macro", "🌍 世界大事 World events"],
                horizontal=True, key="bp_news_src",
                help="Themes and Macro come from The Guardian (dated, archive "
                     "back years). World events is Wikipedia's curated daily "
                     "log — keyless and the deepest archive of the three.",
            )
            query = (game_news.build_query(risk.get("search_terms"))
                     if src.startswith("主题") else game_news.MACRO_QUERY)

            cache = S.setdefault("bp_news_cache", {})
            ckey = f"{today}|{src}|{query}"
            if st.button("📥 Load news for " + today):
                with st.spinner("Fetching…"):
                    if src.startswith("🌍"):
                        cache[ckey] = game_news.fetch_world_events(today)
                    else:
                        cache[ckey] = game_news.fetch_guardian(
                            query, today, window_days=2,
                            redact=[pick["name"], pick["ticker"]])
                st.rerun(scope="fragment")
            if not src.startswith("🌍"):
                st.caption(f"query: `{query}`")

            got = cache.get(ckey)
            if got is None:
                st.caption("Not loaded for this date yet.")
            elif got.get("throttled"):
                st.warning(
                    "Source is rate-limiting right now — this is the feed being "
                    "busy, **not** an absence of news that day. The Guardian's "
                    "shared `test` key is capped; a free key from "
                    "open-platform.theguardian.com/access raises it to 5,000 "
                    "calls/day — add it as `GUARDIAN_API_KEY` in secrets."
                )
            elif not got.get("ok"):
                st.info(f"News unavailable: {got.get('reason')}")
            elif not got["articles"]:
                st.caption("No matching articles in GDELT for this window.")
            else:
                for a in got["articles"]:
                    st.markdown(
                        f"- [{a['title']}]({a['url']})  \n"
                        f"  <span style='font-size:11px;color:#64748b'>"
                        f"{a['domain']} · {a['seendate']}</span>",
                        unsafe_allow_html=True)

    # ── Trade log & reveal ────────────────────────────────────────────────
    lc, rc = st.columns([2, 1])
    with lc:
        with st.expander(f"📒 交易记录 Trade log ({perf['n_trades']})", expanded=False):
            if g.trades:
                st.dataframe(pd.DataFrame(g.trades)[
                    ["day", "date", "side", "shares", "price", "value", "fees"]],
                    hide_index=True, use_container_width=True)
                st.caption(f"手续费合计 fees paid: ¥{perf['fees_paid']:,.2f}")
            else:
                st.caption("No trades yet.")
    with rc:
        st.metric("买入持有 Buy & hold", f"{perf['buy_hold_pct']:+.2f}%",
                  help="What simply holding from session 1 would have returned — "
                       "the benchmark your decisions have to beat.")
        st.metric("你的收益 Your return", f"{perf['pnl_pct']:+.2f}%",
                  delta=f"{perf['pnl_pct'] - perf['buy_hold_pct']:+.2f}% vs B&H",
                  delta_color="inverse")

    st.markdown("---")
    b1, b2 = st.columns([1, 1])
    with b1:
        if g.revealed or over:
            st.info(f"**{pick['name']} ({pick['ticker']})** · "
                    f"{pick.get('industry') or '—'}")
        elif st.button("👁️ 揭晓股票 Reveal the stock (ends the blind run)"):
            g.revealed = True
            st.rerun(scope="fragment")
    with b2:
        if st.button("🔄 换一只 · New stock, restart", use_container_width=True):
            _start_game(g.initial_cash, g.play_bars)
            st.rerun()


game()
