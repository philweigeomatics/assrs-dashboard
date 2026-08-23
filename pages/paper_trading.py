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

import accumulation_signals as acsig
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
        _bars = st.select_slider("要交易多少天 Sessions to play",
                                 options=[30, 60, 90, 120, 180], value=120,
                                 key="bp_bars",
                                 help="How many decisions you get. You begin at "
                                      "session 1 with ~60 sessions of chart "
                                      "history already visible behind you.")
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
    # "1 / 180" is the session you are ON, not a date range: the game starts at
    # session 1 and you advance forward, so 180 is how many decisions remain,
    # not how much history is drawn. The chart still shows ~60 sessions of past
    # data behind day 1 — that history is context, not part of the count.
    m = st.columns(6)
    m[0].metric("已进行 Session", f"{perf['day']} / {perf['play_bars']}",
                help="Sessions played so far out of the number you chose. You "
                     "start at 1 and advance one session per decision; the "
                     "chart behind day 1 is context you did not trade.")
    m[1].metric("当前日期 Date", today)
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

    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.028,
                        row_heights=[0.34, 0.13, 0.15, 0.13, 0.13, 0.12],
                        subplot_titles=("价格 Price", "成交量 & OBV", "MACD",
                                        "RSI", "ADX Trend Analysis",
                                        "Z-Score · Price + Volume"))
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

    # ── 2 · Volume & OBV (OBV on its own scale, as on the TA page) ──
    fig.add_trace(go.Bar(x=dates, y=vis["Volume"], name="Volume", showlegend=False,
                         marker_color=["#dc2626" if c >= o else "#22c55e"
                                       for c, o in zip(vis["Close"], vis["Open"])]),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=vis["Vol_MA20"], name="Vol MA20",
                             line=dict(color="#64748b", width=1)), row=2, col=1)
    _obv = vis["OBV"]
    _vmax = float(vis["Volume"].max() or 1)
    _rng = float(_obv.max() - _obv.min()) or 1.0
    fig.add_trace(go.Scatter(
        x=dates, y=(_obv - _obv.min()) / _rng * _vmax, name="OBV (scaled)",
        line=dict(color="#7c3aed", width=1.4),
        hovertemplate="OBV %{customdata:,.0f}<extra></extra>",
        customdata=_obv), row=2, col=1)

    # ── 3 · MACD with turn markers ──
    fig.add_trace(go.Bar(x=dates, y=vis["MACD_Hist"], name="Hist", showlegend=False,
                         marker_color=["#dc2626" if v >= 0 else "#22c55e"
                                       for v in vis["MACD_Hist"].fillna(0)]),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=vis["MACD"], name="MACD",
                             line=dict(color="#2a78d6", width=1.4)), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=vis["MACD_Signal"], name="Signal",
                             line=dict(color="#eb6834", width=1.2)), row=3, col=1)
    for flag, lab, sym, colour in (("MACD_Bottoming", "MACD Bottoming", "triangle-up", "#dc2626"),
                                   ("MACD_Peaking", "MACD Peaking", "triangle-down", "#15803d"),
                                   ("MACD_GoldenCross", "金叉 Golden", "star", "#dc2626"),
                                   ("MACD_DeadCross", "死叉 Dead", "x", "#15803d")):
        msk = vis[flag].fillna(False).values.astype(bool)
        if msk.any():
            fig.add_trace(go.Scatter(
                x=[d for d, m in zip(dates, msk) if m],
                y=vis["MACD"][msk], mode="markers", name=lab,
                marker=dict(symbol=sym, size=9, color=colour)), row=3, col=1)

    # ── 4 · RSI with dynamic bands + arrows ──
    fig.add_trace(go.Scatter(x=dates, y=vis["RSI_14"], name="RSI",
                             line=dict(color="#7c3aed", width=1.5)), row=4, col=1)
    for col, lab in (("RSI_P90", "P90"), ("RSI_P10", "P10")):
        fig.add_trace(go.Scatter(x=dates, y=vis[col], name=f"RSI {lab}",
                                 line=dict(color="rgba(148,163,184,.8)", width=1,
                                           dash="dot"), showlegend=False), row=4, col=1)
    for flag, lab, sym, colour in (("RSI_Bottoming", "RSI 见底", "arrow-up", "#dc2626"),
                                   ("RSI_Peaking", "RSI 见顶", "arrow-down", "#15803d")):
        msk = vis[flag].fillna(False).values.astype(bool)
        if msk.any():
            fig.add_trace(go.Scatter(
                x=[d for d, m in zip(dates, msk) if m],
                y=vis["RSI_14"][msk], mode="markers", name=lab,
                marker=dict(symbol=sym, size=11, color=colour)), row=4, col=1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line_dash="dot", line_width=1,
                      line_color="rgba(148,163,184,.55)", row=4, col=1)

    # ── 5 · ADX trend analysis ──
    fig.add_trace(go.Scatter(x=dates, y=vis["ADX"], name="ADX",
                             line=dict(color="#0f172a", width=2)), row=5, col=1)
    fig.add_trace(go.Scatter(x=dates, y=vis["DI_Plus"], name="+DI",
                             line=dict(color="#dc2626", width=1.2)), row=5, col=1)
    fig.add_trace(go.Scatter(x=dates, y=vis["DI_Minus"], name="-DI",
                             line=dict(color="#15803d", width=1.2)), row=5, col=1)
    for lvl, txt in ((20, "weak"), (30, "strong")):
        fig.add_hline(y=lvl, line_dash="dot", line_width=1,
                      line_color="rgba(148,163,184,.7)", row=5, col=1)

    # ── 6 · Z-Score price + volume ──
    fig.add_trace(go.Bar(x=dates, y=vis["Volume_ZScore"], name="Vol Z",
                         marker_color="rgba(100,116,139,.55)"), row=6, col=1)
    fig.add_trace(go.Scatter(x=dates, y=vis["Price_ZScore"], name="Price Z",
                             line=dict(color="#2a78d6", width=1.6)), row=6, col=1)
    for lvl in (-2, 0, 2):
        fig.add_hline(y=lvl, line_dash="dot", line_width=1,
                      line_color="rgba(148,163,184,.6)", row=6, col=1)

    fig.update_layout(height=1180, template="plotly_white",
                      xaxis_rangeslider_visible=False, hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10), bargap=0.15,
                      legend=dict(orientation="h", yanchor="bottom", y=1.015, x=0,
                                  font=dict(size=9)))
    fig.update_yaxes(range=[0, 100], row=4, col=1)
    fig.update_yaxes(range=[0, 60], row=5, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    _step = max(1, len(dates) // 10)
    fig.update_xaxes(type="category", showticklabels=False)
    fig.update_xaxes(type="category", tickangle=-45, showticklabels=True,
                     tickmode="array", tickvals=dates[::_step], row=6, col=1)
    st.plotly_chart(fig, use_container_width=True)

    _phase = str(vis["ADX_Phase"].iloc[-1] or "—")
    st.caption(
        f"ADX {float(vis['ADX'].iloc[-1]):.1f} · {_phase} &nbsp;|&nbsp; "
        f"RSI {float(vis['RSI_14'].iloc[-1]):.1f} "
        f"(band {float(vis['RSI_P10'].iloc[-1]):.0f}–{float(vis['RSI_P90'].iloc[-1]):.0f}) "
        f"&nbsp;|&nbsp; Vol Z {float(vis['Volume_ZScore'].iloc[-1]):+.2f}",
        unsafe_allow_html=True)

    # ── 吸筹 / 出货 ───────────────────────────────────────────────────────
    with st.expander("🧭 吸筹 / 出货 · Accumulation vs Distribution", expanded=False):
        # Fed the SAME truncated frame the chart draws, so the panel can only
        # see what the player can see.
        _res = acsig.detect(vis, None, window=20)
        if not _res.get("ok"):
            st.caption(f"Not enough history yet — {_res.get('reason')}")
        else:
            _s = acsig.summarise(_res)
            st.info(f"**{_s['verdict']}** — 吸筹 {_s['n_acc']} / 出货 {_s['n_dist']}"
                    + (f" · 区间位置 {_res['range_position']:.0%}"
                       if _res.get("range_position") is not None else ""))
            _c1, _c2 = st.columns(2)
            for _col, _title, _colour, _live in (
                    (_c1, "🟥 吸筹 Accumulation", "#dc2626", _s["acc_live"]),
                    (_c2, "🟩 出货 Distribution", "#15803d", _s["dist_live"])):
                with _col:
                    st.markdown(f"<b style='color:{_colour}'>{_title}</b>",
                                unsafe_allow_html=True)
                    if not _live:
                        st.caption("—")
                    for _sg in _live:
                        _when = (f"{_sg['count_60']}× · last {_sg['last']}"
                                 if _sg["kind"] == "event"
                                 else f"{_sg['run']}d since {_sg['since']}")
                        st.markdown(f"- **{_sg['cn']}** · {_sg['label']}  \n"
                                    f"  <span style='font-size:11px;color:#64748b'>🕒 {_when}</span>",
                                    unsafe_allow_html=True)
            st.caption(
                "主力资金 detectors are inactive here — money-flow data is not "
                "loaded for the hidden stock, so this is the price/volume half "
                "only. Signals are computed on the visible window alone."
            )

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
                        # Themes look back a week, macro two days. A specific
                        # theme produces only a handful of Guardian hits even
                        # across a full week, so a two-day window on themes is
                        # empty most days — which reads as "broken" rather than
                        # "nothing on that topic happened".
                        _win = 7 if src.startswith("主题") else 2
                        cache[ckey] = game_news.fetch_guardian(
                            query, today, window_days=_win,
                            redact=[pick["name"], pick["ticker"]])
                st.rerun(scope="fragment")
            if not src.startswith("🌍"):
                _w = 7 if src.startswith("主题") else 2
                st.caption(f"query: `{query}` · looking back {_w} days from {today}")

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
