"""
Sector Analysis - Performance & Rotation
Track sector performance, true RRG-style rotation, lead-lag interactions,
and a next-rotation read from the transition matrix.
"""
import json

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import ai_client

from sector_utils import (
    load_v2_data,
    load_csi300_with_regime,
    build_sector_panels,
    create_performance_comparison_chart,
    create_sector_return_heatmap,
    build_rotation_digest,
    create_rolling_correlation_chart,
    compute_rrg_series,
    create_rrg_chart,
    compute_lead_lag_matrix,
    compute_transition_matrix,
    get_today_topk,
    predict_tomorrow,
)

import auth_manager
auth_manager.require_login()


st.title("📈 Performance & Rotation")

# ---- Data ----
v2latest, v2hist, v2date, v2error = load_v2_data()
if v2latest is None:
    st.error(f"Error loading data: {v2error}")
    st.stop()

# Load CSI300 so excess returns and RRG use the real benchmark
csi300_df = load_csi300_with_regime('日线')

close_panel, ret_panel, vol_panel, exret_panel = build_sector_panels(
    v2hist, csi300_df=csi300_df
)

all_sectors = sorted([s for s in v2hist['Sector'].unique() if s != "MARKET_PROXY"])
selected_sectors = st.multiselect(
    "Select sectors to compare",
    all_sectors,
    default=all_sectors[:5]
)

if not selected_sectors:
    st.warning("Please select at least one sector")
    st.stop()

# ---- Performance ----
st.subheader("📊 板块共振还是轮动？ · Moving together, or rotating?")
st.caption(
    "Each cell is one sector's move on one day — **红涨绿跌**. A column of one "
    "colour means the whole market moved as one that day, so only timing "
    "mattered; a column of mixed colour means sectors rotated against each "
    "other, which is when picking sectors pays. Rows are ordered by cumulative "
    "return over the window, shown next to each name. The strip underneath is "
    "the spread between sectors each day."
)
lookback_days = st.slider("Lookback days", 30, 180, 60, 10)
fig = create_sector_return_heatmap(v2hist, lookback_days, selected_sectors)
if fig:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Not enough overlapping sector data to draw the heatmap.")

with st.expander("📈 Cumulative lines (base 100)", expanded=False):
    st.caption(
        "The old view. Good for reading a level on a given date; poor at "
        "showing whether sectors moved together, which is what the heatmap adds."
    )
    _line_fig = create_performance_comparison_chart(v2hist, lookback_days, selected_sectors)
    if _line_fig:
        st.plotly_chart(_line_fig, use_container_width=True)

st.markdown("---")

# ---- AI read of the whole rotation picture ----
# The page's difficulty was never one chart; it's that each section answers a
# fragment and none states the conclusion. This sends the fragments as one
# bundle and asks for the conclusion. Behind a button — one DeepSeek call.
_ROT_PROMPT = """\
You are a senior Chinese A-share strategist briefing a portfolio manager.

You are given measurements of the CURRENT sector regime. Explain what they say
TOGETHER — the state of the market, not a restatement of each number.

Answer, in this order:
1. Is this a rotating market or a co-moving one? Dispersion percentile and
   average pairwise correlation decide this, not the individual returns.
2. Who is actually leading, and is that leadership stable or churning? A high
   leader-switch rate means today's winner tells you little about tomorrow's.
3. Does sector selection even pay right now? When dispersion sits low in its
   own history, sectors move as one and timing matters more than picking. Say
   so plainly when that is the case.

RULES:
- Dispersion is percentile-ranked against its own past year, so "high" means
  high FOR THIS MARKET, not high in absolute terms.
- Sector returns are relative to the CSI 300 figure given; a sector up 3% in a
  market up 5% is lagging, not winning.
- Do not invent numbers, forecast prices, or recommend specific trades. This is
  a description of market structure.
- Write in Chinese. Metric names may stay in English.
- Return ONLY raw JSON (start { end }). No markdown fences.

Schema:
{
  "headline": "One sentence: rotating, co-moving, or in between.",
  "regime": "轮动 | 共振 | 混合",
  "leadership": "Who leads and whether it is holding or churning.",
  "selection_pays": "Whether picking sectors is worth it right now, and why.",
  "watch_outs": ["Caveats — short window, unstable leadership, etc."],
  "bottom_line": "2-3 sentences a PM could act on."
}
"""


@st.cache_data(ttl=1800, show_spinner=False)
def _rotation_ai_read(digest_json: str, _nonce: int) -> dict:
    """Cached on the digest itself, so identical inputs don't re-bill a call."""
    return ai_client.call_json(
        _ROT_PROMPT, digest_json,
        max_tokens=9000,
        temperature=0.3,
        # Reading a fixed set of supplied statistics — classification and
        # explanation, not multi-step derivation.
        reasoning_effort="low",
    )


st.subheader("🤖 综合解读 · What all of this adds up to")
st.caption(
    "Bundles the numbers behind every section on this page — dispersion vs its "
    "own history, each sector's return against CSI 300, how often leadership "
    "changes — and asks for the one conclusion the charts don't state."
)

_digest = build_rotation_digest(
    close_panel, exret_panel, csi300_df, selected_sectors, lookback=lookback_days
)

if _digest is None:
    st.info("Not enough overlapping sector data for a summary at this window.")
else:
    if st.button("🤖 解读当前板块格局 · Read the current regime", key="rot_ai_btn"):
        with st.spinner("Asking DeepSeek to read the regime…"):
            try:
                st.session_state["rot_ai"] = {
                    "data": _rotation_ai_read(json.dumps(_digest, ensure_ascii=False), 0)
                }
            except Exception as exc:
                st.session_state["rot_ai"] = {"error": str(exc)}

    _r = st.session_state.get("rot_ai")
    if _r and "error" in _r:
        st.error(f"AI read failed: {_r['error']}")
    elif _r:
        _d = _r["data"]
        st.info(f"**{_d.get('headline', '—')}**")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("判定 Regime", _d.get("regime", "—"))
            if _digest.get("dispersion_pctile") is not None:
                st.metric("离散度百分位 Dispersion pctile",
                          f"{_digest['dispersion_pctile'] * 100:.0f}%")
        with m2:
            if _digest.get("avg_pairwise_corr") is not None:
                st.metric("平均两两相关 Avg pairwise corr",
                          f"{_digest['avg_pairwise_corr']:.2f}")
            if _digest.get("leader_switch_rate") is not None:
                st.metric("领涨切换率 Leader switch rate",
                          f"{_digest['leader_switch_rate'] * 100:.0f}%")

        if _d.get("leadership"):
            st.markdown("**领涨情况 Leadership**")
            st.write(_d["leadership"])
        if _d.get("selection_pays"):
            st.markdown("**选板块值得吗 Does selection pay?**")
            st.write(_d["selection_pays"])
        for _w in _d.get("watch_outs") or []:
            st.warning(_w)
        if _d.get("bottom_line"):
            st.success(_d["bottom_line"])
        st.caption(
            f"Window {_digest['start']} → {_digest['end']} "
            f"({_digest['sessions']} sessions). Market-structure description, "
            "not investment advice."
        )

    with st.expander("📋 Numbers sent to the model", expanded=False):
        st.json(_digest)

st.markdown("---")

# ---- Rolling correlation (on excess returns now) ----
st.subheader("📊 有没有共同驱动？ · Do these sectors share a driver?")
st.caption(
    "Correlation over time, computed on **excess returns vs CSI300** — the "
    "whole-market move is subtracted first. Two sectors can look 80% correlated "
    "just because the index moved; here a high reading means they share "
    "something *specific*, beyond both being A-shares. Watch for a line that "
    "falls away: that's a relationship breaking down."
)

if len(selected_sectors) < 2:
    st.info("Pick at least 2 sectors above to see rolling correlations.")
else:
    # This section used to read all_sectors while every other section on the
    # page reads selected_sectors, so it ignored the picker at the top. Both
    # widgets now derive from the same selection.
    col1, col2 = st.columns([1, 2])
    with col1:
        # Reset the reference if the top selection no longer contains it.
        if st.session_state.get("rc_ref") not in selected_sectors:
            st.session_state["rc_ref"] = selected_sectors[0]
        reference_sector = st.selectbox("Reference sector", selected_sectors, key="rc_ref")
        window = st.slider("Rolling Window (days)", 5, 60, 20, 5)

    _avail = [s for s in selected_sectors if s != reference_sector]

    # A keyed widget keeps its own state, so `default=` alone would never
    # refresh it — that's the second half of why this pane looked frozen.
    # Re-seed only when the selection or the reference actually changes, so a
    # deliberate hand-picked comparison isn't wiped on every rerun.
    #
    # Seed with EVERY available sector, not a first-N slice. Slicing made the
    # options list follow the top picker while the plotted lines didn't: add a
    # sector up top, watch the chart not change, conclude it's still broken.
    # The top picker is already the place where the choice is narrowed — this
    # pane shouldn't quietly re-narrow it.
    _sig = (tuple(sorted(selected_sectors)), reference_sector)
    if st.session_state.get("_rc_sig") != _sig:
        st.session_state["_rc_sig"] = _sig
        st.session_state["rc_compare"] = _avail

    with col2:
        compare_sectors = st.multiselect("Compare against", _avail, key="rc_compare")

    if compare_sectors:
        fig = create_rolling_correlation_chart(
            exret_panel, reference_sector, compare_sectors, window
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Pick at least one sector to compare against.")

st.markdown("---")

# ---- RRG ----
# Answer first, picture second. The quadrant breakdown always carried the
# information; it just sat underneath a scatter that becomes unreadable once
# several sectors each draw a trajectory. The chart is still here, one click
# away, for anyone who wants to see the paths.
st.subheader("🔄 谁在领先？ · Who is leading, and is it holding?")
st.caption(
    "Each sector's position against CSI 300, and whether that position is "
    "**improving or fading**. Arrows show direction of travel. Sectors tend to "
    "travel clockwise — 改善 → 领先 → 走弱 → 落后 — so a name in 走弱 is often "
    "an exit signal even while it is still ahead."
)

c1, c2, c3 = st.columns(3)
with c1:
    rs_window = st.selectbox("RS window (days)", [10, 20, 40, 60], index=1,
                             help="Window for the relative-strength z-score normalisation.")
with c2:
    mom_window = st.selectbox("Momentum lag (days)", [3, 5, 10, 20], index=1,
                              help="How many days back to compare RS-Ratio for momentum.")
with c3:
    # Default 5 rather than 10: the tails are what turn this chart into
    # spaghetti once more than a few sectors are selected.
    tail_length = st.selectbox("Tail length (days)", [5, 10, 20, 40], index=0,
                               help="How many trading days of trajectory to draw per sector. "
                                    "Shorter is easier to read with many sectors.")

if csi300_df is None or 'Close' not in (csi300_df.columns if csi300_df is not None else []):
    st.error("CSI300 data unavailable — cannot build RRG. Check data_manager.get_index_data_live('000300.SH').")
else:
    rs_ratio_panel, rs_mom_panel = compute_rrg_series(
        close_panel, csi300_df,
        sectors=selected_sectors,
        rs_window=rs_window,
        mom_window=mom_window,
    )

    fig, rotation_df = create_rrg_chart(
        rs_ratio_panel, rs_mom_panel,
        sectors=selected_sectors,
        tail_length=tail_length,
    )

    if fig is None or rotation_df is None or rotation_df.empty:
        st.warning("Not enough overlapping CSI300 / sector data to build the RRG.")
    else:
        # Quadrant cards FIRST — this is the readable answer. Order runs the way
        # sectors actually travel (clockwise) rather than by quadrant name, so
        # the cycle reads left to right.
        _QUADS = [
            ("Improving", "改善 Improving", "behind but catching up",   "#185FA5"),
            ("Leading",   "领先 Leading",   "ahead & still gaining",    "#0F6E56"),
            ("Weakening", "走弱 Weakening", "ahead but losing steam",   "#854F0B"),
            ("Lagging",   "落后 Lagging",   "behind & still sliding",   "#A32D2D"),
        ]
        quad_cols = st.columns(4)
        for _col, (_key, _title, _sub, _colour) in zip(quad_cols, _QUADS):
            with _col:
                in_quad = rotation_df[rotation_df["Quadrant"] == _key]
                st.markdown(
                    f"<div style='font-size:13px;font-weight:600;color:{_colour}'>{_title}"
                    f" ({len(in_quad)})</div>"
                    f"<div style='font-size:11px;color:rgba(120,120,120,.9);"
                    f"margin:2px 0 8px'>{_sub}</div>",
                    unsafe_allow_html=True,
                )
                if in_quad.empty:
                    st.caption("—")
                else:
                    for _, row in in_quad.iterrows():
                        dx, dy = row["ΔRS_Ratio_tail"], row["ΔRS_Momentum_tail"]
                        if dx > 0 and dy > 0:
                            arrow = "↗"
                        elif dx < 0 and dy > 0:
                            arrow = "↖"
                        elif dx < 0 and dy < 0:
                            arrow = "↙"
                        else:
                            arrow = "↘"
                        st.markdown(
                            f"<div style='font-size:14px;line-height:1.9'>{row['Sector']} "
                            f"<span style='color:{_colour}'>{arrow}</span></div>",
                            unsafe_allow_html=True,
                        )

        st.caption("顺时针轮动 · sectors travel clockwise: 改善 → 领先 → 走弱 → 落后 → 改善")

        # Scatter second, and collapsed — it's the part that gets unreadable.
        with st.expander("📉 散点图与轨迹 · Scatter chart & trajectories", expanded=False):
            st.caption(
                "**x = RS-Ratio** (strength vs CSI 300, centred at 100), "
                "**y = RS-Momentum** (whether that strength is rising or falling, "
                "also centred at 100). The tail is each sector's path over the last "
                f"{tail_length} sessions. Shorten the tail above if it's crowded."
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw RRG table", expanded=False):
            st.dataframe(
                rotation_df.round(2).sort_values(
                    ["Quadrant", "RS_Ratio"], ascending=[True, False]
                ),
                hide_index=True,
            )

st.markdown("---")

# ---- Lead-Lag (the actual "interaction" question) ----
st.subheader("🔗 Lead-Lag Interactions (excess returns)")
st.caption(
    "For each pair, the lag in [-max, +max] trading days that maximises "
    "|correlation| of excess returns. **Positive lag = column-sector leads row-sector**."
)

c1, c2 = st.columns(2)
with c1:
    max_lag = st.selectbox("Max lag (days)", [3, 5, 7, 10], index=1)
with c2:
    ll_lookback = st.selectbox("Lookback (days)", [60, 90, 120, 180, 252], index=2)

ll_sectors = selected_sectors
if len(ll_sectors) < 2:
    st.info("Pick at least 2 sectors above to see lead-lag.")
else:
    best_lag, best_corr = compute_lead_lag_matrix(
        exret_panel, ll_sectors, max_lag=max_lag, lookback=ll_lookback
    )
    if best_lag is None:
        st.warning("Not enough data for lead-lag at this window.")
    else:
        tab_lag, tab_corr, tab_summary = st.tabs(["Best lag (days)", "Peak correlation", "Who leads whom"])

        with tab_lag:
            fig_lag = go.Figure(data=go.Heatmap(
                z=best_lag.values,
                x=best_lag.columns.tolist(),
                y=best_lag.index.tolist(),
                colorscale='RdBu',
                zmid=0,
                zmin=-max_lag, zmax=max_lag,
                text=best_lag.values,
                texttemplate='%{text}',
                colorbar=dict(title='Lag (days)<br>+ = col leads row'),
            ))
            fig_lag.update_layout(
                title=f"Best lag at peak |corr| — last {ll_lookback}d",
                height=max(400, len(ll_sectors) * 55),
                template='plotly_white',
            )
            st.plotly_chart(fig_lag, use_container_width=True)

        with tab_corr:
            fig_c = go.Figure(data=go.Heatmap(
                z=best_corr.values,
                x=best_corr.columns.tolist(),
                y=best_corr.index.tolist(),
                colorscale='RdYlGn',
                zmid=0, zmin=-1, zmax=1,
                text=best_corr.values.round(2),
                texttemplate='%{text}',
                colorbar=dict(title='Peak corr'),
            ))
            fig_c.update_layout(
                title=f"Peak |corr| across lags ±{max_lag}d — last {ll_lookback}d",
                height=max(400, len(ll_sectors) * 55),
                template='plotly_white',
            )
            st.plotly_chart(fig_c, use_container_width=True)

        with tab_summary:
            # For each sector, list the strongest leader and strongest follower
            rows = []
            for s in ll_sectors:
                col = best_lag[s]
                corr_col = best_corr[s]
                # leaders of s = rows where col[s] (B=s leads A=row) is positive lag
                leaders = []
                followers = []
                for other in ll_sectors:
                    if other == s:
                        continue
                    lag_other_leads_s = best_lag.loc[s, other]
                    corr_other_leads_s = best_corr.loc[s, other]
                    if lag_other_leads_s > 0 and abs(corr_other_leads_s) >= 0.2:
                        leaders.append((other, int(lag_other_leads_s), float(corr_other_leads_s)))
                    elif lag_other_leads_s < 0 and abs(corr_other_leads_s) >= 0.2:
                        followers.append((other, int(-lag_other_leads_s), float(corr_other_leads_s)))
                leaders.sort(key=lambda t: -abs(t[2]))
                followers.sort(key=lambda t: -abs(t[2]))
                rows.append({
                    'Sector': s,
                    'Led by (sector, lag d, corr)': ", ".join(
                        [f"{n} ({l}d, {c:+.2f})" for n, l, c in leaders[:3]]
                    ) or "—",
                    'Leads (sector, lag d, corr)': ", ".join(
                        [f"{n} ({l}d, {c:+.2f})" for n, l, c in followers[:3]]
                    ) or "—",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("Only pairs with |peak corr| ≥ 0.20 are listed. Lag is in trading days.")

st.markdown("---")

# ---- Next-rotation prediction ----
st.subheader("🎯 Next-Rotation Read (Transition Matrix)")
st.caption(
    "From the transition matrix of daily top-K leaders in excess return, "
    "what tends to follow today's leaders. Same engine as the Interaction Lab."
)

c1, c2 = st.columns(2)
with c1:
    pred_topk = st.selectbox("Top-K leaders / day", [2, 3, 4, 5], index=1, key='rot_topk')
with c2:
    pred_lb = st.selectbox("Training lookback (days)", [30, 60, 90, 120], index=2, key='rot_lb')

probs, counts = compute_transition_matrix(exret_panel, lookback=pred_lb, top_k=pred_topk)
if probs is None:
    st.info("Not enough data for transition matrix at this window.")
else:
    latest_dt, leaders = get_today_topk(exret_panel, pred_topk)
    pred = predict_tomorrow(probs, counts, leaders)
    cA, cB = st.columns(2)
    with cA:
        st.markdown(f"**Today's leaders** ({latest_dt.strftime('%Y-%m-%d')})")
        st.dataframe(
            leaders.reset_index().rename(columns={'index': 'Sector', latest_dt: 'ExcessRet'}),
            hide_index=True, use_container_width=True,
        )
    with cB:
        st.markdown("**Most likely followers tomorrow**")
        st.dataframe(pred, hide_index=True, use_container_width=True)

st.markdown("---")

# ---- Correlation matrix (excess returns now) ----
st.subheader("🔗 Sector Correlation Matrix (excess returns)")
st.caption("On excess returns vs CSI300, so values reflect *inter-sector* co-movement, not shared market beta.")

ex = exret_panel[selected_sectors].dropna()
if len(ex) > 0:
    corr_matrix = ex.corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale='RdYlGn',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont=dict(size=14),
        colorbar=dict(title='Correlation'),
        zmin=-1, zmax=1,
    ))
    fig_corr.update_layout(
        title="Daily Excess-Return Correlation",
        height=max(400, len(selected_sectors) * 60),
        template='plotly_white',
    )
    st.plotly_chart(fig_corr, use_container_width=True)
