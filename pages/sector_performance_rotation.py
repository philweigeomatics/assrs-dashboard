"""
Sector Analysis - Performance & Rotation
Track sector performance, true RRG-style rotation, lead-lag interactions,
and a next-rotation read from the transition matrix.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sector_utils import (
    load_v2_data,
    load_csi300_with_regime,
    build_sector_panels,
    create_performance_comparison_chart,
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
st.subheader("📊 Sector Performance Comparison")
lookback_days = st.slider("Lookback days", 30, 180, 60, 10)
fig = create_performance_comparison_chart(v2hist, lookback_days, selected_sectors)
if fig:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Unable to create performance chart")

st.markdown("---")

# ---- Rolling correlation (on excess returns now) ----
st.subheader("📊 Rolling Excess-Return Correlations")
st.caption("Correlations computed on **excess returns vs CSI300** so co-movement with the market is stripped out.")

col1, col2 = st.columns([1, 2])
with col1:
    reference_sector = st.selectbox("Reference Sector", all_sectors)
    window = st.slider("Rolling Window (days)", 5, 60, 20, 5)
with col2:
    compare_sectors = st.multiselect(
        "Compare Against",
        [s for s in all_sectors if s != reference_sector],
        default=[s for s in all_sectors if s != reference_sector][:4]
    )

if compare_sectors:
    fig = create_rolling_correlation_chart(exret_panel, reference_sector, compare_sectors, window)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---- RRG ----
st.subheader("🔄 Relative Rotation Graph (vs CSI300)")
st.caption(
    "Proper RRG: **x = RS-Ratio** (sector strength vs CSI300, centred at 100), "
    "**y = RS-Momentum** (rate of change of RS-Ratio, centred at 100). "
    "The tail shows the trajectory over the chosen window — direction of travel is "
    "what tells you whether a sector is *rotating into* a quadrant or drifting out."
)

c1, c2, c3 = st.columns(3)
with c1:
    rs_window = st.selectbox("RS window (days)", [10, 20, 40, 60], index=1,
                             help="Window for the relative-strength z-score normalisation.")
with c2:
    mom_window = st.selectbox("Momentum lag (days)", [3, 5, 10, 20], index=1,
                              help="How many days back to compare RS-Ratio for momentum.")
with c3:
    tail_length = st.selectbox("Tail length (days)", [5, 10, 20, 40], index=1,
                               help="How many trading days of trajectory to draw per sector.")

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
        st.plotly_chart(fig, use_container_width=True)

        # Quadrant breakdown — labels match what create_rrg_chart writes
        st.markdown("#### 📊 Quadrant Breakdown")
        quad_cols = st.columns(4)
        for idx, quadrant in enumerate(["Leading", "Improving", "Weakening", "Lagging"]):
            with quad_cols[idx]:
                in_quad = rotation_df[rotation_df['Quadrant'] == quadrant]
                st.markdown(f"**{quadrant}** ({len(in_quad)})")
                if in_quad.empty:
                    st.caption("None")
                else:
                    for _, row in in_quad.iterrows():
                        # Direction-of-travel arrow from tail deltas
                        dx, dy = row['ΔRS_Ratio_tail'], row['ΔRS_Momentum_tail']
                        if dx > 0 and dy > 0:
                            arrow = "↗"
                        elif dx < 0 and dy > 0:
                            arrow = "↖"
                        elif dx < 0 and dy < 0:
                            arrow = "↙"
                        else:
                            arrow = "↘"
                        st.markdown(f"• {row['Sector']} {arrow}")

        with st.expander("Raw RRG table", expanded=False):
            st.dataframe(
                rotation_df.round(2).sort_values(
                    ['Quadrant', 'RS_Ratio'], ascending=[True, False]
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
