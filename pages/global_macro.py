"""
Global Macro — US yield curve movement + overseas equity board.

Context for an A-share dashboard: US rates set the global cost of capital and
drive the dollar, which drives foreign flows into and out of Chinese equities;
Japan and Korea trade concurrently with China and read regional risk appetite.
Neither is a trading signal on its own — this page is the weather report the
domestic pages are read against.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import auth_manager
import data_manager
import global_macro as gmac

auth_manager.require_login()

st.title("🌍 Global Macro · 全球宏观")
st.caption(
    "US Treasury curve from FRED, overseas index board from Tushare or ETF "
    "proxies. Both are end-of-day — free real-time international equity data "
    "does not exist, and for reading the tone into an A-share session the "
    "overnight close is the number that matters anyway."
)


@st.cache_data(ttl=1800, show_spinner=False)
def _curve(lookback_days: int = 450):
    return gmac.fetch_yield_curve(lookback_days=lookback_days)


@st.cache_data(ttl=1800, show_spinner=False)
def _spread(series_id: str):
    return gmac.fetch_spread(series_id, lookback_days=900)


@st.cache_data(ttl=900, show_spinner=False)
def _board():
    try:
        data_manager.init_tushare()
        pro = data_manager.TUSHARE_API
    except Exception:
        pro = None
    return gmac.fetch_global_board(pro_api=pro)


# ── Yield curve ───────────────────────────────────────────────────────────────
st.subheader("📈 美债收益率曲线 · US Treasury Yield Curve")

with st.spinner("Loading Treasury curve from FRED…"):
    _c = _curve()

if not _c["ok"]:
    st.error(f"Yield curve unavailable — {_c.get('reason')}")
    st.caption(
        "Needs a free FRED API key: create one at "
        "https://fredaccount.stlouisfed.org/apikeys then add "
        "`FRED_API_KEY` to app Settings → Secrets."
    )
else:
    df = _c["data"]

    _mv_days = st.select_slider(
        "Compare against", options=[7, 30, 90, 180, 365], value=30,
        format_func=lambda d: {7: "1 week ago", 30: "1 month ago",
                               90: "3 months ago", 180: "6 months ago",
                               365: "1 year ago"}[d],
    )
    move = gmac.classify_curve_move(df, days=_mv_days)

    if move["ok"]:
        _tone = "warning" if move["direction"] == "bear" else "success"
        getattr(st, _tone)(
            f"**{move['label']}** — over {move['days']} days the 2Y moved "
            f"**{move['d_short_bp']:+.0f}bp** and the 10Y **{move['d_long_bp']:+.0f}bp**, "
            f"so the 2s10s spread {'widened' if move['d_spread_bp'] > 0 else 'narrowed'} "
            f"by {abs(move['d_spread_bp']):.0f}bp."
        )
        with st.expander("What the four names mean", expanded=False):
            st.markdown(
                "- **Bear steepening** — long end selling off faster. "
                "Growth, inflation, or supply concerns.\n"
                "- **Bull steepening** — front end rallying faster. Cuts being priced.\n"
                "- **Bear flattening** — front end selling off faster. Hikes being priced.\n"
                "- **Bull flattening** — long end rallying faster. "
                "Growth scare, or a bid for duration.\n\n"
                "*Bull* means yields fell (bond prices rose); *bear* means yields rose. "
                "Direction is taken from the average of the two ends, so a flat long "
                "end doesn't decide the label on its own."
            )

    snap = gmac.curve_snapshots(df, offsets={f"{_mv_days}d ago": _mv_days})
    if not snap.empty and snap.shape[1] >= 2:
        prior_col = [c for c in snap.columns if c != "Today"][0]
        chg_bp = (snap["Today"] - snap[prior_col]) * 100.0

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
            vertical_spacing=0.08,
            subplot_titles=("Yield %", "Change (bp)"),
        )
        fig.add_trace(go.Scatter(
            x=snap.index, y=snap[prior_col], name=prior_col, mode="lines+markers",
            line=dict(color="#94a3b8", width=2, dash="dash"),
            marker=dict(size=6),
            hovertemplate="%{x}<br>" + prior_col + " %{y:.2f}%<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=snap.index, y=snap["Today"], name="Today", mode="lines+markers",
            line=dict(color="#2a78d6", width=3), marker=dict(size=8),
            hovertemplate="%{x}<br>Today %{y:.2f}%<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=chg_bp.index, y=chg_bp.values, showlegend=False,
            marker_color=["#dc2626" if v > 0 else "#15803d" for v in chg_bp.values],
            hovertemplate="%{x}<br>%{y:+.1f}bp<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line_width=1, line_color="rgba(148,163,184,.7)",
                      row=2, col=1)
        fig.update_layout(
            height=520, template="plotly_white",
            margin=dict(l=10, r=10, t=48, b=10), hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"FRED constant-maturity yields, as of **{_c['as_of']}** · "
            "red = yields rose (bond prices fell), green = fell · "
            "updated ~18:00 ET each business day."
            + (f" · missing tenors: {', '.join(_c['missing'])}" if _c.get("missing") else "")
        )

    # ── Spreads ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 期限利差 · Key Spreads")
    st.caption(
        "10Y minus 2Y and 10Y minus 3M. Below zero is an inverted curve — the "
        "market pricing lower rates ahead, historically associated with "
        "recessions at a long and variable lag. The re-steepening back through "
        "zero has usually mattered more than the inversion itself."
    )
    _s1, _s2 = st.columns(2)
    for _col, _sid, _name in ((_s1, "T10Y2Y", "10Y − 2Y"),
                              (_s2, "T10Y3M", "10Y − 3M")):
        with _col:
            sp = _spread(_sid)
            if not sp["ok"]:
                st.info(f"{_name} unavailable ({sp.get('reason')})")
                continue
            s = sp["data"]
            st.metric(f"{_name} ({_sid})", f"{sp['latest']:+.2f}%",
                      delta=f"{(sp['latest'] - float(s.iloc[-22])) * 100:+.0f}bp vs 1M"
                      if len(s) > 22 else None)
            f = go.Figure(go.Scatter(
                x=s.index, y=s.values, mode="lines",
                line=dict(color="#2a78d6", width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.2f}%<extra></extra>"))
            f.add_hline(y=0, line_dash="dash", line_color="#dc2626", line_width=1)
            f.update_layout(height=210, template="plotly_white",
                            margin=dict(l=8, r=8, t=6, b=6), showlegend=False)
            st.plotly_chart(f, use_container_width=True)

# ── Overseas board ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🌏 海外股指 · Overseas Equity Board")

with st.spinner("Loading overseas markets…"):
    _b = _board()

if not _b["ok"]:
    st.info("Overseas board unavailable — check TWELVEDATA_API_KEY, or Tushare points.")
else:
    _rows = _b["rows"]
    _cols = st.columns(len(_rows))
    for _col, _r in zip(_cols, _rows):
        with _col:
            if not _r.get("ok"):
                st.metric(_r["label"], "—")
                st.caption(str(_r.get("reason", ""))[:60])
                continue
            # A-share convention: red = up, green = down. Streamlit's own
            # delta colouring is the Western way round, so it is inverted here.
            st.metric(
                f"{_r['label']}",
                f"{_r['value']:,.2f}",
                delta=f"{_r['pct']:+.2f}%",
                delta_color="inverse",
            )
            # A bare date hides a 14-hour spread across these rows, so show the
            # moment each session actually closed in Beijing terms, and say so
            # when that market has since reopened and this number is a session
            # behind what is happening there now.
            _sx = _r.get("session") or {}
            if _sx.get("ok"):
                _age = _sx["hours_ago"]
                _age_s = (f"{_age:.0f}小时前" if _age < 48
                          else f"{_age / 24:.0f}天前")
                st.caption(
                    f"{_r['cn']} · 收盘 {_sx['closed_bj_str']} 北京时间 · {_age_s}"
                    + ("　🟢 已开盘 trading now" if _sx["is_open"] else "")
                )
            else:
                st.caption(f"{_r['cn']} · {_r['date']}")

    if _b["proxy"]:
        st.warning(
            "⚠️ **These are US-listed ETF proxies, not the indices themselves** "
            "— Tushare `index_global` was unavailable, and Twelve Data's free "
            "tier excludes index symbols, so Nikkei is read through EWJ, KOSPI "
            "through EWY. The gap is not small: on 2026-08-18 **EWY showed "
            "−8.13% while KOSPI actually closed −1.55%**. The proxies are "
            "USD-denominated, so currency moves read as market moves, and they "
            "trade **US hours**, so every row above carries a New York "
            "timestamp rather than that market's own close. Treat direction as "
            "indicative and ignore the magnitude."
        )
    else:
        st.caption(
            "Real index levels, each stamped with its own market's close "
            "converted to Beijing time — the same trade date means very "
            "different moments (Tokyo's close is 14:00 Beijing that day; "
            "New York's is 04:00 Beijing the next), so a US number can be "
            "hours old while an Asian one is a full session behind."
        )
    st.caption(f"Source: {_b['source']}")
