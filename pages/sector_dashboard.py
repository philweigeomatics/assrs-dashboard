"""
Sector Analysis - Dashboard
Shows overview of all sectors with key metrics and CSI 300 chart
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import data_manager as dm
import market_leverage as mlev
import ai_client

from sector_utils import (
    load_v2_data, 
    load_csi300_with_regime,
    create_sector_chart
)

import auth_manager
auth_manager.require_login()


# Load data
v2latest, v2hist, v2date, v2error = load_v2_data()
if v2latest is None:
    st.error(f"Error loading data: {v2error}")
    st.stop()

# ===================================================================
# SECTION 0: MARKET HEATMAP (sector-segmented, market-cap weighted)
# ===================================================================
# Box size = 流通市值 (circulating market cap, from daily_basic in the DB).
# Box colour = last trading day's % change. A-share convention: red = up,
# green = down. Grouped by the same sector map used for market breadth.

@st.cache_data(ttl=1800, show_spinner=False)
def _load_heatmap_data():
    """Return (DataFrame[sector, stock, ticker, mcap, pct_chg], trade_date) or (None, None)."""
    import re
    sector_map = dm.get_sector_stock_map()
    all_tickers = sorted({t for lst in sector_map.values() for t in lst})
    if not all_tickers:
        return None, None

    # Market cap (circulating, 亿) from the DB — one read of the latest date.
    mcap = dm.get_daily_basic_for_tickers(all_tickers)
    if mcap is None or mcap.empty:
        return None, None
    latest_date = str(mcap['trade_date'].iloc[0])
    compact = re.sub(r'\D', '', latest_date)  # YYYYMMDD for Tushare

    mc_by_ticker = dict(zip(mcap['ticker'], pd.to_numeric(mcap['circ_mv_yi'], errors='coerce')))

    # Last-day % change — not stored per stock in the DB, so one Tushare
    # daily(trade_date=…) call returns pct_chg for every A-share in one shot.
    pct_by_ticker = {}
    try:
        dm.init_tushare()
        if dm.TUSHARE_API is not None:
            d = dm.TUSHARE_API.daily(trade_date=compact, fields='ts_code,pct_chg')
            if d is not None and not d.empty:
                d['ticker'] = d['ts_code'].str[:6]
                pct_by_ticker = dict(zip(d['ticker'],
                                         pd.to_numeric(d['pct_chg'], errors='coerce')))
    except Exception as exc:
        print(f"[heatmap] pct_chg fetch failed: {exc}")

    names = {s['ticker']: s['name'] for s in dm.get_all_stock_basic()}

    rows = []
    for sector, tickers in sector_map.items():
        for t in tickers:
            mc = mc_by_ticker.get(t)
            if mc is None or pd.isna(mc) or mc <= 0:
                continue
            rows.append({
                'sector':  sector,
                'stock':   names.get(t, t),     # Chinese name = readable leaf label
                'ticker':  t,
                'mcap':    float(mc),
                'pct_chg': float(pct_by_ticker.get(t, 0.0) or 0.0),
            })
    if not rows:
        return None, None
    return pd.DataFrame(rows), latest_date


st.subheader("🗺️ 市场热力图 · Market Heatmap")
with st.spinner("加载热力图…"):
    _hm_df, _hm_date = _load_heatmap_data()

if _hm_df is None or _hm_df.empty:
    st.info("热力图数据暂不可用（需要 daily_basic 市值数据）。")
else:
    _fmt_d = (f"{_hm_date[:4]}-{_hm_date[4:6]}-{_hm_date[6:8]}"
              if len(str(_hm_date)) == 8 and str(_hm_date).isdigit() else str(_hm_date))
    _maxabs = max(float(_hm_df['pct_chg'].abs().max()), 1.0)

    # Build the treemap node lists manually so every node (root / sector /
    # stock) carries its OWN numeric pct in customdata. We can't use
    # %{color} in the texttemplate — in a treemap that resolves to the box's
    # colour STRING (e.g. "rgb(233,245,243)"), not the value. Sector & root
    # percentages are CAP-WEIGHTED averages of their constituents (same
    # weighting idea as the nightly PPI, but computed live from the same
    # per-stock pct_chg as the leaves so box and children stay consistent).
    _df = _hm_df.copy()
    _df["_w"] = _df["pct_chg"] * _df["mcap"]
    _sec = _df.groupby("sector").agg(mcap=("mcap", "sum"), _w=("_w", "sum"))
    _sec["pct"] = _sec["_w"] / _sec["mcap"].replace(0, pd.NA)
    _total_cap = float(_df["mcap"].sum())
    _root_pct = float(_df["_w"].sum() / _total_cap) if _total_cap > 0 else 0.0

    ids, labels, parents, values, pcts, tickers = [], [], [], [], [], []
    # root
    ids.append("全市场"); labels.append("全市场"); parents.append("")
    values.append(_total_cap); pcts.append(_root_pct); tickers.append("")
    # sectors
    for sector, row in _sec.iterrows():
        sid = f"sec::{sector}"
        ids.append(sid); labels.append(str(sector)); parents.append("全市场")
        values.append(float(row["mcap"]))
        pcts.append(float(row["pct"]) if pd.notna(row["pct"]) else 0.0)
        tickers.append("")
    # stocks
    for _, r in _df.iterrows():
        sid = f"sec::{r['sector']}"
        ids.append(f"{sid}::{r['ticker']}")
        labels.append(str(r["stock"])); parents.append(sid)
        values.append(float(r["mcap"])); pcts.append(float(r["pct_chg"]))
        tickers.append(str(r["ticker"]))

    customdata = list(zip(pcts, tickers))
    fig_hm = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total",
        customdata=customdata,
        marker=dict(
            colors=pcts,
            colorscale=[[0.0, "#15803d"], [0.5, "#f1f5f9"], [1.0, "#dc2626"]],
            cmid=0, cmin=-_maxabs, cmax=_maxabs,
            line=dict(width=1, color="white"),
            colorbar=dict(title="涨跌%", ticksuffix="%", tickformat=".2f"),
        ),
        texttemplate="%{label}<br>%{customdata[0]:+.2f}%",
        hovertemplate="%{label} %{customdata[1]}<br>"
                      "涨跌 %{customdata[0]:+.2f}%<br>市值 %{value:.0f} 亿<extra></extra>",
        textposition="middle center",
        textfont=dict(size=12),
        tiling=dict(pad=2),
    ))
    fig_hm.update_layout(height=560, margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption(
        f"box 大小 = 流通市值 · 颜色 = {_fmt_d} 当日涨跌（红涨绿跌）· "
        "板块 % 为成分股市值加权平均 · 板块分组与市场广度一致。"
    )

# ===================================================================
# SECTION 1: CSI 300 INDEX WITH VOLATILITY INDICATORS
# ===================================================================
st.subheader("🏦 CSI 300 指数")

# Frequency selector
freq = st.radio(
    "时间周期",
    ["日线", "周线"],
    key="csi300_freq",
    horizontal=True,
    label_visibility="collapsed"
)

with st.spinner("加载 CSI 300 数据..."):
    raw_df = load_csi300_with_regime(freq)

    if raw_df is not None and not raw_df.empty:
        if freq == "周线":
            chart_df = raw_df.tail(52).copy()  # 1 year weekly
            title = "CSI 300 指数 - 周K线 (52周)"
        else:
            chart_df = raw_df.tail(180).copy()  # 6 months daily
            title = "CSI 300 指数 - 日K线 (180天)"
    else:
        chart_df = None

if chart_df is not None and not chart_df.empty:
    # Show current regime
    if 'Market_Regime' in chart_df.columns and chart_df['Market_Regime'].notna().any():
        latest_regime = chart_df['Market_Regime'].dropna().iloc[-1]

        # Color-coded regime display
        if "Low" in latest_regime:
            st.success(f"✅ 当前波动状态: {latest_regime} (低波动)")
        elif "Normal" in latest_regime:
            st.info(f"ℹ️ 当前波动状态: {latest_regime} (正常波动)")
        elif "High" in latest_regime:
            st.warning(f"⚠️ 当前波动状态: {latest_regime} (高波动)")
        else:
            st.error(f"🔴 当前波动状态: {latest_regime} (极端波动)")

    # Prepare dates
    dates = chart_df.index.strftime('%Y-%m-%d').tolist()

    # Calculate tick spacing
    total_dates = len(dates)
    tick_interval = max(1, total_dates // 5)
    tick_vals = dates[::tick_interval][:5]
    tick_text = tick_vals

    # Create chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )

    # Regime shading
    regime_colors = {
        'Low Volatility': 'rgba(34, 197, 94, 0.08)',
        'Normal Volatility': 'rgba(59, 130, 246, 0.05)',
        'High Volatility': 'rgba(255, 110, 0, 0.11)',
        'Extreme Volatility': 'rgba(220, 38, 38, 0.12)'
    }

    if 'Market_Regime' in chart_df.columns and chart_df['Market_Regime'].notna().any():
        df_clean = chart_df.dropna(subset=['Market_Regime']).copy()

        # Segment by regime changes
        changes = df_clean['Market_Regime'].ne(df_clean['Market_Regime'].shift(1))
        change_indices = df_clean.index[changes].tolist()

        if len(change_indices) == 0 or change_indices[0] != df_clean.index[0]:
            change_indices.insert(0, df_clean.index[0])

        # Y ranges
        ymin_price = df_clean['Low'].min() * 0.98
        ymax_price = df_clean['High'].max() * 1.02
        ymax_vol = df_clean['Volume'].max() * 1.05

        for i in range(len(change_indices)):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1] if i + 1 < len(change_indices) else df_clean.index[-1]
            regime = df_clean.loc[start_idx, 'Market_Regime']

            if regime not in regime_colors:
                continue

            start_date = start_idx.strftime('%Y-%m-%d')
            end_date = end_idx.strftime('%Y-%m-%d')

            # Price panel shading
            fig.add_shape(
                type="rect",
                x0=start_date, x1=end_date,
                y0=ymin_price, y1=ymax_price,
                fillcolor=regime_colors[regime],
                line=dict(width=0),
                layer="below",
                row=1, col=1
            )

            # Volume panel shading
            fig.add_shape(
                type="rect",
                x0=start_date, x1=end_date,
                y0=0, y1=ymax_vol,
                fillcolor=regime_colors[regime],
                line=dict(width=0),
                layer="below",
                row=2, col=1
            )

    # Candlestick (Chinese style: red=up, green=down)
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name='CSI 300',
            increasing_line_color='#ef4444',
            decreasing_line_color='#22c55e'
        ),
        row=1, col=1
    )

    # Volume bars
    colors = ['#ef4444' if chart_df['Close'].iloc[i] >= chart_df['Open'].iloc[i] else '#22c55e' 
              for i in range(len(chart_df))]

    fig.add_trace(
        go.Bar(
            x=dates,
            y=chart_df['Volume'],
            name='成交量',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )

    # Layout
    fig.update_layout(
        title=title,
        height=500,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=40)
    )

    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)

    # X-axis
    fig.update_xaxes(
        type='category',
        tickmode='array',
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=0,
        row=2, col=1
    )
    fig.update_xaxes(type='category', showticklabels=False, row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("无法加载 CSI 300 数据")

# ════════════════════════════════════════════════════════════════════════════
# 市场杠杆  ·  MARKET LEVERAGE  (customer margin borrowings across CN·US·JP·KR)
# ════════════════════════════════════════════════════════════════════════════
# Margin debt = money customers borrow to buy stock. Rising = risk appetite /
# potential froth; falling = de-leveraging. China 两融余额 (Tushare) and US
# margin debt (FINRA) are live; Japan/Korea await a clean aggregate source and
# degrade to "—" rather than showing fabricated numbers. Fetchers live in
# market_leverage.py; each is cached at the source's natural refresh cadence.

@st.cache_data(ttl=1800, show_spinner=False)      # A-share margin updates daily
def _lev_china():
    dm.init_tushare()
    return mlev.fetch_china_margin(dm.TUSHARE_API)

@st.cache_data(ttl=6 * 3600, show_spinner=False)  # FINRA is monthly
def _lev_us():
    return mlev.fetch_us_margin()

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _lev_jp():
    return mlev.fetch_japan_margin()

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _lev_kr():
    return mlev.fetch_korea_margin()

st.markdown("---")
st.subheader("💳 市场杠杆 · Market Leverage")
st.caption(
    "Customer margin borrowings — money borrowed to buy stock — a direct "
    "risk-appetite gauge. China 两融余额 (Tushare) and US margin debt (FINRA) "
    "are live; Japan & Korea are pending a clean aggregate source. "
    "Red = rising leverage, green = falling (A-share convention)."
)

_levs = [_lev_china(), _lev_us(), _lev_jp(), _lev_kr()]
_lev_cols = st.columns(4)
for _col, _r in zip(_lev_cols, _levs):
    with _col:
        if not _r["ok"] or _r["latest"] is None:
            st.metric(_r["label"], "—",
                      help=f"⚠️ {_r.get('error') or 'Unavailable.'}  {_r['note']}")
            continue
        _val = f"{_r['latest']:,.1f}{_r['unit']}"
        _delta = None
        if _r["prev"] is not None:
            _delta = f"{_r['latest'] - _r['prev']:+,.1f}{_r['unit']}"
        st.metric(
            _r["label"], _val, delta=_delta,
            delta_color="inverse",   # A-share: red = up (rising leverage)
            help=f"As of {_r['asof']} · {_r['freq']}. {_r['note']}"
                 + (f"  Prev {_r['prev']:,.1f}{_r['unit']}." if _r["prev"] is not None else ""),
        )

# Trend for whichever markets returned a usable series.
_lev_ok = {r["label"]: r for r in _levs
           if r["ok"] and r["series"] is not None and len(r["series"]) >= 2}
if _lev_ok:
    _pick = st.selectbox("📈 杠杆走势 · View leverage trend for:",
                         options=list(_lev_ok.keys()), key="lev_trend_pick")
    _lr = _lev_ok[_pick]
    _ld = _lr["series"]
    _rising = _ld["value"].iloc[-1] >= _ld["value"].iloc[0]
    _lcolor = "#dc2626" if _rising else "#16a34a"   # red up / green down
    _fig_lev = go.Figure()
    _fig_lev.add_trace(go.Scatter(
        x=_ld["period"], y=_ld["value"], mode="lines",
        line=dict(color=_lcolor, width=2),
        hovertemplate="%{x}<br>%{y:,.1f}" + _lr["unit"] + "<extra></extra>",
    ))
    _fig_lev.update_layout(
        title=f"{_pick} · {_lr['freq']}", height=340, template="plotly_white",
        margin=dict(t=50, l=60, r=30, b=50),
        xaxis=dict(tickangle=-45), yaxis_title=_lr["unit"].strip(),
    )
    st.plotly_chart(_fig_lev, use_container_width=True)

# ── AI 中文解读 · plain-Chinese read of the leverage picture ──────────────────
# Digest the (already-fetched) leverage dicts into a compact factual block and
# hand it to the AI for a human-readable Chinese summary. The summary is cached
# on the digest text (+ a manual nonce), so it only calls the API when the data
# actually changes or the user hits regenerate — reruns from other widgets hit
# the cache for free.

def _lev_digest(levs: list[dict]) -> str:
    """Compact, model-friendly factual digest of the leverage series."""
    lines = []
    for r in levs:
        u = r["unit"]
        if not r.get("ok") or r.get("latest") is None or r.get("series") is None:
            lines.append(f"- {r['label']}：暂无数据（{r.get('note', '')}）")
            continue
        s = r["series"]
        latest, prev = r["latest"], r.get("prev")
        first = float(s["value"].iloc[0])
        span = f"{s['period'].iloc[0]}→{s['period'].iloc[-1]}"
        seg = [f"最新 {latest:,.1f}{u}（截至 {r['asof']}，{r['freq']}）"]
        if prev is not None:
            dp = latest - prev
            pp = (dp / prev * 100) if prev else 0.0
            seg.append(f"环比 {dp:+,.1f}{u}（{pp:+.2f}%）")
        if first:
            dw = latest - first
            pw = dw / first * 100
            seg.append(f"区间({span}, {len(s)}期) {dw:+,.1f}{u}（{pw:+.2f}%）")
        lines.append(f"- {r['label']}：" + "；".join(seg))
    return "\n".join(lines)


_LEV_SYS_PROMPT = (
    "你是一位专业的市场分析师。根据提供的各市场“杠杆/保证金余额”（融资余额、"
    "margin debt 等）数据，用简洁、专业、通俗易懂的中文写一段总结（150–250字）。"
    "要点：说明各市场当前杠杆水平及其环比/区间变化；解读这些变化所反映的风险偏好"
    "与市场情绪；对比不同市场之间的差异。A股惯例：融资余额上升通常代表风险偏好上升。"
    "若某市场数据暂缺，用一句话说明即可。只做客观描述与解读，不要给出任何具体的"
    "买入/卖出或投资建议。直接输出总结正文，不要加标题或免责声明。"
)


@st.cache_data(ttl=1800, show_spinner=False)
def _lev_ai_summary(digest: str, _nonce: int) -> str:
    """Cached AI summary. `_nonce` lets the regenerate button bust the cache."""
    return ai_client.call_text(_LEV_SYS_PROMPT, digest, max_tokens=2000, temperature=0.3)


_lev_has_data = any(r.get("ok") for r in _levs)
if _lev_has_data:
    _lc1, _lc2 = st.columns([6, 1])
    _lc1.markdown("**🧠 AI 杠杆解读**")
    if _lc2.button("🔄 重新生成", key="lev_ai_regen", use_container_width=True):
        st.session_state["lev_ai_nonce"] = st.session_state.get("lev_ai_nonce", 0) + 1

    _digest = _lev_digest(_levs)
    try:
        with st.spinner("AI 正在解读市场杠杆数据…"):
            _summary = _lev_ai_summary(_digest, st.session_state.get("lev_ai_nonce", 0))
        with st.container(border=True):
            st.markdown(_summary)
        st.caption("由 AI 根据上方杠杆数据自动生成，仅供参考，不构成投资建议。")
    except Exception as _exc:
        st.warning(f"AI 解读暂不可用：{_exc}")
        with st.expander("查看原始杠杆数据摘要"):
            st.text(_digest)

# ════════════════════════════════════════════════════════════════════════════
# 龙虎榜  ·  TOP LIST  (Tushare pro.top_list — daily abnormal-trading list)
# ════════════════════════════════════════════════════════════════════════════
# Each ticker is a clickable anchor → Lead-Lag Analysis page with that
# stock preloaded as the topic. Middle-click / right-click → "Open in new
# tab" works natively because it's a real <a href>, not a Streamlit button.

st.markdown("---")
st.subheader("🐉 龙虎榜 · Top Trading List")
st.caption(
    "Stocks that triggered Shanghai/Shenzhen exchange's abnormal-trading "
    "rules (large pct move / turnover spike / amplitude). Click any ticker "
    "to deep-dive into Lead-Lag Analysis with that stock as the topic."
)


@st.cache_data(ttl=4 * 3600, show_spinner=False)
def _load_top_list_latest(max_lookback_days: int = 10):
    """
    Pull the most recent day's 龙虎榜. Walks back day-by-day for up to
    max_lookback_days because weekends/holidays return an empty frame.
    Cached 4h — the data only updates once per evening after close.
    Returns (df, trade_date_yyyymmdd) or (None, None) on failure.
    """
    try:
        dm.init_tushare()
        if dm.TUSHARE_API is None:
            return None, None
        bj = datetime.now(ZoneInfo("Asia/Shanghai"))
        for delta in range(max_lookback_days):
            d = (bj - timedelta(days=delta)).strftime("%Y%m%d")
            try:
                df = dm.TUSHARE_API.top_list(trade_date=d)
            except Exception:
                continue
            if df is not None and not df.empty:
                return df, d
        return None, None
    except Exception:
        return None, None


with st.spinner("Loading 龙虎榜 data…"):
    tl_df, tl_date = _load_top_list_latest()

if tl_df is None or tl_df.empty:
    st.info("📭 No 龙虎榜 data available for the past 10 days.")
else:
    tl_date_fmt = f"{tl_date[:4]}-{tl_date[4:6]}-{tl_date[6:8]}"

    # Sort by net buying amount descending; show top 30
    sort_col = "net_amount" if "net_amount" in tl_df.columns else "amount"
    tl_df = tl_df.sort_values(sort_col, ascending=False).head(30)

    n_buys  = int((tl_df.get("net_amount", pd.Series(dtype=float)) > 0).sum())
    n_sells = int((tl_df.get("net_amount", pd.Series(dtype=float)) < 0).sum())

    hdr_l, hdr_r = st.columns([3, 2])
    hdr_l.markdown(
        f"**Trade date:** {tl_date_fmt}  ·  **Listings shown:** {len(tl_df)}  "
        f"(net buy: **{n_buys}**, net sell: **{n_sells}**)"
    )

    # Build the HTML table — tickers as anchor links to /lead-lag?ticker=…
    rows_html = ""
    for _, row in tl_df.iterrows():
        ts_code = str(row.get("ts_code", ""))
        ticker6 = ts_code[:6] if ts_code else ""
        name    = row.get("name", "") or ""
        close   = row.get("close", None)
        pct     = row.get("pct_change", None)
        net_amt = row.get("net_amount", None)
        net_rt  = row.get("net_rate", None)
        reason  = row.get("reason", "") or ""

        # A-share colour: red = up, green = down
        if pd.notna(pct) and pct > 0:
            pct_style = "color:#dc2626;font-weight:600"
        elif pd.notna(pct) and pct < 0:
            pct_style = "color:#16a34a;font-weight:600"
        else:
            pct_style = "color:#6b7280"

        if pd.notna(net_amt):
            net_style = "color:#dc2626" if net_amt > 0 else "color:#16a34a" if net_amt < 0 else "color:#6b7280"
            net_str   = f"{net_amt / 1e4:+,.0f} 万"
        else:
            net_style, net_str = "color:#6b7280", "—"

        close_str = f"¥{close:.2f}" if pd.notna(close) else "—"
        pct_str   = f"{pct:+.2f}%"  if pd.notna(pct)   else "—"
        rate_str  = f"{net_rt:+.2f}%" if pd.notna(net_rt) else "—"

        rows_html += f"""
        <tr>
          <td style="padding:6px 10px">
            <a href="/lead-lag?ticker={ticker6}" target="_self"
               style="color:#7c3aed;text-decoration:underline;
                      text-underline-offset:3px;font-weight:600;
                      font-family:ui-monospace,monospace">{ticker6}</a>
          </td>
          <td style="padding:6px 10px">{name}</td>
          <td style="padding:6px 10px;text-align:right">{close_str}</td>
          <td style="padding:6px 10px;text-align:right;{pct_style}">{pct_str}</td>
          <td style="padding:6px 10px;text-align:right;{net_style};font-variant-numeric:tabular-nums">{net_str}</td>
          <td style="padding:6px 10px;text-align:right;{net_style}">{rate_str}</td>
          <td style="padding:6px 10px;font-size:12px;color:#64748b">{reason}</td>
        </tr>"""

    # Fixed height ≈ header + 8 rows; rest scrolls. Header stays sticky on
    # top while scrolling so users always know which column is which.
    table_html = f"""
    <div style="max-height:360px;overflow-y:auto;overflow-x:auto;
                border:1px solid #e5e7eb;border-radius:8px;margin-top:8px">
      <table style="width:100%;border-collapse:collapse;font-size:13px">
        <thead style="position:sticky;top:0;z-index:1">
          <tr style="background:#f8fafc;border-bottom:1px solid #e5e7eb">
            <th style="padding:8px 10px;text-align:left;background:#f8fafc">Ticker</th>
            <th style="padding:8px 10px;text-align:left;background:#f8fafc">Name</th>
            <th style="padding:8px 10px;text-align:right;background:#f8fafc">Close</th>
            <th style="padding:8px 10px;text-align:right;background:#f8fafc">% Chg</th>
            <th style="padding:8px 10px;text-align:right;background:#f8fafc">Net 龙虎榜</th>
            <th style="padding:8px 10px;text-align:right;background:#f8fafc">Net Rate</th>
            <th style="padding:8px 10px;text-align:left;background:#f8fafc">上榜理由 · Reason</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""

    st.markdown(table_html, unsafe_allow_html=True)
    st.caption(
        "Sorted by **net 龙虎榜 amount** (institutional / large-order net flow on the day) descending. "
        "Click a ticker to open Lead-Lag Analysis with that stock as topic — middle/right-click to open in a new tab."
    )

# Market Breadth History
st.markdown("---")
st.subheader("📊 市场宽度历史 (Market Breadth History)")
st.caption("各板块中股价高于MA20的股票占比 - 数据来自数据库")

# Load breadth data from database (single query!)
breadth_df = dm.load_market_breadth_from_db()

if breadth_df is None or breadth_df.empty:
    st.warning("数据库中没有市场宽度数据")
else:
    # Get last 60 dates
    unique_dates = breadth_df.index.sort_values(ascending=False)[:60].tolist()

    if len(unique_dates) == 0:
        st.warning("没有历史宽度数据")
    else:
        # Pagination setup
        DAYS_PER_PAGE = 10
        total_pages = (len(unique_dates) + DAYS_PER_PAGE - 1) // DAYS_PER_PAGE

        if 'breadth_page' not in st.session_state:
            st.session_state.breadth_page = 0

        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            if st.button("⬅ 前10天", disabled=(st.session_state.breadth_page >= total_pages - 1)):
                st.session_state.breadth_page += 1
                st.rerun()

        with col2:
            start_idx = st.session_state.breadth_page * DAYS_PER_PAGE
            page_end = min(start_idx + DAYS_PER_PAGE, len(unique_dates))
            date_range = f"{unique_dates[page_end-1].strftime('%m/%d')} 至 {unique_dates[start_idx].strftime('%m/%d')}"
            st.markdown(
                f"<center><b>第 {st.session_state.breadth_page + 1}/{total_pages} 页</b><br>{date_range}</center>",
                unsafe_allow_html=True
            )

        with col3:
            if st.button("后10天 ➡", disabled=(st.session_state.breadth_page == 0)):
                st.session_state.breadth_page -= 1
                st.rerun()

        # Get dates for current page
        end_idx = start_idx + DAYS_PER_PAGE
        page_dates = unique_dates[start_idx:end_idx]
        page_dates = page_dates[::-1]  # Reverse: Latest dates on RIGHT

        # Filter breadth_df for page dates
        page_df = breadth_df.loc[page_dates].copy()

        # Transpose so dates are columns, sectors are rows
        page_df = page_df.T
        page_df.columns = [d.strftime('%m/%d') for d in page_df.columns]
        page_df = page_df.reset_index()
        page_df = page_df.rename(columns={'index': '板块'})

        # Styling
        def style_breadth_cell(val):
            if pd.isna(val):
                return ''
            if val >= 0.5:
                return 'color: #b91c1c; background-color: #fee2e2; font-weight: 600'
            else:
                return 'color: #15803d; background-color: #dcfce7; font-weight: 600'

        date_cols = [col for col in page_df.columns if col != '板块']

        def format_breadth(val):
            if pd.isna(val):
                return ''
            return f'{val*100:.0f}%'

        styled = page_df.style            .map(style_breadth_cell, subset=date_cols)            .format(format_breadth, subset=date_cols)

        st.dataframe(styled, hide_index=True, use_container_width=True, height=600)
        st.caption("🟢 绿色 <50%: 多数股票低于MA20 (机会). 🔴 红色 >=50%: 多数股票高于MA20 (过热).")

# ============================================================================
# SECTOR ROTATION DETECTION MODULE (DAILY + ADJUSTABLE ROLLING WINDOW)
# Add this at the bottom of your sector_dashboard.py
# ============================================================================

st.markdown("---")
st.subheader("🔄 Sector Rotation Detection")

# Define rotation pairs
ROTATION_PAIRS = {
    'Cyclical vs Defensive': {
        'cyclical': '399395.SZ',  # 国证消费 CNI Consumer
        'defensive': '399396.SZ',  # 国证食品 CNI Food & Beverage
        'cyclical_name': '消费',
        'defensive_name': '食品饮料'
    },
    'Tech vs Utilities': {
        'cyclical': '399932.SZ',  # 中证信息 CSI Info Tech
        'defensive': '000991.SH',  # 全指公用 CSI Utilities
        'cyclical_name': '信息技术',
        'defensive_name': '公用事业'
    },
    'Financial vs Industrial': {
        'cyclical': '399975.SZ',  # 证券公司 CSI Securities
        'defensive': '000993.SH',  # 全指工业 CSI Industrials
        'cyclical_name': '证券',
        'defensive_name': '工业'
    },
    'Healthcare vs Energy': {
        'cyclical': '399989.SZ',  # 中证医疗 CSI Healthcare
        'defensive': '000992.SH',  # 全指能源 CSI Energy
        'cyclical_name': '医疗',
        'defensive_name': '能源'
    }
}

# ============================================================================
# ADJUSTABLE ROLLING WINDOW SELECTOR
# ============================================================================
col_select1, col_select2 = st.columns([1, 3])

with col_select1:
    rolling_window = st.selectbox(
        "Rolling Window (Days):",
        options=[5, 10, 15, 30],
        index=1,  # Default to 10 days
        key="rotation_rolling_window"
    )

with col_select2:
    st.caption(f"Using {rolling_window}-day rolling correlation to measure sector rotation dynamics")

def calculate_rotation_metrics_daily(ts_code1, ts_code2, rolling_days=10, lookback_days=400):
    """
    Calculate rotation metrics between two indices using DAILY data
    Returns: correlation, ratio_change, status, correlation_history
    """
    import tushare as ts
    from datetime import datetime, timedelta

    try:
        pro = ts.pro_api()

        # ✅ FIX: Use Beijing time for end_date
        CHINA_TZ = ZoneInfo("Asia/Shanghai")
        beijing_now = datetime.now(CHINA_TZ)
        end_date = datetime.now(CHINA_TZ).strftime('%Y%m%d')
        start_date = (datetime.now(CHINA_TZ) - timedelta(days=lookback_days)).strftime('%Y%m%d')

        df1 = pro.index_daily(ts_code=ts_code1, start_date=start_date, end_date=end_date)
        df2 = pro.index_daily(ts_code=ts_code2, start_date=start_date, end_date=end_date)

        if df1 is None or df2 is None or df1.empty or df2.empty:
            return None, None, "数据不足", None

        # Sort by date and calculate returns
        df1 = df1.sort_values('trade_date')
        df2 = df2.sort_values('trade_date')


        df1['returns'] = df1['close'].pct_change()
        df2['returns'] = df2['close'].pct_change()

        # Merge data
        merged = pd.merge(df1[['trade_date', 'returns', 'close']], 
                         df2[['trade_date', 'returns', 'close']], 
                         on='trade_date', 
                         suffixes=('_1', '_2'))

        if len(merged) < rolling_days:
            return None, None, "数据不足", None

        # Calculate rolling correlation
        merged['correlation'] = merged['returns_1'].rolling(window=rolling_days).corr(merged['returns_2'])

        # Get latest correlation
        correlation = merged['returns_1'].tail(rolling_days).corr(merged['returns_2'].tail(rolling_days))

        # Get correlation history for chart (last 252 days ~ 1 year of trading days)
        correlation_history = merged[['trade_date', 'correlation']].tail(252).copy()


        # Calculate relative strength ratio change (last 60 days)
        ratio_start = merged['close_1'].iloc[-60] / merged['close_2'].iloc[-60] if len(merged) >= 60 else merged['close_1'].iloc[0] / merged['close_2'].iloc[0]
        ratio_end = merged['close_1'].iloc[-1] / merged['close_2'].iloc[-1]
        ratio_change = ((ratio_end - ratio_start) / ratio_start) * 100

        # Determine rotation status
        if correlation < 0.3:
            status = "🔴 高度轮动"
        elif correlation < 0.5:
            status = "🟡 中度轮动"
        elif correlation < 0.7:
            status = "🟢 低度轮动"
        else:
            status = "⚪ 同步移动"

        return correlation, ratio_change, status, correlation_history

    except Exception as e:
        return None, None, f"错误: {str(e)}", None


# Calculate metrics for all pairs
rotation_results = []
correlation_histories = {}

with st.spinner(f"计算板块轮动指标（{rolling_window}日滚动相关系数）..."):
    for pair_name, pair_info in ROTATION_PAIRS.items():
        correlation, ratio_change, status, corr_history = calculate_rotation_metrics_daily(
            pair_info['cyclical'],
            pair_info['defensive'],
            rolling_days=rolling_window,
            lookback_days=400  # Get ~1+ year of data
        )

        # Store correlation history for chart
        if corr_history is not None:
            correlation_histories[pair_name] = corr_history

        # Determine which is leading
        if ratio_change is not None:
            if ratio_change > 5:
                leader = f"➡️ {pair_info['cyclical_name']} 强"
            elif ratio_change < -5:
                leader = f"⬅️ {pair_info['defensive_name']} 强"
            else:
                leader = "⚖️ 均衡"
        else:
            leader = "N/A"

        rotation_results.append({
            '板块对': pair_name,
            '周期/防御': f"{pair_info['cyclical_name']} vs {pair_info['defensive_name']}",
            '相关系数': f"{correlation:.2f}" if correlation is not None else "N/A",
            '轮动状态': status,
            '相对强度': f"{ratio_change:+.1f}%" if ratio_change is not None else "N/A",
            '领先板块': leader
        })

# Display results table
rotation_df = pd.DataFrame(rotation_results)

# Style the table
def style_rotation_status(val):
    if "高度轮动" in val:
        return "color: #b91c1c; background-color: #fee2e2; font-weight: 600"
    elif "中度轮动" in val:
        return "color: #d97706; background-color: #fef3c7; font-weight: 600"
    elif "低度轮动" in val:
        return "color: #15803d; background-color: #dcfce7; font-weight: 600"
    elif "同步移动" in val:
        return "color: #6b7280; background-color: #f3f4f6; font-weight: 600"
    return ""

def style_correlation(val):
    if val == "N/A":
        return ""
    try:
        corr = float(val)
        if corr < 0.3:
            return "color: #b91c1c; font-weight: 600"
        elif corr < 0.5:
            return "color: #d97706; font-weight: 600"
        elif corr < 0.7:
            return "color: #15803d; font-weight: 600"
        else:
            return "color: #6b7280; font-weight: 600"
    except:
        return ""

styled_rotation = rotation_df.style.map(style_rotation_status, subset=['轮动状态']).map(style_correlation, subset=['相关系数'])

st.dataframe(styled_rotation, hide_index=True, use_container_width=True, height=220)

# ============================================================================
# CORRELATION CHART VISUALIZATION (4 pairs, 1 year of data)
# ============================================================================

if correlation_histories:
    st.markdown("---")
    st.subheader(f"📈 Correlation Trends ({rolling_window}-Day Rolling - Last Year)")

    # Create plotly figure with 4 subplots (2x2 grid)
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(correlation_histories.keys()),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Color mapping for correlation levels
    def get_color(corr_val):
        if pd.isna(corr_val):
            return '#9ca3af'
        if corr_val < 0.3:
            return '#ef4444'  # Red - High rotation
        elif corr_val < 0.5:
            return '#f59e0b'  # Orange - Medium rotation
        elif corr_val < 0.7:
            return '#10b981'  # Green - Low rotation
        else:
            return '#6b7280'  # Gray - Moving together

    # Plot each pair
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, (pair_name, corr_history) in enumerate(correlation_histories.items()):
        row, col = positions[idx]


        # Prepare data - show data every 5 days to avoid overcrowding
        # total_points = len(corr_history)
        # step = max(1, total_points // 50)  # Show 50 bars max
        # sampled_history = corr_history.iloc[::step].copy()

        # # NEW (CORRECT):
        # dates = pd.to_datetime(sampled_history['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d').tolist()

        dates = pd.to_datetime(corr_history['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d').tolist()
        correlations = corr_history['correlation'].tolist()


        # Get colors for each bar
        colors = [get_color(c) for c in correlations]

        # # Add bar chart
        # fig.add_trace(
        #     go.Bar(
        #         x=dates,
        #         y=correlations,
        #         marker_color=colors,
        #         name=pair_name,
        #         showlegend=False,
        #         hovertemplate='<b>%{x}</b><br>Correlation: %{y:.2f}<extra></extra>'
        #     ),
        #     row=row, col=col
        # )

        # Create line chart with gradient colors
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=correlations,
                mode='lines',
                fill='tozeroy',
                line=dict(color='rgb(59, 130, 246)', width=2),
                name=pair_name,
                showlegend=False,
                hovertemplate='<b>Date: %{x}</b><br>' + f'{rolling_window}-day Rolling Correlation: ' + '%{y:.4f}<br><extra></extra>'
            ),
            row=row, col=col
        )



        # Add horizontal reference lines
        fig.add_hline(y=0.3, line_dash="dash", line_color="red", line_width=1, 
                     opacity=0.3, row=row, col=col)
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", line_width=1, 
                     opacity=0.3, row=row, col=col)
        fig.add_hline(y=0.7, line_dash="dash", line_color="green", line_width=1, 
                     opacity=0.3, row=row, col=col)

        # Update y-axis range
        fig.update_yaxes(range=[-0.2, 1.0], row=row, col=col)

    # Update layout
    fig.update_layout(
        height=600,
        template='plotly_white',
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    # Rotate x-axis labels and reduce font size
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=8))

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown(f"""
    **📊 Correlation Levels ({rolling_window}-day rolling):**
    🔴 < 0.3 = High Rotation | 🟡 0.3-0.5 = Medium Rotation | 🟢 0.5-0.7 = Low Rotation | ⚪ > 0.7 = Moving Together
    """)

# ============================================================================
# MARKET CONCLUSION
# ============================================================================

# Calculate overall rotation intensity
correlations = [float(r['相关系数']) for r in rotation_results if r['相关系数'] != "N/A"]
if correlations:
    avg_correlation = sum(correlations) / len(correlations)
    high_rotation_count = sum(1 for c in correlations if c < 0.3)

    st.markdown("---")

    if avg_correlation < 0.4:
        market_status = "🔴 **市场处于高轮动期**"
        interpretation = "各板块走势分化明显，建议精选优势板块，避免弱势板块。"
    elif avg_correlation < 0.6:
        market_status = "🟡 **市场处于中度轮动期**"
        interpretation = "板块有一定分化，存在轮动机会，可考虑配置多个板块。"
    else:
        market_status = "🟢 **市场同步移动**"
        interpretation = "各板块走势趋同，市场趋势明确，建议跟随大盘方向。"

    st.markdown(f"### {market_status}")
    st.info(f"📊 **平均相关系数**: {avg_correlation:.2f} | **高轮动对数**: {high_rotation_count}/4\n\n{interpretation}")

    # Additional insights
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💡 轮动状态说明**")
        st.markdown("""
        - 🔴 **高度轮动** (相关系数 < 0.3): 板块严重分化
        - 🟡 **中度轮动** (0.3-0.5): 板块有所分化
        - 🟢 **低度轮动** (0.5-0.7): 板块轻微分化
        - ⚪ **同步移动** (> 0.7): 板块走势一致
        """)

    with col2:
        st.markdown("**🎯 投资建议**")
        if avg_correlation < 0.4:
            st.markdown("""
            - ✅ 精选领先板块，集中投资
            - ✅ 避免落后板块
            - ✅ 灵活调仓，跟随轮动节奏
            """)
        elif avg_correlation < 0.6:
            st.markdown("""
            - ✅ 均衡配置多个板块
            - ✅ 关注轮动机会
            - ✅ 适度分散风险
            """)
        else:
            st.markdown("""
            - ✅ 跟随市场整体方向
            - ✅ 配置指数型基金
            - ✅ 减少频繁调仓
            """)

st.caption(f"📅 数据基于过去一年日度数据 | 相关系数基于{rolling_window}日滚动计算 | 相对强度基于近60日变化")


# ============================================================================
# STATISTICAL WYCKOFF PHASE DETECTION (CSI 300)
# Append this to the bottom of sector_dashboard.py
# ============================================================================

st.markdown("---")
st.subheader("📊 Statistical Wyckoff Phase Detection (CSI 300)")
st.caption("A pure quantitative approach using 120-day rolling channels, return volatility, and volume Z-scores to mathematically define market regimes without moving average lag.")
import numpy as np

with st.spinner("Calculating Statistical Wyckoff Phases..."):
    # Fetch 500 days to ensure enough history for 120-day rolling windows and means
    wyckoff_stat_df = dm.get_index_data_live('000300.SH', lookback_days=500, freq='daily')

    if wyckoff_stat_df is not None and not wyckoff_stat_df.empty:
        df = wyckoff_stat_df.copy()
        
        lookback = 120 # Approx 6 months of trading days
        
        # 1. Market Structure: Donchian Channels & Range Positioning
        df['Max_120'] = df['High'].rolling(window=lookback).max()
        df['Min_120'] = df['Low'].rolling(window=lookback).min()
        df['Range_120'] = df['Max_120'] - df['Min_120']
        # Where is the price currently sitting within its 6-month range? (0.0 to 1.0)
        df['Position_120'] = (df['Close'] - df['Min_120']) / df['Range_120']
        
        # 2. Volatility: Rolling 20-day standard deviation vs historical baseline
        df['Returns'] = df['Close'].pct_change()
        df['Volat_20'] = df['Returns'].rolling(window=20).std()
        df['Volat_Baseline'] = df['Volat_20'].rolling(window=lookback).mean()
        
        # 3. Clean Volume Metric (Isolated for this module)
        df['Vol_Mean_60'] = df['Volume'].rolling(window=60).mean()
        df['Vol_Std_60'] = df['Volume'].rolling(window=60).std()
        df['Vol_Z'] = (df['Volume'] - df['Vol_Mean_60']) / df['Vol_Std_60']
        
        # 4. Statistical Phase Conditions
        conditions = [
            # MARKUP: Upper quartile of 6-month range, trending up (Close > 20-day mean)
            (df['Position_120'] > 0.75) & (df['Close'] > df['Close'].rolling(20).mean()),
            
            # MARKDOWN: Lower quartile of 6-month range, trending down
            (df['Position_120'] < 0.25) & (df['Close'] < df['Close'].rolling(20).mean()),
            
            # DISTRIBUTION: Upper half of range, but high volatility (churn) and failing to push new highs
            (df['Position_120'] >= 0.5) & (df['Volat_20'] > df['Volat_Baseline']),
            
            # ACCUMULATION: Lower half of range, low volatility (quiet absorption)
            (df['Position_120'] < 0.5) & (df['Volat_20'] <= df['Volat_Baseline'])
        ]
        
        choices = ['Markup', 'Markdown', 'Distribution', 'Accumulation']
        df['Wyckoff_Phase'] = np.select(conditions, choices, default='Transition')
        
        # Drop rows where our longest baseline hasn't calculated yet
        plot_df = df.dropna(subset=['Volat_Baseline']).copy()
        
        # Focus chart on the last 180 trading days
        plot_df = plot_df.tail(180)

        if not plot_df.empty:
            current_phase = plot_df['Wyckoff_Phase'].iloc[-1]
            current_pos = plot_df['Position_120'].iloc[-1]
            current_vol_z = plot_df['Vol_Z'].iloc[-1]
            
            # Display current metrics
            col_s1, col_s2, col_s3 = st.columns([1.5, 1, 1])
            with col_s1:
                st.metric("Current Phase (Statistical)", current_phase)
            with col_s2:
                st.metric("Range Position (120d)", f"{current_pos * 100:.1f}%")
            with col_s3:
                st.metric("Volume Z-Score", f"{current_vol_z:.2f}")

            # Contextual explainer
            if current_phase == 'Accumulation':
                st.info("🟦 **Accumulation:** Price is compressed in the lower half of its 6-month range. Volatility is below average, indicating quiet institutional absorption.")
            elif current_phase == 'Markup':
                st.success("🟩 **Markup:** Price has broken into the top 25% of its 6-month range. Upward momentum is statistically confirmed.")
            elif current_phase == 'Distribution':
                st.warning("🟧 **Distribution:** Price is high, but volatility is spiking above baseline. High churn indicates potential institutional selling.")
            elif current_phase == 'Markdown':
                st.error("🟥 **Markdown:** Price has collapsed into the bottom 25% of its range. Downward momentum is dominating.")
            else:
                st.info("⬜ **Transition:** The market is caught between defined statistical states.")

            # Create Plotly Chart
            fig_stat = go.Figure()

            # Phase colors
            phase_colors = {
                'Accumulation': 'rgba(59, 130, 246, 0.1)',
                'Markup': 'rgba(34, 197, 94, 0.1)',
                'Distribution': 'rgba(245, 158, 11, 0.1)',
                'Markdown': 'rgba(239, 68, 68, 0.1)',
                'Transition': 'rgba(156, 163, 175, 0.1)'
            }

            # Add background shading
            changes = plot_df['Wyckoff_Phase'].ne(plot_df['Wyckoff_Phase'].shift())
            change_indices = plot_df.index[changes].tolist()
            
            if len(change_indices) == 0 or change_indices[0] != plot_df.index[0]:
                change_indices.insert(0, plot_df.index[0])

            ymin = plot_df['Low'].min() * 0.95
            ymax = plot_df['High'].max() * 1.05

            dates_str = plot_df.index.strftime('%Y-%m-%d').tolist()

            for i in range(len(change_indices)):
                start_idx = change_indices[i]
                end_idx = change_indices[i + 1] if i + 1 < len(change_indices) else plot_df.index[-1]
                phase = plot_df.loc[start_idx, 'Wyckoff_Phase']

                if phase in phase_colors:
                    fig_stat.add_shape(
                        type="rect",
                        x0=start_idx.strftime('%Y-%m-%d'), 
                        x1=end_idx.strftime('%Y-%m-%d'),
                        y0=ymin, y1=ymax,
                        fillcolor=phase_colors[phase],
                        line=dict(width=0),
                        layer="below"
                    )

            # Candlestick Trace
            fig_stat.add_trace(go.Candlestick(
                x=dates_str,
                open=plot_df['Open'], high=plot_df['High'],
                low=plot_df['Low'], close=plot_df['Close'],
                name='CSI 300',
                increasing_line_color='#ef4444', decreasing_line_color='#22c55e'
            ))

            # 120-Day Donchian Channels (Max/Min)
            fig_stat.add_trace(go.Scatter(
                x=dates_str, y=plot_df['Max_120'],
                mode='lines', name='120d High',
                line=dict(color='rgba(156, 163, 175, 0.6)', width=1, dash='dash')
            ))
            
            fig_stat.add_trace(go.Scatter(
                x=dates_str, y=plot_df['Min_120'],
                mode='lines', name='120d Low',
                line=dict(color='rgba(156, 163, 175, 0.6)', width=1, dash='dash')
            ))

            # Layout updates
            fig_stat.update_layout(
                title='CSI 300 - Statistical Regimes & 120-Day Channels',
                height=500,
                template='plotly_white',
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # Fix x-axis categorical spacing for weekends
            tick_interval = max(1, len(dates_str) // 6)
            fig_stat.update_xaxes(
                type='category',
                tickmode='array',
                tickvals=dates_str[::tick_interval],
                ticktext=dates_str[::tick_interval]
            )

            st.plotly_chart(fig_stat, use_container_width=True)

    else:
        st.error("Failed to load sufficient data for Statistical Wyckoff calculation.")