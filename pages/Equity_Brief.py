"""
Equity_Brief.py — AURX-style printable equity brief for any A-share stock.

Combines Tushare quantitative data (fundamentals + technicals) with
AI-generated qualitative content (PESTEL / Porter's / SWOT / competitors).

Heavy custom CSS replaces Streamlit's default chrome — most content is
rendered as raw HTML cards/tables to match the AURX print aesthetic.
"""

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import auth_manager
import data_manager
import equity_brief
import trading_strategy
from analysis_engine import run_single_stock_analysis

auth_manager.require_login()
equity_brief.ensure_equity_brief_cache_table()

st.set_page_config(page_title="Equity Brief | 个股研报", page_icon="📄", layout="wide")


# ══════════════════════════════════════════════════════════════════════════════
# AURX-STYLE CSS — overrides Streamlit chrome, supplies cards, KPI tiles, tables
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
:root {
    --bg:        #f7f3ec;
    --bg-2:      #efe9df;
    --bg-3:      #e6dfd2;
    --ink:       #1a1916;
    --ink-2:     #4d4942;
    --ink-3:     #807a70;
    --rule:      #d6cebe;
    --rule-2:    #b8b09f;
    --primary:   #2563a8;
    --primary-2: #d6e4f4;
    --pos:       #2f8a4f;
    --pos-2:     #d8eedb;
    --neg:       #c6432a;
    --neg-2:     #f3dad2;
    --warn:      #b8800f;
    --salmon:    #f3dec5;
}

/* Hide Streamlit chrome we don't want */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.stDeployButton { display: none; }

/* Page body — Newsreader-ish serif feel via Georgia fallback */
.eb-root { font-family: Georgia, "Newsreader", serif; color: var(--ink);
    background: var(--bg); padding: 0 8px; }
.eb-num  { font-family: "JetBrains Mono", ui-monospace, monospace;
    font-feature-settings: "tnum","zero"; }

.eb-eyebrow {
    font-family: ui-monospace, monospace;
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin: 4px 0 10px 0;
}
.eb-h1 { font-size: 44px; font-weight: 600; line-height: 1.1; margin: 0 0 8px 0;
    letter-spacing: -0.01em; }
.eb-h2 { font-size: 28px; font-weight: 600; line-height: 1.2; margin: 0 0 14px 0;
    letter-spacing: -0.01em; border-bottom: 1px solid var(--rule); padding-bottom: 8px; }
.eb-h3 { font-size: 18px; font-weight: 600; margin: 0 0 8px 0; }
.eb-kicker { font-size: 13px; color: var(--ink-2); margin: -4px 0 18px 0;
    max-width: 720px; line-height: 1.55; }

.eb-section { padding: 28px 0; border-bottom: 1px solid var(--rule); }
.eb-section:last-child { border-bottom: 0; }

/* Cards & tiles */
.eb-card { background: #fff; border: 1px solid var(--rule); border-radius: 4px;
    padding: 16px 18px; }

.eb-kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.eb-kpi { background: #fff; border: 1px solid var(--rule); border-radius: 4px;
    padding: 12px 14px; }
.eb-kpi-label { font-family: ui-monospace, monospace; font-size: 10.5px;
    color: var(--ink-3); text-transform: uppercase; letter-spacing: 0.12em; }
.eb-kpi-value { font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 22px; font-weight: 500; color: var(--ink);
    font-feature-settings: "tnum"; margin-top: 4px; }
.eb-kpi-sub { font-size: 11.5px; color: var(--ink-2); margin-top: 2px; }

.eb-pill { display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-family: ui-monospace, monospace; font-size: 10.5px;
    border: 1px solid var(--rule-2); color: var(--ink-2); background: #fff;
    letter-spacing: 0.06em; text-transform: uppercase; }
.eb-pill.pos { color: var(--pos); border-color: var(--pos); background: var(--pos-2); }
.eb-pill.neg { color: var(--neg); border-color: var(--neg); background: var(--neg-2); }
.eb-pill.warn { color: var(--warn); border-color: var(--warn); background: var(--salmon); }

/* Tables */
.eb-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.eb-table th { text-align: left; font-family: ui-monospace, monospace;
    font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--ink-3); border-bottom: 1px solid var(--rule);
    padding: 8px 10px; font-weight: 500; }
.eb-table td { padding: 8px 10px; border-bottom: 1px solid var(--rule);
    color: var(--ink); }
.eb-table td.num { font-family: "JetBrains Mono", ui-monospace, monospace;
    font-feature-settings: "tnum"; text-align: right; }
.eb-table tr:last-child td { border-bottom: 0; }
.eb-table .pos { color: var(--pos); }
.eb-table .neg { color: var(--neg); }

/* Two-column qualitative cards */
.eb-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.eb-2col-card { background: #fff; border: 1px solid var(--rule); border-radius: 4px;
    padding: 14px 16px; }
.eb-2col-card h4 { margin: 0 0 8px 0; font-family: ui-monospace, monospace;
    font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--ink-3); font-weight: 500; }
.eb-2col-card ul { margin: 0; padding-left: 18px; font-size: 13px; line-height: 1.55; }
.eb-2col-card li { margin-bottom: 6px; }

/* Force score bars */
.eb-force { display: grid; grid-template-columns: 130px 100px 1fr; gap: 12px;
    align-items: center; padding: 8px 0; border-bottom: 1px solid var(--rule); }
.eb-force:last-child { border-bottom: 0; }
.eb-force-label { font-family: ui-monospace, monospace; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.1em; color: var(--ink-2); }
.eb-force-bar { display: flex; gap: 3px; }
.eb-force-bar .seg { width: 16px; height: 8px; background: var(--bg-3);
    border-radius: 1px; }
.eb-force-bar .seg.on { background: var(--ink); }

/* Print styles — match AURX brief aesthetic for PDF export */
@media print {
    @page { size: Letter portrait; margin: 0.45in; }
    .stApp > header, [data-testid="stSidebar"], .stButton, .stDownloadButton,
    .stSelectbox, .stTextInput, .stForm, .stAlert, .stSpinner,
    div[data-testid="stToolbar"], iframe[title="streamlit_plotly_events"] {
        display: none !important;
    }
    .eb-section { break-inside: avoid; page-break-inside: avoid; }
    .eb-section + .eb-section { break-before: page; }
    .eb-h1 { font-size: 36px !important; }
    .eb-h2 { font-size: 22px !important; }
    body { background: var(--bg) !important;
        -webkit-print-color-adjust: exact !important; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_yi(v):
    """Format a number assumed to be in 亿元."""
    if v is None or pd.isna(v):
        return "—"
    return f"{v:,.2f}"

def _fmt_num(v, dec=2):
    if v is None or pd.isna(v):
        return "—"
    return f"{v:,.{dec}f}"

def _fmt_pct(v, dec=1, signed=False):
    if v is None or pd.isna(v):
        return "—"
    sign = "+" if (signed and v >= 0) else ""
    return f"{sign}{v:.{dec}f}%"

def _pct_class(v):
    if v is None or pd.isna(v):
        return ""
    return "pos" if v > 0 else ("neg" if v < 0 else "")


@st.cache_data(ttl=3600, show_spinner=False)
def _all_stock_options_eb():
    return [""] + [f"{s['ticker']} · {s['name']}"
                   for s in data_manager.get_all_stock_basic()]


@st.cache_data(ttl=3600, show_spinner=False)
def _stock_basic_row(ticker):
    ts_code = data_manager.get_tushare_ticker(ticker)
    df = data_manager.db.read_table(
        "stock_basic", filters={"ts_code": ts_code},
        columns="ts_code,symbol,name,area,industry,market,list_date", limit=1,
    )
    if df is None or df.empty:
        return None
    return df.iloc[0].to_dict()


@st.cache_data(ttl=300, show_spinner=False, max_entries=20)
def _bundle_fundamentals(ticker: str, periods: int = 8):
    return {
        "income":   data_manager.fetch_income_statement(ticker, periods),
        "balance":  data_manager.fetch_balance_sheet(ticker, periods),
        "cashflow": data_manager.fetch_cashflow(ticker, periods),
        "fina":     data_manager.fetch_full_fina_indicator(ticker, periods),
        "daily":    data_manager.get_latest_daily_basic(ticker),
        "mainbz":   data_manager.fetch_fina_mainbz(ticker, "P"),
    }


@st.cache_data(ttl=300, show_spinner=False, max_entries=10)
def _technical_signal(ticker: str):
    df = data_manager.get_single_stock_data_live(ticker, lookback_years=2)
    if df is None or df.empty:
        return None, None
    analysis = run_single_stock_analysis(df)
    if analysis is None or analysis.empty:
        return None, df
    return analysis, df


def _entry_signal_label(analysis_df) -> tuple[str, str]:
    """
    Return ("label", "css_class") describing the latest entry signal.
    css_class is one of "pos", "neg", "warn", or "" (neutral).
    """
    if analysis_df is None or analysis_df.empty:
        return "—", ""
    last = analysis_df.iloc[-1]
    if last.get("Squeeze_Fired_Bullish", False):
        return "Bullish Squeeze Fired", "pos"
    if last.get("Signal_Accumulation", False):
        return "Accumulation", "pos"
    if last.get("Signal_Squeeze", False):
        return "In Squeeze", "warn"
    if last.get("Squeeze_Fired_Bearish", False):
        return "Bearish Squeeze Fired", "neg"
    if last.get("Exit_MACD_Lead", False):
        return "Exit Signal", "neg"
    return "Neutral", ""


# ══════════════════════════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="eb-root">', unsafe_allow_html=True)

c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    pick = st.selectbox(
        "Stock", options=_all_stock_options_eb(), key="eb_pick",
        format_func=lambda x: "Type to search… (code or name)" if x == "" else x,
        label_visibility="collapsed",
    )
with c2:
    st.write("")
    generate = st.button("📄 Generate Brief", type="primary", use_container_width=True)
with c3:
    st.write("")
    if st.button("✖ Reset", use_container_width=True):
        st.session_state.pop("eb_active_ticker", None)
        st.rerun()

raw = (pick or "").strip()
ticker = raw.split(" · ")[0].strip() if " · " in raw else raw

if generate and len(ticker) == 6 and ticker.isdigit():
    st.session_state["eb_active_ticker"] = ticker

active = st.session_state.get("eb_active_ticker")
if not active:
    st.markdown(
        '<div class="eb-card" style="margin-top:24px">'
        '<div class="eb-eyebrow">Equity Brief · 个股研报</div>'
        '<div class="eb-h2">A printable, AURX-style brief for any A-share stock.</div>'
        '<div class="eb-kicker">Combines fundamentals (Tushare income / balance sheet / '
        'cash flow), the technical signal engine you already use, and AI-generated '
        'qualitative analysis (PESTEL, Porter\'s Five, SWOT, peer set). '
        'Sections cache on first generation — pages reload in &lt;5 s thereafter. '
        'Use Ctrl+P / Cmd+P to print or save as PDF.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

ticker = active
basic = _stock_basic_row(ticker)
if not basic:
    st.error(f"❌ {ticker} not found in stock_basic.")
    st.stop()

company  = basic["name"]
industry = basic.get("industry") or "—"

is_admin = auth_manager.is_admin()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner(f"Loading fundamentals for {company}…"):
    bundle = _bundle_fundamentals(ticker)

inc, bal, cf, fina, daily, mainbz = (
    bundle["income"], bundle["balance"], bundle["cashflow"],
    bundle["fina"], bundle["daily"], bundle["mainbz"],
)
metrics = data_manager.compute_derived_metrics(inc, bal, cf, daily or {})

# Header HTML
list_date = str(basic.get("list_date") or "")
list_str  = f"{list_date[:4]}-{list_date[4:6]}-{list_date[6:8]}" if len(list_date) == 8 else "—"
last_price = (daily or {}).get("close")
mv_yi      = (daily or {}).get("total_mv")

st.markdown(f"""
<div class="eb-section">
  <div class="eb-eyebrow">Equity Brief · {ticker}</div>
  <div class="eb-h1">{company}</div>
  <div style="display:flex;gap:18px;flex-wrap:wrap;color:var(--ink-2);font-size:13px;margin-top:10px">
    <div><span class="eb-pill">{basic.get('market') or 'A-share'}</span></div>
    <div>Industry · <strong>{industry}</strong></div>
    <div>HQ · <strong>{basic.get('area') or '—'}</strong></div>
    <div>Listed · <strong>{list_str}</strong></div>
  </div>
  <div style="margin-top:18px;display:flex;gap:32px;flex-wrap:wrap">
    <div>
      <div class="eb-eyebrow">Last Price</div>
      <div class="eb-num" style="font-size:36px;font-weight:500">¥{_fmt_num(last_price, 2)}</div>
    </div>
    <div>
      <div class="eb-eyebrow">Market Cap</div>
      <div class="eb-num" style="font-size:36px;font-weight:500">{_fmt_yi(mv_yi)}<span style="font-size:14px;color:var(--ink-3)"> 亿</span></div>
    </div>
    <div>
      <div class="eb-eyebrow">Enterprise Value</div>
      <div class="eb-num" style="font-size:36px;font-weight:500">{_fmt_yi(metrics['ev_yi'])}<span style="font-size:14px;color:var(--ink-3)"> 亿</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 01 — SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════

pe   = (daily or {}).get("pe_ttm") or (daily or {}).get("pe")
pb   = (daily or {}).get("pb")
ps   = (daily or {}).get("ps_ttm") or (daily or {}).get("ps")
dvr  = (daily or {}).get("dv_ratio")
roe  = fina.iloc[0]["roe"] if (fina is not None and not fina.empty and "roe" in fina.columns) else None
roa  = fina.iloc[0]["roa"] if (fina is not None and not fina.empty and "roa" in fina.columns) else None
or_yoy = fina.iloc[0]["or_yoy"] if (fina is not None and not fina.empty and "or_yoy" in fina.columns) else None
np_yoy = fina.iloc[0]["netprofit_yoy"] if (fina is not None and not fina.empty and "netprofit_yoy" in fina.columns) else None

kpi_html = f"""
<div class="eb-section">
  <div class="eb-eyebrow">01 · Snapshot</div>
  <div class="eb-h2">Where the numbers stand today.</div>
  <div class="eb-kicker">Trailing-twelve-month metrics from Tushare daily_basic and fina_indicator,
    with EV / EBITDA and FCF yield computed from the last four quarters of statements.</div>
  <div class="eb-kpi-grid">
    <div class="eb-kpi"><div class="eb-kpi-label">P/E (TTM)</div>
      <div class="eb-kpi-value">{_fmt_num(pe, 1)}</div></div>
    <div class="eb-kpi"><div class="eb-kpi-label">P/B</div>
      <div class="eb-kpi-value">{_fmt_num(pb, 2)}</div></div>
    <div class="eb-kpi"><div class="eb-kpi-label">P/S (TTM)</div>
      <div class="eb-kpi-value">{_fmt_num(ps, 2)}</div></div>
    <div class="eb-kpi"><div class="eb-kpi-label">EV / EBITDA</div>
      <div class="eb-kpi-value">{_fmt_num(metrics['ev_ebitda'], 1)}</div></div>
    <div class="eb-kpi"><div class="eb-kpi-label">ROE</div>
      <div class="eb-kpi-value">{_fmt_pct(roe, 1)}</div></div>
    <div class="eb-kpi"><div class="eb-kpi-label">ROA</div>
      <div class="eb-kpi-value">{_fmt_pct(roa, 1)}</div></div>
    <div class="eb-kpi"><div class="eb-kpi-label">Gross Margin (TTM)</div>
      <div class="eb-kpi-value">{_fmt_pct(metrics['gross_margin_ttm_pct'], 1)}</div></div>
    <div class="eb-kpi"><div class="eb-kpi-label">Net Margin (TTM)</div>
      <div class="eb-kpi-value">{_fmt_pct(metrics['net_margin_ttm_pct'], 1)}</div></div>
    <div class="eb-kpi"><div class="eb-kpi-label">Revenue YoY</div>
      <div class="eb-kpi-value">{_fmt_pct(or_yoy, 1, signed=True)}</div></div>
    <div class="eb-kpi"><div class="eb-kpi-label">Net Profit YoY</div>
      <div class="eb-kpi-value">{_fmt_pct(np_yoy, 1, signed=True)}</div></div>
    <div class="eb-kpi"><div class="eb-kpi-label">Debt / Equity</div>
      <div class="eb-kpi-value">{_fmt_num(metrics['debt_to_equity'], 2)}</div></div>
    <div class="eb-kpi"><div class="eb-kpi-label">Net Debt / EBITDA</div>
      <div class="eb-kpi-value">{_fmt_num(metrics['net_debt_ebitda'], 2)}</div></div>
    <div class="eb-kpi"><div class="eb-kpi-label">Current Ratio</div>
      <div class="eb-kpi-value">{_fmt_num(metrics['current_ratio'], 2)}</div></div>
    <div class="eb-kpi"><div class="eb-kpi-label">FCF Yield</div>
      <div class="eb-kpi-value">{_fmt_pct(metrics['fcf_yield_pct'], 2)}</div></div>
    <div class="eb-kpi"><div class="eb-kpi-label">Dividend Yield</div>
      <div class="eb-kpi-value">{_fmt_pct(dvr, 2)}</div></div>
    <div class="eb-kpi"><div class="eb-kpi-label">P / FCF</div>
      <div class="eb-kpi-value">{_fmt_num(metrics['p_fcf'], 1)}</div></div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 02 — TECHNICAL
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="eb-section">
  <div class="eb-eyebrow">02 · Technical</div>
  <div class="eb-h2">Where the chart says we are.</div>
  <div class="eb-kicker">Phase 1 (Accumulation) → Phase 2 (Squeeze) → Phase 3 (Breakout)
    signals from the same engine used in Single Stock Analysis. Below shows the
    last 250 trading days of qfq-adjusted price, with active signals highlighted.</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Running technical analysis…"):
    analysis_df, raw_df = _technical_signal(ticker)

if analysis_df is not None and not analysis_df.empty:
    last250 = analysis_df.tail(250)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=last250.index, open=last250["Open"], high=last250["High"],
        low=last250["Low"], close=last250["Close"],
        name="Price",
        increasing=dict(line=dict(color="#c6432a")),
        decreasing=dict(line=dict(color="#2f8a4f")),
    ))
    if "MA20" in last250.columns:
        fig.add_trace(go.Scatter(x=last250.index, y=last250["MA20"], name="MA20",
                                 line=dict(color="#b8800f", width=1.2, dash="dot")))
    if "MA50" in last250.columns:
        fig.add_trace(go.Scatter(x=last250.index, y=last250["MA50"], name="MA50",
                                 line=dict(color="#2563a8", width=1.5)))
    acc = last250[last250.get("Signal_Accumulation", False)]
    if not acc.empty:
        fig.add_trace(go.Scatter(x=acc.index, y=acc["Low"] * 0.98, mode="markers",
            name="Accumulation", marker=dict(color="#b8800f", size=8, symbol="circle")))
    bull = last250[last250.get("Squeeze_Fired_Bullish", False)]
    if not bull.empty:
        fig.add_trace(go.Scatter(x=bull.index, y=bull["Low"] * 0.95, mode="markers",
            name="Squeeze Fired", marker=dict(color="#2f8a4f", size=12, symbol="triangle-up")))

    fig.update_layout(
        height=420, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#f7f3ec", paper_bgcolor="#f7f3ec",
    )
    fig.update_xaxes(gridcolor="#d6cebe")
    fig.update_yaxes(gridcolor="#d6cebe", title="Price (¥)")
    st.plotly_chart(fig, use_container_width=True)

    sig_label, sig_class = _entry_signal_label(analysis_df)
    pill_class = f"eb-pill {sig_class}" if sig_class else "eb-pill"
    st.markdown(f'<div style="margin-top:8px"><span class="{pill_class}">'
                f'Latest signal · {sig_label}</span></div>', unsafe_allow_html=True)

    # ── Trading Strategy block (within Section 02) ────────────────────────────
    summary = trading_strategy.build_strategy_summary(raw_df, analysis_df)

    if summary:
        score   = summary["score"]
        stg     = summary["stop_targets"]
        zone    = summary["entry_zone"]
        sups    = summary["supports"]
        res     = summary["resistances"]
        price   = summary["price"]
        atr     = summary["atr"]

        # ── Signal gauge as semicircle SVG (-100 left → +100 right) ──────────
        # Map score → angle: -100 = -90°, +100 = +90°
        angle_deg = score["score"] * 0.9
        rad = np.deg2rad(angle_deg - 90)  # -90° offset → start at top
        cx, cy, r = 110, 110, 90
        nx = cx + r * np.cos(rad)
        ny = cy + r * np.sin(rad)
        # Score color
        s = score["score"]
        if s >= 40:    score_color = "#2f8a4f"
        elif s >= 10:  score_color = "#82c69a"
        elif s > -10:  score_color = "#b8800f"
        elif s > -40:  score_color = "#e08a6e"
        else:          score_color = "#c6432a"

        gauge_svg = f"""
        <svg width="220" height="135" viewBox="0 0 220 135">
          <!-- background arc -->
          <path d="M 20 110 A 90 90 0 0 1 200 110"
                fill="none" stroke="#d6cebe" stroke-width="14"/>
          <!-- colored arc up to the score -->
          <path d="M 20 110 A 90 90 0 0 1 {nx:.2f} {ny:.2f}"
                fill="none" stroke="{score_color}" stroke-width="14"
                stroke-linecap="round"/>
          <!-- needle -->
          <line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}"
                stroke="#1a1916" stroke-width="2"/>
          <circle cx="{cx}" cy="{cy}" r="5" fill="#1a1916"/>
          <!-- score number -->
          <text x="{cx}" y="80" text-anchor="middle"
                font-family="JetBrains Mono, monospace" font-size="32"
                font-weight="500" fill="#1a1916">{s:+d}</text>
          <text x="{cx}" y="100" text-anchor="middle"
                font-family="ui-monospace, monospace" font-size="11"
                letter-spacing="0.14em" fill="#807a70" text-transform="uppercase">
                {score['label'].upper()}</text>
          <!-- scale labels -->
          <text x="20" y="128" font-family="ui-monospace, monospace"
                font-size="9" fill="#807a70">-100</text>
          <text x="200" y="128" text-anchor="end" font-family="ui-monospace, monospace"
                font-size="9" fill="#807a70">+100</text>
        </svg>
        """

        # ── Component breakdown bars ─────────────────────────────────────────
        comps = score["components"]
        comp_html = ""
        for k in ("trend", "momentum", "rsi", "volatility", "custom"):
            c = comps[k]
            v, mx = c["value"], c["max"]
            # Map -mx..+mx → 0..100% width, centered at 50%
            center, width_pct = 50, abs(v) / mx * 50
            if v >= 0:
                bar_left, bar_right = center, center + width_pct
                bar_color = "#2f8a4f"
            else:
                bar_left, bar_right = center - width_pct, center
                bar_color = "#c6432a"
            comp_html += f"""
            <div style="display:grid;grid-template-columns:90px 1fr 50px;gap:10px;
                        align-items:center;padding:5px 0;font-size:12px">
              <div style="font-family:ui-monospace,monospace;font-size:10.5px;
                          letter-spacing:0.08em;color:var(--ink-3);text-transform:uppercase">
                {c['label']}</div>
              <div style="position:relative;height:10px;background:#efe9df;border-radius:2px">
                <div style="position:absolute;left:50%;top:-2px;width:1px;height:14px;
                            background:#807a70"></div>
                <div style="position:absolute;left:{bar_left}%;width:{bar_right-bar_left}%;
                            top:0;height:10px;background:{bar_color};border-radius:2px"></div>
              </div>
              <div class="eb-num" style="text-align:right;font-size:12px;color:var(--ink-2)">
                {v:+d}</div>
            </div>"""

        # ── Stop/targets card ────────────────────────────────────────────────
        stop_targets_html = f"""
        <div class="eb-card">
          <div class="eb-eyebrow" style="margin-bottom:10px">Strategy</div>
          <table class="eb-table" style="font-size:13px">
            <tr><td>Current</td>
                <td class="num"><strong>¥{price:.2f}</strong></td>
                <td class="num" style="color:var(--ink-3)">ATR ¥{atr:.2f}</td></tr>
            <tr><td>Stop</td>
                <td class="num">¥{stg['stop']:.2f}</td>
                <td class="num neg">{stg['risk_pct']:+.1f}%</td></tr>
            <tr><td>T1 (1R)</td>
                <td class="num">¥{stg['t1']:.2f}</td>
                <td class="num pos">+{stg['reward_t1_pct']:.1f}%</td></tr>
            <tr><td>T2 (1.5R)</td>
                <td class="num">¥{stg['t2']:.2f}</td>
                <td class="num pos">+{stg['reward_t2_pct']:.1f}%</td></tr>
            <tr><td>T3 (capped at R1)</td>
                <td class="num">¥{stg['t3']:.2f}</td>
                <td class="num pos">+{stg['reward_t3_pct']:.1f}%</td></tr>
            <tr><td><strong>R : R</strong></td>
                <td class="num" colspan="2"><strong>1 : {stg['rr']:.2f}</strong>
                    <span style="color:var(--ink-3);font-size:11px">  · using T2</span></td></tr>
          </table>
        </div>
        """

        # ── S/R table ────────────────────────────────────────────────────────
        sr_rows = ""
        for r_lvl in res:
            d = (r_lvl["price"] - price) / price * 100
            dots = "●" * min(int(r_lvl["strength"]), 5)
            sr_rows += f"""<tr>
              <td><strong style="color:var(--neg)">R</strong></td>
              <td class="num">¥{r_lvl['price']:.2f}</td>
              <td class="num neg">+{d:.1f}%</td>
              <td style="color:var(--ink-3);font-family:ui-monospace,monospace;font-size:11px">{dots}</td>
            </tr>"""
        sr_rows += f"""<tr style="background:#efe9df">
          <td><strong>—</strong></td>
          <td class="num"><strong>¥{price:.2f}</strong></td>
          <td class="num"><strong>now</strong></td>
          <td></td>
        </tr>"""
        for s_lvl in sups:
            d = (s_lvl["price"] - price) / price * 100
            dots = "●" * min(int(s_lvl["strength"]), 5)
            sr_rows += f"""<tr>
              <td><strong style="color:var(--pos)">S</strong></td>
              <td class="num">¥{s_lvl['price']:.2f}</td>
              <td class="num pos">{d:.1f}%</td>
              <td style="color:var(--ink-3);font-family:ui-monospace,monospace;font-size:11px">{dots}</td>
            </tr>"""

        sr_html = f"""
        <div class="eb-card">
          <div class="eb-eyebrow" style="margin-bottom:10px">Support & Resistance · volume-weighted</div>
          <table class="eb-table" style="font-size:13px">
            <thead><tr><th></th><th>Price</th><th>Δ%</th><th>Strength</th></tr></thead>
            <tbody>{sr_rows}</tbody>
          </table>
        </div>
        """

        # ── Entry zone bar ───────────────────────────────────────────────────
        all_zones = zone["all_zones"]
        z_min = min(b["low"] for b in all_zones)
        z_max = max(b["high"] for b in all_zones)
        z_range = max(z_max - z_min, 1e-6)

        bar_segments, label_segments = "", ""
        for b in all_zones:
            left  = (b["low"]  - z_min) / z_range * 100
            width = (b["high"] - b["low"]) / z_range * 100
            bar_segments += (
                f'<div style="position:absolute;left:{left}%;width:{width}%;'
                f'top:0;height:34px;background:{b["color"]};'
                f'border-right:1px solid rgba(0,0,0,0.15)"></div>'
            )
            # Label only if wide enough
            if width > 7:
                label_segments += (
                    f'<div style="position:absolute;left:{left}%;width:{width}%;'
                    f'top:36px;text-align:center;font-family:ui-monospace,monospace;'
                    f'font-size:9.5px;color:var(--ink-2);letter-spacing:0.06em;'
                    f'text-transform:uppercase">{b["label"]}</div>'
                )

        marker_left = (price - z_min) / z_range * 100
        zone_bar_html = f"""
        <div class="eb-card">
          <div class="eb-eyebrow" style="margin-bottom:10px">Entry Zone</div>
          <div style="display:flex;justify-content:space-between;
                      font-size:13px;margin-bottom:8px">
            <div>Current price sits in <strong>{zone['label']}</strong></div>
            <div style="color:var(--ink-2)">{zone['action']}</div>
          </div>
          <div style="position:relative;height:34px;background:#efe9df;
                      border-radius:3px;overflow:hidden">
            {bar_segments}
            <div style="position:absolute;left:{marker_left}%;top:-6px;
                        width:2px;height:46px;background:#1a1916;z-index:5"></div>
            <div style="position:absolute;left:calc({marker_left}% - 30px);
                        top:-22px;font-family:JetBrains Mono,monospace;font-size:11px;
                        font-weight:600;color:#1a1916;z-index:6;width:60px;text-align:center">
              ¥{price:.2f}</div>
          </div>
          <div style="position:relative;height:36px;margin-top:4px">
            {label_segments}
          </div>
          <div style="display:flex;justify-content:space-between;
                      font-family:JetBrains Mono,monospace;font-size:10px;
                      color:var(--ink-3);margin-top:6px">
            <span>¥{z_min:.2f}</span>
            <span>¥{z_max:.2f}</span>
          </div>
        </div>
        """

        # ── Render: 2-column layout (gauge+breakdown | stop/targets) then 2 cols below
        st.markdown(f"""
        <div style="margin-top:24px;display:grid;grid-template-columns:1fr 1fr;gap:14px">
          <div class="eb-card">
            <div class="eb-eyebrow" style="margin-bottom:6px">Composite Signal Score</div>
            <div style="display:flex;align-items:center;gap:18px">
              {gauge_svg}
              <div style="flex:1">{comp_html}</div>
            </div>
          </div>
          {stop_targets_html}
        </div>
        <div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr;gap:14px">
          {sr_html}
          {zone_bar_html}
        </div>
        """, unsafe_allow_html=True)
else:
    st.warning("No technical data available.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 03 — FUNDAMENTALS & YOY GROWTH
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="eb-section">
  <div class="eb-eyebrow">03 · Fundamentals & YoY Growth</div>
  <div class="eb-h2">Eight quarters of trend.</div>
  <div class="eb-kicker">Revenue, gross profit, net profit, and operating cash flow — quarter
    by quarter, with the same-quarter year-over-year change highlighted.</div>
</div>
""", unsafe_allow_html=True)

if inc is not None and not inc.empty:
    df_show = inc.copy()
    df_show["end_date"] = df_show["end_date"].astype(str)
    df_show["gross_profit"] = (df_show.get("total_revenue", 0).fillna(0)
                              - df_show.get("oper_cost", 0).fillna(0))

    # YoY: compare each row to the row 4 quarters older
    def _yoy_series(s: pd.Series) -> pd.Series:
        s2 = s.reset_index(drop=True)
        return ((s2 - s2.shift(-4)) / s2.shift(-4).abs() * 100)

    df_show["rev_yoy"]    = _yoy_series(df_show["total_revenue"])
    df_show["np_yoy"]     = _yoy_series(df_show["n_income_attr_p"])
    df_show["gp_yoy"]     = _yoy_series(df_show["gross_profit"])

    if cf is not None and not cf.empty:
        cf_aligned = cf.set_index(cf["end_date"].astype(str))
        df_show["ocf"] = df_show["end_date"].map(cf_aligned["n_cashflow_act"])
        df_show["ocf_yoy"] = _yoy_series(df_show["ocf"])
    else:
        df_show["ocf"] = None
        df_show["ocf_yoy"] = None

    rows_html = ""
    for _, r in df_show.iterrows():
        rev_y = r.get("rev_yoy"); np_y = r.get("np_yoy")
        gp_y  = r.get("gp_yoy");  ocf_y = r.get("ocf_yoy")
        rows_html += f"""
        <tr>
          <td>{r['end_date']}</td>
          <td class="num">{_fmt_yi(r['total_revenue']/1e8 if pd.notna(r['total_revenue']) else None)}</td>
          <td class="num {_pct_class(rev_y)}">{_fmt_pct(rev_y, 1, signed=True)}</td>
          <td class="num">{_fmt_yi(r['gross_profit']/1e8 if pd.notna(r['gross_profit']) else None)}</td>
          <td class="num {_pct_class(gp_y)}">{_fmt_pct(gp_y, 1, signed=True)}</td>
          <td class="num">{_fmt_yi(r['n_income_attr_p']/1e8 if pd.notna(r['n_income_attr_p']) else None)}</td>
          <td class="num {_pct_class(np_y)}">{_fmt_pct(np_y, 1, signed=True)}</td>
          <td class="num">{_fmt_yi(r['ocf']/1e8 if pd.notna(r['ocf']) else None)}</td>
          <td class="num {_pct_class(ocf_y)}">{_fmt_pct(ocf_y, 1, signed=True)}</td>
        </tr>"""

    st.markdown(f"""
    <div class="eb-card">
    <table class="eb-table">
      <thead><tr>
        <th>Period</th>
        <th>Revenue (亿)</th><th>Rev YoY</th>
        <th>Gross (亿)</th><th>GP YoY</th>
        <th>NP (亿)</th><th>NP YoY</th>
        <th>OCF (亿)</th><th>OCF YoY</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("No income statement data available.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 04 — QUALITATIVE (PESTEL / PORTER'S / SWOT)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="eb-section">
  <div class="eb-eyebrow">04 · Qualitative</div>
  <div class="eb-h2">PESTEL · Porter's Five · SWOT.</div>
  <div class="eb-kicker">AI-generated, grounded in this company's industry context and
    headline metrics. Cached after first generation; admins can regenerate.</div>
</div>
""", unsafe_allow_html=True)

# Build the metrics summary string for SWOT grounding
metrics_summary = "\n".join([
    f"- P/E (TTM): {_fmt_num(pe, 1)}",
    f"- P/B: {_fmt_num(pb, 2)}",
    f"- ROE: {_fmt_pct(roe, 1)}",
    f"- Gross margin (TTM): {_fmt_pct(metrics['gross_margin_ttm_pct'], 1)}",
    f"- Net margin (TTM): {_fmt_pct(metrics['net_margin_ttm_pct'], 1)}",
    f"- Revenue YoY: {_fmt_pct(or_yoy, 1, signed=True)}",
    f"- Net profit YoY: {_fmt_pct(np_yoy, 1, signed=True)}",
    f"- Net debt: {_fmt_yi(metrics['net_debt_yi'])} 亿元",
    f"- EV/EBITDA: {_fmt_num(metrics['ev_ebitda'], 1)}",
    f"- FCF yield: {_fmt_pct(metrics['fcf_yield_pct'], 2)}",
])

# --- PESTEL ---
st.markdown('<h3 class="eb-h3" style="margin-top:18px">PESTEL</h3>', unsafe_allow_html=True)
pcol1, pcol2 = st.columns([5, 1])
with pcol2:
    if is_admin and st.button("🔄 Regen", key="eb_re_pestel", use_container_width=True):
        with st.spinner("Regenerating PESTEL…"):
            try:
                equity_brief.get_pestel(ticker, company, industry, force_refresh=True)
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
try:
    with st.spinner("Loading PESTEL…"):
        pestel = equity_brief.get_pestel(ticker, company, industry)
    p = pestel["payload"]
    cards = []
    for label in ("political","economic","social","technological","environmental","legal"):
        bullets = "".join(f"<li>{b}</li>" for b in p.get(label, []))
        cards.append(f'<div class="eb-2col-card"><h4>{label}</h4><ul>{bullets}</ul></div>')
    st.markdown('<div class="eb-2col" style="grid-template-columns:repeat(3,1fr)">'
                + "".join(cards) + '</div>', unsafe_allow_html=True)
except RuntimeError as exc:
    st.error(f"PESTEL failed: {exc}")

# --- Porter's Five ---
st.markdown('<h3 class="eb-h3" style="margin-top:24px">Porter\'s Five Forces</h3>',
            unsafe_allow_html=True)
fcol1, fcol2 = st.columns([5, 1])
with fcol2:
    if is_admin and st.button("🔄 Regen", key="eb_re_porters", use_container_width=True):
        with st.spinner("Regenerating Porter's…"):
            try:
                equity_brief.get_porters(ticker, company, industry, force_refresh=True)
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
try:
    with st.spinner("Loading Porter's…"):
        porters = equity_brief.get_porters(ticker, company, industry)
    pp = porters["payload"]
    force_html = '<div class="eb-card">'
    for fkey, flabel in [("rivalry","Rivalry"),("suppliers","Suppliers"),
                         ("buyers","Buyers"),("substitutes","Substitutes"),
                         ("new_entrants","New Entrants")]:
        f = pp.get(fkey, {}) or {}
        score = int(f.get("score", 0) or 0)
        summary = f.get("summary", "—")
        bar = "".join(
            f'<span class="seg{" on" if i < score else ""}"></span>'
            for i in range(5))
        force_html += f"""
        <div class="eb-force">
          <div class="eb-force-label">{flabel}</div>
          <div class="eb-force-bar">{bar}</div>
          <div style="font-size:13px;color:var(--ink-2)">{summary}</div>
        </div>"""
    force_html += "</div>"
    st.markdown(force_html, unsafe_allow_html=True)
except RuntimeError as exc:
    st.error(f"Porter's failed: {exc}")

# --- SWOT ---
st.markdown('<h3 class="eb-h3" style="margin-top:24px">SWOT</h3>', unsafe_allow_html=True)
scol1, scol2 = st.columns([5, 1])
with scol2:
    if is_admin and st.button("🔄 Regen", key="eb_re_swot", use_container_width=True):
        with st.spinner("Regenerating SWOT…"):
            try:
                equity_brief.get_swot(ticker, company, industry, metrics_summary,
                                       force_refresh=True)
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
try:
    with st.spinner("Loading SWOT…"):
        swot = equity_brief.get_swot(ticker, company, industry, metrics_summary)
    s = swot["payload"]
    cards = []
    for k, label in [("strengths","Strengths"),("weaknesses","Weaknesses"),
                     ("opportunities","Opportunities"),("threats","Threats")]:
        bullets = "".join(f"<li>{b}</li>" for b in s.get(k, []))
        cards.append(f'<div class="eb-2col-card"><h4>{label}</h4><ul>{bullets}</ul></div>')
    st.markdown('<div class="eb-2col">' + "".join(cards) + '</div>',
                unsafe_allow_html=True)
except RuntimeError as exc:
    st.error(f"SWOT failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 05 — PRODUCT SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="eb-section">
  <div class="eb-eyebrow">05 · Product Segments</div>
  <div class="eb-h2">Where the revenue actually comes from.</div>
  <div class="eb-kicker">Latest annual main-business breakdown from Tushare fina_mainbz,
    with year-over-year comparison if available.</div>
</div>
""", unsafe_allow_html=True)

if mainbz is not None and not mainbz.empty:
    latest_period = mainbz["end_date"].iloc[0]
    latest = mainbz[mainbz["end_date"] == latest_period].copy()
    latest["pct"] = latest["bz_sales"] / latest["bz_sales"].sum() * 100
    latest = latest.sort_values("bz_sales", ascending=False).head(10)

    rows_html = "".join(
        f"""<tr>
          <td>{r['bz_item']}</td>
          <td class="num">{_fmt_yi(r['bz_sales']/1e8)}</td>
          <td class="num">{_fmt_pct(r['pct'], 1)}</td>
        </tr>"""
        for _, r in latest.iterrows()
    )
    st.markdown(f"""
    <div class="eb-card">
      <div class="eb-eyebrow" style="margin-bottom:10px">Period · {latest_period}</div>
      <table class="eb-table">
        <thead><tr><th>Segment</th><th>Revenue (亿)</th><th>Share</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("No segment breakdown available for this company.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 06 — COMPETITORS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="eb-section">
  <div class="eb-eyebrow">06 · Competitor Comparison</div>
  <div class="eb-h2">How peers stack up.</div>
  <div class="eb-kicker">AI-selected A-share peers competing in the same end-market,
    each cross-checked against stock_basic and put through the same fundamentals
    + technical engine for an apples-to-apples comparison.</div>
</div>
""", unsafe_allow_html=True)

ccol1, ccol2 = st.columns([5, 1])
with ccol2:
    if is_admin and st.button("🔄 Regen peers", key="eb_re_peers", use_container_width=True):
        with st.spinner("Regenerating peer set…"):
            try:
                equity_brief.get_competitors(ticker, company, industry, force_refresh=True)
                st.session_state.pop(f"eb_peer_select_{ticker}", None)
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))

try:
    with st.spinner("Loading peers…"):
        comps = equity_brief.get_competitors(ticker, company, industry)
    peer_list = comps["payload"].get("competitors", [])
except RuntimeError as exc:
    st.error(f"Peer discovery failed: {exc}")
    peer_list = []

if not peer_list:
    st.info("No validated peers were returned.")
else:
    # Admin-only: per-peer checkbox for curation
    selected = []
    if is_admin:
        st.markdown('<div style="margin-bottom:8px"><span class="eb-pill warn">'
                    'Admin · uncheck any wrong peers, then save</span></div>',
                    unsafe_allow_html=True)
        sel_key_root = f"eb_peer_chk_{ticker}"
        for p in peer_list:
            checked = st.checkbox(
                f"`{p['ticker']}` {p['name']} — {p.get('why','')}",
                value=True, key=f"{sel_key_root}_{p['ticker']}",
            )
            if checked:
                selected.append(p)
        if st.button("💾 Save curated peer set", key="eb_save_peers"):
            equity_brief.save_competitors_curated(ticker, selected)
            st.success(f"✅ Saved {len(selected)} peers.")
            st.rerun()
    else:
        selected = peer_list

    # Build comparison table
    target_row = {
        "ticker": ticker, "name": company, "is_target": True,
        "pe": pe, "pb": pb, "ev_ebitda": metrics["ev_ebitda"],
        "rev_yoy": or_yoy, "np_yoy": np_yoy, "roe": roe,
        "net_debt_ebitda": metrics["net_debt_ebitda"],
        "signal": _entry_signal_label(analysis_df),
    }
    table_rows = [target_row]

    with st.spinner(f"Pulling fundamentals + technicals for {len(selected)} peers…"):
        for p in selected:
            t = p["ticker"]
            try:
                pdaily = data_manager.get_latest_daily_basic(t)
                pinc   = data_manager.fetch_income_statement(t, 8)
                pbal   = data_manager.fetch_balance_sheet(t, 8)
                pcf    = data_manager.fetch_cashflow(t, 8)
                pfina  = data_manager.fetch_full_fina_indicator(t, 8)
                pmet   = data_manager.compute_derived_metrics(pinc, pbal, pcf, pdaily or {})
                pana, _ = _technical_signal(t)
            except Exception as exc:
                print(f"[Equity_Brief] peer {t}: {exc}")
                continue

            table_rows.append({
                "ticker": t, "name": p["name"], "is_target": False,
                "pe":  (pdaily or {}).get("pe_ttm") or (pdaily or {}).get("pe"),
                "pb":  (pdaily or {}).get("pb"),
                "ev_ebitda": pmet["ev_ebitda"],
                "rev_yoy": pfina.iloc[0]["or_yoy"] if (pfina is not None and not pfina.empty) else None,
                "np_yoy": pfina.iloc[0]["netprofit_yoy"] if (pfina is not None and not pfina.empty) else None,
                "roe":   pfina.iloc[0]["roe"]    if (pfina is not None and not pfina.empty) else None,
                "net_debt_ebitda": pmet["net_debt_ebitda"],
                "signal": _entry_signal_label(pana),
            })

    rows_html = ""
    for r in table_rows:
        sig_lbl, sig_cls = r["signal"]
        sig_pill = f'<span class="eb-pill {sig_cls}">{sig_lbl}</span>' if sig_cls else sig_lbl
        target_marker = ' <span class="eb-pill">Target</span>' if r["is_target"] else ""
        rows_html += f"""
        <tr>
          <td><strong>`{r['ticker']}` {r['name']}</strong>{target_marker}</td>
          <td class="num">{_fmt_num(r['pe'], 1)}</td>
          <td class="num">{_fmt_num(r['pb'], 2)}</td>
          <td class="num">{_fmt_num(r['ev_ebitda'], 1)}</td>
          <td class="num {_pct_class(r['rev_yoy'])}">{_fmt_pct(r['rev_yoy'], 1, signed=True)}</td>
          <td class="num {_pct_class(r['np_yoy'])}">{_fmt_pct(r['np_yoy'], 1, signed=True)}</td>
          <td class="num">{_fmt_pct(r['roe'], 1)}</td>
          <td class="num">{_fmt_num(r['net_debt_ebitda'], 2)}</td>
          <td>{sig_pill}</td>
        </tr>"""

    st.markdown(f"""
    <div class="eb-card" style="margin-top:12px">
      <table class="eb-table">
        <thead><tr>
          <th>Stock</th><th>P/E</th><th>P/B</th><th>EV/EBITDA</th>
          <th>Rev YoY</th><th>NP YoY</th><th>ROE</th>
          <th>Net Debt / EBITDA</th><th>Latest Signal</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

# Footer
gen_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
st.markdown(f"""
<div style="margin-top:30px;padding:18px 0;border-top:1px solid var(--rule);
    color:var(--ink-3);font-family:ui-monospace,monospace;font-size:11px;
    letter-spacing:0.1em;text-transform:uppercase">
  Generated {gen_at} · ASSRS Dashboard ·
  <span style="float:right">Press Ctrl+P to print or save as PDF</span>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close eb-root
