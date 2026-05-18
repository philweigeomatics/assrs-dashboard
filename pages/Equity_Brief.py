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
import streamlit.components.v1 as components

import auth_manager
import data_manager
import equity_brief
import sector_themes as sector_themes_mod
import supply_chain_ui
import trading_strategy
from analysis_engine import run_single_stock_analysis
from nav_helpers import page_link_button

auth_manager.require_login()
equity_brief.ensure_equity_brief_cache_table()
data_manager.ensure_chain_positions_table()

st.set_page_config(page_title="Equity Report | 个股研报", page_icon="📄", layout="wide")


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


def _render_html(html: str) -> None:
    """
    Render HTML reliably. Streamlit's st.markdown(unsafe_allow_html=True) runs
    the input through a markdown parser first — any line indented 4+ spaces is
    treated as a code block, which blows up nested HTML like our grid bars.
    Stripping leading whitespace from every line neutralizes that rule.
    """
    flat = "\n".join(line.lstrip() for line in html.splitlines())
    st.markdown(flat, unsafe_allow_html=True)


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
        "mainbz":     data_manager.fetch_fina_mainbz(ticker, "P"),
        "mainbz_geo": data_manager.fetch_fina_mainbz(ticker, "D"),
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


@st.cache_data(ttl=600, show_spinner=False, max_entries=50)
def _peer_row_data(peer_ticker: str):
    """
    Fetch all fundamentals + derived metrics for ONE peer.  Cached for
    10 minutes so unchecking a peer (which reruns the fragment) does NOT
    re-issue Tushare calls — the previously-fetched peers are served from
    cache instantly.

    Returns a dict ready to merge into a comparison-table row, or None on
    fetch failure.  Caller adds name / signal / target marker.
    """
    try:
        pdaily = data_manager.get_latest_daily_basic(peer_ticker)
        pinc   = data_manager.fetch_income_statement(peer_ticker, 8)
        pbal   = data_manager.fetch_balance_sheet(peer_ticker, 8)
        pcf    = data_manager.fetch_cashflow(peer_ticker, 8)
        pfina  = data_manager.fetch_full_fina_indicator(peer_ticker, 8)
        pmet   = data_manager.compute_derived_metrics(pinc, pbal, pcf, pdaily or {})
    except Exception as exc:
        print(f"[Equity_Brief] peer {peer_ticker}: {exc}")
        return None

    p_roe_q = p_roe_q_period = None
    p_roe_annual = p_roe_annual_period = None
    if pfina is not None and not pfina.empty and "roe" in pfina.columns:
        p_roe_q = pfina.iloc[0]["roe"]
        p_roe_q_period = str(pfina.iloc[0].get("end_date", ""))
        p_annual_rows = pfina[pfina["end_date"].astype(str).str.endswith("1231")]
        if not p_annual_rows.empty:
            p_roe_annual = p_annual_rows.iloc[0]["roe"]
            p_roe_annual_period = str(p_annual_rows.iloc[0]["end_date"])

    return {
        "ticker":            peer_ticker,
        "pe":                (pdaily or {}).get("pe_ttm") or (pdaily or {}).get("pe"),
        "pb":                (pdaily or {}).get("pb"),
        "ev_ebitda":         pmet["ev_ebitda"],
        "rev_yoy":           pfina.iloc[0]["or_yoy"] if (pfina is not None and not pfina.empty) else None,
        "np_yoy":            pfina.iloc[0]["netprofit_yoy"] if (pfina is not None and not pfina.empty) else None,
        "roe_annual":        p_roe_annual,
        "roe_annual_period": p_roe_annual_period,
        "roe_q":             p_roe_q,
        "roe_q_period":      p_roe_q_period,
        "net_debt_ebitda":   pmet["net_debt_ebitda"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="eb-root">', unsafe_allow_html=True)

# Wrap the stock picker in a form so that changing the selectbox does NOT
# trigger a server rerun — only clicking a submit button does.
# This prevents the page from re-rendering the previous stock's brief every
# time the user scrolls through the dropdown to find a new stock.
with st.form("eb_form"):
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        pick = st.selectbox(
            "Stock", options=_all_stock_options_eb(), key="eb_pick",
            format_func=lambda x: "Type to search… (code or name)" if x == "" else x,
            label_visibility="collapsed",
        )
    with c2:
        st.write("")
        generate = st.form_submit_button(
            "📄 Generate Report", type="primary", use_container_width=True)
    with c3:
        st.write("")
        reset = st.form_submit_button("✖ Reset", use_container_width=True)

raw = (pick or "").strip()
ticker = raw.split(" · ")[0].strip() if " · " in raw else raw

if reset:
    st.session_state.pop("eb_active_ticker", None)
    st.rerun()

if generate and len(ticker) == 6 and ticker.isdigit():
    st.session_state["eb_active_ticker"] = ticker

# ── Peer link navigation — ?ticker=XXXXXX in URL ─────────────────────────────
# Peer anchor links in the comparison table navigate here by appending
# ?ticker=XXXXXX to the page URL.  We detect it, set session state, and
# clear the query string so the URL stays clean on subsequent reruns.
_qp_ticker = st.query_params.get("ticker", "").strip()
if _qp_ticker and len(_qp_ticker) == 6 and _qp_ticker.isdigit():
    st.session_state["eb_active_ticker"] = _qp_ticker
    st.query_params.clear()

active = st.session_state.get("eb_active_ticker")
if not active:
    st.markdown(
        '<div class="eb-card" style="margin-top:24px">'
        '<div class="eb-eyebrow">Equity Report · 个股研报</div>'
        '<div class="eb-h2">A printable, AURX-style report for any A-share stock.</div>'
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
user     = auth_manager.get_current_user()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

# No spinner — fetcher is cached (5 min TTL); reruns hit cache instantly.
# Showing a spinner here would flash on every dropdown change even though
# the brief is already loaded.
bundle = _bundle_fundamentals(ticker)

inc, bal, cf, fina, daily, mainbz, mainbz_geo = (
    bundle["income"], bundle["balance"], bundle["cashflow"],
    bundle["fina"], bundle["daily"], bundle["mainbz"], bundle["mainbz_geo"],
)
metrics = data_manager.compute_derived_metrics(inc, bal, cf, daily or {})

# Header HTML
list_date = str(basic.get("list_date") or "")
list_str  = f"{list_date[:4]}-{list_date[4:6]}-{list_date[6:8]}" if len(list_date) == 8 else "—"
last_price = (daily or {}).get("close")
mv_yi      = (daily or {}).get("total_mv")

st.markdown(f"""
<div class="eb-section">
  <div class="eb-eyebrow">Equity Report · {ticker}</div>
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

# ── Watchlist toggle ─────────────────────────────────────────────────────────
_wl_col_spacer, _wl_col_btn = st.columns([5, 1])
with _wl_col_btn:
    _in_wl = data_manager.is_in_watchlist(ticker)
    if _in_wl:
        if st.button("★ In Watchlist — Remove",
                     key=f"eb_wl_remove_{ticker}",
                     use_container_width=True):
            _ok, _msg = data_manager.remove_from_watchlist(ticker)
            (st.success if _ok else st.warning)(_msg)
            st.rerun()
    else:
        if st.button("☆ Add to Watchlist",
                     key=f"eb_wl_add_{ticker}",
                     type="primary",
                     use_container_width=True):
            _ok, _msg = data_manager.add_to_watchlist(ticker, company)
            (st.success if _ok else st.warning)(_msg)
            if _ok:
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 00 — OVERVIEW (what the company does + supply chain graph)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="eb-section">
  <div class="eb-eyebrow">00 · Overview</div>
  <div class="eb-h2">What this company does.</div>
</div>
""", unsafe_allow_html=True)

# ── Company description card (AI, cached) ────────────────────────────────────
ovcol1, ovcol2 = st.columns([5, 1])
with ovcol2:
    if is_admin and st.button("🔄 Regen", key="eb_re_overview", use_container_width=True):
        with st.spinner("Regenerating overview…"):
            try:
                equity_brief.get_company_overview(ticker, company, industry,
                                                   force_refresh=True)
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))

try:
    overview = equity_brief.get_company_overview(ticker, company, industry)
    op = overview["payload"]
    tagline = (op.get("tagline") or "").strip()
    summary = (op.get("summary") or "").strip()
    _render_html(f"""
    <div class="eb-card" style="margin-bottom:14px">
      {f'<div class="eb-eyebrow" style="margin-bottom:6px">{tagline}</div>' if tagline else ''}
      <div style="font-size:14.5px;line-height:1.6;color:var(--ink)">
        {summary}
      </div>
    </div>
    """)
except RuntimeError as exc:
    st.warning(f"Overview unavailable: {exc}")

# ── Supply chain: layer strip helper ─────────────────────────────────────────

def _render_chain_position_strip(
    layers: list,
    active_layer_index: int | None,
    company_short: str,
    matched_items: list | None = None,
) -> None:
    """
    Render all layers of a sector theme as a horizontal strip.
    The company's layer (active_layer_index) is highlighted with a blue border
    and background; all other layers are dimmed.
    If active_layer_index is None, all layers render at equal opacity (unclassified).
    """
    cards = ""
    for layer in sorted(layers, key=lambda x: x.get("layer_index", 0)):
        idx         = layer.get("layer_index", 0)
        is_active   = (active_layer_index is not None and idx == active_layer_index)
        layer_name  = (layer.get("layer_name") or f"Layer {idx}").split("/")[0].strip()
        items       = layer.get("items", [])

        items_html = "".join(
            f'<div style="font-size:10px;color:{"#1a1916" if is_active else "#807a70"};'
            f'margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
            f'· {it.split("/")[0].strip()}</div>'
            for it in items[:5]
        )

        if is_active:
            box_style    = "border:2px solid #2563a8;background:#d6e4f4;"
            eyebrow_col  = "#2563a8"
            company_html = (
                f'<div style="font-size:9px;font-family:monospace;color:#2563a8;'
                f'font-weight:700;margin-top:8px;letter-spacing:0.06em">'
                f'▶ {company_short}</div>'
            )
            matches_html = "".join(
                f'<div style="font-size:9px;color:#2563a8;margin-top:2px">'
                f'✓ {m.split("/")[0].strip()}</div>'
                for m in (matched_items or [])
            )
        else:
            opacity      = "0.55" if active_layer_index is not None else "1"
            box_style    = f"border:1px solid #d6cebe;background:#fff;opacity:{opacity};"
            eyebrow_col  = "#807a70"
            company_html = ""
            matches_html = ""

        cards += (
            f'<div style="flex:1;min-width:120px;max-width:240px;border-radius:6px;'
            f'padding:10px 12px;{box_style}">'
            f'<div style="font-family:monospace;font-size:9px;letter-spacing:0.12em;'
            f'text-transform:uppercase;color:{eyebrow_col}">Layer {idx}</div>'
            f'<div style="font-size:11.5px;font-weight:600;color:#1a1916;'
            f'margin:4px 0 2px 0;line-height:1.3">{layer_name}</div>'
            f'{items_html}{company_html}{matches_html}'
            f'</div>'
        )

    _render_html(
        f'<div style="display:flex;gap:8px;overflow-x:auto;'
        f'padding:2px 0 10px 0;align-items:flex-start;margin-top:8px">'
        f'{cards}</div>'
    )


# ── Supply chain graph (existing or admin-generated) ─────────────────────────
graph       = data_manager.get_supply_chain_graph(ticker)
sc_products = (graph or {}).get("products", [])   # available to the whole page

if graph:
    sc_col1, sc_col2 = st.columns([5, 1])
    with sc_col1:
        st.markdown(
            '<div class="eb-eyebrow" style="margin:6px 0 4px 0">'
            'Industry & Product Supply Chain</div>',
            unsafe_allow_html=True,
        )
    with sc_col2:
        if is_admin and st.button("🔄 Regen graph", key="eb_re_graph",
                                   use_container_width=True):
            with st.spinner(f"Regenerating supply chain for {company}…"):
                try:
                    new_graph = supply_chain_ui.generate_supply_chain_graph(
                        ticker, company)
                    if data_manager.upsert_supply_chain_graph(
                            ticker, company, new_graph):
                        st.success("✅ Supply chain regenerated.")
                        st.rerun()
                    else:
                        st.error("Generated but failed to save.")
                except RuntimeError as exc:
                    st.error(str(exc))
    supply_chain_ui.render_supply_chain_graph(graph, height=520)

    # ── Sector theme positions ────────────────────────────────────────────────
    # Batch-match all macro_sectors to stored themes (ONE AI call max), then
    # show which supply-chain layer this company occupies in each matched theme.
    macro_sectors = graph.get("macro_sectors", [])
    all_themes    = data_manager.get_all_sector_themes()

    if macro_sectors and all_themes is not None:
        _render_html(
            '<div class="eb-eyebrow" style="margin:20px 0 4px 0">'
            'Sector Theme Positions · 产业链位置</div>'
            '<div style="font-size:12px;color:var(--ink-3);margin-bottom:10px">'
            'Where this company fits within stored supply chain themes — '
            'based on its end-markets and products.</div>'
        )

        # ── 1. Batch match all sectors → themes (cached per ticker + theme count) ──
        _batch_key = f"cp_batch_{ticker}_{len(all_themes)}"
        if _batch_key not in st.session_state:
            st.session_state[_batch_key] = (
                sector_themes_mod.match_sectors_to_themes_batch(
                    macro_sectors, all_themes
                )
            )
        _sector_to_theme: dict = st.session_state[_batch_key]

        # ── 2. Deduplicate matched themes by theme_id ──
        _seen_ids:        set  = set()
        _no_layer_ids:    set  = set()   # theme IDs that matched but have no layers
        _matched_themes:  list = []   # [(theme_full, pos), ...]
        _unmatched_secs:  list = []   # [sector_str, ...]

        for _sector in macro_sectors:
            _tm = _sector_to_theme.get(_sector)
            if _tm is None:
                if _sector not in _unmatched_secs:
                    _unmatched_secs.append(_sector)
                continue
            if _tm["id"] in _seen_ids:
                continue
            # Themes with no layers: treat as unmatched so the sector stays visible
            if _tm["id"] in _no_layer_ids:
                if _sector not in _unmatched_secs:
                    _unmatched_secs.append(_sector)
                continue
            _tf = data_manager.get_sector_theme_by_id(_tm["id"])
            if not _tf or not _tf.get("layers"):
                _no_layer_ids.add(_tm["id"])
                if _sector not in _unmatched_secs:
                    _unmatched_secs.append(_sector)
                continue
            _seen_ids.add(_tm["id"])
            _pos = data_manager.get_chain_position(ticker, _tm["id"])
            _matched_themes.append((_tf, _pos))

        # ── 3. Single "Classify All" button for unclassified themes ──
        _unclassified = [tf for tf, pos in _matched_themes if pos is None]
        if is_admin and _unclassified:
            _n = len(_unclassified)
            if st.button(
                f"🔍 Classify All Layer Positions ({_n} theme{'s' if _n > 1 else ''})",
                key=f"cp_cls_all_{ticker}",
                type="primary",
            ):
                with st.spinner(f"Classifying across {_n} theme(s) — one AI call…"):
                    try:
                        _results = sector_themes_mod.classify_ticker_across_themes(
                            ticker, company, sc_products, _unclassified
                        )
                        _saved = 0
                        _skipped = 0
                        for _r in _results:
                            if _r.get("theme_id") is not None and _r.get("layer_index") is not None:
                                data_manager.upsert_chain_position(
                                    ticker, _r["theme_id"],
                                    _r.get("layer_index"),
                                    _r.get("matched_items", []),
                                )
                                _saved += 1
                            elif _r.get("theme_id") is not None:
                                _skipped += 1
                        msg = f"✅ Classified {_saved} theme(s)."
                        if _skipped:
                            msg += f" ({_skipped} skipped — AI could not determine layer, existing data preserved.)"
                        st.success(msg)
                        st.rerun()
                    except RuntimeError as exc:
                        st.error(str(exc))

        # ── 4. Render each matched theme in an expander ──
        for _tf, _pos in _matched_themes:
            _layers   = _tf.get("layers", [])
            _formal   = _tf.get("formal_name") or _tf.get("name", "")
            _theme_id = _tf["id"]

            _exp_label = (
                f"🔗 {_formal} — Layer {_pos['layer_index']} / {len(_layers)}"
                if _pos and _pos.get("layer_index") else
                f"🔗 {_formal} — position unknown"
            )
            with st.expander(_exp_label, expanded=(_pos is not None)):
                if _pos is not None:
                    _render_chain_position_strip(
                        _layers,
                        _pos.get("layer_index"),
                        company,
                        _pos.get("matched_items", []),
                    )
                    if is_admin and st.button(
                        "🔄 Reclassify",
                        key=f"cp_recls_{ticker}_{_theme_id}",
                        help="Re-run AI classification for this theme",
                    ):
                        with st.spinner("Reclassifying…"):
                            try:
                                _result = sector_themes_mod.classify_ticker_in_theme(
                                    ticker, company, sc_products, _tf,
                                )
                                if _result.get("layer_index") is not None:
                                    data_manager.upsert_chain_position(
                                        ticker, _theme_id,
                                        _result.get("layer_index"),
                                        _result.get("matched_items", []),
                                    )
                                    st.rerun()
                                else:
                                    st.warning("AI could not determine layer — existing classification preserved.")
                            except RuntimeError as exc:
                                st.error(str(exc))
                else:
                    st.caption(
                        "Layer position not yet classified. "
                        "Use the button above to classify all themes at once."
                    )
                    _render_chain_position_strip(_layers, None, company)

        # ── 5. Unmatched sectors — offer to generate a theme (admins only) ──
        for _sector in _unmatched_secs:
            with st.expander(
                f"📦 {_sector} — no theme loaded", expanded=False
            ):
                st.caption("No sector theme in the database matches this market yet.")
                if is_admin:
                    _raw_inp = _sector.split("/")[0].strip().lower()
                    if st.button(
                        f"🌱 Generate theme for '{_sector.split('/')[0].strip()}'",
                        key=f"cp_gen_{ticker}_{_sector[:24]}",
                    ):
                        with st.spinner("Generating sector theme…"):
                            try:
                                _tdata = sector_themes_mod.generate_sector_theme(_raw_inp)
                                data_manager.add_sector_theme(
                                    raw_input=_raw_inp,
                                    formal_name=_tdata["name"],
                                    layers_data=_tdata,
                                    created_by=user.get("username", "admin"),
                                )
                                # Preserve existing batch matches; inject the new one
                                # so the re-render doesn't trigger a fresh AI batch-match
                                # (which would scramble other sectors' assignments).
                                # Re-fetch all_themes so _future_key reflects the ACTUAL
                                # DB count — guards against UNIQUE-constraint conflicts
                                # where the insert silently fails and count is unchanged.
                                _themes_now    = data_manager.get_all_sector_themes()
                                _new_theme_rec = data_manager.get_sector_theme_by_raw_input(_raw_inp)
                                _carried = dict(st.session_state.get(_batch_key, {}))
                                if _new_theme_rec:
                                    _carried[_sector] = {"id": _new_theme_rec["id"]}
                                _future_key = f"cp_batch_{ticker}_{len(_themes_now)}"
                                st.session_state[_future_key] = _carried
                                if _future_key != _batch_key:
                                    st.session_state.pop(_batch_key, None)
                                st.success("✅ Theme generated.")
                                st.rerun()
                            except RuntimeError as exc:
                                st.error(str(exc))

    elif macro_sectors and not all_themes:
        st.caption(
            "No sector themes loaded yet. "
            "Build themes in Sector Explorer first — they'll appear here automatically."
        )

elif is_admin:
    st.markdown(
        '<div class="eb-card" style="margin-top:8px">'
        '<div class="eb-eyebrow" style="margin-bottom:6px">'
        'Industry & Product Supply Chain</div>'
        f'<div style="font-size:13px;color:var(--ink-2);margin-bottom:10px">'
        f'No supply chain graph exists yet for <strong>{ticker} {company}</strong>. '
        f'As an admin you can generate one — it will be saved and shared with all users.'
        '</div></div>',
        unsafe_allow_html=True,
    )
    if st.button("🧬 Generate Supply Chain", type="primary", key="eb_gen_graph"):
        with st.spinner(f"Generating supply chain for {company}…"):
            try:
                new_graph = supply_chain_ui.generate_supply_chain_graph(
                    ticker, company)
                if data_manager.upsert_supply_chain_graph(
                        ticker, company, new_graph):
                    st.success("✅ Supply chain generated.")
                    st.rerun()
                else:
                    st.error("Generated but failed to save.")
            except RuntimeError as exc:
                st.error(str(exc))
else:
    st.markdown(
        '<div class="eb-card" style="margin-top:8px;color:var(--ink-3);'
        'font-size:13px">No supply chain graph available for this stock '
        '(an admin can add one).</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 01 — SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════

pe   = (daily or {}).get("pe_ttm") or (daily or {}).get("pe")
pb   = (daily or {}).get("pb")
ps   = (daily or {}).get("ps_ttm") or (daily or {}).get("ps")
dvr  = (daily or {}).get("dv_ratio")
roa  = fina.iloc[0]["roa"] if (fina is not None and not fina.empty and "roa" in fina.columns) else None
or_yoy = fina.iloc[0]["or_yoy"] if (fina is not None and not fina.empty and "or_yoy" in fina.columns) else None
np_yoy = fina.iloc[0]["netprofit_yoy"] if (fina is not None and not fina.empty and "netprofit_yoy" in fina.columns) else None

# ── ROE: split annual (latest 12-31 report) vs latest accumulated quarter ──
# Chinese periodic reports are CUMULATIVE: Q1 covers Jan-Mar, H1 = Jan-Jun,
# Q3 = Jan-Sep, Annual = Jan-Dec. You cannot annualise Q1 to get full-year ROE.
# We show both values so the reader can judge trend across the year.
roe_q        = None   # latest quarter (accumulated up to that date)
roe_q_period = "—"
roe_annual        = None   # most recent full-year (end_date ends '1231')
roe_annual_period = "—"
if fina is not None and not fina.empty and "roe" in fina.columns:
    roe_q = fina.iloc[0]["roe"]
    roe_q_period = str(fina.iloc[0].get("end_date", ""))
    annual_rows = fina[fina["end_date"].astype(str).str.endswith("1231")]
    if not annual_rows.empty:
        roe_annual = annual_rows.iloc[0]["roe"]
        roe_annual_period = str(annual_rows.iloc[0]["end_date"])

# Keep a single `roe` alias so the SWOT metrics summary still works
roe = roe_annual if roe_annual is not None else roe_q

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
    <div class="eb-kpi">
      <div class="eb-kpi-label">ROE · Annual</div>
      <div class="eb-kpi-value">{_fmt_pct(roe_annual, 1)}</div>
      <div class="eb-kpi-sub">{roe_annual_period or "—"}</div>
    </div>
    <div class="eb-kpi">
      <div class="eb-kpi-label">ROE · Latest Qtr</div>
      <div class="eb-kpi-value">{_fmt_pct(roe_q, 1)}</div>
      <div class="eb-kpi-sub">{roe_q_period or "—"} · cumulative YTD</div>
    </div>
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

# Cached — no spinner needed
analysis_df, raw_df = _technical_signal(ticker)

if analysis_df is not None and not analysis_df.empty:
    last250 = analysis_df.tail(250).copy()

    # Use string dates as categorical x-axis so weekends and Chinese public
    # holidays (which have no trading data) produce no gaps on the chart.
    def _xdates(df_or_series):
        idx = df_or_series.index
        if hasattr(idx, "strftime"):
            return idx.strftime("%Y-%m-%d").tolist()
        return [str(d) for d in idx]

    x_all = _xdates(last250)

    # Compute strategy summary BEFORE building the chart so we can overlay S/R
    summary = trading_strategy.build_strategy_summary(raw_df, analysis_df)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x_all, open=last250["Open"], high=last250["High"],
        low=last250["Low"], close=last250["Close"],
        name="Price",
        increasing=dict(line=dict(color="#c6432a")),
        decreasing=dict(line=dict(color="#2f8a4f")),
    ))
    if "MA20" in last250.columns:
        fig.add_trace(go.Scatter(x=x_all, y=last250["MA20"], name="MA20",
                                 line=dict(color="#b8800f", width=1.2, dash="dot")))
    if "MA50" in last250.columns:
        fig.add_trace(go.Scatter(x=x_all, y=last250["MA50"], name="MA50",
                                 line=dict(color="#2563a8", width=1.5)))

    # ── S/R overlay ───────────────────────────────────────────────────────────
    # xref="paper" makes the lines span the full plot width regardless of the
    # categorical x-axis.  Right-side annotations sit just outside the plot area.
    # Chinese convention: red (涨) for resistance (above price), green (跌) for
    # support (below price) — consistent with stop/targets table convention.
    if summary:
        _res = summary.get("resistances", [])
        _sup = summary.get("supports", [])

        for idx_r, r_lvl in enumerate(_res):
            r_price = r_lvl["price"]
            strength = min(int(r_lvl.get("strength", 1)), 3)
            lw = 0.8 + strength * 0.3      # 1.1 → 1.4 → 1.7 px by strength
            fig.add_shape(
                type="line", xref="paper", yref="y",
                x0=0, x1=1, y0=r_price, y1=r_price,
                line=dict(color="#c6432a", width=lw, dash="dash"),
            )
            fig.add_annotation(
                x=1.01, xref="paper", y=r_price, yref="y",
                text=f"R{idx_r + 1} ¥{r_price:.2f}",
                showarrow=False, xanchor="left",
                font=dict(size=9.5, color="#c6432a",
                          family="JetBrains Mono, ui-monospace, monospace"),
            )

        for idx_s, s_lvl in enumerate(_sup):
            s_price = s_lvl["price"]
            strength = min(int(s_lvl.get("strength", 1)), 3)
            lw = 0.8 + strength * 0.3
            fig.add_shape(
                type="line", xref="paper", yref="y",
                x0=0, x1=1, y0=s_price, y1=s_price,
                line=dict(color="#2f8a4f", width=lw, dash="dash"),
            )
            fig.add_annotation(
                x=1.01, xref="paper", y=s_price, yref="y",
                text=f"S{idx_s + 1} ¥{s_price:.2f}",
                showarrow=False, xanchor="left",
                font=dict(size=9.5, color="#2f8a4f",
                          family="JetBrains Mono, ui-monospace, monospace"),
            )

    if "Signal_Accumulation" in last250.columns:
        acc = last250[last250["Signal_Accumulation"] == True]
        if not acc.empty:
            fig.add_trace(go.Scatter(x=_xdates(acc), y=acc["Low"] * 0.98,
                mode="markers", name="Accumulation",
                marker=dict(color="#b8800f", size=8, symbol="circle")))
    if "Squeeze_Fired_Bullish" in last250.columns:
        bull = last250[last250["Squeeze_Fired_Bullish"] == True]
        if not bull.empty:
            fig.add_trace(go.Scatter(x=_xdates(bull), y=bull["Low"] * 0.95,
                mode="markers", name="Squeeze Fired",
                marker=dict(color="#2f8a4f", size=12, symbol="triangle-up")))

    fig.update_layout(
        height=420, template="plotly_white",
        # Right margin widened to 90px so S/R annotations don't get clipped
        margin=dict(l=10, r=90, t=10, b=10),
        xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#f7f3ec", paper_bgcolor="#f7f3ec",
    )
    # type='category' treats each trading day as a discrete slot — no gaps
    # for weekends or Chinese public holidays.
    fig.update_xaxes(type="category", nticks=10, gridcolor="#d6cebe")
    fig.update_yaxes(gridcolor="#d6cebe", title="Price (¥)")
    st.plotly_chart(fig, use_container_width=True)

    sig_label, sig_class = _entry_signal_label(analysis_df)
    pill_class = f"eb-pill {sig_class}" if sig_class else "eb-pill"
    st.markdown(f'<div style="margin-top:8px"><span class="{pill_class}">'
                f'Latest signal · {sig_label}</span></div>', unsafe_allow_html=True)

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
        comp_rows = []
        for k in ("trend", "momentum", "rsi", "volatility", "custom"):
            c = comps[k]
            v, mx = c["value"], c["max"]
            width_pct = abs(v) / mx * 50  # half the bar at most (50% of 100%)
            if v >= 0:
                bar_left, bar_color = 50, "#2f8a4f"
            else:
                bar_left, bar_color = 50 - width_pct, "#c6432a"
            reason = c.get("reason", "")
            comp_rows.append(
                f'<div class="bar-row">'
                f'<div>'
                f'<div class="bar-label">{c["label"]}</div>'
                f'<div class="bar-reason">{reason}</div>'
                f'</div>'
                f'<div class="bar-track">'
                f'<div class="bar-mid"></div>'
                f'<div class="bar-fill" style="left:{bar_left}%;width:{width_pct}%;background:{bar_color}"></div>'
                f'</div>'
                f'<div class="bar-val">{v:+d}</div>'
                f'</div>'
            )
        comp_html = "".join(comp_rows)

        # ── Stop/targets card ────────────────────────────────────────────────
        # Chinese A-share colour convention: red (涨) = up, green (跌) = down.
        # Stop is a downside level → green.  Targets are upside → red.
        _cn_up   = "color:#c6432a"   # red  = 涨 (positive / up move)
        _cn_down = "color:#2f8a4f"   # green = 跌 (negative / down move)
        stop_targets_html = f"""
        <div class="eb-card">
          <div class="eb-eyebrow" style="margin-bottom:6px">Stop &amp; Targets
            <span style="font-size:9px;letter-spacing:0.08em;color:var(--ink-3);
                         margin-left:8px">红涨绿跌 · red=up · green=down</span>
          </div>
          <table class="eb-table" style="font-size:13px">
            <thead><tr>
              <th>Level</th><th>Price</th><th>Δ from entry</th><th>Role</th>
            </tr></thead>
            <tbody>
            <tr>
              <td>Current</td>
              <td class="num"><strong>¥{price:.2f}</strong></td>
              <td class="num" style="color:var(--ink-3)">ATR ¥{atr:.2f}</td>
              <td style="font-size:11px;color:var(--ink-3)">entry reference</td>
            </tr>
            <tr>
              <td><strong>Stop Loss</strong></td>
              <td class="num">¥{stg['stop']:.2f}</td>
              <td class="num" style="{_cn_down}">−{stg['risk_pct']:.1f}%</td>
              <td style="font-size:11px;color:var(--ink-3)">exit if breached; invalidates setup</td>
            </tr>
            <tr>
              <td>T1 · Partial exit</td>
              <td class="num">¥{stg['t1']:.2f}</td>
              <td class="num" style="{_cn_up}">+{stg['reward_t1_pct']:.1f}%</td>
              <td style="font-size:11px;color:var(--ink-3)">scale out ~⅓ at first resistance</td>
            </tr>
            <tr>
              <td>T2 · Main target</td>
              <td class="num">¥{stg['t2']:.2f}</td>
              <td class="num" style="{_cn_up}">+{stg['reward_t2_pct']:.1f}%</td>
              <td style="font-size:11px;color:var(--ink-3)">primary exit; basis for R:R</td>
            </tr>
            <tr>
              <td>T3 · Full run</td>
              <td class="num">¥{stg['t3']:.2f}</td>
              <td class="num" style="{_cn_up}">+{stg['reward_t3_pct']:.1f}%</td>
              <td style="font-size:11px;color:var(--ink-3)">if momentum carries through next resistance</td>
            </tr>
            <tr>
              <td><strong>R : R</strong></td>
              <td class="num" colspan="3"><strong>1 : {stg['rr']:.2f}</strong>
                <span style="color:var(--ink-3);font-size:11px"> · measured to T2</span>
              </td>
            </tr>
            </tbody>
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

        # Build segments. Labels go INSIDE the colored bar (overlaid in white)
        # so they never collide with the price endpoints below the bar.
        bar_segments = ""
        for b in all_zones:
            left  = (b["low"]  - z_min) / z_range * 100
            width = (b["high"] - b["low"]) / z_range * 100
            # Inline label only if the band is wide enough to fit the text
            label_html = ""
            if width >= 12:
                label_html = (
                    f'<span style="position:absolute;left:0;right:0;top:50%;'
                    f'transform:translateY(-50%);text-align:center;'
                    f'font-family:ui-monospace,monospace;font-size:9px;'
                    f'color:rgba(255,255,255,0.95);letter-spacing:0.06em;'
                    f'text-transform:uppercase;white-space:nowrap;'
                    f'overflow:hidden;text-overflow:ellipsis;'
                    f'text-shadow:0 1px 2px rgba(0,0,0,0.3);pointer-events:none">'
                    f'{b["label"]}</span>'
                )
            bar_segments += (
                f'<div style="position:absolute;left:{left}%;width:{width}%;'
                f'top:0;height:38px;background:{b["color"]};'
                f'border-right:1px solid rgba(0,0,0,0.15)">{label_html}</div>'
            )

        # Build anchor note — shows which S/R levels were used as zone boundaries
        _s1p = sups[0]["price"] if len(sups) >= 1 else price - 2 * atr
        _s2p = sups[1]["price"] if len(sups) >= 2 else _s1p - 2 * atr
        _r1p = res[0]["price"]  if len(res)  >= 1 else price + 2 * atr
        _r2p = res[1]["price"]  if len(res)  >= 2 else _r1p + 2 * atr
        anchor_note = (
            f'<div style="font-family:JetBrains Mono,monospace;font-size:10px;'
            f'color:var(--ink-3);margin-bottom:10px;letter-spacing:0.03em">'
            f'Anchored to: '
            f'<span style="color:var(--pos)">S2 ¥{_s2p:.2f}</span>'
            f' · <span style="color:var(--pos)">S1 ¥{_s1p:.2f}</span>'
            f' · <span style="color:var(--neg)">R1 ¥{_r1p:.2f}</span>'
            f' · <span style="color:var(--neg)">R2 ¥{_r2p:.2f}</span>'
            f' · ATR ¥{atr:.2f}</div>'
        )

        marker_left = (price - z_min) / z_range * 100
        zone_bar_html = f"""
        <div class="eb-card">
          <div class="eb-eyebrow" style="margin-bottom:8px">Entry Zone</div>
          {anchor_note}
          <div style="display:flex;justify-content:space-between;
                      font-size:13px;margin-bottom:8px;gap:8px">
            <div>Current sits in <strong>{zone['label']}</strong></div>
            <div style="color:var(--ink-2);text-align:right">{zone['action']}</div>
          </div>
          <div style="position:relative;height:38px;background:#efe9df;
                      border-radius:3px;overflow:visible;margin-top:24px">
            {bar_segments}
            <div style="position:absolute;left:{marker_left}%;top:-6px;
                        width:2px;height:50px;background:#1a1916;z-index:5"></div>
            <div style="position:absolute;left:calc({marker_left}% - 30px);
                        top:-22px;font-family:JetBrains Mono,monospace;font-size:11px;
                        font-weight:600;color:#1a1916;z-index:6;width:60px;text-align:center">
              ¥{price:.2f}</div>
          </div>
          <div style="display:flex;justify-content:space-between;
                      font-family:JetBrains Mono,monospace;font-size:10px;
                      color:var(--ink-3);margin-top:8px">
            <span>¥{z_min:.2f}</span>
            <span>¥{z_max:.2f}</span>
          </div>
        </div>
        """

        # ── Render in a sandboxed iframe so markdown processing doesn't
        #    chew our nested HTML. Self-contained doc with inlined CSS
        #    (the parent page's :root variables don't reach the iframe).
        strategy_doc = f"""
<!DOCTYPE html>
<html><head><style>
:root {{ --bg:#f7f3ec; --bg-2:#efe9df; --bg-3:#e6dfd2;
        --ink:#1a1916; --ink-2:#4d4942; --ink-3:#807a70;
        --rule:#d6cebe; --rule-2:#b8b09f;
        --pos:#2f8a4f; --neg:#c6432a; --warn:#b8800f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:0; background:var(--bg); color:var(--ink);
       font-family: Georgia, "Newsreader", serif; font-size:14px; }}
.eb-num {{ font-family:"JetBrains Mono",ui-monospace,monospace;
          font-feature-settings:"tnum","zero"; }}
.eb-card {{ background:#fff; border:1px solid var(--rule);
           border-radius:4px; padding:16px 18px; }}
.eb-eyebrow {{ font-family:ui-monospace,monospace; font-size:11px;
              letter-spacing:0.18em; text-transform:uppercase;
              color:var(--ink-3); }}
.eb-pill {{ display:inline-block; padding:2px 8px; border-radius:999px;
           font-family:ui-monospace,monospace; font-size:10.5px;
           border:1px solid var(--rule-2); color:var(--ink-2);
           background:#fff; letter-spacing:0.06em; text-transform:uppercase; }}
.eb-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.eb-table th {{ text-align:left; font-family:ui-monospace,monospace;
               font-size:10.5px; text-transform:uppercase;
               letter-spacing:0.1em; color:var(--ink-3);
               border-bottom:1px solid var(--rule); padding:8px 10px;
               font-weight:500; }}
.eb-table td {{ padding:8px 10px; border-bottom:1px solid var(--rule); }}
.eb-table td.num {{ font-family:"JetBrains Mono",ui-monospace,monospace;
                   font-feature-settings:"tnum"; text-align:right; }}
.eb-table tr:last-child td {{ border-bottom:0; }}
.eb-table .pos {{ color:var(--pos); }}
.eb-table .neg {{ color:var(--neg); }}
.row {{ display:grid; grid-template-columns:1fr 1fr; gap:14px;
       margin-bottom:14px; }}
.bar-row {{ display:grid; grid-template-columns:130px 1fr 42px; gap:8px;
           align-items:center; padding:6px 0; font-size:12px; }}
.bar-label {{ font-family:ui-monospace,monospace; font-size:10.5px;
             letter-spacing:0.08em; color:var(--ink-3); text-transform:uppercase; }}
.bar-reason {{ font-size:9px; color:var(--ink-3); margin-top:3px; opacity:0.75;
              font-style:italic; white-space:nowrap; overflow:hidden;
              text-overflow:ellipsis; font-family:Georgia,serif; }}
.bar-track {{ position:relative; height:10px; background:var(--bg-2);
             border-radius:2px; }}
.bar-mid {{ position:absolute; left:50%; top:-2px; width:1px; height:14px;
           background:var(--ink-3); }}
.bar-fill {{ position:absolute; top:0; height:10px; border-radius:2px; }}
.bar-val {{ text-align:right; font-size:12px; color:var(--ink-2);
           font-family:"JetBrains Mono",ui-monospace,monospace; }}
</style></head><body>

<div class="row">
  <div class="eb-card">
    <div class="eb-eyebrow" style="margin-bottom:6px">Composite Signal Score</div>
    <div style="display:flex;align-items:center;gap:18px">
      {gauge_svg}
      <div style="flex:1">{comp_html}</div>
    </div>
  </div>
  {stop_targets_html}
</div>
<div class="row">
  {sr_html}
  {zone_bar_html}
</div>

</body></html>
"""
        components.html(strategy_doc, height=680, scrolling=False)
else:
    st.warning("No technical data available.")

# ── Links to related pages ───────────────────────────────────────────────────
_tc1, _tc2, _tc3 = st.columns([4, 1, 1])
with _tc2:
    page_link_button(
        "single-stock-analysis",
        "Full Analysis →",
        params={"ticker": ticker},
        use_container_width=True,
        help="Open Technical Analysis pre-loaded with this stock. "
             "Middle/right-click → Open in new tab.",
    )
with _tc3:
    page_link_button(
        "lead-lag",
        "Lead-Lag →",
        params={"ticker": ticker},
        use_container_width=True,
        help="Open Lead-Lag Analysis pre-loaded with this stock. "
             "Middle/right-click → Open in new tab.",
    )


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

# ── Latest Earnings Commentary (forecast + express, company's own words) ────
@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def _earnings_commentary(ticker_):
    return {
        "forecast": data_manager.fetch_forecast(ticker_, periods=4),
        "express":  data_manager.fetch_express(ticker_,  periods=4),
    }

ec = _earnings_commentary(ticker)
fc_df, ex_df = ec["forecast"], ec["express"]

# Determine the latest formally-reported quarter from the income statement we
# already pulled. We use this as the cutoff:
#   - end_date >  latest_reported → upcoming (worth seeing prominently)
#   - end_date == latest_reported → past quarter (worth seeing, marked as such)
#   - end_date <  latest_reported → stale, skip silently
latest_reported_period = (
    str(inc.iloc[0]["end_date"]) if (inc is not None and not inc.empty) else ""
)

def _filing_status(filing_period_str):
    """Return ('upcoming' | 'past' | 'stale', display_label)."""
    if not filing_period_str or not latest_reported_period:
        return "upcoming", ""  # treat unknowns as worth showing
    if filing_period_str >  latest_reported_period: return "upcoming", "Upcoming Quarter"
    if filing_period_str == latest_reported_period: return "past",     "Past Quarter · already reported"
    return "stale", ""


def _filing_too_old(row: "pd.Series") -> bool:
    """
    Return True when the filing is more than 90 days old by wall-clock time.
    We prefer ann_date (announcement date) over end_date (reporting period end)
    because ann_date is when the filing became public information.
    Falls back to end_date if ann_date is absent.
    Date format from Tushare: YYYYMMDD (8-char string).
    """
    _today = datetime.utcnow().date()
    for col in ("ann_date", "end_date"):
        raw = str(row.get(col) or "")
        if len(raw) == 8 and raw.isdigit():
            try:
                filing_date = datetime.strptime(raw, "%Y%m%d").date()
                return (_today - filing_date).days > 90
            except ValueError:
                continue
    return False  # unknown → assume fresh enough


# Pre-filter both DataFrames so we never render stale rows
def _filter_status(df):
    if df is None or df.empty:
        return df, None  # (df, status)
    row0 = df.iloc[0]
    # Wall-clock recency check — supersedes the formal-report comparison
    if _filing_too_old(row0):
        return None, "stale"
    end_date_str = str(row0.get("end_date") or "")
    status, _label = _filing_status(end_date_str)
    if status == "stale":
        return None, "stale"
    return df, status

fc_df, fc_status = _filter_status(fc_df)
ex_df, ex_status = _filter_status(ex_df)

# Show only if at least one survived the staleness filter
if fc_df is not None or ex_df is not None:
    cards_html = ""

    # Forecast card — preliminary earnings warning with narrative
    if fc_df is not None and not fc_df.empty:
        f0 = fc_df.iloc[0]
        ann = str(f0.get("ann_date") or "")
        ann_fmt = f"{ann[:4]}-{ann[4:6]}-{ann[6:8]}" if len(ann) == 8 else "—"
        period = str(f0.get("end_date") or "")
        period_fmt = f"{period[:4]}-{period[4:6]}" if len(period) == 8 else "—"
        ftype = (f0.get("type") or "").strip() or "—"
        pmin, pmax = f0.get("p_change_min"), f0.get("p_change_max")
        npmin, npmax = f0.get("net_profit_min"), f0.get("net_profit_max")
        summary_txt = (f0.get("summary") or "").strip()
        reason_txt  = (f0.get("change_reason") or "").strip()

        range_str = ""
        if pd.notna(pmin) and pd.notna(pmax):
            range_str = f"{float(pmin):+.1f}% to {float(pmax):+.1f}%"
        elif pd.notna(pmin):
            range_str = f"≥ {float(pmin):+.1f}%"

        np_range = ""
        if pd.notna(npmin) and pd.notna(npmax):
            np_range = f"{float(npmin)/1e4:.0f}–{float(npmax)/1e4:.0f} 万元"

        # Color the type pill: 预增/续盈/扭亏/略增 = pos; 预减/续亏/略减/首亏 = neg
        type_class = "pill"
        if any(k in ftype for k in ("预增", "续盈", "扭亏", "略增")):
            type_class = "pill pos"
        elif any(k in ftype for k in ("预减", "续亏", "略减", "首亏")):
            type_class = "pill neg"
        elif "不确定" in ftype:
            type_class = "pill warn"

        # Build narrative block — show change_reason if available, else summary, else nothing
        narrative_html = ""
        if reason_txt:
            narrative_html = (
                f'<div style="font-size:10.5px;font-family:ui-monospace,monospace;'
                f'letter-spacing:0.1em;text-transform:uppercase;color:var(--ink-3);'
                f'margin:14px 0 6px 0">Why (company\'s words)</div>'
                f'<div style="font-size:13px;line-height:1.6;color:var(--ink);'
                f'white-space:pre-wrap">{reason_txt}</div>'
            )
        elif summary_txt:
            narrative_html = (
                f'<div style="font-size:10.5px;font-family:ui-monospace,monospace;'
                f'letter-spacing:0.1em;text-transform:uppercase;color:var(--ink-3);'
                f'margin:14px 0 6px 0">Summary</div>'
                f'<div style="font-size:13px;line-height:1.6;color:var(--ink);'
                f'white-space:pre-wrap">{summary_txt}</div>'
            )

        # Status pill — Upcoming (warn) vs Past (neutral, dimmed)
        period_pill_html = ""
        if fc_status == "upcoming":
            period_pill_html = '<span class="eb-pill warn">Upcoming Quarter</span>'
        elif fc_status == "past":
            period_pill_html = ('<span class="eb-pill" style="opacity:0.7">'
                                'Past Quarter · already reported</span>')

        cards_html += f"""
        <div class="eb-card" style="margin-top:14px">
          <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:10px">
            <div>
              <div class="eb-eyebrow">业绩预告 · Earnings Forecast</div>
              <div style="font-size:13px;color:var(--ink-2);margin-top:2px">
                Period {period_fmt} · Filed {ann_fmt}
              </div>
            </div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end">
              {period_pill_html}
              <span class="eb-pill {type_class.replace('pill ','').strip()}">{ftype}</span>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px">
            <div>
              <div class="eb-eyebrow">Net Profit YoY</div>
              <div class="eb-num" style="font-size:22px">{range_str or '—'}</div>
            </div>
            <div>
              <div class="eb-eyebrow">Net Profit Range</div>
              <div class="eb-num" style="font-size:22px">{np_range or '—'}</div>
            </div>
          </div>
          {narrative_html}
        </div>
        """

    # Express card — full preliminary results with headline numbers
    if ex_df is not None and not ex_df.empty:
        e0 = ex_df.iloc[0]
        ann = str(e0.get("ann_date") or "")
        ann_fmt = f"{ann[:4]}-{ann[4:6]}-{ann[6:8]}" if len(ann) == 8 else "—"
        period = str(e0.get("end_date") or "")
        period_fmt = f"{period[:4]}-{period[4:6]}" if len(period) == 8 else "—"

        rev = e0.get("revenue")
        ni  = e0.get("n_income")
        roe = e0.get("diluted_roe") if pd.notna(e0.get("diluted_roe")) else e0.get("yoy_roe")

        yoy_sales = e0.get("yoy_sales")
        yoy_np    = e0.get("yoy_dedu_np") if pd.notna(e0.get("yoy_dedu_np")) else None

        def _ex_metric(label, val_yi, yoy_pct):
            v_str = f"{val_yi:.2f} 亿" if val_yi is not None and pd.notna(val_yi) else "—"
            yoy_str = ""
            if yoy_pct is not None and pd.notna(yoy_pct):
                cls = "pos" if float(yoy_pct) >= 0 else "neg"
                yoy_str = (f'<span class="{cls}" style="font-size:12px;'
                           f'margin-left:8px">{float(yoy_pct):+.1f}% YoY</span>')
            return (f'<div><div class="eb-eyebrow">{label}</div>'
                    f'<div class="eb-num" style="font-size:22px">{v_str}{yoy_str}</div></div>')

        rev_yi = float(rev) / 1e8 if pd.notna(rev) else None
        ni_yi  = float(ni) / 1e8 if pd.notna(ni) else None

        ex_period_pill_html = ""
        if ex_status == "upcoming":
            ex_period_pill_html = '<span class="eb-pill warn">Upcoming Quarter</span>'
        elif ex_status == "past":
            ex_period_pill_html = ('<span class="eb-pill" style="opacity:0.7">'
                                   'Past Quarter · already reported</span>')

        cards_html += f"""
        <div class="eb-card" style="margin-top:14px">
          <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:10px">
            <div>
              <div class="eb-eyebrow">业绩快报 · Earnings Express</div>
              <div style="font-size:13px;color:var(--ink-2);margin-top:2px">
                Period {period_fmt} · Filed {ann_fmt}
              </div>
            </div>
            {ex_period_pill_html}
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:14px">
            {_ex_metric("Revenue", rev_yi, yoy_sales)}
            {_ex_metric("Net Profit", ni_yi, yoy_np)}
            <div><div class="eb-eyebrow">Diluted ROE</div>
                 <div class="eb-num" style="font-size:22px">
                   {f"{float(roe):.2f}%" if pd.notna(roe) else "—"}
                 </div></div>
          </div>
        </div>
        """

    has_upcoming = (fc_status == "upcoming") or (ex_status == "upcoming")
    section_title = ("Upcoming Earnings Preview" if has_upcoming
                     else "Most Recent Earnings Filings")
    section_blurb = (
        "The company's own pre-announcement for a quarter that hasn't been "
        "formally reported yet."
        if has_upcoming else
        "The company's pre-announcement filings for the quarter just reported. "
        "Useful as a sanity-check companion to the formal numbers above."
    )

    st.markdown(
        f"""
        <h3 class="eb-h3" style="margin-top:24px">{section_title}</h3>
        <div style="font-size:13px;color:var(--ink-2);max-width:780px;line-height:1.55;
                    margin-top:-4px;margin-bottom:6px">
          {section_blurb}
        </div>
        {cards_html}
        """,
        unsafe_allow_html=True,
    )


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
_roe_label = (f"ROE annual ({roe_annual_period[:4]})" if roe_annual is not None
              else f"ROE Q ({roe_q_period})")
metrics_summary = "\n".join([
    f"- P/E (TTM): {_fmt_num(pe, 1)}",
    f"- P/B: {_fmt_num(pb, 2)}",
    f"- {_roe_label}: {_fmt_pct(roe, 1)}",
    f"- ROE latest quarter ({roe_q_period}): {_fmt_pct(roe_q, 1)} (cumulative YTD, not annualised)",
    f"- Gross margin (TTM): {_fmt_pct(metrics['gross_margin_ttm_pct'], 1)}",
    f"- Net margin (TTM): {_fmt_pct(metrics['net_margin_ttm_pct'], 1)}",
    f"- Revenue YoY: {_fmt_pct(or_yoy, 1, signed=True)}",
    f"- Net profit YoY: {_fmt_pct(np_yoy, 1, signed=True)}",
    f"- Net debt (short+long borrowings + bonds − cash): {_fmt_yi(metrics['net_debt_yi'])} 亿元",
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
# SECTION 05 — REVENUE SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="eb-section">
  <div class="eb-eyebrow">05 · Revenue Segmentation</div>
  <div class="eb-h2">Where the revenue actually comes from.</div>
  <div class="eb-kicker">Product and geographic breakdown from Tushare fina_mainbz,
    with year-over-year changes where prior-year data is available.</div>
</div>
""", unsafe_allow_html=True)


def _seg_table_html(df: "pd.DataFrame", heading: str) -> str | None:
    """
    Build an HTML table card for one mainbz slice (product OR geographic).
    Returns None if df is empty. Includes YoY column when prior-year period exists.
    """
    if df is None or df.empty:
        return None

    latest_period = str(df["end_date"].iloc[0])
    latest = df[df["end_date"] == latest_period].copy()
    total = latest["bz_sales"].sum()
    latest["_pct"] = (latest["bz_sales"] / total * 100) if total > 0 else 0.0
    latest = latest.sort_values("bz_sales", ascending=False).head(10)

    # Prior-year same period (YYYYMMDD → (YYYY-1)MMDD)
    prior_map: dict = {}
    try:
        prior_period = f"{int(latest_period[:4]) - 1}{latest_period[4:]}"
        prior_df = df[df["end_date"] == prior_period]
        if not prior_df.empty:
            prior_map = prior_df.set_index("bz_item")["bz_sales"].to_dict()
    except Exception:
        pass

    has_yoy = bool(prior_map)

    # Header
    header_cols = (
        "<th>Segment</th><th>Revenue (亿)</th><th>Share</th><th>YoY</th>"
        if has_yoy else
        "<th>Segment</th><th>Revenue (亿)</th><th>Share</th>"
    )

    rows = ""
    for _, r in latest.iterrows():
        item  = r["bz_item"]
        sales = float(r["bz_sales"])
        pct   = float(r["_pct"])
        yoy_td = ""
        if has_yoy:
            prev = prior_map.get(item)
            if prev and prev > 0:
                yoy = (sales - prev) / prev * 100
                cls = "pos" if yoy > 0 else ("neg" if yoy < 0 else "")
                yoy_td = f'<td class="num {cls}">{_fmt_pct(yoy, 1, signed=True)}</td>'
            else:
                yoy_td = '<td class="num">—</td>'
        rows += (
            f"<tr>"
            f"<td>{item}</td>"
            f'<td class="num">{_fmt_yi(sales / 1e8)}</td>'
            f'<td class="num">{_fmt_pct(pct, 1)}</td>'
            f"{yoy_td}"
            f"</tr>"
        )

    period_fmt = (
        f"{latest_period[:4]}-{latest_period[4:6]}-{latest_period[6:]}"
        if len(latest_period) == 8 else latest_period
    )

    return f"""
<div class="eb-card" style="margin-bottom:16px">
  <div class="eb-eyebrow" style="margin-bottom:6px">{heading} · Period {period_fmt}</div>
  <table class="eb-table">
    <thead><tr>{header_cols}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
"""


_seg_prod = _seg_table_html(mainbz,     "By Product")
_seg_geo  = _seg_table_html(mainbz_geo, "By Geography")

if _seg_prod or _seg_geo:
    _render_html((_seg_prod or "") + (_seg_geo or ""))
else:
    st.info("No revenue segmentation data available for this company.")


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

@st.fragment
def _competitors_section():
    """
    Wrapped in @st.fragment so 'Regen peers' and 'Save curation' only
    rerender this block — not the whole page.

    IMPORTANT: the comparison table renders from the SAVED peer list
    (whatever's persisted in equity_brief_cache), NOT from live checkbox
    state.  Toggling a checkbox does NOT rebuild the table — it only
    affects the next save.  This means:
      - Unchecking a peer is instant (no Tushare calls).
      - Clicking 'Save' persists the curated set + reruns the fragment;
        the table then re-renders for the new (smaller) saved peer list.
      - Per-peer fundamentals are cached in _peer_row_data, so even on a
        full re-render only the peers we haven't seen recently hit Tushare.
    """
    ccol1, ccol2 = st.columns([5, 1])
    with ccol2:
        if is_admin and st.button("🔄 Regen peers", key="eb_re_peers",
                                  use_container_width=True):
            with st.spinner("Regenerating peer set…"):
                try:
                    equity_brief.get_competitors(ticker, company, industry,
                                                 force_refresh=True,
                                                 core_products=sc_products)
                    st.rerun(scope="fragment")
                except Exception as exc:
                    st.error(str(exc))

    try:
        with st.spinner("Loading peers…"):
            comps = equity_brief.get_competitors(ticker, company, industry,
                                                core_products=sc_products)
        peer_list = comps["payload"].get("competitors", [])
    except Exception as exc:
        st.error(f"Peer discovery failed: {exc}")
        peer_list = []

    if not peer_list:
        st.info("No validated peers were returned.")
        return

    # ── Comparison table — built from the SAVED peer list ───────────────────
    target_row = {
        "ticker": ticker, "name": company, "is_target": True,
        "pe": pe, "pb": pb, "ev_ebitda": metrics["ev_ebitda"],
        "rev_yoy": or_yoy, "np_yoy": np_yoy,
        "roe_annual": roe_annual, "roe_annual_period": roe_annual_period,
        "roe_q": roe_q, "roe_q_period": roe_q_period,
        "net_debt_ebitda": metrics["net_debt_ebitda"],
        "signal": _entry_signal_label(analysis_df),
    }
    table_rows = [target_row]

    with st.spinner(f"Loading peer fundamentals ({len(peer_list)})…"):
        for p in peer_list:
            row = _peer_row_data(p["ticker"])     # ← cached, fast on rerun
            if row is None:
                continue
            row["name"] = p["name"]
            row["is_target"] = False
            pana, _ = _technical_signal(p["ticker"])   # also cached
            row["signal"] = _entry_signal_label(pana)
            table_rows.append(row)

    rows_html = ""
    for r in table_rows:
        sig_lbl, sig_cls = r["signal"]
        sig_pill = f'<span class="eb-pill {sig_cls}">{sig_lbl}</span>' if sig_cls else sig_lbl
        target_marker = ' <span class="eb-pill">Target</span>' if r["is_target"] else ""
        annual_yr   = (r.get("roe_annual_period") or "")[:4] or "—"
        q_lbl       = (r.get("roe_q_period") or "")
        q_lbl_short = f"{q_lbl[:4]}-{q_lbl[4:6]}" if len(q_lbl) >= 6 else q_lbl or "—"
        if r["is_target"]:
            _name_cell = f'<strong>{r["ticker"]} {r["name"]}</strong>{target_marker}'
        else:
            _name_cell = (
                f'<a href="?ticker={r["ticker"]}" '
                f'style="color:var(--primary);text-decoration:underline;'
                f'text-underline-offset:3px;font-weight:600;">'
                f'{r["ticker"]} {r["name"]}</a>'
            )
        rows_html += f"""
        <tr>
          <td>{_name_cell}</td>
          <td class="num">{_fmt_num(r['pe'], 1)}</td>
          <td class="num">{_fmt_num(r['pb'], 2)}</td>
          <td class="num">{_fmt_num(r['ev_ebitda'], 1)}</td>
          <td class="num {_pct_class(r['rev_yoy'])}">{_fmt_pct(r['rev_yoy'], 1, signed=True)}</td>
          <td class="num {_pct_class(r['np_yoy'])}">{_fmt_pct(r['np_yoy'], 1, signed=True)}</td>
          <td class="num">{_fmt_pct(r.get('roe_annual'), 1)}
            <div style="font-size:9.5px;color:var(--ink-3)">{annual_yr} annual</div></td>
          <td class="num">{_fmt_pct(r.get('roe_q'), 1)}
            <div style="font-size:9.5px;color:var(--ink-3)">{q_lbl_short} YTD</div></td>
          <td class="num">{_fmt_num(r['net_debt_ebitda'], 2)}</td>
          <td>{sig_pill}</td>
        </tr>"""

    st.markdown(f"""
    <div class="eb-card" style="margin-top:12px">
      <table class="eb-table">
        <thead><tr>
          <th>Stock</th><th>P/E</th><th>P/B</th><th>EV/EBITDA</th>
          <th>Rev YoY</th><th>NP YoY</th>
          <th>ROE (Annual)</th><th>ROE (Latest Qtr)</th>
          <th>Net Debt/EBITDA</th><th>Latest Signal</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Admin curation panel ─────────────────────────────────────────────────
    # WHY st.form here:
    #   Any st.widget outside a form inside a @st.fragment triggers an
    #   immediate fragment rerun on every interaction.  Checkboxes are widgets,
    #   so every toggle was causing a full table rebuild — expensive and jarring.
    #
    #   st.form batches all widget changes: nothing fires until the submit
    #   button is clicked.  Result:
    #     • Toggling checkboxes  → zero reruns, zero Tushare calls
    #     • Clicking Save        → exactly one fragment rerun; DB is written
    #                             first, then a second rerun re-reads the new
    #                             peer list and rebuilds the table
    if is_admin:
        with st.expander("🛠 Edit peer set (admin)", expanded=False):
            st.markdown(
                '<div style="margin-bottom:10px;font-size:13px;color:var(--ink-2)">'
                'Uncheck any incorrect peers then click <strong>Save</strong>. '
                'Checkbox changes are <em>batched</em> — the comparison table '
                'will not refresh until you save.'
                '</div>',
                unsafe_allow_html=True,
            )
            sel_key_root = f"eb_peer_chk_{ticker}"
            with st.form(key=f"eb_peers_form_{ticker}"):
                for p in peer_list:
                    st.checkbox(
                        f"`{p['ticker']}` {p['name']} — {p.get('why', '')}",
                        value=True,
                        key=f"{sel_key_root}_{p['ticker']}",
                    )
                submitted = st.form_submit_button(
                    "💾 Save curated peer set",
                    type="primary",
                    use_container_width=True,
                )
            # Intentionally OUTSIDE the form block so session_state is readable
            if submitted:
                selected = [
                    p for p in peer_list
                    if st.session_state.get(f"{sel_key_root}_{p['ticker']}", True)
                ]
                current_tickers  = {p["ticker"] for p in peer_list}
                selected_tickers = {p["ticker"] for p in selected}
                if selected_tickers == current_tickers:
                    st.info("No changes — all peers are still checked.")
                else:
                    removed = current_tickers - selected_tickers
                    equity_brief.save_competitors_curated(ticker, selected)
                    st.success(
                        f"✅ Saved — removed {len(removed)} peer(s): "
                        + ", ".join(sorted(removed))
                    )
                    st.rerun(scope="fragment")


_competitors_section()

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
