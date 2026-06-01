"""
Macro & Commodities · 宏观与大宗商品

Two tabs:
  • Macro       — China macro series from Tushare (CPI, PPI, PMI, GDP, M1/M2,
                  SHIBOR, US Treasury yields).
  • Commodities — Domestic futures continuous-main contracts from Tushare
                  (metals, ferrous, energy, agri, chemicals).

Design notes (memory-conscious — this app runs on ~1 GB shared RAM):
  • Cards use native st.metric (cheap) instead of 20 simultaneous Plotly
    figures. Only ONE trend chart is rendered per tab, for the series the
    user selects.
  • All Tushare reads are cached 1 h (macro updates monthly/daily, not
    intraday).
  • A-share colour convention: st.metric delta_color="inverse" → red = up,
    green = down on every card.
  • Every fetch is defensive: a bad endpoint / contract code degrades to
    "—" with the error in the tooltip rather than crashing the page.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import auth_manager
import data_manager

auth_manager.require_login()

st.set_page_config(page_title="Macro & Commodities | 宏观与大宗", page_icon="🌍", layout="wide")

st.title("🌍 Macro & Commodities · 宏观与大宗商品")
st.caption(
    "China macro indicators and domestic commodity-futures prices, sourced live "
    "from Tushare. Red = up, green = down (A-share convention)."
)

CHINA_TZ = ZoneInfo("Asia/Shanghai")
_NOW = datetime.now(CHINA_TZ)

# ── Generic cached Tushare fetcher ───────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch(endpoint: str, **kwargs) -> pd.DataFrame:
    """Call a Tushare pro endpoint by name. Returns df, or a 1-row df with an
    `_error` column on failure so the caller can surface tier/credit issues."""
    try:
        data_manager.init_tushare()
        api = data_manager.TUSHARE_API
        if api is None:
            return pd.DataFrame({"_error": ["Tushare not initialised"]})
        fn = getattr(api, endpoint)
        df = fn(**kwargs)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        return pd.DataFrame({"_error": [str(exc)]})


def _series(df: pd.DataFrame, period_col: str, value_col: str):
    """
    Return (latest_value, prev_value, period_label, trend_df) from a fetched
    DataFrame, or (None, None, None, None) if unavailable.
    trend_df is a 2-col frame [period, value] sorted ascending for charting.
    """
    if df is None or df.empty or "_error" in df.columns:
        return None, None, None, None
    if period_col not in df.columns or value_col not in df.columns:
        return None, None, None, None
    d = df[[period_col, value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col]).sort_values(period_col)
    if d.empty:
        return None, None, None, None
    latest = float(d[value_col].iloc[-1])
    prev   = float(d[value_col].iloc[-2]) if len(d) >= 2 else None
    label  = str(d[period_col].iloc[-1])
    return latest, prev, label, d.rename(columns={period_col: "period", value_col: "value"})


def _error_of(df: pd.DataFrame) -> str | None:
    if df is not None and not df.empty and "_error" in df.columns:
        return str(df["_error"].iloc[0])
    return None


def _fmt_period(p: str) -> str:
    """Pretty-print a YYYYMM / YYYYMMDD / YYYYQN period string."""
    p = str(p)
    if len(p) == 6 and p.isdigit():           # YYYYMM
        return f"{p[:4]}-{p[4:6]}"
    if len(p) == 8 and p.isdigit():           # YYYYMMDD
        return f"{p[:4]}-{p[4:6]}-{p[6:8]}"
    return p


def _render_cards(specs: list[dict], cols_per_row: int = 4):
    """
    Render a grid of st.metric cards from a list of series specs and return a
    dict {label: trend_df} for the ones that loaded (used by the trend picker).
    """
    trends: dict[str, pd.DataFrame] = {}
    # Group by 'group' key, render a subheader per group.
    groups: dict[str, list[dict]] = {}
    for s in specs:
        groups.setdefault(s["group"], []).append(s)

    for group_name, group_specs in groups.items():
        st.markdown(f"**{group_name}**")
        cols = st.columns(cols_per_row)
        for i, s in enumerate(group_specs):
            df = s["_df"]
            latest, prev, period, trend = _series(df, s["period_col"], s["value_col"])
            col = cols[i % cols_per_row]
            if latest is None:
                err = _error_of(df)
                col.metric(s["label"], "—",
                           help=(f"⚠️ {err}" if err else "No data returned for this series."))
                continue
            unit = s.get("unit", "")
            val_str = f"{latest:,.2f}{unit}"
            delta = None
            if prev is not None:
                delta = f"{latest - prev:+.2f}{unit}"
            col.metric(
                s["label"], val_str, delta=delta,
                delta_color="inverse",  # A-share: red = up, green = down
                help=f"Latest period: {_fmt_period(period)}"
                     + (f" · prev {prev:,.2f}{unit}" if prev is not None else ""),
            )
            if trend is not None and len(trend) >= 2:
                trends[s["label"]] = trend
        st.markdown("")  # spacer
    return trends


def _trend_chart(trends: dict[str, pd.DataFrame], key: str, default_unit: str = ""):
    """Single on-demand trend chart for one selected series."""
    if not trends:
        return
    st.markdown("---")
    pick = st.selectbox("📈 View trend for:", options=list(trends.keys()), key=key)
    d = trends[pick]
    rising = d["value"].iloc[-1] >= d["value"].iloc[0]
    color = "#dc2626" if rising else "#16a34a"   # red up / green down
    fill  = "rgba(220,38,38,0.10)" if rising else "rgba(22,163,74,0.10)"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[_fmt_period(p) for p in d["period"]],
        y=d["value"],
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=4, color=color),
        fill="tozeroy", fillcolor=fill,
        hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=pick, height=340, template="plotly_white",
        margin=dict(t=50, l=50, r=30, b=50),
        xaxis=dict(tickangle=-45), yaxis_title=default_unit or "Value",
    )
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Date windows
# ════════════════════════════════════════════════════════════════════════════
_M_START = (_NOW - timedelta(days=365 * 3)).strftime("%Y%m")     # 3 years monthly
_M_END   = _NOW.strftime("%Y%m")
_D_START = (_NOW - timedelta(days=120)).strftime("%Y%m%d")        # 120 days daily
_D_END   = _NOW.strftime("%Y%m%d")
# Quarterly window for GDP: last ~3 years of quarters
_Q_START = f"{_NOW.year - 3}Q1"
_Q_END   = f"{_NOW.year}Q4"


tab_macro, tab_comm = st.tabs(["📊 Macro 宏观", "🛢 Commodities 大宗商品"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — MACRO
# ════════════════════════════════════════════════════════════════════════════
with tab_macro:
    st.caption(
        "Inflation, growth, liquidity and rates. China series are monthly/quarterly; "
        "SHIBOR and US Treasury yields are daily. Delta is change vs the prior period."
    )

    # Fetch once per series (cached)
    cpi_df   = _fetch("cn_cpi", start_m=_M_START, end_m=_M_END)
    ppi_df   = _fetch("cn_ppi", start_m=_M_START, end_m=_M_END)
    pmi_df   = _fetch("cn_pmi", start_m=_M_START, end_m=_M_END)
    m_df     = _fetch("cn_m",   start_m=_M_START, end_m=_M_END)
    gdp_df   = _fetch("cn_gdp", start_q=_Q_START, end_q=_Q_END)
    shibor_df= _fetch("shibor", start_date=_D_START, end_date=_D_END)
    ustycr_df= _fetch("us_tycr", start_date=_D_START, end_date=_D_END)

    macro_specs = [
        # Inflation
        {"label": "China CPI YoY",      "group": "Inflation", "unit": "%",
         "_df": cpi_df, "period_col": "month", "value_col": "nt_yoy"},
        {"label": "China PPI YoY",      "group": "Inflation", "unit": "%",
         "_df": ppi_df, "period_col": "month", "value_col": "ppi_yoy"},
        # Growth
        {"label": "Manufacturing PMI",  "group": "Growth", "unit": "",
         "_df": pmi_df, "period_col": "month", "value_col": "pmi010000"},
        {"label": "GDP YoY",            "group": "Growth", "unit": "%",
         "_df": gdp_df, "period_col": "quarter", "value_col": "gdp_yoy"},
        # Liquidity
        {"label": "M2 Money Supply YoY","group": "Liquidity", "unit": "%",
         "_df": m_df, "period_col": "month", "value_col": "m2_yoy"},
        {"label": "M1 Money Supply YoY","group": "Liquidity", "unit": "%",
         "_df": m_df, "period_col": "month", "value_col": "m1_yoy"},
        # China rates
        {"label": "SHIBOR Overnight",   "group": "China Rates", "unit": "%",
         "_df": shibor_df, "period_col": "date", "value_col": "on"},
        {"label": "SHIBOR 3M",          "group": "China Rates", "unit": "%",
         "_df": shibor_df, "period_col": "date", "value_col": "3m"},
        # US rates
        {"label": "US 2Y Treasury",     "group": "US Rates", "unit": "%",
         "_df": ustycr_df, "period_col": "date", "value_col": "y2"},
        {"label": "US 10Y Treasury",    "group": "US Rates", "unit": "%",
         "_df": ustycr_df, "period_col": "date", "value_col": "y10"},
    ]

    macro_trends = _render_cards(macro_specs, cols_per_row=4)
    _trend_chart(macro_trends, key="macro_trend_pick")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMMODITIES
# ════════════════════════════════════════════════════════════════════════════
with tab_comm:
    st.caption(
        "Domestic commodity-futures continuous-main contracts (settlement / close), "
        "from Tushare `fut_daily`. Delta is day-over-day close change."
    )

    # Continuous-main contract codes (<SYMBOL>.<EXCHANGE>):
    #   SHFE→.SHF · DCE→.DCE · CZCE→.ZCE · INE→.INE
    COMMODITIES = [
        {"label": "Copper 沪铜",      "ts_code": "CU.SHF", "group": "Base Metals"},
        {"label": "Aluminum 沪铝",    "ts_code": "AL.SHF", "group": "Base Metals"},
        {"label": "Zinc 沪锌",        "ts_code": "ZN.SHF", "group": "Base Metals"},
        {"label": "Gold 沪金",        "ts_code": "AU.SHF", "group": "Precious Metals"},
        {"label": "Silver 沪银",      "ts_code": "AG.SHF", "group": "Precious Metals"},
        {"label": "Rebar 螺纹钢",     "ts_code": "RB.SHF", "group": "Ferrous"},
        {"label": "Iron Ore 铁矿石",  "ts_code": "I.DCE",  "group": "Ferrous"},
        {"label": "Coking Coal 焦煤", "ts_code": "JM.DCE", "group": "Energy"},
        {"label": "Crude Oil 原油",   "ts_code": "SC.INE", "group": "Energy"},
        {"label": "Soybean Meal 豆粕","ts_code": "M.DCE",  "group": "Agriculture"},
        {"label": "Palm Oil 棕榈油",  "ts_code": "P.DCE",  "group": "Agriculture"},
        {"label": "PTA",              "ts_code": "TA.ZCE", "group": "Chemicals"},
    ]

    # Build specs by fetching each contract's recent daily bars.
    comm_specs = []
    for c in COMMODITIES:
        df = _fetch("fut_daily", ts_code=c["ts_code"], start_date=_D_START, end_date=_D_END)
        comm_specs.append({
            "label": c["label"], "group": c["group"], "unit": "",
            "_df": df, "period_col": "trade_date", "value_col": "close",
        })

    comm_trends = _render_cards(comm_specs, cols_per_row=4)
    _trend_chart(comm_trends, key="comm_trend_pick", default_unit="Price")

    st.caption(
        "ⓘ Contract codes use Tushare's continuous-main format "
        "(`<SYMBOL>.<EXCHANGE>`). If any card shows “—”, that specific code may "
        "need adjusting for your data tier — note which ones and they can be fixed."
    )
