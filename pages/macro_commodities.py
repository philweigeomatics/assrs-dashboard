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


# ════════════════════════════════════════════════════════════════════════════
# Futures term-structure helpers
# ════════════════════════════════════════════════════════════════════════════
# A futures price is meaningless without its delivery month. To read whether a
# market is in contango (far > near → storage/carry priced in) or backwardation
# (near > far → tight spot / supply stress), we need every active contract's
# settlement price on the same trade date, sorted by maturity = the forward
# curve. From the front two points we derive market state + annualised roll
# yield; the full set gives the calendar-spread matrix.

@st.cache_data(ttl=3600, show_spinner=False)
def _latest_fut_date() -> str | None:
    """Most recent futures trade date with data (probed via a liquid contract)."""
    df = _fetch("fut_daily", ts_code="CU.SHF", start_date=_D_START, end_date=_D_END)
    if df is None or df.empty or "_error" in df.columns or "trade_date" not in df.columns:
        return None
    return str(df["trade_date"].max())


@st.cache_data(ttl=3600, show_spinner=False)
def _fut_basic(exchange: str) -> pd.DataFrame:
    """Contract directory for an exchange — maturity (delist_date) + product code."""
    return _fetch("fut_basic", exchange=exchange, fut_type="1",
                  fields="ts_code,symbol,fut_code,name,list_date,delist_date")


@st.cache_data(ttl=3600, show_spinner=False)
def _fut_day(trade_date: str) -> pd.DataFrame:
    """All futures daily bars for one trade date (one call, filtered locally)."""
    return _fetch("fut_daily", trade_date=trade_date)


def _forward_curve(code: str, exchange: str, trade_date: str) -> pd.DataFrame | None:
    """
    Build the forward curve for product `code` on `trade_date`:
    every active contract's settlement price, sorted near→far by maturity.
    Returns a DataFrame [symbol, maturity (YYYYMMDD), price, oi, vol] or None.
    """
    basic = _fut_basic(exchange)
    day   = _fut_day(trade_date)
    if (basic is None or basic.empty or "_error" in basic.columns
            or day is None or day.empty or "_error" in day.columns):
        return None
    b = basic[basic["fut_code"].astype(str).str.upper() == code.upper()].copy()
    if b.empty:
        return None
    # Active = not yet delisted as of the trade date
    b = b[b["delist_date"].astype(str) >= str(trade_date)]
    if b.empty:
        return None
    keep = [c for c in ("ts_code", "settle", "close", "pre_settle", "oi", "vol") if c in day.columns]
    m = b.merge(day[keep], on="ts_code", how="inner")
    if m.empty:
        return None
    m["price"] = pd.to_numeric(m.get("settle"), errors="coerce")
    if "close" in m.columns:
        m["price"] = m["price"].fillna(pd.to_numeric(m["close"], errors="coerce"))
    m = m.dropna(subset=["price"])
    m = m[m["price"] > 0]
    if m.empty:
        return None
    m = m.sort_values("delist_date").reset_index(drop=True)
    m["maturity"] = m["delist_date"].astype(str)
    cols = ["symbol", "maturity", "price"]
    for extra in ("oi", "vol"):
        if extra in m.columns:
            cols.append(extra)
    return m[cols]


def _term_structure(curve: pd.DataFrame) -> dict | None:
    """From the front two contracts derive market state + annualised roll yield."""
    if curve is None or len(curve) < 2:
        return None
    front, nxt = curve.iloc[0], curve.iloc[1]
    spread = float(front["price"]) - float(nxt["price"])
    try:
        d0 = datetime.strptime(str(front["maturity"]), "%Y%m%d")
        d1 = datetime.strptime(str(nxt["maturity"]), "%Y%m%d")
        days = max((d1 - d0).days, 1)
    except Exception:
        days = 30
    # Roll yield: rolling from front to next. Positive = backwardation (you
    # gain rolling down the curve), negative = contango (you bleed carry).
    roll_ann = (float(front["price"]) / float(nxt["price"]) - 1.0) * (365.0 / days)
    state = "Backwardation" if spread > 0 else ("Contango" if spread < 0 else "Flat")
    return {
        "front_symbol": front["symbol"], "front_price": float(front["price"]),
        "next_symbol":  nxt["symbol"],   "next_price":  float(nxt["price"]),
        "spread": spread, "roll_ann_pct": roll_ann * 100, "state": state,
    }


def _liquid_only(curve: pd.DataFrame, frac: float = 0.10) -> pd.DataFrame:
    """
    Keep only liquid contracts — those whose open interest is ≥ `frac` of the
    most-liquid contract's OI. In Chinese commodity futures, liquidity clusters
    in the 1/5/9 delivery months; the off-months barely trade and their
    settlements are administrative, producing fake kinks in a full curve.
    Filtering by OI surfaces the real liquidity nodes (which ARE 1/5/9 for most
    metals) without hard-coding a month rule that breaks for crude / some ags.
    Falls back to the full curve if OI is missing or all-zero.
    """
    if curve is None or curve.empty or "oi" not in curve.columns:
        return curve
    oi = pd.to_numeric(curve["oi"], errors="coerce").fillna(0)
    if oi.max() <= 0:
        return curve
    keep = curve[oi >= oi.max() * frac]
    return keep if len(keep) >= 2 else curve


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
    # fut_code (product) + exchange param. ts_code suffix differs from the
    # exchange param: SHFE→.SHF, DCE→.DCE, CZCE→.ZCE, INE→.INE.
    FUTURES = [
        {"label": "Copper 沪铜",       "code": "CU", "exchange": "SHFE", "unit": "¥/t"},
        {"label": "Aluminum 沪铝",     "code": "AL", "exchange": "SHFE", "unit": "¥/t"},
        {"label": "Zinc 沪锌",         "code": "ZN", "exchange": "SHFE", "unit": "¥/t"},
        {"label": "Gold 沪金",         "code": "AU", "exchange": "SHFE", "unit": "¥/g"},
        {"label": "Silver 沪银",       "code": "AG", "exchange": "SHFE", "unit": "¥/kg"},
        {"label": "Rebar 螺纹钢",      "code": "RB", "exchange": "SHFE", "unit": "¥/t"},
        {"label": "Iron Ore 铁矿石",   "code": "I",  "exchange": "DCE",  "unit": "¥/t"},
        {"label": "Coking Coal 焦煤",  "code": "JM", "exchange": "DCE",  "unit": "¥/t"},
        {"label": "Crude Oil 原油",    "code": "SC", "exchange": "INE",  "unit": "¥/bbl"},
        {"label": "Soybean Meal 豆粕", "code": "M",  "exchange": "DCE",  "unit": "¥/t"},
        {"label": "Palm Oil 棕榈油",   "code": "P",  "exchange": "DCE",  "unit": "¥/t"},
        {"label": "PTA",               "code": "TA", "exchange": "CZCE", "unit": "¥/t"},
    ]
    _LABEL_BY_CODE = {f["code"]: f for f in FUTURES}

    trade_date = _latest_fut_date()
    if not trade_date:
        st.error("Could not determine the latest futures trade date from Tushare.")
        st.stop()
    st.caption(
        f"Settlement prices for trade date **{_fmt_period(trade_date)}**, from "
        "Tushare `fut_daily` / `fut_basic`. Each price is a dated delivery "
        "contract — red = up, green = down."
    )

    # ── Headline cards: front-month settlement for the 4 key metals ──────────
    st.markdown("**Key metals · front-month settlement**")
    head_cols = st.columns(4)
    for col, code in zip(head_cols, ["CU", "AL", "AU", "AG"]):
        spec = _LABEL_BY_CODE[code]
        curve = _forward_curve(code, spec["exchange"], trade_date)
        if curve is None or curve.empty:
            col.metric(spec["label"], "—", help="No active contracts returned.")
            continue
        front = curve.iloc[0]
        ts = _term_structure(curve)
        delta = f"{ts['spread']:+.2f} vs {ts['next_symbol']}" if ts else None
        col.metric(
            f"{spec['label']} · {front['symbol']}",
            f"{front['price']:,.2f} {spec['unit']}",
            delta=delta, delta_color="inverse",
            help=f"Front contract {front['symbol']} settlement on {_fmt_period(trade_date)}."
                 + (f" Term structure: {ts['state']}." if ts else ""),
        )

    st.markdown("---")

    # ── Per-commodity term-structure drill-down ──────────────────────────────
    st.markdown("### 🔬 Term-structure analysis")
    sel_label = st.selectbox(
        "Commodity",
        options=[f["label"] for f in FUTURES],
        key="comm_select",
    )
    sel = next(f for f in FUTURES if f["label"] == sel_label)
    full_curve = _forward_curve(sel["code"], sel["exchange"], trade_date)

    if full_curve is None or full_curve.empty:
        st.warning(
            f"No active contracts returned for {sel_label} ({sel['code']}.{sel['exchange']}). "
            "The product code or exchange may need adjusting for your data tier."
        )
    else:
        # Liquidity toggle. Default ON so the curve, roll yield and spread
        # table all reflect the real liquidity nodes (1/5/9 for most metals),
        # not the dead off-months whose settlements are administrative.
        liquid_only = st.toggle(
            "Liquid months only (filter by open interest)",
            value=True,
            key="comm_liquid_toggle",
            help="Chinese commodity liquidity clusters in the 1/5/9 contracts. "
                 "ON keeps contracts with OI ≥ 10% of the most-liquid month, "
                 "removing dead-month noise so the curve slope is accurate. "
                 "OFF shows every listed month.",
        )
        curve = _liquid_only(full_curve) if liquid_only else full_curve
        if liquid_only and len(curve) < len(full_curve):
            st.caption(f"Showing {len(curve)} liquid of {len(full_curve)} listed contracts.")

        ts = _term_structure(curve)

        # ── Term-structure status panel ──────────────────────────────────────
        if ts:
            state = ts["state"]
            # A-share convention: backwardation (near richer) = red, contango = green
            if state == "Backwardation":
                badge_color, blurb = "#dc2626", "near > far — tight spot / supply stress; positive roll for longs"
            elif state == "Contango":
                badge_color, blurb = "#16a34a", "far > near — carry/storage priced in; negative roll for longs"
            else:
                badge_color, blurb = "#64748b", "front and next roughly equal"

            st.markdown(
                f"<div style='padding:10px 14px;border-radius:8px;"
                f"background:{badge_color}1a;border:1px solid {badge_color};"
                f"display:inline-block;margin-bottom:6px'>"
                f"<b style='color:{badge_color}'>Market state: {state}</b> "
                f"<span style='color:var(--text-color,#475569)'>· {blurb}</span></div>",
                unsafe_allow_html=True,
            )

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(f"Front · {ts['front_symbol']}", f"{ts['front_price']:,.2f}")
            k2.metric(f"Next · {ts['next_symbol']}",   f"{ts['next_price']:,.2f}",
                      delta=f"{ts['spread']:+.2f}", delta_color="inverse",
                      help="Front − Next settlement spread.")
            k3.metric("Calendar spread", f"{ts['spread']:+.2f} {sel['unit']}")
            k4.metric("Annualised roll yield", f"{ts['roll_ann_pct']:+.2f}%",
                      delta_color="inverse",
                      help="Annualised gain/loss from rolling the front contract to "
                           "the next. Positive = backwardation, negative = contango.")

        # ── Forward curve chart with OI-sized bubbles ────────────────────────
        # Bubble size ∝ open interest: big solid node = dominant/liquid month
        # to trust; tiny dot = dead contract whose price is suspect. This stops
        # a thin contract near expiry from faking a steep curve segment.
        rising = curve["price"].iloc[-1] >= curve["price"].iloc[0]
        line_color = "#16a34a" if rising else "#dc2626"   # up=contango green, down=backwardation red

        oi_vals = pd.to_numeric(curve.get("oi", pd.Series([0] * len(curve))), errors="coerce").fillna(0)
        if oi_vals.max() > 0:
            sizes = (10 + 40 * (oi_vals / oi_vals.max())).tolist()   # 10–50 px
        else:
            sizes = [14] * len(curve)

        fig = go.Figure()
        # Connecting line
        fig.add_trace(go.Scatter(
            x=curve["symbol"], y=curve["price"],
            mode="lines", line=dict(color=line_color, width=2),
            hoverinfo="skip", showlegend=False,
        ))
        # OI bubbles
        fig.add_trace(go.Scatter(
            x=curve["symbol"], y=curve["price"],
            mode="markers",
            marker=dict(size=sizes, color=line_color, opacity=0.55,
                        line=dict(color=line_color, width=1.5)),
            customdata=oi_vals,
            hovertemplate="%{x}<br>Settle %{y:,.2f}<br>OI %{customdata:,.0f}<extra></extra>",
            showlegend=False,
        ))
        fig.update_layout(
            title=f"{sel_label} forward curve · {_fmt_period(trade_date)}  (bubble = open interest)",
            height=400, template="plotly_white",
            margin=dict(t=50, l=50, r=30, b=50),
            xaxis_title="Contract (near → far)",
            yaxis_title=f"Settlement ({sel['unit']})",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Spread table: Contract · Price · Spread vs front · State ─────────
        st.markdown("**Calendar spreads** · each contract vs the front, with state")
        front_price = float(curve["price"].iloc[0])
        spread_rows = []
        for _, r in curve.iterrows():
            spr = float(r["price"]) - front_price
            if abs(spr) < 1e-9:
                state_lbl = "— front"
            elif spr > 0:
                state_lbl = "Contango (far richer)"      # far > near
            else:
                state_lbl = "Backwardation (far cheaper)"  # far < near
            spread_rows.append({
                "Contract":  r["symbol"],
                "Price":     round(float(r["price"]), 2),
                "Spread vs front": round(spr, 2),
                "State":     state_lbl,
            })
        spread_df = pd.DataFrame(spread_rows)

        def _state_color(val):
            # A-share: contango (far richer) green, backwardation (far cheaper) red
            if "Contango" in str(val):
                return "color:#16a34a;font-weight:600"
            if "Backwardation" in str(val):
                return "color:#dc2626;font-weight:600"
            return "color:#64748b"

        st.dataframe(
            spread_df.style
                .format({"Price": "{:,.2f}", "Spread vs front": "{:+,.2f}"})
                .map(_state_color, subset=["State"]),
            use_container_width=True, hide_index=True,
        )

        # ── Underlying contract table (raw OI / volume) ──────────────────────
        with st.expander("📋 Contract detail (settle · OI · volume)", expanded=False):
            disp = full_curve.copy()
            disp["maturity"] = disp["maturity"].map(_fmt_period)
            disp = disp.rename(columns={
                "symbol": "Contract", "maturity": "Delist date",
                "price": "Settle", "oi": "Open interest", "vol": "Volume",
            })
            st.dataframe(disp, use_container_width=True, hide_index=True)

    st.caption(
        "ⓘ Forward curve = each contract's settlement, sorted near→far; bubble "
        "size = open interest (trust the slope between big bubbles, ignore tiny "
        "ones). Upward slope = contango, downward = backwardation. In Chinese "
        "commodities, liquidity clusters in the 1/5/9 months — the liquidity "
        "toggle keeps only those real nodes. Roll yield = annualised front→next "
        "carry on the displayed (liquid) curve."
    )
