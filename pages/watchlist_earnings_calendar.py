"""Watchlist Earnings Disclosure Calendar — shows Tushare disclosure_date data."""

import calendar as cal_module
import html as html_module
from datetime import datetime

import pandas as pd
import pytz
import streamlit as st

import auth_manager
import data_manager as dm

auth_manager.require_login()

# ── Page Header ────────────────────────────────────────────────────────
st.title("📅 Earnings Calendar 财报披露日历")
st.caption("Earnings disclosure dates for your watchlist stocks · Data from Tushare")

# ── Watchlist ──────────────────────────────────────────────────────────
watchlist = dm.get_watchlist()
if not watchlist:
    st.info("Your watchlist is empty. Add stocks on the **Watchlist** page first.")
    st.stop()

ticker_to_name: dict[str, str] = {item["ticker"]: item["stock_name"] for item in watchlist}
ts_codes = [dm.get_tushare_ticker(t) for t in ticker_to_name]
ts_to_ticker = {dm.get_tushare_ticker(t): t for t in ticker_to_name}

# ── Reporting Period Selection ─────────────────────────────────────────
BEIJING_TZ = pytz.timezone("Asia/Shanghai")
today = datetime.now(BEIJING_TZ).date()


def build_periods(ref_date):
    y = ref_date.year
    items = [
        {"label": f"Annual {y - 1} 年报", "end_date": f"{y - 1}1231"},
        {"label": f"Q1 {y} 一季报",       "end_date": f"{y}0331"},
    ]
    if ref_date.month >= 7:
        items.append({"label": f"H1 {y} 中报", "end_date": f"{y}0630"})
    if ref_date.month >= 10:
        items.append({"label": f"Q3 {y} 三季报", "end_date": f"{y}0930"})
    return items


periods = build_periods(today)
period_labels = [p["label"] for p in periods]

left_col, _ = st.columns([3, 7])
with left_col:
    sel_idx = st.selectbox(
        "Reporting Period 报告期",
        range(len(periods)),
        format_func=lambda i: period_labels[i],
        index=len(periods) - 1,
        key="ecal_period_idx",
    )

selected_period = periods[sel_idx]
end_date = selected_period["end_date"]

# ── Data Fetching ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_disclosures(end_date: str, ts_codes_key: tuple) -> pd.DataFrame:
    """Fetch disclosure dates from Tushare; falls back to per-ticker if bulk is empty."""
    if not dm.init_tushare():
        return pd.DataFrame()

    try:
        # Bulk fetch by period — much faster, one credit
        df = dm.TUSHARE_API.disclosure_date(
            end_date=end_date,
            fields="ts_code,ann_date,end_date,pre_date,actual_date",
        )
        if df is not None and not df.empty:
            filtered = df[df["ts_code"].isin(list(ts_codes_key))].copy()
            if not filtered.empty:
                return filtered
    except Exception:
        pass

    # Fallback: individual calls
    frames = []
    for ts_code in ts_codes_key:
        try:
            df = dm.TUSHARE_API.disclosure_date(
                ts_code=ts_code,
                end_date=end_date,
                fields="ts_code,ann_date,end_date,pre_date,actual_date",
            )
            if df is not None and not df.empty:
                frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


with st.spinner("Fetching disclosure dates 正在获取财报披露日期…"):
    raw = load_disclosures(end_date, tuple(sorted(ts_codes)))

if raw.empty:
    st.warning(
        f"No disclosure data found for your watchlist in **{selected_period['label']}**.\n\n"
        "Possible reasons: Tushare API credits exhausted, data not yet available, "
        "or none of your watchlist stocks match this reporting period."
    )
    st.stop()

# ── Data Processing ────────────────────────────────────────────────────
raw = raw.copy()
raw["ticker"] = raw["ts_code"].map(ts_to_ticker)
raw["stock_name"] = raw["ticker"].map(ticker_to_name)

# Normalise empty strings to NaN
for col in ("pre_date", "actual_date"):
    raw[col] = raw[col].replace("", pd.NA)

raw["effective_date"] = raw["actual_date"].fillna(raw["pre_date"])
raw["is_reported"] = raw["actual_date"].notna()


def classify_status(row) -> str:
    if row["is_reported"]:
        return "reported"
    eff = row["effective_date"]
    if pd.notna(eff) and eff:
        try:
            eff_date = datetime.strptime(str(eff), "%Y%m%d").date()
            return "overdue" if eff_date < today else "scheduled"
        except ValueError:
            pass
    return "unknown"


raw["status"] = raw.apply(classify_status, axis=1)

# ── Calendar Session State ─────────────────────────────────────────────
period_key = f"ecal_view_{end_date}"

if period_key not in st.session_state:
    valid_dates = raw[raw["effective_date"].notna()]["effective_date"]
    if not valid_dates.empty:
        parsed = pd.to_datetime(valid_dates, format="%Y%m%d", errors="coerce").dropna()
        if not parsed.empty:
            periods_by_month = parsed.dt.to_period("M").value_counts()
            best = periods_by_month.idxmax()
            st.session_state[period_key] = (best.year, best.month)
        else:
            st.session_state[period_key] = (today.year, today.month)
    else:
        st.session_state[period_key] = (today.year, today.month)

cal_year, cal_month = st.session_state[period_key]

# ── Build Events Dict ──────────────────────────────────────────────────
# events = { "YYYYMMDD": [{"ticker", "name", "status"}, ...] }
events: dict[str, list] = {}
for _, row in raw.iterrows():
    d = row["effective_date"]
    if d and isinstance(d, str) and len(d) == 8:
        events.setdefault(d, []).append(
            {
                "ticker": row["ticker"] or "",
                "name": row["stock_name"] or "",
                "status": row["status"],
            }
        )

# ── Calendar Renderer ──────────────────────────────────────────────────
STATUS_STYLE = {
    "reported": {"bg": "#22c55e22", "border": "#22c55e", "icon": "✅"},
    "scheduled": {"bg": "#3b82f622", "border": "#3b82f6", "icon": "📋"},
    "overdue":   {"bg": "#ef444422", "border": "#ef4444", "icon": "⚠️"},
    "unknown":   {"bg": "#6b728022", "border": "#6b7280", "icon": "❓"},
}


def build_calendar_html(year: int, month: int, events: dict, today) -> str:
    weeks = cal_module.monthcalendar(year, month)
    parts: list[str] = []

    parts.append(
        '<table style="width:100%;border-collapse:collapse;table-layout:fixed;'
        'font-family:var(--font-sans,-apple-system,sans-serif)">'
    )
    parts.append('<colgroup>' + '<col style="width:14.285%">' * 7 + '</colgroup>')
    parts.append("<thead><tr>")
    for day_name in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        parts.append(
            f'<th style="padding:8px 4px;background:#1e3a5f;color:#93c5fd;'
            f'text-align:center;font-size:12px;font-weight:600;'
            f'border:1px solid #1e3a5f">{day_name}</th>'
        )
    parts.append("</tr></thead><tbody>")

    for week in weeks:
        parts.append("<tr>")
        for day in week:
            if day == 0:
                parts.append(
                    '<td style="border:1px solid #1f2937;background:#0a0a0f;'
                    'min-height:90px;height:90px;padding:0"></td>'
                )
                continue

            date_key = f"{year:04d}{month:02d}{day:02d}"
            is_today = year == today.year and month == today.month and day == today.day

            bg = "#1a237e1a" if is_today else "#0e1117"
            day_color = "#fbbf24" if is_today else "#6b7280"
            today_ring = "box-shadow:inset 0 0 0 2px #fbbf24;" if is_today else ""

            parts.append(
                f'<td style="border:1px solid #1f2937;background:{bg};'
                f"vertical-align:top;padding:6px 5px;min-height:90px;height:90px;"
                f'{today_ring}">'
            )
            parts.append(
                f'<div style="font-size:13px;font-weight:700;color:{day_color};margin-bottom:4px">'
                f"{day}</div>"
            )

            for stock in events.get(date_key, []):
                st_info = STATUS_STYLE.get(stock["status"], STATUS_STYLE["unknown"])
                ticker = html_module.escape(stock["ticker"])
                name = html_module.escape(stock["name"])
                tooltip = f"{ticker} {name}"
                parts.append(
                    f'<div style="background:{st_info["bg"]};border-left:3px solid {st_info["border"]};'
                    f"padding:2px 5px;margin:2px 0;border-radius:3px;overflow:hidden;"
                    f'white-space:nowrap;text-overflow:ellipsis;font-size:11px" title="{tooltip}">'
                    f'<span style="color:{st_info["border"]};font-weight:700">'
                    f'{st_info["icon"]} {ticker}</span> '
                    f'<span style="color:#9ca3af">{name}</span>'
                    f"</div>"
                )

            parts.append("</td>")
        parts.append("</tr>")

    parts.append("</tbody></table>")
    return "".join(parts)


# ── Navigation Controls ────────────────────────────────────────────────
nav_l, nav_c, nav_r = st.columns([1, 6, 1])

with nav_l:
    if st.button("◀", use_container_width=True, key="ecal_prev"):
        new_month = 12 if cal_month == 1 else cal_month - 1
        new_year = cal_year - 1 if cal_month == 1 else cal_year
        st.session_state[period_key] = (new_year, new_month)
        st.rerun()

with nav_c:
    st.markdown(
        f"<h3 style='text-align:center;margin:4px 0;color:#e5e7eb'>"
        f"{cal_module.month_name[cal_month]} {cal_year}</h3>",
        unsafe_allow_html=True,
    )

with nav_r:
    if st.button("▶", use_container_width=True, key="ecal_next"):
        new_month = 1 if cal_month == 12 else cal_month + 1
        new_year = cal_year + 1 if cal_month == 12 else cal_year
        st.session_state[period_key] = (new_year, new_month)
        st.rerun()

# ── Render Calendar ────────────────────────────────────────────────────
st.markdown(build_calendar_html(cal_year, cal_month, events, today), unsafe_allow_html=True)

# Legend
st.markdown(
    '<div style="display:flex;gap:20px;margin-top:10px;font-size:12px;color:#9ca3af">'
    "<span>✅ <span style='color:#22c55e'>Reported 已披露</span></span>"
    "<span>📋 <span style='color:#3b82f6'>Scheduled 预计</span></span>"
    "<span>⚠️ <span style='color:#ef4444'>Overdue 逾期</span></span>"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()

# ── Summary Stats ──────────────────────────────────────────────────────
counts = raw["status"].value_counts()
s_col1, s_col2, s_col3, s_col4 = st.columns(4)
s_col1.metric("Total Tracked 追踪", len(raw))
s_col2.metric("✅ Reported 已披露",  counts.get("reported",  0))
s_col3.metric("📋 Scheduled 预计",   counts.get("scheduled", 0))
s_col4.metric("⚠️ Overdue 逾期",     counts.get("overdue",   0))

# ── Detail Table ───────────────────────────────────────────────────────
st.subheader("Disclosure Details 披露详情")


def fmt_date(d) -> str:
    if pd.isna(d) or not d:
        return "—"
    try:
        return datetime.strptime(str(d), "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return str(d)


STATUS_LABEL = {
    "reported": "✅ Reported 已披露",
    "scheduled": "📋 Scheduled 预计",
    "overdue":   "⚠️ Overdue 逾期",
    "unknown":   "❓ Unknown",
}

detail = raw[raw["ticker"].notna()][
    ["ticker", "stock_name", "end_date", "pre_date", "actual_date", "effective_date", "status"]
].copy()

detail["Period End 报告期末"]     = detail["end_date"].apply(fmt_date)
detail["Planned Date 预计披露"]   = detail["pre_date"].apply(fmt_date)
detail["Actual Date 实际披露"]    = detail["actual_date"].apply(fmt_date)
detail["Effective Date 生效日期"] = detail["effective_date"].apply(fmt_date)
detail["Status 状态"]             = detail["status"].map(STATUS_LABEL)

display_df = (
    detail[
        ["ticker", "stock_name", "Period End 报告期末",
         "Planned Date 预计披露", "Actual Date 实际披露",
         "Effective Date 生效日期", "Status 状态"]
    ]
    .rename(columns={"ticker": "Code 代码", "stock_name": "Name 名称"})
    .sort_values("Effective Date 生效日期", na_position="last")
    .reset_index(drop=True)
)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Watchlist stocks with no disclosure data ───────────────────────────
disclosed_tickers = set(raw["ticker"].dropna())
missing = set(ticker_to_name.keys()) - disclosed_tickers
if missing:
    missing_str = ", ".join(
        f"{t}({ticker_to_name.get(t, '')})" for t in sorted(missing)
    )
    st.info(f"No disclosure data found for: {missing_str}")
