"""
Today's Opportunities & Alerts - 今日提醒
Scans all stocks and shows buy/sell signals in a clean table
扫描所有股票并显示买卖信号
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import data_manager

# Import from shared engine
from analysis_engine import run_single_stock_analysis
import box_detection as bxd

import auth_manager
auth_manager.require_login()

# Initialize tables
data_manager.create_watchlist_table()
MY_WATCHLIST = data_manager.get_watchlist_tickers()


if 'force_rescan' not in st.session_state:
    st.session_state.force_rescan = False

# ==================== SIGNAL CRITERIA ====================
# The rules live in watchlist_scan.py, shared with the nightly GitHub Actions
# scan. Two copies would drift, and a cached row would stop meaning what the
# chart marker means.
import watchlist_scan


def get_beijing_date():
    """Get current date in Beijing timezone"""
    beijing_tz = pytz.timezone('Asia/Shanghai')
    return datetime.now(beijing_tz).date()


def expected_latest_session():
    """
    The trading session whose close should be in the cache by now.

    Tushare publishes the day's bars by ~18:30 Beijing and the nightly scan
    runs at 20:00, so before 20:30 the freshest possible snapshot is the
    previous weekday. No holiday calendar: on a holiday this names a session
    that never happened, which only makes the freshness note over-cautious.
    It never causes a scan.
    """
    now = datetime.now(pytz.timezone('Asia/Shanghai'))
    d = now.date()
    if now.hour < 20 or (now.hour == 20 and now.minute < 30):
        d = d - pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d = d - pd.Timedelta(days=1)
    return d


def init_signals_tables():
    """Initialize the signals cache tables"""
    data_manager.create_signals_tables()


def scan_my_watchlist():
    """
    Interactive scan of the logged-in user's watchlist. Same per-stock code as
    the nightly job (watchlist_scan.scan_ticker); only the progress UI and the
    session-scoped save differ.

    Returns (df, seconds, data_date).
    """
    import time
    start_time = time.time()

    if not MY_WATCHLIST:
        st.error("❌ Your watchlist is empty! Add stocks on the Watchlist page.")
        return None, 0, None

    rows, dates = [], []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(MY_WATCHLIST)
    for idx, ticker in enumerate(MY_WATCHLIST, 1):
        progress_bar.progress(idx / total)
        status_text.text(f"🔍 调取并分析 {idx}/{total}: {ticker} - {idx/total*100:.1f}%")
        r = watchlist_scan.scan_ticker(ticker)
        if r.get("data_date"):
            dates.append(r["data_date"])
        if r["status"] == "ok":
            rows.append(r["row"])
        elif r["status"] == "no_data":
            st.warning(f"⚠️ {ticker}: 数据不足 (需要至少{watchlist_scan.MIN_BARS}天)")
        elif r["status"] == "error":
            st.warning(f"⚠️ {ticker} 分析失败: {r.get('error', '')}")

    progress_bar.empty()
    status_text.empty()
    data_date = max(set(dates), key=dates.count) if dates else None
    return watchlist_scan.rank_rows(rows), time.time() - start_time, data_date


# ==================== MAIN PAGE ====================
st.set_page_config(page_title="Today's Alerts | 今日提醒", page_icon="🎯", layout="wide")

st.title("🎯 Today's Opportunities & Alerts | 今日提醒")

# Initialize tables in existing database
# init_signals_tables()

# Get today's date in Beijing time
today_beijing = get_beijing_date()
today_str = today_beijing.strftime('%Y-%m-%d')

st.markdown(f"**Beijing Date:** {today_str} {today_beijing.strftime('%A')}")

# ── Which snapshot to show ────────────────────────────────────────
# NEVER scan on page load. An 80-stock scan is ~24 min of CPU in one script
# run (~15.5s/stock in the walk-forward HMM) and Community Cloud throttles
# exactly that. The nightly GitHub Actions job (scan_watchlists.py) fills the
# cache; this page reads it. It used to look up only today's calendar date and
# scan on a miss, which made every morning a full rescan of data that had not
# changed since the previous close.
snap_date = data_manager.get_latest_signal_snapshot_date()
cached_df = data_manager.get_cached_signals(snap_date) if snap_date else None
metadata = data_manager.get_scan_metadata(snap_date) if snap_date else None

if st.session_state.force_rescan:
    cached_df = None  # Ignore cache if force rescan flag is set
    metadata = None

# Cache version guard: old snapshots are one-row-PER-ALERT (Type values
# '🚀 Opportunity' / '⚠️ Alert'). The new layout is one-row-PER-STOCK with a
# Bias Type ('🚀 Bullish' / '⚠️ Bearish' / '⚖️ Mixed').
if cached_df is not None and 'Type' in cached_df.columns:
    _old_labels = {'🚀 Opportunity', '⚠️ Alert'}
    if cached_df['Type'].astype(str).isin(_old_labels).any():
        cached_df = None
        metadata = None

_expected = expected_latest_session()
_stale = bool(snap_date) and snap_date < _expected.strftime('%Y-%m-%d')

# Show cache status
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

with col1:
    if cached_df is not None:
        if _stale:
            st.warning(f"📦 快照 {snap_date}")
        else:
            st.success(f"✅ 最新快照 {snap_date}")
    else:
        st.info("📭 暂无快照")

with col2:
    if st.button("🔄 Force Rescan", type="secondary"):
        st.session_state.force_rescan = True
        st.rerun()
    if len(MY_WATCHLIST) > 25:
        st.caption(f"⚠️ {len(MY_WATCHLIST)} 只约需 {len(MY_WATCHLIST) * 18 // 60} 分钟 CPU，"
                   f"免费版可能被限流")

with col3:
    filter_type = st.selectbox(
        "Filter", ["All", "🚀 Bullish", "⚠️ Bearish", "⚖️ Mixed"])

with col4:
    min_signals = st.selectbox("Min Signals", [1, 2, 3, 4], index=0)

# Show scan metadata if available
if metadata:
    with st.expander("📊 Scan Information", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stocks Scanned", metadata['total_stocks_scanned'])
        with col2:
            st.metric("Scan Duration", f"{metadata['scan_duration_seconds']:.1f}s")
        with col3:
            st.metric("Opportunities", metadata['opportunities_found'])
        with col4:
            st.metric("Alerts", metadata['alerts_found'])
        st.caption(f"Last scanned: {metadata['created_at']}")

st.markdown("---")

# ==================== 筹码结构扫描 ====================
# Computed nightly by scan_watchlists.py in GitHub Actions and stored per
# ticker in chip_scan — the page only reads it, same as the signal snapshot.
# Kept out of daily_signals on purpose: that cache is a fixed 10-column
# schema, and chip structure is a dozen numbers per stock.
#
# Defined up here and called through _stop() because the signal section below
# st.stop()s on "no snapshot" and "no signals" — a quiet day still has a chip
# structure, and it used to be unreachable on exactly those days.

_CHIP_DISPLAY = {
    "ticker": "Ticker", "name": "Name", "setup_score": "评分", "setup_label": "结构",
    "price": "价格", "peak_price": "主峰", "n_peaks": "峰数", "winner_rate": "获利盘",
    "concentration": "集中度", "weight_avg": "平均成本", "pct_from_peak": "距主峰%",
    "converged": "收敛",
}


@st.cache_data(ttl=600, show_spinner=False)
def _chip_snapshot(day: str, decay: float, tickers: tuple):
    return data_manager.get_chip_scan(day, decay, list(tickers))


@st.cache_data(ttl=3600, show_spinner=False)
def _chip_live(tickers: tuple, day: str, decay: float):
    """Manual fallback, e.g. for a stock added since last night's run."""
    out, failed = [], []
    for tk in tickers:
        r = watchlist_scan.scan_chips(tk, decays=(decay,))
        if r["status"] == "ok":
            out.extend(r["rows"])
        else:
            failed.append(tk)
    return pd.DataFrame(out), failed


def render_chip_scan():
    st.subheader("🧮 筹码结构扫描 · Chip Structure Scan")
    st.caption(
        "谁在什么价位持有这只股票。找的是 **低位单峰密集**：筹码集中在一个价位、"
        "上方套牢盘轻、主峰在现价下方构成支撑。评分只描述结构，不是涨跌预测。"
    )
    if not MY_WATCHLIST:
        st.caption("自选股为空。")
        return

    chip_date = data_manager.get_latest_chip_scan_date()
    _cs1, _cs2, _cs3 = st.columns([1, 1, 2])
    with _cs1:
        decay = st.select_slider("衰减系数", options=list(watchlist_scan.CHIP_DECAYS),
                                 value=1.0, key="chipscan_decay")
    with _cs2:
        go_live = st.button("🔄 实时重算筹码", type="secondary", key="chipscan_go")
    with _cs3:
        if chip_date:
            st.caption(f"📦 夜间快照 {chip_date}（北京时间 20:00 由 GitHub Actions 计算）。"
                       f"实时重算约 {len(MY_WATCHLIST)} 只 × ~2 秒。")
        else:
            st.caption("还没有夜间快照。实时重算约 "
                       f"{len(MY_WATCHLIST)} 只 × ~2 秒。")

    if go_live:
        st.session_state["chipscan_live"] = True

    failed, missing = [], []
    if st.session_state.get("chipscan_live"):
        with st.spinner("计算筹码分布…"):
            cdf, failed = _chip_live(tuple(MY_WATCHLIST), today_str, decay)
        source = "实时计算"
    elif chip_date:
        cdf = _chip_snapshot(chip_date, decay, tuple(MY_WATCHLIST))
        source = f"快照 {chip_date}"
        have = set() if cdf is None or cdf.empty else set(cdf["ticker"].astype(str))
        missing = [t for t in MY_WATCHLIST if str(t) not in have]
    else:
        st.info("📭 还没有筹码快照。今晚的夜间扫描会生成；也可以点 **实时重算筹码**。")
        return

    if cdf is None or cdf.empty:
        st.info("没有可计算的股票（需要换手率数据与至少 120 个交易日）。")
        return

    cdf = cdf[[c for c in _CHIP_DISPLAY if c in cdf.columns]].rename(columns=_CHIP_DISPLAY)
    cdf["获利盘"] = (cdf["获利盘"] * 100).round(1)
    cdf["收敛"] = cdf["收敛"].map(lambda v: "✓" if bool(v) else "⚠")
    cdf = cdf.sort_values("评分", ascending=False)

    st.caption(f"数据来源：{source}")
    _unconv = int((cdf["收敛"] == "⚠").sum())
    if _unconv:
        st.caption(
            f"⚠ {_unconv} 只标记为未收敛：换手率太低，三年历史不足以冲掉初始假设，"
            f"它们的数字参考价值有限。")
    st.dataframe(
        cdf, use_container_width=True, hide_index=True,
        column_config={
            "评分": st.column_config.ProgressColumn(
                "评分", min_value=0.0, max_value=1.0, format="%.2f"),
            "获利盘": st.column_config.NumberColumn("获利盘 %", format="%.1f%%"),
            "距主峰%": st.column_config.NumberColumn("距主峰 %", format="%+.1f%%"),
        })
    st.caption(
        "**主峰** = 持有量最大的成本价位。**距主峰%** 为正表示现价在主峰上方"
        "（主峰构成支撑），为负表示主峰在上方（是压力）。"
        "**集中度** 越小筹码越集中。**峰数** >1 说明上方或下方还有另一批成本。")
    if missing:
        st.caption(f"快照中没有 {len(missing)} 只（今晚扫描后出现，或点实时重算）："
                   f"{', '.join(missing[:12])}" + ("…" if len(missing) > 12 else ""))
    if failed:
        st.caption(f"跳过 {len(failed)} 只：{', '.join(failed[:12])}"
                   + ("…" if len(failed) > 12 else ""))


def _stop():
    """st.stop(), but show the chip structure first — it does not need signals."""
    st.markdown("---")
    render_chip_scan()
    st.stop()


# ── Decide: use snapshot, explicit scan, or wait ─────────────────
if cached_df is not None:
    df = cached_df
    if _stale:
        st.info(f"📦 显示 {snap_date} 收盘的快照。夜间扫描在北京时间 20:00 运行；"
                f"若 {_expected} 是交易日，新快照会在那之后出现。")
    else:
        st.info(f"📦 显示 {snap_date} 收盘的快照（最新交易日）")
elif st.session_state.force_rescan:
    st.info(f"🎯 Scanning your watchlist ({len(MY_WATCHLIST)} stocks)")
    with st.spinner("🔍 Scanning all stocks for signals... This may take a few minutes."):
        df, scan_duration, data_date = scan_my_watchlist()
    st.session_state.force_rescan = False

    if df is None:
        st.error("❌ Scanning failed! ")
        _stop()

    if not df.empty:
        # Keyed by the session the data belongs to, same as the nightly job,
        # so the two never write competing snapshots under different dates.
        save_success = data_manager.save_signals_to_cache(
            df, data_date or today_str, scan_duration)
        if save_success:
            st.success(f"✅ Scan complete in {scan_duration:.1f}s — snapshot saved.")
        else:
            st.warning("⚠️ Scan complete but failed to save snapshot.")
    else:
        st.success("✨ No signals detected in your watchlist today.")
        st.info("💡 Market may be consolidating, or no strong trends detected.")
        _stop()
else:
    st.info("📭 还没有快照。夜间扫描（北京时间 20:00）会自动生成；"
            "也可以点上方 **Force Rescan** 立即扫描——自选股多时可能被免费版限流。")
    _stop()

if df.empty:
    st.success("✨ No signals detected today. Market is quiet!")
    st.info("💡 This could mean:\n- All stocks are in neutral zones\n- No strong trends detected\n- Market is consolidating")
    _stop()

# Derive bull/bear counts from the ▲/▼ markers in the Signals string so the
# breakdown survives a cache round-trip (the DB stores only the 10-col schema).
def _count_dir(sig, mark):
    return str(sig).count(mark)
df['Bull'] = df['Signals'].apply(lambda s: _count_dir(s, '▲'))
df['Bear'] = df['Signals'].apply(lambda s: _count_dir(s, '▼'))

# Apply filters
filtered_df = df.copy()
if filter_type == "🚀 Bullish":
    filtered_df = filtered_df[filtered_df['Type'] == '🚀 Bullish']
elif filter_type == "⚠️ Bearish":
    filtered_df = filtered_df[filtered_df['Type'] == '⚠️ Bearish']
elif filter_type == "⚖️ Mixed":
    filtered_df = filtered_df[filtered_df['Type'] == '⚖️ Mixed']

filtered_df = filtered_df[filtered_df['Signal_Count'] >= min_signals]

# ==================== SUMMARY STATS ====================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🚀 Bullish", int((df['Type'] == '🚀 Bullish').sum()))
with col2:
    st.metric("⚠️ Bearish", int((df['Type'] == '⚠️ Bearish').sum()))
with col3:
    st.metric("⚖️ Mixed", int((df['Type'] == '⚖️ Mixed').sum()))
with col4:
    st.metric("📊 Stocks with signals", len(df))

st.markdown("---")

# ==================== 箱体 EDGE ALERTS ====================
# Pulled back out of the Signals string so this works from a cached snapshot
# too — the cache keeps only the 10-column schema, and the string is the one
# field that survives the round trip.
import re as _re

_BOX_PAT = _re.compile(
    r"([▲▼])\s*箱体(下沿|上沿|突破|跌破)\s*\[([\d.]+)-([\d.]+)\]\s*"
    r"位置(-?\d+)%\s*触(\d+)/(\d+)\s*质([\d.]+)")

_box_rows = []
for _, _r in df.iterrows():
    for _m in _BOX_PAT.finditer(str(_r.get("Signals", ""))):
        _mark, _where, _bot, _top, _pos, _tt, _tb, _q = _m.groups()
        _box_rows.append({
            "Ticker": _r["Ticker"], "Name": _r["Name"],
            "Price": float(_r.get("Price", 0)),
            "位置": f"{_pos}%",
            "_pos": int(_pos),
            "箱体": f"¥{float(_bot):.2f} – ¥{float(_top):.2f}",
            "幅度": f"{(float(_top)/float(_bot)-1)*100:.1f}%",
            "信号": f"{_mark} {_where}",
            "触及": f"上{_tt}/下{_tb}",
            "质量": float(_q),
        })

st.subheader("📦 箱体边缘提醒 · Box Edge Alerts")
if not _box_rows:
    st.info("观察列表中没有股票正处于箱体上下沿。"
            "（只有上下沿都被反复确认、且走势没有方向的区间才算箱体）")
else:
    _bdf = pd.DataFrame(_box_rows).sort_values(
        ["信号", "质量"], ascending=[True, False])
    _c1, _c2 = st.columns(2)
    # NOTE: these must be if/else STATEMENTS, not `a() if c else b()` used as
    # an expression statement. Streamlit's magic renders the value of any bare
    # expression in the script, so the conditional-expression form handed it
    # the DeltaGenerator returned by st.dataframe and it printed that object's
    # repr and full docstring onto the page.
    with _c1:
        _low = _bdf[_bdf["信号"].str.contains("下沿|跌破")]
        st.markdown(f"**🔻 贴近下沿 / 跌破 · {len(_low)}**")
        st.caption("支撑位——箱体交易的买入侧，跌破则是失效信号")
        if not _low.empty:
            st.dataframe(_low.drop(columns=["_pos"]), use_container_width=True,
                         hide_index=True)
        else:
            st.caption("—")
    with _c2:
        _high = _bdf[_bdf["信号"].str.contains("上沿|突破")]
        st.markdown(f"**🔺 贴近上沿 / 突破 · {len(_high)}**")
        st.caption("压力位——箱体交易的卖出侧，突破则是启动信号")
        if not _high.empty:
            st.dataframe(_high.drop(columns=["_pos"]), use_container_width=True,
                         hide_index=True)
        else:
            st.caption("—")
    st.caption("质量 0–1：由上下沿触及次数、走势平坦度、收盘留在箱内的比例、"
               "以及持续时间加权得出。0.7 以上是结构清晰的箱体。")

st.markdown("---")

render_chip_scan()

st.markdown("---")

# ==================== DISPLAY TABLE ====================
if filtered_df.empty:
    st.warning(f"No results match your filters (Filter: {filter_type}, Min Signals: {min_signals})")
else:
    st.subheader(f"Found {len(filtered_df)} stocks with signals")

    display_df = filtered_df.copy()
    display_df['Price']  = display_df['Price'].apply(lambda x: f"¥{x:.2f}")
    display_df['RSI']    = display_df['RSI'].apply(lambda x: f"{x:.1f}")
    display_df['ADX']    = display_df['ADX'].apply(lambda x: f"{x:.1f}")
    display_df['MACD']   = display_df['MACD'].apply(lambda x: f"{x:.4f}")
    display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")

    # Code → deep link into Technical Analysis (?ticker= seeds active_ticker).
    display_df['Ticker'] = display_df['Ticker'].apply(
        lambda t: f"/single-stock-analysis?ticker={t}")

    # One row per stock: Bias · Code · Name · ▲ · ▼ · Signals · indicators
    display_df = display_df[['Type', 'Ticker', 'Name', 'Bull', 'Bear',
                             'Signals', 'Price', 'RSI', 'ADX', 'MACD', 'Volume']]
    display_df.columns = ['Bias', 'Code', 'Stock Name', '▲', '▼',
                          'Signal Details', 'Price', 'RSI', 'ADX', 'MACD', 'Volume']

    st.dataframe(
        display_df,
        use_container_width=True,
        height=600,
        hide_index=True,
        column_config={
            "Bias": st.column_config.TextColumn(
                "Bias", width="small",
                help="Net bias: 🚀 Bullish (only bull signals) · ⚠️ Bearish (only bear) · ⚖️ Mixed (both)"),
            "Code": st.column_config.LinkColumn(
                "Code", width="small",
                display_text=r"ticker=(.+)$",
                help="Click to open this stock in Technical Analysis 技术分析"),
            "Stock Name": st.column_config.TextColumn("Stock Name", width="medium"),
            "▲": st.column_config.NumberColumn("▲", width="small", help="Bullish signal count"),
            "▼": st.column_config.NumberColumn("▼", width="small", help="Bearish signal count"),
            "Signal Details": st.column_config.TextColumn(
                "Signal Details", width="large",
                help="▲ = bullish marker, ▼ = bearish marker — same set as the Technical Analysis chart"),
            "Price": st.column_config.TextColumn("Price", width="small"),
            "RSI": st.column_config.TextColumn("RSI", width="small"),
            "ADX": st.column_config.TextColumn("ADX", width="small"),
            "MACD": st.column_config.TextColumn("MACD", width="small"),
            "Volume": st.column_config.TextColumn("Volume", width="small"),
        }
    )
    
    # ==================== DOWNLOAD BUTTON ====================
    st.markdown("---")
    
    csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"signals_{today_str}.csv",
        mime="text/csv"
    )

# ==================== LEGEND ====================
st.markdown("---")
st.markdown("### 📖 Signal Definitions")
st.caption(
    "These are the **exact same discrete signals plotted on the Technical "
    "Analysis chart** — an alert here means the same marker appears there. "
    "ADX-based buy/sell signals are direction-gated by +DI vs −DI (the chart's "
    "Entry/Exit candidates), not a separate price-trend rule."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**▲ Bullish markers**")
    st.markdown("""
    - **Strength Returning (ADX)** — ADX Bottoming / Reversing Up while +DI dominant
    - **Trend Accelerating (ADX)** — ADX Accelerating Up while +DI dominant
    - **DI Screaming Buy 🚀** — fresh +DI bullish cross with momentum blow-out
    - **Bullish Squeeze Breakout 🚀** — Bollinger squeeze fired upward
    - **Phase 1: Accumulation** — OBV-divergence accumulation phase
    - **MACD Bottoming** — MACD stopped falling, turning up
    - **MACD Bullish Crossover** — MACD crossed above its signal line
    - **RSI Bottoming** — RSI in the bottom decile, turning up
    """)

with col2:
    st.markdown("**▼ Bearish markers**")
    st.markdown("""
    - **Trend Topping (ADX)** — ADX Peaking / Reversing Down while +DI dominant
    - **Trend Collapsing (ADX)** — ADX Accelerating Down while +DI dominant
    - **DI Screaming Sell 🛑** — fresh −DI bearish cross with momentum blow-out
    - **Bearish Squeeze Drop 🩸** — Bollinger squeeze fired downward
    - **MACD Exit Signal** — MACD-lead exit (bearish cross / MA cross-down)
    - **MACD Peaking** — MACD stopped rising, turning down
    - **MACD Bearish Crossover** — MACD crossed below its signal line
    - **RSI Peaking** — RSI in the top decile, turning down
    """)

st.markdown("---")
st.caption(
    "💡 One row per stock · **Bias** = 🚀 Bullish (only ▲) / ⚠️ Bearish (only ▼) / "
    "⚖️ Mixed (both). Results are cached daily; click 'Force Rescan' for fresh data."
)
