"""
Watchlist Management — 观察列表管理
Lean version tuned for Streamlit Community Cloud (~1 GB shared RAM).

Memory-saving choices:
  - st.dataframe replaces st-aggrid (AgGrid keeps a heavy JS+server state).
  - Watchlist + daily_basic merged into ONE cached DataFrame per user.
    Cache key = user_id only (no ticker tuples) → at most max_entries=20.
  - Caches explicitly .clear()'d after mutations.
  - @st.fragment scopes row-click reruns so the add-stock form / bulk
    importer above never re-execute on a click.
"""

import streamlit as st
import pandas as pd

import data_manager
import auth_manager
import supply_chain_ui
from nav_helpers import page_link_button

@st.cache_data(ttl=3600, show_spinner=False)
def _all_stock_options_wl():
    stocks = data_manager.get_all_stock_basic()
    return [""] + [f"{s['ticker']} · {s['name']}" for s in stocks]

# ── Auth ───────────────────────────────────────────────────────────────────────
auth_manager.require_login()
user = auth_manager.get_current_user()
_uid = auth_manager.get_current_user_id()

# ── Cached loaders (1 entry per user, capped) ──────────────────────────────────
@st.cache_data(ttl=60, max_entries=20, show_spinner=False)
def _load_enriched_watchlist(user_id):
    """Return watchlist DF with daily-basic columns merged in. One DB hit per cache."""
    items = data_manager.get_watchlist()
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)[["ticker", "stock_name", "added_date"]]
    daily = data_manager.get_daily_basic_latest(df["ticker"].tolist())
    if daily is not None and not daily.empty:
        df = df.merge(daily, on="ticker", how="left")
    return df.sort_values("added_date", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=60, max_entries=20, show_spinner=False)
def _load_graphed_tickers(user_id):
    return data_manager.get_all_supply_chain_tickers()

def _clear_data_cache():
    _load_enriched_watchlist.clear()

def _clear_graph_cache():
    _load_graphed_tickers.clear()

# ── Supply Chain Graph Dialog ──────────────────────────────────────────────────
@st.dialog("Supply Chain Graph 供应链图谱", width="large")
def _show_graph_dialog(ticker, graph_data):
    company  = graph_data.get("company_name", ticker)
    products = graph_data.get("products", [])
    sectors  = graph_data.get("macro_sectors", [])
    st.caption(f"**{company}** · {ticker}")
    c1, c2 = st.columns(2)
    c1.markdown("📦 **Products:** " + " · ".join(products))
    c2.markdown("🏭 **Sectors:** "  + " · ".join(sectors))
    supply_chain_ui.render_supply_chain_graph(graph_data, height=570)

# ── Page Header ────────────────────────────────────────────────────────────────
st.title("📋 Watchlist Management | 观察列表管理")
st.caption(f"👤 {user['username']}'s watchlist")
st.markdown("---")

# ── Add Stock ──────────────────────────────────────────────────────────────────
st.subheader("➕ Add Stock to Watchlist")
c1, c2 = st.columns([3, 1])
with c1:
    new_ticker_raw = st.selectbox(
        "Stock Code or Name 股票代码或名称",
        options=_all_stock_options_wl(),
        key="add_ticker",
        format_func=lambda x: "Type to search… (code or name)" if x == "" else x,
    )
with c2:
    st.write("")
    st.write("")
    if st.button("➕ Add Stock", type="primary", use_container_width=True):
        _raw = (new_ticker_raw or "").strip()
        t = _raw.split(" · ")[0].strip() if " · " in _raw else _raw
        if not t:
            st.warning("⚠️ Please select a stock first")
        elif not (len(t) == 6 and t.isdigit()):
            st.error("❌ Invalid selection — please choose from the list")
        else:
            ok, msg = data_manager.add_to_watchlist(t)
            if ok:
                _clear_data_cache()
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

st.markdown("---")

# ── Bulk Import ────────────────────────────────────────────────────────────────
with st.expander("📥 Bulk Import | 批量导入", expanded=False):
    st.caption("Paste ticker codes, one per line (6 digits only)")
    bulk = st.text_area(
        "Ticker Codes", placeholder="600519\n000333\n300750\n…",
        height=140, key="bulk_input",
    )
    if st.button("📥 Import All", type="primary"):
        if bulk:
            tickers = [t.strip() for t in bulk.split("\n") if t.strip()]
            tickers = [t for t in tickers if len(t) == 6 and t.isdigit()]
            if tickers:
                with st.spinner(f"Importing {len(tickers)} stocks…"):
                    n_ok, n_fail, msgs = data_manager.bulk_add_to_watchlist(tickers)
                _clear_data_cache()
                st.success(f"✅ Imported {n_ok} stocks")
                if n_fail:
                    st.warning(f"⚠️ {n_fail} failed")
                    with st.expander("Show errors"):
                        for m in msgs:
                            st.write(m)
                st.rerun()
            else:
                st.warning("⚠️ No valid 6-digit tickers found")
        else:
            st.warning("⚠️ Please paste ticker codes")

st.markdown("---")


# ── Watchlist Grid (fragment-scoped) ───────────────────────────────────────────
@st.fragment
def _watchlist_grid():
    st.subheader("📋 Current Watchlist")

    df = _load_enriched_watchlist(_uid)
    if df.empty:
        st.info("📭 Your watchlist is empty. Add stocks above to get started.")
        return

    graphed = _load_graphed_tickers(_uid)

    # ── Search ─────────────────────────────────────────────────────────────────
    search = st.text_input(
        "🔍 Search by ticker or name | 搜索代码或名称",
        placeholder="e.g., 600519 or 茅台",
        key="search_input",
    )

    if search:
        mask = (df["ticker"].str.contains(search, case=False, na=False)
                | df["stock_name"].str.contains(search, case=False, na=False))
        view = df[mask]
        if view.empty:
            st.warning(f"⚠️ No matches for '{search}'")
            view = df
        else:
            st.caption(f"📊 {len(view)} of {len(df)} match '{search}'")
    else:
        view = df

    # ── Pagination state ───────────────────────────────────────────────────────
    PAGE_SIZE = 10
    n_pages = max((len(view) - 1) // PAGE_SIZE + 1, 1)

    if "watchlist_page" not in st.session_state:
        st.session_state.watchlist_page = 0
    if search != st.session_state.get("last_search", ""):
        st.session_state.watchlist_page = 0
        st.session_state.last_search = search
    page = max(0, min(st.session_state.watchlist_page, n_pages - 1))
    st.session_state.watchlist_page = page

    s, e = page * PAGE_SIZE, min((page + 1) * PAGE_SIZE, len(view))
    page_df = view.iloc[s:e].copy().reset_index(drop=True)
    page_df["graph"] = page_df["ticker"].apply(lambda t: "✅" if t in graphed else "—")

    # ── Delete Confirmation ────────────────────────────────────────────────────
    if st.session_state.get("pending_delete"):
        td = st.session_state.pending_delete
        st.warning(f"⚠️ Remove **{td}** from your watchlist?")
        cf, cn, _ = st.columns([1, 1, 4])
        if cf.button("✅ Confirm", type="primary"):
            ok, msg = data_manager.remove_from_watchlist(td)
            (st.success if ok else st.error)(msg)
            _clear_data_cache()
            st.session_state.pending_delete = None
            st.rerun()
        if cn.button("❌ Cancel"):
            st.session_state.pending_delete = None
            st.rerun()
        st.divider()

    # ── Native st.dataframe (much lighter than AgGrid) ─────────────────────────
    cols = ["ticker", "stock_name", "graph", "close", "pe_ttm", "pb",
            "turnover_rate", "circ_mv_yi", "trade_date", "added_date"]
    display_df = page_df[[c for c in cols if c in page_df.columns]]

    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(80 + len(display_df) * 36, 440),
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "ticker":        st.column_config.TextColumn("代码",        width="small"),
            "stock_name":    st.column_config.TextColumn("名称",        width="medium"),
            "graph":         st.column_config.TextColumn("图谱",        width="small"),
            "close":         st.column_config.NumberColumn("收盘价",     format="%.2f"),
            "pe_ttm":        st.column_config.NumberColumn("PE(TTM)",   format="%.1f"),
            "pb":            st.column_config.NumberColumn("PB",        format="%.2f"),
            "turnover_rate": st.column_config.NumberColumn("换手率%",   format="%.2f"),
            "circ_mv_yi":    st.column_config.NumberColumn("流通市值(亿)", format="%.1f"),
            "trade_date":    st.column_config.TextColumn("数据日期"),
            "added_date":    st.column_config.TextColumn("添加日期"),
        },
    )

    # ── Pagination Controls ────────────────────────────────────────────────────
    p1, p2, p3 = st.columns([1, 2, 1])
    if p1.button("⬅️ Previous", disabled=(page == 0), use_container_width=True):
        st.session_state.watchlist_page -= 1
        st.rerun()
    p2.markdown(
        f"<div style='text-align:center;padding-top:8px'>"
        f"Page <b>{page + 1}</b> / <b>{n_pages}</b> · "
        f"Showing {s + 1}–{e} of {len(view)}</div>",
        unsafe_allow_html=True,
    )
    if p3.button("➡️ Next", disabled=(page >= n_pages - 1), use_container_width=True):
        st.session_state.watchlist_page += 1
        st.rerun()

    # ── Row Action Buttons ─────────────────────────────────────────────────────
    sel_rows = getattr(getattr(event, "selection", None), "rows", []) or []
    if not sel_rows:
        st.caption("💡 Click a row to select it, then use the action buttons below")
        return

    idx        = sel_rows[0]
    sel_ticker = page_df.iloc[idx]["ticker"]
    sel_name   = page_df.iloc[idx]["stock_name"]
    has_graph  = sel_ticker in graphed

    st.info(f"Selected: **{sel_name} ({sel_ticker})**")

    if has_graph:
        ca, cv, cu, cr, _ = st.columns([1.2, 1.2, 1.4, 1.2, 2.5])
    else:
        ca, cg, cr, _ = st.columns([1.2, 1.6, 1.2, 3.5])

    with ca:
        page_link_button(
            "single-stock-analysis",
            "🔍 Analyze",
            params={"ticker": sel_ticker},
            style="primary",
            use_container_width=True,
            help="Open in Single Stock Analysis. "
                 "Middle/right-click → Open in new tab.",
        )

    if has_graph:
        if cv.button("📊 View Graph", use_container_width=True):
            gd = data_manager.get_supply_chain_graph(sel_ticker)
            if gd:
                _show_graph_dialog(sel_ticker, gd)
            else:
                st.error("Graph record found but data could not be loaded.")

        if cu.button("🔄 Update Graph", use_container_width=True):
            with st.spinner(f"Re-generating graph for {sel_name}…"):
                try:
                    gd = supply_chain_ui.generate_supply_chain_graph(sel_ticker, sel_name)
                    if data_manager.upsert_supply_chain_graph(sel_ticker, sel_name, gd):
                        _clear_graph_cache()
                        st.success(f"✅ Updated for {sel_name}")
                        st.rerun()
                    else:
                        st.error("❌ Graph generated but failed to save.")
                except RuntimeError as exc:
                    st.error(f"❌ {exc}")

        if cr.button("🗑️ Remove", use_container_width=True):
            st.session_state.pending_delete = sel_ticker
            st.rerun()
    else:
        if cg.button("🧬 Generate Graph", use_container_width=True):
            with st.spinner(f"Generating graph for {sel_name}…"):
                try:
                    gd = supply_chain_ui.generate_supply_chain_graph(sel_ticker, sel_name)
                    if data_manager.upsert_supply_chain_graph(sel_ticker, sel_name, gd):
                        _clear_graph_cache()
                        st.success(f"✅ Generated for {sel_name}")
                        st.rerun()
                    else:
                        st.error("❌ Graph generated but failed to save.")
                except RuntimeError as exc:
                    st.error(f"❌ {exc}")

        if cr.button("🗑️ Remove", use_container_width=True):
            st.session_state.pending_delete = sel_ticker
            st.rerun()


_watchlist_grid()
