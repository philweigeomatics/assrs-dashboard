"""
Watchlist Management — 观察列表管理
Each user manages their own watchlist.

Performance notes:
  - @st.cache_data wrappers keep the 3 Supabase round-trips out of the hot
    path.  Cache is keyed by user_id so multi-user data never crosses.
  - @st.fragment scopes row-click reruns to just the grid section; the add
    stock form and bulk import above are never re-executed on a row click.
  - st.switch_page cannot be called inside a fragment, so "Analyze" writes a
    session-state flag that the navigation bridge (outside the fragment) picks
    up on the next full-app rerun.
"""

import streamlit as st
import pandas as pd
import data_manager
import auth_manager
import supply_chain_ui
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# ── Auth ───────────────────────────────────────────────────────────────────────
auth_manager.require_login()
user = auth_manager.get_current_user()
_uid = auth_manager.get_current_user_id()  # used as cache-key discriminator

# ── Per-user cached data loaders ───────────────────────────────────────────────
# user_id in the signature makes each entry user-specific (no cross-user leakage).
# Call .clear() after any mutation so the next read is fresh.

@st.cache_data(ttl=30, show_spinner=False)
def _load_watchlist(user_id):
    return data_manager.get_watchlist()

@st.cache_data(ttl=30, show_spinner=False)
def _load_graphed_tickers(user_id):
    return data_manager.get_all_supply_chain_tickers()

@st.cache_data(ttl=300, show_spinner=False)
def _load_daily_basic(tickers_tuple, user_id):
    return data_manager.get_daily_basic_latest(list(tickers_tuple))

# ── Supply Chain Graph Dialog ──────────────────────────────────────────────────
# Must be at module level (not inside a fragment or function).
@st.dialog("Supply Chain Graph 供应链图谱", width="large")
def _show_graph_dialog(ticker: str, graph_data: dict) -> None:
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
col1, col2 = st.columns([3, 1])
with col1:
    new_ticker = st.text_input(
        "Stock Code 股票代码",
        placeholder="e.g., 600519",
        key="add_ticker",
        help="Enter 6-digit stock code",
    )
with col2:
    st.write("")
    st.write("")
    if st.button("➕ Add Stock", type="primary", use_container_width=True):
        if new_ticker:
            ticker_clean = new_ticker.strip()
            if not (len(ticker_clean) == 6 and ticker_clean.isdigit()):
                st.error("❌ Invalid ticker format. Must be 6 digits (e.g., 600519)")
            else:
                success, message = data_manager.add_to_watchlist(ticker_clean)
                if success:
                    _load_watchlist.clear()
                    _load_daily_basic.clear()
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        else:
            st.warning("⚠️ Please enter a stock code")

st.markdown("---")

# ── Bulk Import ────────────────────────────────────────────────────────────────
with st.expander("📥 Bulk Import | 批量导入", expanded=False):
    st.markdown("**Import multiple stocks at once**")
    st.caption("Paste ticker codes, one per line (6 digits only)")
    bulk_input = st.text_area(
        "Ticker Codes",
        placeholder="600519\n000333\n300750\n...",
        height=150,
        key="bulk_input",
    )
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📥 Import All", type="primary"):
            if bulk_input:
                tickers = [t.strip() for t in bulk_input.split("\n") if t.strip()]
                tickers = [t for t in tickers if len(t) == 6 and t.isdigit()]
                if tickers:
                    with st.spinner(f"Importing {len(tickers)} stocks..."):
                        success_count, failed_count, msgs = data_manager.bulk_add_to_watchlist(tickers)
                    _load_watchlist.clear()
                    _load_daily_basic.clear()
                    st.success(f"✅ Imported {success_count} stocks")
                    if failed_count > 0:
                        st.warning(f"⚠️ {failed_count} failed (already in watchlist or errors)")
                        with st.expander("Show errors"):
                            for msg in msgs:
                                st.write(msg)
                    st.rerun()
                else:
                    st.warning("⚠️ No valid 6-digit tickers found")
            else:
                st.warning("⚠️ Please paste ticker codes")

st.markdown("---")

# ── Navigation bridge ──────────────────────────────────────────────────────────
# st.switch_page cannot be called inside @st.fragment, so the Analyze button
# sets this flag and triggers a full-app rerun; we intercept it here.
if st.session_state.get("_goto_analysis"):
    ticker = st.session_state.pop("_goto_analysis")
    st.session_state.active_ticker = ticker
    st.switch_page("pages/2_Single_Stock_Analysis_个股分析.py")


# ── Watchlist Grid ─────────────────────────────────────────────────────────────
# Wrapped in @st.fragment: row clicks, pagination, and search only rerun this
# section — the add-stock form and bulk import above are untouched.
@st.fragment
def _watchlist_grid():
    st.subheader("📋 Current Watchlist")
    watchlist_data = _load_watchlist(_uid)

    if not watchlist_data:
        st.info("📭 Your watchlist is empty. Add stocks above to get started.")
        return

    st.write(f"**Total: {len(watchlist_data)} stocks**")
    df = pd.DataFrame(watchlist_data)[["ticker", "stock_name", "added_date"]]
    df = df.sort_values("added_date", ascending=False)

    graphed_tickers: set = _load_graphed_tickers(_uid)

    # ── Search ─────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input(
            "🔍 Search by ticker or name | 搜索代码或名称",
            placeholder="e.g., 600519 or 茅台",
            key="search_input",
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Clear Search", use_container_width=True):
            st.rerun()

    if search_term:
        filtered_df = df[
            df["ticker"].str.contains(search_term, case=False, na=False)
            | df["stock_name"].str.contains(search_term, case=False, na=False)
        ]
        if filtered_df.empty:
            st.warning(f"⚠️ No stocks found matching '{search_term}'")
            filtered_df = df
        else:
            st.info(f"📊 Found {len(filtered_df)} stock(s) matching '{search_term}'")
    else:
        filtered_df = df

    # ── Pagination ─────────────────────────────────────────────────────────────
    ITEMS_PER_PAGE = 10
    total_items = len(filtered_df)
    total_pages = max((total_items - 1) // ITEMS_PER_PAGE + 1, 1)

    if "watchlist_page" not in st.session_state:
        st.session_state.watchlist_page = 0
    if "last_search" not in st.session_state:
        st.session_state.last_search = ""

    if search_term != st.session_state.last_search:
        st.session_state.watchlist_page = 0
        st.session_state.last_search = search_term

    st.session_state.watchlist_page = max(
        0, min(st.session_state.watchlist_page, total_pages - 1)
    )

    start_idx = st.session_state.watchlist_page * ITEMS_PER_PAGE
    end_idx   = min(start_idx + ITEMS_PER_PAGE, total_items)

    # ── Enrich: daily basic + graph status ─────────────────────────────────────
    all_tickers = filtered_df["ticker"].tolist()
    daily_df    = _load_daily_basic(tuple(sorted(all_tickers)), _uid)
    if not daily_df.empty:
        filtered_df = filtered_df.merge(daily_df, on="ticker", how="left")

    filtered_df["supply_chain"] = filtered_df["ticker"].apply(
        lambda t: "✅" if t in graphed_tickers else "—"
    )

    page_df = filtered_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    # ── Delete Confirmation ────────────────────────────────────────────────────
    if st.session_state.get("pending_delete"):
        ticker_to_delete = st.session_state.pending_delete
        st.warning(
            f"⚠️ Are you sure you want to remove **{ticker_to_delete}** from your watchlist?"
        )
        col_confirm, col_cancel, _ = st.columns([1, 1, 4])
        if col_confirm.button("✅ Confirm Delete", type="primary"):
            success, message = data_manager.remove_from_watchlist(ticker_to_delete)
            st.success(message) if success else st.error(message)
            _load_watchlist.clear()
            _load_daily_basic.clear()
            st.session_state.pending_delete = None
            st.rerun()
        if col_cancel.button("❌ Cancel"):
            st.session_state.pending_delete = None
            st.rerun()
        st.divider()

    # ── AgGrid Table ───────────────────────────────────────────────────────────
    gb = GridOptionsBuilder.from_dataframe(page_df)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_column("ticker",        header_name="代码",        width=110)
    gb.configure_column("stock_name",    header_name="名称",        flex=1)
    gb.configure_column("supply_chain",  header_name="图谱",        width=70)
    gb.configure_column("close",         header_name="收盘价",      width=100)
    gb.configure_column("pe_ttm",        header_name="PE(TTM)",     width=100)
    gb.configure_column("pb",            header_name="PB",          width=90)
    gb.configure_column("turnover_rate", header_name="换手率%",     width=100)
    gb.configure_column("circ_mv_yi",    header_name="流通市值(亿)", width=130)
    gb.configure_column("trade_date",    header_name="数据日期",    width=120)
    gb.configure_column("added_date",    header_name="添加日期",    width=120)
    gb.configure_grid_options(rowHeight=40, suppressMovableColumns=True)
    grid_options = gb.build()

    grid_result = AgGrid(
        page_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        use_container_width=True,
        height=min(80 + len(page_df) * 40, 480),
        theme="streamlit",
    )

    # ── Row Action Buttons ─────────────────────────────────────────────────────
    selected_rows = grid_result.get("selected_rows", [])
    if selected_rows is not None and not selected_rows.empty:
        selected_ticker = selected_rows.iloc[0]["ticker"]
        selected_name   = selected_rows.iloc[0]["stock_name"]
        has_graph       = selected_ticker in graphed_tickers

        st.info(f"Selected: **{selected_name} ({selected_ticker})**")

        if has_graph:
            c_analyze, c_view, c_update, c_remove, _ = st.columns([1.2, 1.2, 1.4, 1.2, 2.5])
        else:
            c_analyze, c_generate, c_remove, _ = st.columns([1.2, 1.6, 1.2, 3.5])

        # Analyze — needs full-app rerun so st.switch_page works outside the fragment
        if c_analyze.button("🔍 Analyze", type="primary", use_container_width=True):
            st.session_state["_goto_analysis"] = selected_ticker
            st.rerun(scope="app")

        if has_graph:
            if c_view.button("📊 View Graph", use_container_width=True):
                graph_data = data_manager.get_supply_chain_graph(selected_ticker)
                if graph_data:
                    _show_graph_dialog(selected_ticker, graph_data)
                else:
                    st.error("Graph record found but data could not be loaded.")

            if c_update.button("🔄 Update Graph", use_container_width=True):
                with st.spinner(f"Re-generating graph for {selected_name}…"):
                    try:
                        graph_data = supply_chain_ui.generate_supply_chain_graph(
                            selected_ticker, selected_name
                        )
                        if data_manager.upsert_supply_chain_graph(
                            selected_ticker, selected_name, graph_data
                        ):
                            _load_graphed_tickers.clear()
                            st.success(f"✅ Graph updated for {selected_name}")
                            st.rerun()
                        else:
                            st.error("❌ Graph generated but failed to save.")
                    except RuntimeError as exc:
                        st.error(f"❌ {exc}")

            if c_remove.button("🗑️ Remove", use_container_width=True):
                st.session_state.pending_delete = selected_ticker
                st.rerun()

        else:
            if c_generate.button("🧬 Generate Graph", use_container_width=True):
                with st.spinner(f"Generating supply chain graph for {selected_name}…"):
                    try:
                        graph_data = supply_chain_ui.generate_supply_chain_graph(
                            selected_ticker, selected_name
                        )
                        if data_manager.upsert_supply_chain_graph(
                            selected_ticker, selected_name, graph_data
                        ):
                            _load_graphed_tickers.clear()
                            st.success(f"✅ Graph generated for {selected_name}!")
                            st.rerun()
                        else:
                            st.error("❌ Graph generated but failed to save.")
                    except RuntimeError as exc:
                        st.error(f"❌ {exc}")

            if c_remove.button("🗑️ Remove", use_container_width=True):
                st.session_state.pending_delete = selected_ticker
                st.rerun()
    else:
        st.caption("💡 Click a row to select it, then use the action buttons")

    # ── Pagination Controls ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button(
            "⬅️ Previous",
            disabled=(st.session_state.watchlist_page == 0),
            use_container_width=True,
        ):
            st.session_state.watchlist_page -= 1
            st.rerun()
    with col2:
        st.markdown(
            f"<div style='text-align:center; padding-top:8px'>"
            f"Page <b>{st.session_state.watchlist_page + 1}</b> of <b>{total_pages}</b>"
            f"&nbsp;|&nbsp; Showing {start_idx + 1}–{end_idx} of {total_items}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col3:
        if st.button(
            "➡️ Next",
            disabled=(st.session_state.watchlist_page >= total_pages - 1),
            use_container_width=True,
        ):
            st.session_state.watchlist_page += 1
            st.rerun()


_watchlist_grid()
