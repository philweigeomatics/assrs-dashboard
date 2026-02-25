"""
Watchlist Management - 观察列表管理
Each user manages their own watchlist
"""
import streamlit as st
import pandas as pd
import data_manager
import auth_manager
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# ==================== PAGE CONFIG ====================
auth_manager.require_login()
user = auth_manager.get_current_user()

st.title("📋 Watchlist Management | 观察列表管理")
st.caption(f"👤 {user['username']}'s watchlist")
st.markdown("---")

# ==================== ADD STOCK SECTION ====================
st.subheader("➕ Add Stock to Watchlist")
col1, col2 = st.columns([3, 1])
with col1:
    new_ticker = st.text_input(
        "Stock Code 股票代码",
        placeholder="e.g., 600519",
        key="add_ticker",
        help="Enter 6-digit stock code"
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
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        else:
            st.warning("⚠️ Please enter a stock code")

st.markdown("---")

# ==================== BULK IMPORT SECTION ====================
with st.expander("📥 Bulk Import | 批量导入", expanded=False):
    st.markdown("**Import multiple stocks at once**")
    st.caption("Paste ticker codes, one per line (6 digits only)")
    bulk_input = st.text_area(
        "Ticker Codes",
        placeholder="600519\n000333\n300750\n...",
        height=150,
        key="bulk_input"
    )
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📥 Import All", type="primary"):
            if bulk_input:
                tickers = [t.strip() for t in bulk_input.split('\n') if t.strip()]
                tickers = [t for t in tickers if len(t) == 6 and t.isdigit()]
                if tickers:
                    with st.spinner(f"Importing {len(tickers)} stocks..."):
                        success_count, failed_count, msgs = data_manager.bulk_add_to_watchlist(tickers)
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

# ==================== WATCHLIST DISPLAY ====================
st.subheader("📋 Current Watchlist")
watchlist_data = data_manager.get_watchlist()

if not watchlist_data:
    st.info("📭 Your watchlist is empty. Add stocks above to get started.")
else:
    st.write(f"**Total: {len(watchlist_data)} stocks**")
    df = pd.DataFrame(watchlist_data)
    df = df[['ticker', 'stock_name', 'added_date']]
    df = df.sort_values('added_date', ascending=False)

    # ==================== SEARCH FEATURE ====================
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input(
            "🔍 Search by ticker or name | 搜索代码或名称",
            placeholder="e.g., 600519 or 茅台",
            key="search_input"
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Clear Search", use_container_width=True):
            st.rerun()

    if search_term:
        filtered_df = df[
            df['ticker'].str.contains(search_term, case=False, na=False) |
            df['stock_name'].str.contains(search_term, case=False, na=False)
        ]
        if filtered_df.empty:
            st.warning(f"⚠️ No stocks found matching '{search_term}'")
            filtered_df = df
        else:
            st.info(f"📊 Found {len(filtered_df)} stock(s) matching '{search_term}'")
    else:
        filtered_df = df

    # ==================== PAGINATION ====================
    ITEMS_PER_PAGE = 10
    total_items = len(filtered_df)
    total_pages = (total_items - 1) // ITEMS_PER_PAGE + 1 if total_items > 0 else 1

    if 'watchlist_page' not in st.session_state:
        st.session_state.watchlist_page = 0
    if 'last_search' not in st.session_state:
        st.session_state.last_search = ""

    if search_term != st.session_state.last_search:
        st.session_state.watchlist_page = 0
        st.session_state.last_search = search_term

    st.session_state.watchlist_page = max(0, min(st.session_state.watchlist_page, total_pages - 1))

    start_idx = st.session_state.watchlist_page * ITEMS_PER_PAGE
    end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)
    page_df = filtered_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    # ==================== DELETE CONFIRMATION ====================
    if st.session_state.get('pending_delete'):
        ticker_to_delete = st.session_state.pending_delete
        st.warning(f"⚠️ Are you sure you want to remove **{ticker_to_delete}** from your watchlist?")
        col_confirm, col_cancel, _ = st.columns([1, 1, 4])
        if col_confirm.button("✅ Confirm Delete", type="primary"):
            success, message = data_manager.remove_from_watchlist(ticker_to_delete)
            st.success(message) if success else st.error(message)
            st.session_state.pending_delete = None
            st.rerun()
        if col_cancel.button("❌ Cancel"):
            st.session_state.pending_delete = None
            st.rerun()
        st.divider()

    # ==================== AGGRID TABLE ====================
    gb = GridOptionsBuilder.from_dataframe(page_df)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_column("ticker",      header_name="Stock Code 代码",   width=150)
    gb.configure_column("stock_name",  header_name="Name 名称",          flex=1)
    gb.configure_column("added_date",  header_name="Added Date 添加日期", width=160)
    gb.configure_grid_options(rowHeight=40, suppressMovableColumns=True)
    grid_options = gb.build()

    grid_result = AgGrid(
        page_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        use_container_width=True,
        height=min(80 + len(page_df) * 40, 480),  # dynamic height
        theme="streamlit",
    )

    # ==================== ROW ACTION BUTTONS ====================
    selected_rows = grid_result.get('selected_rows', [])
    if selected_rows is not None and not selected_rows.empty:
        selected_ticker = selected_rows.iloc[0]['ticker']
        selected_name   = selected_rows.iloc[0]['stock_name']
        st.info(f"Selected: **{selected_name} ({selected_ticker})**")
        col_analyze, col_remove, _ = st.columns([1.2, 1.2, 5])
        if col_analyze.button("🔍 Analyze", type="primary", use_container_width=True):
            st.session_state.active_ticker = selected_ticker
            st.switch_page("pages/2_Single_Stock_Analysis_个股分析.py")
        if col_remove.button("🗑️ Remove", use_container_width=True):
            st.session_state.pending_delete = selected_ticker
            st.rerun()
    else:
        st.caption("💡 Click a row to select it, then Analyze or Remove")

    # ==================== PAGINATION CONTROLS ====================
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Previous", disabled=(st.session_state.watchlist_page == 0), use_container_width=True):
            st.session_state.watchlist_page -= 1
            st.rerun()
    with col2:
        st.markdown(
            f"<div style='text-align:center; padding-top:8px'>"
            f"Page <b>{st.session_state.watchlist_page + 1}</b> of <b>{total_pages}</b> "
            f"&nbsp;|&nbsp; Showing {start_idx + 1}–{end_idx} of {total_items}"
            f"</div>",
            unsafe_allow_html=True
        )
    with col3:
        if st.button("➡️ Next", disabled=(st.session_state.watchlist_page >= total_pages - 1), use_container_width=True):
            st.session_state.watchlist_page += 1
            st.rerun()
