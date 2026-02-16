"""
Watchlist Management - 观察列表管理
Admin-only page to manage your stock watchlist
需要密码访问的管理页面
"""

import streamlit as st
import pandas as pd
import data_manager

# ==================== PASSWORD PROTECTION ====================
# Simple password check (replace with your own password)
ADMIN_PASSWORD = "000000"  # Change this!

def check_password():
    """Returns True if the user entered the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == ADMIN_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # First run or password not correct yet
    if "password_correct" not in st.session_state:
        # Show password input
        st.text_input(
            "🔒 Enter Password | 输入密码",
            type="password",
            on_change=password_entered,
            key="password",
            placeholder="Admin password required"
        )
        st.info("💡 This page is admin-only. Enter password to access.")
        return False
    
    # Password incorrect
    elif not st.session_state["password_correct"]:
        st.text_input(
            "🔒 Enter Password | 输入密码",
            type="password",
            on_change=password_entered,
            key="password",
            placeholder="Admin password required"
        )
        st.error("❌ Incorrect password. Please try again.")
        return False
    
    # Password correct
    else:
        return True

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Watchlist | 观察列表",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Watchlist Management | 观察列表管理")

# Check password first
if not check_password():
    st.stop()

# Initialize tables
data_manager.create_watchlist_table()

# ==================== MAIN PAGE ====================

# Add logout button
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🚪 Logout"):
        st.session_state["password_correct"] = False
        st.rerun()

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
            
            # Validate ticker format (6 digits)
            if not (len(ticker_clean) == 6 and ticker_clean.isdigit()):
                st.error("❌ Invalid ticker format. Must be 6 digits (e.g., 600519)")
            else:
                # Add to watchlist
                success, message = data_manager.add_to_watchlist(ticker_clean)
                
                if success:
                    st.success(message)
                    st.rerun()  # Just rerun, don't manually clear
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
                # Parse tickers
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
# ==================== WATCHLIST DISPLAY ====================
st.subheader("📋 Current Watchlist")

watchlist_data = data_manager.get_watchlist()

if not watchlist_data:
    st.info("📭 Your watchlist is empty. Add stocks above to get started.")
else:
    # Display total count
    st.write(f"**Total: {len(watchlist_data)} stocks**")
    
    # Create DataFrame
    df = pd.DataFrame(watchlist_data)
    df = df[['ticker', 'stock_name', 'added_date']]
    
    # Sort by added_date descending (newest first)
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
    
    # Filter dataframe based on search
    if search_term:
        search_lower = search_term.lower()
        filtered_df = df[
            df['ticker'].str.contains(search_term, case=False, na=False) |
            df['stock_name'].str.contains(search_term, case=False, na=False)
        ]
        
        if filtered_df.empty:
            st.warning(f"⚠️ No stocks found matching '{search_term}'")
            filtered_df = df  # Show all if no results
        else:
            st.info(f"📊 Found {len(filtered_df)} stock(s) matching '{search_term}'")
    else:
        filtered_df = df
    
    # ==================== PAGINATION ====================
    ITEMS_PER_PAGE = 10
    total_items = len(filtered_df)
    total_pages = (total_items - 1) // ITEMS_PER_PAGE + 1 if total_items > 0 else 1
    
    # Initialize page number in session state
    if 'watchlist_page' not in st.session_state:
        st.session_state.watchlist_page = 0
    
    # Reset to page 0 if search changed
    if 'last_search' not in st.session_state:
        st.session_state.last_search = ""
    
    if search_term != st.session_state.last_search:
        st.session_state.watchlist_page = 0
        st.session_state.last_search = search_term
    
    # Ensure page number is within bounds
    if st.session_state.watchlist_page >= total_pages:
        st.session_state.watchlist_page = total_pages - 1
    if st.session_state.watchlist_page < 0:
        st.session_state.watchlist_page = 0
    
    # Calculate slice indices
    start_idx = st.session_state.watchlist_page * ITEMS_PER_PAGE
    end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)
    
    # Get current page data
    page_df = filtered_df.iloc[start_idx:end_idx].copy()
    
    # Add Remove checkbox column
    page_df.insert(0, 'Remove', False)
    
    # ==================== DISPLAY TABLE ====================
    edited_df = st.data_editor(
        page_df,
        column_config={
            "Remove": st.column_config.CheckboxColumn(
                "🗑️ Remove",
                help="Check to remove from watchlist",
                default=False,
                width="small"
            ),
            "ticker": st.column_config.TextColumn(
                "Stock Code 代码",
                disabled=True,
                width="medium"
            ),
            "stock_name": st.column_config.TextColumn(
                "Name 名称",
                disabled=True,
                width="large"
            ),
            "added_date": st.column_config.TextColumn(
                "Added Date 添加日期",
                disabled=True,
                width="medium"
            ),
        },
        hide_index=True,
        use_container_width=True,
        disabled=["ticker", "stock_name", "added_date"],
        key=f"watchlist_editor_page_{st.session_state.watchlist_page}"
    )
    
    # ==================== PAGINATION CONTROLS ====================
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous", disabled=(st.session_state.watchlist_page == 0), use_container_width=True):
            st.session_state.watchlist_page -= 1
            st.rerun()
    
    with col2:
        st.markdown(f"<div style='text-align: center; padding: 10px;'><b>Page {st.session_state.watchlist_page + 1} of {total_pages}</b><br/>Showing {start_idx + 1}-{end_idx} of {total_items} stocks</div>", unsafe_allow_html=True)
    
    with col3:
        if st.button("Next ➡️", disabled=(st.session_state.watchlist_page >= total_pages - 1), use_container_width=True):
            st.session_state.watchlist_page += 1
            st.rerun()
    
    st.markdown("---")
    
    # ==================== REMOVE BUTTON ====================
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Remove Selected", type="secondary", use_container_width=True):
            # Get tickers marked for removal
            to_remove = edited_df[edited_df['Remove'] == True]['ticker'].tolist()
            
            if not to_remove:
                st.warning("⚠️ No stocks selected for removal")
            else:
                # Remove each selected stock
                removed_count = 0
                failed_count = 0
                
                for ticker in to_remove:
                    success, msg = data_manager.remove_from_watchlist(ticker)
                    if success:
                        removed_count += 1
                    else:
                        failed_count += 1
                        st.error(f"Failed to remove {ticker}: {msg}")
                
                if removed_count > 0:
                    st.success(f"✅ Removed {removed_count} stock(s) from watchlist")
                    st.rerun()
                
                if failed_count > 0:
                    st.error(f"❌ Failed to remove {failed_count} stock(s)")
    
    with col2:
        st.caption("💡 Check the boxes next to stocks you want to remove, then click 'Remove Selected'")
    
    st.markdown("---")
    
    # ==================== EXPORT ====================
    st.subheader("📥 Export Watchlist")
    col1, col2 = st.columns([1, 4])
    
    with col1:
        # Export filtered results or all
        export_df = filtered_df if search_term else df
        csv = export_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"watchlist_{search_term if search_term else 'all'}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if search_term:
            st.caption(f"Download filtered results ({len(filtered_df)} stocks)")
        else:
            st.caption(f"Download complete watchlist ({len(df)} stocks)")
