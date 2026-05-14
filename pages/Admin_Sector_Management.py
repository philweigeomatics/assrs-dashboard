"""
Admin: Sector Management

Lets admins:
  1. View current sector composition
  2. Add / remove stocks from existing sectors
  3. Create new sectors with an initial stock list
  4. Trigger a full PPI → breadth → regime-score rebuild (background thread)
  5. Monitor rebuild job progress in real-time
"""

import time
import json

import streamlit as st
import pandas as pd

import auth_manager
import data_manager as dm
import db_config
import api_config
from rebuild_runner import start_rebuild_thread

st.set_page_config(
    page_title="⚙️ Sector Management | Admin",
    page_icon="⚙️",
    layout="wide",
)

auth_manager.require_admin()

st.title("⚙️ Sector Management")
st.caption("Admin only — changes here affect PPI calculations and all downstream analysis.")

# ── ensure tables exist on every load ───────────────────────────────────────
dm.ensure_admin_tables_exist()
dm.seed_sector_stock_map()

# ── thread registry (per Streamlit session) ─────────────────────────────────
if '_rebuild_threads' not in st.session_state:
    st.session_state['_rebuild_threads'] = {}


# ── Stock lookup helpers (used by multiple tabs) ─────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _all_stock_options():
    """All stocks from stock_basic as 'CODE · 名称' strings for selectbox."""
    stocks = dm.get_all_stock_basic()
    return [""] + [f"{s['ticker']} · {s['name']}" for s in stocks]

@st.cache_data(ttl=3600, show_spinner=False)
def _build_name_map():
    """Dict of {6-digit ticker: company name} from stock_basic."""
    stocks = dm.get_all_stock_basic()
    return {s['ticker']: s['name'] for s in stocks}

def _label(ticker, name_map):
    name = name_map.get(ticker, "")
    return f"{ticker} · {name}" if name else ticker


# ============================================================
# TABS
# ============================================================
tab_overview, tab_edit, tab_new_sector, tab_jobs = st.tabs([
    "📋 Sector Overview",
    "✏️ Edit Sector",
    "➕ New Sector",
    "🔄 Rebuild Jobs",
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab_overview:
    st.subheader("Current Sector Composition")

    raw = dm.get_all_sector_stock_map_raw()
    if raw.empty:
        st.info("No sector data in database yet. It will be seeded from the hardcoded map automatically.")
    else:
        active = raw[raw['is_active'] == 1]
        inactive = raw[raw['is_active'] == 0]

        sector_map = dm.get_sector_stock_map()
        summary_rows = []
        for sector, tickers in sector_map.items():
            summary_rows.append({'Sector': sector, 'Active Stocks': len(tickers),
                                  'Tickers': ', '.join(tickers[:5]) + ('…' if len(tickers) > 5 else '')})
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with st.expander("Show full composition table"):
            name_map_ov = _build_name_map()
            full = active[['sector', 'ticker', 'added_at']].copy()
            full.insert(2, 'name', full['ticker'].map(lambda t: name_map_ov.get(t, '—')))
            st.dataframe(full.rename(columns={'sector': 'Sector', 'ticker': 'Ticker',
                                              'name': 'Name', 'added_at': 'Added At'}),
                         use_container_width=True, hide_index=True)

        if not inactive.empty:
            with st.expander(f"Removed stocks ({len(inactive)})"):
                st.dataframe(inactive[['sector', 'ticker', 'removed_at']].rename(
                    columns={'sector': 'Sector', 'ticker': 'Ticker', 'removed_at': 'Removed At'}
                ), use_container_width=True, hide_index=True)


# ============================================================
# TAB 2 — EDIT SECTOR
# ============================================================
with tab_edit:
    st.subheader("Edit Existing Sector")
    st.info("After saving changes, go to **Rebuild Jobs** to trigger a full PPI rebuild.")

    sector_map = dm.get_sector_stock_map()
    if not sector_map:
        st.warning("No sectors found. Use 'New Sector' tab to create one.")
    else:
        selected_sector = st.selectbox("Select sector", sorted(sector_map.keys()),
                                       key="edit_sector_select")

        if selected_sector:
            current_tickers = sector_map[selected_sector]
            name_map = _build_name_map()

            col_add, col_remove = st.columns(2)

            # ── Add a stock ─────────────────────────────────────────────────
            with col_add:
                st.markdown("**Add a stock**")
                all_opts = _all_stock_options()
                picked = st.selectbox(
                    "Search by code or name",
                    options=all_opts,
                    key="add_ticker_select",
                    format_func=lambda x: "Type to search… (code or name)" if x == "" else x,
                )
                new_ticker = picked.split(" · ")[0].strip() if picked else ""

                if st.button("Add stock", key="btn_add_stock"):
                    if not new_ticker:
                        st.error("Please select a stock.")
                    elif new_ticker in current_tickers:
                        st.warning(f"{_label(new_ticker, name_map)} is already in {selected_sector}.")
                    else:
                        dm.add_stock_to_sector(selected_sector, new_ticker)
                        st.success(f"Added {_label(new_ticker, name_map)} to {selected_sector}. "
                                   "Trigger a rebuild when ready.")
                        st.rerun()

            # ── Remove a stock ───────────────────────────────────────────────
            with col_remove:
                st.markdown("**Remove a stock**")
                remove_opts = [f"{t} · {name_map[t]}" if t in name_map else t
                               for t in current_tickers]
                remove_pick = st.selectbox(
                    "Select stock to remove",
                    options=remove_opts,
                    key="remove_ticker_select",
                )
                ticker_to_remove = remove_pick.split(" · ")[0].strip() if remove_pick else ""

                if st.button("Remove stock", type="primary", key="btn_remove_stock"):
                    if len(current_tickers) <= 2:
                        st.error("A sector needs at least 2 stocks for the PPI model.")
                    else:
                        dm.remove_stock_from_sector(selected_sector, ticker_to_remove)
                        st.success(f"Removed {_label(ticker_to_remove, name_map)} from {selected_sector}. "
                                   "Trigger a rebuild when ready.")
                        st.rerun()

            st.divider()

            # ── Current stock list with names ────────────────────────────────
            st.markdown(f"**Current stocks in {selected_sector}** ({len(current_tickers)} total)")
            stock_rows = [{"Ticker": t, "Name": name_map.get(t, "—")} for t in current_tickers]
            st.dataframe(pd.DataFrame(stock_rows), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("**Trigger rebuild for this sector only**")
            if st.button(f"Rebuild '{selected_sector}' now", key="btn_rebuild_single"):
                job_id = dm.create_rebuild_job('sector_rebuild', [selected_sector])
                thread = start_rebuild_thread(job_id, [selected_sector])
                st.session_state['_rebuild_threads'][job_id] = thread
                st.success(f"Rebuild job `{job_id}` started. "
                           "Switch to the **Rebuild Jobs** tab to monitor progress.")


# ============================================================
# TAB 3 — NEW SECTOR
# ============================================================
with tab_new_sector:
    st.subheader("Create a New Sector")
    st.info("After creating the sector, a rebuild job starts automatically.")

    if '_new_sector_stocks' not in st.session_state:
        st.session_state['_new_sector_stocks'] = []

    new_sector_name = st.text_input(
        "Sector name (Chinese or English)",
        placeholder="e.g. 储能 or Energy_Storage",
        key="new_sector_name_input",
    ).strip()

    st.markdown("**Add stocks to this sector**")
    col_pick, col_add_btn = st.columns([4, 1])
    with col_pick:
        all_opts_ns = _all_stock_options()
        ns_picked = st.selectbox(
            "Search by code or name",
            options=all_opts_ns,
            key="new_sector_stock_pick",
            format_func=lambda x: "Type to search… (code or name)" if x == "" else x,
        )
        ns_ticker = ns_picked.split(" · ")[0].strip() if ns_picked else ""
    with col_add_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add", key="btn_ns_add"):
            if not ns_ticker:
                st.warning("Select a stock first.")
            elif ns_ticker in st.session_state['_new_sector_stocks']:
                st.warning(f"{ns_ticker} already in list.")
            else:
                st.session_state['_new_sector_stocks'].append(ns_ticker)
                st.rerun()

    # Display current stock list
    stocks = st.session_state['_new_sector_stocks']
    if stocks:
        name_map_ns = _build_name_map()
        st.markdown(f"**Stock list** ({len(stocks)} stocks)")
        for i, t in enumerate(stocks):
            col_t, col_del = st.columns([5, 1])
            with col_t:
                st.text(_label(t, name_map_ns))
            with col_del:
                if st.button("✕", key=f"ns_del_{i}"):
                    st.session_state['_new_sector_stocks'].pop(i)
                    st.rerun()
    else:
        st.caption("No stocks added yet.")

    st.divider()

    col_create, col_clear = st.columns([3, 1])
    with col_create:
        if st.button("Create sector", type="primary", key="btn_create_sector"):
            if not new_sector_name:
                st.error("Sector name cannot be empty.")
            elif len(stocks) < 2:
                st.error("Please add at least 2 stocks.")
            else:
                existing_sectors = dm.get_sector_stock_map()
                if new_sector_name in existing_sectors:
                    st.error(f"Sector '{new_sector_name}' already exists. Use the Edit tab to modify it.")
                else:
                    missing_ppi = dm.get_missing_ppi_tables([new_sector_name])
                    missing_breadth = dm.get_missing_breadth_columns([new_sector_name])
                    if (missing_ppi or missing_breadth) and db_config.USE_SUPABASE:
                        st.error(
                            "**Supabase requires manual steps before this sector can be created.**\n\n"
                            "Run the SQL below in the **Supabase SQL editor**, then come back and click Create again."
                        )
                        sql_parts = []
                        if missing_ppi:
                            sql_parts.append(
                                f'CREATE TABLE IF NOT EXISTS "PPI_{new_sector_name}" ('
                                f'"Date" TEXT PRIMARY KEY, "Open" REAL, "High" REAL, '
                                f'"Low" REAL, "Close" REAL, "Norm_Vol_Metric" REAL);'
                            )
                        if missing_breadth:
                            sql_parts.append(
                                f'ALTER TABLE market_breadth ADD COLUMN "{new_sector_name}" DOUBLE PRECISION;'
                            )
                        st.code("\n".join(sql_parts), language="sql")
                    else:
                        dm.add_new_sector(new_sector_name, stocks)
                        st.success(f"Created sector '{new_sector_name}' with {len(stocks)} stocks.")
                        job_id = dm.create_rebuild_job('sector_rebuild', [new_sector_name])
                        thread = start_rebuild_thread(job_id, [new_sector_name])
                        st.session_state['_rebuild_threads'][job_id] = thread
                        st.session_state['_new_sector_stocks'] = []
                        st.success(f"Rebuild job `{job_id}` started. Switch to **Rebuild Jobs** to monitor.")
    with col_clear:
        if st.button("Clear list", key="btn_ns_clear"):
            st.session_state['_new_sector_stocks'] = []
            st.rerun()


# ============================================================
# TAB 4 — REBUILD JOBS
# ============================================================
with tab_jobs:
    st.subheader("Rebuild Jobs")

    # ── Trigger full rebuild ─────────────────────────────────────────────────
    with st.expander("🔴 Trigger full rebuild (all sectors)", expanded=False):
        st.warning(
            "This rebuilds PPI and market breadth for **every** sector "
            "from the data start date. It runs in the background and takes 20–60 minutes "
            "depending on API rate limits. The dashboard remains live during the rebuild."
        )
        if st.button("Start full rebuild", type="primary", key="btn_full_rebuild"):
            job_id = dm.create_rebuild_job('full_rebuild', '__all__')
            thread = start_rebuild_thread(job_id, '__all__')
            st.session_state['_rebuild_threads'][job_id] = thread
            st.success(f"Full rebuild job `{job_id}` started. Monitor below.")

    st.divider()

    # ── Job list and live progress ───────────────────────────────────────────
    jobs_df = dm.get_recent_rebuild_jobs(limit=15)

    if jobs_df.empty:
        st.info("No rebuild jobs yet.")
    else:
        # Check if any job is still running
        any_running = False

        for _, job_row in jobs_df.iterrows():
            job_id = job_row['job_id']
            status = job_row.get('status', 'unknown')
            pct = int(job_row.get('progress', 0))
            msg = job_row.get('progress_message', '')
            created = job_row.get('created_at', '')
            sectors_raw = job_row.get('sectors', '[]')
            try:
                sectors_label = ', '.join(json.loads(sectors_raw)) if sectors_raw != '__all__' else 'ALL'
            except Exception:
                sectors_label = sectors_raw

            with st.container(border=True):
                header_col, status_col = st.columns([3, 1])
                with header_col:
                    st.markdown(f"**Job `{job_id}`** — {job_row.get('job_type', '')}  "
                                f"| {sectors_label}  | Created: {created}")
                with status_col:
                    if status == 'completed':
                        st.success("Completed")
                    elif status == 'failed':
                        st.error("Failed")
                    elif status == 'running':
                        st.warning("Running")
                        any_running = True
                    else:
                        st.info("Pending")

                if status in ('running', 'pending'):
                    st.progress(pct / 100, text=f"{pct}% — {msg}")
                elif status == 'completed':
                    st.markdown(f"✅ {msg}  |  Finished: {job_row.get('completed_at', '')}")
                elif status == 'failed':
                    st.error(f"Error: {job_row.get('error_message', 'unknown error')}")

        # Auto-refresh every 4 seconds while a job is running
        if any_running:
            st.caption("Auto-refreshing every 4 seconds while job is running…")
            time.sleep(4)
            st.rerun()
        else:
            if st.button("Refresh job list", key="btn_refresh_jobs"):
                st.rerun()
