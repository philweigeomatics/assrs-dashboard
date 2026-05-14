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
# Keeps track of running thread objects so we can check liveness
if '_rebuild_threads' not in st.session_state:
    st.session_state['_rebuild_threads'] = {}   # {job_id: thread}


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
            st.dataframe(active[['sector', 'ticker', 'added_at']].rename(
                columns={'sector': 'Sector', 'ticker': 'Ticker', 'added_at': 'Added At'}
            ), use_container_width=True, hide_index=True)

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

            col_add, col_remove = st.columns(2)

            # ── Add a stock ─────────────────────────────────────────────────
            with col_add:
                st.markdown("**Add a stock**")
                new_ticker = st.text_input(
                    "Ticker (6-digit A-share code)",
                    placeholder="e.g. 600519",
                    key="add_ticker_input",
                ).strip()

                if st.button("Add stock", key="btn_add_stock"):
                    if not new_ticker:
                        st.error("Please enter a ticker.")
                    elif new_ticker in current_tickers:
                        st.warning(f"{new_ticker} is already in {selected_sector}.")
                    else:
                        dm.add_stock_to_sector(selected_sector, new_ticker)
                        st.success(f"Added {new_ticker} to {selected_sector}. "
                                   "Trigger a rebuild when ready.")
                        st.rerun()

            # ── Remove a stock ───────────────────────────────────────────────
            with col_remove:
                st.markdown("**Remove a stock**")
                ticker_to_remove = st.selectbox(
                    "Select stock to remove",
                    options=current_tickers,
                    key="remove_ticker_select",
                )

                if st.button("Remove stock", type="primary", key="btn_remove_stock"):
                    if len(current_tickers) <= 2:
                        st.error("A sector needs at least 2 stocks for the PPI model.")
                    else:
                        dm.remove_stock_from_sector(selected_sector, ticker_to_remove)
                        st.success(f"Removed {ticker_to_remove} from {selected_sector}. "
                                   "Trigger a rebuild when ready.")
                        st.rerun()

            st.divider()
            st.markdown(f"**Current stocks in {selected_sector}** ({len(current_tickers)} total)")
            st.write(", ".join(current_tickers))

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

    with st.form("new_sector_form"):
        sector_name = st.text_input(
            "Sector name (Chinese or English)",
            placeholder="e.g. 储能 or Energy_Storage",
        ).strip()

        tickers_raw = st.text_area(
            "Initial stock list",
            placeholder="One ticker per line, or comma-separated\ne.g.\n300750\n002594\n600886",
            height=150,
        )

        submitted = st.form_submit_button("Create sector")

    if submitted:
        if not sector_name:
            st.error("Sector name cannot be empty.")
        else:
            import re
            tickers = [t.strip() for t in re.split(r'[,\n\r]+', tickers_raw) if t.strip()]
            tickers = list(dict.fromkeys(tickers))

            existing_sectors = dm.get_sector_stock_map()
            if sector_name in existing_sectors:
                st.error(f"Sector '{sector_name}' already exists. Use the Edit tab to modify it.")
            elif len(tickers) < 2:
                st.error("Please provide at least 2 tickers.")
            else:
                # Check if market_breadth needs a new column before we start
                missing_cols = dm.get_missing_breadth_columns([sector_name])
                if missing_cols and db_config.USE_SUPABASE:
                    st.error(
                        f"**Supabase requires a manual step before this rebuild can run.**\n\n"
                        f"The `market_breadth` table needs a new column for `{sector_name}`. "
                        f"Run this in the **Supabase SQL editor** first, then come back and create the sector:"
                    )
                    st.code(
                        f'ALTER TABLE market_breadth ADD COLUMN "{sector_name}" DOUBLE PRECISION;',
                        language="sql",
                    )
                else:
                    dm.add_new_sector(sector_name, tickers)
                    st.success(f"Created sector '{sector_name}' with {len(tickers)} stocks: "
                               f"{', '.join(tickers)}")

                    job_id = dm.create_rebuild_job('sector_rebuild', [sector_name])
                    thread = start_rebuild_thread(job_id, [sector_name])
                    st.session_state['_rebuild_threads'][job_id] = thread
                    st.success(f"Rebuild job `{job_id}` started. Switch to **Rebuild Jobs** to monitor.")


# ============================================================
# TAB 4 — REBUILD JOBS
# ============================================================
with tab_jobs:
    st.subheader("Rebuild Jobs")

    # ── Trigger full rebuild ─────────────────────────────────────────────────
    with st.expander("🔴 Trigger full rebuild (all sectors)", expanded=False):
        st.warning(
            "This rebuilds PPI, market breadth, and regime scores for **every** sector "
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
