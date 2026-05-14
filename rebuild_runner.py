"""
rebuild_runner.py

Background thread that performs a full PPI → market breadth rebuild
for a set of sectors.  All progress is written to the rebuild_jobs table so
the admin UI can poll without holding an HTTP connection open.

Usage (from admin page):
    from rebuild_runner import start_rebuild_thread
    thread = start_rebuild_thread(job_id, sectors, tushare_token)
"""

import threading
import traceback
from datetime import datetime

import pandas as pd

import data_manager as dm
import api_config
import db_config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _breadth_proxy(ppi_df, date, lookback=20):
    """PPI vs MA20 breadth proxy, same logic as main.py."""
    try:
        df = ppi_df[ppi_df.index <= date].copy()
        if len(df) < lookback:
            return 0.5
        df['_ma'] = df['Close'].rolling(window=lookback).mean()
        if date not in df.index:
            return 0.5
        close = df.loc[date, 'Close']
        ma = df.loc[date, '_ma']
        if pd.isna(close) or pd.isna(ma) or ma == 0:
            return 0.5
        return max(0.0, min(1.0, ((close - ma) / ma + 0.05) / 0.10))
    except Exception:
        return 0.5


def _log(job_id, msg):
    print(f"[rebuild:{job_id}] {msg}")


# ---------------------------------------------------------------------------
# Main rebuild routine (runs in a thread)
# ---------------------------------------------------------------------------

def run_full_rebuild(job_id, sectors_to_rebuild, tushare_token):
    """
    Full rebuild pipeline for the given sectors.

    sectors_to_rebuild: list of sector names, or '__all__' for every sector.
    Writes progress (0-100) and status to rebuild_jobs table throughout.
    """
    now_str = lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def progress(pct, msg):
        _log(job_id, f"[{pct}%] {msg}")
        dm.update_rebuild_job(job_id, progress=pct, progress_message=msg)

    try:
        dm.update_rebuild_job(job_id, status='running',
                              started_at=now_str(), progress=0,
                              progress_message='Starting...')

        # ── Step 1: Tushare ─────────────────────────────────────────────────
        progress(2, 'Connecting to Tushare API...')
        if not dm.init_tushare(tushare_token):
            raise RuntimeError("Tushare initialisation failed — check token")

        # ── Step 2: Resolve sector list ──────────────────────────────────────
        progress(4, 'Loading sector composition from database...')
        sector_map = dm.get_sector_stock_map()

        if sectors_to_rebuild == '__all__':
            rebuild_sectors = list(sector_map.keys())
        else:
            rebuild_sectors = [s for s in sectors_to_rebuild if s in sector_map]
            if not rebuild_sectors:
                raise RuntimeError(f"None of the requested sectors found in DB: {sectors_to_rebuild}")

        _log(job_id, f"Rebuilding {len(rebuild_sectors)} sectors: {rebuild_sectors}")

        # ── Step 2b: Supabase pre-flight — all required tables must exist ────
        if db_config.USE_SUPABASE:
            missing_ppi = dm.get_missing_ppi_tables(rebuild_sectors)
            missing_breadth = dm.get_missing_breadth_columns(rebuild_sectors)
            if missing_ppi or missing_breadth:
                sql_parts = []
                for s in missing_ppi:
                    sql_parts.append(
                        f'CREATE TABLE IF NOT EXISTS "PPI_{s}" ('
                        f'"Date" TEXT PRIMARY KEY, "Open" REAL, "High" REAL, '
                        f'"Low" REAL, "Close" REAL, "Norm_Vol_Metric" REAL);'
                    )
                for s in missing_breadth:
                    sql_parts.append(
                        f'ALTER TABLE market_breadth ADD COLUMN "{s}" DOUBLE PRECISION;'
                    )
                raise RuntimeError(
                    f"Supabase is missing tables/columns for: "
                    f"PPI={missing_ppi}, breadth={missing_breadth}.\n"
                    f"Run in the Supabase SQL editor first:\n\n" + "\n".join(sql_parts)
                )

        # ── Step 3: Wipe existing PPI tables for these sectors ───────────────
        progress(6, f'Clearing old PPI data for {len(rebuild_sectors)} sectors...')
        for sector in rebuild_sectors:
            table = f"PPI_{sector}"
            if dm.db.table_exists(table):
                dm.db.delete_all_records(table)

        # ── Step 4: Full PPI rebuild (one sector at a time for progress) ─────
        progress(8, f'Rebuilding PPIs (0/{len(rebuild_sectors)})...')
        all_ppi = {}
        n = len(rebuild_sectors)

        for i, sector in enumerate(rebuild_sectors):
            pct = 8 + int((i / n) * 52)   # 8 → 60 %
            progress(pct, f'Rebuilding PPI [{i+1}/{n}]: {sector}')

            single_ppi = dm.aggregate_ppi_data(
                sector_start_dates={sector: None},      # None = full rebuild
                sector_stock_map={sector: sector_map[sector]},
            )
            if single_ppi:
                dm.save_ppi_data_to_db(single_ppi)
                all_ppi.update(single_ppi)
                _log(job_id, f"  ✅ PPI done: {sector}")
            else:
                _log(job_id, f"  ⚠️  PPI failed: {sector} (skipped)")

        # ── Step 5: Load all PPIs from DB (need unrebuilt sectors too) ───────
        progress(62, 'Loading all PPIs from database...')
        all_ppi_db = dm.load_ppi_data_from_db()
        if not all_ppi_db:
            raise RuntimeError("No PPIs found in database after rebuild")

        # ── Step 6: Rebuild market breadth for the rebuilt sectors ───────────
        progress(64, 'Rebuilding market breadth...')

        all_dates: set = set()
        for s in rebuild_sectors:
            if s in all_ppi_db:
                all_dates.update(all_ppi_db[s].index)

        date_range = pd.DatetimeIndex(sorted(all_dates))
        breadth_by_date: dict = {}
        total_d = len(date_range)

        for di, date in enumerate(date_range):
            if di % 25 == 0:
                pct = 64 + int((di / total_d) * 14)   # 64 → 78 %
                progress(pct, f'Breadth [{di}/{total_d} dates]...')

            row: dict = {}
            for s in rebuild_sectors:
                if s in all_ppi_db:
                    row[s] = _breadth_proxy(all_ppi_db[s], date)
            if row:
                breadth_by_date[date] = row

        if breadth_by_date:
            try:
                dm.save_market_breadth_to_db(breadth_by_date)
            except RuntimeError as e:
                # Missing Supabase column — surface the SQL and abort
                raise RuntimeError(str(e))
        progress(99, 'Market breadth saved.')

        # ── Done ─────────────────────────────────────────────────────────────
        dm.update_rebuild_job(job_id, status='completed', progress=100,
                              progress_message='Rebuild complete ✅',
                              completed_at=now_str())
        _log(job_id, "Rebuild completed successfully.")

    except Exception as exc:
        msg = str(exc)
        _log(job_id, f"FAILED: {msg}")
        traceback.print_exc()
        dm.update_rebuild_job(job_id, status='failed',
                              progress_message=f'Failed: {msg}',
                              error_message=msg,
                              completed_at=now_str())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def start_rebuild_thread(job_id, sectors, tushare_token=None):
    """
    Kick off a rebuild in a daemon thread.  Returns the thread object.

    The thread writes all progress to rebuild_jobs[job_id] in the DB;
    callers should poll that row rather than joining the thread.
    """
    if tushare_token is None:
        tushare_token = api_config.TUSHARE_TOKEN

    t = threading.Thread(
        target=run_full_rebuild,
        args=(job_id, sectors, tushare_token),
        daemon=True,
        name=f"rebuild_{job_id}",
    )
    t.start()
    return t
