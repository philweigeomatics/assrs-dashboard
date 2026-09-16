-- chip_scan: nightly 筹码结构 snapshot, written by scan_watchlists.py in
-- GitHub Actions and read by the Today's Alerts page, so the chip model never
-- has to run inside Streamlit.
--
-- One row per (session, ticker, decay). Per TICKER, not per user: a stock's
-- chip structure does not depend on who watches it. The page filters to the
-- logged-in user's watchlist.
--
-- ADDITIVE ONLY. Creates one table; alters nothing existing.
-- Run once in Supabase → SQL Editor.

create table if not exists public.chip_scan (
  scan_date     text             not null,
  ticker        text             not null,
  decay         double precision not null,
  name          text,
  setup_score   double precision,
  setup_label   text,
  price         double precision,
  peak_price    double precision,
  n_peaks       integer,
  winner_rate   double precision,
  concentration double precision,
  weight_avg    double precision,
  pct_from_peak double precision,
  converged     boolean,
  created_at    text,
  primary key (scan_date, ticker, decay)
);

create index if not exists idx_chip_scan_scan_date on public.chip_scan (scan_date);

-- RLS on with no policies, same as auth_user_link: only server-side code with
-- the service_role key (the nightly job, the Streamlit app, the API) touches it.
alter table public.chip_scan enable row level security;
