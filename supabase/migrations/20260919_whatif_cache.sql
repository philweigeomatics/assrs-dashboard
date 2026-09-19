-- 尾盘推演 cache: the AI read of a stock's latest real session.
--
-- One row per ticker, not one per (ticker, day). The read is a pure function
-- of the bar it describes, so it is valid for exactly as long as that bar is
-- the newest one — and a read for a bar three weeks gone can never be served
-- again. Keeping it would be dead weight that grows by one row per stock per
-- trading day forever.
--
-- `bar_date` is therefore not part of the key: it is the VALIDITY CHECK. A row
-- whose bar_date no longer matches the latest session is a miss, and the next
-- generation overwrites it.
--
-- Not cached here: the simulated ("ghost") read. Its inputs are a continuous
-- slider, so the same hypothetical is almost never asked for twice.

create table if not exists public.whatif_cache (
    ticker       text primary key,
    -- The trading session the read describes, e.g. '2026-09-18'.
    bar_date     text        not null,
    -- The 吸筹/出货 window the read was built with. A different window is a
    -- different brief and therefore a different answer, so it invalidates.
    ad_window    integer     not null default 20,
    -- {mode, read, crossings, bar_date} as returned by the API.
    payload      text        not null,
    generated_at text        not null
);

-- The only query this table serves.
create index if not exists whatif_cache_bar_idx
    on public.whatif_cache (ticker, bar_date);

alter table public.whatif_cache enable row level security;

-- Read-through cache of a model call about PUBLIC market data — no user owns
-- a row and nothing here is personal, which is why there is no user_id. The
-- API reaches it through service_role; anon and authenticated get nothing.
revoke all on public.whatif_cache from anon, authenticated;
