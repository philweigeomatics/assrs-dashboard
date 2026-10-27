-- 🚦 市场风险 · 涨停家数: one row per session, derived from `daily` against
-- `stk_limit` (limit_list_d is outside the current Tushare tier).
--
-- Two whole-market calls per session and ~500 sessions since 2024-10-14 is
-- too much to repeat on every cold start, so each session is computed once
-- and kept. A past session's count never changes; there is no expiry.
-- An unpublished session is never written, so a gap here always means
-- "not fetched yet", never "zero".

create table if not exists public.market_limit_stats (
    trade_date  text    primary key,    -- YYYYMMDD
    up_count    integer not null,       -- closed at the limit-up price
    down_count  integer not null,       -- closed at the limit-down price
    touched     integer not null,       -- reached limit-up intraday
    total       integer not null        -- stocks with a limit that day
);

alter table public.market_limit_stats enable row level security;

-- Market-wide numbers, but written and read only by the API through
-- service_role, which bypasses RLS; anon and authenticated get nothing.
revoke all on public.market_limit_stats from anon, authenticated;
