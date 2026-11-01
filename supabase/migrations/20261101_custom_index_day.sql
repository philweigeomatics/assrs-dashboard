-- 自建指数: every index's value, stored day by day.
--
-- Computed once when an index is made (its history from the start date) and
-- then one row per trading day by the daily job, continuing from the last
-- stored close. A stored day is not recomputed: a rebalance or a membership
-- change only ever affects the days after it, as at an index provider.
--
-- Safe to re-run.

create table if not exists public.custom_index_day (
    index_id  bigint not null references public.custom_index(id) on delete cascade,
    d         date   not null,
    -- The index at the open: yesterday's holdings at this morning's opening
    -- prices. With the close it splits each day's move into overnight (the
    -- gap) and the session. Null on the start day, or when a member that
    -- traded has no opening price.
    open      double precision,
    close     double precision not null,
    -- Turnover of the members held through the session, in the market's
    -- currency (yuan or dollars). Summed, as index providers publish it.
    amount    double precision,
    -- At this close: each holding's value in index points ("v"), its price
    -- ("p"), its running contribution ("c"), cash, halted names ("h"),
    -- names whose weight input had to be guessed ("g"); on the latest day
    -- also the next rebalance's targets ("t"). Values, not share counts:
    -- a dividend rescales every adjusted price before it, a value does not.
    state     jsonb,
    -- What happened at this close: start, add, remove, rebalance.
    events    jsonb,
    primary key (index_id, d)
);

-- The same rules replayed from up to ten years before the start date and
-- joined to the index on it, for the dashboard's 200-day and stretch
-- readings: {"from": date, "dates": [...], "close": [...]}. Hypothetical by
-- construction, so kept apart from the index's own record above.
alter table public.custom_index add column if not exists backtest jsonb;

alter table public.custom_index_day enable row level security;
revoke all on public.custom_index_day from anon, authenticated;
