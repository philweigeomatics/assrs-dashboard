-- 🌍 宏观数据 → 指数: US and China economic releases, as eco_cal lists them.
--
-- Kept because eco_cal cannot be asked for one series' history: its event
-- filter wants an exact name, and every month's name is different ("(九月)").
-- Rows are stored as given; a release listed twice (scheduled with no value,
-- then with it) is resolved when read — see eco_impact.dedupe. The reactions
-- are computed on read from index prices, so changing the rule never needs a
-- refetch.
--
-- Backfill with `python eco_impact.py backfill 2021-11-01`; the daily job
-- runs `python eco_impact.py` to re-read the last two weeks and the next ten
-- days. Safe to re-run.

create table if not exists public.eco_release (
    -- 美国 / 中国
    country   text        not null,
    -- The event exactly as eco_cal names it, period included.
    event     text        not null,
    -- The event with its period suffix removed: what a history is read by.
    series    text        not null,
    -- The release instant (eco_cal gives Beijing time; stored as an instant).
    at        timestamptz not null,
    -- Actual, forecast and previous, as strings with their units ("4.9%").
    value     text,
    forecast  text,
    prev      text,
    primary key (country, at, event)
);

create index if not exists eco_release_series_idx
    on public.eco_release (country, series, at);

alter table public.eco_release enable row level security;
revoke all on public.eco_release from anon, authenticated;
