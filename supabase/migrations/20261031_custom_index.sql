-- 自建指数: an index you define and watch. Private to its owner.
--
-- The level is never stored. It is a pure function of these rows and market
-- prices (custom_index.py), so an edit to a member's dates rebuilds the
-- history correctly instead of leaving a stored series that disagrees with
-- its own definition.

create table if not exists public.custom_index (
    id               bigserial primary key,
    app_user_id      bigint      not null,
    name             text        not null,
    -- One market per index: one currency, one calendar, one benchmark.
    market           text        not null check (market in ('CN', 'US')),
    -- equal | equal_hold | cap | cap_capped | price | custom | inv_vol
    method           text        not null,
    -- M | Q | H | Y | never
    rebalance        text        not null,
    -- Single-name cap in percent, for cap_capped only.
    cap_pct          real,
    base_date        date        not null,
    base_value       real        not null default 1000,
    -- What the index is compared with: a US industry ETF or an A-share theme
    -- index from the dashboard board. Null = the broad market.
    bench            text,
    created_at       timestamptz not null default now(),
    -- The last time the definition changed. History before it was computed
    -- with hindsight and is drawn as 回溯（假设）.
    rules_updated_at timestamptz not null default now(),
    unique (app_user_id, name)
);

create table if not exists public.custom_index_member (
    id            bigserial primary key,
    index_id      bigint not null references public.custom_index(id) on delete cascade,
    -- Bare six digits for A-shares, US:TICKER for US.
    ticker        text   not null,
    name          text,
    -- For the custom method only.
    weight_pct    real,
    -- In from the close of added_on; out at the close of removed_on.
    added_on      date   not null,
    removed_on    date,
    -- 'user' or 'delisted'.
    removal       text,
    exit_price    real,
    -- JSON {date: adjusted close}, written once when a member stops trading,
    -- because Yahoo later drops an acquired company's history entirely.
    frozen_closes text
);

-- Safe to re-run: a database created from the first version of this file
-- gains the comparison column without touching anything else.
alter table public.custom_index add column if not exists bench text;

create index if not exists custom_index_member_index_id on public.custom_index_member (index_id);
create index if not exists custom_index_app_user_id on public.custom_index (app_user_id);

alter table public.custom_index enable row level security;
alter table public.custom_index_member enable row level security;

-- Reached only through the API with service_role, which bypasses RLS and
-- scopes every query to the caller; anon and authenticated get nothing.
revoke all on public.custom_index from anon, authenticated;
revoke all on public.custom_index_member from anon, authenticated;
