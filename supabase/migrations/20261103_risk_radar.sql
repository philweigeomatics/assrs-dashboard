-- 🧭 风险雷达: the evidence behind each stock's risk map, and the AI text built on it.
--
-- risk_radar_evidence is one row per fact or story, per stock: what was seen
-- (date, source, title, link) and how it was read (which risk, which way, how
-- strong, fact or opinion). Kept rather than refetched because most sources
-- only serve their latest items — a news feed's 200 newest, not last month —
-- so the record only grows if each refresh adds to it. Scores are computed from
-- these rows on read, so changing a weight never needs a refetch.
--
-- risk_radar_cache holds per-stock AI output (the company profile and the
-- bull/bear synthesis) and the last refresh's sensitivity readings, as JSON.
-- Safe to re-run.

create table if not exists public.risk_radar_evidence (
    symbol      text not null,
    eid         text not null,
    d           date not null,
    source      text not null,
    kind        text not null,
    title       text not null,
    url         text,
    risk        text,
    dir         text,
    strength    integer,
    fact        boolean,
    note        text,
    tagged_by   text,
    fetched_at  timestamptz not null default now(),
    primary key (symbol, eid)
);

create index if not exists risk_radar_evidence_symbol_d
    on public.risk_radar_evidence (symbol, d desc);

create table if not exists public.risk_radar_cache (
    symbol       text not null,
    section      text not null,
    payload      text not null,
    generated_at timestamptz not null default now(),
    primary key (symbol, section)
);

alter table public.risk_radar_evidence enable row level security;
alter table public.risk_radar_cache enable row level security;
revoke all on public.risk_radar_evidence from anon, authenticated;
revoke all on public.risk_radar_cache from anon, authenticated;
