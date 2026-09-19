-- 配对搜索 cache: a two-minute, eighty-request search, kept until something
-- actually invalidates it.
--
-- There is deliberately NO expiry column. The result is a pure function of
-- the watchlist, the search parameters and the latest published session, and
-- all of those live in `cache_key` — so a time-to-live could only ever throw
-- away a still-correct answer or serve an already-wrong one. See
-- discover_cache.key_of.
--
-- One row per (user, kind). A search over a watchlist the user no longer
-- holds can never be served again, so it is overwritten rather than kept.

create table if not exists public.discover_cache (
    app_user_id  bigint not null,
    -- 'pair-trade' or 'lead-lag' — different searches over the same universe.
    kind         text   not null,
    -- JSON: every input that can change the answer, including a fingerprint
    -- of the watchlist and the date of the newest bar. Compared whole; any
    -- difference is a miss.
    cache_key    text   not null,
    payload      text   not null,
    generated_at text   not null,
    primary key (app_user_id, kind)
);

alter table public.discover_cache enable row level security;

-- Per-user rows holding one user's watchlist analysis. Reached only through
-- service_role, which bypasses RLS; anon and authenticated get nothing.
revoke all on public.discover_cache from anon, authenticated;
