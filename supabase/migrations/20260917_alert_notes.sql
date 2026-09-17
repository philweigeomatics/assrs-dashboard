-- Prediction notes on Today's Alerts.
--
-- One row per note: free text plus a list of machine-checkable claims about
-- the NEXT trading session. `outcome` stays NULL until a bar dated strictly
-- after `scan_date` exists, which is what keeps a Friday note pending over the
-- weekend without any calendar logic.
--
-- The point of storing the claims rather than only the prose is that they can
-- be marked by something other than memory. Hindsight rewrites a vague
-- prediction into whatever happened; a stored one cannot be rewritten.

create table if not exists public.alert_notes (
    id            bigserial primary key,
    user_id       bigint      not null,
    ticker        text        not null,
    -- The trading session the note was written AGAINST, not the wall clock —
    -- a note typed on Saturday is still a call on Friday's close.
    scan_date     date        not null,
    note          text,
    -- Normalised claims, validated by alert_notes.normalise before insert.
    predictions   jsonb       not null default '[]'::jsonb,
    created_at    timestamptz not null default now(),
    resolved_date date,
    resolved_at   timestamptz,
    -- {bar, claims[], hits, decided, score_pct}
    outcome       jsonb
);

-- The two queries the app makes: everything for a user, and the pending ones.
create index if not exists alert_notes_user_idx
    on public.alert_notes (user_id, scan_date desc);
create index if not exists alert_notes_pending_idx
    on public.alert_notes (user_id) where outcome is null;

alter table public.alert_notes enable row level security;

-- The API reaches this through the service_role key, which bypasses RLS; the
-- policy is what stops one user reading another's notes if the table is ever
-- exposed through the anon key.
drop policy if exists alert_notes_owner on public.alert_notes;
create policy alert_notes_owner on public.alert_notes
    for all
    using (user_id = (current_setting('request.jwt.claims', true)::jsonb ->> 'app_user_id')::bigint)
    with check (user_id = (current_setting('request.jwt.claims', true)::jsonb ->> 'app_user_id')::bigint);
