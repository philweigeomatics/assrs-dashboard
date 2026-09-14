-- auth_user_link: bridges a Supabase Auth login (new site) to the existing
-- app_users row (Streamlit app), so both apps share one set of watchlists,
-- search history and portfolios.
--
-- ADDITIVE ONLY. Creates one table; alters nothing Streamlit uses.
-- Run once in Supabase → SQL Editor.

create table if not exists public.auth_user_link (
  auth_user_id uuid primary key
    references auth.users (id) on delete cascade,
  app_user_id  bigint not null unique
    references public.app_users (id) on delete cascade,
  email        text not null,
  linked_at    timestamptz not null default now()
);

-- Row-level security ON with NO policies: the browser's publishable key can
-- neither read nor write this table. Only server-side code using the
-- service_role key — the API and the Streamlit app — can, and that key
-- bypasses RLS by design. A readable link table would let a signed-in user
-- enumerate which login maps to which account.
alter table public.auth_user_link enable row level security;

comment on table public.auth_user_link is
  'Supabase Auth user -> app_users.id. Written by api/auth.py on first login, '
  'only after confirming the login email is verified.';
