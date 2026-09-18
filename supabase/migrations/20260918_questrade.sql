-- Questrade connection, one row per app user.
--
-- This table holds a live brokerage credential, so it is deliberately the most
-- locked-down table in the schema:
--
--   * RLS is ON and there is NO policy. That is not an oversight. With RLS
--     enabled and no policy, every role except service_role is denied by
--     default — including the anon key the browser holds. The API reads this
--     through service_role and never sends any of it to a client.
--   * Only READ permissions are requested when the app is registered in the
--     Questrade App Hub, so even a leaked refresh token cannot place an order.
--
-- `refresh_token` is the irreplaceable part: Questrade rotates it on every
-- exchange and the replacement arrives exactly once, in the response that
-- killed the previous one. questrade.py writes the new value before handing
-- the access token to anything, which is why this table has no history — an
-- older row is never useful, only misleading.

create table if not exists public.questrade_tokens (
    app_user_id   bigint      primary key,
    -- Rotates on every refresh. Losing it means the user re-generates by hand.
    refresh_token text        not null,
    -- Short-lived (30 minutes). Cached only to avoid refreshing per request,
    -- since every refresh burns the refresh token.
    access_token  text,
    -- Questrade tells you which host to use; it is not always api01.
    api_server    text,
    expires_at    double precision default 0,
    connected_at  double precision default 0,
    -- Why the last attempt failed, shown to the user so a broken chain is
    -- actionable instead of just "no data".
    last_error    text default '',
    updated_at    timestamptz not null default now()
);

alter table public.questrade_tokens enable row level security;

-- Belt and braces: revoke the API roles explicitly as well as relying on the
-- empty policy set, so a policy added later by accident cannot open this up.
revoke all on public.questrade_tokens from anon, authenticated;
