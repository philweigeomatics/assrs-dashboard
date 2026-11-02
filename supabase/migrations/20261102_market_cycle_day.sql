-- 💳 杠杆周期: the whole-market series its rules read, one row per session.
--
-- Raw facts, not results: 融资 balances and flows, 流通市值 and 成交额, the
-- 中证全指 close, and broad-ETF net creations. The rules (leverage_cycle.py)
-- are computed from these on read, so changing a rule never needs a refetch.
-- The daily job appends new sessions and re-reads the last few, because
-- margin data can publish a day late. Backfill from 2010 with
-- `python leverage_cycle.py`. Safe to re-run.

create table if not exists public.market_cycle_day (
    d           date primary key,
    -- 融资余额 / 融资买入额 / 融资偿还额 / 融券余额, SSE + SZSE, 亿元.
    rzye        double precision,
    rzmre       double precision,
    rzche       double precision,
    rqye        double precision,
    -- Shanghai + Shenzhen 流通市值 and 成交额, 亿元.
    float_mv    double precision,
    amount      double precision,
    -- 中证全指 (000985) close.
    idx_close   double precision,
    -- Broad-index ETFs' net creation that day, 亿元, and per fund.
    etf_flow    double precision,
    etf_detail  jsonb
);

alter table public.market_cycle_day enable row level security;
revoke all on public.market_cycle_day from anon, authenticated;
