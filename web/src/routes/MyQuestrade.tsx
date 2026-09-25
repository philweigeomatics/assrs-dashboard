/**
 * MyQuestrade — the real book, not a watchlist.
 *
 * Three questions, in the order they get asked: what do I hold, where is it
 * held, and what is it doing to my risk.
 *
 * Accounts are aggregated into ONE portfolio because that is the thing that
 * has a risk profile — a TFSA and a margin account holding the same name are
 * one position as far as concentration and beta are concerned. Market value
 * stays per account, because that is the number you reconcile against the
 * statement, and because the tax treatment differs: a loss in an RRSP cannot
 * be harvested and a gain in a TFSA is not taxable.
 *
 * The risk report is a separate query from the book. Positions move with the
 * tape and are cached for minutes; three years of daily history is not, and
 * re-downloading it every time a position ticks would make the page unusable.
 */

import { useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import type { QtBook, QtScope } from "../lib/types";
import { NavBar } from "../components/NavBar";
import { ConnectCard } from "../components/questrade/ConnectCard";
import { Holdings } from "../components/questrade/Holdings";
import { RiskPanel } from "../components/questrade/RiskPanel";
import { ExposurePanel } from "../components/questrade/ExposurePanel";
import { OptimisePanel } from "../components/questrade/OptimisePanel";
import { Transactions } from "../components/questrade/Transactions";
import { usePersistentState } from "../lib/usePersistentState";
import { fixed, signed } from "../lib/format";

/**
 * Two views of the same connection. 组合 is everything as it stands today;
 * 流水 is what happened over a year, which is a different question and a
 * different shape, and stacking it under five more sections would have
 * buried it.
 */
const VIEWS = [
  { id: "book", label: "💼 组合" },
  { id: "ledger", label: "📒 交易流水" },
] as const;

type View = typeof VIEWS[number]["id"];

//: Questrade served activity for 2023 on a live account; there is no
//: endpoint that reports how far back a connection goes, so the picker
//: offers a window and an empty year simply comes back empty.
const YEARS = Array.from({ length: 6 },
  (_, i) => new Date().getFullYear() - i);

const BENCHMARKS = [
  { id: "^GSPC", name: "S&P 500" },
  { id: "^IXIC", name: "NASDAQ" },
  { id: "^GSPTSE", name: "S&P/TSX" },
];

/** North America: green up, red down. */
const NA = false;

function money(v: number | null | undefined, ccy: string, nd = 0): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const sym = ccy === "CAD" ? "C$" : "$";
  return `${sym}${v.toLocaleString(undefined, {
    minimumFractionDigits: nd, maximumFractionDigits: nd })}`;
}

export function MyQuestrade() {
  useEffect(() => { document.title = "ASSRS · MyQuestrade"; }, []);
  const qc = useQueryClient();

  const [base, setBase] = usePersistentState<"CAD" | "USD">("assrs.qt.base", "CAD");
  const [benchmark, setBenchmark] = usePersistentState<string>("assrs.qt.bench", "^GSPC");
  const [scope, setScope] = usePersistentState<QtScope>("assrs.qt.scope", "all");
  const [method, setMethod] = usePersistentState<string>("assrs.qt.method", "min_var");
  const [cap, setCap] = usePersistentState<number>("assrs.qt.cap", 0.25);

  const [view, setView] = usePersistentState<View>("assrs.qt.view", "book");
  const [year, setYear] = usePersistentState<number>(
    "assrs.qt.year", new Date().getFullYear());

  const status = useQuery({ queryKey: ["qt", "status"], queryFn: api.qtStatus,
    staleTime: 60_000, retry: false });
  const connected = status.data?.connected === true;

  const book = useQuery({
    queryKey: ["qt", "book", base], queryFn: () => api.qtBook(base),
    enabled: connected, staleTime: 2 * 60_000, retry: false,
  });
  const risk = useQuery({
    queryKey: ["qt", "risk", benchmark, base, scope],
    queryFn: () => api.qtRisk(benchmark, base, scope),
    enabled: connected, staleTime: 20 * 60_000, retry: false,
  });
  // Its own query: a first call reads a profile per holding from Yahoo, which
  // is slow enough that the rest of the page must not wait behind it.
  const exposure = useQuery({
    queryKey: ["qt", "exposure", base], queryFn: () => api.qtExposure(base),
    enabled: connected, staleTime: 6 * 3600_000, retry: false,
  });
  const alloc = useQuery({
    queryKey: ["qt", "opt", base, scope, method, cap],
    queryFn: () => api.qtOptimise(base, scope, method, cap),
    enabled: connected, staleTime: 20 * 60_000, retry: false,
  });

  const disconnect = useMutation({
    mutationFn: () => api.qtDisconnect(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["qt"] }),
  });

  // A broken token chain comes back as 409, deliberately not 401 — the app
  // must not sign you out of Supabase because a brokerage link expired.
  const ledger = useQuery({
    queryKey: ["qt", "txn", year],
    queryFn: () => api.qtTransactions(year),
    enabled: connected && view === "ledger",
    // Thirteen requests per account per year on the server; a closed year
    // never changes, so never refetch it on a remount.
    staleTime: 6 * 3600_000,
  });

  const needsReconnect = [book.error, risk.error, status.error]
    .some((e) => e instanceof ApiError && e.status === 409);

  // A status that could not even be read — the table missing, most likely —
  // has to reach the screen. Otherwise setup fails with an empty connect form
  // and nothing anywhere saying which migration has not been run.
  const setupError = status.isError && !needsReconnect
    ? (status.error as ApiError)?.message : null;

  return (
    <div className="min-h-screen">
      <NavBar>
        {connected && book.data && (
          <span className="label truncate">
            {book.data.as_of} · 汇率 {fixed(book.data.fx.rates.USD, 4)} ({book.data.fx.source})
            {book.data.delayed && <b className="text-brand-ink"> · 部分报价延迟</b>}
          </span>
        )}
      </NavBar>

      <main className="max-w-[1800px] mx-auto px-3 py-3 flex flex-col gap-3">
        {status.isPending && <div className="card p-10 text-center label">正在检查连接…</div>}

        {(!connected || needsReconnect) && !status.isPending && (
          <ConnectCard reason={
            needsReconnect ? "Questrade 连接已失效，请重新生成并粘贴一个刷新令牌。"
                           : setupError ?? status.data?.reason} />
        )}

        {connected && !needsReconnect && (
          <>
            <section className="card p-2 flex flex-wrap items-center gap-1">
              {VIEWS.map((v) => (
                <button key={v.id} onClick={() => setView(v.id)}
                  className={`h-8 px-3 rounded-lg text-[13px] font-medium transition-colors ${
                    view === v.id ? "bg-elevated text-ink" : "text-ink-mute hover:text-ink"
                  }`}>
                  {v.label}
                </button>
              ))}
            </section>
          </>
        )}

        {connected && !needsReconnect && view === "ledger" && (
          <section className="card p-3 flex flex-col gap-2">
            <h2 className="text-[14.5px] font-semibold">📒 交易流水</h2>
            <p className="label">
              买卖、股息、利息、存取、费用，全部按账户分开。为报税整理，
              所以注册账户与非注册账户分得很清楚，CAD 与 USD 从不相加。
            </p>
            <Body q={ledger}>
              {ledger.data && (
                <Transactions d={ledger.data} year={year} onYear={setYear}
                  years={YEARS} />
              )}
            </Body>
          </section>
        )}

        {connected && !needsReconnect && view === "book" && (
          <>
            <section className="card p-3 flex flex-col gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-[14.5px] font-semibold">💼 组合总览</h2>
                <div className="ml-auto flex items-center gap-2">
                  <div className="flex rounded-lg bg-sunken p-0.5" title="所有金额换算成哪种货币">
                    {(["CAD", "USD"] as const).map((c) => (
                      <button key={c} onClick={() => setBase(c)}
                        className={`px-2.5 h-7 rounded-md text-[12.5px] font-medium ${
                          base === c ? "bg-panel text-ink shadow-sm" : "text-ink-mute"}`}>
                        {c}
                      </button>
                    ))}
                  </div>
                  <button onClick={() => {
                    book.refetch(); risk.refetch(); exposure.refetch(); alloc.refetch();
                  }}
                    className="h-7 px-2.5 rounded-lg bg-sunken text-[12.5px] text-ink-dim">
                    ↻ 刷新
                  </button>
                  <button onClick={() => disconnect.mutate()}
                    className="h-7 px-2.5 text-[12.5px] text-ink-mute hover:text-up">
                    断开连接
                  </button>
                </div>
              </div>

              <Body q={book}>
                {book.data && <Overview book={book.data} base={base} />}
              </Body>
            </section>

            <section className="card p-3 flex flex-col gap-2">
              <h2 className="text-[14.5px] font-semibold">📋 持仓明细</h2>
              <p className="label">
                同一只股票跨账户合并为一行；点「N 个账户」展开可看各账户的数量与成本。
                均价按成本加权 —— 各账户的 ACB 请看展开行。
              </p>
              <Body q={book}>
                {book.data && book.data.holdings.length > 0
                  ? <Holdings book={book.data} />
                  : <p className="label py-6 text-center">这些账户里没有持仓。</p>}
              </Body>
            </section>

            <section className="card p-3 flex flex-col gap-2">
              <h2 className="text-[14.5px] font-semibold">📊 风险与收益分析</h2>
              <Body q={risk}>
                {risk.data && (
                  <RiskPanel risk={risk.data} benchmarks={BENCHMARKS}
                    benchmark={benchmark} onBenchmark={setBenchmark}
                    scope={scope} onScope={setScope} />
                )}
              </Body>
            </section>

            <section className="card p-3 flex flex-col gap-2">
              <h2 className="text-[14.5px] font-semibold">🏭 行业暴露</h2>
              <p className="label">
                ETF 已穿透到其成分行业 —— 否则一个半仓指数基金的组合只会告诉你
                「50% 是 ETF」，而那既不是行业，也不是答案。
              </p>
              <Body q={exposure}>
                {exposure.data && <ExposurePanel data={exposure.data} />}
              </Body>
            </section>

            <section className="card p-3 flex flex-col gap-2">
              <h2 className="text-[14.5px] font-semibold">⚖️ 配置优化</h2>
              <p className="label">
                在<b>你已经选好的标的</b>之间重新分配权重，不引入新标的。
                顶部的「仅股票 / 仅 ETF」同时作用于这里。
              </p>
              <Body q={alloc}>
                {alloc.data && (
                  <OptimisePanel data={alloc.data} method={method} onMethod={setMethod}
                    cap={cap} onCap={setCap} />
                )}
              </Body>
            </section>
          </>
        )}
      </main>
    </div>
  );
}

function Body({ q, children }: {
  q: { isPending: boolean; isError: boolean; error: unknown; refetch: () => void };
  children: React.ReactNode;
}) {
  if (q.isPending) return <div className="py-10 text-center label">加载中…</div>;
  if (q.isError) {
    const err = q.error as ApiError;
    if (err?.status === 409) return null;         // handled by the reconnect card
    return (
      <div className="py-6 text-center flex flex-col gap-1.5">
        <p className="text-[12.5px] text-up">{err?.message}</p>
        <button onClick={() => q.refetch()} className="text-cyan text-[13px]">重试</button>
      </div>
    );
  }
  return <>{children}</>;
}

function Overview({ book, base }: { book: QtBook; base: "CAD" | "USD" }) {
  const t = book.totals;
  const pnlTone = (t.open_pnl > 0) === NA ? "text-up" : "text-down";

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <Big label="总权益" value={money(t.equity, base)}
          hint="持仓市值 + 现金，全部账户合计。" />
        <Big label="持仓市值" value={money(t.market_value, base)}
          hint={`${t.positions} 只，分布在 ${t.accounts} 个账户。`
            + (t.dust ? ` 另有 ${t.dust} 笔零碎持仓（不足 0.01）未显示。` : "")}
          sub={t.dust ? `${t.positions} 只 · ${t.dust} 笔零碎已隐藏` : `${t.positions} 只`} />
        <Big label="现金" value={money(t.cash, base)} hint="各账户各币种现金换算合计。" />
        <Big label="浮动盈亏" value={money(t.open_pnl, base)} cls={pnlTone}
          hint="持仓市值 − 持仓成本。已实现盈亏不在其中。"
          sub={`${signed(t.open_pnl_pct, 2)}%`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-3">
        <div className="flex flex-col gap-1.5">
          <span className="label">各账户市值（{base}）</span>
          {book.accounts.map((a) => (
            <div key={a.id} className="flex items-baseline gap-2 text-[12.5px]
                                       rounded-lg bg-sunken px-2.5 py-1.5">
              <span className="font-medium">{a.label}</span>
              <span className="label font-mono">{a.id}</span>
              <span className="label ml-auto">{a.positions} 只</span>
              <span className="tnum w-24 text-right" title="现金">
                {money(a.cash_base, base)}
              </span>
              <span className="tnum w-28 text-right font-semibold">
                {money(a.market_value_base, base)}
              </span>
            </div>
          ))}
        </div>

        <div className="flex flex-col gap-2">
          <Mix title="按币种" rows={book.mix.currency} base={base} />
          <Mix title="按类型" rows={book.mix.kind} base={base} />
        </div>
      </div>

      {book.warnings.length > 0 && (
        <ul className="rounded-lg bg-sunken px-2.5 py-2 flex flex-col gap-0.5">
          {book.warnings.map((w) => (
            <li key={w} className="text-[11.5px] text-brand-ink leading-snug">⚠ {w}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Mix({ title, rows, base }: {
  title: string; rows: { name: string; value: number; pct: number | null }[];
  base: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="label">{title}</span>
      <div className="flex h-5 rounded-md overflow-hidden bg-sunken">
        {rows.map((r, i) => (
          <div key={r.name} style={{
            width: `${r.pct ?? 0}%`,
            background: ["#0062cc", "#5856d6", "#ff9500", "#1f7a35", "#8e8e93"][i % 5],
          }} title={`${r.name} ${fixed(r.pct, 1)}%`} />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5">
        {rows.map((r, i) => (
          <span key={r.name} className="text-[11.5px] text-ink-dim flex items-center gap-1">
            <i className="w-2 h-2 rounded-sm inline-block" style={{
              background: ["#0062cc", "#5856d6", "#ff9500", "#1f7a35", "#8e8e93"][i % 5],
            }} />
            {r.name} {fixed(r.pct, 1)}%
            <span className="text-ink-mute tnum">{money(r.value, base)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

function Big({ label, value, hint, cls = "", sub }: {
  label: string; value: string; hint: string; cls?: string; sub?: string;
}) {
  return (
    <div className="rounded-lg bg-sunken px-2.5 py-2" title={hint}>
      <div className="label truncate">{label}</div>
      <div className={`text-[19px] font-semibold tnum ${cls}`}>{value}</div>
      {sub && <div className={`text-[12px] tnum ${cls}`}>{sub}</div>}
    </div>
  );
}
