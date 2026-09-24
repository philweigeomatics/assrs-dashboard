/**
 * ⚖️ 调仓 — replace the mandate.
 *
 * The positions held today close at the end of it, and the new weights take
 * effect at tomorrow's open — which is what `execute_fund_rebalance` has
 * always done. Old weights are retired with an end date rather than deleted,
 * so what this fund used to target stays readable afterwards.
 *
 * Every rule here is enforced again on the server. A disabled button is a
 * courtesy to the user, not a control on the request.
 */

import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";
import type { FundDetail, StockRef } from "../../lib/types";
import { useSymbolSearch } from "../../lib/useSymbolSearch";
import { fixed, signed } from "../../lib/format";

const TOLERANCE = 0.1;

type Row = { t: string; n: string; weight_pct: number };

export function Rebalance({ d, onDone }: { d: FundDetail; onDone: () => void }) {
  const start = useMemo<Row[]>(
    () => d.holdings.map((h) => ({ t: h.t, n: h.n, weight_pct: h.weight_pct })),
    [d.holdings]);
  const [rows, setRows] = useState<Row[]>(start);
  const [confirming, setConfirming] = useState(false);

  const actual = new Map(d.drift.map((x) => [x.t, x.actual_pct]));
  const total = rows.reduce((a, r) => a + r.weight_pct, 0);
  const balanced = Math.abs(total - 100) <= TOLERANCE;
  const changed = rows.length !== start.length
    || rows.some((r, i) => start[i]?.t !== r.t
                        || Math.abs(start[i].weight_pct - r.weight_pct) > 0.001);

  const run = useMutation({
    mutationFn: () => api.rebalanceFund(d.id, rows
      .filter((r) => r.weight_pct > 0)
      .map((r) => ({ t: r.t, weight_pct: r.weight_pct }))),
    onSuccess: () => { setConfirming(false); onDone(); },
  });

  const set = (t: string, v: number) => setRows((p) => p.map(
    (r) => (r.t === t ? { ...r, weight_pct: Math.max(0, Math.min(100, v)) } : r)));

  /** Snap targets to what the market has already made the weights. */
  const adopt = () => setRows((p) => p.map(
    (r) => ({ ...r, weight_pct: actual.get(r.t) ?? r.weight_pct })));

  return (
    <div className="flex flex-col gap-2.5">
      <p className="label leading-snug">
        今天的持仓收盘后结束，新的目标权重从明天开盘生效。旧权重不会被删除，只是标上结束日期
        —— 这个组合以前对标过什么，之后还查得到。
      </p>

      <div className="flex flex-col gap-1.5">
        {rows.map((r) => {
          const now = actual.get(r.t);
          return (
            <div key={r.t} className="flex items-center gap-2 text-[12.5px]">
              <span className="w-20 sm:w-24 shrink-0 truncate" title={r.n}>
                {r.n}
              </span>
              <span className="hidden lg:block w-20 shrink-0 font-mono tnum
                text-[11px] text-ink-mute">
                {r.t}
              </span>
              <span className="hidden md:block w-20 shrink-0 tnum text-[11.5px]
                text-ink-mute" title="市值涨跌后的当前实际权重">
                现 {now == null ? "—" : `${fixed(now, 1)}%`}
              </span>
              <input type="range" min={0} max={100} step={0.5} value={r.weight_pct}
                onChange={(e) => set(r.t, Number(e.target.value))}
                aria-label={`${r.n} 目标权重`}
                className="flex-1 min-w-[48px] accent-[var(--color-cyan)]" />
              <input type="number" min={0} max={100} step={0.5} value={r.weight_pct}
                onChange={(e) => set(r.t, Number(e.target.value))}
                className="w-14 sm:w-16 h-7 px-1.5 rounded-md bg-sunken text-[12px]
                  tnum text-right outline-none shrink-0" />
              <button onClick={() => setRows((p) => p.filter((x) => x.t !== r.t))}
                aria-label={`移除 ${r.n}`}
                className="text-ink-mute hover:text-up px-1">✕</button>
            </div>
          );
        })}
      </div>

      <AddRow have={rows.map((r) => r.t)}
        onAdd={(s) => setRows((p) => [...p, { t: s.t, n: s.n, weight_pct: 0 }])} />

      <div className="flex flex-wrap items-center gap-2">
        <span className={`text-[12.5px] tnum font-medium ${
          balanced ? "text-up" : "text-brand-ink"}`}>
          合计 {fixed(total, 2)}%
          {!balanced && <span className="font-normal"> · 必须正好 100%</span>}
        </span>
        <button onClick={adopt} disabled={d.drift.length === 0}
          title="把目标权重改成当前实际权重 —— 等于承认漂移，不做交易"
          className="h-7 px-2 rounded-md bg-sunken text-[12px] disabled:opacity-50">
          采用当前权重
        </button>
        <button onClick={() => setRows(start)} disabled={!changed}
          className="h-7 px-2 rounded-md bg-sunken text-[12px] disabled:opacity-50">
          还原
        </button>

        <div className="ml-auto flex items-center gap-2">
          {confirming ? (
            <>
              <span className="text-[12px] text-ink-dim">
                {rows.filter((r) => r.weight_pct > 0).length} 只，明天开盘生效？
              </span>
              <button onClick={() => run.mutate()} disabled={run.isPending}
                className="h-8 px-3 rounded-lg bg-cyan text-white text-[12.5px]
                  font-semibold disabled:opacity-60">
                {run.isPending ? "执行中…" : "确认调仓"}
              </button>
              <button onClick={() => setConfirming(false)}
                className="h-8 px-2 rounded-lg bg-sunken text-[12.5px]">取消</button>
            </>
          ) : (
            <button onClick={() => setConfirming(true)}
              disabled={!balanced || !changed}
              title={!changed ? "还没有改动"
                : !balanced ? "权重合计必须正好 100%" : undefined}
              className="h-8 px-3 rounded-lg bg-cyan text-white text-[12.5px]
                font-semibold disabled:opacity-50">
              🚀 执行调仓
            </button>
          )}
        </div>
      </div>

      {run.isError && (
        <p className="text-[12.5px] text-up">{(run.error as ApiError).message}</p>
      )}
      {run.isSuccess && (
        <p className="text-[12.5px] text-up">{run.data.message}</p>
      )}

      <Diff start={start} rows={rows} />
    </div>
  );
}

/** What the rebalance actually does, named as trades rather than numbers. */
function Diff({ start, rows }: { start: Row[]; rows: Row[] }) {
  const before = new Map(start.map((r) => [r.t, r.weight_pct]));
  const moves = rows
    .map((r) => ({ ...r, was: before.get(r.t) ?? 0 }))
    .filter((r) => Math.abs(r.weight_pct - r.was) > 0.05)
    .concat(start.filter((s) => !rows.some((r) => r.t === s.t))
      .map((s) => ({ ...s, weight_pct: 0, was: s.weight_pct })));

  if (moves.length === 0) return null;
  return (
    <div className="rounded-lg bg-sunken p-2.5 flex flex-col gap-1">
      <span className="label">这次调仓要做的事</span>
      {moves.map((m) => (
        <div key={m.t} className="flex items-baseline gap-2 text-[12px]">
          <span className="w-24 truncate">{m.n}</span>
          <span className="tnum text-ink-mute">
            {fixed(m.was, 1)}% → {fixed(m.weight_pct, 1)}%
          </span>
          <span className={`tnum font-medium ${
            m.weight_pct > m.was ? "text-up" : "text-down"}`}>
            {m.weight_pct > m.was ? "加仓"
              : m.weight_pct === 0 ? "清仓" : "减仓"}
            {" "}{signed(m.weight_pct - m.was, 1, "pp")}
          </span>
        </div>
      ))}
    </div>
  );
}

function AddRow({ have, onAdd }: { have: string[]; onAdd: (s: StockRef) => void }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const stocks = useQuery({ queryKey: ["stocks"], queryFn: api.stocks,
    staleTime: 6 * 3600_000 });
  const { items } = useSymbolSearch({
    query: q, stocks: stocks.data ?? [], markets: ["CN"], limit: 8 });
  // Held codes carry an exchange suffix; the picker offers bare ones.
  const bare = new Set(have.map((t) => t.split(".")[0]));
  const hits = items.filter((s) => !bare.has(s.t.split(".")[0]));

  return (
    <div className="relative">
      <input value={q} onChange={(e) => { setQ(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onBlur={() => globalThis.setTimeout(() => setOpen(false), 150)}
        placeholder="加一只新的…"
        className="h-7 w-52 px-2 rounded-md bg-sunken text-[12.5px] outline-none
          focus:ring-2 focus:ring-cyan/40" />
      {open && hits.length > 0 && (
        <div className="card absolute left-0 mt-1 z-30 w-64 py-1">
          {hits.map((s) => (
            <button key={s.t} onMouseDown={(e) => e.preventDefault()}
              onClick={() => { onAdd({ t: s.t, n: s.n }); setQ(""); setOpen(false); }}
              className="w-full flex items-baseline gap-2 px-2 py-1 text-left
                hover:bg-elevated">
              <span className="font-mono tnum text-[12px] text-ink-dim shrink-0">
                {s.t}
              </span>
              <span className="text-[12.5px] truncate">{s.n}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
