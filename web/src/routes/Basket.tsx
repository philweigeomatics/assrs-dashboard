/**
 * 多股对比 — ranking a group, with no chart.
 *
 * Built for one strategy: several names are moving together and all trending
 * up, so sell the ones moving least and concentrate into the leader. Six lines
 * on a chart cannot answer that; a ranked table with beta and alpha can.
 *
 * The page deliberately leads with the two things that decide whether the
 * strategy applies at all, before the ranking itself:
 *
 *   * 平均相关性 — are they actually one trade? Below 0.5 the premise is
 *     missing and the ranking is comparing unrelated bets.
 *   * α 排名 next to 涨幅排名 — is the laggard weak, or just lower-beta? A
 *     0.6-beta name in a 1.4-beta basket is supposed to rise less, and selling
 *     it for that is selling low volatility to buy high volatility.
 *
 * Same market only, for the same reason the pairwise comparison is: one beta
 * needs one index, and two currencies cannot share a ranking.
 */

import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { BasketStats, BasketStock, MarketCode, StockRef } from "../lib/types";
import { NavBar } from "../components/NavBar";
import { useSymbolSearch } from "../lib/useSymbolSearch";
import { usePersistentState } from "../lib/usePersistentState";
import { fixed, signed } from "../lib/format";
import { splitMarket } from "../lib/symbols";

const WINDOWS = [
  { id: "60", label: "3个月" },
  { id: "120", label: "6个月" },
  { id: "252", label: "1年" },
  { id: "all", label: "全部" },
];

const MAX = 8;

export function Basket() {
  useEffect(() => { document.title = "ASSRS · 多股对比"; }, []);

  const [picked, setPicked] = usePersistentState<StockRef[]>("assrs.basket.v1", []);
  const [window, setWindow] = useState("252");

  const stocks = useQuery({ queryKey: ["stocks"], queryFn: api.stocks, staleTime: 6 * 3600_000 });
  const symbols = picked.map((s) => s.t);

  const q = useQuery({
    queryKey: ["basket", symbols.join(","), window],
    queryFn: () => api.basket(symbols, window),
    enabled: symbols.length >= 2,
    staleTime: 20 * 60_000,
  });

  // The first pick fixes the market; everything after must match it.
  const market = picked.length ? splitMarket(picked[0]!.t) : null;

  function add(s: StockRef) {
    if (picked.length >= MAX || picked.some((p) => p.t === s.t)) return;
    setPicked([...picked, s]);
  }

  return (
    <div className="min-h-screen">
      <NavBar />
      <main className="max-w-[1800px] mx-auto px-3 py-3 flex flex-col gap-3">
        <Picker picked={picked} market={market} stocks={stocks.data ?? []}
          onAdd={add} onRemove={(t) => setPicked(picked.filter((p) => p.t !== t))}
          onClear={() => setPicked([])} window={window} setWindow={setWindow} />

        {symbols.length < 2 && (
          <div className="card p-10 text-center label">
            至少选择两只同市场股票 — 适合用来比较一组同涨同跌的标的
          </div>
        )}

        {symbols.length >= 2 && q.isPending && (
          <div className="card p-10 text-center label">正在计算…</div>
        )}
        {q.isError && (
          <div className="card p-6 text-center">
            <p className="text-up text-[13px]">{(q.error as Error).message}</p>
            <button onClick={() => q.refetch()} className="mt-2 text-cyan text-[13px]">重试</button>
          </div>
        )}
        {q.data && <Body d={q.data} />}
      </main>
    </div>
  );
}

function Picker({ picked, market, stocks, onAdd, onRemove, onClear, window, setWindow }: {
  picked: StockRef[]; market: MarketCode | null; stocks: StockRef[];
  onAdd: (s: StockRef) => void; onRemove: (t: string) => void; onClear: () => void;
  window: string; setWindow: (w: string) => void;
}) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const { items, searching } = useSymbolSearch({
    query: q, stocks,
    markets: market ? [market] : ["CN", "US", "CA"],
    limit: 8,
  });
  const chosen = new Set(picked.map((p) => p.t));
  const hits = useMemo(() => items.filter((s) => !chosen.has(s.t)), [items, chosen]);

  return (
    <section className="card p-3 flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-[14px] font-semibold">🧺 多股对比</h2>
        {market && (
          <span className="label">
            仅限同市场 · {market === "CN" ? "A股" : market === "US" ? "美股" : "加股"}
          </span>
        )}
        <div className="ml-auto flex items-center gap-1">
          {WINDOWS.map((w) => (
            <button key={w.id} onClick={() => setWindow(w.id)}
              className={`h-7 px-2 rounded-md text-[12.5px] border transition-colors ${
                w.id === window ? "border-cyan bg-cyan text-white" : "border-line bg-panel hover:bg-elevated"
              }`}>
              {w.label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        {picked.map((s) => (
          <span key={s.t}
            className="h-7 pl-2 pr-1 rounded-md bg-sunken flex items-center gap-1.5 text-[12.5px]">
            <span className="truncate max-w-[130px]">{s.n}</span>
            <span className="font-mono tnum text-[11px] text-ink-mute">{s.t}</span>
            <button onClick={() => onRemove(s.t)} aria-label={`移除 ${s.n}`}
              className="text-ink-mute hover:text-ink px-1">✕</button>
          </span>
        ))}

        {picked.length < MAX && (
          <div className="relative">
            <input
              value={q} onChange={(e) => { setQ(e.target.value); setOpen(true); }}
              onFocus={() => setOpen(true)}
              onBlur={() => globalThis.setTimeout(() => setOpen(false), 150)}
              placeholder={picked.length ? "再加一只…" : "输入代码或名称添加…"}
              className="h-7 w-48 px-2 rounded-md bg-sunken text-[12.5px] outline-none focus:ring-2 focus:ring-cyan/40"
            />
            {open && (hits.length > 0 || searching) && (
              <div className="card absolute left-0 mt-1 z-30 w-64 py-1">
                {hits.map((s) => (
                  <button key={s.t} onMouseDown={(e) => e.preventDefault()}
                    onClick={() => { onAdd(s); setQ(""); setOpen(false); }}
                    className="w-full flex items-baseline gap-2 px-2 py-1 text-left hover:bg-elevated">
                    <span className="font-mono tnum text-[12px] text-ink-dim shrink-0">{s.t}</span>
                    <span className="text-[12.5px] truncate">{s.n}</span>
                  </button>
                ))}
                {searching && hits.length === 0 && <div className="px-2 py-1 label">正在搜索…</div>}
              </div>
            )}
          </div>
        )}

        {picked.length > 0 && (
          <button onClick={onClear} className="text-[12px] text-cyan ml-1">清空</button>
        )}
        {picked.length >= MAX && <span className="label">最多 {MAX} 只</span>}
      </div>
    </section>
  );
}

function Body({ d }: { d: BasketStats }) {
  return (
    <>
      <Summary d={d} />
      <section className="card p-3 overflow-x-auto">
        <Table d={d} />
      </section>
      <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,420px)] items-start">
        <Verdicts d={d} />
        <Correlation d={d} />
      </div>
      <p className="label leading-snug">
        β/α 对 {d.benchmark?.label ?? "基准"} 回归（对数收益），无风险利率取 0。
        「与同伴相关」为与篮子中其余标的的平均相关系数。历史统计，非预测。
      </p>
    </>
  );
}

function Summary({ d }: { d: BasketStats }) {
  const b = d.basket;
  return (
    <section className="card p-3 flex flex-wrap items-baseline gap-x-5 gap-y-1 text-[12.5px]">
      <span className="label">{d.from} → {d.to} · {d.bars} 个共同交易日</span>
      <span>篮子等权 <b className="font-mono tnum">{signed(b.total_return_pct, 1, "%")}</b></span>
      <span>
        平均相关性{" "}
        <b className={`font-mono tnum ${b.cohesive ? "" : "text-brand-ink"}`}>
          {fixed(b.avg_correlation, 2)}
        </b>
        <span className="text-ink-mute"> {b.cohesive ? "· 同涨同跌" : "· 并未一起动"}</span>
      </span>
      <span>首尾差 <b className="font-mono tnum">{fixed(b.spread_pct, 0)}pp</b></span>
      {d.benchmark && (
        <span className="text-ink-mute">
          同期{d.benchmark.label} {signed(d.benchmark.total_return_pct, 1, "%")}
        </span>
      )}
    </section>
  );
}

const COLS: { k: keyof BasketStock; label: string; nd: number; suffix?: string;
              hint?: string; lowerBetter?: boolean }[] = [
  { k: "total_return_pct", label: "区间涨幅", nd: 1, suffix: "%" },
  { k: "vs_basket", label: "vs 篮子", nd: 2, hint: "1.20 = 比等权篮子多涨两成" },
  { k: "beta", label: "β", nd: 2, hint: "市场每涨1%，它涨多少" },
  { k: "alpha_annual_pct", label: "α 年化", nd: 0, suffix: "%", hint: "市场解释不了的部分" },
  { k: "sharpe", label: "夏普", nd: 2 },
  { k: "vol_annual_pct", label: "年化波动", nd: 0, suffix: "%", lowerBetter: true },
  { k: "max_drawdown_pct", label: "最大回撤", nd: 0, suffix: "%", lowerBetter: true },
  { k: "corr_to_peers", label: "与同伴相关", nd: 2, hint: "与篮子其余标的的平均相关性，低=不是同一笔交易" },
];

function Table({ d }: { d: BasketStats }) {
  const best = new Map<string, number>();
  for (const c of COLS) {
    const vals = d.stocks.map((s) => s[c.k] as number | null).filter((v): v is number => v != null);
    if (vals.length) best.set(String(c.k), c.lowerBetter ? Math.min(...vals.map(Math.abs)) : Math.max(...vals));
  }

  return (
    <table className="w-full text-[12.5px] border-collapse">
      <thead>
        <tr className="text-ink-mute">
          <th className="text-left font-normal pb-1 pr-2">#</th>
          <th className="text-left font-normal pb-1 pr-3">股票</th>
          {COLS.map((c) => (
            <th key={String(c.k)} title={c.hint}
              className="text-right font-normal pb-1 px-2 whitespace-nowrap">
              {c.label}{c.hint && <span className="text-ink-mute"> ⓘ</span>}
            </th>
          ))}
          <th className="text-right font-normal pb-1 pl-2 whitespace-nowrap"
            title="涨幅排名 / α排名。两者都垫底才是真的弱">涨/α 排名</th>
        </tr>
      </thead>
      <tbody>
        {d.stocks.map((s) => (
          <tr key={s.symbol} className="border-t border-line">
            <td className="py-1 pr-2 text-ink-mute font-mono tnum">{s.rank_return}</td>
            <td className="py-1 pr-3">
              <Link to={`/?t=${encodeURIComponent(s.symbol)}`} className="hover:text-cyan">
                <span className="truncate">{s.label}</span>{" "}
                <span className="font-mono tnum text-[11px] text-ink-mute">{s.symbol}</span>
              </Link>
            </td>
            {COLS.map((c) => {
              const v = s[c.k] as number | null;
              const top = v != null
                && (c.lowerBetter ? Math.abs(v) : v) === best.get(String(c.k));
              return (
                <td key={String(c.k)}
                  className={`py-1 px-2 text-right font-mono tnum whitespace-nowrap ${
                    top ? "font-semibold" : "text-ink-dim"}`}>
                  {v == null ? "—" : `${fixed(v, c.nd)}${c.suffix ?? ""}`}
                </td>
              );
            })}
            <td className="py-1 pl-2 text-right font-mono tnum text-ink-dim">
              {s.rank_return}/{s.rank_alpha ?? "—"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const TONE: Record<string, { bg: string; icon: string }> = {
  warn: { bg: "bg-brand/10", icon: "⚠️" },
  weak: { bg: "bg-sunken", icon: "🔻" },
  ok: { bg: "bg-sunken", icon: "✅" },
  mixed: { bg: "bg-sunken", icon: "ℹ️" },
};

function Verdicts({ d }: { d: BasketStats }) {
  if (d.verdicts.length === 0) return <div />;
  return (
    <section className="card p-3 flex flex-col gap-2">
      <h2 className="text-[13px] font-semibold">该不该换掉涨得少的那只</h2>
      {d.verdicts.map((v, i) => (
        <p key={i} className={`rounded-lg px-2.5 py-2 text-[12.5px] leading-snug ${TONE[v.kind]?.bg ?? "bg-sunken"}`}>
          {TONE[v.kind]?.icon} {v.text}
        </p>
      ))}
    </section>
  );
}

function Correlation({ d }: { d: BasketStats }) {
  const n = d.symbols.length;
  const label = (sym: string) =>
    d.stocks.find((s) => s.symbol === sym)?.label ?? sym;

  // One hue, darker with correlation; negatives get the opposite hue so the
  // sign is never carried by shade alone.
  const cell = (v: number | null) => {
    if (v == null) return { background: "transparent", color: "inherit" };
    const a = Math.min(Math.abs(v), 1);
    return {
      background: v >= 0
        ? `rgba(0, 98, 204, ${0.08 + a * 0.55})`
        : `rgba(180, 83, 9, ${0.08 + a * 0.55})`,
      color: a > 0.6 ? "#fff" : "var(--color-ink-dim)",
    };
  };

  return (
    <section className="card p-3 overflow-x-auto">
      <h2 className="text-[13px] font-semibold mb-2">两两相关性</h2>
      <table className="text-[11.5px] border-collapse">
        <thead>
          <tr>
            <th />
            {d.symbols.map((s) => (
              <th key={s} className="px-1 pb-1 font-normal text-ink-mute whitespace-nowrap">
                {label(s).slice(0, 4)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {d.symbols.map((row, i) => (
            <tr key={row}>
              <td className="pr-2 text-ink-mute whitespace-nowrap">{label(row).slice(0, 6)}</td>
              {Array.from({ length: n }, (_, j) => (
                <td key={j} className="px-1 py-0.5">
                  <span className="block rounded px-1.5 py-1 text-center font-mono tnum"
                    style={cell(d.correlation[i]?.[j] ?? null)}>
                    {fixed(d.correlation[i]?.[j], 2)}
                  </span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
