/**
 * 组合优化 — pick names, choose how to weight them, see what that would have done.
 *
 * The screen is arranged as the decision is actually made: which market,
 * which names, then how to weight them, then the evidence. The evidence is
 * the part most optimiser UIs skip — an optimiser always produces a
 * portfolio, and on the history it was fitted to that portfolio always looks
 * good, so equal weight and the index are drawn beside it every time.
 *
 * The method's own caveat is shown with the method, not buried in help:
 * 最大夏普 fits historical means, historical means do not repeat, and it
 * routinely returns three names at the cap. Saying so where it is chosen is
 * the difference between a tool and a slot machine.
 */

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";
import type { PortfolioBuild, StockRef } from "../../lib/types";
import { useSymbolSearch } from "../../lib/useSymbolSearch";
import { usePersistentState } from "../../lib/usePersistentState";
import { fixed, signed } from "../../lib/format";
import { Curves } from "./Curves";
import { Frontier } from "./Frontier";
import { CorrMatrix } from "./CorrMatrix";

const MAX_NAMES = 30;

export type Market = "CN" | "US";

const MARKETS: { id: Market; label: string; hint: string }[] = [
  { id: "CN", label: "A 股", hint: "对标沪深300" },
  { id: "US", label: "美股", hint: "对标标普500" },
];

export function Optimiser({ onSaved }: { onSaved: () => void }) {
  // Kept per market: switching back should not lose the basket you built.
  const [market, setMarket] = usePersistentState<Market>("assrs.pf.market", "CN");
  const [cnPicks, setCnPicks] = usePersistentState<StockRef[]>("assrs.pf.cn", []);
  const [usPicks, setUsPicks] = usePersistentState<StockRef[]>("assrs.pf.us", []);
  const picked = market === "CN" ? cnPicks : usPicks;
  const setPicked = market === "CN" ? setCnPicks : setUsPicks;

  const [method, setMethod] = usePersistentState<string>("assrs.pf.method", "min_var");
  const [capPct, setCapPct] = usePersistentState<number>("assrs.pf.cap", 25);
  const [lookback, setLookback] = usePersistentState<number>("assrs.pf.lb", 242);

  const methods = useQuery({ queryKey: ["pf", "methods"], queryFn: api.optMethods,
    staleTime: 24 * 3600_000 });
  const run = useMutation({
    mutationFn: () => api.portfolioBuild({
      symbols: picked.map((p) => p.t), method, cap_pct: capPct,
      lookback, duration: 1, rf_pct: 0,
    }),
  });

  const chosen = methods.data?.methods.find((m) => m.id === method);
  const ready = picked.length >= 2;

  return (
    <div className="flex flex-col gap-3">
      <section className="card p-3 flex flex-col gap-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1">
            {MARKETS.map((m) => (
              <button key={m.id} onClick={() => setMarket(m.id)}
                title={m.hint}
                className={`h-8 px-3 rounded-lg text-[13px] font-medium transition-colors ${
                  market === m.id ? "bg-elevated text-ink" : "text-ink-mute hover:text-ink"
                }`}>
                {m.label}
              </button>
            ))}
          </div>
          <span className="label">
            一个组合只能是一个市场 —— 两边交易日历不同，混在一起的协方差是两边都没经历过的样本
          </span>
        </div>

        <Picker market={market} picked={picked} onChange={setPicked} />

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <label className="label flex items-center gap-1.5">
            方法
            <select value={method} onChange={(e) => setMethod(e.target.value)}
              className="h-7 px-1.5 rounded-md bg-sunken text-[12.5px] outline-none">
              {(methods.data?.methods ?? []).map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
          </label>
          <label className="label flex items-center gap-1.5"
            title="任何一只的最大权重。太紧会退化成等权，所以实际下限是 2/n。">
            单只上限
            <input type="range" min={5} max={100} step={5} value={capPct}
              onChange={(e) => setCapPct(Number(e.target.value))}
              className="w-28 accent-[var(--color-cyan)]" />
            <span className="tnum w-9">{capPct}%</span>
          </label>
          <label className="label flex items-center gap-1.5" title="协方差用多少个交易日估计">
            回看
            <select value={lookback} onChange={(e) => setLookback(Number(e.target.value))}
              className="h-7 px-1.5 rounded-md bg-sunken text-[12.5px] tnum outline-none">
              {[60, 120, 242, 504].map((d) => <option key={d} value={d}>{d} 天</option>)}
            </select>
          </label>
          <button onClick={() => run.mutate()} disabled={!ready || run.isPending}
            className="ml-auto h-8 px-4 rounded-lg bg-cyan text-white text-[13px]
              font-semibold disabled:opacity-60">
            {run.isPending ? "计算中…" : ready ? "优化" : "先选两只以上"}
          </button>
        </div>

        {chosen && (
          <p className={`text-[12px] leading-snug ${
            chosen.id === "max_sharpe" ? "text-brand-ink" : "label"}`}>
            <b>{chosen.label}</b>：{chosen.means}
          </p>
        )}
        {run.isError && (
          <p className="text-[12.5px] text-up">{(run.error as ApiError).message}</p>
        )}
      </section>

      {run.data && <Result d={run.data} market={market} onSaved={onSaved} />}
    </div>
  );
}

function Picker({ market, picked, onChange }: {
  market: Market; picked: StockRef[]; onChange: (v: StockRef[]) => void;
}) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const stocks = useQuery({ queryKey: ["stocks"], queryFn: api.stocks,
    staleTime: 6 * 3600_000 });
  const { items } = useSymbolSearch({
    query: q, stocks: stocks.data ?? [],
    markets: market === "CN" ? ["CN"] : ["US"], limit: 8,
  });
  const have = new Set(picked.map((p) => p.t));
  const hits = items.filter((s) => !have.has(s.t));

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {picked.map((s) => (
        <span key={s.t}
          className="h-7 pl-2 pr-1 rounded-md bg-sunken flex items-center gap-1.5 text-[12.5px]">
          <span className="truncate max-w-[130px]">{s.n}</span>
          <span className="font-mono tnum text-[11px] text-ink-mute">{s.t}</span>
          <button onClick={() => onChange(picked.filter((p) => p.t !== s.t))}
            aria-label={`移除 ${s.n}`}
            className="text-ink-mute hover:text-ink px-1">✕</button>
        </span>
      ))}
      {picked.length < MAX_NAMES && (
        <div className="relative">
          <input value={q} onChange={(e) => { setQ(e.target.value); setOpen(true); }}
            onFocus={() => setOpen(true)}
            onBlur={() => globalThis.setTimeout(() => setOpen(false), 150)}
            placeholder={picked.length ? "再加一只…" : "输入代码或名称添加…"}
            className="h-7 w-48 px-2 rounded-md bg-sunken text-[12.5px] outline-none
              focus:ring-2 focus:ring-cyan/40" />
          {open && hits.length > 0 && (
            <div className="card absolute left-0 mt-1 z-30 w-64 py-1">
              {hits.map((s) => (
                <button key={s.t} onMouseDown={(e) => e.preventDefault()}
                  onClick={() => { onChange([...picked, { t: s.t, n: s.n }]); setQ(""); setOpen(false); }}
                  className="w-full flex items-baseline gap-2 px-2 py-1 text-left hover:bg-elevated">
                  <span className="font-mono tnum text-[12px] text-ink-dim shrink-0">{s.t}</span>
                  <span className="text-[12.5px] truncate">{s.n}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      {picked.length > 0 && (
        <button onClick={() => onChange([])} className="text-[12px] text-cyan ml-1">清空</button>
      )}
      <span className="label ml-auto tnum">{picked.length}/{MAX_NAMES}</span>
    </div>
  );
}

function Result({ d, market, onSaved }: {
  d: PortfolioBuild; market: Market; onSaved: () => void;
}) {
  const held = d.holdings.filter((h) => h.weight_pct > 0);
  const capBit = d.cap_pct > d.cap_asked_pct + 0.05;

  return (
    <>
      <section className="card p-3 flex flex-col gap-2.5">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h3 className="text-[13.5px] font-semibold">🎯 {d.method_label}目标配置</h3>
          <span className="label">
            {d.from} → {d.to} · {d.lookback} 个交易日 · {held.length} 只有权重
          </span>
          <SaveFund d={d} market={market} onSaved={onSaved} />
        </div>

        {capBit && (
          <p className="text-[12px] text-brand-ink leading-snug">
            单只上限设的是 {d.cap_asked_pct}%，实际用的是 <b>{d.cap_pct}%</b> ——
            比 1/n 还紧的上限会让「权重合计 100%」无解，所以下限固定在等权的两倍。
          </p>
        )}
        {d.missing.length > 0 && (
          <p className="text-[12px] text-up">
            读不到行情，已排除：{d.missing.join("、")}
          </p>
        )}

        <div className="flex flex-col gap-1">
          {d.holdings.map((h) => (
            <div key={h.t} className="flex items-center gap-2 text-[12.5px]">
              <span className="w-28 shrink-0 truncate">{h.n}</span>
              <span className="w-16 shrink-0 font-mono tnum text-[11px] text-ink-mute">
                {h.t}
              </span>
              <div className="flex-1 h-4 rounded-sm bg-sunken overflow-hidden">
                <div className="h-full rounded-sm bg-cyan"
                  style={{ width: `${Math.min(100, h.weight_pct)}%` }} />
              </div>
              <span className={`w-14 text-right tnum ${
                h.weight_pct <= 0 ? "text-ink-mute" : "font-medium"}`}>
                {fixed(h.weight_pct, 1)}%
              </span>
            </div>
          ))}
        </div>

        <Compare d={d} />
      </section>

      <section className="card p-3 flex flex-col gap-2">
        <h3 className="text-[13.5px] font-semibold">📈 历史表现</h3>
        <p className="label">
          用最终权重回看这段历史。优化本来就是在这段数据上做的，所以这条线一定好看 ——
          等权和{d.benchmark ? d.benchmark.label : "基准"}画在一起才读得出来。
        </p>
        <Curves d={d} />
      </section>

      <div className="grid gap-3 lg:grid-cols-2 items-start">
        <section className="card p-3 flex flex-col gap-2">
          <h3 className="text-[13.5px] font-semibold">📉 有效前沿</h3>
          <p className="label">
            每个风险水平下历史上能达到的最高收益。横轴波动、纵轴年化 ——
            看形状，别从上面挑点：纵轴是历史均值，而历史均值不重复。
          </p>
          <Frontier d={d} />
        </section>

        <section className="card p-3 flex flex-col gap-2">
          <h3 className="text-[13.5px] font-semibold">🔥 相关性矩阵</h3>
          <p className="label">
            两两之间的日收益相关性。整片深色说明这组股票其实是一个赌注，
            分散只是名义上的。
          </p>
          <CorrMatrix c={d.correlation} holdings={d.holdings} />
        </section>
      </div>
    </>
  );
}

function Compare({ d }: { d: PortfolioBuild }) {
  const row = (label: string, s: PortfolioBuild["stats"], strong?: boolean) => (
    <tr className={`border-t border-line ${strong ? "font-medium" : "text-ink-dim"}`}>
      <td className="py-1 pr-3">{label}</td>
      <td className="py-1 px-2 text-right tnum">{signed(s.ann_return_pct, 1, "%")}</td>
      <td className="py-1 px-2 text-right tnum">{fixed(s.ann_vol_pct, 1)}%</td>
      <td className="py-1 px-2 text-right tnum">{fixed(s.sharpe, 2)}</td>
      <td className="py-1 pl-2 text-right tnum">{fixed(s.max_drawdown_pct, 1)}%</td>
    </tr>
  );
  return (
    <table className="w-full text-[12.5px] border-collapse mt-1">
      <thead>
        <tr className="text-ink-mute">
          <th className="text-left font-normal pb-1 pr-3">组合</th>
          <th className="text-right font-normal pb-1 px-2">年化收益</th>
          <th className="text-right font-normal pb-1 px-2">年化波动</th>
          <th className="text-right font-normal pb-1 px-2"
            title="年化收益 ÷ 年化波动，未扣无风险利率">夏普</th>
          <th className="text-right font-normal pb-1 pl-2">最大回撤</th>
        </tr>
      </thead>
      <tbody>
        {row(d.method_label, d.stats, true)}
        {row("等权重（基准线）", d.equal_stats)}
      </tbody>
    </table>
  );
}

function SaveFund({ d, market, onSaved }: {
  d: PortfolioBuild; market: Market; onSaved: () => void;
}) {
  const [name, setName] = useState("");
  const [open, setOpen] = useState(false);
  const save = useMutation({
    mutationFn: () => api.saveFund({
      name: name.trim(),
      benchmark: d.benchmark?.label ?? null,
      holdings: d.holdings.filter((h) => h.weight_pct > 0)
        .map((h) => ({ t: h.t, weight_pct: h.weight_pct })),
    }),
    onSuccess: () => { setOpen(false); setName(""); onSaved(); },
  });

  if (!open) {
    return (
      <button onClick={() => setOpen(true)}
        className="ml-auto h-8 px-3 rounded-lg bg-sunken text-[12.5px] font-medium">
        💾 存为组合
      </button>
    );
  }
  return (
    <div className="ml-auto flex items-center gap-1.5">
      <input value={name} onChange={(e) => setName(e.target.value)}
        placeholder={`${market === "CN" ? "A股" : "美股"}组合名…`} autoFocus
        className="h-8 w-40 px-2 rounded-md bg-sunken text-[12.5px] outline-none
          focus:ring-2 focus:ring-cyan/40" />
      <button onClick={() => save.mutate()}
        disabled={!name.trim() || save.isPending}
        className="h-8 px-3 rounded-lg bg-cyan text-white text-[12.5px]
          font-semibold disabled:opacity-60">
        {save.isPending ? "保存中…" : "保存"}
      </button>
      <button onClick={() => setOpen(false)}
        className="h-8 px-2 rounded-lg bg-sunken text-[12.5px]">取消</button>
      {save.isError && (
        <span className="text-[12px] text-up">{(save.error as ApiError).message}</span>
      )}
    </div>
  );
}
