/**
 * 组合优化 — the ported mean-variance page.
 *
 * Default is maximum Sharpe, as the Streamlit page had it. Clicking a point
 * on the frontier switches to target-return mode and re-solves for the
 * minimum-variance weights that reach it: the curve is the control, not an
 * illustration beside a fixed answer.
 *
 * The assessment comes first in the results, as it did on the page — a
 * 0-100 score across five risk dimensions with its reasons spelled out,
 * before any of the charts.
 */

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";
import type { BuildMode, PortfolioBuild, StockRef } from "../../lib/types";
import { useSymbolSearch } from "../../lib/useSymbolSearch";
import { usePersistentState } from "../../lib/usePersistentState";
import { BUILD_KEY, signature, type CachedBuild } from "../../lib/buildCache";
import { Allocation, useAllocation } from "./Allocation";
import { AnalysisWindow, useAnalysisWindow } from "./AnalysisWindow";
import { Curves } from "./Curves";
import { Frontier } from "./Frontier";
import { CorrMatrix } from "./CorrMatrix";
import { Industries } from "./Industries";
import { CandidateBar, useCandidates, type CandidatesState } from "./Candidates";

const MAX_NAMES = 30;

/**
 * What "optimise" is being asked to mean.
 *
 * The three differ in which input they trust. Max Sharpe trusts the
 * historical means, which are the least stable thing in the problem.
 * Minimum variance ignores them entirely and minimises volatility. Risk
 * parity also ignores them, and instead equalises how much risk each holding
 * carries — which is not the same as equalising weights.
 */
const BUILD_MODES: { id: BuildMode; label: string; hint: string }[] = [
  { id: "max_sharpe", label: "最大夏普",
    hint: "每单位波动换来的超额收益最高。用到历史均值收益，而均值是最不稳定的那个输入。" },
  { id: "min_variance", label: "最小方差",
    hint: "只求波动最小，完全不看预期收益。前沿曲线最左端就是它。" },
  { id: "risk_parity", label: "风险平价",
    hint: "让每只股票承担同样多的风险，而不是同样多的仓位。也不预测收益，只用协方差。" },
  { id: "equal", label: "等权重",
    hint: "每只一样多，什么都不估计，单只上限对它不起作用。其他几种方法都得先赢过它才算有用。" },
];

export type Market = "CN" | "US";

const MARKETS: { id: Market; label: string; hint: string }[] = [
  { id: "CN", label: "A 股", hint: "对标沪深300" },
  { id: "US", label: "美股", hint: "对标标普500" },
];

export function Optimiser({ onSaved }: { onSaved: () => void }) {
  const [market, setMarket] = usePersistentState<Market>("assrs.pf.market", "CN");
  const [cnPicks, setCnPicks] = usePersistentState<StockRef[]>("assrs.pf.cn", []);
  const [usPicks, setUsPicks] = usePersistentState<StockRef[]>("assrs.pf.us", []);
  const picked = market === "CN" ? cnPicks : usPicks;
  const setPicked = market === "CN" ? setCnPicks : setUsPicks;

  const [mode, setMode] = usePersistentState<BuildMode>(
    "assrs.pf.mode", "max_sharpe");
  const [maxWeight, setMaxWeight] = usePersistentState<number>("assrs.pf.cap", 30);
  const [lookback, setLookback] = usePersistentState<number>("assrs.pf.lb", 242);
  const [duration, setDuration] = usePersistentState<number>("assrs.pf.dur", 1);
  const [rf, setRf] = usePersistentState<number>("assrs.pf.rf", 3);

  // Both markets pick a stretch of history on their own index; 回看 is what is
  // left when that index fails to load.
  const cnWin = useAnalysisWindow("CN", market === "CN");
  const usWin = useAnalysisWindow("US", market === "US");
  const win = market === "CN" ? cnWin : usWin;
  const useWindow = win.from != null && win.to != null;
  const showLookback = win.index.isError;

  const cands = useCandidates(market);

  const qc = useQueryClient();
  const sig = signature({ symbols: picked.map((p) => p.t),
                          maxWeight, lookback, duration, rf, mode,
                          window: useWindow ? `${win.from}~${win.to}` : undefined });

  const run = useMutation({
    mutationFn: (t: number | null) => api.portfolioBuild({
      symbols: picked.map((p) => p.t),
      // A frontier click always means target mode, whatever button is lit.
      mode: t === null ? mode : "target",
      target_return_pct: t === null ? null : Number((t * 100).toFixed(4)),
      max_weight_pct: maxWeight, lookback, duration, rf_pct: rf,
      start: useWindow ? win.from ?? null : null,
      end: useWindow ? win.to ?? null : null,
    }),
    onSuccess: (d) => qc.setQueryData<CachedBuild>(BUILD_KEY, { d, sig }),
  });

  // Cache-only: `enabled: false` means the fn never runs, so this is purely a
  // subscription to whatever the last successful build wrote.
  const kept = useQuery({
    queryKey: BUILD_KEY,
    queryFn: () => null as CachedBuild | null,
    enabled: false,
    staleTime: Infinity,
    gcTime: Infinity,
  }).data ?? null;

  const ready = picked.length >= 2 && (useWindow || showLookback);
  const go = (t: number | null) => run.mutate(t);

  return (
    <div className="flex flex-col gap-3">
      <section className="card p-3 flex flex-col gap-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1">
            {MARKETS.map((m) => (
              <button key={m.id} onClick={() => setMarket(m.id)} title={m.hint}
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
      </section>

      <AnalysisWindow w={win} picks={picked} rf={rf} />

      <section className="card p-3 flex flex-col gap-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="label">配置方法</span>
          <div className="flex items-center gap-1 flex-wrap">
            {BUILD_MODES.map((m) => (
              <button key={m.id} onClick={() => setMode(m.id)} title={m.hint}
                className={`h-8 px-3 rounded-lg text-[12.5px] font-medium transition-colors ${
                  mode === m.id ? "bg-elevated text-ink" : "text-ink-mute hover:text-ink"
                }`}>
                {m.label}
              </button>
            ))}
          </div>
          <span className="label flex-1 min-w-[200px] leading-snug">
            {BUILD_MODES.find((m) => m.id === mode)?.hint}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <label className="label flex items-center gap-1.5"
            title="任何一只的最大权重">
            单只上限
            <input type="range" min={5} max={100} step={5} value={maxWeight}
              onChange={(e) => setMaxWeight(Number(e.target.value))}
              className="w-24 accent-[var(--color-cyan)]" />
            <span className="tnum w-9">{maxWeight}%</span>
          </label>
          {showLookback ? (
            <label className="label flex items-center gap-1.5"
              title="协方差和预期收益用多少个交易日估计。242 ≈ A 股一年。">
              回看
              <select value={lookback} onChange={(e) => setLookback(Number(e.target.value))}
                className="h-7 px-1.5 rounded-md bg-sunken text-[12.5px] tnum outline-none">
                {[60, 90, 120, 180, 242].map((d) => <option key={d} value={d}>{d} 天</option>)}
              </select>
            </label>
          ) : useWindow && (
            <span className="label tnum" title="在上面的时间窗口里调整">
              窗口 {win.from} → {win.to}
            </span>
          )}
          <label className="label flex items-center gap-1.5"
            title="1=日线，5=周，20=月。只影响协方差矩阵；历史模拟和日度风险指标始终用日收益。">
            收益周期
            <input type="number" min={1} max={30} value={duration}
              onChange={(e) => setDuration(Math.max(1, Number(e.target.value)))}
              className="w-14 h-7 px-1.5 rounded-md bg-sunken text-[12.5px] tnum outline-none" />
          </label>
          <label className="label flex items-center gap-1.5" title="年化无风险利率，用于夏普">
            无风险
            <input type="number" min={0} max={10} step={0.5} value={rf}
              onChange={(e) => setRf(Number(e.target.value))}
              className="w-16 h-7 px-1.5 rounded-md bg-sunken text-[12.5px] tnum outline-none" />
            <span>%</span>
          </label>
          <button onClick={() => go(null)} disabled={!ready || run.isPending}
            className="ml-auto h-8 px-4 rounded-lg bg-cyan text-white text-[13px]
              font-semibold disabled:opacity-60">
            {run.isPending ? "计算中…"
              : ready ? `🚀 优化（${BUILD_MODES.find((m) => m.id === mode)?.label}）`
              : picked.length >= 2 ? "读取时间窗口…"
              : "先选两只以上"}
          </button>
        </div>

        {run.isError && (
          <p className="text-[12.5px] text-up">{(run.error as ApiError).message}</p>
        )}
      </section>

      {kept && (
        <Result d={kept.d} market={market} busy={run.isPending} cands={cands}
          stale={kept.sig !== sig} onPick={go} onSaved={onSaved}
          onSaveWeights={() => {
            // Scroll the save box into view: it lives in the assessment
            // card at the top, a long way from the editor.
            document.querySelector("[data-save-fund]")
              ?.scrollIntoView({ behavior: "smooth", block: "center" });
          }} />
      )}
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

const TONE: Record<string, string> = {
  good: "text-up border-up/40 bg-up/5",
  warn: "text-brand-ink border-brand-ink/40 bg-brand-ink/5",
  bad: "text-up border-up/50 bg-up/10",
};

function Result({ d, market, busy, stale, onPick, onSaved, onSaveWeights, cands }: {
  d: PortfolioBuild; market: Market; busy: boolean; stale: boolean;
  onPick: (t: number | null) => void; onSaved: () => void;
  onSaveWeights: () => void;
  cands: CandidatesState;
}) {
  const alloc = useAllocation(d);
  // The candidate last loaded or saved: 更新 writes over this one.
  const [activeId, setActiveId] = useState<string | null>(null);
  useEffect(() => { setActiveId(null); }, [d]);
  const { tweak } = alloc;
  const a = tweak?.assessment ?? d.assessment;
  const [saving, setSaving] = useState(false);
  // Everything the weights feed: faint once they are out of date, fainter
  // while the new numbers are on their way.
  const fade = alloc.weighing ? "opacity-40" : alloc.dirty ? "opacity-60" : "";
  return (
    <div className={`flex flex-col gap-3 transition-opacity ${
      busy ? "opacity-50" : ""}`}>
      {stale && (
        <p className="card px-3 py-2 text-[12.5px] text-brand-ink
          border border-brand-ink/40 bg-brand-ink/5">
          输入改过了 —— 下面还是上一次的结果，重新算一次才对得上。
        </p>
      )}
      {/* The assessment leads, as it did on the page. */}
      <section className={`card p-3 flex flex-col gap-2 border ${TONE[a.tone] ?? ""}`}>
        <div className={`flex flex-col gap-2 transition-opacity ${fade}`}>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="text-[15px] font-semibold">{a.verdict}</span>
            {tweak && <span className="label">按你的权重</span>}
            <span className="text-[13px] tnum font-semibold">{a.score}/{a.max}</span>
            <div className="flex-1 min-w-[120px] h-2 rounded-full bg-sunken overflow-hidden">
              <div className="h-full rounded-full bg-current"
                style={{ width: `${a.score}%` }} />
            </div>
          </div>
          <p className="text-[12.5px] text-ink leading-snug">{a.summary}</p>
          <div className="grid gap-1 sm:grid-cols-2 lg:grid-cols-3 text-[12px]">
            {a.strengths.map((x, i) => (
              <span key={`s${i}`} className="text-ink-dim">✅ {x}</span>
            ))}
            {a.notes.map((x, i) => (
              <span key={`n${i}`} className="text-ink-mute">📊 {x}</span>
            ))}
            {a.warnings.map((x, i) => (
              <span key={`w${i}`} className="text-up">⚠️ {x}</span>
            ))}
          </div>
        </div>
        <div className="flex justify-end" data-save-fund>
          <SaveFund d={d} market={market} onSaved={onSaved}
            holdings={alloc.changed ? alloc.holdings : d.holdings}
            blocked={alloc.balanced ? null : `权重合计 ${alloc.total.toFixed(2)}%，必须正好 100%`}
            open={saving} setOpen={setSaving} />
        </div>
      </section>

      <Allocation d={d} a={alloc} fade={fade}
        onSave={() => { setSaving(true); onSaveWeights(); }}
        candidates={<CandidateBar d={d} a={alloc} cands={cands}
          activeId={activeId} setActiveId={setActiveId} />} />

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 items-start">
        <section className={`card p-3 flex flex-col gap-2 transition-opacity ${fade}`}>
          <h3 className="text-[13.5px] font-semibold">📉 有效前沿 · 点选目标</h3>
          <Frontier d={d} onPick={onPick} busy={busy}
            custom={tweak && {
              vol_pct: tweak.ann_vol_pct,
              ann_return_pct: tweak.ann_return_pct,
            }} />
        </section>
        <div className="flex flex-col gap-3">
          <section className={`card p-3 flex flex-col gap-2 transition-opacity ${fade}`}>
            <h3 className="text-[13.5px] font-semibold">🏭 行业分布</h3>
            <Industries d={tweak?.industries ?? d.industries} />
          </section>
          <section className="card p-3 flex flex-col gap-2">
            <h3 className="text-[13.5px] font-semibold">🔥 相关性矩阵</h3>
            <CorrMatrix c={d.correlation} stocks={d.holdings} />
          </section>
        </div>
      </div>

      <section className={`card p-3 flex flex-col gap-2 transition-opacity ${fade}`}>
        <h3 className="text-[13.5px] font-semibold">📈 历史模拟</h3>
        <Curves d={d} tweak={tweak} cands={cands.list} activeId={activeId} />
      </section>
    </div>
  );
}

/**
 * Saving, either the optimiser's weights or the ones you edited by hand.
 *
 * `holdings` is what actually gets written: the weights in the panel, so
 * saving never silently reverts to the optimiser's answer.
 */
function SaveFund({ d, market, onSaved, holdings, blocked, open, setOpen }: {
  d: PortfolioBuild; market: Market; onSaved: () => void;
  holdings: { t: string; n: string; weight_pct: number }[];
  /** Why these weights cannot be saved yet, if they cannot. */
  blocked: string | null;
  open: boolean; setOpen: (v: boolean) => void;
}) {
  const [name, setName] = useState("");
  const edited = holdings !== d.holdings;
  const save = useMutation({
    mutationFn: () => api.saveFund({
      name: name.trim(),
      benchmark: d.benchmark?.label ?? null,
      holdings: holdings.filter((h) => h.weight_pct > 0)
        .map((h) => ({ t: h.t, weight_pct: h.weight_pct })),
    }),
    onSuccess: () => { setOpen(false); setName(""); onSaved(); },
  });

  if (!open) {
    return (
      <button onClick={() => setOpen(true)}
        className="h-8 px-3 rounded-lg bg-sunken text-[12.5px] font-medium text-ink">
        💾 存为组合
      </button>
    );
  }
  return (
    <div className="flex flex-wrap items-center justify-end gap-1.5">
      {edited && (
        <span className="text-[12px] text-brand-ink">存你改过的权重</span>
      )}
      <input value={name} onChange={(e) => setName(e.target.value)}
        placeholder={`${market === "CN" ? "A股" : "美股"}组合名…`} autoFocus
        className="h-8 w-40 px-2 rounded-md bg-sunken text-[12.5px] text-ink outline-none
          focus:ring-2 focus:ring-cyan/40" />
      {blocked && <span className="text-[12px] text-up">{blocked}</span>}
      <button onClick={() => save.mutate()}
        disabled={!name.trim() || save.isPending || blocked != null}
        className="h-8 px-3 rounded-lg bg-cyan text-white text-[12.5px]
          font-semibold disabled:opacity-60">
        {save.isPending ? "保存中…" : "保存"}
      </button>
      <button onClick={() => setOpen(false)}
        className="h-8 px-2 rounded-lg bg-sunken text-[12.5px] text-ink">取消</button>
      {save.isError && (
        <span className="text-[12px] text-up">{(save.error as ApiError).message}</span>
      )}
    </div>
  );
}
