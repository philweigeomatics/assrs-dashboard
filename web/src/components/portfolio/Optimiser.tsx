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

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";
import type { BuildMode, PortfolioBuild, StockRef } from "../../lib/types";
import { useSymbolSearch } from "../../lib/useSymbolSearch";
import { usePersistentState } from "../../lib/usePersistentState";
import { BUILD_KEY, signature, type CachedBuild } from "../../lib/buildCache";
import { fixed, signed } from "../../lib/format";
import { Curves } from "./Curves";
import { Frontier } from "./Frontier";
import { CorrMatrix } from "./CorrMatrix";
import { Industries } from "./Industries";
import { WeightEditor } from "./WeightEditor";
import type { WeighResult } from "../../lib/types";

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

  const qc = useQueryClient();
  const sig = signature({ symbols: picked.map((p) => p.t),
                          maxWeight, lookback, duration, rf, mode });

  const run = useMutation({
    mutationFn: (t: number | null) => api.portfolioBuild({
      symbols: picked.map((p) => p.t),
      // A frontier click always means target mode, whatever button is lit.
      mode: t === null ? mode : "target",
      target_return_pct: t === null ? null : Number((t * 100).toFixed(4)),
      max_weight_pct: maxWeight, lookback, duration, rf_pct: rf,
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

  const ready = picked.length >= 2;
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
          <label className="label flex items-center gap-1.5"
            title="协方差和预期收益用多少个交易日估计。242 ≈ A 股一年。">
            回看
            <select value={lookback} onChange={(e) => setLookback(Number(e.target.value))}
              className="h-7 px-1.5 rounded-md bg-sunken text-[12.5px] tnum outline-none">
              {[60, 90, 120, 180, 242].map((d) => <option key={d} value={d}>{d} 天</option>)}
            </select>
          </label>
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
              : "先选两只以上"}
          </button>
        </div>

        {run.isError && (
          <p className="text-[12.5px] text-up">{(run.error as ApiError).message}</p>
        )}
      </section>

      {kept && (
        <Result d={kept.d} market={market} busy={run.isPending}
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

function Result({ d, market, busy, stale, onPick, onSaved, onSaveWeights }: {
  d: PortfolioBuild; market: Market; busy: boolean; stale: boolean;
  onPick: (t: number | null) => void; onSaved: () => void;
  onSaveWeights: (h: { t: string; n: string; weight_pct: number }[]) => void;
}) {
  const a = d.assessment;
  const r = d.risk;
  const [tweak, setTweak] = useState<WeighResult | null>(null);
  // What "存为组合" writes: the optimiser's weights until the editor hands
  // over its own.
  const [edited, setEdited] = useState<
    { t: string; n: string; weight_pct: number }[] | null>(null);
  const [saving, setSaving] = useState(false);
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
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1"
          data-save-fund>
          <span className="text-[15px] font-semibold">{a.verdict}</span>
          <span className="text-[13px] tnum font-semibold">{a.score}/{a.max}</span>
          <div className="flex-1 min-w-[120px] h-2 rounded-full bg-sunken overflow-hidden">
            <div className="h-full rounded-full bg-current"
              style={{ width: `${a.score}%` }} />
          </div>
          <SaveFund d={d} market={market} onSaved={onSaved}
            holdings={edited ?? d.holdings}
            open={saving} setOpen={(v) => {
              setSaving(v);
              if (!v) setEdited(null);
            }} />
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
      </section>

      <section className="card p-3 flex flex-col gap-2.5">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h3 className="text-[13.5px] font-semibold">
            🎯 {d.mode === "target"
              ? `目标年化 ${fixed(d.target_return_pct, 1)}% 的最小方差配置`
              : `${d.mode_label}配置`}
          </h3>
          <span className="label">
            {d.from} → {d.to} · {d.lookback} 个交易日 · 上限 {d.max_weight_pct}%
            · 无风险 {d.rf_pct}%
            {d.duration > 1 && ` · 收益周期 ${d.duration} 天`}
          </span>
        </div>

        {d.missing.length > 0 && (
          <p className="text-[12px] text-up">读不到行情，已排除：{d.missing.join("、")}</p>
        )}

        <Weights d={d} />

        <table className="w-full text-[12.5px] border-collapse mt-1">
          <thead>
            <tr className="text-ink-mute">
              <th className="text-left font-normal pb-1 pr-3">组合</th>
              <th className="text-right font-normal pb-1 px-2">年化收益</th>
              <th className="text-right font-normal pb-1 px-2">年化波动</th>
              <th className="text-right font-normal pb-1 px-2">夏普</th>
              <th className="text-right font-normal pb-1 pl-2">最大回撤</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-t border-line font-medium">
              <td className="py-1 pr-3" title="优化器算出来的：把各股年化均值按权重加权，再用协方差矩阵算波动。假设每天都调回目标权重。">优化器预期（按权重加权）</td>
              <td className="py-1 px-2 text-right tnum">{signed(d.opt.ann_return_pct, 1, "%")}</td>
              <td className="py-1 px-2 text-right tnum">{fixed(d.opt.ann_vol_pct, 1)}%</td>
              <td className="py-1 px-2 text-right tnum">{fixed(d.opt.sharpe, 2)}</td>
              <td className="py-1 pl-2 text-right tnum text-ink-mute">—</td>
            </tr>
            <tr className="border-t border-line text-ink-dim">
              <td className="py-1 pr-3" title="同一组权重买入后一直持有，在这段真实行情上滚出来的结果。会复利，权重也会随涨跌漂移，所以和上面那行不一样 —— 只有这一行能算出最大回撤。">买入持有实测（同一段行情）</td>
              <td className="py-1 px-2 text-right tnum">{signed(d.stats.ann_return_pct, 1, "%")}</td>
              <td className="py-1 px-2 text-right tnum">{fixed(d.stats.ann_vol_pct, 1)}%</td>
              <td className="py-1 px-2 text-right tnum">{fixed(d.stats.sharpe, 2)}</td>
              <td className="py-1 pl-2 text-right tnum">{fixed(d.stats.max_drawdown_pct, 1)}%</td>
            </tr>
            <tr className="border-t border-line text-ink-dim">
              <td className="py-1 pr-3" title="每只一样多，同样买入持有。优化如果赢不过这一行，就没有产生价值。">等权买入持有（对照）</td>
              <td className="py-1 px-2 text-right tnum">{signed(d.equal_stats.ann_return_pct, 1, "%")}</td>
              <td className="py-1 px-2 text-right tnum">{fixed(d.equal_stats.ann_vol_pct, 1)}%</td>
              <td className="py-1 px-2 text-right tnum">{fixed(d.equal_stats.sharpe, 2)}</td>
              <td className="py-1 pl-2 text-right tnum">{fixed(d.equal_stats.max_drawdown_pct, 1)}%</td>
            </tr>
          </tbody>
        </table>

        <div className="grid gap-x-4 gap-y-1 sm:grid-cols-2 lg:grid-cols-5 text-[12px] mt-1">
          <Metric label="有效仓位数" v={fixed(r.enb, 2)}
            hint="1/Σw²。接近股票只数说明是真的分散，接近 1 说明是一个赌注。" />
          <Metric label="分散比" v={fixed(r.div_ratio, 2)}
            hint="加权平均波动 ÷ 组合波动。大于 1 才有分散化收益。" />
          <Metric label="VaR 95%" v={`${fixed(r.var_95_pct, 2)}%`}
            hint="95% 的交易日里，单日亏损不会超过这个数。" />
          <Metric label="CVaR 95%" v={`${fixed(r.cvar_95_pct, 2)}%`}
            hint="真的跌破 VaR 的那些天，平均亏这么多。" />
          <Metric label="尾部比 95%" v={fixed(r.tail_95, 2)}
            hint="上尾 ÷ 下尾。大于 1 表示极端行情里上行空间更大。" />
          <Metric label="VaR 99%" v={`${fixed(r.var_99_pct, 2)}%`} />
          <Metric label="CVaR 99%" v={`${fixed(r.cvar_99_pct, 2)}%`} />
          <Metric label="尾部比 99%" v={fixed(r.tail_99, 2)} />
          <Metric label="最差单日" v={`${fixed(r.worst_day_pct, 2)}%`} />
          <Metric label="最差 5 日均值" v={`${fixed(r.avg_worst5_pct, 2)}%`} />
        </div>
      </section>

      <div className="grid gap-3 lg:grid-cols-2 items-start">
        <section className="card p-3 flex flex-col gap-2">
          <h3 className="text-[13.5px] font-semibold">🏭 行业分布</h3>
          <Industries d={d.industries} />
        </section>
        <section className="card p-3 flex flex-col gap-2">
          <h3 className="text-[13.5px] font-semibold">⚖️ 调整权重</h3>
          <WeightEditor d={d} onResult={setTweak}
            onSave={(h) => { setEdited(h); setSaving(true); onSaveWeights(h); }} />
        </section>
      </div>

      <div className="grid gap-3 lg:grid-cols-2 items-start">
        <section className="card p-3 flex flex-col gap-2">
          <h3 className="text-[13.5px] font-semibold">📉 有效前沿 · 点选目标</h3>
          <Frontier d={d} onPick={onPick} busy={busy}
            custom={tweak && {
              vol_pct: tweak.ann_vol_pct,
              ann_return_pct: tweak.ann_return_pct,
            }} />
        </section>
        <section className="card p-3 flex flex-col gap-2">
          <h3 className="text-[13.5px] font-semibold">🔥 相关性矩阵</h3>
          <CorrMatrix c={d.correlation} holdings={d.holdings} />
        </section>
      </div>

      <section className="card p-3 flex flex-col gap-2">
        <h3 className="text-[13.5px] font-semibold">📈 历史模拟</h3>
        <p className="label">
          用最终权重回看这段历史。优化就是在这段数据上做的，所以这条线一定好看 ——
          等权和{d.benchmark ? d.benchmark.label : "基准"}画在一起才读得出来。
        </p>
        <Curves d={d} />
      </section>
    </div>
  );
}

/**
 * Weight against risk share, one row each.
 *
 * These come apart badly and the gap is the point. On six A-shares the
 * max-Sharpe book put 30% into one semiconductor name and that single
 * position carried 89% of the portfolio volatility — a number the weights
 * bar alone gives you no way to see.
 */
function Weights({ d }: { d: PortfolioBuild }) {
  const held = d.holdings.filter((h) => h.weight_pct > 0);
  const worst = held.reduce(
    (a, h) => (h.risk_pct > (a?.risk_pct ?? -1) ? h : a),
    null as PortfolioBuild["holdings"][number] | null);
  const lopsided = worst != null && worst.risk_pct - worst.weight_pct >= 15;

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2 text-[11px] text-ink-mute">
        <span className="w-28 shrink-0" />
        <span className="hidden md:block w-16 shrink-0" />
        <span className="flex-1">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-2 rounded-sm inline-block bg-cyan" />权重
            <span className="w-3 h-2 rounded-sm inline-block ml-2"
              style={{ background: "#f59e0b" }} />占风险
          </span>
        </span>
        <span className="w-14 text-right">权重</span>
        <span className="w-14 text-right">风险</span>
      </div>

      {d.holdings.map((h) => (
        <div key={h.t} className="flex items-center gap-2 text-[12.5px]">
          <span className="w-28 shrink-0 truncate" title={h.n}>{h.n}</span>
          <span className="hidden md:block w-16 shrink-0 font-mono tnum
            text-[11px] text-ink-mute">{h.t}</span>
          <div className="flex-1 flex flex-col gap-0.5 min-w-[60px]">
            <div className="h-2.5 rounded-sm bg-sunken overflow-hidden">
              <div className="h-full rounded-sm bg-cyan"
                style={{ width: `${Math.min(100, h.weight_pct)}%` }} />
            </div>
            <div className="h-2.5 rounded-sm bg-sunken overflow-hidden">
              <div className="h-full rounded-sm"
                style={{ width: `${Math.min(100, h.risk_pct)}%`,
                         background: "#f59e0b" }} />
            </div>
          </div>
          <span className={`w-14 text-right tnum ${
            h.weight_pct <= 0 ? "text-ink-mute" : "font-medium"}`}>
            {fixed(h.weight_pct, 1)}%
          </span>
          <span className={`w-14 text-right tnum ${
            h.risk_pct - h.weight_pct >= 15 ? "text-up font-medium"
              : "text-ink-dim"}`}>
            {fixed(h.risk_pct, 1)}%
          </span>
        </div>
      ))}

      {d.parity ? <Parity p={d.parity} /> : lopsided && worst && (
        <p className="text-[12px] text-brand-ink leading-snug">
          ⚠️ {worst.n} 只占 {fixed(worst.weight_pct, 1)}% 的仓位，
          却承担了 {fixed(worst.risk_pct, 1)}% 的组合波动 ——
          仓位分散不等于风险分散。想让每只承担一样的风险，选「风险平价」。
        </p>
      )}
    </div>
  );
}

/** Whether risk parity actually got there — the cap can make it unreachable. */
function Parity({ p }: { p: NonNullable<PortfolioBuild["parity"]> }) {
  if (p.reached) {
    return (
      <p className="text-[12px] text-up leading-snug">
        ✅ 每只都承担 {fixed(p.equal_pct, 1)}% 的组合波动，风险真的被均分了。
        注意权重并不相等 —— 波动小的那只本来就该拿得多。
      </p>
    );
  }
  return (
    <p className="text-[12px] text-brand-ink leading-snug">
      ⚠️ 单只上限挡住了完全的风险平价：目标是每只 {fixed(p.equal_pct, 1)}%，
      实际落在 {fixed(p.min_pct, 1)}%–{fixed(p.max_pct, 1)}%（相差 {fixed(p.spread_pp, 1)}pp）。
      把上限放宽可以更接近。
    </p>
  );
}

function Metric({ label, v, hint }: { label: string; v: string; hint?: string }) {
  return (
    <span className="flex items-baseline gap-1.5" title={hint}>
      <span className="text-ink-mute">{label}</span>
      {hint && <span className="text-ink-mute text-[10px]">ⓘ</span>}
      <span className="ml-auto tnum font-medium">{v}</span>
    </span>
  );
}

/**
 * Saving, either the optimiser's weights or the ones you edited by hand.
 *
 * `holdings` is what actually gets written — the editor passes its own, so
 * "存为组合" down there saves what is on screen rather than silently
 * reverting to the optimiser's answer.
 */
function SaveFund({ d, market, onSaved, holdings, open, setOpen }: {
  d: PortfolioBuild; market: Market; onSaved: () => void;
  holdings: { t: string; n: string; weight_pct: number }[];
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
        className="ml-auto h-8 px-3 rounded-lg bg-sunken text-[12.5px] font-medium text-ink">
        💾 存为组合
      </button>
    );
  }
  return (
    <div className="ml-auto flex items-center gap-1.5">
      {edited && (
        <span className="text-[12px] text-brand-ink">存你改过的权重</span>
      )}
      <input value={name} onChange={(e) => setName(e.target.value)}
        placeholder={`${market === "CN" ? "A股" : "美股"}组合名…`} autoFocus
        className="h-8 w-40 px-2 rounded-md bg-sunken text-[12.5px] text-ink outline-none
          focus:ring-2 focus:ring-cyan/40" />
      <button onClick={() => save.mutate()} disabled={!name.trim() || save.isPending}
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
