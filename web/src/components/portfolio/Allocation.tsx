/**
 * 🎯 配置 — the weights, what they carry, and what they earn, in one panel.
 *
 * The weights here are the controls. Moving one changes nothing else until
 * 计算 is pressed: a full re-pricing is a server round trip, and numbers that
 * jump on every slider tick are harder to read than numbers that wait. So
 * while the draft differs from what was last calculated, everything derived
 * from it is dimmed and says so, because on screen it is now out of date.
 *
 * The total is reported, never silently normalised. Weights summing to 94%
 * is a thing you need to see, not something to quietly fix — and saving is
 * blocked until it is exactly 100%, the same rule the Streamlit page had.
 */

import { useEffect, useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";
import type { PortfolioBuild, WeighResult } from "../../lib/types";
import { fixed, signed } from "../../lib/format";
import {
  equalWeights, isBalanced, sameWeights, sumOf, weightsOf, type WeightDraft,
} from "../../lib/weightDraft";
import { NumberInput } from "../NumberInput";

/**
 * The draft, and the calculated result it may or may not still match.
 *
 * It lives above the panel because the score, 行业分布, the frontier and
 * 历史模拟 all show the same calculation.
 */
export function useAllocation(d: PortfolioBuild) {
  const start = useMemo(() => weightsOf(d.holdings), [d.holdings]);
  const [w, setW] = useState<WeightDraft>(start);
  const [tweak, setTweak] = useState<WeighResult | null>(null);
  // Which weights the numbers on screen belong to.
  const [calculated, setCalculated] = useState<WeightDraft>(start);

  const run = useMutation({
    mutationFn: (weights: WeightDraft) => api.portfolioWeigh({
      symbols: d.holdings.map((h) => h.t), weights,
      max_weight_pct: d.max_weight_pct, lookback: d.lookback,
      duration: d.duration, rf_pct: d.rf_pct,
      start: d.window ? d.from : null,
      end: d.window ? d.to : null,
    }),
  });

  // Back to the optimiser's own numbers, which need no calculating. Resetting
  // the mutation also drops the answer to a request still in flight.
  const backToStart = () => {
    run.reset();
    setTweak(null);
    setCalculated(start);
  };
  // A fresh optimisation replaces the draft: the old numbers belonged to a
  // different answer.
  useEffect(() => {
    setW(start);
    backToStart();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [start]);

  const total = sumOf(w);
  return {
    w,
    tweak,
    total,
    balanced: isBalanced(w),
    /** The draft no longer matches the numbers on screen. */
    dirty: !sameWeights(w, calculated),
    /** The draft is not the optimiser's answer. */
    changed: !sameWeights(w, start),
    weighing: run.isPending,
    error: run.isError ? (run.error as ApiError).message : null,
    holdings: d.holdings
      .filter((h) => (w[h.t] ?? 0) > 0)
      .map((h) => ({ t: h.t, n: h.n, weight_pct: w[h.t] ?? 0 })),
    set: (t: string, v: number) =>
      setW((prev) => ({ ...prev, [t]: Math.max(0, Math.min(100, v)) })),
    spread: () => setW(equalWeights(d.holdings.map((h) => h.t))),
    restore: () => { setW(start); backToStart(); },
    calculate: () => {
      if (total <= 0 || run.isPending) return;
      const sent = { ...w };
      if (sameWeights(sent, start)) { backToStart(); return; }
      run.mutate(sent, {
        onSuccess: (res) => { setTweak(res); setCalculated(sent); },
      });
    },
  };
}

export type AllocationState = ReturnType<typeof useAllocation>;

type Risk = PortfolioBuild["risk"];

const METRICS: { label: string; hint?: string; get: (r: Risk) => string }[] = [
  { label: "有效仓位数", get: (r) => fixed(r.enb, 2),
    hint: "1/Σw²。接近股票只数说明是真的分散，接近 1 说明是一个赌注。" },
  { label: "分散比", get: (r) => fixed(r.div_ratio, 2),
    hint: "加权平均波动 ÷ 组合波动。大于 1 才有分散化收益。" },
  { label: "VaR 95%", get: (r) => `${fixed(r.var_95_pct, 2)}%`,
    hint: "95% 的交易日里，单日亏损不会超过这个数。" },
  { label: "CVaR 95%", get: (r) => `${fixed(r.cvar_95_pct, 2)}%`,
    hint: "真的跌破 VaR 的那些天，平均亏这么多。" },
  { label: "尾部比 95%", get: (r) => fixed(r.tail_95, 2),
    hint: "上尾 ÷ 下尾。大于 1 表示极端行情里上行空间更大。" },
  { label: "VaR 99%", get: (r) => `${fixed(r.var_99_pct, 2)}%` },
  { label: "CVaR 99%", get: (r) => `${fixed(r.cvar_99_pct, 2)}%` },
  { label: "尾部比 99%", get: (r) => fixed(r.tail_99, 2) },
  { label: "最差单日", get: (r) => `${fixed(r.worst_day_pct, 2)}%` },
  { label: "最差 5 日均值", get: (r) => `${fixed(r.avg_worst5_pct, 2)}%` },
];

export function Allocation({ d, a, onSave, fade }: {
  d: PortfolioBuild;
  a: AllocationState;
  onSave: () => void;
  /** Dims what the draft has made out of date. */
  fade: string;
}) {
  const { tweak } = a;
  const name = d.mode === "target" ? "目标组合" : d.mode_label;
  const risk = tweak?.risk ?? d.risk;

  type Row = {
    label: string; title?: string; strong?: boolean;
    ret: number | null; vol: number | null; sharpe: number | null;
    dd: number | null;
  };
  const expected = (o: { ann_return_pct: number; ann_vol_pct: number; sharpe: number | null }) => ({
    ret: o.ann_return_pct, vol: o.ann_vol_pct, sharpe: o.sharpe, dd: null,
  });
  const realised = (s: PortfolioBuild["stats"]) => ({
    ret: s.ann_return_pct, vol: s.ann_vol_pct, sharpe: s.sharpe, dd: s.max_drawdown_pct,
  });
  const expectedTitle = "把各股年化均值按权重加权，再用协方差矩阵算波动。假设每天都调回目标权重。";
  const equalRow: Row[] = d.mode === "equal" ? [] : [{
    label: "等权买入持有（对照）", ...realised(d.equal_stats),
    title: "每只一样多，同样买入持有。优化如果赢不过这一行，就没有产生价值。",
  }];
  const rows: Row[] = tweak
    ? [
        { label: "你的权重 · 预期（按权重加权）", strong: true, title: expectedTitle, ...expected(tweak) },
        { label: "你的权重 · 实测（同一段行情）", strong: true, ...realised(tweak.realised),
          title: "这组权重在这段真实行情上逐日算出来的结果。" },
        { label: `${name} · 预期（按权重加权）`, title: expectedTitle, ...expected(d.opt) },
        { label: `${name} · 实测（同一段行情）`, ...realised(d.stats),
          title: "原来那组权重在这段真实行情上逐日算出来的结果。" },
        ...equalRow,
      ]
    : [
        { label: d.mode === "equal" ? "预期（按权重加权）" : "优化器预期（按权重加权）", strong: true,
          title: `${d.mode === "equal" ? "" : "优化器算出来的："}${expectedTitle}`, ...expected(d.opt) },
        { label: "买入持有实测（同一段行情）", ...realised(d.stats),
          title: "同一组权重买入后一直持有，在这段真实行情上滚出来的结果。会复利，权重也会随涨跌漂移，所以和上面那行不一样 —— 只有这一行能算出最大回撤。" },
        ...equalRow,
      ];

  return (
    <section className="card p-3 flex flex-col gap-2.5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="text-[13.5px] font-semibold">
          🎯 {tweak
            ? `你的权重（从${name}调整）`
            : d.mode === "target"
              ? `目标年化 ${fixed(d.target_return_pct, 1)}% 的最小方差配置`
              : `${d.mode_label}配置`}
        </h3>
        <span className="label">
          {d.from} → {d.to} · {d.lookback} 个交易日
          {d.mode !== "equal" && ` · 上限 ${d.max_weight_pct}%`}
          {` · 无风险 ${d.rf_pct}%`}
          {d.duration > 1 && ` · 收益周期 ${d.duration} 天`}
        </span>
      </div>

      {d.missing.length > 0 && (
        <p className="text-[12px] text-up">读不到行情，已排除：{d.missing.join("、")}</p>
      )}
      {d.limited_by && (
        <p className="text-[12px] text-brand-ink leading-snug">
          {d.limited_by.n} 从 {d.limited_by.from} 才有行情，每只股票都要同时有价，
          所以实际从这天开始算，比你选的窗口短。
        </p>
      )}

      <div className={a.weighing ? "opacity-40 transition-opacity" : "transition-opacity"}>
        <WeightRows d={d} a={a} name={name} />
      </div>

      {/* Stays on screen while a long list of names is being adjusted. */}
      <div className="sticky bottom-0 z-10 -mx-3 px-3 py-2 flex flex-col gap-1.5
        border-y border-line" style={{ background: "var(--color-panel)" }}>
        <div className="flex flex-wrap items-center gap-2">
          <span className={`text-[12.5px] tnum font-medium ${
            a.balanced ? "text-ink" : "text-brand-ink"}`}>
            合计 {fixed(a.total, 2)}%
            {!a.balanced && <span className="font-normal"> · 存为组合必须正好 100%</span>}
          </span>
          <button onClick={a.spread}
            className="h-7 px-2 rounded-md bg-sunken text-[12px]">等权</button>
          <button onClick={a.restore} disabled={!a.changed && !tweak}
            className="h-7 px-2 rounded-md bg-sunken text-[12px] disabled:opacity-50">
            还原
          </button>
          <button onClick={a.calculate}
            disabled={!a.dirty || a.weighing || a.total <= 0}
            className="ml-auto h-8 px-4 rounded-lg bg-cyan text-white text-[13px]
              font-semibold disabled:opacity-50">
            {a.weighing ? "计算中…" : "计算"}
          </button>
          <button onClick={onSave} disabled={!a.balanced}
            title={a.balanced ? "存成组合" : "权重合计必须正好 100%"}
            className="h-8 px-3 rounded-lg bg-sunken text-[12.5px] font-medium
              disabled:opacity-50">
            💾 存为组合
          </button>
        </div>

        {a.weighing ? (
          <p className="text-[12.5px] font-medium text-cyan">⏳ 按新的权重计算中，下面的数字马上更新…</p>
        ) : a.dirty ? (
          <p className="text-[12.5px] leading-snug rounded-md px-2 py-1 text-brand-ink
            border border-brand-ink/40 bg-brand-ink/5">
            ⚠️ 权重已调整 —— 风险占比、收益、风险指标、评分、行业分布、前沿图和历史模拟
            都还是上一次算的，现在不准。点「计算」重新算。
          </p>
        ) : tweak && (
          <p className="label leading-snug">
            现在显示的是你的权重。橙点是它在前沿图上的位置，离曲线越远，
            说明同样的收益本可以用更小的波动拿到。
            {!tweak.balanced && ` 合计 ${fixed(tweak.sum_pct, 2)}%，没有归一，差的部分按现金算。`}
          </p>
        )}
        {a.error && <p className="text-[12.5px] text-up">{a.error}</p>}
      </div>

      <div className={`flex flex-col gap-2.5 transition-opacity ${fade}`}>
        <table className="w-full text-[12.5px] border-collapse">
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
            {rows.map((r) => (
              <tr key={r.label} className={`border-t border-line ${
                r.strong ? "font-medium" : "text-ink-dim"}`}>
                <td className="py-1 pr-3" title={r.title}>{r.label}</td>
                <td className="py-1 px-2 text-right tnum">{signed(r.ret, 1, "%")}</td>
                <td className="py-1 px-2 text-right tnum">{fixed(r.vol, 1)}%</td>
                <td className="py-1 px-2 text-right tnum">{fixed(r.sharpe, 2)}</td>
                <td className={`py-1 pl-2 text-right tnum ${r.dd == null ? "text-ink-mute" : ""}`}>
                  {r.dd == null ? "—" : `${fixed(r.dd, 1)}%`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="grid gap-x-4 gap-y-1 sm:grid-cols-2 lg:grid-cols-5 text-[12px]">
          {METRICS.map((m) => (
            <Metric key={m.label} label={m.label} hint={m.hint} v={m.get(risk)}
              was={tweak ? m.get(d.risk) : undefined} />
          ))}
        </div>
      </div>
    </section>
  );
}

/**
 * Weight against risk share, one row each, with the weight as the control.
 *
 * These come apart badly and the gap is the point. On six A-shares the
 * max-Sharpe book put 30% into one semiconductor name and that single
 * position carried 89% of the portfolio volatility — a number the weights
 * alone give you no way to see. The slider and the risk bar share a scale,
 * so the distance between them is that gap.
 */
function WeightRows({ d, a, name }: { d: PortfolioBuild; a: AllocationState; name: string }) {
  // The risk numbers belong to the last calculation, not the draft.
  const { tweak } = a;
  const calc = new Map<string, { weight_pct: number; risk_pct: number }>(
    (tweak ? tweak.holdings : d.holdings).map((h) => [h.t, h]));
  const rows = d.holdings.map((h) => ({
    ...h,
    held: calc.get(h.t)?.weight_pct ?? 0,
    risk: calc.get(h.t)?.risk_pct ?? 0,
    now: a.w[h.t] ?? 0,
  }));
  const worst = rows.reduce<(typeof rows)[number] | null>(
    (x, h) => (h.held > 0 && h.risk > (x?.risk ?? -1) ? h : x), null);
  const lopsided = worst != null && worst.risk - worst.held >= 15;
  const stale = a.dirty ? "opacity-50" : "";

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2 text-[11px] text-ink-mute">
        <span className="w-20 sm:w-28 shrink-0" />
        <span className="hidden md:block w-16 shrink-0" />
        <span className="hidden lg:block w-14 shrink-0 text-right" title="这只股票单独持有，这段时间的年化收益">年化</span>
        <span className="hidden lg:block w-12 shrink-0 text-right" title="这只股票单独持有，这段时间的年化波动">波动</span>
        <span className="hidden lg:block w-10 shrink-0 text-right mr-2" title="这只股票单独持有，这段时间的夏普">夏普</span>
        <span className="flex-1">
          <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
            <span className="w-3 h-2 rounded-sm inline-block bg-cyan" />权重
            <span className="w-3 h-2 rounded-sm inline-block ml-2"
              style={{ background: "#f59e0b" }} />占风险
          </span>
        </span>
        <span className="w-16 text-right shrink-0">权重 %</span>
        <span className="w-14 text-right shrink-0">风险</span>
        <span className="hidden sm:block w-14 text-right shrink-0">变化</span>
      </div>

      {rows.map((h) => {
        const moved = h.now - h.weight_pct;
        const solo = h.solo;
        return (
          <div key={h.t} className="flex items-center gap-2 text-[12.5px]">
            <div className="w-20 sm:w-28 shrink-0 min-w-0">
              <span className="block truncate" title={h.n}>{h.n}</span>
              <span className="lg:hidden block label tnum truncate">
                年化 {signed(solo.ann_return_pct, 0, "%")} · 夏普 {fixed(solo.sharpe, 1)}
              </span>
            </div>
            <span className="hidden md:block w-16 shrink-0 font-mono tnum
              text-[11px] text-ink-mute">{h.t}</span>
            <span className={`hidden lg:block w-14 shrink-0 text-right tnum text-[12px] ${
              (solo.ann_return_pct ?? 0) >= 0 ? "text-up" : "text-down"}`}>
              {signed(solo.ann_return_pct, 1, "%")}
            </span>
            <span className="hidden lg:block w-12 shrink-0 text-right tnum text-[12px] text-ink-dim">
              {fixed(solo.ann_vol_pct, 1)}%
            </span>
            <span className="hidden lg:block w-10 shrink-0 text-right tnum text-[12px] text-ink-dim mr-2">
              {fixed(solo.sharpe, 2)}
            </span>
            <div className="flex-1 flex flex-col gap-0.5 min-w-[60px]">
              <input type="range" min={0} max={100} step={0.5} value={h.now}
                onChange={(e) => a.set(h.t, Number(e.target.value))}
                aria-label={`${h.n} 权重`}
                className="block w-full h-3.5 m-0 accent-[var(--color-cyan)]" />
              <div className={`h-2 rounded-sm bg-sunken overflow-hidden transition-opacity ${stale}`}>
                <div className="h-full rounded-sm"
                  style={{ width: `${Math.min(100, h.risk)}%`, background: "#f59e0b" }} />
              </div>
            </div>
            <NumberInput value={h.now} set={(v) => a.set(h.t, v)} step={0.5} max={100}
              label={`${h.n} 权重百分比`}
              className="w-16 h-7 px-1.5 rounded-md bg-sunken text-[12.5px] tnum font-medium
                text-right outline-none shrink-0 focus:ring-2 focus:ring-cyan/40" />
            <span className={`w-14 text-right tnum shrink-0 transition-opacity ${stale} ${
              h.risk - h.held >= 15 ? "text-up font-medium" : "text-ink-dim"}`}>
              {fixed(h.risk, 1)}%
            </span>
            <span className={`hidden sm:block w-14 text-right tnum text-[11.5px] shrink-0 ${
              Math.abs(moved) < 0.05 ? "text-ink-mute"
                : moved > 0 ? "text-up" : "text-down"}`}
              title={`相对${name}的变化`}>
              {Math.abs(moved) < 0.05 ? "—" : signed(moved, 1, "pp")}
            </span>
          </div>
        );
      })}

      {!a.dirty && (d.parity && !tweak ? <Parity p={d.parity} /> : lopsided && worst && (
        <p className="text-[12px] text-brand-ink leading-snug">
          ⚠️ {worst.n} 只占 {fixed(worst.held, 1)}% 的仓位，
          却承担了 {fixed(worst.risk, 1)}% 的组合波动 ——
          仓位分散不等于风险分散。想让每只承担一样的风险，选「风险平价」。
        </p>
      ))}
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

function Metric({ label, v, hint, was }: {
  label: string; v: string; hint?: string;
  /** The optimiser's figure, once the weights are your own. */
  was?: string;
}) {
  return (
    <span className="flex items-baseline gap-1.5" title={hint}>
      <span className="text-ink-mute">{label}</span>
      {hint && <span className="text-ink-mute text-[10px]">ⓘ</span>}
      <span className="ml-auto tnum font-medium">{v}</span>
      {was && was !== v && (
        <span className="tnum text-[10.5px] text-ink-mute">原 {was}</span>
      )}
    </span>
  );
}
