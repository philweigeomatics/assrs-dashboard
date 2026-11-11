/**
 * 历史模拟: the optimiser's weights, your weights, equal weight and the index
 * on one rebased axis.
 *
 * Several lines rather than one, because the portfolio's own line is
 * unfalsifiable on its own: it was fitted to this history. What the reader
 * needs is the distance between it and the things it has to beat.
 *
 * The range is separate from the optimiser's 回看 on purpose. 回看 decides
 * which days the weights are fitted to, so lengthening it would change the
 * weights. Here the weights stay as they are and only the replay gets longer,
 * and the days left of the dashed line are ones the optimiser never saw.
 *
 * Saved 候选 join them, each in its own colour, and the table underneath
 * scores every line over exactly the days drawn — that table is where
 * candidates actually get compared.
 */

import { useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";
import type { PortfolioBuild, WeighResult } from "../../lib/types";
import { fixed, signed } from "../../lib/format";
import { usePersistentState } from "../../lib/usePersistentState";
import {
  colourOf, curveStats, matching, type Candidate, type CurveStats,
} from "../../lib/candidates";
import { weightsOf } from "../../lib/weightDraft";

const VW = 1000;
const VH = 190;

/** The same amber as the 你的权重 dot on the frontier. */
const MINE = "#f59e0b";

const RANGES = [
  { years: 0, label: "优化窗口" },
  { years: 1, label: "1年" },
  { years: 2, label: "2年" },
  { years: 3, label: "3年" },
  { years: 5, label: "5年" },
  { years: 10, label: "10年" },
];

const pcts = (rows: { t: string; weight_pct: number }[]) =>
  Object.fromEntries(rows.filter((h) => h.weight_pct > 0).map((h) => [h.t, h.weight_pct]));

export function Curves({ d, tweak, cands = [], activeId = null, view = null }: {
  d: PortfolioBuild; tweak: WeighResult | null;
  cands?: Candidate[]; activeId?: string | null;
  /** The 时间窗口 selected now, when it differs from the one the weights were fitted on. */
  view?: { from: string; to: string } | null;
}) {
  const [years, setYears] = usePersistentState<number>("assrs.pf.sim.years", 0);
  const [hidden, setHidden] = useState<Set<string>>(() => new Set());
  // A candidate may hold a name the current list does not: it is priced too.
  const symbols = [...new Set([...d.holdings.map((h) => h.t),
    ...cands.flatMap((c) => Object.keys(c.weights))])];
  const weights = pcts(d.holdings);
  // 时间窗口 follows the window picked above. Moved away from the one the
  // weights were fitted on, every line is replayed over the new stretch with
  // the weights unchanged — which is the test that matters.
  const moved = years === 0 && view != null;
  const mine = tweak && (years > 0 || moved) ? pcts(tweak.holdings) : null;
  const book = cands.length ? Object.fromEntries(cands.map((c) => [c.id, c.weights])) : null;
  // Over the optimiser's own window the other lines are already in hand, so
  // the request is only for the candidates, replayed over exactly those days.
  const span = years > 0 ? { start: null, end: null }
    : moved ? { start: view.from, end: view.to }
    : { start: d.from, end: d.to };

  const sim = useQuery({
    queryKey: ["portfolio", "simulate", symbols, weights, mine, years, span, book],
    queryFn: () => api.portfolioSimulate({
      symbols, weights, mine, years, ...span, candidates: book }),
    enabled: years > 0 || moved || book != null,
    staleTime: 10 * 60_000,
    retry: false,
    placeholderData: keepPreviousData,
  });
  const got = years > 0 || moved || book != null ? sim.data ?? null : null;
  const long = years > 0 || moved ? got : null;
  // A placeholder from a longer range is rebased to a different day, so its
  // candidate lines cannot be laid on the window's.
  const candResp = long ? got : sim.isPlaceholderData ? null : got;

  const dates = long?.dates ?? d.dates;
  const opt = long?.curve ?? d.curve;
  const equal = long?.equal_curve ?? d.equal_curve;
  const bench = long ? long.benchmark : d.benchmark;
  // By date, not by position: the two answers are separate requests, and a
  // session can close between them.
  let own: number[] | null = long?.mine_curve ?? null;
  if (!long && tweak) {
    const at = new Map(tweak.dates.map((t, i) => [t, tweak.curve[i]]));
    own = d.dates.map((t) => at.get(t) ?? NaN);
  }

  // Equal weight fits nothing, so it has no comparison line to duplicate and
  // no window its weights came from.
  const isEqual = d.mode === "equal";

  // When the draft is exactly a saved candidate, that candidate's line is it.
  const active = cands.find((c) => c.id === activeId) ?? null;
  const mineIs = tweak ? matching(cands, weightsOf(tweak.holdings)) : null;
  if (mineIs) own = null;
  const mineLabel = active ? `${active.name}（改动中）` : "你的权重";

  type Line = { key: string; label: string; values: number[]; colour: string; waiting?: boolean };
  const candLines: Line[] = cands.map((c) => {
    const line = candResp?.candidates?.[c.id];
    let values: number[] = [];
    if (line && candResp) {
      const at = new Map(candResp.dates.map((t, i) => [t, line[i]]));
      values = dates.map((t) => at.get(t) ?? NaN);
    }
    return { key: `c:${c.id}`, label: c.name, values, colour: colourOf(cands, c.id),
             waiting: !line };
  });

  // Likewise the optimiser's own answer, once it has been saved under a name.
  const optIs = matching(cands, weights);
  // A candidate saved from an earlier run often carries the method's name.
  const method = d.mode === "target" ? "目标组合" : d.mode_label;
  const optLabel = cands.some((c) => c.name === method) ? `${method}（本次优化）` : method;

  const series: Line[] = [
    ...(optIs ? [] : [{ key: "opt", label: optLabel, values: opt, colour: "var(--color-cyan)" }]),
    ...(own
      ? [{ key: "mine", label: tweak && !tweak.balanced ? `${mineLabel}（合计 ${tweak.sum_pct}%）` : mineLabel, values: own, colour: MINE }]
      : []),
    ...candLines,
    ...(isEqual ? [] : [{ key: "eq", label: "等权重", values: equal, colour: "#a855f7" }]),
    ...(bench
      ? [{
          key: "bm", label: bench.label,
          values: bench.curve.map((v) => (v == null ? NaN : v)),
          colour: "var(--color-ink-mute)",
        }]
      : []),
  ];
  const shown = series.filter((s) => !hidden.has(s.key));
  // Yours and the candidates are drawn last, so they stay visible where they
  // cross the others.
  const order = (k: string) => (k === "mine" ? 2 : k.startsWith("c:") ? 1 : 0);
  const drawn = [...shown].sort((a, b) => order(a.key) - order(b.key));
  const toggle = (k: string) => setHidden((h) => {
    const next = new Set(h);
    if (next.has(k)) next.delete(k); else next.add(k);
    return next;
  });

  let lo = Infinity;
  let hi = -Infinity;
  for (const s of shown) {
    for (const v of s.values) {
      if (!Number.isFinite(v)) continue;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  if (!Number.isFinite(lo)) { lo = 0; hi = 0; }
  const pad = Math.max(1, (hi - lo) * 0.08);
  const top = hi + pad;
  const bottom = lo - pad;

  const n = dates.length;
  const x = (i: number) => (n < 2 ? 0 : (i / (n - 1)) * VW);
  const y = (v: number) => VH - ((v - bottom) / (top - bottom)) * VH;
  const path = (values: number[]) => {
    let out = "";
    let pen = false;
    values.forEach((v, i) => {
      if (!Number.isFinite(v)) { pen = false; return; }
      out += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`;
      pen = true;
    });
    return out;
  };

  // Which of the days drawn the weights were fitted on: shaded, because the
  // lines are bound to look good there and nowhere else is that promised.
  let f0 = -1;
  let f1 = -1;
  if (long && !isEqual) {
    f0 = dates.findIndex((t) => t >= d.from);
    for (let i = n - 1; i >= 0; i--) if (dates[i]! <= d.to) { f1 = i; break; }
  }
  const overlap = f0 >= 0 && f1 >= f0;
  const inside = overlap && f0 === 0 && f1 === n - 1;
  const shade = overlap && !inside;
  const benchLabel = d.benchmark ? d.benchmark.label : "基准";

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-1">
        {RANGES.map((r) => (
          <button key={r.years} onClick={() => setYears(r.years)}
            title={r.years ? `权重不变，回放最近 ${r.years} 年`
              : d.window ? "跟着上面选中的时间窗口走" : "算权重用的那段行情"}
            className={`h-7 px-2.5 rounded-md text-[12px] font-medium transition-colors ${
              years === r.years ? "bg-elevated text-ink" : "text-ink-mute hover:text-ink"}`}>
            {r.years === 0 && d.window ? "时间窗口" : r.label}
          </button>
        ))}
        {sim.isFetching && (
          <span className="label ml-1">
            {years > 0 ? `读取 ${years} 年行情…` : moved ? "按新窗口回放…" : "计算候选…"}
          </span>
        )}
      </div>

      <p className="label leading-snug">
        {years === 0 && !moved
          ? isEqual
            ? `每只一样多，权重不是从这段行情里算出来的，所以这条线可以直接和${benchLabel}比；不过股票本身是今天选的，仍然带着后见之明。`
            : `用最终权重回看这段历史。优化就是在这段数据上做的，所以这条线一定好看 —— 等权和${benchLabel}画在一起才读得出来。`
          : `权重不变，${moved ? `按上面选中的 ${view.from} → ${view.to} 回放` : "只把回放拉长"}。${
              isEqual ? ""
              : inside ? `这段全在算权重用的 ${d.from} → ${d.to} 里面，所以本次优化的线一定好看；从别的窗口存下来的候选不受这个影响。`
              : overlap ? `浅色那段是算权重用的 ${d.from} → ${d.to}；其余的行情优化器没见过，更能说明这组权重的成色，不过股票本身是今天选的，仍然带着后见之明。`
              : `这段行情完全在算权重用的 ${d.from} → ${d.to} 之外，优化器一天也没见过 —— 这是对这组权重最干净的检验，不过股票本身是今天选的，仍然带着后见之明。`}`}
        {" "}每天按同一组权重计算，从起点的 0% 复利；股票含分红，{benchLabel}是价格指数、不含分红。
        {!tweak && !cands.length && " 在上面的配置里改权重、点「计算」，这里会多一条你的权重的线；存成候选的每一组也各有一条。"}
      </p>

      {(years > 0 || moved || book) && sim.isError && (
        <p className="text-[12.5px] text-up">
          {(sim.error as ApiError).message} —— {years > 0 || moved ? "下面还是优化窗口的结果。" : "候选的线画不出来。"}
        </p>
      )}
      {got?.limited_by && (
        <p className="text-[12px] text-brand-ink leading-snug">
          {got.limited_by.n} 从 {got.limited_by.from} 才有行情，每只股票都要同时有价，
          所以模拟最早只能从这天开始。
        </p>
      )}
      {got && got.missing.length > 0 && (
        <p className="text-[12px] text-up">读不到行情，这段模拟里没有算：{got.missing.join("、")}</p>
      )}

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px]">
        {series.map((s) => {
          const last = [...s.values].reverse().find((v) => Number.isFinite(v));
          const off = hidden.has(s.key);
          return (
            <button key={s.key} onClick={() => toggle(s.key)}
              title={off ? "点一下显示" : "点一下隐藏"}
              className={`flex items-center gap-1.5 rounded px-1 -mx-1 hover:bg-elevated ${
                off ? "opacity-40" : ""}`}>
              <span className="w-4 h-[3px] rounded-full" style={{ background: s.colour }} />
              <span className={off ? "line-through" : ""}>{s.label}</span>
              <span className="tnum font-medium">
                {s.waiting ? "…" : signed(last ?? null, 1, "%")}
              </span>
            </button>
          );
        })}
      </div>

      <div className="relative">
        <svg viewBox={`0 0 ${VW} ${VH}`} preserveAspectRatio="none"
          className="w-full h-[190px] block" role="img"
          aria-label="组合、你的权重、等权与基准的累计收益对比">
          {bottom < 0 && top > 0 && (
            <line x1={0} x2={VW} y1={y(0)} y2={y(0)} stroke="var(--color-line-bright)"
              vectorEffect="non-scaling-stroke" />
          )}
          {shade && (
            <rect x={x(f0)} width={Math.max(1, x(f1) - x(f0))} y={0} height={VH}
              fill="var(--color-cyan)" opacity={0.07} />
          )}
          {shade && f0 > 0 && (
            <line x1={x(f0)} x2={x(f0)} y1={0} y2={VH}
              stroke="var(--color-ink-mute)" strokeDasharray="4 4"
              vectorEffect="non-scaling-stroke" />
          )}
          {shade && f1 < n - 1 && (
            <line x1={x(f1)} x2={x(f1)} y1={0} y2={VH}
              stroke="var(--color-ink-mute)" strokeDasharray="4 4"
              vectorEffect="non-scaling-stroke" />
          )}
          {drawn.map((s) => (
            <path key={s.key} d={path(s.values)} fill="none" stroke={s.colour}
              strokeWidth={s.key === "opt" || s.key === "mine" ? 1.8 : 1.3}
              vectorEffect="non-scaling-stroke" />
          ))}
        </svg>
        {shade && (
          <span className="absolute top-0 label font-mono tnum px-1 whitespace-nowrap"
            style={f0 / (n - 1) < 0.6
              ? { left: `${(f0 / (n - 1)) * 100}%` }
              : { right: `${(1 - f1 / (n - 1)) * 100}%` }}>
            算权重 {d.from} → {d.to}
          </span>
        )}
      </div>

      <div className="flex justify-between label font-mono tnum">
        <span>{dates[0]}</span>
        <span>{dates[n - 1]}</span>
      </div>

      <Scores rows={series.filter((s) => !s.waiting)
        .map((s) => ({ key: s.key, label: s.label, colour: s.colour,
                       off: hidden.has(s.key), st: curveStats(s.values, d.rf_pct) }))}
        bench={bench ? "bm" : null} />
    </div>
  );
}

type ScoreRow = { key: string; label: string; colour: string; off: boolean; st: CurveStats };

const COLS: {
  k: keyof CurveStats; label: string; hi: boolean;
  fmt: (v: number | null) => string; hint?: string;
}[] = [
  { k: "total", label: "区间收益", hi: true, fmt: (v) => signed(v, 1, "%") },
  { k: "ann", label: "年化", hi: true, fmt: (v) => signed(v, 1, "%") },
  { k: "vol", label: "波动", hi: false, fmt: (v) => `${fixed(v, 1)}%` },
  { k: "sharpe", label: "夏普", hi: true, fmt: (v) => fixed(v, 2) },
  { k: "maxdd", label: "最大回撤", hi: true, fmt: (v) => `${fixed(v, 1)}%`,
    hint: "从此前最高点算起的最大跌幅。越接近 0 越好。" },
];

/**
 * Every line, scored over the days on the chart. The best in each column is
 * starred among the portfolios — the index is listed for reference but never
 * wins, since it is what they are trying to beat.
 */
function Scores({ rows, bench }: { rows: ScoreRow[]; bench: string | null }) {
  const contenders = rows.filter((r) => r.key !== bench && !r.off);
  const best: Partial<Record<keyof CurveStats, number>> = {};
  for (const c of COLS) {
    const vals = contenders.map((r) => r.st[c.k]).filter((v): v is number => v != null);
    if (vals.length >= 2) best[c.k] = c.hi ? Math.max(...vals) : Math.min(...vals);
  }
  return (
    <div className="flex flex-col gap-1 mt-1">
      <div className="overflow-x-auto">
        <table className="w-full text-[12.5px] border-collapse min-w-[440px]">
          <thead>
            <tr className="text-ink-mute">
              <th className="text-left font-normal pb-1 pr-3">图上这段行情</th>
              {COLS.map((c) => (
                <th key={c.k} title={c.hint}
                  className="text-right font-normal pb-1 px-2 whitespace-nowrap">{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key} className={`border-t border-line ${r.off ? "opacity-40" : ""} ${
                r.key === bench ? "text-ink-dim" : ""}`}>
                <td className="py-1 pr-3">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full shrink-0"
                      style={{ background: r.colour }} />
                    <span className="truncate max-w-[180px]">{r.label}</span>
                  </span>
                </td>
                {COLS.map((c) => {
                  const v = r.st[c.k];
                  const top = !r.off && r.key !== bench && v != null && best[c.k] === v;
                  return (
                    <td key={c.k} className={`py-1 px-2 text-right tnum whitespace-nowrap ${
                      top ? "font-semibold text-cyan" : ""}`}>
                      {c.fmt(v)}{top && " ★"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="label leading-snug">
        ★ 是每一列里组合中最好的那个{bench ? "，基准只作参照" : ""}；点图例可以隐藏一条线，隐藏的不参与比较。
        年化按每年 242 个交易日，夏普扣掉无风险利率，和上面「买入持有实测」同一个算法。
        优化窗口里的成绩一定偏好看 —— 切到更长的区间，看优化器没见过的行情里谁还站得住。
      </p>
    </div>
  );
}
