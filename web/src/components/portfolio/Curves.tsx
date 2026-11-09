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
 */

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";
import type { PortfolioBuild, WeighResult } from "../../lib/types";
import { signed } from "../../lib/format";
import { usePersistentState } from "../../lib/usePersistentState";

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

export function Curves({ d, tweak }: { d: PortfolioBuild; tweak: WeighResult | null }) {
  const [years, setYears] = usePersistentState<number>("assrs.pf.sim.years", 0);
  const symbols = d.holdings.map((h) => h.t);
  const weights = pcts(d.holdings);
  const mine = tweak ? pcts(tweak.holdings) : null;

  const sim = useQuery({
    queryKey: ["portfolio", "simulate", symbols, weights, mine, years],
    queryFn: () => api.portfolioSimulate({ symbols, weights, mine, years }),
    enabled: years > 0,
    staleTime: 10 * 60_000,
    retry: false,
    placeholderData: keepPreviousData,
  });
  const long = years > 0 ? sim.data ?? null : null;

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

  const series = [
    { key: "opt", label: d.mode === "target" ? "目标组合" : d.mode_label, values: opt, colour: "var(--color-cyan)" },
    ...(own
      ? [{ key: "mine", label: tweak && !tweak.balanced ? `你的权重（合计 ${tweak.sum_pct}%）` : "你的权重", values: own, colour: MINE }]
      : []),
    ...(isEqual ? [] : [{ key: "eq", label: "等权重", values: equal, colour: "#a855f7" }]),
    ...(bench
      ? [{
          key: "bm", label: bench.label,
          values: bench.curve.map((v) => (v == null ? NaN : v)),
          colour: "var(--color-ink-mute)",
        }]
      : []),
  ];
  // Yours is drawn last, so it stays visible where it crosses the others.
  const drawn = [...series.filter((s) => s.key !== "mine"), ...series.filter((s) => s.key === "mine")];

  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series) {
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

  // Where the optimiser's window begins, when the replay starts before it.
  const fitted = long && !isEqual ? dates.indexOf(d.from) : -1;
  const benchLabel = d.benchmark ? d.benchmark.label : "基准";

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-1">
        {RANGES.map((r) => (
          <button key={r.years} onClick={() => setYears(r.years)}
            title={r.years ? `权重不变，回放最近 ${r.years} 年` : "算权重用的那段行情"}
            className={`h-7 px-2.5 rounded-md text-[12px] font-medium transition-colors ${
              years === r.years ? "bg-elevated text-ink" : "text-ink-mute hover:text-ink"}`}>
            {r.label}
          </button>
        ))}
        {years > 0 && sim.isFetching && (
          <span className="label ml-1">读取 {years} 年行情…</span>
        )}
      </div>

      <p className="label leading-snug">
        {years === 0
          ? isEqual
            ? `每只一样多，权重不是从这段行情里算出来的，所以这条线可以直接和${benchLabel}比；不过股票本身是今天选的，仍然带着后见之明。`
            : `用最终权重回看这段历史。优化就是在这段数据上做的，所以这条线一定好看 —— 等权和${benchLabel}画在一起才读得出来。`
          : `权重不变，只把回放拉长。${fitted > 0
              ? "虚线右边是算权重用的那段；左边的行情优化器没见过，更能说明这组权重的成色，不过股票本身是今天选的，仍然带着后见之明。"
              : ""}`}
        {" "}每天按同一组权重计算，从起点的 0% 复利；股票含分红，{benchLabel}是价格指数、不含分红。
        {!tweak && " 在「调整权重」里点「看看落在哪」，这里会多一条你的权重的线。"}
      </p>

      {years > 0 && sim.isError && (
        <p className="text-[12.5px] text-up">
          {(sim.error as ApiError).message} —— 下面还是优化窗口的结果。
        </p>
      )}
      {long?.limited_by && (
        <p className="text-[12px] text-brand-ink leading-snug">
          {long.limited_by.n} 从 {long.limited_by.from} 才有行情，每只股票都要同时有价，
          所以模拟最早只能从这天开始。
        </p>
      )}
      {long && long.missing.length > 0 && (
        <p className="text-[12px] text-up">读不到行情，这段模拟里没有算：{long.missing.join("、")}</p>
      )}

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px]">
        {series.map((s) => {
          const last = [...s.values].reverse().find((v) => Number.isFinite(v));
          return (
            <span key={s.key} className="flex items-center gap-1.5">
              <span className="w-4 h-[3px] rounded-full" style={{ background: s.colour }} />
              <span>{s.label}</span>
              <span className="tnum font-medium">{signed(last ?? null, 1, "%")}</span>
            </span>
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
          {fitted > 0 && (
            <line x1={x(fitted)} x2={x(fitted)} y1={0} y2={VH}
              stroke="var(--color-ink-mute)" strokeDasharray="4 4"
              vectorEffect="non-scaling-stroke" />
          )}
          {drawn.map((s) => (
            <path key={s.key} d={path(s.values)} fill="none" stroke={s.colour}
              strokeWidth={s.key === "opt" || s.key === "mine" ? 1.8 : 1.3}
              vectorEffect="non-scaling-stroke" />
          ))}
        </svg>
        {fitted > 0 && (
          <span className="absolute top-0 label font-mono tnum px-1 whitespace-nowrap"
            style={fitted / (n - 1) < 0.7
              ? { left: `${(fitted / (n - 1)) * 100}%` }
              : { right: `${(1 - fitted / (n - 1)) * 100}%` }}>
            {d.from} 起算权重
          </span>
        )}
      </div>

      <div className="flex justify-between label font-mono tnum">
        <span>{dates[0]}</span>
        <span>{dates[n - 1]}</span>
      </div>
    </div>
  );
}
