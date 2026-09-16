/**
 * 板块相关性 / 板块轮动 — which sectors this stock trades like, and which one
 * has been leading it.
 *
 * Left: today's correlation against every sector index, sorted. Right: the
 * same correlations over the past year as lines, with a strip underneath
 * showing which sector was on top each day — where the colour changes, the
 * stock changed theme. A stock whose strip is one solid colour has a sector
 * identity; one that looks like confetti does not, and its top correlation
 * this week means very little.
 *
 * 沪深300 is drawn in amber and kept out of the "sector" reading: a stock
 * correlated with everything is just correlated with the market.
 *
 * Bars follow the Chinese convention — red for positive, green for negative —
 * so the colour means the same thing here as everywhere else in the app.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { SectorAnalysis, SectorRow } from "../lib/types";
import { usePersistentState } from "../lib/usePersistentState";

const WINDOWS = [5, 10, 20, 30, 60];

const VW = 900; // line-chart viewBox width; the SVG itself is fluid
const VH = 210;
const STRIP = 22;

export function SectorPanel({ ticker }: { ticker: string }) {
  const [window, setWindow] = usePersistentState<number>("assrs.sector.window", 20);
  const q = useQuery({
    queryKey: ["sectors", ticker, window],
    queryFn: () => api.sectors(ticker, window),
    staleTime: 20 * 60_000,
  });

  return (
    <section className="card p-3 flex flex-col gap-3">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="text-[14px] font-semibold">🧲 板块相关性 · 轮动</h2>
        <p className="label grow min-w-[240px]">
          个股日收益率与各板块指数（市值加权）的滚动皮尔逊相关系数
        </p>
        <div className="flex items-center gap-1">
          <span className="label">滚动窗口</span>
          {WINDOWS.map((w) => (
            <button key={w} onClick={() => setWindow(w)}
              className={`h-7 px-2 rounded-md text-[12.5px] border transition-colors ${
                w === window ? "border-cyan bg-cyan text-white" : "border-line bg-panel hover:bg-elevated"
              }`}>
              {w}日
            </button>
          ))}
        </div>
      </div>

      {q.isPending && <p className="label py-6 text-center">正在计算板块相关性…</p>}
      {q.isError && (
        <p className="text-[12.5px] text-up py-4 text-center">
          {(q.error as Error).message}
          <button onClick={() => q.refetch()} className="ml-2 text-cyan">重试</button>
        </p>
      )}
      {q.data && <Body d={q.data} />}
    </section>
  );
}

function Body({ d }: { d: SectorAnalysis }) {
  const { summary: s } = d;
  const byName = new Map(d.sectors.map((x) => [x.name, x]));

  return (
    <>
      <Verdict d={d} />

      <div className="grid gap-3 xl:grid-cols-[minmax(0,320px)_minmax(0,1fr)] items-start">
        <Bars rows={d.sectors} />
        <div className="min-w-0 flex flex-col gap-2">
          <Rotation d={d} byName={byName} />
          <Leaders d={d} byName={byName} />
        </div>
      </div>

      <p className="label leading-snug">
        相关性不等于因果，也不等于板块归属：一只股票可以与某板块同涨同跌却完全不在其中。
        近 {s.sessions} 个交易日 · {d.sectors.length} 个板块指数。
      </p>
    </>
  );
}

function Verdict({ d }: { d: SectorAnalysis }) {
  const s = d.summary;
  const trend = s.trend === "strengthening" ? "📈 增强中"
    : s.trend === "weakening" ? "📉 减弱中" : "➡️ 稳定";
  const verdict = s.verdict === "low"
    ? { icon: "✅", text: "板块属性稳定，价格走势长期由同一板块解释。" }
    : s.verdict === "high"
      ? { icon: "⚠️", text: "轮动频繁，该股不断切换题材——板块相关性只是短期信号。" }
      : { icon: "ℹ️", text: "中等轮动：有主导板块，但会随题材阶段性切换。" };

  return (
    <div className="rounded-lg bg-sunken px-3 py-2 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[12.5px]">
      <span>
        主导板块 <b>{s.top}</b>{" "}
        <span className="font-mono tnum">{fmtR(s.top_r)}</span>
      </span>
      <span className="text-ink-dim">
        {trend}（近5日 {fmtR(s.r5)} vs 近20日 {fmtR(s.r20)}）
      </span>
      <span className="text-ink-dim">
        轮动 {s.rotations} 次 · {s.n_leaders} 个板块曾领先
      </span>
      <span className="basis-full text-ink-dim">
        {verdict.icon} {verdict.text}
        {s.self_index && (
          <span className="text-brand-ink">
            {" "}该股本身是「{s.top}」指数的成分股且权重可观，相关性接近 1 属于构成效应，并非独立信号。
          </span>
        )}
      </span>
    </div>
  );
}

/** Today's correlation per sector, as a diverging bar list. */
function Bars({ rows }: { rows: SectorRow[] }) {
  const [hover, setHover] = useState<string | null>(null);
  return (
    <div className="flex flex-col">
      <div className="flex items-center justify-between label mb-1">
        <span>−1</span><span>相关系数 r</span><span>+1</span>
      </div>
      {rows.map((row) => {
        const r = row.r ?? 0;
        const pct = Math.min(Math.abs(r), 1) * 50;
        const color = row.benchmark ? "var(--color-brand)"
          : r >= 0 ? "var(--color-up)" : "var(--color-down)";
        return (
          <div key={row.name}
            onMouseEnter={() => setHover(row.name)} onMouseLeave={() => setHover(null)}
            title={`${row.name}  当前 ${fmtR(row.r)} · 窗口均值 ${fmtR(row.mean_r)}${row.member ? " · 成分股" : ""}`}
            className={`relative h-[17px] ${hover === row.name ? "bg-elevated" : ""}`}>
            {/* the zero rule */}
            <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
            <div className="absolute inset-y-[3px] rounded-sm" style={{
              background: color,
              left: r >= 0 ? "50%" : `${50 - pct}%`,
              width: `${pct}%`,
            }} />
            <div className="absolute inset-0 flex items-center justify-between px-1 text-[11px] pointer-events-none">
              <span className="truncate max-w-[45%]" style={{ textShadow: "0 0 3px #fff, 0 0 3px #fff" }}>
                {row.name}{row.member ? " ·" : ""}
              </span>
              <span className="font-mono tnum" style={{ textShadow: "0 0 3px #fff, 0 0 3px #fff" }}>
                {fmtR(row.r)}
              </span>
            </div>
          </div>
        );
      })}
      <p className="label mt-1">「·」= 该股是此板块指数的成分股</p>
    </div>
  );
}

/** Rolling correlation lines + the dominant-sector strip. */
function Rotation({ d, byName }: { d: SectorAnalysis; byName: Map<string, SectorRow> }) {
  const n = d.dates.length;
  const x = (i: number) => (n < 2 ? 0 : (i / (n - 1)) * VW);
  const y = (r: number) => ((1 - r) / 2) * VH;

  const path = (series: (number | null)[]) => {
    let out = "";
    let pen = false;
    series.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      out += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`;
      pen = true;
    });
    return out;
  };

  const back = d.sectors.filter((s) => !s.top && !s.benchmark);
  const front = d.sectors.filter((s) => s.top || s.benchmark);

  return (
    <div className="min-w-0">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 mb-1">
        <span className="text-[12.5px] font-medium">🔄 板块轮动</span>
        {front.map((s) => (
          <span key={s.name} className="text-[11px] flex items-center gap-1">
            <i className="inline-block w-2.5 h-[3px] rounded-sm" style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </div>

      <svg viewBox={`0 0 ${VW} ${VH}`} preserveAspectRatio="none"
        className="w-full h-[170px] block" role="img" aria-label="滚动相关系数">
        <line x1={0} x2={VW} y1={y(0)} y2={y(0)} stroke="var(--color-line)"
          strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
        {[0.5, -0.5].map((v) => (
          <line key={v} x1={0} x2={VW} y1={y(v)} y2={y(v)} stroke="var(--color-line)"
            strokeWidth={0.5} opacity={0.5} vectorEffect="non-scaling-stroke" />
        ))}
        {back.map((s) => (
          <path key={s.name} d={path(s.series)} fill="none" stroke="#828282" strokeWidth={0.8}
            opacity={0.18} vectorEffect="non-scaling-stroke" />
        ))}
        {front.map((s) => (
          <path key={s.name} d={path(s.series)} fill="none" stroke={s.color} strokeWidth={2}
            vectorEffect="non-scaling-stroke" />
        ))}
      </svg>

      {/* Which sector led on each day. */}
      <svg viewBox={`0 0 ${VW} ${STRIP}`} preserveAspectRatio="none"
        className="w-full h-[16px] block mt-0.5" role="img" aria-label="每日领先板块">
        {d.dominant.map((run, i) => {
          const x0 = x(run.from);
          const x1 = run.to >= n - 1 ? VW : x(run.to + 1);
          return (
            <rect key={i} x={x0} y={0} width={Math.max(x1 - x0, 0.7)} height={STRIP}
              fill={byName.get(run.sector)?.color ?? "#64748b"}>
              <title>{`${run.sector}  ${d.dates[run.from]} → ${d.dates[run.to]}`}</title>
            </rect>
          );
        })}
      </svg>

      <div className="flex justify-between label mt-0.5">
        <span>{d.dates[0]}</span>
        <span>每日相关性最高的板块</span>
        <span>{d.dates[n - 1]}</span>
      </div>
    </div>
  );
}

function Leaders({ d, byName }: { d: SectorAnalysis; byName: Map<string, SectorRow> }) {
  if (d.summary.leaders.length === 0) return null;
  return (
    <div className="flex flex-col gap-0.5">
      <span className="label">领先天数</span>
      {d.summary.leaders.map((l) => (
        <div key={l.sector} className="flex items-center gap-2 text-[11.5px]">
          <span className="w-24 truncate font-medium" style={{ color: byName.get(l.sector)?.color }}>
            {l.sector}
          </span>
          <span className="h-2 rounded-sm" style={{
            width: `${Math.max(l.pct, 1)}%`, background: byName.get(l.sector)?.color ?? "#64748b",
          }} />
          <span className="font-mono tnum text-ink-mute shrink-0">
            {l.days}天 ({l.pct.toFixed(0)}%)
          </span>
        </div>
      ))}
    </div>
  );
}

function fmtR(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(3)}`;
}
