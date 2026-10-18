/**
 * 季节性 — the last few full years on one Jan–Dec axis.
 *
 * Read top to bottom: what the numbers conclude (and whether that beats
 * chance), the years drawn over each other with the agreeing windows shaded,
 * the windows themselves year by year, then the grid of every week or month.
 *
 * Closed until opened: it is its own fetch of up to eleven years of bars,
 * which the page does not otherwise need.
 */

import { useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import type { SeasonBasis, Seasonality } from "../lib/types";
import { signed } from "../lib/format";
import { usePersistentState } from "../lib/usePersistentState";
import {
  MONTHS, MONTH_SLOTS, cellFill, cellInk, slotLabel, valueAt, weekOf, weeklyPath,
  yearColour, type Point,
} from "../lib/season";

type Grain = "d" | "w";
type Basis = "abs" | "excess";

export function SeasonalityPanel({ ticker }: { ticker: string }) {
  const [open, setOpen] = usePersistentState("assrs.season.open", false);
  const [years, setYears] = usePersistentState<number>("assrs.season.years", 3);
  const [grain, setGrain] = usePersistentState<Grain>("assrs.season.grain", "d");
  const [basis, setBasis] = usePersistentState<Basis>("assrs.season.basis", "abs");

  const q = useQuery({
    queryKey: ["season", ticker, years],
    queryFn: () => api.seasonality(ticker, years),
    enabled: open,
    staleTime: 3600_000,
    retry: (n, err) => !(err instanceof ApiError && err.status < 500) && n < 1,
  });

  return (
    <section className="card">
      <button onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-elevated rounded-[10px]">
        <span className="text-[14px] font-semibold">📅 季节性</span>
        <span className="label">历年同期叠加 · 哪段时间往往涨、哪段往往跌 · 是否超出随机</span>
        <span className="ml-auto label">{open ? "收起 ▲" : "展开 ▼"}</span>
      </button>
      {open && (
        <div className="px-3 pb-3 pt-1 flex flex-col gap-3 border-t border-line">
          <Controls years={years} setYears={setYears} grain={grain} setGrain={setGrain}
            basis={basis} setBasis={setBasis}
            benchmark={q.data?.benchmark_name} hasExcess={q.data ? !!q.data.excess : true} />
          {q.isPending && (
            <div className="py-10 flex flex-col items-center gap-2 text-ink-mute">
              <div className="h-6 w-6 rounded-full border-2 border-line border-t-cyan animate-spin" />
              <span className="label">读取 {years + 1} 年日线并做 300 次随机检验…</span>
            </div>
          )}
          {q.isError && (
            <p className="text-[12.5px] text-up py-4">
              {(q.error as Error).message}
              <button onClick={() => q.refetch()} className="ml-2 text-cyan">重试</button>
            </p>
          )}
          {q.data && (
            <Body d={q.data} grain={grain}
              basis={basis === "excess" && q.data.excess ? "excess" : "abs"} />
          )}
        </div>
      )}
    </section>
  );
}

function Seg<T extends string | number>({ value, set, options }: {
  value: T; set: (v: T) => void; options: { v: T; label: string; disabled?: boolean; title?: string }[];
}) {
  return (
    <div className="flex rounded-md bg-sunken p-0.5">
      {options.map((o) => (
        <button key={String(o.v)} disabled={o.disabled} title={o.title}
          onClick={() => set(o.v)}
          className={`h-6 px-2.5 rounded text-[12px] disabled:opacity-40 ${
            value === o.v ? "bg-panel shadow-sm font-semibold" : "text-ink-mute hover:text-ink"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Controls({ years, setYears, grain, setGrain, basis, setBasis, benchmark, hasExcess }: {
  years: number; setYears: (v: number) => void;
  grain: Grain; setGrain: (v: Grain) => void;
  basis: Basis; setBasis: (v: Basis) => void;
  benchmark?: string; hasExcess: boolean;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2 pt-2">
      <Seg value={years} set={setYears}
        options={[3, 5, 10].map((v) => ({ v, label: `${v}年` }))} />
      <Seg value={grain} set={setGrain}
        options={[{ v: "d" as Grain, label: "日线" }, { v: "w" as Grain, label: "周线" }]} />
      <Seg value={basis} set={setBasis}
        options={[
          { v: "abs" as Basis, label: "股价" },
          { v: "excess" as Basis, label: `相对${benchmark ?? "大盘"}`, disabled: !hasExcess,
            title: hasExcess ? "扣除大盘后的超额收益：剩下的才是这只股票自己的季节性" : "指数数据不可用" },
        ]} />
    </div>
  );
}

function Body({ d, grain, basis }: { d: Seasonality; grain: Grain; basis: Basis }) {
  const b = d[basis] as SeasonBasis;
  const full = d.years.map((y) => y.year);
  const all = [...full, ...(b.daily[String(d.current_year)] ? [d.current_year] : [])];
  const [hidden, setHidden] = useState<Set<number>>(new Set());

  const paths = useMemo(() => {
    const out: Record<number, Point[]> = {};
    for (const y of all) {
      out[y] = grain === "d"
        ? (b.daily[String(y)] ?? [])
        : weeklyPath(b.weekly[String(y)] ?? []);
    }
    return out;
  }, [b, grain, all.join(",")]);   // eslint-disable-line react-hooks/exhaustive-deps

  const avg: Point[] = useMemo(() => {
    if (grain === "d") return b.avg_daily.map((v, i) => [i, v] as Point);
    let lvl = 1;
    const out: Point[] = [[0, 0]];
    b.week_stats.forEach((s, k) => {
      lvl *= 1 + s.mean / 100;
      out.push([k === 51 ? 365 : 7 * k + 6, +((lvl - 1) * 100).toFixed(2)]);
    });
    return out;
  }, [b, grain]);

  return (
    <div className="flex flex-col gap-3">
      <Conclusions d={d} b={b} />
      <Overlay d={d} b={b} paths={paths} avg={avg} all={all} full={full}
        hidden={hidden} setHidden={setHidden} />
      <Windows d={d} b={b} full={full} />
      {grain === "w"
        ? <WeekGrid d={d} b={b} all={all} />
        : <MonthGrid d={d} b={b} all={all} />}
      {d.market === "CN" && b.spring.length > 0 && <Spring d={d} b={b} />}
      <p className="label leading-snug">
        每年从上一年最后一个收盘价起算，按月-日对齐（2月29日单独一格）。周按 1月1日起每 7 天一段，
        最后一段延至 12月31日；休市周（国庆、春节）记为不动并标灰。窗口要求所有年份同向（超过 5 年时允许
        每 5 年有 1 年例外），且最弱的一年也动了至少 1%，再按"平均涨跌 ÷ 该股自身波动"排序。
        显著性：把每一年的周收益随机错开若干周（保留每年的涨跌幅度与趋势、只打乱日期），重做同样的搜索
        {b.significance.shuffles} 次，看随机日历下能否找到同样强的窗口。季节性只是历史统计，不构成投资建议。
      </p>
    </div>
  );
}

const VERDICT: Record<string, { label: string; cls: string }> = {
  strong: { label: "超出随机", cls: "bg-cyan/15 text-cyan" },
  weak: { label: "证据偏弱", cls: "bg-brand/15 text-brand-ink" },
  none: { label: "与随机无异", cls: "bg-sunken text-ink-mute" },
};

function Conclusions({ d, b }: { d: Seasonality; b: SeasonBasis }) {
  const v = VERDICT[b.significance.verdict] ?? VERDICT.none!;
  return (
    <div className="rounded-lg bg-sunken p-3 flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[13px] font-semibold">结论</span>
        <span className={`px-2 py-0.5 rounded-full text-[11.5px] font-semibold ${v.cls}`}>{v.label}</span>
        <span className="label">数据截至 {d.as_of}</span>
        {d.skipped.length > 0 && (
          <span className="label">（{d.skipped.join("、")} 年数据不完整，未计入）</span>
        )}
      </div>
      <ul className="flex flex-col gap-1 list-disc pl-4">
        {b.conclusions.map((c, i) => (
          <li key={i} className="text-[12.5px] leading-snug text-ink-dim">{c}</li>
        ))}
      </ul>
    </div>
  );
}

// ── the overlay ──────────────────────────────────────────────────────────────
const VW = 1000;
const VH = 260;

function Overlay({ d, b, paths, avg, all, full, hidden, setHidden }: {
  d: Seasonality; b: SeasonBasis; paths: Record<number, Point[]>; avg: Point[];
  all: number[]; full: number[]; hidden: Set<number>; setHidden: (s: Set<number>) => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const shown = all.filter((y) => !hidden.has(y));
  const rise = d.up_is_red ? "var(--color-up)" : "var(--color-down)";
  const fall = d.up_is_red ? "var(--color-down)" : "var(--color-up)";

  let lo = 0;
  let hi = 0;
  for (const y of shown) for (const [, v] of paths[y] ?? []) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  for (const [, v] of avg) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const pad = Math.max(2, (hi - lo) * 0.06);
  const top = hi + pad;
  const bottom = lo - pad;
  const x = (slot: number) => (slot / 365) * VW;
  const y = (v: number) => VH - ((v - bottom) / (top - bottom)) * VH;
  const line = (pts: Point[]) =>
    pts.map(([s, v], i) => `${i ? "L" : "M"}${x(s).toFixed(1)} ${y(v).toFixed(1)}`).join("");

  const ticks = niceTicks(bottom, top);
  const legendVal = (yr: number) => {
    const p = paths[yr] ?? [];
    return p.length ? p[p.length - 1]![1] : null;
  };

  function move(e: React.MouseEvent) {
    const r = box.current?.getBoundingClientRect();
    if (!r) return;
    setHover(Math.max(0, Math.min(365, Math.round(((e.clientX - r.left) / r.width) * 365))));
  }

  const tip = hover == null ? null : [
    ...[...shown].sort((a, c) => c - a).map((yr) => ({
      k: String(yr), label: yr === d.current_year ? `${yr}（今年）` : String(yr),
      colour: yearColour(yr, full, d.current_year), v: valueAt(paths[yr] ?? [], hover),
    })),
    { k: "avg", label: `${full.length}年平均`, colour: "var(--color-ink-mute)", v: valueAt(avg, hover) },
  ];

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px]">
        {[...all].sort((a, c) => c - a).map((yr) => {
          const off = hidden.has(yr);
          const regime = d.years.find((r) => r.year === yr)?.regime;
          return (
            <button key={yr} onClick={() => {
              const n = new Set(hidden);
              if (off) n.delete(yr); else n.add(yr);
              setHidden(n);
            }}
              title={off ? "点击显示" : "点击隐藏"}
              className={`flex items-center gap-1.5 ${off ? "opacity-35" : ""}`}>
              <span className="w-4 rounded-full"
                style={{ height: yr === d.current_year ? 3 : 2,
                         background: yearColour(yr, full, d.current_year) }} />
              <span>{yr}{yr === d.current_year ? " 至今" : ""}</span>
              {regime === "pre" && <span className="label">924前</span>}
              <span className="tnum font-medium">{signed(legendVal(yr), 1, "%")}</span>
            </button>
          );
        })}
        <span className="flex items-center gap-1.5">
          <span className="w-4 border-t-2 border-dashed border-ink-mute" />
          <span>平均</span>
        </span>
      </div>

      <div className="flex gap-1.5">
        <div className="relative w-9 shrink-0 h-[260px]">
          {ticks.map((t) => (
            <span key={t} className="absolute right-0 -translate-y-1/2 label font-mono tnum"
              style={{ top: `${(y(t) / VH) * 100}%` }}>{t > 0 ? "+" : ""}{t}%</span>
          ))}
        </div>
        <div ref={box} className="relative flex-1 min-w-0 h-[260px]"
          onMouseMove={move} onMouseLeave={() => setHover(null)}>
          <svg viewBox={`0 0 ${VW} ${VH}`} preserveAspectRatio="none"
            className="absolute inset-0 w-full h-full block" role="img"
            aria-label="历年同期累计涨跌叠加">
            {b.windows.map((w, i) => (
              <rect key={i} x={x(w.start * 7)} width={x(Math.min(366, (w.start + w.weeks) * 7)) - x(w.start * 7)}
                y={0} height={VH} fill={w.dir === "up" ? rise : fall} opacity={0.08} />
            ))}
            {MONTH_SLOTS.slice(1).map((s) => (
              <line key={s} x1={x(s)} x2={x(s)} y1={0} y2={VH} stroke="var(--color-line)"
                strokeWidth={1} vectorEffect="non-scaling-stroke" />
            ))}
            {ticks.map((t) => (
              <line key={t} x1={0} x2={VW} y1={y(t)} y2={y(t)}
                stroke={t === 0 ? "var(--color-line-bright)" : "var(--color-line)"}
                strokeWidth={1} strokeDasharray={t === 0 ? undefined : "2 4"}
                vectorEffect="non-scaling-stroke" />
            ))}
            {[...shown].sort((a, c) => a - c).map((yr) => (
              <path key={yr} d={line(paths[yr] ?? [])} fill="none"
                stroke={yearColour(yr, full, d.current_year)}
                strokeWidth={yr === d.current_year ? 2.4 : 1.5}
                strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
            ))}
            <path d={line(avg)} fill="none" stroke="var(--color-ink-mute)" strokeWidth={1.6}
              strokeDasharray="6 4" vectorEffect="non-scaling-stroke" />
            {hover != null && (
              <line x1={x(hover)} x2={x(hover)} y1={0} y2={VH} stroke="var(--color-ink-mute)"
                strokeWidth={1} vectorEffect="non-scaling-stroke" />
            )}
          </svg>

          {/* Window numbers, matched to the table below. */}
          {b.windows.map((w, i) => (
            <span key={i} className="absolute top-1 -translate-x-1/2 label font-semibold"
              style={{ left: `${((w.start * 7 + w.weeks * 3.5) / 365) * 100}%` }}>{i + 1}</span>
          ))}
          {/* 春节, per year, in that year's colour: it moves by up to a month. */}
          {d.market === "CN" && shown.map((yr) => {
            const sd = d.spring_dates[String(yr)];
            if (!sd) return null;
            const [m, dd] = sd.split("-").map(Number);
            const slot = MONTH_SLOTS[(m ?? 1) - 1]! + (dd ?? 1) - 1;
            return (
              <span key={yr} title={`${yr} 春节 ${sd}`}
                className="absolute bottom-0 w-[2px] h-2.5 -translate-x-1/2"
                style={{ left: `${(slot / 365) * 100}%`, background: yearColour(yr, full, d.current_year) }} />
            );
          })}

          {tip && hover != null && (
            <div className="absolute top-2 z-10 card px-2 py-1.5 pointer-events-none text-[12px] min-w-[132px]"
              style={hover > 240 ? { right: `${100 - (hover / 365) * 100 + 1}%` } : { left: `${(hover / 365) * 100 + 1}%` }}>
              <div className="font-semibold mb-0.5 font-mono">{slotLabel(hover)}</div>
              {tip.map((t) => (
                <div key={t.k} className="flex items-center gap-1.5">
                  <span className="w-2.5 h-[2px] rounded-full" style={{ background: t.colour }} />
                  <span className="text-ink-dim">{t.label}</span>
                  <span className="ml-auto tnum font-medium">{signed(t.v, 1, "%")}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className="flex ml-[42px]">
        {MONTHS.map((m, i) => (
          <span key={m} className="label text-center"
            style={{ width: `${(((MONTH_SLOTS[i + 1] ?? 366) - MONTH_SLOTS[i]!) / 366) * 100}%` }}>{m}</span>
        ))}
      </div>
      {d.market === "CN" && <span className="label">底部短线：各年春节所在日期（颜色同年份）</span>}
    </div>
  );
}

function niceTicks(lo: number, hi: number): number[] {
  const span = hi - lo;
  const raw = span / 5;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) out.push(+t.toFixed(6));
  return out;
}

// ── the windows ──────────────────────────────────────────────────────────────
function pctOf(p: number): string {
  return p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`;
}

function Windows({ d, b, full }: { d: Seasonality; b: SeasonBasis; full: number[] }) {
  if (!b.windows.length) {
    return <p className="label">没有任何 2–8 周的时段在这些年里方向一致且幅度超过 1%。</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[12.5px] tnum">
        <thead>
          <tr className="text-left label">
            <th className="py-1 pr-2 font-normal">#</th>
            <th className="py-1 pr-2 font-normal">时段</th>
            <th className="py-1 pr-2 font-normal">方向</th>
            {full.map((y) => <th key={y} className="py-1 pr-2 font-normal text-right">{y}</th>)}
            <th className="py-1 pr-2 font-normal text-right">平均</th>
            <th className="py-1 pr-2 font-normal text-right"
              title="打乱日历后，最强窗口至少这么强的比例。越小越不像巧合；超过 20% 基本可以当作随机">
              随机也能找到 ⓘ
            </th>
          </tr>
        </thead>
        <tbody>
          {b.windows.map((w, i) => (
            <tr key={i} className="border-t border-line">
              <td className="py-1 pr-2 label">{i + 1}</td>
              <td className="py-1 pr-2 font-mono whitespace-nowrap">
                {w.from} ~ {w.to} <span className="label">（{w.weeks}周）</span>
              </td>
              <td className="py-1 pr-2 whitespace-nowrap">
                <span className={(w.dir === "up") === d.up_is_red ? "text-up" : "text-down"}>
                  {w.dir === "up" ? "▲ 涨" : "▼ 跌"}
                </span>
                <span className="label ml-1">{w.hits}/{full.length}</span>
              </td>
              {full.map((y) => {
                const v = w.per_year[String(y)] ?? null;
                return (
                  <td key={y} className="py-1 pr-2 text-right"
                    style={{ color: v == null ? undefined : v > 0 === d.up_is_red ? "var(--color-up)" : "var(--color-down)" }}>
                    {signed(v, 1, "%")}
                  </td>
                );
              })}
              <td className="py-1 pr-2 text-right font-semibold">{signed(w.mean, 1, "%")}</td>
              <td className={`py-1 pr-2 text-right ${w.p < 0.05 ? "font-semibold text-cyan" : w.p < 0.2 ? "" : "text-ink-mute"}`}>
                {pctOf(w.p)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── the grids ────────────────────────────────────────────────────────────────
function WeekGrid({ d, b, all }: { d: Seasonality; b: SeasonBasis; all: number[] }) {
  const CLAMP = 8;
  const rows = [...all].sort((a, c) => c - a);
  const nowWeek = weekOf(slotOf(d.as_of));
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[13px] font-semibold">周度涨跌</span>
      <div className="overflow-x-auto">
        <div className="min-w-[760px]">
          <div className="grid gap-[2px]" style={{ gridTemplateColumns: "52px repeat(52, minmax(11px, 1fr))" }}>
            <span />
            {d.week_labels.map((l, k) => {
              const m = Number(l.slice(0, 2));
              const first = k === 0 || Number(d.week_labels[k - 1]!.slice(0, 2)) !== m;
              return <span key={k} className="label text-[10px] leading-none whitespace-nowrap overflow-visible">{first ? `${m}月` : ""}</span>;
            })}
            {rows.map((y) => (
              <Row key={y} label={String(y)}>
                {(b.weekly[String(y)] ?? []).map((v, k) => {
                  const closed = b.traded[String(y)]?.[k] === false && v != null;
                  return (
                    <span key={k}
                      title={`${y} ${d.week_labels[k]}：${v == null ? "—" : closed ? "休市" : signed(v, 2, "%")}`}
                      className={`h-5 rounded-[3px] ${y === d.current_year && k === nowWeek ? "ring-1 ring-ink" : ""}`}
                      style={{ background: closed ? "repeating-linear-gradient(135deg, var(--color-line) 0 2px, transparent 2px 4px)" : v == null ? "transparent" : cellFill(v, CLAMP, d.up_is_red) }} />
                  );
                })}
              </Row>
            ))}
            <Row label="平均">
              {b.week_stats.map((s, k) => (
                <span key={k} title={`${d.week_labels[k]}：平均 ${signed(s.mean, 2, "%")}，中位 ${signed(s.median, 2, "%")}`}
                  className="h-5 rounded-[3px] mt-1" style={{ background: cellFill(s.mean, CLAMP / 2, d.up_is_red) }} />
              ))}
            </Row>
            <Row label="上涨年数">
              {b.week_stats.map((s, k) => (
                <span key={k} title={`${d.week_labels[k]}：${s.ups}/${s.n} 年上涨`}
                  className="h-5 text-center text-[10px] leading-5 tnum text-ink-mute">{s.ups}</span>
              ))}
            </Row>
          </div>
        </div>
      </div>
      <span className="label">悬停看数值；斜纹 = 休市周；今年当前周加框。颜色上限 ±{CLAMP}%（平均行 ±{CLAMP / 2}%）。</span>
    </div>
  );
}

function slotOf(iso: string): number {
  const [, m, dd] = iso.split("-").map(Number);
  return MONTH_SLOTS[(m ?? 1) - 1]! + (dd ?? 1) - 1;
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <span className="label text-[11px] leading-5 tnum whitespace-nowrap">{label}</span>
      {children}
    </>
  );
}

function MonthGrid({ d, b, all }: { d: Seasonality; b: SeasonBasis; all: number[] }) {
  const CLAMP = 15;
  const rows = [...all].sort((a, c) => c - a);
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[13px] font-semibold">月度涨跌</span>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-[12px] tnum border-separate border-spacing-[2px]">
          <thead>
            <tr>
              <th />
              {MONTHS.map((m) => <th key={m} className="label font-normal">{m}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((y) => (
              <tr key={y}>
                <td className="label pr-1 whitespace-nowrap">{y}</td>
                {(b.monthly[String(y)] ?? []).map((v, i) => (
                  <td key={i} className="text-center rounded-[4px] h-6"
                    style={{ background: v == null ? "transparent" : cellFill(v, CLAMP, d.up_is_red),
                             color: cellInk(v, CLAMP) }}>
                    {v == null ? "" : signed(v, 1)}
                  </td>
                ))}
              </tr>
            ))}
            <tr>
              <td className="label pr-1 pt-1">平均</td>
              {b.month_stats.map((s, i) => (
                <td key={i} className="text-center rounded-[4px] h-6 font-semibold"
                  style={{ background: cellFill(s.mean, CLAMP / 2, d.up_is_red), color: cellInk(s.mean, CLAMP / 2) }}>
                  {signed(s.mean, 1)}
                </td>
              ))}
            </tr>
            <tr>
              <td className="label pr-1">上涨年数</td>
              {b.month_stats.map((s, i) => (
                <td key={i} className="text-center label">{s.ups}/{s.n}</td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Spring({ d, b }: { d: Seasonality; b: SeasonBasis }) {
  const tone = (v: number | null) =>
    v == null ? undefined : { color: v > 0 === d.up_is_red ? "var(--color-up)" : "var(--color-down)" };
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[13px] font-semibold">🧧 春节效应 <span className="label font-normal">节前 / 节后各 10 个交易日</span></span>
      <div className="overflow-x-auto">
        <table className="text-[12.5px] tnum">
          <thead>
            <tr className="label text-left">
              <th className="pr-4 font-normal">年份</th>
              <th className="pr-4 font-normal">春节</th>
              <th className="pr-4 font-normal text-right">节前</th>
              <th className="pr-4 font-normal text-right">节后</th>
            </tr>
          </thead>
          <tbody>
            {[...b.spring].reverse().map((s) => (
              <tr key={s.year} className="border-t border-line">
                <td className="pr-4 py-0.5">{s.year}</td>
                <td className="pr-4 py-0.5 font-mono">{s.date}</td>
                <td className="pr-4 py-0.5 text-right" style={tone(s.before)}>{signed(s.before, 1, "%")}</td>
                <td className="pr-4 py-0.5 text-right" style={tone(s.after)}>{signed(s.after, 1, "%")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
