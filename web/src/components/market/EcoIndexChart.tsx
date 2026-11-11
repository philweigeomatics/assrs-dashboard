/**
 * 📈 An index under the economic calendar, with every past release of one
 * series marked on it.
 *
 * Each mark sits on the session the release could first move: the first
 * session that CLOSED after it, on that exchange's own calendar. So a Fed
 * decision at 11:00 Vancouver marks that day on 标普500 and the next day on
 * 沪深300. See eco_impact for the rule.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import {
  CandlestickSeries, ColorType, createChart, createSeriesMarkers, CrosshairMode,
  type IChartApi, type ISeriesApi, type ISeriesMarkersPluginApi, type MouseEventParams,
  type SeriesMarker, type Time,
} from "lightweight-charts";
import { api, ApiError } from "../../lib/api";
import type { EcoHistoryRow, EcoIndexBars } from "../../lib/types";
import { usePersistentState } from "../../lib/usePersistentState";
import { localClock, localDay } from "../../lib/localDay";
import { moveClass, signed } from "../../lib/format";
import { seriesOf } from "../../lib/ecoSeries";

export type EcoPick = { country: string; event: string };

const INDEXES = [
  { code: "^GSPC", label: "标普500" },
  { code: "^NDX", label: "纳指100" },
  { code: "000300.SH", label: "沪深300" },
  { code: "000001.SH", label: "上证指数" },
  { code: "399006.SZ", label: "创业板指" },
];

/** The releases most worth a look, by the series name the server stores. */
export const KEY_SERIES: { country: string; label: string; event: string }[] = [
  { country: "美国", label: "美联储利率决议", event: "美联储利率决议" },
  { country: "美国", label: "非农就业", event: "美国季调后非农就业人口变动" },
  { country: "美国", label: "失业率", event: "美国失业率" },
  { country: "美国", label: "CPI", event: "美国CPI月率(%)(月度)" },
  { country: "美国", label: "核心PCE", event: "美国核心PCE物价指数月率(%)(月度)" },
  { country: "美国", label: "零售销售", event: "美国零售销售月率(%)(月度)" },
  { country: "美国", label: "ISM制造业", event: "美国ISM制造业PMI" },
  { country: "美国", label: "GDP", event: "美国实际GDP年化季率(季度)" },
  { country: "美国", label: "FOMC纪要", event: "美联储FOMC货币政策会议纪要" },
  { country: "中国", label: "CPI", event: "中国CPI年率(%)(年度)" },
  { country: "中国", label: "PPI", event: "中国PPI年率(%)(年度)" },
  { country: "中国", label: "制造业PMI", event: "中国官方制造业PMI" },
  { country: "中国", label: "社融", event: "中国社会融资规模(人民币十亿元)" },
  { country: "中国", label: "出口", event: "中国出口年率-美元计价(%)(年度)" },
  { country: "中国", label: "GDP", event: "中国季度GDP年率(%)(年度)" },
  { country: "中国", label: "LPR", event: "中国央行贷款市场报价利率(LPR)" },
];

const RANGES = [1, 3, 5];

// Red up, green down, on every market.
const RED = "#ff3b30";
const GREEN = "#34c759";
const GRID = "#eef0f3";
const AXIS = "#8a8a8e";
// The marks: direction against forecast, not good or bad news.
const ABOVE = "#0891b2";
const BELOW = "#9333ea";
const PLAIN = "#f59e0b";

type Hit = { rows: EcoHistoryRow[]; ret?: number; x?: number | null };

export function EcoIndexChart({ pick, setPick }: {
  pick: EcoPick | null; setPick: (p: EcoPick | null) => void;
}) {
  const [code, setCode] = usePersistentState<string>("assrs.cal.chart.code", "^GSPC");
  const [years, setYears] = usePersistentState<number>("assrs.cal.chart.years", 3);

  const bars = useQuery({
    queryKey: ["cal", "ecoIndex", code],
    queryFn: () => api.ecoIndex(code),
    staleTime: 30 * 60_000,
    retry: 1,
    placeholderData: keepPreviousData,
  });
  const hist = useQuery({
    queryKey: ["cal", "ecoHist", pick?.country, pick ? seriesOf(pick.event) : null, code],
    queryFn: () => api.ecoHistory(pick!.country, pick!.event, code),
    enabled: !!pick,
    staleTime: 30 * 60_000,
    retry: false,
    placeholderData: keepPreviousData,
  });
  const h = pick && hist.data && hist.data.series === seriesOf(pick.event) ? hist.data : null;

  // Releases by the session they landed on; two on one day share a mark.
  const byDay = useMemo(() => {
    const m = new Map<string, Hit>();
    for (const r of h?.rows ?? []) {
      const x = r.react[code];
      if (!x?.done) continue;
      const hit = m.get(x.day) ?? { rows: [], ret: x.ret, x: x.x };
      hit.rows.push(r);
      m.set(x.day, hit);
    }
    return m;
  }, [h, code]);

  const next = useMemo(() => {
    const now = Date.now();
    const ahead = (h?.rows ?? []).filter((r) => new Date(r.at).getTime() > now);
    return ahead.length ? ahead[ahead.length - 1] : null;
  }, [h]);

  const series = pick ? seriesOf(pick.event) : null;
  const keyed = KEY_SERIES.some((k) => k.country === pick?.country && k.event === series);
  const s = h?.summary[code];
  const label = INDEXES.find((i) => i.code === code)?.label ?? code;

  return (
    <section className="card p-3 flex flex-col gap-2.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <h2 className="text-[14.5px] font-semibold">📈 指数与宏观数据</h2>
        <select value={code} onChange={(e) => setCode(e.target.value)} aria-label="指数"
          className="h-7 px-2 rounded-lg bg-sunken text-[12.5px] outline-none
            focus:ring-2 focus:ring-cyan/40">
          {INDEXES.map((i) => <option key={i.code} value={i.code}>{i.label}</option>)}
        </select>
        <div className="flex items-center gap-0.5">
          {RANGES.map((y) => (
            <button key={y} onClick={() => setYears(y)}
              className={`h-7 px-2 rounded-md text-[12px] ${
                years === y ? "bg-elevated text-ink font-medium" : "text-ink-mute hover:text-ink"}`}>
              {y}年
            </button>
          ))}
        </div>
        {(bars.isFetching || hist.isFetching) && <span className="label">读取中…</span>}
      </div>

      <div className="flex flex-col gap-1">
        {(["美国", "中国"] as const).map((c) => (
          <div key={c} className="flex flex-wrap items-center gap-1">
            <span className="label w-8 shrink-0">{c}</span>
            {KEY_SERIES.filter((k) => k.country === c).map((k) => {
              const on = pick?.country === c && series === k.event;
              return (
                <button key={k.event}
                  onClick={() => setPick(on ? null : { country: c, event: k.event })}
                  className={`h-6 px-2 rounded-md text-[12px] transition-colors ${
                    on ? "bg-cyan/15 text-ink ring-1 ring-cyan/50 font-medium"
                      : "bg-sunken text-ink-dim hover:text-ink"}`}>
                  {k.label}
                </button>
              );
            })}
          </div>
        ))}
        {pick && !keyed && (
          <div className="flex items-center gap-1">
            <span className="label w-8 shrink-0">已选</span>
            <button onClick={() => setPick(null)}
              className="h-6 px-2 rounded-md text-[12px] bg-cyan/15 ring-1 ring-cyan/50">
              {series} ✕
            </button>
          </div>
        )}
      </div>

      {bars.isError && <p className="text-[12.5px] text-up">{(bars.error as ApiError).message}</p>}
      {hist.isError && pick && (
        <p className="text-[12.5px] text-up">{(hist.error as ApiError).message}</p>
      )}

      {bars.data ? (
        <Chart bars={bars.data} byDay={byDay} years={years} />
      ) : bars.isPending ? (
        <div className="h-[360px] grid place-items-center label">读取指数行情…</div>
      ) : null}

      {pick && h && s && (
        <p className="text-[12.5px] leading-relaxed">
          <b>{series}</b> 共 {s.n} 次落在 {label} 的交易日上：
          那天的波动平均是普通一天的 <b className="tnum">{s.mean_x?.toFixed(2) ?? "—"}×</b>，
          收涨 <span className="tnum">{s.up != null ? Math.round(s.up * 100) : "—"}%</span>。
          {s.above.n > 0 && (
            <> 高于预期 {s.above.n} 次，平均{" "}
              <span className={`tnum ${moveClass(s.above.ret)}`}>{signed(s.above.ret, 2, "%")}</span>；</>
          )}
          {s.below.n > 0 && (
            <> 低于预期 {s.below.n} 次，平均{" "}
              <span className={`tnum ${moveClass(s.below.ret)}`}>{signed(s.below.ret, 2, "%")}</span>。</>
          )}
          {next && (
            <span className="text-ink-mute">
              {" "}下一次：{localDay(new Date(next.at))} {localClock(next.at)}（你的时间）。
            </span>
          )}
        </p>
      )}
      {pick && h && h.rows.length === 0 && (
        <p className="label">这项数据还没有历史记录。</p>
      )}

      <p className="label leading-snug">
        在上面的日历里点美国或中国的数据名，或者点这里的按钮，它过去每次发布的位置就会标在图上：
        <span style={{ color: ABOVE }}>●</span> 高于预期，
        <span style={{ color: BELOW }}>●</span> 低于预期，
        <span style={{ color: PLAIN }}>●</span> 持平或没有预期值（方向不代表利好利空）。
        标记放在发布后<b>第一个收盘</b>的交易日：美国白天发布的，美股是当天、A股是第二天；
        中国上午发布的，两边都是当天。指上去看每次的数值和当天涨跌。
        「×」是当天波动相对前 60 个交易日平均的倍数，1.0× 就是普通的一天；
        同一天还有别的消息，所以要看多次的平均，单次说明不了什么。
      </p>
    </section>
  );
}

function Chart({ bars, byDay, years }: {
  bars: EcoIndexBars; byDay: Map<string, Hit>; years: number;
}) {
  const box = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const candles = useRef<ISeriesApi<"Candlestick", Time> | null>(null);
  const marks = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const [hover, setHover] = useState<string | null>(null);

  useEffect(() => {
    if (!box.current) return;
    const c = createChart(box.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: "transparent" },
                textColor: AXIS, fontSize: 10, attributionLogo: false },
      grid: { vertLines: { color: GRID }, horzLines: { color: GRID } },
      rightPriceScale: { borderColor: GRID },
      timeScale: { borderColor: GRID, rightOffset: 3 },
      crosshair: { mode: CrosshairMode.Normal },
    });
    const s = c.addSeries(CandlestickSeries, {
      upColor: RED, downColor: GREEN, borderUpColor: RED, borderDownColor: GREEN,
      wickUpColor: RED, wickDownColor: GREEN,
    });
    const fn = (p: MouseEventParams<Time>) => setHover(p.time ? String(p.time) : null);
    c.subscribeCrosshairMove(fn);
    chart.current = c;
    candles.current = s;
    marks.current = createSeriesMarkers(s, []);
    return () => { c.unsubscribeCrosshairMove(fn); c.remove(); chart.current = null; };
  }, []);

  useEffect(() => {
    candles.current?.setData(bars.dates.map((t, i) => ({
      time: t as Time, open: bars.o[i]!, high: bars.h[i]!, low: bars.l[i]!, close: bars.c[i]!,
    })));
  }, [bars]);

  useEffect(() => {
    const have = new Set(bars.dates);
    const list: SeriesMarker<Time>[] = [];
    for (const [day, hit] of byDay) {
      if (!have.has(day)) continue;
      const dirs = new Set(hit.rows.map((r) => r.dir));
      const color = dirs.size === 1 && dirs.has(1) ? ABOVE
        : dirs.size === 1 && dirs.has(-1) ? BELOW : PLAIN;
      list.push({ time: day as Time, position: "aboveBar", shape: "circle", color,
                  size: (hit.x ?? 0) >= 1.5 ? 1.6 : 1 });
    }
    list.sort((a, b) => String(a.time).localeCompare(String(b.time)));
    marks.current?.setMarkers(list);
  }, [byDay, bars]);

  useEffect(() => {
    const c = chart.current;
    const last = bars.dates[bars.dates.length - 1];
    if (!c || !last) return;
    const from = new Date(`${last}T00:00:00`);
    from.setFullYear(from.getFullYear() - years);
    const start = bars.dates.find((d) => d >= localDay(from)) ?? bars.dates[0]!;
    c.timeScale().setVisibleRange({ from: start as Time, to: last as Time });
  }, [bars, years]);

  // A mark is one bar wide; snap to one within two sessions of the cursor so
  // it does not take a pixel-perfect aim to read it.
  let i = hover ? bars.dates.indexOf(hover) : bars.dates.length - 1;
  if (hover && i >= 0 && !byDay.has(hover)) {
    const near = [-1, 1, -2, 2].map((k) => i + k)
      .find((j) => j >= 0 && j < bars.dates.length && byDay.has(bars.dates[j]!));
    if (near != null) i = near;
  }
  const day = bars.dates[i];
  const chg = i > 0 ? (bars.c[i]! / bars.c[i - 1]! - 1) * 100 : null;
  const hit = day ? byDay.get(day) : undefined;

  return (
    <div className="flex flex-col gap-1">
      <div className="text-[12px] tnum min-h-[36px] leading-snug">
        {day && (
          <div>
            <span className="font-mono text-ink-mute">{day}</span>{" "}
            {bars.label} 收 {bars.c[i]?.toFixed(2)}{" "}
            <span className={moveClass(chg)}>{signed(chg, 2, "%")}</span>
          </div>
        )}
        {hit ? hit.rows.map((r) => (
          <div key={`${r.at}|${r.event}`} className="text-ink-dim">
            📌 {r.event} · 实 <b className="text-ink">{r.value ?? "—"}</b> 预 {r.forecast ?? "—"} 前{" "}
            {r.prev ?? "—"}
            <span className="text-ink-mute">（你的时间 {localDay(new Date(r.at))} {localClock(r.at)} 发布）</span>
            {hit.x != null && <> · 当天 <b className="text-ink">{hit.x.toFixed(1)}×</b> 普通一天</>}
          </div>
        )) : <div className="label">{byDay.size ? "指到标记上看那次发布" : ""}</div>}
      </div>
      <div ref={box} className="h-[360px] w-full" />
    </div>
  );
}
