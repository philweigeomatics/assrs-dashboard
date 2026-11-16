/**
 * 自建指数 as daily candles, from its start date to today — and, while the
 * US session runs, today's candle live at the end.
 *
 * Open and close are the stored ones. High and low come from the members:
 * what the index held at the previous close, each holding moved to its own
 * high (low) of the day (index_api.candles / custom_index.candle_range).
 * Members seldom peak at the same minute, so a wick is the most the index
 * could have reached — the usual convention for an index built from daily
 * bars. Today's candle is the same rule on Yahoo's 5-minute bars, so once
 * the close is stored it turns into the same candle.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  CandlestickSeries, ColorType, createChart, createSeriesMarkers, CrosshairMode, HistogramSeries,
  LineSeries, PriceScaleMode, type IChartApi, type ISeriesApi, type Time,
} from "lightweight-charts";
import { api } from "../../lib/api";
import type { CustomIndexDetail } from "../../lib/types";
import type { LiveRead } from "../../lib/indexLive";
import { fixed, moveClass, signed } from "../../lib/format";

export const MA_LINES = [
  { n: 50, color: "#2563eb" },
  { n: 120, color: "#9333ea" },
  { n: 200, color: "#b45309" },
] as const;
export type MaN = (typeof MA_LINES)[number]["n"];

/** Trailing mean of the last `n` values; null until there are `n`, and across a gap. */
export function movingAverage(vs: (number | null)[], n: number): (number | null)[] {
  const out: (number | null)[] = [];
  let sum = 0, run = 0;
  for (let i = 0; i < vs.length; i++) {
    const v = vs[i];
    if (v == null) { sum = 0; run = 0; out.push(null); continue; }
    sum += v; run++;
    if (run > n) sum -= vs[i - n]!;
    out.push(run >= n ? sum / n : null);
  }
  return out;
}

const RED = "#ff3b30", GREEN = "#34c759", GRID = "#eef0f3", AXIS = "#8a8a8e";

type Bar = { date: string; o: number; h: number; l: number; c: number; amount: number | null; live?: boolean };

export function IndexKChart({ d, lv, mas, log }: {
  d: CustomIndexDetail; lv: LiveRead | null; mas: MaN[]; log: boolean;
}) {
  const red = d.index.market === "CN";
  const [up, down] = red ? [RED, GREEN] : [GREEN, RED];
  const q = useQuery({
    queryKey: ["idx", "candles", d.index.id, d.as_of],
    queryFn: () => api.indexCandles(d.index.id),
    staleTime: 30 * 60_000,
  });

  const bars = useMemo<Bar[]>(() => {
    const k = q.data;
    if (!k) return [];
    const out: Bar[] = [];
    k.dates.forEach((date, i) => {
      const c = k.close[i];
      if (c == null) return;
      out.push({ date, c, o: k.open[i] ?? c, h: k.high[i] ?? c, l: k.low[i] ?? c, amount: k.amount[i] ?? null });
    });
    const last = out[out.length - 1];
    if (lv?.bar && last && lv.session > last.date) {
      out.push({ date: lv.session, o: lv.bar.open, h: lv.bar.high, l: lv.bar.low, c: lv.level, amount: null, live: true });
    }
    return out;
  }, [q.data, lv]);

  const maLines = useMemo(() => {
    const closes = bars.map((b) => b.c);
    return Object.fromEntries(MA_LINES.map((m) => [m.n, movingAverage(closes, m.n)])) as Record<MaN, (number | null)[]>;
  }, [bars]);

  const box = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const candle = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const vol = useRef<ISeriesApi<"Histogram"> | null>(null);
  const maSeries = useRef<Map<number, ISeriesApi<"Line">>>(new Map());
  const fitted = useRef(false);
  const [hover, setHover] = useState<number | null>(null);

  // The chart itself, once.
  useEffect(() => {
    if (!box.current) return;
    const c = createChart(box.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: AXIS, fontSize: 10,
                attributionLogo: false },
      grid: { vertLines: { color: GRID }, horzLines: { color: GRID } },
      rightPriceScale: { borderColor: GRID, scaleMargins: { top: 0.06, bottom: 0.2 } },
      timeScale: { borderColor: GRID, rightOffset: 3 },
      crosshair: { mode: CrosshairMode.Normal,
                   vertLine: { color: "#0062cc88", labelBackgroundColor: "#0062cc" },
                   horzLine: { color: "#0062cc88", labelBackgroundColor: "#0062cc" } },
    });
    vol.current = c.addSeries(HistogramSeries, { priceScaleId: "vol", priceFormat: { type: "volume" },
                                                lastValueVisible: false, priceLineVisible: false });
    c.priceScale("vol").applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });
    candle.current = c.addSeries(CandlestickSeries, {
      priceFormat: { type: "price", precision: 2, minMove: 0.01 },
    });
    c.subscribeCrosshairMove((p) => {
      const i = p.logical;
      setHover(p.time != null && i != null ? Math.round(i) : null);
    });
    chart.current = c;
    const ma = maSeries.current;
    return () => { c.remove(); chart.current = null; ma.clear(); fitted.current = false; };
  }, []);

  useEffect(() => {
    candle.current?.applyOptions({ upColor: up, downColor: down, borderUpColor: up, borderDownColor: down,
                                   wickUpColor: up, wickDownColor: down });
  }, [up, down]);

  useEffect(() => {
    chart.current?.priceScale("right").applyOptions({
      mode: log ? PriceScaleMode.Logarithmic : PriceScaleMode.Normal });
  }, [log]);

  // Data: every stored day, and today's while it runs — hollow, so a candle
  // still forming is not read as a settled one.
  useEffect(() => {
    const c = chart.current, s = candle.current, v = vol.current;
    if (!c || !s || !v || !bars.length) return;
    s.setData(bars.map((b) => ({
      time: b.date as Time, open: b.o, high: b.h, low: b.l, close: b.c,
      ...(b.live ? { color: "rgba(0,0,0,0)" } : {}),
    })));
    v.setData(bars.map((b) => ({
      time: b.date as Time, value: b.amount ?? 0,
      color: (b.c >= b.o) ? (red ? "rgba(255,59,48,.28)" : "rgba(52,199,89,.28)")
                          : (red ? "rgba(52,199,89,.28)" : "rgba(255,59,48,.28)"),
    })));
    const markers = createSeriesMarkers(s, []);
    if (d.live_from) {
      const at = bars.find((b) => b.date >= d.live_from!);
      if (at) markers.setMarkers([{ time: at.date as Time, position: "aboveBar", shape: "arrowDown",
                                    color: "#8a8a8e", text: "实盘起" }]);
    }
    if (!fitted.current) { c.timeScale().fitContent(); fitted.current = true; }
    return () => { markers.detach(); };
  }, [bars, red, d.live_from]);

  // The moving averages, one line each, only the ones switched on.
  useEffect(() => {
    const c = chart.current;
    if (!c) return;
    const have = maSeries.current;
    for (const m of MA_LINES) {
      const on = mas.includes(m.n);
      let s = have.get(m.n);
      if (!on) { if (s) { c.removeSeries(s); have.delete(m.n); } continue; }
      if (!s) {
        s = c.addSeries(LineSeries, { color: m.color, lineWidth: 1, priceLineVisible: false,
                                      lastValueVisible: false, crosshairMarkerVisible: false });
        have.set(m.n, s);
      }
      const line = maLines[m.n];
      s.setData(bars.flatMap((b, i) => (line[i] != null ? [{ time: b.date as Time, value: line[i]! }] : [])));
    }
  }, [mas, maLines, bars]);

  const i = hover != null && hover >= 0 && hover < bars.length ? hover : bars.length - 1;
  const b = bars[i];
  const prev = i > 0 ? bars[i - 1] : undefined;
  const chg = b && prev ? (b.c / prev.c - 1) * 100 : null;
  const partial = q.data?.partial ?? [];

  return (
    <div className="flex flex-col gap-1 min-w-0">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-[12px] tnum min-h-[18px]">
        {b && <>
          <span className="font-mono">{b.date}{b.live ? (lv?.open ? ` 盘中 ${lv.as_of ?? ""}` : " 未入库") : ""}</span>
          <span>开 <b>{fixed(b.o, 2)}</b></span>
          <span>高 <b>{fixed(b.h, 2)}</b></span>
          <span>低 <b>{fixed(b.l, 2)}</b></span>
          <span>收 <b className={moveClass(chg, red)}>{fixed(b.c, 2)}</b></span>
          {chg != null && <b className={moveClass(chg, red)}>{signed(chg, 2, "%")}</b>}
          {MA_LINES.filter((m) => mas.includes(m.n)).map((m) => (
            <span key={m.n} style={{ color: m.color }}>MA{m.n} {fixed(maLines[m.n][i], 2)}</span>
          ))}
        </>}
      </div>
      <div ref={box} className="relative w-full h-[300px] sm:h-[360px]">
        {q.isPending && <div className="absolute inset-0 grid place-items-center label">读取成分股的每日高低点…</div>}
        {q.isError && <div className="absolute inset-0 grid place-items-center text-[13px] text-up px-4 text-center">
          {(q.error as Error).message}</div>}
      </div>
      <p className="label leading-snug">
        开盘、收盘为存下的点位；最高、最低 = 前一收盘的持仓按各成分股当日最高 / 最低价计算 —— 成分股很少在同一时刻见顶，
        所以影线是指数当天<b>最多</b>能到的位置。{lv?.bar ? "最后一根空心 K 线是今天的，盘中每分钟更新，收盘入库后变成实心。" : ""}
        {partial.length > 0 && ` 有 ${partial.length} 天部分成分缺少高低点，按其收盘价计入（如 ${partial[partial.length - 1]}）。`}
      </p>
    </div>
  );
}
