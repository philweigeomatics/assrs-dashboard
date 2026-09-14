/**
 * The Technical Analysis chart: a price pane plus indicator panes.
 *
 * Each pane is its OWN chart instance (as in BlindlyTrade), which buys
 * independently resizable panes, a per-pane legend, and real separation.
 * The cost is wiring them together, which is what most of this file does.
 *
 * Invariant: ONE POINT PER BAR in every series, nulls sent as whitespace.
 * Panes sync scroll via logical ranges and the plugins draw by logical index;
 * a series that dropped its leading nulls would put bar 48 at index 23 and
 * the pane would render shifted.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  CandlestickSeries,
  ColorType,
  createChart,
  createSeriesMarkers,
  CrosshairMode,
  HistogramSeries,
  LineSeries,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type LogicalRange,
  type MouseEventParams,
  type SeriesMarker,
  type SeriesType,
  type Time,
} from "lightweight-charts";
import type { Analysis } from "../../lib/types";
import {
  buildPanes,
  defaultHidden,
  type LineSpec,
  type MarkerSpec,
  type PaneSpec,
} from "../../lib/panes";
import { compact } from "../../lib/format";
import { BandsPrimitive, BoxesPrimitive } from "./primitives";
import { Legend } from "./Legend";
import { Readout } from "./Readout";
import { ResizablePane } from "./ResizablePane";
import { usePersistentState } from "../../lib/usePersistentState";

const UP = "#ff3b30";
const DOWN = "#34c759";
const GRID = "#eef0f3";
const AXIS = "#8a8a8e";
const SCALE_WIDTH = 72;

type Handles = {
  charts: Map<string, IChartApi>;
  series: Map<string, ISeriesApi<SeriesType, Time>>;
  markers: Map<string, { plugin: ISeriesMarkersPluginApi<Time>; specs: MarkerSpec[] }>;
};

function baseOptions(showTime: boolean, compactScale: boolean) {
  return {
    autoSize: true,
    layout: {
      background: { type: ColorType.Solid, color: "transparent" },
      textColor: AXIS,
      fontSize: 10,
      attributionLogo: false,
    },
    grid: { vertLines: { color: GRID }, horzLines: { color: GRID } },
    rightPriceScale: { borderColor: GRID, minimumWidth: SCALE_WIDTH },
    timeScale: { borderColor: GRID, visible: showTime, rightOffset: 3 },
    crosshair: {
      mode: CrosshairMode.Normal,
      vertLine: { color: "#0062cc88", labelBackgroundColor: "#0062cc" },
      horzLine: { color: "#0062cc88", labelBackgroundColor: "#0062cc" },
    },
    localization: compactScale ? { priceFormatter: compact } : {},
  } as const;
}

const styleOf = (s: LineSpec["style"]) =>
  s === "dashed" ? LineStyle.Dashed : s === "dotted" ? LineStyle.Dotted : LineStyle.Solid;

function lineData(dates: string[], values: (number | null)[]) {
  return dates.map((t, i) => {
    const v = values[i];
    return v != null && Number.isFinite(v) ? { time: t as Time, value: v } : { time: t as Time };
  });
}

function histData(d: Analysis, spec: LineSpec) {
  const [pos, neg] = spec.signColors ?? [UP, DOWN];
  return d.dates.map((t, i) => {
    const v = spec.values[i];
    if (v == null || !Number.isFinite(v)) return { time: t as Time };
    const up =
      spec.signBy === "body"
        ? (d.ohlcv.c[i] ?? 0) > (d.ohlcv.o[i] ?? 0)
        : v >= 0;
    return { time: t as Time, value: v, color: up ? pos : neg };
  });
}

function markerList(d: Analysis, specs: MarkerSpec[]): SeriesMarker<Time>[] {
  const out: SeriesMarker<Time>[] = [];
  for (const m of specs) {
    for (const i of m.idx) {
      const t = d.dates[i];
      if (!t) continue;
      out.push({ time: t as Time, position: m.position, shape: m.shape, color: m.color, text: m.text });
    }
  }
  // Markers must be in time order or lightweight-charts drops them.
  return out.sort((a, b) => (a.time < b.time ? -1 : a.time > b.time ? 1 : 0));
}

export function ChartStack({ data }: { data: Analysis }) {
  const { main, subs } = useMemo(() => buildPanes(data), [data]);
  const panes = useMemo(() => [main, ...subs], [main, subs]);

  const [hidden, setHidden] = usePersistentState<string[]>(
    "assrs.chart.hidden.v1",
    defaultHidden(main, subs),
  );
  const [heights, setHeights] = usePersistentState<Record<string, number>>(
    "assrs.chart.heights.v1",
    {},
  );
  const [hover, setHover] = useState<number | null>(null);

  const hosts = useRef<Record<string, HTMLDivElement | null>>({});
  const handles = useRef<Handles>({ charts: new Map(), series: new Map(), markers: new Map() });

  // ── build every pane ──────────────────────────────────────────────────
  useEffect(() => {
    const h: Handles = { charts: new Map(), series: new Map(), markers: new Map() };
    handles.current = h;
    const n = data.dates.length;
    if (n === 0) return;

    panes.forEach((pane, pi) => {
      const el = hosts.current[pane.id];
      if (!el) return;
      const isMain = pane.id === "main";
      const chart = createChart(el, baseOptions(pi === panes.length - 1, !isMain));
      h.charts.set(pane.id, chart);

      let anchor: ISeriesApi<SeriesType, Time> | null = null;

      if (isMain) {
        const candles = chart.addSeries(CandlestickSeries, {
          upColor: UP, downColor: DOWN, borderUpColor: UP, borderDownColor: DOWN,
          wickUpColor: UP, wickDownColor: DOWN,
          priceFormat: { type: "price", precision: 2, minMove: 0.01 },
        });
        candles.setData(
          data.dates.map((t, i) => ({
            time: t as Time,
            open: data.ohlcv.o[i] ?? NaN, high: data.ohlcv.h[i] ?? NaN,
            low: data.ohlcv.l[i] ?? NaN, close: data.ohlcv.c[i] ?? NaN,
          })),
        );
        h.series.set("main:__candles", candles);
        anchor = candles;
        if (pane.bands.length) candles.attachPrimitive(new BandsPrimitive(pane.bands));
        if (data.boxes.length) candles.attachPrimitive(new BoxesPrimitive(data.boxes));
      }

      drawPane(chart, pane, data, h);
      if (!anchor) anchor = h.series.get(`${pane.id}:${pane.lines[0]?.key}`) ?? null;
      if (!isMain && anchor && pane.bands.some((b) => b.segments.length)) {
        anchor.attachPrimitive(new BandsPrimitive(pane.bands));
      }

      // Markers, grouped by the series they ride on.
      const byAnchor = new Map<string, MarkerSpec[]>();
      for (const m of pane.markers) {
        const key = isMain ? "main:__candles" : `${pane.id}:${m.anchor ?? pane.lines[0]?.key}`;
        byAnchor.set(key, [...(byAnchor.get(key) ?? []), m]);
      }
      for (const [seriesKey, specs] of byAnchor) {
        const s = h.series.get(seriesKey);
        if (!s) continue;
        h.markers.set(seriesKey, { plugin: createSeriesMarkers(s, []), specs });
      }
    });

    // Keep every pane on the same bars. Guard the echo: applying a range to
    // a chart fires that chart's own subscriber.
    const charts = [...h.charts.values()];
    let syncing = false;
    const offRange = charts.map((c) => {
      const fn = (r: LogicalRange | null) => {
        if (!r || syncing) return;
        syncing = true;
        for (const o of charts) if (o !== c) o.timeScale().setVisibleLogicalRange(r);
        syncing = false;
      };
      c.timeScale().subscribeVisibleLogicalRangeChange(fn);
      return () => c.timeScale().unsubscribeVisibleLogicalRangeChange(fn);
    });

    // One crosshair: moving over any pane draws the vertical line on all of
    // them and points every legend at the same bar.
    const firstSeries = new Map<IChartApi, ISeriesApi<SeriesType, Time>>();
    for (const [id, c] of h.charts) {
      const s = id === "main" ? h.series.get("main:__candles") : [...h.series.entries()].find(([k]) => k.startsWith(`${id}:`))?.[1];
      if (s) firstSeries.set(c, s);
    }
    let crossSyncing = false;
    const offCross = charts.map((c) => {
      const fn = (p: MouseEventParams<Time>) => {
        if (crossSyncing) return;
        const i = p.logical != null ? Math.round(p.logical) : null;
        setHover(i != null && i >= 0 && i < n ? i : null);
        crossSyncing = true;
        for (const o of charts) {
          if (o === c) continue;
          const s = firstSeries.get(o);
          if (p.time != null && s) o.setCrosshairPosition(0, p.time, s);
          else o.clearCrosshairPosition();
        }
        crossSyncing = false;
      };
      c.subscribeCrosshairMove(fn);
      return () => c.unsubscribeCrosshairMove(fn);
    });

    const from = Math.max(0, n - data.initial_visible);
    for (const c of charts) c.timeScale().setVisibleLogicalRange({ from, to: n + 3 });

    return () => {
      offRange.forEach((f) => f());
      offCross.forEach((f) => f());
      for (const c of charts) c.remove();
      handles.current = { charts: new Map(), series: new Map(), markers: new Map() };
    };
  }, [data, panes]);

  // ── visibility, without rebuilding the charts ─────────────────────────
  useEffect(() => {
    const off = new Set(hidden);
    for (const [id, s] of handles.current.series) {
      if (id.endsWith(":__candles")) continue;
      s.applyOptions({ visible: !off.has(id) });
    }
    for (const [seriesKey, { plugin, specs }] of handles.current.markers) {
      const pane = seriesKey.split(":")[0];
      plugin.setMarkers(markerList(data, specs.filter((m) => !off.has(`${pane}:${m.key}`))));
    }
  }, [hidden, data, panes]);

  const toggle = (id: string) =>
    setHidden((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

  const idx = hover ?? data.dates.length - 1;

  return (
    <div className="flex flex-col gap-1.5">
      {panes.map((pane) => (
        <ResizablePane
          key={pane.id}
          height={heights[pane.id] ?? pane.height}
          minHeight={pane.id === "main" ? 220 : 56}
          onResize={(hgt) => setHeights((prev) => ({ ...prev, [pane.id]: hgt }))}
          hostRef={(el) => {
            hosts.current[pane.id] = el;
          }}
          overlay={
            <>
              {pane.id === "main" && <Readout data={data} index={idx} />}
              <Legend pane={pane} index={idx} hidden={hidden} onToggle={toggle} />
            </>
          }
        />
      ))}
    </div>
  );
}

function drawPane(chart: IChartApi, pane: PaneSpec, d: Analysis, h: Handles) {
  for (const g of pane.guides) {
    const s = chart.addSeries(LineSeries, {
      color: g.color ?? "#d1d5db", lineWidth: 1, lineStyle: LineStyle.Dashed,
      priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
      ...(g.scaleId ? { priceScaleId: g.scaleId } : {}),
    });
    s.setData(d.dates.map((t) => ({ time: t as Time, value: g.value })));
  }

  for (const line of pane.lines) {
    const common = {
      priceLineVisible: false,
      lastValueVisible: false,
      ...(line.scaleId ? { priceScaleId: line.scaleId } : {}),
    };
    let s: ISeriesApi<SeriesType, Time>;
    if (line.kind === "histogram") {
      s = chart.addSeries(HistogramSeries, { ...common, color: line.color });
      s.setData(histData(d, line));
    } else {
      s = chart.addSeries(LineSeries, {
        ...common,
        color: line.color,
        lineWidth: line.width ?? 1,
        lineStyle: styleOf(line.style),
        crosshairMarkerVisible: false,
      });
      s.setData(lineData(d.dates, line.values));
    }
    if (line.scaleId && line.scaleMargins) {
      chart.priceScale(line.scaleId).applyOptions({ scaleMargins: line.scaleMargins });
    }
    h.series.set(`${pane.id}:${line.key}`, s);
  }

  if (pane.ribbon?.length) {
    // The ADX lifecycle ribbon: a thin coloured strip along the pane floor.
    const pos = new Map(pane.ribbon.map((r) => [r.i, r.color]));
    const s = chart.addSeries(HistogramSeries, {
      priceScaleId: "ribbon", priceLineVisible: false, lastValueVisible: false,
    });
    chart.priceScale("ribbon").applyOptions({ scaleMargins: { top: 0.93, bottom: 0 } });
    s.setData(
      d.dates.map((t, i) => {
        const c = pos.get(i);
        return c ? { time: t as Time, value: 1, color: c } : { time: t as Time };
      }),
    );
  }
}
