/**
 * The Technical Analysis chart: a price pane plus indicator panes.
 *
 * Each pane is its OWN chart instance, which buys independently resizable
 * panes, a per-pane legend and real separation; the cost is wiring them
 * together, which is most of this file.
 *
 * Invariant: ONE POINT PER BAR in every series, nulls sent as whitespace.
 * Panes sync scroll via logical ranges and the plugins draw by logical index;
 * a series that dropped its leading nulls would put bar 48 at index 23.
 *
 * Three things live ON TOP of that, and are deliberately separate so one can
 * be cleared without disturbing the others:
 *   * the GHOST — a hypothetical next bar, drawn dashed, redrawn on every
 *     What-If edit without rebuilding the charts;
 *   * the COMPARISON — a second stock, in either scaling;
 *   * DRAWINGS — your measure boxes and levels, stored as bar+price so they
 *     stay put through zoom and resize.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  type Logical,
  type LogicalRange,
  type MouseEventParams,
  type SeriesMarker,
  type SeriesType,
  type Time,
} from "lightweight-charts";
import type { Analysis, CompareResult, SimResult } from "../../lib/types";
import {
  buildPanes,
  defaultHidden,
  type LineSpec,
  type MarkerSpec,
  type PaneSpec,
} from "../../lib/panes";
import { compact } from "../../lib/format";
import { BandsPrimitive, BoxesPrimitive } from "./primitives";
import { DrawingsPrimitive, newId, type Drawing } from "./drawings";
import { Legend } from "./Legend";
import { Readout } from "./Readout";
import { ResizablePane } from "./ResizablePane";
import { usePersistentState } from "../../lib/usePersistentState";

const UP = "#ff3b30";
const DOWN = "#34c759";
const GRID = "#eef0f3";
const AXIS = "#8a8a8e";
const SCALE_WIDTH = 72;
const GHOST_DASH = LineStyle.LargeDashed;

export type Tool = "none" | "zoom" | "measure" | "hline";

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
    const up = spec.signBy === "body" ? (d.ohlcv.c[i] ?? 0) > (d.ohlcv.o[i] ?? 0) : v >= 0;
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
  return out.sort((a, b) => (a.time < b.time ? -1 : a.time > b.time ? 1 : 0));
}

export function ChartStack({
  data,
  ghost,
  compare,
  compareMode,
  tool,
  onToolDone,
  resetSignal,
  drawings,
  setDrawings,
}: {
  data: Analysis;
  ghost: SimResult | null;
  compare: CompareResult | null;
  compareMode: "pct" | "price";
  tool: Tool;
  /** Fired after a one-shot tool (measure, hline) completes a drawing. */
  onToolDone: () => void;
  /** Incrementing counter: bump it to snap the view back to the default range. */
  resetSignal: number;
  /**
   * Owned by the page, not here. Two usePersistentState hooks over one key are
   * two independent states that overwrite each other: the toolbar's count
   * never moved and "clear" was undone by this component writing its own copy
   * back.
   */
  drawings: Drawing[];
  setDrawings: (fn: (d: Drawing[]) => Drawing[]) => void;
}) {
  const { main, subs } = useMemo(() => buildPanes(data), [data]);
  const panes = useMemo(() => [main, ...subs], [main, subs]);

  const [hidden, setHidden] = usePersistentState<string[]>(
    "assrs.chart.hidden.v1", defaultHidden(main, subs));
  const [heights, setHeights] = usePersistentState<Record<string, number>>(
    "assrs.chart.heights.v1", {});
  const [hover, setHover] = useState<number | null>(null);
  const [rubber, setRubber] = useState<null | { x1: number; y1: number; x2: number; y2: number }>(null);

  const hosts = useRef<Record<string, HTMLDivElement | null>>({});
  const handles = useRef<Handles>({ charts: new Map(), series: new Map(), markers: new Map() });
  const ghostSeries = useRef<Map<string, ISeriesApi<SeriesType, Time>>>(new Map());
  const cmpSeries = useRef<Map<string, ISeriesApi<SeriesType, Time>>>(new Map());
  const drawingsRef = useRef<Drawing[]>(drawings);
  drawingsRef.current = drawings;

  const defaultRange = useCallback((): LogicalRange => {
    const n = data.dates.length;
    return { from: Math.max(0, n - data.initial_visible) as Logical, to: (n + 3) as Logical };
  }, [data]);

  // ── build every pane ──────────────────────────────────────────────────
  useEffect(() => {
    const h: Handles = { charts: new Map(), series: new Map(), markers: new Map() };
    handles.current = h;
    ghostSeries.current = new Map();
    cmpSeries.current = new Map();
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
        candles.setData(data.dates.map((t, i) => ({
          time: t as Time,
          open: data.ohlcv.o[i] ?? NaN, high: data.ohlcv.h[i] ?? NaN,
          low: data.ohlcv.l[i] ?? NaN, close: data.ohlcv.c[i] ?? NaN,
        })));
        h.series.set("main:__candles", candles);
        anchor = candles;
        if (pane.bands.length) candles.attachPrimitive(new BandsPrimitive(pane.bands));
        if (data.boxes.length) candles.attachPrimitive(new BoxesPrimitive(data.boxes));
        candles.attachPrimitive(
          new DrawingsPrimitive(() => drawingsRef.current) as never);
      }

      drawPane(chart, pane, data, h);
      if (!anchor) anchor = h.series.get(`${pane.id}:${pane.lines[0]?.key}`) ?? null;
      if (!isMain && anchor && pane.bands.some((b) => b.segments.length)) {
        anchor.attachPrimitive(new BandsPrimitive(pane.bands));
      }

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

    const firstSeries = new Map<IChartApi, ISeriesApi<SeriesType, Time>>();
    for (const [id, c] of h.charts) {
      const s = id === "main"
        ? h.series.get("main:__candles")
        : [...h.series.entries()].find(([k]) => k.startsWith(`${id}:`))?.[1];
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

    for (const c of charts) c.timeScale().setVisibleLogicalRange(defaultRange());

    return () => {
      offRange.forEach((f) => f());
      offCross.forEach((f) => f());
      for (const c of charts) c.remove();
      handles.current = { charts: new Map(), series: new Map(), markers: new Map() };
      ghostSeries.current = new Map();
      cmpSeries.current = new Map();
    };
  }, [data, panes, defaultRange]);

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

  // ── the ghost bar ─────────────────────────────────────────────────────
  // Its own series, added and removed on their own, so a What-If edit never
  // rebuilds the charts (which would throw away your zoom) and clearing the
  // ghost leaves everything else untouched.
  useEffect(() => {
    const h = handles.current;
    const gs = ghostSeries.current;

    const clear = () => {
      for (const [key, s] of gs) {
        const paneId = key.split(":")[0]!;
        h.charts.get(paneId)?.removeSeries(s);
      }
      gs.clear();
    };

    if (!ghost) {
      clear();
      return;
    }
    clear();

    const lastDate = data.dates[data.dates.length - 1];
    const gDate = ghost.date as Time;
    if (!lastDate) return;

    const mainChart = h.charts.get("main");
    if (mainChart && ghost.ohlcv.c != null) {
      const gc = mainChart.addSeries(CandlestickSeries, {
        upColor: "rgba(255,59,48,0.35)", downColor: "rgba(52,199,89,0.35)",
        borderUpColor: UP, borderDownColor: DOWN,
        wickUpColor: "rgba(255,59,48,0.6)", wickDownColor: "rgba(52,199,89,0.6)",
        priceFormat: { type: "price", precision: 2, minMove: 0.01 },
        lastValueVisible: false, priceLineVisible: false,
      });
      gc.setData([{
        time: gDate,
        open: ghost.ohlcv.o ?? ghost.ohlcv.c, high: ghost.ohlcv.h ?? ghost.ohlcv.c,
        low: ghost.ohlcv.l ?? ghost.ohlcv.c, close: ghost.ohlcv.c,
      }]);
      gs.set("main:__ghostcandle", gc);
    }

    // One dashed two-point segment per indicator: last real value → ghost.
    for (const pane of panes) {
      const chart = h.charts.get(pane.id);
      if (!chart) continue;
      for (const line of pane.lines) {
        const gv = ghost.series[line.key as keyof typeof ghost.series];
        if (gv == null) continue;
        const prev = line.values[line.values.length - 1];
        if (prev == null) continue;
        const s = chart.addSeries(LineSeries, {
          color: line.color, lineWidth: 2, lineStyle: GHOST_DASH,
          priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
          ...(line.scaleId ? { priceScaleId: line.scaleId } : {}),
        });
        s.setData([
          { time: lastDate as Time, value: prev },
          { time: gDate, value: gv },
        ]);
        gs.set(`${pane.id}:__ghost_${line.key}`, s);
      }
      // Volume gets a ghost bar rather than a line.
      if (pane.id === "volume" && ghost.ohlcv.v != null) {
        const s = chart.addSeries(HistogramSeries, {
          priceLineVisible: false, lastValueVisible: false,
        });
        const up = (ghost.ohlcv.c ?? 0) > (ghost.ohlcv.o ?? 0);
        s.setData([{ time: gDate, value: ghost.ohlcv.v, color: up ? "rgba(239,68,68,0.35)" : "rgba(34,197,94,0.35)" }]);
        gs.set("volume:__ghostvol", s);
      }
    }
  }, [ghost, data, panes]);

  // ── the comparison stock ──────────────────────────────────────────────
  useEffect(() => {
    const h = handles.current;
    const cs = cmpSeries.current;
    const chart = h.charts.get("main");
    if (!chart) return;

    for (const [, s] of cs) chart.removeSeries(s);
    cs.clear();
    if (!compare) {
      // Hide the left axis too. Leaving it visible after ✕ keeps an empty
      // ~50px strip that squeezes the candles for no reason.
      chart.priceScale("left").applyOptions({ visible: false });
      return;
    }

    // Both scalings are built; the toggle only flips visibility, so switching
    // is instant and needs no refetch. "pct" rides the main ¥ axis rebased to
    // this stock's first close; "price" gets its own left axis.
    const rebased = chart.addSeries(LineSeries, {
      color: "#7c3aed", lineWidth: 2, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false,
      visible: compareMode === "pct",
    });
    rebased.setData(lineData(data.dates, compare.rebased));
    cs.set("rebased", rebased);

    const price = chart.addSeries(LineSeries, {
      color: "#7c3aed", lineWidth: 2, priceLineVisible: false,
      lastValueVisible: true, crosshairMarkerVisible: false,
      priceScaleId: "left", visible: compareMode === "price",
    });
    price.setData(lineData(data.dates, compare.price));
    cs.set("price", price);
    chart.priceScale("left").applyOptions({ visible: compareMode === "price", borderColor: GRID });
  }, [compare, compareMode, data]);

  // ── reset view ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!resetSignal) return;
    for (const c of handles.current.charts.values()) {
      c.timeScale().setVisibleLogicalRange(defaultRange());
    }
  }, [resetSignal, defaultRange]);

  // Redraw the drawings layer when the list changes.
  useEffect(() => {
    handles.current.series.get("main:__candles")?.applyOptions({});
  }, [drawings]);

  const toggle = (id: string) =>
    setHidden((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

  const idx = hover ?? data.dates.length - 1;

  // ── pointer tools on the price pane ───────────────────────────────────
  function onLayerPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    if (tool === "none") return;
    const chart = handles.current.charts.get("main");
    const candles = handles.current.series.get("main:__candles");
    if (!chart || !candles) return;
    const box = e.currentTarget.getBoundingClientRect();
    const start = { x: e.clientX - box.left, y: e.clientY - box.top };
    e.currentTarget.setPointerCapture?.(e.pointerId);

    if (tool === "hline") {
      const p = candles.coordinateToPrice(start.y);
      if (p != null) setDrawings((d) => [...d, { id: newId(), kind: "hline", p1: p }]);
      onToolDone();
      return;
    }

    const move = (ev: PointerEvent) => {
      setRubber({
        x1: start.x, y1: start.y,
        x2: ev.clientX - box.left, y2: ev.clientY - box.top,
      });
    };
    const up = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      setRubber(null);
      const end = { x: ev.clientX - box.left, y: ev.clientY - box.top };
      if (Math.abs(end.x - start.x) < 4 && Math.abs(end.y - start.y) < 4) {
        onToolDone();
        return;
      }
      const l1 = chart.timeScale().coordinateToLogical(Math.min(start.x, end.x));
      const l2 = chart.timeScale().coordinateToLogical(Math.max(start.x, end.x));
      if (l1 == null || l2 == null) {
        onToolDone();
        return;
      }
      const maxBar = data.dates.length - 1;
      const clamp = (v: number) => Math.max(0, Math.min(maxBar, Math.round(v)));
      if (tool === "zoom") {
        for (const c of handles.current.charts.values()) {
          c.timeScale().setVisibleLogicalRange({ from: l1, to: l2 });
        }
      } else if (tool === "measure") {
        const p1 = candles.coordinateToPrice(start.y);
        const p2 = candles.coordinateToPrice(end.y);
        if (p1 != null && p2 != null) {
          // Clamped to real bars: a drag past the right edge maps to a
          // logical index beyond the data, and a box anchored there would
          // float in empty space.
          setDrawings((d) => [...d, {
            id: newId(), kind: "measure",
            i1: clamp(l1), i2: clamp(l2), p1, p2,
          }]);
        }
      }
      onToolDone();
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  const layer =
    tool === "none" ? null : (
      <div
        onPointerDown={onLayerPointerDown}
        className={`absolute inset-0 z-20 ${tool === "hline" ? "cursor-pointer" : "cursor-crosshair"}`}
        style={{ pointerEvents: "auto" }}
      >
        {rubber && (
          <div
            className={`absolute border-2 ${tool === "zoom" ? "border-cyan bg-cyan/10" : "border-brand bg-brand/10"}`}
            style={{
              left: Math.min(rubber.x1, rubber.x2),
              top: tool === "zoom" ? 0 : Math.min(rubber.y1, rubber.y2),
              width: Math.abs(rubber.x2 - rubber.x1),
              height: tool === "zoom" ? "100%" : Math.abs(rubber.y2 - rubber.y1),
            }}
          />
        )}
      </div>
    );

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
          layer={pane.id === "main" ? layer : null}
          overlay={
            <>
              {pane.id === "main" && (
                <Readout data={data} index={idx} ghost={ghost} compare={compare} />
              )}
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
        ...common, color: line.color, lineWidth: line.width ?? 1,
        lineStyle: styleOf(line.style), crosshairMarkerVisible: false,
      });
      s.setData(lineData(d.dates, line.values));
    }
    if (line.scaleId && line.scaleMargins) {
      chart.priceScale(line.scaleId).applyOptions({ scaleMargins: line.scaleMargins });
    }
    h.series.set(`${pane.id}:${line.key}`, s);
  }

  if (pane.ribbon?.length) {
    const pos = new Map(pane.ribbon.map((r) => [r.i, r.color]));
    const s = chart.addSeries(HistogramSeries, {
      priceScaleId: "ribbon", priceLineVisible: false, lastValueVisible: false,
    });
    chart.priceScale("ribbon").applyOptions({ scaleMargins: { top: 0.93, bottom: 0 } });
    s.setData(d.dates.map((t, i) => {
      const c = pos.get(i);
      return c ? { time: t as Time, value: 1, color: c } : { time: t as Time };
    }));
  }
}
