/**
 * User drawings: the measure box and horizontal levels.
 *
 * Stored as DATA (bar index + price), never as pixels, so they survive zoom,
 * pan and a pane resize — a pixel rectangle would slide off the bars it was
 * drawn against the moment the chart moved.
 *
 * Kept apart from everything the analysis computes (boxes, bands, markers):
 * "clear drawings" must never remove a 箱体 the detector found, and redrawing
 * the analysis must never wipe a measurement.
 */

import type {
  IPrimitivePaneRenderer,
  ISeriesApi,
  Logical,
  SeriesType,
  Time,
} from "lightweight-charts";

export type Drawing =
  | { id: string; kind: "measure"; i1: number; i2: number; p1: number; p2: number }
  | { id: string; kind: "hline"; p1: number };

type Target = Parameters<IPrimitivePaneRenderer["draw"]>[0];

export function newId() {
  return Math.random().toString(36).slice(2, 9);
}

/** % change between the two prices, plus how many bars the box spans. */
export function measureLabel(d: Extract<Drawing, { kind: "measure" }>) {
  const lo = Math.min(d.p1, d.p2);
  const hi = Math.max(d.p1, d.p2);
  const pct = lo ? ((hi - lo) / lo) * 100 : 0;
  const bars = Math.abs(d.i2 - d.i1) + 1;
  // Direction follows the drag: down-to-up reads as a gain, up-to-down a loss.
  const rising = d.p2 >= d.p1;
  return {
    pct: `${rising ? "+" : "−"}${pct.toFixed(2)}%`,
    range: `${lo.toFixed(2)} → ${hi.toFixed(2)}`,
    bars: `${bars} 根`,
    rising,
  };
}

export class DrawingsPrimitive {
  private chart: { timeScale(): { logicalToCoordinate(l: Logical): number | null } } | null = null;
  private series: ISeriesApi<SeriesType, Time> | null = null;
  private readonly views: unknown[];

  constructor(private getDrawings: () => Drawing[]) {
    const renderer: IPrimitivePaneRenderer = { draw: (t) => this.paint(t) };
    this.views = [{ zOrder: () => "top", renderer: () => renderer }];
  }

  attached(p: { chart: unknown; series: unknown }) {
    this.chart = p.chart as never;
    this.series = p.series as never;
  }

  detached() {
    this.chart = null;
    this.series = null;
  }

  paneViews() {
    return this.views as never;
  }

  private paint(target: Target) {
    const chart = this.chart;
    const series = this.series;
    if (!chart || !series) return;
    target.useMediaCoordinateSpace(({ context: ctx, mediaSize }) => {
      ctx.save();
      ctx.font = "11px -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif";
      for (const d of this.getDrawings()) {
        if (d.kind === "hline") {
          const y = series.priceToCoordinate(d.p1);
          if (y == null) continue;
          ctx.strokeStyle = "#0062cc";
          ctx.setLineDash([5, 3]);
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(mediaSize.width, y);
          ctx.stroke();
          ctx.setLineDash([]);
          label(ctx, `${d.p1.toFixed(2)}`, 4, y - 4, "#0062cc");
          continue;
        }

        const x1 = chart.timeScale().logicalToCoordinate(d.i1 as Logical);
        const x2 = chart.timeScale().logicalToCoordinate(d.i2 as Logical);
        const y1 = series.priceToCoordinate(d.p1);
        const y2 = series.priceToCoordinate(d.p2);
        if (x1 == null || x2 == null || y1 == null || y2 == null) continue;

        const left = Math.min(x1, x2);
        const top = Math.min(y1, y2);
        const w = Math.abs(x2 - x1);
        const h = Math.abs(y2 - y1);
        const m = measureLabel(d);
        // A-share convention: a gain is red, a loss is green.
        const stroke = m.rising ? "#dc2626" : "#16a34a";

        ctx.fillStyle = m.rising ? "rgba(220,38,38,0.10)" : "rgba(22,163,74,0.10)";
        ctx.fillRect(left, top, w, h);
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(left + 0.5, top + 0.5, w - 1, h - 1);

        label(ctx, `${m.pct}  ${m.range}  ${m.bars}`, left + 4, top - 5, stroke);
      }
      ctx.restore();
    });
  }
}

function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string) {
  const w = ctx.measureText(text).width + 8;
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.fillRect(x - 2, y - 12, w, 15);
  ctx.fillStyle = color;
  ctx.fillText(text, x + 2, y);
}
