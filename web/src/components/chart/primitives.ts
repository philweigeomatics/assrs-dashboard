/**
 * Canvas plugins for the two things lightweight-charts has no series for:
 * background bands (volatility regime, large-trend shading) and 箱体
 * rectangles with their edge zones.
 *
 * Both position by LOGICAL index (bar number), not by date. Every series in a
 * pane has one point per bar, so logical index i IS bar i — the same invariant
 * the pane-sync relies on — and logicalToCoordinate works for bars scrolled
 * out of view, where a date lookup would return null.
 */

import type {
  IChartApiBase,
  IPrimitivePaneRenderer,
  IPrimitivePaneView,
  ISeriesApi,
  ISeriesPrimitive,
  Logical,
  SeriesAttachedParameter,
  SeriesType,
  Time,
} from "lightweight-charts";
import type { ChartBox } from "../../lib/types";
import type { BandSpec } from "../../lib/panes";

type Target = Parameters<IPrimitivePaneRenderer["draw"]>[0];

abstract class BasePrimitive implements ISeriesPrimitive<Time> {
  protected chart: IChartApiBase<Time> | null = null;
  protected series: ISeriesApi<SeriesType, Time> | null = null;
  private readonly views: IPrimitivePaneView[];

  constructor(zOrder: "bottom" | "normal" | "top") {
    const renderer: IPrimitivePaneRenderer = {
      draw: (target) => {
        if (zOrder !== "bottom") this.paint(target);
      },
      drawBackground: (target) => {
        if (zOrder === "bottom") this.paint(target);
      },
    };
    this.views = [{ zOrder: () => zOrder, renderer: () => renderer }];
  }

  attached(p: SeriesAttachedParameter<Time>): void {
    this.chart = p.chart;
    this.series = p.series;
  }

  detached(): void {
    this.chart = null;
    this.series = null;
  }

  paneViews(): readonly IPrimitivePaneView[] {
    return this.views;
  }

  /** Left and right pixel edges of bars [from, to], or null if unplaceable. */
  protected span(from: number, to: number): [number, number] | null {
    if (!this.chart) return null;
    const ts = this.chart.timeScale();
    const x0 = ts.logicalToCoordinate(from as Logical);
    const x1 = ts.logicalToCoordinate(to as Logical);
    if (x0 == null || x1 == null) return null;
    const half = ts.options().barSpacing / 2;
    return [x0 - half, x1 + half];
  }

  protected abstract paint(target: Target): void;
}

/** Full-height background bands: regime shading, large-trend shading. */
export class BandsPrimitive extends BasePrimitive {
  constructor(private readonly bands: BandSpec[]) {
    super("bottom");
  }

  protected paint(target: Target): void {
    target.useMediaCoordinateSpace(({ context: ctx, mediaSize }) => {
      for (const band of this.bands) {
        ctx.fillStyle = band.color;
        for (const seg of band.segments) {
          const x = this.span(seg.from, seg.to);
          if (!x) continue;
          const left = Math.max(0, x[0]);
          const right = Math.min(mediaSize.width, x[1]);
          if (right > left) ctx.fillRect(left, 0, right - left, mediaSize.height);
        }
      }
    });
  }
}

/**
 * 箱体 rectangles. A box gets a translucent body, dashed outline and a grey
 * strip at each edge showing the touch ZONE (support is a band of prices, not
 * one number). Channels have no rectangle — a box drawn over a 通道 implies
 * edges nobody defended — so they get a label only.
 */
export class BoxesPrimitive extends BasePrimitive {
  constructor(private readonly boxes: ChartBox[]) {
    super("normal");
  }

  protected paint(target: Target): void {
    const series = this.series;
    if (!series) return;
    target.useMediaCoordinateSpace(({ context: ctx }) => {
      ctx.save();
      ctx.font = "11px -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif";
      for (const b of this.boxes) {
        const x = this.span(b.from, b.to);
        if (!x) continue;
        const [left, right] = x;

        if (b.kind !== "BOX" || b.top == null || b.bot == null) {
          const label = `${b.status_cn}${b.drift_pct != null ? ` ${b.drift_pct > 0 ? "+" : ""}${b.drift_pct}%` : ""}`;
          ctx.fillStyle = "rgba(100,116,139,0.9)";
          ctx.fillText(label, left + 4, 14);
          continue;
        }

        const yTop = series.priceToCoordinate(b.top);
        const yBot = series.priceToCoordinate(b.bot);
        if (yTop == null || yBot == null) continue;
        const top = Math.min(yTop, yBot);
        const h = Math.abs(yBot - yTop);

        ctx.fillStyle = b.is_active ? "rgba(255,149,0,0.10)" : "rgba(255,149,0,0.06)";
        ctx.fillRect(left, top, right - left, h);

        if (b.zone != null) {
          const zTop = series.priceToCoordinate(b.top - b.zone);
          const zBot = series.priceToCoordinate(b.bot + b.zone);
          ctx.fillStyle = "rgba(107,114,128,0.16)";
          if (zTop != null) ctx.fillRect(left, top, right - left, Math.abs(zTop - yTop));
          if (zBot != null) ctx.fillRect(left, Math.min(zBot, yBot), right - left, Math.abs(yBot - zBot));
        }

        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = b.is_active ? "rgba(234,88,12,0.75)" : "rgba(0,0,0,0.3)";
        ctx.lineWidth = 1;
        ctx.strokeRect(left + 0.5, top + 0.5, right - left - 1, h - 1);
        ctx.setLineDash([]);

        const label =
          `箱体 ${b.bot.toFixed(2)}–${b.top.toFixed(2)}  ` +
          `触${b.touches_top ?? 0}/${b.touches_bot ?? 0}  质${b.quality?.toFixed(2) ?? "—"}  ${b.status_cn}`;
        const w = ctx.measureText(label).width + 8;
        ctx.fillStyle = "rgba(255,255,255,0.88)";
        ctx.fillRect(right - w, top - 16, w, 15);
        ctx.fillStyle = b.is_active ? "#b34700" : "#3c3c43";
        ctx.fillText(label, right - w + 4, top - 5);
      }
      ctx.restore();
    });
  }
}
