/**
 * The geometry behind the 配对交易 detail panes.
 *
 * Kept out of the component because all of it is the kind of arithmetic that
 * fails silently: a trade marker one index off sits over the wrong day and
 * still looks perfectly reasonable, a zoom window that forgets to clamp
 * indexes the price array out of bounds, and a label collision rule that is
 * off by one draws two labels on top of each other.
 */

import type { PairTrade } from "./types";

/** One trade placed on the shared date axis. -1 = not on it at all. */
export type Span = { i: number; t: PairTrade; a: number; b: number };

/**
 * Locate every trade on the date axis.
 *
 * Entry and exit are dates from the same series the chart draws, so this is
 * an exact lookup rather than a nearest match — a "nearest" here would
 * quietly place a marker on a day the engine never traded. An open trade has
 * no exit yet and runs to the last bar we have.
 */
export function placeTrades(dates: string[], trades: PairTrade[]): Span[] {
  const at = new Map<string, number>();
  dates.forEach((d, i) => at.set(d, i));
  const last = dates.length - 1;
  return trades.map((t, i) => ({
    i,
    t,
    a: at.get(t.entry) ?? -1,
    b: at.get(t.exit) ?? (t.open ? last : -1),
  }));
}

/**
 * The index range to draw: the whole history, or one trade with room round it.
 *
 * The padding is proportional, and never less than eight bars — a two-day
 * trade drawn edge to edge shows the convergence and nothing of the
 * divergence that set it up, which is the half that explains the trade.
 */
export function zoomWindow(sel: Span | null, n: number, minPad = 8): [number, number] {
  if (n <= 0) return [0, 0];
  if (!sel || sel.a < 0) return [0, n - 1];
  const pad = Math.max(minPad, Math.round((sel.b - sel.a) * 0.6));
  return [Math.max(0, sel.a - pad), Math.min(n - 1, Math.max(sel.b, sel.a) + pad)];
}

/**
 * Which trades get a text label rather than a bare dot.
 *
 * Left to right, dropping any that lands within `gap` of the last one drawn.
 * The focused trade is always labelled: it is the one the reader just asked
 * about.
 *
 * `at` picks the coordinate — entry markers and exit markers sit at different
 * places along the axis, so they are two separate rows of labels and have to
 * be thinned separately. `gap` is in the same units `at` returns.
 */
export function pickLabels(shown: Span[], at: (s: Span) => number,
                           focus: number | null, gap: number): Set<number> {
  const out = new Set<number>();
  let last = -Infinity;
  for (const s of [...shown].sort((p, q) => at(p) - at(q))) {
    if (s.a < 0) continue;
    if (s.i === focus || at(s) - last >= gap) {
      out.add(s.i);
      last = at(s);
    }
  }
  return out;
}

/**
 * A gap in on-screen pixels, in the viewBox units the chart is drawn in.
 *
 * The panes stretch to the container (preserveAspectRatio="none"), so one
 * viewBox unit is not one pixel and the ratio changes with every resize.
 * Thinning labels by viewBox units means a narrow pane draws them right on
 * top of each other, which is exactly the case that needed thinning.
 */
export function gapInUnits(px: number, width: number, vw: number): number {
  return width > 0 ? (px / width) * vw : 0;
}

/**
 * Percent change of `px[i]` against `px[base]`.
 *
 * NaN when either bar is missing, so it propagates into a gap in the line
 * instead of a stand-in value that would draw as a real price.
 */
export function rebase(px: number[], base: number, i: number): number {
  const b = px[base];
  const v = px[i];
  if (b == null || v == null || b === 0) return NaN;
  return (v / b - 1) * 100;
}

/** The min/max of both legs over the window, ignoring gaps. */
export function extent(a: number[], b: number[], base: number,
                       lo: number, hi: number): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (let i = lo; i <= hi; i += 1) {
    for (const v of [rebase(a, base, i), rebase(b, base, i)]) {
      if (!Number.isFinite(v)) continue;
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
  }
  return Number.isFinite(min) ? [min, max] : [0, 0];
}
