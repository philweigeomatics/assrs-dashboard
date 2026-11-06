/**
 * How big a day's move is, judged against the days on screen.
 *
 * "Up 5%" is a huge day for a utility and an ordinary one for a small-cap
 * chip stock, and an ordinary day in a quiet year can be a big one in a wild
 * year. So the yardstick is the daily changes of the bars currently in view:
 * zoom out to five years and a move is ranked against five years, zoom into a
 * month and it is ranked against that month.
 *
 * Changes are close-to-close on the chart's own (adjusted) closes, so an
 * ex-dividend or 送转 day is not a fake drop.
 */

import type { Num } from "./types";

/** Fewer days than this and a percentile says more about luck than the stock. */
export const MIN_DAYS = 20;

/** Close-to-close change in %, one per bar; the first bar (no previous close) is null. */
export function dailyChanges(close: Num[]): Num[] {
  return close.map((c, i) => {
    const p = i > 0 ? close[i - 1] : null;
    return c != null && p != null && p > 0 && Number.isFinite(c) && Number.isFinite(p)
      ? (c / p - 1) * 100
      : null;
  });
}

export type MoveStats = {
  /** First and last bar index of the window, inclusive. */
  from: number;
  to: number;
  /** Days in the window that have a change. */
  n: number;
  sorted: number[];
  sortedAbs: number[];
  p5: number;
  p95: number;
  /** Median size of a day's move, up or down. */
  medianAbs: number;
};

/** Linear-interpolated quantile of an ascending array, q in [0, 1]. */
export function quantile(sorted: number[], q: number): number {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (pos - lo);
}

/** Count of values strictly below x, and equal to x, in an ascending array. */
function below(sorted: number[], x: number): [number, number] {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (sorted[m]! < x) lo = m + 1;
    else hi = m;
  }
  let eq = lo;
  while (eq < sorted.length && sorted[eq] === x) eq++;
  return [lo, eq - lo];
}

/**
 * Percentile rank, 0–100: the share of the window's days below x, a tie
 * counting half (so the median day ranks 50, not 51 or 49).
 */
export function rankOf(sorted: number[], x: number): number {
  if (!sorted.length) return NaN;
  const [lt, eq] = below(sorted, x);
  return ((lt + eq / 2) / sorted.length) * 100;
}

/**
 * The window's statistics, for the bars between two logical positions of the
 * chart's time scale (fractional, and running past either end of the data
 * when the view is scrolled to the edge). Null when the window holds fewer
 * than MIN_DAYS changes.
 */
export function windowStats(changes: Num[], from: number, to: number): MoveStats | null {
  const a = Math.max(0, Math.ceil(from));
  const b = Math.min(changes.length - 1, Math.floor(to));
  const vals: number[] = [];
  for (let i = a; i <= b; i++) {
    const v = changes[i];
    if (v != null && Number.isFinite(v)) vals.push(v);
  }
  if (vals.length < MIN_DAYS) return null;
  const sorted = [...vals].sort((x, y) => x - y);
  const sortedAbs = vals.map(Math.abs).sort((x, y) => x - y);
  return {
    from: a, to: b, n: vals.length, sorted, sortedAbs,
    p5: quantile(sorted, 0.05),
    p95: quantile(sorted, 0.95),
    medianAbs: quantile(sortedAbs, 0.5),
  };
}

export type MoveRank = {
  change: number;
  /** Where the day sits among the window's changes, 0 = the worst day, 100 = the best. */
  rank: number;
  /** Share of the window's days that moved LESS, either way. */
  sizeRank: number;
  /** The move in multiples of the window's median move. */
  multiple: number;
};

export function rankDay(stats: MoveStats | null, change: Num): MoveRank | null {
  if (!stats || change == null || !Number.isFinite(change)) return null;
  return {
    change,
    rank: rankOf(stats.sorted, change),
    sizeRank: rankOf(stats.sortedAbs, Math.abs(change)),
    multiple: stats.medianAbs > 0 ? Math.abs(change) / stats.medianAbs : NaN,
  };
}

/** Words for a move's size, from the share of days that moved less. */
export function sizeWord(sizeRank: number): string {
  if (sizeRank >= 99) return "极端";
  if (sizeRank >= 95) return "很大";
  if (sizeRank >= 80) return "偏大";
  if (sizeRank >= 40) return "平常";
  return "偏小";
}
