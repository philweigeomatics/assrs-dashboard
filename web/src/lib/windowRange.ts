/**
 * The analysis window as positions in a list of trading days.
 *
 * Saved as dates, because the list grows every day. `to: null` means "up to
 * the latest session", so a window that ends today still ends today tomorrow.
 */

/** The optimiser needs 60 sessions of returns before a covariance means anything. */
export const MIN_WINDOW = 60;
export const DEFAULT_WINDOW = 242;

export type Range = [number, number];
export type SavedWindow = { from: string; to: string | null };

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** The latest `sessions` days, or all of them if there are fewer. */
export function lastSessions(n: number, sessions: number): Range {
  return [Math.max(0, n - 1 - sessions), n - 1];
}

/** First position on or after `date`; `dates.length` if there is none. */
function lowerBound(dates: string[], date: string) {
  let lo = 0;
  let hi = dates.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (dates[mid]! < date) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/**
 * Where a saved window sits on this index's calendar.
 *
 * Indices start on different days (科创50 only exists from 2019-12-31), so a
 * saved window can fall partly or wholly before the one now on screen.
 */
export function resolveRange(dates: string[], saved: SavedWindow | null): Range {
  const n = dates.length;
  if (n === 0) return [0, 0];
  if (!saved || (saved.to && saved.to < dates[0]!) || saved.from > dates[n - 1]!) {
    return lastSessions(n, DEFAULT_WINDOW);
  }
  let i = Math.min(lowerBound(dates, saved.from), n - 1);
  let j = n - 1;
  if (saved.to) {
    const k = lowerBound(dates, saved.to);
    j = k < n && dates[k] === saved.to ? k : Math.max(0, k - 1);
  }
  if (j - i < MIN_WINDOW) {
    j = Math.min(n - 1, i + MIN_WINDOW);
    i = Math.max(0, j - MIN_WINDOW);
  }
  return [i, j];
}

export const toSaved = (dates: string[], [i, j]: Range): SavedWindow => ({
  from: dates[i]!,
  to: j >= dates.length - 1 ? null : dates[j]!,
});

/** Slide the window, keeping its length, stopping at either end of the data. */
export function moveWindow([i, j]: Range, by: number, n: number): Range {
  const len = j - i;
  const start = clamp(i + by, 0, Math.max(0, n - 1 - len));
  return [start, start + len];
}

export function resizeStart([, j]: Range, to: number): Range {
  return [clamp(to, 0, Math.max(0, j - MIN_WINDOW)), j];
}

export function resizeEnd([i]: Range, to: number, n: number): Range {
  return [i, clamp(to, Math.min(i + MIN_WINDOW, n - 1), n - 1)];
}
