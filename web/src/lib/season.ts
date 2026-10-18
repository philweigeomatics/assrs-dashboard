/**
 * Helpers for the seasonality panel. Every year lives on one leap-year
 * calendar: slot 0 is Jan 1, slot 59 is Feb 29, slot 365 is Dec 31 — the
 * same month-day is the same slot in every year.
 */

export type Point = [number, number];

/** First slot of each month on the leap calendar. */
export const MONTH_SLOTS = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335];
export const MONTHS = ["1月", "2月", "3月", "4月", "5月", "6月",
  "7月", "8月", "9月", "10月", "11月", "12月"];

/** "MM-DD" for a slot. */
export function slotLabel(slot: number): string {
  const s = Math.max(0, Math.min(365, Math.round(slot)));
  let m = 0;
  while (m < 11 && MONTH_SLOTS[m + 1]! <= s) m++;
  const d = s - MONTH_SLOTS[m]! + 1;
  return `${String(m + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
}

/** The week block a slot falls in (fixed 7-day blocks from Jan 1; the last runs to Dec 31). */
export function weekOf(slot: number): number {
  return Math.min(Math.floor(slot / 7), 51);
}

/**
 * Weekly returns as a cumulative path, one point at the end of each block.
 * Starts at [0, 0] — the previous year's close — and stops at the first
 * missing week, which is where this year has not got to yet.
 */
export function weeklyPath(weekly: (number | null)[]): Point[] {
  const out: Point[] = [[0, 0]];
  let lvl = 1;
  for (let k = 0; k < weekly.length; k++) {
    const v = weekly[k];
    if (v == null || !Number.isFinite(v)) break;
    lvl *= 1 + v / 100;
    out.push([k === 51 ? 365 : 7 * k + 6, +((lvl - 1) * 100).toFixed(2)]);
  }
  return out;
}

/**
 * The path's value on a slot: the last point at or before it, as a price
 * holds through a closed market. null before the first point, and null
 * after the last one, so this year's line does not claim days it has not seen.
 */
export function valueAt(path: Point[], slot: number): number | null {
  if (!path.length || slot < path[0]![0] || slot > path[path.length - 1]![0] + 6) return null;
  let lo = 0;
  let hi = path.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (path[mid]![0] <= slot) lo = mid; else hi = mid - 1;
  }
  return path[lo]![1];
}

/**
 * A cell's fill for a move, as a share of the market's own up or down
 * colour over the panel. `clamp` is the move that gets full strength.
 * Direction follows the market: red is a rise on an A-share page and a fall
 * on a North American one.
 */
export function cellFill(v: number | null | undefined, clamp: number, upIsRed: boolean): string {
  if (v == null || !Number.isFinite(v) || v === 0) return "var(--color-panel-alt)";
  const rise = upIsRed ? "var(--color-up)" : "var(--color-down)";
  const fall = upIsRed ? "var(--color-down)" : "var(--color-up)";
  const share = Math.round(12 + 70 * Math.min(1, Math.abs(v) / clamp));
  return `color-mix(in srgb, ${v > 0 ? rise : fall} ${share}%, var(--color-panel))`;
}

/** Text on a cell: white once the fill is dark enough to need it. */
export function cellInk(v: number | null | undefined, clamp: number): string {
  return v != null && Math.abs(v) / clamp > 0.6 ? "#fff" : "var(--color-ink-dim)";
}

/**
 * Line colours by recency. The three most recent full years get validated
 * categorical hues (blue, amber, violet — checked for colour-blind
 * separation, and kept clear of the red/green that mean up and down here);
 * older years recede to grey so ten lines stay readable; this year is ink.
 */
export const YEAR_HUES = ["#0062cc", "#b45309", "#7c3aed"];
export const OLD_YEAR = "#b4b4bb";

export function yearColour(year: number, fullYears: number[], current: number): string {
  if (year === current) return "var(--color-ink)";
  const recent = [...fullYears].sort((a, b) => b - a);
  const i = recent.indexOf(year);
  return i >= 0 && i < YEAR_HUES.length ? YEAR_HUES[i]! : OLD_YEAR;
}
