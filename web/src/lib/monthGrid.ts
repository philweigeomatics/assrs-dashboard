/**
 * A month laid out as Monday-first weeks, for a calendar grid.
 *
 * Its own module because the arithmetic is the part that breaks quietly:
 * a month starting on Sunday needs six leading blanks rather than none, and
 * getting that wrong shifts every disclosure in the grid by a day.
 *
 * Everything here is total. The first version was not, and it took the whole
 * 日历 route down with "Invalid array length":
 *
 *   · `Number("undefined")` is NaN, and `??` does not catch NaN — only null
 *     and undefined — so a missing month string sailed past the fallback and
 *     reached `Array(NaN)`, which throws.
 *   · `shiftMonth` on a bad input built its result from an Invalid Date and
 *     returned the literal string "NaN-NaN", so one bad value poisoned every
 *     later render rather than failing once.
 *
 * A calendar with no data to show should show an empty month, not an error
 * page, so a value that cannot be parsed falls back to the current month.
 */

/** "YYYY-MM", or null if this is not one. */
export function parseMonth(ym: unknown): { year: number; month: number } | null {
  if (typeof ym !== "string") return null;
  const m = /^(\d{4})-(\d{1,2})/.exec(ym.trim());
  if (!m) return null;
  const year = Number(m[1]);
  const month = Number(m[2]);
  if (!Number.isInteger(year) || year < 1000 || year > 9999) return null;
  if (!Number.isInteger(month) || month < 1 || month > 12) return null;
  return { year, month };
}

/** The current month, as the fallback for anything unparseable. */
export function thisMonth(now: Date = new Date()): string {
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`;
}

/** Whatever was given, or the current month — never something that throws. */
export function safeMonth(ym: unknown, now: Date = new Date()): string {
  const got = parseMonth(ym);
  return got ? `${got.year}-${String(got.month).padStart(2, "0")}`
             : thisMonth(now);
}

/** Padding cells, before the 1st and after the last, are 0. */
export function monthGrid(ym: unknown): number[][] {
  const { year, month } = parseMonth(ym) ?? parseMonth(thisMonth())!;

  const first = new Date(year, month - 1, 1);
  const days = new Date(year, month, 0).getDate();
  // JS weeks start on Sunday; these grids start on Monday.
  const lead = (first.getDay() + 6) % 7;

  const cells: number[] = Array(lead).fill(0);
  for (let d = 1; d <= days; d += 1) cells.push(d);
  while (cells.length % 7) cells.push(0);

  const weeks: number[][] = [];
  for (let i = 0; i < cells.length; i += 7) weeks.push(cells.slice(i, i + 7));
  return weeks;
}

/** "2026-08" → the month before or after, wrapping the year. */
export function shiftMonth(ym: unknown, by: number): string {
  const { year, month } = parseMonth(ym) ?? parseMonth(thisMonth())!;
  const step = Number.isFinite(by) ? Math.trunc(by) : 0;
  const next = new Date(year, month - 1 + step, 1);
  return `${next.getFullYear()}-${String(next.getMonth() + 1).padStart(2, "0")}`;
}

/** The "YYYY-MM-DD" key for one cell, matching the API's date strings. */
export function cellDate(ym: unknown, day: number): string {
  return `${safeMonth(ym)}-${String(day).padStart(2, "0")}`;
}
