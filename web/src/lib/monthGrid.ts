/**
 * A month laid out as Monday-first weeks, for a calendar grid.
 *
 * Its own module because the arithmetic is the part that breaks quietly:
 * a month starting on Sunday needs six leading blanks rather than none, and
 * getting that wrong shifts every disclosure in the grid by a day without
 * looking broken.
 */

/** Padding cells, before the 1st and after the last, are 0. */
export function monthGrid(year: number, month: number): number[][] {
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
export function shiftMonth(ym: string, by: number): string {
  const [y, m] = ym.split("-").map(Number);
  const next = new Date(y ?? 1970, (m ?? 1) - 1 + by, 1);
  return `${next.getFullYear()}-${String(next.getMonth() + 1).padStart(2, "0")}`;
}

/** The "YYYY-MM-DD" key for one cell, matching the API's date strings. */
export function cellDate(ym: string, day: number): string {
  return `${ym}-${String(day).padStart(2, "0")}`;
}
