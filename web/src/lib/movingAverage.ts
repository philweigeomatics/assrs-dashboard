/** The moving averages a level chart can show, each in its own colour (none red or green). */
export const MA_LINES = [
  { n: 50, color: "#2563eb" },
  { n: 120, color: "#9333ea" },
  { n: 200, color: "#b45309" },
] as const;
export type MaN = (typeof MA_LINES)[number]["n"];

/** Trailing mean of the last `n` values; null until there are `n`, and across a gap. */
export function movingAverage(vs: (number | null)[], n: number): (number | null)[] {
  const out: (number | null)[] = [];
  let sum = 0, run = 0;
  for (let i = 0; i < vs.length; i++) {
    const v = vs[i];
    if (v == null) { sum = 0; run = 0; out.push(null); continue; }
    sum += v; run++;
    if (run > n) sum -= vs[i - n]!;
    out.push(run >= n ? sum / n : null);
  }
  return out;
}
