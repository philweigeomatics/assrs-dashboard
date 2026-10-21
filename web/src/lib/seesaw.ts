/**
 * Helpers for the 跷跷板 panel. Correlation colours are blue for "move
 * together" and orange for "take turns" — deliberately not red and green,
 * which on this app mean up and down and flip between markets.
 */

export const POS = "#0062cc";
export const NEG = "#c2410c";

/** Pearson correlation of two equal-length arrays, skipping pairs with a gap. */
export function corr(a: (number | null)[], b: (number | null)[]): number | null {
  let n = 0, sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    if (x == null || y == null) continue;
    n++; sa += x; sb += y;
  }
  if (n < 3) return null;
  const ma = sa / n, mb = sb / n;
  let cov = 0, va = 0, vb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    if (x == null || y == null) continue;
    cov += (x - ma) * (y - mb); va += (x - ma) ** 2; vb += (y - mb) ** 2;
  }
  return va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : null;
}

/** Correlation over the trailing `win` periods at every point; null until there are enough. */
export function rollingCorr(a: (number | null)[], b: (number | null)[], win: number): (number | null)[] {
  return a.map((_, i) => (i + 1 < win ? null : corr(a.slice(i + 1 - win, i + 1), b.slice(i + 1 - win, i + 1))));
}

/** A correlation cell's fill: blue for together, orange for opposite, stronger with size. */
export function corrFill(r: number | null | undefined): string {
  if (r == null || !Number.isFinite(r)) return "var(--color-panel-alt)";
  const share = Math.round(8 + 72 * Math.min(1, Math.abs(r)));
  return `color-mix(in srgb, ${r >= 0 ? POS : NEG} ${share}%, var(--color-panel))`;
}

export function corrInk(r: number | null | undefined): string {
  return r != null && Math.abs(r) > 0.55 ? "#fff" : "var(--color-ink)";
}

/** Did one rise while the other fell? Flat periods are not a seesaw. */
export function opposite(x: number | null | undefined, y: number | null | undefined): boolean {
  return x != null && y != null && x !== 0 && y !== 0 && (x > 0) !== (y > 0);
}
