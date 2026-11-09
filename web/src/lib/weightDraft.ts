/** A set of portfolio weights being edited, as percentages by ticker. */
export type WeightDraft = Record<string, number>;

export const TOLERANCE = 0.1;

export const weightsOf = (holdings: { t: string; weight_pct: number }[]): WeightDraft =>
  Object.fromEntries(holdings.map((h) => [h.t, h.weight_pct]));

export const sumOf = (w: WeightDraft) => Object.values(w).reduce((a, b) => a + b, 0);

export const isBalanced = (w: WeightDraft) => Math.abs(sumOf(w) - 100) <= TOLERANCE;

/** A name absent from one side counts as 0, which is what it would be held at. */
export function sameWeights(a: WeightDraft, b: WeightDraft) {
  return Object.keys({ ...a, ...b }).every(
    (t) => Math.abs((a[t] ?? 0) - (b[t] ?? 0)) <= 0.001);
}

/** Equal weights to two decimals, with the last name taking the rounding. */
export function equalWeights(tickers: string[]): WeightDraft {
  const each = Math.round((100 / tickers.length) * 100) / 100;
  return Object.fromEntries(tickers.map((t, i) => [t,
    i === tickers.length - 1
      ? Math.round((100 - each * (tickers.length - 1)) * 100) / 100
      : each]));
}
