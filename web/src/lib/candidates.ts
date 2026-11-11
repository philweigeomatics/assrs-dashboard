/**
 * 候选组合 — weight sets saved on the page so several can be compared.
 *
 * They live in this browser only (localStorage), one list per market: they
 * are scratch work towards a portfolio, not the portfolio. 存为组合 is still
 * the way to keep one for real.
 */

import { sameWeights, type WeightDraft } from "./weightDraft";

export type Candidate = {
  id: string;
  name: string;
  /** Percent by ticker; only names held above zero. */
  weights: WeightDraft;
  /** Display names by ticker, so the list reads without a lookup. */
  names: Record<string, string>;
  /** What the weights started from: 最大夏普, 风险平价, … */
  source: string;
  /** The optimiser window it was made in. */
  from: string;
  to: string;
  saved: string;
};

export const MAX_CANDIDATES = 8;

/**
 * One colour per slot, chosen to stay apart from the lines already on the
 * chart: cyan (optimiser), amber (draft), purple (equal) and grey (index).
 * Red and green are left out because on these pages they mean up and down.
 */
export const CANDIDATE_COLOURS = [
  "#3b82f6", "#ec4899", "#0d9488", "#a16207",
  "#6366f1", "#f97316", "#0ea5e9", "#be185d",
];

export const colourOf = (list: Candidate[], id: string) =>
  CANDIDATE_COLOURS[Math.max(0, list.findIndex((c) => c.id === id)) % CANDIDATE_COLOURS.length]
  ?? "#3b82f6";

/** Only what is actually held, rounded the way the inputs are. */
export const heldOnly = (w: WeightDraft): WeightDraft =>
  Object.fromEntries(Object.entries(w)
    .filter(([, v]) => v > 0)
    .map(([t, v]) => [t, Math.round(v * 100) / 100]));

/** "候选 1", "候选 2", … — the first number not already taken. */
export function nextName(list: Candidate[]) {
  const taken = new Set(list.map((c) => c.name));
  for (let i = 1; ; i++) if (!taken.has(`候选 ${i}`)) return `候选 ${i}`;
}

/** Names a candidate holds that the current stock list does not have. */
export const outside = (c: Candidate, tickers: string[]) => {
  const have = new Set(tickers);
  return Object.keys(c.weights).filter((t) => !have.has(t));
};

/** The candidate these weights are, if they are exactly one of them. */
export const matching = (list: Candidate[], w: WeightDraft) =>
  list.find((c) => sameWeights(c.weights, heldOnly(w))) ?? null;

export const newId = () =>
  `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;

export type CurveStats = {
  total: number | null;
  ann: number | null;
  vol: number | null;
  sharpe: number | null;
  maxdd: number | null;
};

/**
 * Realised statistics from a cumulative-return line (in %, starting at 0).
 *
 * Same definitions as the server's stats_of, so a line's row here agrees
 * with 买入持有实测 above it: compounded annual return over 242 sessions a
 * year, sample volatility of the daily returns, Sharpe as (ann − rf) / vol,
 * and drawdown from the running peak. Gaps (NaN) are skipped, not zeroed.
 */
export function curveStats(values: number[], rfPct = 0, perYear = 242): CurveStats {
  const growth = values.filter((v) => Number.isFinite(v)).map((v) => 1 + v / 100);
  const empty = { total: null, ann: null, vol: null, sharpe: null, maxdd: null };
  if (growth.length < 3) return empty;
  const rets: number[] = [];
  for (let i = 1; i < growth.length; i++) rets.push(growth[i]! / growth[i - 1]! - 1);
  const n = rets.length;
  const mean = rets.reduce((a, b) => a + b, 0) / n;
  const variance = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1);
  const vol = Math.sqrt(variance) * Math.sqrt(perYear);
  const whole = growth[growth.length - 1]! / growth[0]!;
  const ann = whole ** (perYear / n) - 1;
  let peak = -Infinity;
  let dd = 0;
  for (const g of growth) {
    peak = Math.max(peak, g);
    dd = Math.min(dd, g / peak - 1);
  }
  return {
    total: (whole - 1) * 100,
    ann: ann * 100,
    vol: vol * 100,
    sharpe: vol > 0 ? (ann - rfPct / 100) / vol : null,
    maxdd: dd * 100,
  };
}
