/**
 * Helpers for the 板块周期 tab. The phase colours follow the A-share
 * convention — red rising, green falling — with the two turning phases in
 * the in-between hues, and every chip also carries its word, so colour is
 * never the only way to tell them apart.
 */
import type { CyclePair, CyclePhase, CycleRow } from "./types";

export const PHASE_ORDER: CyclePhase[] = ["up", "topping", "down", "bottoming"];

export const PHASE_LABEL: Record<CyclePhase, string> = {
  up: "上行", topping: "回落", down: "下行", bottoming: "回升",
};

export const PHASE_HINT: Record<CyclePhase, string> = {
  up: "在 20 日均线上方，且还在走高",
  topping: "还在均线上方，但已开始回落",
  down: "在均线下方，且还在走低",
  bottoming: "还在均线下方，但已开始回升",
};

/** Tailwind classes for a phase chip. */
export const PHASE_CLASS: Record<CyclePhase, string> = {
  up: "bg-up/15 text-up",
  topping: "bg-brand/15 text-brand-ink",
  down: "bg-down/15 text-down",
  bottoming: "bg-cyan/15 text-cyan",
};

/** How far through its typical stay a sector is, 0..1+ (null when unknown). */
export function progress(r: Pick<CycleRow, "days" | "typical">): number | null {
  return r.typical ? r.days / r.typical : null;
}

const above = (p: CyclePhase | null) => p === "up" || p === "topping";

/**
 * Proven see-saw pairs (q ≤ `q`) and whether they are on opposite sides of
 * their trends right now — the list that ties the history to today.
 */
export function livePairs(pairs: CyclePair[], board: CycleRow[], q = 0.1) {
  const by = new Map(board.map((r) => [r.sector, r]));
  return pairs
    .filter((p) => p.q != null && p.q <= q)
    .map((p) => {
      const a = by.get(p.a), b = by.get(p.b);
      const opposite = !!a?.phase && !!b?.phase && above(a.phase) !== above(b.phase);
      return { pair: p, a, b, opposite };
    });
}

/** SVG path for a sparkline over [lo, hi], skipping gaps. */
export function sparkPath(vals: (number | null)[], w: number, h: number, lo = -3, hi = 3): string {
  const n = vals.length;
  let d = "", pen = false;
  vals.forEach((v, i) => {
    if (v == null || !Number.isFinite(v)) { pen = false; return; }
    const x = n < 2 ? 0 : (i / (n - 1)) * w;
    const c = Math.max(lo, Math.min(hi, v));
    const y = h - ((c - lo) / (hi - lo)) * h;
    d += `${pen ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
    pen = true;
  });
  return d;
}
