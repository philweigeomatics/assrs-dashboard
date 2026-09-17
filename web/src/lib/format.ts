import type { Num } from "./types";

export const dash = "—";

export function fixed(v: Num | undefined, nd = 2): string {
  return v == null || !Number.isFinite(v) ? dash : v.toFixed(nd);
}

export function signed(v: Num | undefined, nd = 2, suffix = ""): string {
  if (v == null || !Number.isFinite(v)) return dash;
  return `${v > 0 ? "+" : ""}${v.toFixed(nd)}${suffix}`;
}

/** 亿 with sensible precision: 7788.8亿 → "7,789亿", 12.3亿 → "12.3亿". */
export function yi(v: Num | undefined): string {
  if (v == null || !Number.isFinite(v)) return dash;
  if (Math.abs(v) >= 10000) return `${(v / 10000).toFixed(2)}万亿`;
  if (Math.abs(v) >= 100) return `${Math.round(v).toLocaleString()}亿`;
  return `${v.toFixed(1)}亿`;
}

/**
 * Colour for a move. `upIsRed` defaults to the A-share convention, so every
 * existing call site keeps its behaviour; North American pages pass false.
 *
 * The CSS tokens are named for the DIRECTION (--color-up / --color-down), not
 * the colour, so this swaps which token a rise gets rather than swapping the
 * tokens themselves.
 */
export function moveClass(v: Num | undefined, upIsRed = true): string {
  if (v == null || !Number.isFinite(v) || v === 0) return "text-flat";
  const rising = v > 0;
  return rising === upIsRed ? "text-up" : "text-down";
}

/**
 * A market capitalisation, written the way that market is read.
 *
 * 亿 and 万亿 for A-shares; B and T for North America. Rendering a US company
 * as "48513亿" is not wrong so much as unreadable to anyone who trades it.
 */
export function money(v: Num | undefined, currency: string, symbol = ""): string {
  if (v == null || !Number.isFinite(v)) return dash;
  const a = Math.abs(v);
  if (currency === "CNY") {
    if (a >= 1e12) return `${(v / 1e12).toFixed(2)}万亿`;
    if (a >= 1e10) return `${Math.round(v / 1e8).toLocaleString()}亿`;
    return `${(v / 1e8).toFixed(1)}亿`;
  }
  if (a >= 1e12) return `${symbol}${(v / 1e12).toFixed(2)}T`;
  if (a >= 1e9) return `${symbol}${(v / 1e9).toFixed(1)}B`;
  if (a >= 1e6) return `${symbol}${(v / 1e6).toFixed(0)}M`;
  return `${symbol}${Math.round(v).toLocaleString()}`;
}

/** 3450000 → 3.45M. Keeps every price scale the same width. */
export function compact(value: number): string {
  const a = Math.abs(value);
  if (a >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (a >= 1e4) return `${(value / 1e3).toFixed(1)}K`;
  if (a >= 100) return value.toFixed(1);
  if (a >= 1) return value.toFixed(2);
  return value.toFixed(3);
}
