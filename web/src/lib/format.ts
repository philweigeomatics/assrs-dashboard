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

/** Chinese convention: up red, down green. */
export function moveClass(v: Num | undefined): string {
  if (v == null || !Number.isFinite(v) || v === 0) return "text-flat";
  return v > 0 ? "text-up" : "text-down";
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
