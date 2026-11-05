/**
 * 🧭 风险雷达 helpers that are worth testing apart from the page.
 *
 * The synthesis cites evidence by id ("[#3f2a…]"), not by position, because
 * the evidence is re-ranked by weight every day and a position would point at
 * a different item tomorrow. The page numbers evidence by its CURRENT rank, so
 * an id is turned into that number here — and a citation whose item has aged
 * out of the list is dropped rather than shown pointing at nothing.
 */

import type { RiskEvidence, RiskTier } from "./types";

export type Segment = { text: string } | { cite: number };

const CITE = /\[#([0-9a-f]{6,})\]/g;

export function citations(text: string | null, evidence: RiskEvidence[]): Segment[] {
  if (!text) return [];
  const rank = new Map(evidence.map((e, i) => [e.eid, i + 1]));
  const out: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(CITE)) {
    if (m.index! > last) out.push({ text: text.slice(last, m.index) });
    const n = rank.get(m[1]!);
    if (n !== undefined) out.push({ cite: n });
    last = m.index! + m[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last) });
  return out;
}

/** Tailwind classes for a 等级 chip. Downside reads green: red up, green down. */
export function tierClass(t: RiskTier): string {
  switch (t) {
    case "高": return "bg-down text-white";
    case "偏高": return "bg-[#d9ecdf] text-down";
    case "中": return "bg-elevated text-ink-dim";
    case "证据不足": return "border border-dashed border-line-bright text-ink-mute";
    default: return "bg-sunken text-ink-mute";
  }
}

export function leanClass(lean: string): string {
  return lean === "偏空" ? "text-down" : lean === "偏多" ? "text-up" : "text-ink-mute";
}

/** "x,y x,y …" for an SVG polyline over a 0–100 series. */
export function sparkPoints(values: number[], w: number, h: number): string {
  if (values.length < 2) return "";
  const step = w / (values.length - 1);
  return values
    .map((v, i) => `${(i * step).toFixed(1)},${(h - 2 - (Math.max(0, Math.min(100, v)) / 100) * (h - 4)).toFixed(1)}`)
    .join(" ");
}

export const KIND_LABEL: Record<RiskEvidence["kind"], string> = {
  filing: "公告/申报", data: "数据", qa: "问答", macro: "政策", research: "研报", news: "新闻",
};
