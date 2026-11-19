/**
 * A custom index's members as 组合优化 picks.
 *
 * Only names the index holds now: removed and delisted members are history,
 * not candidates. Halted and pending ones are still members and stay in —
 * the optimiser reports a name it cannot price rather than dropping it
 * silently here.
 *
 * When the index holds more than the optimiser takes, the heaviest go in
 * first, so a cut loses the names that mattered least to the index.
 */

import type { IndexMember, StockRef } from "./types";

const GONE: IndexMember["status"][] = ["removed", "delisted"];

export function indexPicks(members: IndexMember[], cap: number): {
  picks: StockRef[]; dropped: StockRef[];
} {
  const seen = new Set<string>();
  const live = members
    .filter((m) => !GONE.includes(m.status) && !m.removed_on)
    .filter((m) => (seen.has(m.ticker) ? false : (seen.add(m.ticker), true)))
    .sort((a, b) => (b.weight ?? b.target ?? 0) - (a.weight ?? a.target ?? 0))
    .map((m) => ({ t: m.ticker, n: m.name }));
  return { picks: live.slice(0, cap), dropped: live.slice(cap) };
}
