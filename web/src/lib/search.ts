import type { HistoryRef, StockRef } from "./types";

export type Suggestion = StockRef & { fromHistory: boolean };

/**
 * One list for the combobox, from one query.
 *
 * Empty query → your recent searches, newest first. That is the "pick
 * directly from history" half: focus the box and the list is already there.
 *
 * Typed query → matches from the full list, ranked so the thing you most
 * likely meant is on top:
 *   0  exact code            600519
 *   1  code prefix           6005…
 *   2  name prefix           贵州…
 *   3  name contains         …茅台
 * Within a rank, stocks you searched before come first, then by code. A stock
 * that is in your history is still flagged, so the ⟲ mark shows either way.
 */
export function suggest(
  query: string,
  stocks: StockRef[],
  history: HistoryRef[],
  limit = 12,
): Suggestion[] {
  const q = query.trim().toLowerCase();
  const recent = new Map(history.map((h, i) => [h.t, i]));

  if (!q) {
    return history.slice(0, limit).map((h) => ({ t: h.t, n: h.n, fromHistory: true }));
  }

  const scored: { s: StockRef; rank: number }[] = [];
  for (const s of stocks) {
    const name = s.n.toLowerCase();
    let rank = -1;
    if (s.t === q) rank = 0;
    else if (s.t.startsWith(q)) rank = 1;
    else if (name.startsWith(q)) rank = 2;
    else if (name.includes(q)) rank = 3;
    if (rank >= 0) scored.push({ s, rank });
  }

  scored.sort((a, b) => {
    if (a.rank !== b.rank) return a.rank - b.rank;
    const ha = recent.get(a.s.t) ?? Infinity;
    const hb = recent.get(b.s.t) ?? Infinity;
    if (ha !== hb) return ha - hb;
    return a.s.t.localeCompare(b.s.t);
  });

  return scored.slice(0, limit).map(({ s }) => ({ ...s, fromHistory: recent.has(s.t) }));
}
