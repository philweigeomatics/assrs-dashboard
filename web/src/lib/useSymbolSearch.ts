/**
 * Finding an instrument, wherever it lives.
 *
 * A-shares filter in the browser — all 5,600 arrive once from /stocks, so
 * every keystroke is free. US and Canadian listings have no equivalent list to
 * download, so they are fetched, debounced, and only once the query is long
 * enough to be a real search rather than a half-typed code.
 *
 * `markets` scopes the search. Pass a single market to keep the results inside
 * it — the comparison box does that, because beta against two different
 * indices in two different currencies is not a comparison, and offering a US
 * stock as the match for an A-share would only produce a rejected request.
 */

import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { suggest, type Suggestion } from "./search";
import type { HistoryRef, MarketCode, StockRef } from "./types";

/** Below this, a query is still being typed and every keystroke is a request. */
export const MIN_REMOTE_CHARS = 2;
const DEBOUNCE_MS = 300;

export function useSymbolSearch({
  query,
  stocks,
  history = [],
  markets = ["CN", "US", "CA"],
  limit = 16,
}: {
  query: string;
  stocks: StockRef[];
  history?: HistoryRef[];
  markets?: MarketCode[];
  limit?: number;
}): { items: Suggestion[]; searching: boolean } {
  const wantCN = markets.includes("CN");
  const remoteMarkets = markets.filter((m) => m !== "CN") as ("US" | "CA")[];

  const local = useMemo(
    () => (wantCN ? suggest(query, stocks, history, limit) : []),
    [wantCN, query, stocks, history, limit]);

  const [debounced, setDebounced] = useState("");
  useEffect(() => {
    const q = query.trim();
    const id = window.setTimeout(
      () => setDebounced(q.length >= MIN_REMOTE_CHARS ? q : ""), DEBOUNCE_MS);
    return () => window.clearTimeout(id);
  }, [query]);

  const remote = useQuery({
    queryKey: ["search", debounced, remoteMarkets.join(",")],
    queryFn: async () => {
      const lists = await Promise.all(
        // One failing market must not empty the other's results.
        remoteMarkets.map((m) => api.search(debounced, m).catch(() => [])));
      return lists.flat();
    },
    enabled: remoteMarkets.length > 0 && debounced.length >= MIN_REMOTE_CHARS,
    staleTime: 5 * 60_000,
  });

  const items = useMemo<Suggestion[]>(() => {
    const seen = new Set(local.map((s) => s.t));
    const extra: Suggestion[] = (remote.data ?? [])
      .filter((s) => !seen.has(s.t))
      .map((s) => ({ ...s, fromHistory: false }));
    // A-shares first: they are the ones that filter instantly. Foreign hits
    // append rather than interleave, so the list never reshuffles under the
    // cursor when the request lands.
    return [...local, ...extra].slice(0, limit);
  }, [local, remote.data, limit]);

  return { items, searching: remote.isFetching };
}
