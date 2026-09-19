/**
 * Add or remove the stock on screen from your watchlist.
 *
 * The watchlist is what the nightly scan walks and what every screen is built
 * on — 今日提醒, 做T候选, 反转候选 — so this is the one control that decides
 * what the rest of the app has anything to say about.
 *
 * Every market, now that there are two nightly runs rather than one. A-shares
 * are scanned at 20:00 Beijing through Tushare; US and Canadian names after
 * the 16:00 ET close through Yahoo. One table holds both — the canonical
 * symbol says which market a row is — and each job takes only its own, so a
 * US ticker can no longer break the 20:00 run.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";

export function WatchlistButton({ ticker }: { ticker: string }) {
  const qc = useQueryClient();
  // Every market in one call: the button only needs to know whether THIS
  // symbol is on the list, and scoping the query would make it miss a row
  // that is there.
  const list = useQuery({
    queryKey: ["watchlist", "all"], queryFn: () => api.watchlist(), staleTime: 60_000,
  });

  const inList = (list.data ?? []).some((s) => s.t === ticker);

  const toggle = useMutation({
    mutationFn: () => (inList ? api.watchlistRemove(ticker) : api.watchlistAdd(ticker)),
    // Refetch rather than patching locally: the server decides what the list
    // is, and a failed add must not leave the button lying about it.
    onSettled: () => qc.invalidateQueries({ queryKey: ["watchlist"] }),
  });

  return (
    <div className="flex items-center gap-2 shrink-0">
      <button
        onClick={() => toggle.mutate()}
        disabled={toggle.isPending || list.isPending}
        title={inList ? "从自选股移除" : "加入自选股 — 夜间扫描与今日提醒都基于自选股"}
        className={`h-7 px-2.5 rounded-md text-[12.5px] border transition-colors disabled:opacity-60 ${
          inList
            ? "border-line bg-sunken text-ink-dim hover:bg-elevated"
            : "border-cyan bg-cyan text-white"
        }`}
      >
        {toggle.isPending ? "…" : inList ? "★ 已在自选股" : "☆ 加入自选股"}
      </button>
      {toggle.isError && (
        <span className="text-[11.5px] text-up">{(toggle.error as Error).message}</span>
      )}
    </div>
  );
}
