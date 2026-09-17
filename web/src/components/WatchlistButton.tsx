/**
 * Add or remove the stock on screen from your watchlist.
 *
 * The watchlist is what the nightly scan walks and what every screen is built
 * on — 今日提醒, 做T候选, 反转候选 — so this is the one control that decides
 * what the rest of the app has anything to say about.
 *
 * A-shares only, and the button says so rather than failing on click: the
 * nightly job and all three screens go through Tushare, so a US ticker in the
 * list would break a job that runs at 20:00 with nobody watching.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { MarketCode } from "../lib/types";

export function WatchlistButton({ ticker, market }: { ticker: string; market: MarketCode }) {
  const qc = useQueryClient();
  const list = useQuery({ queryKey: ["watchlist"], queryFn: api.watchlist, staleTime: 60_000 });

  const inList = (list.data ?? []).some((s) => s.t === ticker);

  const toggle = useMutation({
    mutationFn: () => (inList ? api.watchlistRemove(ticker) : api.watchlistAdd(ticker)),
    // Refetch rather than patching locally: the server decides what the list
    // is, and a failed add must not leave the button lying about it.
    onSettled: () => qc.invalidateQueries({ queryKey: ["watchlist"] }),
  });

  if (market !== "CN") {
    return (
      <span className="label shrink-0" title="自选股与夜间扫描目前仅支持 A 股">
        自选股仅支持 A 股
      </span>
    );
  }

  return (
    <div className="flex items-center gap-2 shrink-0">
      <button
        onClick={() => toggle.mutate()}
        disabled={toggle.isPending || list.isPending}
        title={inList ? "从自选股移除" : "加入自选股 — 夜间扫描与各策略都基于自选股"}
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
