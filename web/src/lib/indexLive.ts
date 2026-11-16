/**
 * The US session live, for every place a custom index is shown.
 *
 * One query, shared by key with 📈 对比: the server rebuilds each US index
 * through the current session from what it held at the previous close
 * (index_api.live) and fetches Yahoo at most once a minute for everyone.
 *
 * A stored close is the record; a live read only REPLACES it on screen when
 * it is for a later session than the one stored — during the session, and in
 * the hour after the bell before the close is written down. Once stored, the
 * two are the same day and the stored one is shown.
 */

import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { EtfRow, IndexLive, MineBoard } from "./types";

export type LiveRead = {
  level: number; chg: number;
  ref_date: string; ref_level: number | null;
  open: boolean; session: string; as_of: string | null;
  times: string[]; levels: (number | null)[];
  members: Record<string, { chg: number | null; contrib: number | null }>;
};

export function useIndexLive(enabled: boolean) {
  return useQuery({
    queryKey: ["idx", "live"], queryFn: api.indexLive, enabled,
    // Every minute while the bell is open; every five otherwise, so a page
    // left open before 9:30 New York picks the session up on its own.
    refetchInterval: (q) => (q.state.data?.open ? 60_000 : 5 * 60_000),
    staleTime: 50_000,
  });
}

/** Index `id`'s live read, or null when it is not newer than the stored close of `storedAsOf`. */
export function liveFor(live: IndexLive | undefined, id: number, storedAsOf: string | null | undefined): LiveRead | null {
  if (!live?.session || !storedAsOf || live.session <= storedAsOf) return null;
  const x = live.indexes.find((y) => y.id === id);
  if (!x || x.error || x.level == null || x.chg == null || !x.ref_date) return null;
  return {
    level: x.level, chg: x.chg, ref_date: x.ref_date, ref_level: x.ref_level ?? null,
    open: live.open, session: live.session, as_of: live.as_of,
    times: live.times ?? [], levels: x.levels ?? [], members: x.members ?? {},
  };
}

/**
 * 我的指数 rows on the dashboard with each US index's price and 1-day change
 * swapped for the live session's. Every other column — the 50/200-day reads,
 * the stretch, the returns over weeks — stays on the stored closes it was
 * computed from; the row says which session its price is.
 */
export function withLive(mine: MineBoard | undefined, live: IndexLive | undefined): MineBoard | undefined {
  if (!mine || !live?.session) return mine;
  const rows: EtfRow[] = mine.rows.map((r) => {
    if (r.index_id == null) return r;
    const lv = liveFor(live, r.index_id, r.date);
    if (!lv) return r;
    return { ...r, price: lv.level, ret: { ...r.ret, "1d": lv.chg },
             live: { open: lv.open, as_of: lv.as_of, session: lv.session, ref_date: lv.ref_date } };
  });
  return { ...mine, rows };
}
