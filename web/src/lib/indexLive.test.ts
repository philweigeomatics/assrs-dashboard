import { describe, expect, it } from "vitest";
import { liveFor, withLive } from "./indexLive";
import type { EtfRow, IndexLive, MineBoard } from "./types";

const live = (session: string, open = true): IndexLive => ({
  open, session, as_of: "11:35", now: `${session} 11:35`, ttl: 60, times: ["09:30", "09:35"],
  indexes: [
    { id: 1, name: "A", ref_date: "2026-11-11", ref_level: 1000, level: 1012.5, chg: 1.25, levels: [1001, 1012.5] },
    { id: 2, name: "B", error: "最近一周没有存下的收盘" },
  ],
});

const row = (index_id: number | undefined, date: string): EtfRow => ({
  ticker: `IDX${index_id}`, name: "x", group: "我的指数", price: 1000, date,
  ret: { "1d": -0.4, "5d": 1, "1m": 2, "3m": 3, "1y": 4, ytd: 5 },
  d50: 1, d200: 2, slope200: 0.1, golden: true, cross_days: 10, state: "多头",
  hi52: 1100, lo52: 900, from_high: -9, range_pos: 0.5, stretch_rank: 50, history_years: 1,
  vol20: 15, vol_rank: 40, rs_3m: 1, after: null, spark: [], spark_ma200: [],
  ...(index_id != null ? { index_id } : {}),
} as EtfRow);

describe("liveFor", () => {
  it("replaces the stored close only with a later session", () => {
    expect(liveFor(live("2026-11-12"), 1, "2026-11-11")?.level).toBe(1012.5);
    expect(liveFor(live("2026-11-12"), 1, "2026-11-12")).toBeNull();     // today's close already stored
    expect(liveFor(live("2026-11-11"), 1, "2026-11-11")).toBeNull();     // before the bell: yesterday's session
  });
  it("skips an index the server could not read", () => {
    expect(liveFor(live("2026-11-12"), 2, "2026-11-11")).toBeNull();
    expect(liveFor(live("2026-11-12"), 9, "2026-11-11")).toBeNull();
  });
});

describe("withLive", () => {
  it("swaps price and the 1-day change, and nothing else", () => {
    const mine: MineBoard = { rows: [row(1, "2026-11-11"), row(2, "2026-11-11")], young: [], failed: [] };
    const out = withLive(mine, live("2026-11-12"))!;
    expect(out.rows[0]!.price).toBe(1012.5);
    expect(out.rows[0]!.ret["1d"]).toBe(1.25);
    expect(out.rows[0]!.ret["5d"]).toBe(1);
    expect(out.rows[0]!.d200).toBe(2);
    expect(out.rows[0]!.live?.open).toBe(true);
    expect(out.rows[1]).toBe(mine.rows[1]);
  });
  it("leaves the board alone with no session", () => {
    const mine: MineBoard = { rows: [row(1, "2026-11-11")], young: [], failed: [] };
    expect(withLive(mine, undefined)).toBe(mine);
  });
});
