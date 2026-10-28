import { describe, expect, it } from "vitest";
import { livePairs, progress, sparkPath } from "./cycle";
import type { CyclePair, CycleRow } from "./types";

const row = (sector: string, phase: CycleRow["phase"]): CycleRow => ({
  sector, beta: null, osc: 0, slope: 0, phase, days: 3, typical: 6,
  typical_all: {}, last_cross: null, spark: [],
});
const pair = (a: string, b: string, q: number): CyclePair => ({
  a, b, best_lag: 0, best_corr: -0.5, corr0: -0.5, p: 0.001, q, opposite: 0.7, opposite_expected: 0.5,
});

describe("cycle helpers", () => {
  it("shows only proven pairs, and says whether they are on opposite sides now", () => {
    const board = [row("银行", "topping"), row("半导体", "bottoming"), row("白酒", "up")];
    const live = livePairs([pair("银行", "半导体", 0.02), pair("银行", "白酒", 0.05), pair("白酒", "半导体", 0.4)], board);
    expect(live.map((x) => [x.pair.a, x.pair.b, x.opposite])).toEqual([
      ["银行", "半导体", true],
      ["银行", "白酒", false],
    ]);
  });
  it("measures progress against the sector's own typical stay", () => {
    expect(progress({ days: 3, typical: 6 })).toBe(0.5);
    expect(progress({ days: 3, typical: null })).toBeNull();
  });
  it("draws a sparkline that breaks over gaps and clamps outliers", () => {
    const d = sparkPath([0, 10, null, -10], 30, 10);
    expect(d.startsWith("M0.0 5.0")).toBe(true);
    expect(d).toContain("L10.0 0.0");
    expect(d).toContain("M30.0 10.0");
  });
});
