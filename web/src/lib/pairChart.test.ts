/**
 * The arithmetic under the pair-trade chart, which fails silently when wrong.
 *
 *     npm test -- pairChart
 */

import { describe, expect, it } from "vitest";
import { extent, gapInUnits, pickLabels, placeTrades, rebase, zoomWindow }
  from "./pairChart";
import type { PairTrade } from "./types";

const DATES = Array.from({ length: 20 },
  (_, i) => `2026-01-${String(i + 1).padStart(2, "0")}`);

function trade(entry: string, exit: string, over: Partial<PairTrade> = {}): PairTrade {
  return {
    entry, exit, entry_z: -2.1, exit_z: 0.05,
    direction: "BUY_A", open: false,
    buy_code: "600584", buy_name: "长电科技",
    entry_price: 10, exit_price: 11, pnl_pct: 10,
    a_ret_pct: 10, b_ret_pct: 4, pattern: "BOTH_UP",
    ...over,
  };
}

describe("placeTrades", () => {
  it("puts each marker on the day the engine actually traded", () => {
    const [s] = placeTrades(DATES, [trade("2026-01-05", "2026-01-12")]);
    expect([s!.a, s!.b]).toEqual([4, 11]);
  });

  it("runs an open trade to the last bar rather than dropping it", () => {
    const [s] = placeTrades(DATES, [trade("2026-01-05", "", { open: true })]);
    expect([s!.a, s!.b]).toEqual([4, 19]);
  });

  it("reports a date that is not on the axis instead of guessing a near one", () => {
    // A nearest match would place the marker on a day nothing happened, and
    // the chart would look entirely plausible.
    const [s] = placeTrades(DATES, [trade("2025-12-31", "2026-01-05")]);
    expect(s!.a).toBe(-1);
  });
});

describe("zoomWindow", () => {
  it("shows the whole history when nothing is selected", () => {
    expect(zoomWindow(null, 20)).toEqual([0, 19]);
  });

  it("leaves room either side of the trade", () => {
    const [s] = placeTrades(DATES, [trade("2026-01-08", "2026-01-13")]);
    const [lo, hi] = zoomWindow(s!, 20);
    expect(lo).toBeLessThan(s!.a);
    expect(hi).toBeGreaterThan(s!.b);
  });

  it("never runs off either end of the series", () => {
    const first = placeTrades(DATES, [trade("2026-01-01", "2026-01-02")])[0]!;
    const last = placeTrades(DATES, [trade("2026-01-19", "2026-01-20")])[0]!;
    expect(zoomWindow(first, 20)[0]).toBe(0);
    expect(zoomWindow(last, 20)[1]).toBe(19);
  });

  it("gives a one-day trade a readable window anyway", () => {
    const s = placeTrades(DATES, [trade("2026-01-10", "2026-01-10")])[0]!;
    const [lo, hi] = zoomWindow(s, 20);
    expect(hi - lo).toBeGreaterThanOrEqual(16);
  });

  it("falls back to the full range for a trade it could not place", () => {
    const s = placeTrades(DATES, [trade("2020-01-01", "2020-02-01")])[0]!;
    expect(zoomWindow(s, 20)).toEqual([0, 19]);
  });
});

describe("pickLabels", () => {
  const spans = (...days: number[]) =>
    days.map((d, i) => ({ i, t: trade("", ""), a: d, b: d + 12 }));
  const entry = (s: { a: number }) => s.a * 10;
  const exitAt = (s: { b: number }) => s.b * 10;

  it("drops a label that would land on top of the last one", () => {
    expect(pickLabels(spans(0, 2, 20), entry, null, 96)).toEqual(new Set([0, 2]));
  });

  it("always labels the trade the reader just clicked", () => {
    expect(pickLabels(spans(0, 2, 20), entry, 1, 96).has(1)).toBe(true);
  });

  it("measures the gap from the last label drawn, not the last trade seen", () => {
    // 0, 50, 100: the middle one is dropped, so the third is 100px from the
    // first — far enough. Measuring from the dropped one would lose it too.
    expect(pickLabels(spans(0, 5, 10), entry, null, 96)).toEqual(new Set([0, 2]));
  });

  it("works left to right whatever order the trades arrive in", () => {
    const out = spans(20, 0, 2);
    expect(pickLabels([out[0]!, out[1]!, out[2]!], entry, null, 96))
      .toEqual(new Set([1, 0]));
  });

  it("thins exits by where the exits are, not by where the entries are", () => {
    // Two trades entered far apart but exiting on top of each other: thinned
    // on the entry coordinate, both exit labels would be drawn overlapping.
    const close = [{ i: 0, t: trade("", ""), a: 0, b: 30 },
                   { i: 1, t: trade("", ""), a: 20, b: 31 }];
    expect(pickLabels(close, entry, null, 96)).toEqual(new Set([0, 1]));
    expect(pickLabels(close, exitAt, null, 96)).toEqual(new Set([0]));
  });
});

describe("gapInUnits", () => {
  it("converts a pixel gap into the viewBox units the chart is drawn in", () => {
    // Half the on-screen width of the viewBox: 84px of a 500px pane is 168
    // of 1000 units.
    expect(gapInUnits(84, 500, 1000)).toBeCloseTo(168);
    expect(gapInUnits(84, 1000, 1000)).toBeCloseTo(84);
  });

  it("is a narrower pane that needs the BIGGER gap, not the smaller", () => {
    expect(gapInUnits(84, 280, 1000)).toBeGreaterThan(gapInUnits(84, 900, 1000));
  });

  it("asks for no thinning before the pane has been measured", () => {
    // First render, width 0. A gap of Infinity would label nothing at all.
    expect(gapInUnits(84, 0, 1000)).toBe(0);
  });
});

describe("rebase", () => {
  it("is zero at the base and a percentage elsewhere", () => {
    const px = [10, 11, 9];
    expect(rebase(px, 0, 0)).toBe(0);
    expect(rebase(px, 0, 1)).toBeCloseTo(10);
    expect(rebase(px, 0, 2)).toBeCloseTo(-10);
  });

  it("rebasing at the entry makes the exit value the trade's return", () => {
    const px = [100, 102, 105, 110];
    expect(rebase(px, 1, 3)).toBeCloseTo((110 / 102 - 1) * 100);
  });

  it("is NaN rather than a number when a bar is missing", () => {
    expect(rebase([10, 11], 0, 7)).toBeNaN();
    expect(rebase([0, 11], 0, 1)).toBeNaN();
  });
});

describe("extent", () => {
  it("covers both legs, not just the one that moved most", () => {
    const [min, max] = extent([100, 120], [100, 80], 0, 0, 1);
    expect(min).toBeCloseTo(-20);
    expect(max).toBeCloseTo(20);
  });

  it("skips gaps instead of collapsing the scale onto NaN", () => {
    const [min, max] = extent([100, 110, 121], [100, NaN, 90], 0, 0, 2);
    expect(min).toBeCloseTo(-10);
    expect(max).toBeCloseTo(21);
  });

  it("degrades to a flat scale when there is nothing to draw", () => {
    expect(extent([], [], 0, 0, 0)).toEqual([0, 0]);
  });
});
