import { describe, expect, it } from "vitest";
import { dailyChanges, MIN_DAYS, quantile, rankDay, rankOf, sizeWord, windowStats } from "./moveRank";

describe("dailyChanges", () => {
  it("is close-to-close in percent, null where a close is missing", () => {
    const c = dailyChanges([10, 11, null, 11, 9.9]);
    expect(c[0]).toBeNull();
    expect(c[1]).toBeCloseTo(10);
    expect(c[2]).toBeNull();
    expect(c[3]).toBeNull();            // no previous close
    expect(c[4]).toBeCloseTo(-10);
  });
});

describe("rankOf", () => {
  it("counts ties half, so the median ranks 50", () => {
    expect(rankOf([1, 2, 3, 4, 5], 3)).toBe(50);
    expect(rankOf([1, 2, 3, 4, 5], 5)).toBe(90);
    expect(rankOf([1, 2, 3, 4, 5], 1)).toBe(10);
    expect(rankOf([1, 2, 3, 4, 5], 9)).toBe(100);
    expect(rankOf([1, 2, 3, 4, 5], 0)).toBe(0);
    expect(rankOf([2, 2, 2, 2], 2)).toBe(50);
  });
});

describe("quantile", () => {
  it("interpolates", () => {
    expect(quantile([0, 10], 0.5)).toBe(5);
    expect(quantile([0, 1, 2, 3, 4], 0.95)).toBeCloseTo(3.8);
  });
});

describe("windowStats", () => {
  // 1..100 as daily changes, bar 0 has none.
  const changes = [null, ...Array.from({ length: 100 }, (_, i) => i + 1 - 50)];

  it("uses only the bars in the window, clamped to the data", () => {
    const s = windowStats(changes, -5.3, 400)!;
    expect(s.from).toBe(0);
    expect(s.to).toBe(100);
    expect(s.n).toBe(100);
    const t = windowStats(changes, 50.2, 80.9)!;
    expect(t.from).toBe(51);
    expect(t.to).toBe(80);
    expect(t.n).toBe(30);
    expect(t.sorted[0]).toBe(1);        // bar 51 = 51 − 50
  });

  it("refuses a window too short to rank against", () => {
    expect(windowStats(changes, 1, MIN_DAYS - 1)).toBeNull();
    expect(windowStats(changes, 1, MIN_DAYS)).not.toBeNull();
  });

  it("the same move ranks differently in a calm and a wild window", () => {
    const calm = Array.from({ length: 60 }, (_, i) => (i % 2 ? 1 : -1) * 0.8);
    const wild = Array.from({ length: 60 }, (_, i) => (i % 2 ? 1 : -1) * 7);
    const day = 5;
    const a = rankDay(windowStats([...calm, day], 0, 60), day)!;
    const b = rankDay(windowStats([...wild, day], 0, 60), day)!;
    expect(a.sizeRank).toBeGreaterThan(98);
    expect(sizeWord(a.sizeRank)).toBe("极端");
    expect(sizeWord(96)).toBe("很大");
    expect(sizeWord(50)).toBe("平常");
    expect(b.sizeRank).toBeLessThan(55);
    expect(a.multiple).toBeCloseTo(5 / 0.8);
  });

  it("ranks a drop low by direction but high by size", () => {
    const vals = Array.from({ length: 50 }, (_, i) => ((i % 5) - 2) * 0.5);   // −1 … +1
    const r = rankDay(windowStats([...vals, -4], 0, 50), -4)!;
    expect(r.rank).toBeLessThan(2);
    expect(r.sizeRank).toBeGreaterThan(98);
  });
});
