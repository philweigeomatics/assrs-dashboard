import { describe, expect, it } from "vitest";
import { corr, corrFill, opposite, rollingCorr } from "./seesaw";

describe("corr", () => {
  it("is +1 for the same series and -1 for its mirror", () => {
    const a = [1, -2, 3, 0.5, -1];
    expect(corr(a, a)).toBeCloseTo(1);
    expect(corr(a, a.map((v) => -v))).toBeCloseTo(-1);
  });
  it("skips gaps rather than treating them as zero", () => {
    expect(corr([1, 2, null, 3, 4], [2, 4, 100, 6, 8])).toBeCloseTo(1);
  });
  it("has no answer for a flat series or too few points", () => {
    expect(corr([1, 1, 1, 1], [1, 2, 3, 4])).toBeNull();
    expect(corr([1, 2], [2, 1])).toBeNull();
  });
});

describe("rollingCorr", () => {
  it("waits for a full window, then follows a regime change", () => {
    const a = [1, -1, 2, -2, 1, -1, 2, -2];
    const b = [1, -1, 2, -2, -1, 1, -2, 2];       // together, then mirrored
    const r = rollingCorr(a, b, 4);
    expect(r.slice(0, 3)).toEqual([null, null, null]);
    expect(r[3]).toBeCloseTo(1);
    expect(r[7]).toBeCloseTo(-1);
  });
});

describe("colour and opposite", () => {
  it("uses blue for together and orange for opposite, never the up/down colours", () => {
    expect(corrFill(0.5)).toContain("#0062cc");
    expect(corrFill(-0.5)).toContain("#c2410c");
    expect(corrFill(0.5)).not.toContain("--color-up");
  });
  it("counts only a rise against a fall", () => {
    expect(opposite(1, -1)).toBe(true);
    expect(opposite(1, 2)).toBe(false);
    expect(opposite(0, -1)).toBe(false);
    expect(opposite(null, 1)).toBe(false);
  });
});
