import { describe, expect, it } from "vitest";
import {
  curveStats, heldOnly, matching, nextName, outside, type Candidate,
} from "./candidates";

const cand = (id: string, name: string, weights: Record<string, number>): Candidate => ({
  id, name, weights, names: {}, source: "最大夏普", from: "2025-01-02", to: "2025-12-31",
  saved: "2026-11-11",
});

describe("candidates", () => {
  it("names the first free slot", () => {
    expect(nextName([])).toBe("候选 1");
    expect(nextName([cand("a", "候选 1", {}), cand("b", "候选 3", {})])).toBe("候选 2");
    expect(nextName([cand("a", "最大夏普", {})])).toBe("候选 1");
  });

  it("keeps only what is held", () => {
    expect(heldOnly({ A: 50, B: 0, C: 49.996 })).toEqual({ A: 50, C: 50 });
  });

  it("finds which saved candidate a draft is, zeros and all", () => {
    const list = [cand("a", "x", { A: 60, B: 40 }), cand("b", "y", { A: 100 })];
    expect(matching(list, { A: 100, B: 0 })?.id).toBe("b");
    expect(matching(list, { A: 60, B: 40 })?.id).toBe("a");
    expect(matching(list, { A: 59, B: 41 })).toBeNull();
  });

  it("lists names the current stock list does not have", () => {
    expect(outside(cand("a", "x", { A: 50, Z: 50 }), ["A", "B"])).toEqual(["Z"]);
  });
});

describe("curveStats", () => {
  it("agrees with compounding on a steady line", () => {
    // +0.1% every day for 242 days.
    const values = Array.from({ length: 243 }, (_, i) => (1.001 ** i - 1) * 100);
    const s = curveStats(values);
    expect(s.total).toBeCloseTo((1.001 ** 242 - 1) * 100, 6);
    expect(s.ann).toBeCloseTo((1.001 ** 242 - 1) * 100, 6);
    expect(s.vol).toBeCloseTo(0, 6);
    expect(s.maxdd).toBeCloseTo(0, 6);
  });

  it("measures the drawdown from the running peak", () => {
    const s = curveStats([0, 10, 21, -1, 5]);
    // Peak 1.21 → trough 0.99.
    expect(s.maxdd).toBeCloseTo((0.99 / 1.21 - 1) * 100, 6);
    expect(s.total).toBeCloseTo(5, 6);
  });

  it("subtracts the risk-free rate before dividing by volatility", () => {
    const values = [0, 1, -0.5, 2, 1.5, 3];
    const a = curveStats(values, 0);
    const b = curveStats(values, 3);
    expect(b.sharpe!).toBeCloseTo(a.sharpe! - 0.03 / (a.vol! / 100), 6);
  });

  it("skips gaps and gives nothing for too short a line", () => {
    expect(curveStats([0, NaN, 1]).total).toBeNull();
    expect(curveStats([0, NaN, 1, 2]).total).toBeCloseTo(2, 6);
  });
});
