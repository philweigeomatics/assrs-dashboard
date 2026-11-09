import { describe, expect, it } from "vitest";
import { equalWeights, isBalanced, sameWeights, sumOf, weightsOf } from "./weightDraft";

describe("weight drafts", () => {
  it("reads the weights out of the holdings", () => {
    expect(weightsOf([{ t: "A", weight_pct: 60 }, { t: "B", weight_pct: 40 }]))
      .toEqual({ A: 60, B: 40 });
  });

  it("treats a missing name as held at zero", () => {
    expect(sameWeights({ A: 100, B: 0 }, { A: 100 })).toBe(true);
    expect(sameWeights({ A: 90, B: 10 }, { A: 100 })).toBe(false);
  });

  it("notices a small change but not floating-point noise", () => {
    expect(sameWeights({ A: 33.3 }, { A: 33.8 })).toBe(false);
    expect(sameWeights({ A: 0.1 + 0.2 }, { A: 0.3 })).toBe(true);
  });

  it("makes equal weights that can always be saved", () => {
    for (let n = 2; n <= 30; n += 1) {
      const w = equalWeights(Array.from({ length: n }, (_, i) => `S${i}`));
      expect(Object.keys(w)).toHaveLength(n);
      expect(sumOf(w)).toBeCloseTo(100, 6);
      expect(isBalanced(w)).toBe(true);
    }
  });

  it("refuses a total that is off by more than the tolerance", () => {
    expect(isBalanced({ A: 50, B: 49.95 })).toBe(true);
    expect(isBalanced({ A: 50, B: 49 })).toBe(false);
  });
});
