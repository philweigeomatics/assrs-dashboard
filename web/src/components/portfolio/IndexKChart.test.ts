import { describe, expect, it } from "vitest";
import { movingAverage } from "./IndexKChart";

describe("movingAverage", () => {
  it("is null until there are n values, then the trailing mean", () => {
    expect(movingAverage([1, 2, 3, 4, 5], 3)).toEqual([null, null, 2, 3, 4]);
  });
  it("restarts after a gap rather than averaging across it", () => {
    expect(movingAverage([1, 2, null, 4, 6, 8], 2)).toEqual([null, 1.5, null, null, 5, 7]);
  });
});
