import { describe, expect, it } from "vitest";
import type { IndexMember } from "./types";
import { indexPicks } from "./indexPicks";

const m = (ticker: string, weight: number | null,
           status: IndexMember["status"] = "active",
           removed_on: string | null = null): IndexMember => ({
  id: 0, ticker, name: ticker.toLowerCase(), status, weight, target: null,
  price: null, ret_1d: null,
  contrib: { "1d": null, "5d": null, "1m": null, "3m": null, all: null },
  added_on: "2025-01-02", removed_on, exit_price: null,
});

describe("indexPicks", () => {
  it("keeps current members, heaviest first, as ticker + name", () => {
    const got = indexPicks([m("A", 0.2), m("B", 0.5), m("C", null, "halted")], 30);
    expect(got.picks).toEqual([{ t: "B", n: "b" }, { t: "A", n: "a" }, { t: "C", n: "c" }]);
    expect(got.dropped).toEqual([]);
  });

  it("leaves out removed and delisted members and duplicate rows", () => {
    const got = indexPicks([
      m("A", 0.3), m("X", 0, "removed", "2025-06-01"), m("Y", 0, "delisted"),
      m("A", 0.1, "active", "2025-03-01"),
    ], 30);
    expect(got.picks.map((p) => p.t)).toEqual(["A"]);
  });

  it("cuts at the cap from the lightest end", () => {
    const got = indexPicks([m("A", 0.1), m("B", 0.6), m("C", 0.3)], 2);
    expect(got.picks.map((p) => p.t)).toEqual(["B", "C"]);
    expect(got.dropped.map((p) => p.t)).toEqual(["A"]);
  });
});
