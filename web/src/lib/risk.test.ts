import { describe, expect, it } from "vitest";
import { citations, sparkPoints } from "./risk";
import type { RiskEvidence } from "./types";

const ev = (eid: string) => ({ eid }) as RiskEvidence;

describe("citations", () => {
  it("numbers a citation by the item's current rank", () => {
    expect(citations("毛利承压[#aaaaaa1]，订单增长[#bbbbbb2]", [ev("bbbbbb2"), ev("aaaaaa1")]))
      .toEqual([{ text: "毛利承压" }, { cite: 2 }, { text: "，订单增长" }, { cite: 1 }]);
  });

  it("drops a citation whose item is no longer listed", () => {
    expect(citations("旧消息[#cccccc3]。", [ev("aaaaaa1")])).toEqual([{ text: "旧消息" }, { text: "。" }]);
  });

  it("returns nothing for no text", () => {
    expect(citations(null, [])).toEqual([]);
  });
});

describe("sparkPoints", () => {
  it("spans the width and clamps to the scale", () => {
    const pts = sparkPoints([0, 150], 100, 40).split(" ");
    expect(pts[0]).toBe("0.0,38.0");
    expect(pts[1]).toBe("100.0,2.0");
  });
});
