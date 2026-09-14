import { describe, expect, it } from "vitest";
import { suggest } from "./search";

const stocks = [
  { t: "600519", n: "贵州茅台" },
  { t: "600036", n: "招商银行" },
  { t: "000001", n: "平安银行" },
  { t: "601318", n: "中国平安" },
  { t: "300750", n: "宁德时代" },
];
const history = [
  { t: "601318", n: "中国平安" },
  { t: "600519", n: "贵州茅台" },
];

describe("suggest", () => {
  it("shows recent searches, newest first, for an empty query", () => {
    expect(suggest("", stocks, history).map((s) => s.t)).toEqual(["601318", "600519"]);
    expect(suggest("  ", stocks, history).every((s) => s.fromHistory)).toBe(true);
  });

  it("puts an exact code above code prefixes", () => {
    const r = suggest("600519", stocks, []);
    expect(r[0]?.t).toBe("600519");
  });

  it("matches by code prefix", () => {
    expect(suggest("6005", stocks, []).map((s) => s.t)).toEqual(["600519"]);
  });

  it("matches by name, prefix before contains", () => {
    // 平安银行 starts with 平安; 中国平安 only contains it.
    expect(suggest("平安", stocks, []).map((s) => s.t)).toEqual(["000001", "601318"]);
  });

  it("ranks previously searched stocks first within the same rank", () => {
    // Both are "contains 银行" (rank 3)… no history → by code.
    expect(suggest("银行", stocks, []).map((s) => s.t)).toEqual(["000001", "600036"]);
    // …with 600036 in history it jumps ahead.
    const r = suggest("银行", stocks, [{ t: "600036", n: "招商银行" }]);
    expect(r.map((s) => s.t)).toEqual(["600036", "000001"]);
    expect(r[0]?.fromHistory).toBe(true);
  });

  it("returns nothing for a query that matches nothing", () => {
    expect(suggest("zzz", stocks, history)).toEqual([]);
  });

  it("respects the limit", () => {
    // "6" prefixes three codes; the limit keeps two.
    expect(suggest("6", stocks, [], 2)).toHaveLength(2);
  });
});
