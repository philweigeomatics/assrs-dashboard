import { describe, expect, it } from "vitest";
import { MAX_PER_LEG, addLeg, addToLeg, emptyChain, moveLeg, problem, removeTicker, renameLeg, toRequest } from "./chain";

describe("chain", () => {
  it("a stock lives in one leg: adding it elsewhere moves it", () => {
    let c = emptyChain();
    const [a, b] = c.legs;
    c = addToLeg(c, a!.id, ["300308", "300502"], "光模块中游");
    c = addToLeg(c, b!.id, ["300502"], "HPC");
    expect(c.legs[0]!.tickers).toEqual(["300308"]);
    expect(c.legs[1]!.tickers).toEqual(["300502"]);
  });

  it("names an empty leg after the 板块 it was filled from, without clashing", () => {
    let c = emptyChain();
    c = addToLeg(c, c.legs[0]!.id, ["000001"], "银行");
    c = addToLeg(c, c.legs[1]!.id, ["600036"], "银行");
    expect(c.legs.map((l) => l.name)).toEqual(["银行", "银行2"]);
  });

  it("keeps a name the user typed", () => {
    let c = emptyChain();
    c = renameLeg(c, c.legs[0]!.id, "上游");
    c = addToLeg(c, c.legs[0]!.id, ["000001"], "银行");
    expect(c.legs[0]!.name).toBe("上游");
  });

  it("stops at the per-leg limit", () => {
    let c = emptyChain();
    const many = Array.from({ length: 20 }, (_, i) => String(600000 + i));
    c = addToLeg(c, c.legs[0]!.id, many, "x");
    expect(c.legs[0]!.tickers.length).toBe(MAX_PER_LEG);
  });

  it("reorders legs upstream and downstream, and not past the ends", () => {
    let c = addLeg(emptyChain());
    const ids = c.legs.map((l) => l.id);
    c = moveLeg(c, ids[2]!, -1);
    expect(c.legs.map((l) => l.id)).toEqual([ids[0], ids[2], ids[1]]);
    expect(moveLeg(c, ids[0]!, -1)).toBe(c);
  });

  it("says what is missing before it can run", () => {
    let c = emptyChain();
    expect(problem(c)).toMatch(/两个/);
    c = addToLeg(c, c.legs[0]!.id, ["300308"], "A");
    c = addToLeg(c, c.legs[1]!.id, ["601138"]);
    expect(problem(c)).toMatch(/名字/);
    c = renameLeg(c, c.legs[1]!.id, "A");
    expect(problem(c)).toMatch(/重复/);
    c = renameLeg(c, c.legs[1]!.id, "B");
    expect(problem(c)).toBeNull();
  });

  it("sends only filled legs, upstream first", () => {
    let c = addLeg(emptyChain());
    c = addToLeg(c, c.legs[0]!.id, ["300308"], "A");
    c = addToLeg(c, c.legs[2]!.id, ["601138"], "C");
    c = removeTicker(c, c.legs[0]!.id, "nope");
    expect(toRequest(c)).toEqual([{ name: "A", tickers: ["300308"] }, { name: "C", tickers: ["601138"] }]);
  });
});
