/**
 * The explanations, and the two ways they can quietly become wrong.
 *
 * A tooltip is looked up by the column's label. A typo there does not throw
 * and does not show up in review — it renders an empty tooltip, which looks
 * exactly like a column that was never meant to have one. And a threshold
 * written into the prose instead of taken from the engine goes stale the
 * moment someone tunes a gate, leaving the page confidently explaining a rule
 * that is no longer the rule.
 *
 *     npm test -- indicators
 */

import { describe, expect, it } from "vitest";
import { DISCOVER_INDICATORS, pairIndicators } from "./indicators";
import type { PairGates } from "./types";
// The components' own source, so a tooltip lookup that no longer resolves is
// caught here rather than rendering as a blank tooltip nobody notices.
import pairSrc from "../components/PairTrade.tsx?raw";
import discoverSrc from "../components/Discover.tsx?raw";

const GATES: PairGates = {
  coint_p: 0.1, adf_p: 0.1,
  hurst_max: 0.45, hl_min: 5, hl_max: 30,
  entry_z: 2, watch_z: 1.5,
  good_score: 7, max_score: 11,
  z_window: 60, ols_window: 252,
};

/** Every `tip("…")` a component asks for, read from its own source. */
const asked = (src: string) =>
  [...src.matchAll(/\btip\(\s*"([^"]+)"\s*\)/g)].map((m) => m[1]!);

describe("every column that asks for an explanation gets one", () => {
  it("covers the pair table", () => {
    const have = new Set(pairIndicators(GATES).map((i) => i.label));
    const want = asked(pairSrc);
    expect(want.length).toBeGreaterThan(5);
    expect(want.filter((l) => !have.has(l))).toEqual([]);
  });

  it("covers the discovery table", () => {
    const have = new Set(DISCOVER_INDICATORS.map((i) => i.label));
    const want = asked(discoverSrc);
    expect(want.length).toBeGreaterThan(3);
    expect(want.filter((l) => !have.has(l))).toEqual([]);
  });

  it("notices a label that does not resolve", () => {
    // The failure mode being guarded: a lookup that misses returns "" rather
    // than throwing, so the column silently loses its tooltip.
    const have = new Set(pairIndicators(GATES).map((i) => i.label));
    expect(have.has("协整 p")).toBe(true);
    expect(have.has("协整p")).toBe(false);      // no space — a realistic typo
  });
});

describe("nothing is explained with an empty string", () => {
  it.each([...pairIndicators(GATES), ...DISCOVER_INDICATORS])(
    "$label", (i) => {
      for (const field of [i.short, i.what, i.reads]) {
        expect(field.trim().length).toBeGreaterThan(8);
      }
      expect(i.caveat ?? "x").not.toBe("");
    });
});

describe("thresholds come from the engine, not from the prose", () => {
  const text = (g: PairGates) =>
    pairIndicators(g).map((i) => [i.short, i.what, i.pass, i.reads, i.caveat]
      .filter(Boolean).join(" ")).join("\n");

  it("moves when the gate moves", () => {
    // Every gate moved together, so any surviving default is a threshold
    // that was typed into the prose instead of read from the engine.
    const moved = text({ ...GATES, coint_p: 0.03, adf_p: 0.07,
                         hurst_max: 0.6, hl_min: 3, hl_max: 45 });
    expect(moved).toContain("0.03");
    expect(moved).toContain("0.07");
    expect(moved).toContain("0.60");
    expect(moved).toContain("3–45 天");
    expect(moved).not.toContain("0.10");
    expect(moved).not.toContain("0.45");
    expect(moved).not.toContain("5–30");
  });

  it("carries the windows and the entry level too", () => {
    const moved = text({ ...GATES, z_window: 90, ols_window: 504,
                         entry_z: 2.5, watch_z: 1.8 });
    expect(moved).toContain("90 天");
    expect(moved).toContain("504 天");
    expect(moved).toContain("2.5");
    expect(moved).toContain("1.8");
    // Absence matters as much as presence: one field still saying 60 while
    // another says 90 is the exact shape of a threshold typed into the prose,
    // and asserting only on the new value cannot see it.
    expect(moved).not.toContain("60 天");
    expect(moved).not.toContain("252 天");
    expect(moved).not.toContain("2.0");
    expect(moved).not.toContain("1.5");
  });

  it("states a threshold for each of the four gates", () => {
    const by = Object.fromEntries(pairIndicators(GATES).map((i) => [i.label, i]));
    for (const label of ["协整 p", "ADF p", "Hurst", "半衰期"]) {
      expect(by[label]!.pass, label).toBeTruthy();
    }
  });
});

describe("the entries say what the number does NOT tell you", () => {
  it("every gate carries a caveat", () => {
    // The line a reader actually needs. A column that only ever says
    // "< 0.10 通过" teaches that passing means the trade will work.
    const by = Object.fromEntries(pairIndicators(GATES).map((i) => [i.label, i]));
    for (const label of ["协整 p", "ADF p", "Hurst", "半衰期", "Z", "历史"]) {
      expect(by[label]!.caveat, label).toBeTruthy();
    }
  });

  it("says the z baseline is a rolling one", () => {
    const z = pairIndicators(GATES).find((i) => i.label === "Z")!;
    expect(z.caveat).toContain("滚动");
  });

  it("says the bought leg is unhedged", () => {
    const h = pairIndicators(GATES).find((i) => i.label === "历史")!;
    expect(h.caveat).toContain("做空");
  });
});
