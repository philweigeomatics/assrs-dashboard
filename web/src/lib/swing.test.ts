import { describe, expect, it } from "vitest";
import { scenario } from "./swing";
import type { Swing } from "./types";

const d = {
  symbols: ["A", "B"],
  stats: {
    A: { vol_annual_pct: 60, avg_abs_daily_pct: 3, beta_up: 2.5, beta_down: 1.5, resid_vol_daily_pct: 2 },
    B: { vol_annual_pct: 30, avg_abs_daily_pct: 1, beta_up: 0.8, beta_down: 1.2, resid_vol_daily_pct: 1 },
  },
} as unknown as Swing;

describe("scenario", () => {
  it("uses the up-day multiple for a rise and the down-day one for a fall", () => {
    const up = scenario(d, 10, 20);
    const down = scenario(d, -10, 20);
    expect(up.find((r) => r.s === "A")!.mid).toBeCloseTo(25);
    expect(down.find((r) => r.s === "A")!.mid).toBeCloseTo(-15);
    expect(up.find((r) => r.s === "B")!.mid).toBeCloseTo(8);
    expect(down.find((r) => r.s === "B")!.mid).toBeCloseTo(-12);
  });
  it("widens the band with the horizon, by the square root of time", () => {
    const short = scenario(d, 10, 20).find((r) => r.s === "A")!.band;
    const long = scenario(d, 10, 80).find((r) => r.s === "A")!.band;
    expect(long / short).toBeCloseTo(2);
  });
  it("puts the market first, moving exactly as asked", () => {
    expect(scenario(d, -7, 20)[0]).toEqual({ s: "__market__", mid: -7, band: 0 });
  });
});
