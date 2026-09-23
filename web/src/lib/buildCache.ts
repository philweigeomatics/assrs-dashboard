/**
 * Where an optimisation result lives between renders.
 *
 * Not in the mutation that produced it. A mutation's `data` dies with the
 * component, and TanStack v5 also clears it the moment the next `mutate`
 * goes pending — so switching to 我的组合 and back threw away an analysis
 * that took real time to compute, and clicking a frontier point blanked the
 * whole panel while the re-solve was in flight. Parked in the query cache it
 * survives both: the previous answer stays up until a new one replaces it.
 */

import type { PortfolioBuild } from "./types";

export const BUILD_KEY = ["pf", "build"] as const;

export type CachedBuild = { d: PortfolioBuild; sig: string };

export type BuildInputs = {
  symbols: string[];
  maxWeight: number;
  lookback: number;
  duration: number;
  rf: number;
};

/**
 * What produced a result, so a kept one can admit the form has moved on.
 *
 * Order-insensitive on symbols — the same names picked in a different order
 * is the same portfolio, and re-sorting the chips should not make a result
 * look stale. The target return is deliberately absent: picking a point on
 * the frontier is a different question about the same inputs, not a change
 * to them.
 */
export function signature(i: BuildInputs): string {
  return `${[...i.symbols].sort().join(",")}|${i.maxWeight}|${i.lookback}`
    + `|${i.duration}|${i.rf}`;
}
