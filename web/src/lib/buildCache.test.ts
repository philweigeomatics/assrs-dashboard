/**
 * The bug these pin: an optimisation vanished when you left the tab.
 *
 * Both halves of it were the same cause — the result lived in the mutation —
 * so both are checked here against a real QueryClient rather than a mock.
 */

import { describe, it, expect } from "vitest";
import { QueryClient, QueryObserver, MutationObserver } from "@tanstack/react-query";
import { BUILD_KEY, signature, type CachedBuild } from "./buildCache";

const build = (tag: string) => ({ tag } as unknown as CachedBuild["d"]);

/** What Optimiser mounts: a cache-only subscription to the last build. */
function watch(qc: QueryClient) {
  return new QueryObserver<CachedBuild | null>(qc, {
    queryKey: BUILD_KEY as unknown as string[],
    queryFn: () => null,
    enabled: false,
    staleTime: Infinity,
    gcTime: Infinity,
  });
}

describe("signature", () => {
  const base = { symbols: ["600519", "000001"], maxWeight: 30,
                 lookback: 242, duration: 1, rf: 3, mode: "max_sharpe" };

  it("does not change when the same names are picked in another order", () => {
    expect(signature({ ...base, symbols: ["000001", "600519"] }))
      .toBe(signature(base));
  });

  it("changes when any input that feeds the optimiser changes", () => {
    for (const diff of [
      { symbols: ["600519"] }, { maxWeight: 25 },
      { lookback: 120 }, { duration: 5 }, { rf: 2 },
      // Changing the objective changes the answer.
      { mode: "risk_parity" }, { mode: "min_variance" },
    ]) {
      expect(signature({ ...base, ...diff })).not.toBe(signature(base));
    }
  });
});

describe("the kept build", () => {
  it("survives the component unmounting — the reported bug", () => {
    const qc = new QueryClient();
    const first = watch(qc);
    const stop = first.subscribe(() => {});
    qc.setQueryData<CachedBuild>(BUILD_KEY, { d: build("run-1"), sig: "a" });
    expect(first.getCurrentResult().data?.d).toEqual(build("run-1"));

    stop();                                   // switch to 我的组合
    const second = watch(qc);                 // …and switch back
    second.subscribe(() => {});
    expect(second.getCurrentResult().data?.d).toEqual(build("run-1"));
  });

  it("stays on screen while the next solve is in flight", async () => {
    const qc = new QueryClient();
    const view = watch(qc);
    view.subscribe(() => {});
    qc.setQueryData<CachedBuild>(BUILD_KEY, { d: build("run-1"), sig: "a" });

    const run = new MutationObserver(qc, {
      mutationFn: async () => { await new Promise((r) => setTimeout(r, 20)); },
      onSuccess: () =>
        qc.setQueryData<CachedBuild>(BUILD_KEY, { d: build("run-2"), sig: "a" }),
    });
    run.subscribe(() => {});
    const inFlight = run.mutate();

    // The mutation's own data is gone the instant it goes pending — that is
    // what used to blank the panel on a frontier click.
    expect(run.getCurrentResult().data).toBeUndefined();
    expect(view.getCurrentResult().data?.d).toEqual(build("run-1"));

    await inFlight;
    expect(view.getCurrentResult().data?.d).toEqual(build("run-2"));
  });

  it("is never fetched — the query function must not run", async () => {
    const qc = new QueryClient();
    let calls = 0;
    const obs = new QueryObserver<CachedBuild | null>(qc, {
      queryKey: BUILD_KEY as unknown as string[],
      queryFn: () => { calls += 1; return null; },
      enabled: false, staleTime: Infinity, gcTime: Infinity,
    });
    obs.subscribe(() => {});
    await new Promise((r) => setTimeout(r, 30));
    expect(calls).toBe(0);
  });
});
