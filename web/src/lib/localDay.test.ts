/**
 * Local calendar days.
 *
 * The regression this guards: `toISOString().slice(0, 10)` converts to UTC
 * first, so local midnight east of Greenwich lands on the previous day and a
 * week built from it is off by one all the way across.
 *
 * Node runs these in whatever zone the machine is in, so the assertions are
 * written to hold in any of them rather than pinning one.
 */

import { describe, it, expect } from "vitest";
import {
  localClock, localDay, localMonday, offBeijing, viewerZone, zoneLabel,
} from "./localDay";

describe("localDay", () => {
  it("reads the local calendar day, not the UTC one", () => {
    const d = new Date(2026, 9, 5, 0, 0, 0);     // local midnight, 5 Oct
    expect(localDay(d)).toBe("2026-10-05");
  });

  it("is stable right up to local midnight", () => {
    expect(localDay(new Date(2026, 9, 5, 23, 59, 59))).toBe("2026-10-05");
    expect(localDay(new Date(2026, 9, 6, 0, 0, 0))).toBe("2026-10-06");
  });

  it("pads single digits so keys sort and compare as strings", () => {
    expect(localDay(new Date(2026, 0, 3))).toBe("2026-01-03");
  });

  it("would disagree with the UTC shortcut east of Greenwich", () => {
    // Not an assertion about this machine's zone — just that the two differ
    // whenever the local offset pushes midnight across the UTC date line.
    const d = new Date(2026, 9, 5, 0, 0, 0);
    const utc = d.toISOString().slice(0, 10);
    if (d.getTimezoneOffset() < 0) expect(utc).not.toBe(localDay(d));
    else expect(utc).toBe(localDay(d));
  });
});

describe("localMonday", () => {
  it("walks back to Monday", () => {
    // 2026-10-04 is a Sunday; its week starts Monday the 28th.
    expect(localDay(localMonday(new Date(2026, 9, 4)))).toBe("2026-09-28");
  });

  it("leaves a Monday where it is", () => {
    expect(localDay(localMonday(new Date(2026, 9, 5)))).toBe("2026-10-05");
  });

  it("crosses a month boundary", () => {
    // Thursday 1 October 2026 belongs to the week of Monday 28 September.
    expect(localDay(localMonday(new Date(2026, 9, 1)))).toBe("2026-09-28");
  });

  it("returns local midnight, so the day never slips", () => {
    const m = localMonday(new Date(2026, 9, 4, 23, 30));
    expect([m.getHours(), m.getMinutes(), m.getSeconds()]).toEqual([0, 0, 0]);
  });
});

describe("localClock", () => {
  it("renders an offset instant on the viewer's clock", () => {
    const at = "2026-10-08T02:00:00+08:00";
    const want = new Date(at);
    expect(localClock(at)).toBe(
      `${String(want.getHours()).padStart(2, "0")}`
      + `:${String(want.getMinutes()).padStart(2, "0")}`);
  });

  it("is empty for an unparseable instant rather than NaN:NaN", () => {
    expect(localClock("nonsense")).toBe("");
  });
});

describe("offBeijing", () => {
  it("is false when the viewer's reading matches Tushare's label", () => {
    const at = "2026-10-08T02:00:00+08:00";
    const d = new Date(at);
    expect(offBeijing(at, localDay(d), localClock(at))).toBe(false);
  });

  it("is true when the clock differs", () => {
    // Tushare says 02:00 Beijing; anywhere else reads a different time.
    const at = "2026-10-08T02:00:00+08:00";
    const differs = offBeijing(at, "2026-10-08", "02:00");
    expect(differs).toBe(localClock(at) !== "02:00"
                         || localDay(new Date(at)) !== "2026-10-08");
  });

  it("does not throw on a bad instant", () => {
    expect(offBeijing("nope", "2026-10-08", "02:00")).toBe(false);
  });
});

describe("zone helpers", () => {
  it("name a zone and an offset without throwing", () => {
    expect(typeof viewerZone()).toBe("string");
    expect(typeof zoneLabel()).toBe("string");
  });
});
