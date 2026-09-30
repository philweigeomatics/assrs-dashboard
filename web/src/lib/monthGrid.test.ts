/**
 * Grid arithmetic, and the totality that took the 日历 route down.
 *
 * The regression: `Number("undefined")` is NaN, `??` does not catch NaN, so a
 * missing month string reached `Array(NaN)` and threw "Invalid array length".
 * Half these tests exist to keep that from coming back.
 */

import { describe, it, expect } from "vitest";
import {
  cellDate, monthGrid, parseMonth, safeMonth, shiftMonth, thisMonth,
} from "./monthGrid";

describe("monthGrid layout", () => {
  it("starts the week on Monday", () => {
    // 1 October 2026 is a Thursday: three blanks, then the 1st in slot four.
    expect(monthGrid("2026-10")[0]).toEqual([0, 0, 0, 1, 2, 3, 4]);
  });

  it("gives a month starting on Sunday six leading blanks, not none", () => {
    // 1 February 2026 is a Sunday — the case a Sunday-first grid gets wrong.
    expect(monthGrid("2026-02")[0]).toEqual([0, 0, 0, 0, 0, 0, 1]);
  });

  it("gives a month starting on Monday no leading blanks", () => {
    expect(monthGrid("2026-06")[0]?.[0]).toBe(1);   // 1 June 2026 is a Monday
  });

  it("holds every day of the month exactly once", () => {
    for (const [ym, days] of [["2026-01", 31], ["2026-02", 28],
                              ["2026-04", 30], ["2024-02", 29]] as const) {
      const flat = monthGrid(ym).flat().filter((d) => d > 0);
      expect(flat).toEqual(Array.from({ length: days }, (_, i) => i + 1));
    }
  });

  it("returns whole weeks, so the grid is never ragged", () => {
    for (let m = 1; m <= 12; m += 1) {
      const ym = `2026-${String(m).padStart(2, "0")}`;
      for (const week of monthGrid(ym)) expect(week).toHaveLength(7);
    }
  });

  it("pads the tail rather than spilling into the next month", () => {
    const last = monthGrid("2026-10").at(-1)!;
    expect(last.filter((d) => d === 0).length).toBeGreaterThan(0);
    expect(Math.max(...last)).toBe(31);
  });
});

// ── the crash ────────────────────────────────────────────────────────────────
const BAD = [undefined, null, "", "undefined", "null", "NaN-NaN", "abc-de",
             "2026-13", "2026-00", "0000-01", 202610, {}, [], NaN];

describe("monthGrid never throws", () => {
  it.each(BAD)("survives %p and shows a real month", (bad) => {
    const weeks = monthGrid(bad as unknown);
    expect(weeks.length).toBeGreaterThan(0);
    for (const w of weeks) expect(w).toHaveLength(7);
    expect(weeks.flat().filter((d) => d > 0).length).toBeGreaterThanOrEqual(28);
  });

  it("would have thrown Invalid array length before the fix", () => {
    // The exact failing expression, kept as the record of what broke.
    const [y] = "undefined".split("-").map(Number);
    const year = y ?? new Date().getFullYear();     // ?? does NOT catch NaN
    expect(Number.isNaN(year)).toBe(true);
    expect(() => Array((new Date(year, 0, 1).getDay() + 6) % 7))
      .toThrow(/Invalid array length/);
  });
});

describe("shiftMonth never produces a poisoned value", () => {
  it("steps forward and back", () => {
    expect(shiftMonth("2026-08", 1)).toBe("2026-09");
    expect(shiftMonth("2026-08", -1)).toBe("2026-07");
  });

  it("wraps the year in both directions", () => {
    expect(shiftMonth("2026-12", 1)).toBe("2027-01");
    expect(shiftMonth("2026-01", -1)).toBe("2025-12");
  });

  it.each(BAD)("never returns NaN-NaN for %p", (bad) => {
    const got = shiftMonth(bad as unknown, 1);
    expect(got).toMatch(/^\d{4}-\d{2}$/);
    // The poisoning path: a bad value used to survive one render and come
    // back as "NaN-NaN", so the NEXT render was the one that threw.
    expect(monthGrid(got).length).toBeGreaterThan(0);
  });

  it("a bad step does not move the month", () => {
    expect(shiftMonth("2026-08", NaN)).toBe("2026-08");
  });
});

describe("parseMonth", () => {
  it("accepts what the API sends", () => {
    expect(parseMonth("2026-08")).toEqual({ year: 2026, month: 8 });
  });

  it("accepts a full date, taking its month", () => {
    expect(parseMonth("2026-08-26")).toEqual({ year: 2026, month: 8 });
  });

  it.each(BAD)("rejects %p", (bad) => {
    expect(parseMonth(bad as unknown)).toBeNull();
  });
});

describe("safeMonth", () => {
  it("passes a good value through, zero-padded", () => {
    expect(safeMonth("2026-8")).toBe("2026-08");
    expect(safeMonth("2026-08")).toBe("2026-08");
  });

  it("falls back to the current month", () => {
    expect(safeMonth(undefined, new Date(2026, 8, 29))).toBe("2026-09");
    expect(safeMonth("rubbish", new Date(2026, 0, 5))).toBe("2026-01");
  });
});

describe("thisMonth", () => {
  it("pads single-digit months", () => {
    expect(thisMonth(new Date(2026, 2, 9))).toBe("2026-03");
  });
});

describe("cellDate", () => {
  it("matches the API's date strings", () => {
    expect(cellDate("2026-08", 5)).toBe("2026-08-05");
    expect(cellDate("2026-08", 26)).toBe("2026-08-26");
  });

  it("still produces a usable key from a bad month", () => {
    expect(cellDate(undefined, 5)).toMatch(/^\d{4}-\d{2}-05$/);
  });
});
