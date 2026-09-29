/**
 * Grid arithmetic, which is the part that fails without looking broken:
 * an off-by-one in the leading blanks shifts every disclosure by a day.
 */

import { describe, it, expect } from "vitest";
import { cellDate, monthGrid, shiftMonth } from "./monthGrid";

describe("monthGrid", () => {
  it("starts the week on Monday", () => {
    // 1 October 2026 is a Thursday: three blanks, then the 1st in slot four.
    const weeks = monthGrid(2026, 10);
    expect(weeks[0]).toEqual([0, 0, 0, 1, 2, 3, 4]);
  });

  it("gives a month starting on Sunday six leading blanks, not none", () => {
    // 1 February 2026 is a Sunday — the case a Sunday-first grid gets wrong.
    const weeks = monthGrid(2026, 2);
    expect(weeks[0]).toEqual([0, 0, 0, 0, 0, 0, 1]);
  });

  it("gives a month starting on Monday no leading blanks", () => {
    // 1 June 2026 is a Monday.
    expect(monthGrid(2026, 6)[0]?.[0]).toBe(1);
  });

  it("holds every day of the month exactly once", () => {
    for (const [y, m, days] of [[2026, 1, 31], [2026, 2, 28], [2026, 4, 30],
                                [2024, 2, 29]] as const) {
      const flat = monthGrid(y, m).flat().filter((d) => d > 0);
      expect(flat).toEqual(Array.from({ length: days }, (_, i) => i + 1));
    }
  });

  it("returns whole weeks, so the grid never has a ragged last row", () => {
    for (let m = 1; m <= 12; m += 1) {
      for (const week of monthGrid(2026, m)) expect(week).toHaveLength(7);
    }
  });

  it("pads the tail rather than spilling into the next month", () => {
    const last = monthGrid(2026, 10).at(-1)!;
    expect(last.filter((d) => d === 0).length).toBeGreaterThan(0);
    expect(Math.max(...last)).toBe(31);
  });
});

describe("shiftMonth", () => {
  it("steps forward and back", () => {
    expect(shiftMonth("2026-08", 1)).toBe("2026-09");
    expect(shiftMonth("2026-08", -1)).toBe("2026-07");
  });

  it("wraps the year in both directions", () => {
    expect(shiftMonth("2026-12", 1)).toBe("2027-01");
    expect(shiftMonth("2026-01", -1)).toBe("2025-12");
  });

  it("keeps the zero padding the API uses", () => {
    expect(shiftMonth("2026-10", -1)).toBe("2026-09");
  });
});

describe("cellDate", () => {
  it("matches the API's date strings", () => {
    expect(cellDate("2026-08", 5)).toBe("2026-08-05");
    expect(cellDate("2026-08", 26)).toBe("2026-08-26");
  });
});
