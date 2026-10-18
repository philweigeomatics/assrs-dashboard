/**
 * The seasonality panel's calendar must agree with seasonality.py's: slot 59
 * is Feb 29, weeks are 7-day blocks from Jan 1, and this year's line must not
 * claim days that have not happened.
 */
import { describe, expect, it } from "vitest";
import { cellFill, slotLabel, valueAt, weekOf, weeklyPath, yearColour } from "./season";

describe("calendar", () => {
  it("labels slots on a leap calendar", () => {
    expect(slotLabel(0)).toBe("01-01");
    expect(slotLabel(59)).toBe("02-29");
    expect(slotLabel(60)).toBe("03-01");
    expect(slotLabel(365)).toBe("12-31");
  });
  it("puts Dec 31 in the last week block", () => {
    expect(weekOf(6)).toBe(0);
    expect(weekOf(7)).toBe(1);
    expect(weekOf(365)).toBe(51);
  });
});

describe("weeklyPath", () => {
  it("compounds from the previous close", () => {
    const p = weeklyPath([10, 10, ...Array(50).fill(0)]);
    expect(p[0]).toEqual([0, 0]);
    expect(p[2]).toEqual([13, 21]);
    expect(p[p.length - 1]![0]).toBe(365);
  });
  it("stops where this year has not got to yet", () => {
    const p = weeklyPath([1, 2, null, null]);
    expect(p.length).toBe(3);
  });
});

describe("valueAt", () => {
  const path: [number, number][] = [[2, 1], [5, 3], [9, -2]];
  it("holds the last value through closed days", () => {
    expect(valueAt(path, 6)).toBe(3);
    expect(valueAt(path, 9)).toBe(-2);
  });
  it("is empty before the first session and well after the last", () => {
    expect(valueAt(path, 1)).toBeNull();
    expect(valueAt(path, 40)).toBeNull();
  });
});

describe("colour", () => {
  it("follows the market: a rise is red on an A-share page, green in North America", () => {
    expect(cellFill(5, 10, true)).toContain("--color-up");
    expect(cellFill(5, 10, false)).toContain("--color-down");
    expect(cellFill(-5, 10, false)).toContain("--color-up");
  });
  it("leaves no move neutral", () => {
    expect(cellFill(0, 10, true)).toBe("var(--color-panel-alt)");
    expect(cellFill(null, 10, true)).toBe("var(--color-panel-alt)");
  });
  it("gives recent years hues, older years grey, this year ink", () => {
    const full = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
    expect(yearColour(2026, full, 2026)).toBe("var(--color-ink)");
    expect(yearColour(2025, full, 2026)).toBe("#0062cc");
    expect(yearColour(2023, full, 2026)).toBe("#7c3aed");
    expect(yearColour(2016, full, 2026)).toBe("#b4b4bb");
  });
});
