import { describe, expect, it } from "vitest";
import {
  MIN_WINDOW, lastSessions, lastYears, moveWindow, resizeEnd, resizeStart,
  resolveRange, toSaved,
} from "./windowRange";

const day = (i: number) => {
  const d = new Date(Date.UTC(2020, 0, 1 + i));
  return d.toISOString().slice(0, 10);
};
const dates = Array.from({ length: 1000 }, (_, i) => day(i));

describe("resolveRange", () => {
  it("starts on the most recent year when nothing is saved", () => {
    expect(resolveRange(dates, null)).toEqual(lastYears(dates, 1));
  });

  it("finds a saved window by its dates", () => {
    expect(resolveRange(dates, { from: day(100), to: day(400) })).toEqual([100, 400]);
  });

  it("follows the latest session when the window ends today", () => {
    const saved = toSaved(dates, [100, 999]);
    expect(saved.to).toBeNull();
    const grown = [...dates, day(1000), day(1001)];
    expect(resolveRange(grown, saved)).toEqual([100, 1001]);
  });

  it("treats a non-trading end date as the session before it", () => {
    const gappy = dates.filter((_, i) => i !== 400);
    expect(gappy[resolveRange(gappy, { from: day(100), to: day(400) })[1]]).toBe(day(399));
  });

  it("falls back when the window is entirely before this index existed", () => {
    const young = dates.slice(500);
    expect(resolveRange(young, { from: day(10), to: day(300) }))
      .toEqual(lastYears(young, 1));
  });

  it("clips a window that starts before this index existed", () => {
    const young = dates.slice(500);
    expect(resolveRange(young, { from: day(10), to: day(800) })).toEqual([0, 300]);
  });

  it("widens anything clipped to under the minimum", () => {
    const young = dates.slice(500);
    const [i, j] = resolveRange(young, { from: day(10), to: day(510) });
    expect(j - i).toBe(MIN_WINDOW);
  });
});

describe("moving the window", () => {
  it("keeps its length and stops at both ends", () => {
    expect(moveWindow([100, 300], 50, 1000)).toEqual([150, 350]);
    expect(moveWindow([100, 300], -500, 1000)).toEqual([0, 200]);
    expect(moveWindow([100, 300], 5000, 1000)).toEqual([799, 999]);
  });

  it("cannot be resized below the minimum from either side", () => {
    expect(resizeStart([100, 300], 290)).toEqual([300 - MIN_WINDOW, 300]);
    expect(resizeEnd([100, 300], 110, 1000)).toEqual([100, 100 + MIN_WINDOW]);
  });

  it("cannot be resized past the data", () => {
    expect(resizeStart([100, 300], -20)).toEqual([0, 300]);
    expect(resizeEnd([100, 300], 5000, 1000)).toEqual([100, 999]);
  });

  it("takes all the days when there are fewer than asked for", () => {
    expect(lastSessions(30, 242)).toEqual([0, 29]);
    expect(lastYears(dates, 10)).toEqual([0, 999]);
  });
});

describe("lastYears", () => {
  it("counts calendar years, whatever the number of sessions in them", () => {
    // 2022-09-26 back to 2021-09-26, on a calendar with a session every day.
    expect(lastYears(dates, 1)).toEqual([634, 999]);
    expect(dates[lastYears(dates, 1)[0]]).toBe("2021-09-26");
    const weekdays = dates.filter((d) => ![0, 6].includes(new Date(d).getUTCDay()));
    const [i, j] = lastYears(weekdays, 1);
    expect(weekdays[i]! >= "2021-09-26" && weekdays[i - 1]! < "2021-09-26").toBe(true);
    expect(j).toBe(weekdays.length - 1);
  });

  it("starts on the next session when the date a year back was a holiday", () => {
    const gappy = dates.filter((d) => d !== "2021-09-26");
    expect(gappy[lastYears(gappy, 1)[0]]).toBe("2021-09-27");
  });

  it("never gives less than the minimum window", () => {
    const short = dates.slice(0, 40);
    expect(lastYears(short, 1)).toEqual([0, 39]);
  });
});
