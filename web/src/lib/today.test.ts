import { describe, expect, it } from "vitest";
import type { EarningsCalendar, EarningsRow, EcoEvent, EcoWeek } from "./types";
import { localDay } from "./localDay";
import { countriesOf, ecoDayOf, released, shouldPop, todayEarnings, todayEco }
  from "./today";

const ev = (over: Partial<EcoEvent>): EcoEvent => ({
  at: null, date: "2026-11-17", time: "", country: "美国", currency: "USD",
  event: "x", value: null, prev: null, forecast: null, ...over,
});

const week = (events: EcoEvent[]): EcoWeek => ({
  from: "2026-11-16", to: "2026-11-22", fetched_from: "2026-11-15",
  fetched_to: "2026-11-23", source_tz: "Asia/Shanghai", events,
  total: events.length,
});

describe("todayEco", () => {
  it("places a timed release on the viewer's day, not Tushare's", () => {
    // Whatever zone the test runs in, the instant decides the day.
    const at = "2026-11-17T21:30:00+08:00";
    const e = ev({ at, date: "2026-11-17", time: "21:30" });
    expect(ecoDayOf(e)).toBe(localDay(new Date(at)));
    expect(todayEco(week([e]), localDay(new Date(at)))).toEqual([e]);
  });

  it("keeps the Beijing date for an untimed release and sorts it last", () => {
    const timed = ev({ at: "2026-11-17T12:00:00+08:00", event: "timed" });
    const day = localDay(new Date(timed.at!));
    const untimed = ev({ date: day, event: "untimed" });
    const early = ev({ at: "2026-11-17T09:00:00+08:00", event: "early" });
    const got = todayEco(week([untimed, timed, early]), day)
      .filter((e) => ecoDayOf(e) === day).map((e) => e.event);
    expect(got[got.length - 1]).toBe("untimed");
    expect(got.indexOf("timed")).toBeGreaterThan(-1);
  });

  it("is empty with nothing loaded", () => {
    expect(todayEco(undefined, "2026-11-17")).toEqual([]);
  });
});

describe("countriesOf", () => {
  it("counts and orders busiest first", () => {
    const got = countriesOf([ev({ country: "中国" }), ev({}), ev({})]);
    expect(got).toEqual([{ country: "美国", count: 2 }, { country: "中国", count: 1 }]);
  });
});

describe("todayEarnings", () => {
  const row = (n: string, market: "CN" | "NA", when?: string): EarningsRow => ({
    t: n, code: n, n, market, date: "2026-11-17", status: "scheduled", when,
  });
  const cal = (rows: EarningsRow[]) => ({
    by_date: [{ date: "2026-11-17", rows }, { date: "2026-11-18", rows: [row("Z", "NA")] }],
  }) as unknown as EarningsCalendar;

  it("takes today's names only, 盘前 before 盘后 before A-shares", () => {
    const got = todayEarnings(cal([
      row("cn", "CN"), row("after", "NA", "盘后"), row("before", "NA", "盘前"),
    ]), "2026-11-17").map((r) => r.n);
    expect(got).toEqual(["before", "after", "cn"]);
  });

  it("is empty on a day with nothing", () => {
    expect(todayEarnings(cal([]), "2026-11-19")).toEqual([]);
    expect(todayEarnings(undefined, "2026-11-17")).toEqual([]);
  });
});

describe("released / shouldPop", () => {
  it("a release with a value has come out", () => {
    expect(released(ev({ value: "3.1%" }))).toBe(true);
    expect(released(ev({}))).toBe(false);
  });

  it("pops once a day, and not when storage throws", () => {
    expect(shouldPop("2026-11-17", () => null)).toBe(true);
    expect(shouldPop("2026-11-17", () => "2026-11-16")).toBe(true);
    expect(shouldPop("2026-11-17", () => "2026-11-17")).toBe(false);
    expect(shouldPop("2026-11-17", () => { throw new Error("blocked"); })).toBe(false);
  });
});
