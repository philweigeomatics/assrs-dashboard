/**
 * 今日 — what the calendar holds for the viewer's today, for the 🔔 brief.
 *
 * Built from the same two responses the 日历 page uses (and under the same
 * query keys, so opening one warms the other): the week of economic releases
 * and the watchlist's earnings dates. Nothing here fetches.
 */

import type { EarningsCalendar, EarningsRow, EcoEvent, EcoWeek } from "./types";
import { localDay } from "./localDay";

/** Countries the brief shows until the viewer picks their own. */
export const DEFAULT_COUNTRIES = ["美国", "中国"];

/**
 * The release's day where the viewer is. Without an instant (no time from
 * Tushare) it keeps Tushare's Beijing date rather than inventing midnight —
 * the same rule as the calendar's week view.
 */
export function ecoDayOf(e: EcoEvent): string {
  return e.at ? localDay(new Date(e.at)) : e.date;
}

/** Today's releases in time order; untimed ones last. */
export function todayEco(week: EcoWeek | undefined, day: string): EcoEvent[] {
  if (!week) return [];
  return week.events
    .filter((e) => ecoDayOf(e) === day)
    .sort((a, b) => (a.at ?? "￿").localeCompare(b.at ?? "￿"));
}

/** Countries among `events`, busiest first. */
export function countriesOf(events: EcoEvent[]): { country: string; count: number }[] {
  const n = new Map<string, number>();
  for (const e of events) n.set(e.country, (n.get(e.country) ?? 0) + 1);
  return [...n.entries()]
    .map(([country, count]) => ({ country, count }))
    .sort((a, b) => b.count - a.count || a.country.localeCompare(b.country));
}

/** 盘前 reports before the open, so it comes first; A-shares carry no hour. */
const WHEN_ORDER: Record<string, number> = { 盘前: 0, 盘中: 1, 盘后: 2 };

/**
 * Watched names reporting today. A-share dates are the exchange's (Beijing)
 * date; for a viewer in the Americas a filing on that date lands overnight or
 * in their morning of the same date, so the date is matched as given.
 */
export function todayEarnings(cal: EarningsCalendar | undefined,
                              day: string): EarningsRow[] {
  if (!cal) return [];
  const rows = (cal.by_date ?? []).find((b) => b.date === day)?.rows ?? [];
  const rank = (r: EarningsRow) =>
    r.market === "NA" ? (WHEN_ORDER[r.when ?? ""] ?? 1) : 3;
  return [...rows].sort((a, b) => rank(a) - rank(b) || a.n.localeCompare(b.n));
}

/** Has the number come out yet? Tushare fills `value` once it has. */
export function released(e: EcoEvent): boolean {
  return !!e.value;
}

/** localStorage key holding the last day the brief opened by itself. */
export const SEEN_KEY = "assrs.today.seen";

/** Whether the brief should open by itself now. Storage failures say no. */
export function shouldPop(day: string, read: () => string | null): boolean {
  try {
    return read() !== day;
  } catch {
    return false;
  }
}
