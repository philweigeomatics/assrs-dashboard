/**
 * Local calendar days, done from the local clock rather than from UTC.
 *
 * `toISOString().slice(0, 10)` is the obvious way to get "YYYY-MM-DD" and it
 * is wrong for this: it converts to UTC first, so local midnight anywhere east
 * of Greenwich lands on the previous day. In Shanghai, Monday 00:00 is Sunday
 * 16:00 UTC, and a week that starts on "Sunday" is off by one all the way
 * across.
 */

/** "YYYY-MM-DD" for the day this Date falls on where the viewer is. */
export function localDay(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`
    + `-${String(d.getDate()).padStart(2, "0")}`;
}

/** Monday of the week containing `d`, at local midnight. */
export function localMonday(d: Date): Date {
  const out = new Date(d);
  out.setDate(out.getDate() - ((out.getDay() + 6) % 7));
  out.setHours(0, 0, 0, 0);
  return out;
}

/** The viewer's IANA zone, e.g. "America/Toronto". */
export function viewerZone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "";
  } catch {
    return "";
  }
}

/** "GMT-4" style label, for saying which clock the times are on. */
export function zoneLabel(d: Date = new Date()): string {
  try {
    const parts = new Intl.DateTimeFormat(undefined, { timeZoneName: "shortOffset" })
      .formatToParts(d);
    return parts.find((p) => p.type === "timeZoneName")?.value ?? "";
  } catch {
    return "";
  }
}

/** "14:00" on the viewer's clock. */
export function localClock(at: string): string {
  const d = new Date(at);
  if (Number.isNaN(d.getTime())) return "";
  return `${String(d.getHours()).padStart(2, "0")}`
    + `:${String(d.getMinutes()).padStart(2, "0")}`;
}

/** True when the viewer is not on Beijing time, so a reference is worth showing. */
export function offBeijing(at: string, beijingDay: string,
                           beijingClock: string): boolean {
  const d = new Date(at);
  if (Number.isNaN(d.getTime())) return false;
  return localDay(d) !== beijingDay || localClock(at) !== beijingClock;
}
