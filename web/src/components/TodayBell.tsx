/**
 * 🔔 今日 — today's macro releases and the watchlist's earnings, in one pop-up.
 *
 * Opens by itself the first time the app is opened each day, and right after
 * signing in (LoginPage clears the day it last opened). After that it is the
 * bell in the header, with today's count on it.
 *
 * It is a doorway, not a second calendar: a release takes you to 日历 with it
 * already ticked, ready for 合并解读; a stock takes you to its 个股分析.
 */

import { useCallback, useEffect, useMemo, useRef, useSyncExternalStore } from "react";
import { createPortal } from "react-dom";
import { Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import type { EarningsRow, EcoEvent } from "../lib/types";
import { localClock, localDay, localMonday } from "../lib/localDay";
import { usePersistentState } from "../lib/usePersistentState";
import { countriesOf, DEFAULT_COUNTRIES, released, SEEN_KEY, shouldPop,
         todayEarnings, todayEco } from "../lib/today";

/** Matches the calendar's 合并解读 limit (eco_advisor.MAX_EVENTS). */
const MAX_PICK = 40;
const WEEKDAYS = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];

/** What the calendar reads from the navigation to tick on arrival. */
export type EcoHandoff = { ecoPick: EcoEvent[] };

// Open or shut, held outside React. Every page renders its own header, and
// some render a different one while loading, so the bell is remounted often;
// state inside it would close the pop-up the moment it had opened.
let isOpen = false;
const listeners = new Set<() => void>();
function setOpen(v: boolean) {
  if (isOpen === v) return;
  isOpen = v;
  listeners.forEach((f) => f());
}
function useOpen() {
  return useSyncExternalStore(
    (f) => { listeners.add(f); return () => { listeners.delete(f); }; },
    () => isOpen);
}

export function TodayBell() {
  const now = new Date();
  const day = localDay(now);
  const start = localDay(localMonday(now));

  // The calendar page's own query keys, so either one warms the other.
  const eco = useQuery({
    queryKey: ["cal", "eco", start],
    queryFn: () => api.ecoCalendar(start),
    staleTime: 30 * 60_000,
  });
  const earn = useQuery({
    queryKey: ["cal", "earn", null],
    queryFn: () => api.earningsCalendar(null),
    staleTime: 30 * 60_000,
  });

  const [countries, setCountries] = usePersistentState<string[]>(
    "assrs.today.countries", DEFAULT_COUNTRIES);
  const open = useOpen();
  const shut = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!shouldPop(day, () => localStorage.getItem(SEEN_KEY))) return;
    setOpen(true);
    try { localStorage.setItem(SEEN_KEY, day); } catch { /* shown anyway */ }
  }, [day]);

  const allEco = useMemo(() => todayEco(eco.data, day), [eco.data, day]);
  const shownEco = allEco.filter((e) => countries.includes(e.country));
  const stocks = useMemo(() => todayEarnings(earn.data, day), [earn.data, day]);
  const count = shownEco.length + stocks.length;

  return (
    <>
      <button onClick={() => setOpen(true)}
        aria-label={`今日提醒${count ? `：${count} 项` : ""}`} aria-expanded={open}
        title="今天的宏观数据和自选股财报"
        className="relative w-10 h-10 shrink-0 grid place-items-center rounded-lg
          text-ink-dim hover:text-ink hover:bg-elevated active:bg-elevated">
        <BellIcon />
        {count > 0 && (
          <span className="absolute top-1 right-0.5 min-w-[17px] h-[17px] px-1 rounded-full
            bg-cyan text-white text-[10.5px] font-bold leading-[17px] text-center tnum">
            {count > 99 ? "99+" : count}
          </span>
        )}
      </button>
      {open && (
        <Panel onClose={shut} now={now}
          eco={eco} allEco={allEco} shownEco={shownEco}
          earn={earn} stocks={stocks}
          countries={countries} setCountries={setCountries} />
      )}
    </>
  );
}

type Q = { isPending: boolean; isError: boolean; error: unknown; refetch: () => void };

function Panel({ onClose, now, eco, allEco, shownEco, earn, stocks,
                 countries, setCountries }: {
  onClose: () => void; now: Date;
  eco: Q; allEco: EcoEvent[]; shownEco: EcoEvent[];
  earn: Q & { data?: { watched?: { total: number } } }; stocks: EarningsRow[];
  countries: string[]; setCountries: (c: string[]) => void;
}) {
  const close = useRef<HTMLButtonElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    close.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const toCalendar = (picked: EcoEvent[]) => {
    onClose();
    navigate("/calendar", { state: { ecoPick: picked.slice(0, MAX_PICK) } satisfies EcoHandoff });
  };
  const flip = (c: string) => setCountries(
    countries.includes(c) ? countries.filter((x) => x !== c) : [...countries, c]);

  const present = countriesOf(allEco);
  const pending = shownEco.filter((e) => !released(e)).length;

  // Portalled: the header's backdrop-blur would otherwise be the containing
  // block for `fixed` and pin the panel inside its 56px strip.
  return createPortal(
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label="今日提醒">
      <div onClick={onClose} className="absolute inset-0 bg-black/30" />
      <div className="absolute inset-x-2 top-[calc(60px+env(safe-area-inset-top))]
        lg:inset-x-auto lg:right-4 lg:w-[440px]
        max-h-[calc(100dvh-76px-env(safe-area-inset-top))]
        rounded-xl bg-panel shadow-xl border border-line flex flex-col overflow-hidden">
        <div className="px-3 h-12 flex items-center gap-2 border-b border-line shrink-0">
          <span className="text-[15px] font-semibold">🔔 今日</span>
          <span className="label tnum">
            {now.getMonth() + 1} 月 {now.getDate()} 日 {WEEKDAYS[now.getDay()]}
          </span>
          <button ref={close} onClick={onClose} aria-label="关闭今日提醒"
            className="ml-auto w-9 h-9 -mr-1 grid place-items-center rounded-lg text-ink-mute
              hover:bg-elevated active:bg-elevated text-[20px] leading-none">
            ×
          </button>
        </div>

        <div className="flex-1 overflow-y-auto overscroll-contain p-3 flex flex-col gap-4">
          <section className="flex flex-col gap-1.5">
            <h3 className="text-[13.5px] font-semibold flex items-baseline gap-2">
              📄 自选股财报
              {!earn.isPending && !earn.isError && (
                <span className="label tnum font-normal">{stocks.length} 只</span>
              )}
            </h3>
            <State q={earn}>
              {stocks.length === 0 ? (
                <p className="label">
                  {earn.data?.watched?.total === 0
                    ? "自选股是空的。" : "今天没有自选股发财报。"}
                </p>
              ) : (
                <ul className="flex flex-col">
                  {stocks.map((r) => (
                    <li key={`${r.market}-${r.code}`}>
                      <Link to={`/?t=${encodeURIComponent(r.t)}`} onClick={onClose}
                        className="flex items-center gap-2 px-2 py-2 -mx-2 rounded-lg
                          hover:bg-elevated active:bg-elevated">
                        <span className="min-w-0 flex-1">
                          <span className="text-[13.5px] font-medium">{r.n}</span>
                          <span className="ml-1.5 font-mono tnum text-[11px] text-ink-mute">
                            {r.t}
                          </span>
                          <span className="block label tnum">
                            {[r.market === "CN" ? "A股" : "美加",
                              r.when || (r.market === "CN" ? "交易所披露日" : ""),
                              r.eps_estimate != null ? `预估 EPS ${r.eps_estimate}` : "",
                              r.eps_reported != null ? `实际 EPS ${r.eps_reported}` : "",
                              r.estimated ? "日期为预估" : "",
                            ].filter(Boolean).join(" · ")}
                          </span>
                        </span>
                        <span className="text-[12px] text-cyan shrink-0">个股分析 →</span>
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </State>
          </section>

          <section className="flex flex-col gap-1.5">
            <h3 className="text-[13.5px] font-semibold flex items-baseline gap-2">
              🌍 宏观数据
              {!eco.isPending && !eco.isError && (
                <span className="label tnum font-normal">
                  {shownEco.length} 条{pending ? ` · ${pending} 条待公布` : ""}
                </span>
              )}
            </h3>
            <State q={eco}>
              {present.length > 0 && (
                <div className="flex flex-wrap gap-1" aria-label="显示哪些国家/地区">
                  {present.map((c) => {
                    const on = countries.includes(c.country);
                    return (
                      <button key={c.country} onClick={() => flip(c.country)}
                        aria-pressed={on}
                        className={`h-7 px-2 rounded-md text-[12px] tnum ${
                          on ? "bg-cyan/15 text-ink ring-1 ring-cyan/40"
                             : "bg-sunken text-ink-mute"}`}>
                        {c.country} {c.count}
                      </button>
                    );
                  })}
                </div>
              )}
              {shownEco.length === 0 ? (
                <p className="label">
                  {allEco.length === 0 ? "今天没有经济数据发布。"
                    : "选中的国家/地区今天没有发布，点上面的国家加进来。"}
                </p>
              ) : (
                <>
                  <ul className="flex flex-col">
                    {shownEco.map((e) => (
                      <li key={`${e.date}|${e.time}|${e.country}|${e.event}`}>
                        <button onClick={() => toCalendar([e])}
                          title="去日历，已勾选这一条，可再选几条一起让 AI 解读"
                          className="w-full text-left flex items-start gap-2 px-2 py-1.5 -mx-2
                            rounded-lg hover:bg-elevated active:bg-elevated">
                          <span className="font-mono tnum text-[12px] text-ink-mute w-[38px]
                            shrink-0 pt-px">
                            {e.at ? localClock(e.at) : "—"}
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className={`text-[13px] leading-snug ${
                              released(e) ? "text-ink-dim" : ""}`}>
                              <span className="text-ink-mute">{e.country} · </span>
                              {e.event}
                            </span>
                            {(e.value || e.forecast || e.prev) && (
                              <span className="flex flex-wrap gap-x-2 text-[11.5px] tnum">
                                {e.value
                                  ? <span className="font-medium">实 {e.value}</span>
                                  : <span className="text-cyan">待公布</span>}
                                {e.forecast && <span className="text-ink-mute">预 {e.forecast}</span>}
                                {e.prev && <span className="text-ink-mute">前 {e.prev}</span>}
                              </span>
                            )}
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                  <button onClick={() => toCalendar(shownEco)}
                    className="self-start h-8 px-3 rounded-lg bg-violet text-white
                      text-[12.5px] font-semibold">
                    🤖 把这 {Math.min(shownEco.length, MAX_PICK)} 条带去日历解读 →
                  </button>
                  <p className="label leading-snug">
                    点一条进日历，它已勾选好；再勾几条，点「合并解读」让 AI 一起读。
                    时间是你所在时区。
                  </p>
                </>
              )}
            </State>
          </section>
        </div>
      </div>
    </div>,
    document.body,
  );
}

function State({ q, children }: { q: Q; children: React.ReactNode }) {
  if (q.isPending) return <p className="label">读取中…（首次约 5–9 秒）</p>;
  if (q.isError) {
    return (
      <p className="text-[12.5px] text-up">
        {(q.error as ApiError).message}{" "}
        <button onClick={() => q.refetch()} className="text-cyan">重试</button>
      </p>
    );
  }
  return <>{children}</>;
}

function BellIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden fill="none"
      stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 8.5a5 5 0 0 1 10 0c0 3.2 1.2 4.8 2 5.5H3c.8-.7 2-2.3 2-5.5Z" />
      <path d="M8.2 16.5a2 2 0 0 0 3.6 0" />
    </svg>
  );
}
