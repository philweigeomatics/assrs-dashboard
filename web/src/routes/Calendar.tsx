/**
 * 🗓️ 日历 — what is scheduled, from two sources that answer different questions.
 *
 * 经济数据 is the world's macro release schedule; 财报披露 is when the stocks
 * you actually watch will report. They share a page because they share a
 * use — knowing what is coming this week — and nothing else, so they are a
 * toggle rather than two panels stacked.
 *
 * The economic view is a WEEK, not a month, and that is a constraint rather
 * than a preference: Tushare's eco_cal allows 20 calls a minute and complete
 * data needs one call per day. See calendar_data for the measurements.
 */

import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api, ApiError } from "../lib/api";
import type { EarningsCalendar, EarningsRow, EcoEvent, EcoWeek }
  from "../lib/types";
import { NavBar } from "../components/NavBar";
import { usePersistentState } from "../lib/usePersistentState";
import { signed } from "../lib/format";
import { cellDate, monthGrid, parseMonth, safeMonth, shiftMonth }
  from "../lib/monthGrid";
import { localClock, localDay, localMonday, offBeijing, viewerZone, zoneLabel }
  from "../lib/localDay";

const VIEWS = [
  { id: "eco", label: "🌍 经济数据" },
  { id: "earn", label: "📄 财报披露" },
] as const;

type View = typeof VIEWS[number]["id"];

const WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"];
const ALL = "__all__";

// Both of these used to go through toISOString(), which converts to UTC
// first — so local midnight anywhere east of Greenwich landed on the previous
// day and the whole week was off by one. See lib/localDay.
const monday = localMonday;
const iso = localDay;

export function Calendar() {
  useEffect(() => { document.title = "ASSRS · 日历"; }, []);
  const [view, setView] = usePersistentState<View>("assrs.cal.view", "eco");
  const [offset, setOffset] = useState(0);
  const [period, setPeriod] = useState<string | null>(null);

  const start = useMemo(() => {
    const d = monday(new Date());
    d.setDate(d.getDate() + offset * 7);
    return iso(d);
  }, [offset]);

  const eco = useQuery({
    queryKey: ["cal", "eco", start],
    queryFn: () => api.ecoCalendar(start),
    enabled: view === "eco",
    staleTime: 30 * 60_000,
  });
  const earn = useQuery({
    queryKey: ["cal", "earn", period],
    queryFn: () => api.earningsCalendar(period),
    enabled: view === "earn",
    staleTime: 30 * 60_000,
  });

  return (
    <div className="min-h-screen">
      <NavBar />
      <main className="max-w-[1800px] mx-auto px-3 py-3 flex flex-col gap-3">
        <section className="card p-2 flex flex-wrap items-center gap-1">
          {VIEWS.map((v) => (
            <button key={v.id} onClick={() => setView(v.id)}
              className={`h-8 px-3 rounded-lg text-[13px] font-medium transition-colors ${
                view === v.id ? "bg-elevated text-ink" : "text-ink-mute hover:text-ink"
              }`}>
              {v.label}
            </button>
          ))}
        </section>

        {view === "eco" && (
          <section className="card p-3 flex flex-col gap-2">
            <Header
              title="🌍 经济数据日历"
              note="全球宏观数据发布时间，来自 Tushare。时间已换算成你所在时区。"
              left={
                <div className="flex items-center gap-1">
                  <Step onClick={() => setOffset(offset - 1)}>← 上周</Step>
                  <button onClick={() => setOffset(0)} disabled={offset === 0}
                    className="h-7 px-2 rounded-md bg-sunken text-[12px] disabled:opacity-50">
                    本周
                  </button>
                  <Step onClick={() => setOffset(offset + 1)}>下周 →</Step>
                </div>
              } />
            <Body q={eco}>{eco.data && <EcoWeekView d={eco.data} />}</Body>
          </section>
        )}

        {view === "earn" && (
          <section className="card p-3 flex flex-col gap-2">
            <Header
              title="📄 财报披露日历"
              note="你的自选股里 A 股部分的财报披露日期。预约日与实际披露日不同的会标出来。"
              left={earn.data && (
                // Fall back to a period that is actually in the list: a
                // <select> whose value matches no option silently displays
                // the first one, which would name the wrong quarter.
                <select value={
                  earn.data.periods.some((p) => p.end === (period ?? earn.data.period))
                    ? (period ?? earn.data.period)
                    : earn.data.periods[earn.data.periods.length - 1]?.end ?? ""}
                  onChange={(e) => setPeriod(e.target.value)}
                  aria-label="报告期"
                  className="h-7 px-2 rounded-lg bg-sunken text-[12.5px] outline-none
                    focus:ring-2 focus:ring-cyan/40">
                  {earn.data.periods.map((p) => (
                    <option key={p.end} value={p.end}>{p.label}</option>
                  ))}
                </select>
              )} />
            <Body q={earn}>{earn.data && <EarningsView d={earn.data} />}</Body>
          </section>
        )}
      </main>
    </div>
  );
}

function Header({ title, note, left }: {
  title: string; note: string; left?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
      <h2 className="text-[14.5px] font-semibold">{title}</h2>
      {left}
      <p className="label leading-snug flex-1 min-w-[220px]">{note}</p>
    </div>
  );
}

function Step({ onClick, children }: {
  onClick: () => void; children: React.ReactNode;
}) {
  return (
    <button onClick={onClick}
      className="h-7 px-2 rounded-md bg-sunken text-[12px]">{children}</button>
  );
}

function Body({ q, children }: {
  q: { isPending: boolean; isError: boolean; error: unknown; refetch: () => void };
  children: React.ReactNode;
}) {
  if (q.isPending) {
    return <div className="py-10 text-center label">读取中…（首次约 5–9 秒）</div>;
  }
  if (q.isError) {
    return (
      <div className="py-6 text-center flex flex-col gap-1.5">
        <p className="text-[12.5px] text-up">{(q.error as ApiError).message}</p>
        <button onClick={() => q.refetch()} className="text-cyan text-[13px]">重试</button>
      </div>
    );
  }
  return <>{children}</>;
}

/**
 * Seven columns, one per day — the VIEWER's day.
 *
 * Tushare reports eco_cal in Beijing time, date included, and that is not a
 * cosmetic detail: on one real week 128 of 409 events belonged to a different
 * calendar day in Toronto than the one Tushare labelled them with. So the
 * grouping is done here from the instant, and the server pads the range a day
 * either side so the first and last columns are complete.
 */
function EcoWeekView({ d }: { d: EcoWeek }) {
  const [country, setCountry] = useState<string>(ALL);
  const [showBeijing, setShowBeijing] = usePersistentState<boolean>(
    "assrs.cal.bjtime", false);
  const today = iso(new Date());
  const zone = viewerZone();
  const offset = zoneLabel();
  const beijing = zone === d.source_tz;

  // Bucket by the viewer's calendar day, then keep only the week asked for —
  // the padding days exist to fill these buckets, not to be shown.
  const { days, countries, total } = useMemo(() => {
    const wanted: string[] = [];
    const base = new Date(`${d.from}T00:00:00`);
    for (let i = 0; i < 7; i += 1) {
      const day = new Date(base);
      day.setDate(day.getDate() + i);
      wanted.push(iso(day));
    }

    const bucket = new Map<string, EcoEvent[]>(wanted.map((k) => [k, []]));
    const tally = new Map<string, number>();
    for (const e of d.events) {
      // No instant means no time from Tushare; fall back to its Beijing date
      // rather than inventing midnight and shifting it into another day.
      const key = e.at ? localDay(new Date(e.at)) : e.date;
      const slot = bucket.get(key);
      if (!slot) continue;
      slot.push(e);
      tally.set(e.country, (tally.get(e.country) ?? 0) + 1);
    }

    for (const list of bucket.values()) {
      list.sort((a, b) => (a.at ?? "").localeCompare(b.at ?? ""));
    }
    return {
      days: wanted.map((k) => ({
        date: k,
        weekday: (new Date(`${k}T00:00:00`).getDay() + 6) % 7,
        events: bucket.get(k) ?? [],
      })),
      countries: [...tally.entries()]
        .map(([c, n]) => ({ country: c, count: n }))
        .sort((a, b) => b.count - a.count),
      total: [...bucket.values()].reduce((a, l) => a + l.length, 0),
    };
  }, [d]);

  const keep = (c: string) => country === ALL || c === country;
  const shown = days.map((day) => ({
    ...day, events: day.events.filter((e) => keep(e.country)),
  }));
  const visible = shown.reduce((a, x) => a + x.events.length, 0);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <select value={country} onChange={(e) => setCountry(e.target.value)}
          aria-label="国家/地区"
          className="h-7 px-2 rounded-lg bg-sunken text-[12.5px] outline-none
            focus:ring-2 focus:ring-cyan/40">
          <option value={ALL}>全部地区（{total}）</option>
          {countries.map((c) => (
            <option key={c.country} value={c.country}>
              {c.country}（{c.count}）
            </option>
          ))}
        </select>
        <span className="label tnum">{d.from} → {d.to} · {visible} 条</span>
        {!beijing && (
          <label className="label flex items-center gap-1.5 cursor-pointer"
            title="同时显示 Tushare 原始的北京时间，方便对账">
            <input type="checkbox" checked={showBeijing}
              onChange={(e) => setShowBeijing(e.target.checked)}
              className="accent-[var(--color-cyan)]" />
            并显示北京时间
          </label>
        )}
      </div>

      <div className="grid gap-2 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7
        items-start">
        {shown.map((day) => (
          <div key={day.date}
            className={`rounded-lg p-2 flex flex-col gap-1.5 min-w-0 ${
              day.date === today ? "bg-cyan/10 ring-1 ring-cyan/40" : "bg-sunken"}`}>
            <div className="flex items-baseline gap-1.5">
              <span className="text-[12.5px] font-semibold">
                {WEEKDAYS[day.weekday]}
              </span>
              <span className="label font-mono tnum">{day.date.slice(5)}</span>
              <span className="ml-auto label tnum">{day.events.length}</span>
            </div>
            {day.events.length === 0 ? (
              <span className="label">—</span>
            ) : (
              <div className="flex flex-col gap-1.5 max-h-[420px] overflow-y-auto">
                {day.events.map((e, i) => {
                  const clock = e.at ? localClock(e.at) : e.time;
                  const moved = !!e.at && !beijing
                    && offBeijing(e.at, e.date, e.time);
                  return (
                    <div key={i} className="flex flex-col gap-0.5 text-[11.5px]
                      border-t border-line pt-1 first:border-0 first:pt-0">
                      <div className="flex items-baseline gap-1.5">
                        <span className="font-mono tnum text-ink-mute shrink-0">
                          {clock || "—"}
                        </span>
                        <span className="text-ink-dim truncate">{e.country}</span>
                        {showBeijing && moved && (
                          <span className="ml-auto font-mono tnum text-[10.5px]
                            text-ink-mute shrink-0"
                            title="Tushare 给的北京时间">
                            京 {e.date.slice(5)} {e.time}
                          </span>
                        )}
                      </div>
                      <span className="leading-snug" title={e.event}>{e.event}</span>
                      {(e.value || e.forecast || e.prev) && (
                        <div className="flex flex-wrap gap-x-2 text-[11px] tnum">
                          {e.value && (
                            <span className="font-medium">实 {e.value}</span>
                          )}
                          {e.forecast && (
                            <span className="text-ink-mute">预 {e.forecast}</span>
                          )}
                          {e.prev && (
                            <span className="text-ink-mute">前 {e.prev}</span>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ))}
      </div>
      <p className="label leading-snug">
        时间已换算成<b>你所在时区</b>
        {zone ? ` ${zone}` : ""}{offset ? `（${offset}）` : ""} ——
        Tushare 给的是北京时间，包括日期，所以在 UTC+8 以外有相当一部分事件
        本来就不属于它标的那一天。
        一次显示一周：eco_cal 每分钟只允许 20 次调用，而要拿全一天的数据就得
        单独请求那一天。已发生的日期会永久缓存，来回翻周是免费的。
      </p>
    </div>
  );
}

const STATUS: Record<EarningsRow["status"],
                     { label: string; dot: string; chip: string }> = {
  reported: { label: "已披露", dot: "var(--color-ink-mute)",
              chip: "bg-sunken text-ink-dim" },
  scheduled: { label: "待披露", dot: "var(--color-cyan)",
               chip: "bg-cyan/10 text-ink" },
  overdue: { label: "已过预约日", dot: "var(--color-up)",
             chip: "bg-up/10 text-up font-medium" },
  unknown: { label: "无日期", dot: "var(--color-line-bright)",
             chip: "bg-sunken text-ink-mute" },
};

const DOW = ["一", "二", "三", "四", "五", "六", "日"];

function EarningsView({ d }: { d: EarningsCalendar }) {
  const today = iso(new Date());

  // Open on the month that actually holds the dates. A quarter's A-share
  // disclosures cluster into a few weeks, so landing on today's month would
  // often show an empty grid a page away from everything.
  //
  // An API that does not send `focus` is not hypothetical: Pages deploys in
  // seconds and Cloud Run takes minutes, so the new page runs against the old
  // API every time. It used to take the route down.
  const focus = safeMonth(d.focus);
  const months = d.months ?? [];
  const [ym, setYm] = useState(focus);
  useEffect(() => { setYm(focus); }, [focus]);

  const byDate = useMemo(
    () => new Map((d.by_date ?? []).map((b) => [b.date, b.rows])),
    [d.by_date]);

  const watched = d.watched ?? { cn: 0, na: 0, total: 0 };
  if (watched.total === 0) {
    return (
      <p className="label py-6 text-center">
        自选股是空的 —— 财报日历看的是你自选的那些股票。
      </p>
    );
  }

  const { year, month } = parseMonth(ym) ?? parseMonth(focus)!;
  const weeks = monthGrid(ym);
  const shift = (by: number) => setYm(shiftMonth(ym, by));
  const inMonth = (d.rows ?? []).filter((r) => r.date?.startsWith(ym)).length;
  const chronological = [...months].sort((a, b) => a.ym.localeCompare(b.ym));

  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1">
          <Step onClick={() => shift(-1)}>◀</Step>
          <span className="text-[13.5px] font-semibold tnum w-[104px] text-center">
            {year} 年 {month} 月
          </span>
          <Step onClick={() => shift(1)}>▶</Step>
        </div>
        <span className="label tnum">本月 {inMonth} 只</span>
        {months.length > 0 && ym !== focus && (
          <button onClick={() => setYm(focus)} className="text-[12px] text-cyan">
            回到 {focus.replace("-", " 年 ")} 月
          </button>
        )}
        <div className="ml-auto flex flex-wrap items-center gap-x-3 gap-y-1">
          {(Object.keys(STATUS) as EarningsRow["status"][])
            .filter((k) => d.counts?.[k])
            .map((k) => (
              <span key={k} className="flex items-center gap-1.5 text-[12px]">
                <span className="w-2.5 h-2.5 rounded-full"
                  style={{ background: STATUS[k].dot }} />
                {STATUS[k].label}
                <span className="tnum font-medium">{d.counts[k]}</span>
              </span>
            ))}
          <span className="label tnum">
            A股 {watched.cn} · 美加 {watched.na}
          </span>
        </div>
      </div>

      {/* A seven-column month cannot collapse to one column and still be a
          calendar, so on a narrow screen it scrolls sideways instead. */}
      <div className="overflow-x-auto">
        <div className="min-w-[680px]">
          <div className="grid grid-cols-7 gap-px bg-line rounded-lg overflow-hidden">
            {DOW.map((day, i) => (
              <div key={day}
                className={`bg-elevated px-2 py-1 text-[12px] font-medium text-center ${
                  i >= 5 ? "text-ink-mute" : "text-ink-dim"}`}>
                {day}
              </div>
            ))}

            {weeks.flat().map((day, i) => {
              if (day === 0) {
                return <div key={`p${i}`} className="bg-canvas min-h-[92px]" />;
              }
              const key = cellDate(ym, day);
              const rows = byDate.get(key) ?? [];
              const isToday = key === today;
              return (
                <div key={key}
                  className={`bg-panel min-h-[92px] p-1.5 flex flex-col gap-1 ${
                    isToday ? "ring-2 ring-inset ring-cyan" : ""}`}>
                  <div className="flex items-baseline gap-1">
                    <span className={`text-[12px] tnum ${
                      isToday ? "text-cyan font-bold" : "text-ink-mute"}`}>
                      {day}
                    </span>
                    {rows.length > 0 && (
                      <span className="ml-auto text-[10.5px] tnum text-ink-mute">
                        {rows.length}
                      </span>
                    )}
                  </div>
                  <div className="flex flex-col gap-0.5 max-h-[150px] overflow-y-auto">
                    {rows.map((r) => <Chip key={`${r.market}-${r.code}`} r={r} />)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {chronological.length > 1 && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="label">其他有安排的月份：</span>
          {chronological.filter((mm) => mm.ym !== ym).map((mm) => (
            <button key={mm.ym} onClick={() => setYm(mm.ym)}
              className="h-6 px-2 rounded-md bg-sunken text-[12px]">
              {mm.ym.replace("-", " 年 ")} 月 · {mm.count}
            </button>
          ))}
        </div>
      )}

      <details className="mt-1">
        <summary className="text-[12.5px] cursor-pointer text-ink-dim">
          全部明细（{(d.rows ?? []).length} 条，按日期排序）
        </summary>
        <div className="overflow-x-auto mt-1.5">
          <table className="w-full text-[12px] border-collapse">
            <thead>
              <tr className="text-ink-mute">
                <th className="text-left font-normal pb-1 pr-2">股票</th>
                <th className="text-left font-normal pb-1 px-2">市场</th>
                <th className="text-left font-normal pb-1 px-2">日期</th>
                <th className="text-left font-normal pb-1 px-2">时段 / 预约</th>
                <th className="text-right font-normal pb-1 px-2">预估 EPS</th>
                <th className="text-right font-normal pb-1 px-2">实际 EPS</th>
                <th className="text-left font-normal pb-1 pl-2">状态</th>
              </tr>
            </thead>
            <tbody>
              {(d.rows ?? []).map((r) => (
                <tr key={`${r.market}-${r.code}`} className="border-t border-line">
                  <td className="py-1 pr-2">
                    <StockLink r={r} />
                  </td>
                  <td className="py-1 px-2 text-ink-mute">
                    {r.market === "CN" ? "A股" : "美加"}
                  </td>
                  <td className="py-1 px-2 font-mono tnum">{r.date ?? "—"}</td>
                  <td className="py-1 px-2 text-ink-mute">
                    {r.market === "NA" ? (r.when || "—")
                      : (r.pre_date ?? "—")}
                  </td>
                  <td className="py-1 px-2 text-right tnum text-ink-mute">
                    {r.eps_estimate ?? "—"}
                  </td>
                  <td className="py-1 px-2 text-right tnum">
                    {r.eps_reported ?? "—"}
                    {r.surprise_pct != null && (
                      <span className={r.surprise_pct >= 0 ? "text-up" : "text-down"}>
                        {" "}{signed(r.surprise_pct, 1, "%")}
                      </span>
                    )}
                  </td>
                  <td className="py-1 pl-2">
                    {STATUS[r.status].label}
                    {r.moved && <span className="text-brand-ink"> · 改期</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      {((d.missing?.cn?.length ?? 0) + (d.missing?.na?.length ?? 0)) > 0 && (
        <p className="label leading-snug">
          暂时没有日期的自选股：
          {(d.missing?.cn?.length ?? 0) > 0
            && ` A股 ${d.missing.cn.length} 只（${d.missing.cn.slice(0, 8).join("、")}${
              d.missing.cn.length > 8 ? " 等" : ""}）`}
          {(d.missing?.na?.length ?? 0) > 0
            && ` 美加 ${d.missing.na.length} 只（${d.missing.na.slice(0, 8).join("、")}${
              d.missing.na.length > 8 ? " 等" : ""}）`}
          。ETF 本来就不发财报。
        </p>
      )}
      <p className="label leading-snug">
        A股日期来自交易所预约/实际披露日；美加来自 Yahoo 的财报日程，带时段
        （盘前 / 盘后），未来的日期多为<b>预估</b>，公司正式公告后可能变动。
        点格子里的股票可直接打开它的个股分析。
      </p>
    </div>
  );
}

/**
 * One stock in a day cell.
 *
 * A link, not a label: the calendar is where you notice a name is reporting,
 * and the next thing anyone wants is its chart. Carries its own timing too —
 * 盘前/盘后 for a US name is the difference between tonight's gap and
 * tomorrow's, and a date alone throws that away.
 */
function Chip({ r }: { r: EarningsRow }) {
  const na = r.market === "NA";
  const title = [
    `${r.n} ${r.t}`,
    r.date ?? "",
    na ? (r.when || "") : "",
    na && r.eps_estimate != null ? `预估 EPS ${r.eps_estimate}` : "",
    na && r.eps_reported != null ? `实际 EPS ${r.eps_reported}` : "",
    r.moved ? `预约 ${r.pre_date} → 实际 ${r.actual_date}` : "",
    STATUS[r.status].label,
    "点击查看个股分析",
  ].filter(Boolean).join(" · ");

  return (
    <Link to={`/?t=${encodeURIComponent(r.t)}`} title={title}
      className={`px-1 py-0.5 rounded text-[11px] leading-[15px] shrink-0
        flex items-baseline gap-1 hover:ring-1 hover:ring-cyan/50
        ${STATUS[r.status].chip}`}>
      <span className="truncate">{r.n}</span>
      {na && r.when && (
        <span className="ml-auto shrink-0 text-[9.5px] text-ink-mute">
          {r.when === "盘前" ? "前" : r.when === "盘后" ? "后" : "中"}
        </span>
      )}
      {r.moved && <span className="shrink-0 text-brand-ink">⚑</span>}
    </Link>
  );
}

function StockLink({ r }: { r: EarningsRow }) {
  return (
    <Link to={`/?t=${encodeURIComponent(r.t)}`}
      className="hover:text-cyan truncate inline-block max-w-[150px]"
      title={`${r.n} ${r.t} · 点击查看个股分析`}>
      {r.n}
      <span className="ml-1 font-mono tnum text-[10.5px] text-ink-mute">
        {r.t}
      </span>
    </Link>
  );
}
