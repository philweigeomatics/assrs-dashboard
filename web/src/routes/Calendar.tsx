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
import { api, ApiError } from "../lib/api";
import type { EarningsCalendar, EarningsRow, EcoWeek } from "../lib/types";
import { NavBar } from "../components/NavBar";
import { usePersistentState } from "../lib/usePersistentState";

const VIEWS = [
  { id: "eco", label: "🌍 经济数据" },
  { id: "earn", label: "📄 财报披露" },
] as const;

type View = typeof VIEWS[number]["id"];

const WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"];
const ALL = "__all__";

/** Monday of the week containing `d`. */
function monday(d: Date): Date {
  const out = new Date(d);
  out.setDate(out.getDate() - ((out.getDay() + 6) % 7));
  out.setHours(0, 0, 0, 0);
  return out;
}

const iso = (d: Date) => d.toISOString().slice(0, 10);

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
              note="全球宏观数据发布时间，来自 Tushare。时间为北京时间。"
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

/** Seven columns, one per day, each a scrollable list of releases. */
function EcoWeekView({ d }: { d: EcoWeek }) {
  const [country, setCountry] = useState<string>(ALL);
  const today = iso(new Date());

  const keep = (c: string) => country === ALL || c === country;
  const shown = d.days.map((day) => ({
    ...day, events: day.events.filter((e) => keep(e.country)),
  }));
  const total = shown.reduce((a, x) => a + x.events.length, 0);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <select value={country} onChange={(e) => setCountry(e.target.value)}
          aria-label="国家/地区"
          className="h-7 px-2 rounded-lg bg-sunken text-[12.5px] outline-none
            focus:ring-2 focus:ring-cyan/40">
          <option value={ALL}>全部地区（{d.total}）</option>
          {d.countries.map((c) => (
            <option key={c.country} value={c.country}>
              {c.country}（{c.count}）
            </option>
          ))}
        </select>
        <span className="label tnum">{d.from} → {d.to} · {total} 条</span>
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
                {day.events.map((e, i) => (
                  <div key={i} className="flex flex-col gap-0.5 text-[11.5px]
                    border-t border-line pt-1 first:border-0 first:pt-0">
                    <div className="flex items-baseline gap-1.5">
                      <span className="font-mono tnum text-ink-mute shrink-0">
                        {e.time || "—"}
                      </span>
                      <span className="text-ink-dim truncate">{e.country}</span>
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
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
      <p className="label leading-snug">
        一次显示一周 —— Tushare 的 eco_cal 每分钟只允许 20 次调用，而要拿全一天的数据
        就得单独请求那一天。已发生的日期会永久缓存，所以来回翻周是免费的。
      </p>
    </div>
  );
}

const STATUS: Record<EarningsRow["status"],
                     { label: string; cls: string; dot: string }> = {
  reported: { label: "已披露", cls: "text-ink-mute", dot: "var(--color-ink-mute)" },
  scheduled: { label: "待披露", cls: "text-cyan", dot: "var(--color-cyan)" },
  overdue: { label: "已过预约日", cls: "text-up font-medium", dot: "var(--color-up)" },
  unknown: { label: "无日期", cls: "text-ink-mute", dot: "var(--color-line-bright)" },
};

function EarningsView({ d }: { d: EarningsCalendar }) {
  const today = iso(new Date());

  if (d.watched === 0) {
    return (
      <p className="label py-6 text-center">
        自选股里还没有 A 股 —— 财报披露日期只对 A 股有。
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        {(Object.keys(STATUS) as EarningsRow["status"][])
          .filter((k) => d.counts[k])
          .map((k) => (
            <span key={k} className="flex items-center gap-1.5 text-[12px]">
              <span className="w-2.5 h-2.5 rounded-full"
                style={{ background: STATUS[k].dot }} />
              {STATUS[k].label}
              <span className="tnum font-medium">{d.counts[k]}</span>
            </span>
          ))}
        <span className="label tnum">共 {d.watched} 只 A 股自选</span>
      </div>

      {d.by_date.length === 0 ? (
        <p className="label py-6 text-center">这个报告期还没有披露日期。</p>
      ) : (
        <div className="grid gap-2 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3
          xl:grid-cols-5 items-start">
          {d.by_date.map((bucket) => {
            const past = bucket.date < today;
            return (
              <div key={bucket.date}
                className={`rounded-lg p-2 flex flex-col gap-1 min-w-0 ${
                  bucket.date === today ? "bg-cyan/10 ring-1 ring-cyan/40"
                    : "bg-sunken"}`}>
                <div className="flex items-baseline gap-1.5">
                  <span className={`text-[12.5px] font-semibold font-mono tnum ${
                    past ? "text-ink-mute" : ""}`}>
                    {bucket.date}
                  </span>
                  <span className="ml-auto label tnum">{bucket.rows.length}</span>
                </div>
                {bucket.rows.map((r) => (
                  <div key={r.code} className="flex items-baseline gap-1.5 text-[12px]">
                    <span className="w-1.5 h-1.5 rounded-full shrink-0 mt-1.5"
                      style={{ background: STATUS[r.status].dot }} />
                    <span className="truncate" title={`${r.n} ${r.t}`}>{r.n}</span>
                    {r.moved && (
                      <span className="text-brand-ink shrink-0"
                        title={`预约 ${r.pre_date} → 实际 ${r.actual_date}`}>
                        ⚑
                      </span>
                    )}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-[12px] border-collapse">
          <thead>
            <tr className="text-ink-mute">
              <th className="text-left font-normal pb-1 pr-2">股票</th>
              <th className="text-left font-normal pb-1 px-2">代码</th>
              <th className="text-left font-normal pb-1 px-2">披露日</th>
              <th className="text-left font-normal pb-1 px-2">预约日</th>
              <th className="text-left font-normal pb-1 px-2">实际日</th>
              <th className="text-left font-normal pb-1 pl-2">状态</th>
            </tr>
          </thead>
          <tbody>
            {d.rows.map((r) => (
              <tr key={r.code} className="border-t border-line">
                <td className="py-1 pr-2 truncate max-w-[140px]">{r.n}</td>
                <td className="py-1 px-2 font-mono tnum text-ink-mute">{r.t}</td>
                <td className="py-1 px-2 font-mono tnum">{r.date ?? "—"}</td>
                <td className="py-1 px-2 font-mono tnum text-ink-mute">
                  {r.pre_date ?? "—"}
                </td>
                <td className={`py-1 px-2 font-mono tnum ${
                  r.moved ? "text-brand-ink font-medium" : "text-ink-mute"}`}>
                  {r.actual_date ?? "—"}
                </td>
                <td className={`py-1 pl-2 ${STATUS[r.status].cls}`}>
                  {STATUS[r.status].label}
                  {r.moved && <span className="text-brand-ink"> · 改期</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {d.missing.length > 0 && (
        <p className="label leading-snug">
          这个报告期还没有披露日期的自选股（{d.missing.length} 只）：
          {d.missing.slice(0, 12).join("、")}
          {d.missing.length > 12 && ` 等 ${d.missing.length} 只`}。
        </p>
      )}
      <p className="label leading-snug">
        「披露日」优先取实际披露日，没有就取交易所预约日。⚑ 表示公司实际披露的日子
        和自己预约的不是同一天。
      </p>
    </div>
  );
}
