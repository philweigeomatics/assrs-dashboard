/**
 * 📈 对比 — your indexes on one chart, each rebased to 100 at the start of
 * the range, so a 1,000-point index and a 3,000-point one are read as moves.
 *
 * 今日 is the US session live: every five minutes, from what each index held
 * at the previous close (index_api.live — nothing stored, one Yahoo fetch a
 * minute shared by everyone). A-share indexes have no live feed on this data
 * tier; they join on the other ranges once their close is stored.
 *
 * Lines are told apart by a categorical palette, never red/green — and each
 * return in the legend uses its own market's colour convention.
 */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../lib/api";
import type { IndexCompare as Compare, IndexLive } from "../../lib/types";
import { fixed, moveClass, signed } from "../../lib/format";
import { usePersistentState } from "../../lib/usePersistentState";
import { useIndexLive } from "../../lib/indexLive";

const PALETTE = ["#0062cc", "#ff9500", "#8e44ad", "#16a085", "#b45309", "#d63384", "#5b6770", "#2e86de", "#6c8e00", "#c0392b"];
type Range = "today" | "1m" | "3m" | "ytd" | "1y" | "all";
const RANGES: { id: Range; label: string }[] = [
  { id: "today", label: "今日（美股盘中）" }, { id: "1m", label: "1 月" }, { id: "3m", label: "3 月" },
  { id: "ytd", label: "今年" }, { id: "1y", label: "1 年" }, { id: "all", label: "全部" },
];

function startOf(range: Range, last: string): string {
  const d = new Date(last + "T00:00:00");
  if (range === "ytd") return `${d.getFullYear()}-01-01`;
  if (range === "all") return "0000-00-00";
  const months = { "1m": 1, "3m": 3, "1y": 12 }[range as "1m" | "3m" | "1y"];
  d.setMonth(d.getMonth() - months);
  return d.toISOString().slice(0, 10);
}

type Line = { id: number; name: string; market: "CN" | "US"; color: string; ret: number | null; v: (number | null)[] };

export function IndexCompare() {
  const cmp = useQuery({ queryKey: ["idx", "compare"], queryFn: api.indexCompare, staleTime: 10 * 60_000 });
  const hasUS = (cmp.data?.indexes ?? []).some((x) => x.market === "US");
  const [range, setRange] = usePersistentState<Range>("assrs.index.cmpRange", "3m");
  const live = useIndexLive(hasUS);
  const [hidden, setHidden] = usePersistentState<number[]>("assrs.index.cmpHidden", []);

  if (cmp.isPending) return <section className="card p-6 label text-center">读取全部自建指数…</section>;
  if (cmp.isError) return <section className="card p-4 text-[13px] text-up">{(cmp.error as Error).message}</section>;
  const all = cmp.data!.indexes;
  if (!all.length) return <section className="card p-6 label text-center">还没有自建指数。</section>;
  const color = (id: number) => PALETTE[all.findIndex((x) => x.id === id) % PALETTE.length]!;
  const toggle = (id: number) => setHidden(hidden.includes(id) ? hidden.filter((x) => x !== id) : [...hidden, id]);

  return (
    <section className="card p-3 flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-[14px] font-semibold">📈 指数对比</h3>
        <span className="label">每条线在区间起点 = 100，看的是涨跌幅而不是点位</span>
        <div className="ml-auto flex rounded-lg bg-sunken p-0.5 overflow-x-auto max-w-full">
          {RANGES.map((r) => (
            <button key={r.id} onClick={() => setRange(r.id)} disabled={r.id === "today" && !hasUS}
              className={`h-7 px-2.5 rounded-md text-[12.5px] whitespace-nowrap disabled:opacity-40 ${
                range === r.id ? "bg-panel shadow-sm font-semibold" : "text-ink-mute hover:text-ink"}`}>{r.label}</button>
          ))}
        </div>
      </div>
      {hasUS && <LiveStrip live={live.data} loading={live.isPending} error={live.isError ? (live.error as Error).message : null} />}
      {range === "today"
        ? <Today live={live.data} all={all} hidden={hidden} color={color} toggle={toggle} />
        : <History all={all} range={range} hidden={hidden} color={color} toggle={toggle} />}
    </section>
  );
}

function LiveStrip({ live, loading, error }: { live?: IndexLive; loading: boolean; error: string | null }) {
  if (error) return <p className="label">美股盘中读取失败：{error}</p>;
  if (loading || !live) return <p className="label animate-pulse">读取美股盘中价格…</p>;
  if (!live.session) return <p className="label">美股暂无盘中数据。</p>;
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-lg bg-sunken px-2.5 py-1.5">
      <span className={`text-[12px] font-semibold ${live.open ? "text-cyan" : "text-ink-mute"}`}>
        {live.open ? "● 美股盘中" : "美股已收盘"}
      </span>
      <span className="label">{live.session} · 截至纽约 {live.as_of}{live.open ? " · 每分钟更新" : ""}</span>
      {live.indexes.map((x) => (
        <span key={x.id} className="text-[12.5px] tnum whitespace-nowrap" title={x.error ?? `前收 ${x.ref_date} ${fixed(x.ref_level, 2)}`}>
          <span className="text-ink-dim">{x.name}</span>{" "}
          {x.error ? <span className="label">—</span> : <>
            <b>{fixed(x.level, 2)}</b> <span className={moveClass(x.chg, false)}>{signed(x.chg, 2, "%")}</span>
          </>}
        </span>
      ))}
    </div>
  );
}

function History({ all, range, hidden, color, toggle }: {
  all: Compare["indexes"]; range: Exclude<Range, "today">; hidden: number[];
  color: (id: number) => string; toggle: (id: number) => void;
}) {
  const { dates, lines } = useMemo(() => {
    const shown = all.filter((x) => !hidden.includes(x.id));
    const last = all.reduce((m, x) => (x.dates[x.dates.length - 1]! > m ? x.dates[x.dates.length - 1]! : m), "");
    // A common start: the range's, or the latest first day among the shown, whichever is later.
    let from = startOf(range, last);
    for (const x of shown) if (x.dates[0]! > from) from = x.dates[0]!;
    const set = new Set<string>();
    for (const x of shown) for (const d of x.dates) if (d >= from) set.add(d);
    const dates = [...set].sort();
    const lines: Line[] = all.map((x) => {
      const m = new Map(x.dates.map((d, i) => [d, x.close[i]!]));
      let base: number | null = null, carry: number | null = null;
      const v = dates.map((d) => {
        const c = m.get(d) ?? carry;                     // a market holiday carries the last close
        carry = c;
        if (c == null) return null;
        if (base == null) base = c;
        return (c / base) * 100;
      });
      const lastV = [...v].reverse().find((y) => y != null) ?? null;
      return { id: x.id, name: x.name, market: x.market, color: color(x.id), v,
               ret: lastV != null ? lastV - 100 : null };
    });
    return { dates, lines };
  }, [all, range, hidden, color]);
  return <Chart xs={dates} lines={lines} hidden={hidden} toggle={toggle}
    note={`共同起点 ${dates[0] ?? "—"}：选中指数中最晚开始的那一天，或区间起点`} />;
}

function Today({ live, all, hidden, color, toggle }: {
  live?: IndexLive; all: Compare["indexes"]; hidden: number[];
  color: (id: number) => string; toggle: (id: number) => void;
}) {
  if (!live?.times?.length) return <p className="label py-6 text-center">美股盘中数据读取中，或今日尚未开盘。</p>;
  const lines: Line[] = live.indexes.filter((x) => !x.error && x.levels).map((x) => {
    // Starting from the previous close makes the gap at the opening bell visible.
    const v = [100, ...x.levels!.map((y) => (y != null && x.ref_level ? (y / x.ref_level) * 100 : null))];
    return { id: x.id, name: x.name, market: "US" as const, color: color(x.id), v, ret: x.chg ?? null };
  });
  const cn = all.filter((x) => x.market === "CN").map((x) => x.name);
  return <Chart xs={["前收", ...live.times]} lines={lines} hidden={hidden} toggle={toggle}
    note={`前一日收盘 = 100 · 纽约时间 · 5 分钟一点${cn.length ? ` · A 股指数（${cn.join("、")}）没有盘中数据，收盘后在其他区间里出现` : ""}`} />;
}

function Chart({ xs, lines, hidden, toggle, note }: {
  xs: string[]; lines: Line[]; hidden: number[]; toggle: (id: number) => void; note: string;
}) {
  const W = 1000, H = 300, PAD = 8;
  const [hover, setHover] = useState<number | null>(null);
  const shown = lines.filter((l) => !hidden.includes(l.id));
  const vals = shown.flatMap((l) => l.v).filter((v): v is number => v != null);
  const lo = Math.min(100, ...vals), hi = Math.max(100, ...vals);
  const n = xs.length;
  const x = (i: number) => (i / Math.max(1, n - 1)) * W;
  const y = (v: number) => H - PAD - ((v - lo) / (hi - lo || 1)) * (H - 2 * PAD);
  const path = (v: (number | null)[]) => {
    let s = "", pen = false;
    v.forEach((p, i) => { if (p == null) { pen = false; return; } s += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(p).toFixed(1)}`; pen = true; });
    return s;
  };
  const h = hover ?? n - 1;
  return (
    <div className="flex flex-col gap-2">
      <div className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[300px]" preserveAspectRatio="none"
          onMouseMove={(e) => {
            const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            setHover(Math.max(0, Math.min(n - 1, Math.round(((e.clientX - r.left) / r.width) * (n - 1)))));
          }}
          onMouseLeave={() => setHover(null)}>
          <line x1={0} x2={W} y1={y(100)} y2={y(100)} stroke="var(--color-line-bright)" strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
          {shown.map((l) => (
            <path key={l.id} d={path(l.v)} fill="none" stroke={l.color} strokeWidth={1.8} vectorEffect="non-scaling-stroke" />
          ))}
          {hover != null && <line x1={x(hover)} x2={x(hover)} y1={0} y2={H} stroke="var(--color-line-bright)" vectorEffect="non-scaling-stroke" />}
        </svg>
        <div className="absolute left-1 top-1 label tnum">{fixed(hi, 1)}</div>
        <div className="absolute left-1 bottom-1 label tnum">{fixed(lo, 1)}</div>
      </div>
      <div className="flex justify-between label tnum text-[11px] -mt-1">
        <span>{xs[0]}</span><span>{hover != null ? xs[h] : "虚线 = 起点 100"}</span><span>{xs[n - 1]}</span>
      </div>
      <div className="flex flex-col gap-1">
        {lines.map((l) => {
          const off = hidden.includes(l.id);
          const at = l.v[h];
          return (
            <button key={l.id} onClick={() => toggle(l.id)}
              className={`flex items-center gap-2 text-[12.5px] text-left rounded-md px-1.5 py-0.5 hover:bg-sunken ${off ? "opacity-40" : ""}`}>
              <span className="inline-block w-3 h-3 rounded-sm shrink-0" style={{ background: off ? "transparent" : l.color, border: `2px solid ${l.color}` }} />
              <span className="min-w-0 truncate">{l.name}</span>
              <span className="label">{l.market === "CN" ? "A股" : "美股"}</span>
              <span className="ml-auto tnum flex gap-3">
                {hover != null && at != null && <span className="label">{fixed(at, 1)}</span>}
                <span className={`w-16 text-right ${moveClass(l.ret, l.market === "CN")}`}>{signed(l.ret, 1, "%")}</span>
              </span>
            </button>
          );
        })}
      </div>
      <p className="label leading-snug">
        {note}。点图例显示 / 隐藏一条线。百分比用各自市场的颜色习惯（A 股红涨绿跌，美股绿涨红跌）。
        「今日」由前一日收盘时每只成分的持仓乘以此刻价格算出，与指数收盘时的记账方式一致；盘中只算不存，收盘后存下当天收盘。
      </p>
    </div>
  );
}
