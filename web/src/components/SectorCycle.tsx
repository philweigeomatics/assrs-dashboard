/**
 * 板块周期 — where each 板块 is in its own trend cycle right now, and which
 * pairs have taken turns before.
 *
 * ① 现在 answers "is it going up or coming down right now": a phase per
 *    sector, how many days in, and how long that phase usually lasts for
 *    THIS sector. On top, the proven see-saw pairs and whether they are on
 *    opposite sides today.
 * ② 两两对照 answers "has this happened before": both oscillators over
 *    two years, the stretches they spent on opposite sides shaded, every
 *    turn marked, and the evidence that it beats chance.
 *
 * The statistics and their checks are in sector_cycle.py.
 */

import { useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { CyclePairDetail, CyclePhase, CycleRow, SectorCycle as Cycle } from "../lib/types";
import { fixed, signed } from "../lib/format";
import { usePersistentState } from "../lib/usePersistentState";
import {
  PHASE_CLASS, PHASE_HINT, PHASE_LABEL, PHASE_ORDER, livePairs, progress, sparkPath,
} from "../lib/cycle";

type Basis = "abs" | "rel";
const A_COL = "#0062cc";
const B_COL = "#b45309";

export function SectorCycle() {
  const [basis, setBasis] = usePersistentState<Basis>("assrs.cycle.basis", "abs");
  const [view, setView] = usePersistentState<"now" | "pair">("assrs.cycle.view", "now");
  const [pair, setPair] = usePersistentState<[string, string] | null>("assrs.cycle.pair", null);
  const [focus, setFocus] = useState<string | null>(null);

  const q = useQuery({
    queryKey: ["sector-cycle", basis],
    queryFn: () => api.sectorCycle(basis),
    staleTime: 60 * 60_000,
  });

  const open = (a: string, b: string) => { setPair([a, b]); setView("pair"); };

  return (
    <section className="card p-3 flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-[14px] font-semibold">📈 板块周期</h2>
        <span className="label">每个板块在自己的趋势周期里走到哪了，哪些板块此起彼伏</span>
        <div className="ml-auto flex flex-wrap gap-2">
          <Seg value={view} set={setView} options={[{ v: "now", label: "① 现在" }, { v: "pair", label: "② 两两对照" }]} />
          <Seg value={basis} set={setBasis} options={[
            { v: "abs", label: "绝对", title: "板块指数本身的涨跌" },
            { v: "rel", label: "相对沪深300", title: "每个板块先减去 β × 沪深300：只看大盘方向解释不了的部分" },
          ]} />
        </div>
      </div>
      {q.isPending && <p className="label py-8 text-center">读取 26 个板块指数并做随机对照检验，首次约 20–30 秒…</p>}
      {q.isError && (
        <p className="text-[12.5px] text-up">{(q.error as Error).message}
          <button onClick={() => q.refetch()} className="ml-2 text-cyan">重试</button></p>
      )}
      {q.data && (view === "now"
        ? <Now d={q.data} focus={focus} open={open}
            pick={(sec) => { setFocus(sec); setView("pair"); }} />
        : <Pairs d={q.data} basis={basis} pair={pair} setPair={setPair} focus={focus} setFocus={setFocus} />)}
    </section>
  );
}

function Seg<T extends string>({ value, set, options }: {
  value: T; set: (v: T) => void; options: { v: T; label: string; title?: string }[];
}) {
  return (
    <div className="flex rounded-md bg-sunken p-0.5">
      {options.map((o) => (
        <button key={o.v} title={o.title} onClick={() => set(o.v)}
          className={`h-6 px-2.5 rounded text-[12px] ${value === o.v ? "bg-panel shadow-sm font-semibold" : "text-ink-mute hover:text-ink"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

function PhaseChip({ phase }: { phase: CyclePhase | null }) {
  if (!phase) return <span className="label">—</span>;
  return (
    <span title={PHASE_HINT[phase]}
      className={`inline-block px-2 py-0.5 rounded-full text-[11.5px] font-semibold ${PHASE_CLASS[phase]}`}>
      {PHASE_LABEL[phase]}
    </span>
  );
}

// ── ① now ────────────────────────────────────────────────────────────────────
function Now({ d, focus, open, pick }: {
  d: Cycle; focus: string | null; open: (a: string, b: string) => void; pick: (sector: string) => void;
}) {
  const live = useMemo(() => livePairs(d.pairs, d.board), [d]);
  const counts = PHASE_ORDER.map((p) => [p, d.board.filter((r) => r.phase === p).length] as const);
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2 text-[12.5px]">
        <span className="label">{d.end} · {d.board.length} 个板块</span>
        {counts.map(([p, n]) => (
          <span key={p} className="flex items-center gap-1"><PhaseChip phase={p} /><b className="tnum">{n}</b></span>
        ))}
        <span className="label">四个阶段依次轮转：上行 → 回落 → 下行 → 回升 → 上行</span>
      </div>

      <div className="rounded-lg bg-sunken p-2.5 flex flex-col gap-1.5">
        <span className="text-[13px] font-semibold">
          已被检验的跷跷板 <span className="label font-normal">
            {d.start} 以来、校正 {d.pairs.length} 对检验后仍成立的 {live.length} 对 · 点一下看历史</span>
        </span>
        {live.length === 0 && <span className="label">这个口径下，没有哪一对板块的反向联动超出随机。试试「相对沪深300」。</span>}
        <div className="flex flex-wrap gap-1.5">
          {live.map(({ pair: p, a, b, opposite }) => (
            <button key={`${p.a}|${p.b}`} onClick={() => open(p.a, p.b)}
              className={`h-7 px-2 rounded-md text-[12px] border flex items-center gap-1.5 ${
                opposite ? "border-brand bg-brand/10" : "border-line bg-panel"}`}
              title={`相关 ${fixed(p.best_corr, 2)} · ${Math.round((p.opposite ?? 0) * 100)}% 的天数一上一下（随机 ${Math.round((p.opposite_expected ?? 0) * 100)}%）· q=${fixed(p.q, 3)}`}>
              <span>{p.a}</span><PhaseChip phase={a?.phase ?? null} />
              <span className="text-ink-mute">⇅</span>
              <span>{p.b}</span><PhaseChip phase={b?.phase ?? null} />
              {opposite && <span className="text-brand-ink font-semibold">此刻对立</span>}
            </button>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[720px] text-[12.5px] tnum">
          <thead>
            <tr className="label text-left">
              <th className="font-normal py-1">板块</th>
              <th className="font-normal">近半年（σ）</th>
              <th className="font-normal">阶段</th>
              <th className="font-normal">已持续 / 典型</th>
              <th className="font-normal text-right">偏离均线</th>
              <th className="font-normal text-right">上次穿越均线</th>
              {d.basis === "rel" && <th className="font-normal text-right" title="对沪深300的 β，已扣除">β</th>}
            </tr>
          </thead>
          <tbody>
            {PHASE_ORDER.flatMap((ph) => d.board.filter((r) => r.phase === ph)).concat(d.board.filter((r) => !r.phase))
              .map((r) => <Row key={r.sector} r={r} on={focus === r.sector} basis={d.basis}
                onClick={() => pick(r.sector)} />)}
          </tbody>
        </table>
      </div>
      <p className="label leading-snug">
        偏离均线 = 板块离自己 20 日均线多远，以它自己过去 120 天的正常偏离幅度为单位（+1σ = 比平时还要高出一截），
        所以平稳的白酒和剧烈的光模块可以放在同一把尺子上。阶段经过 3 日平滑并设缓冲区，避免在均线附近来回翻。
        「典型」= 这个板块自 {d.start} 以来同一阶段持续天数的中位数。点击一行在「两两对照」里查看它的搭档。
      </p>
    </div>
  );
}

function Row({ r, on, onClick, basis }: { r: CycleRow; on: boolean; onClick: () => void; basis: Basis }) {
  const W = 160, H = 26;
  const pr = progress(r);
  const zero = H / 2;
  return (
    <tr onClick={onClick} className={`border-t border-line cursor-pointer ${on ? "bg-cyan/5" : "hover:bg-elevated"}`}>
      <td className="py-1 pr-2 whitespace-nowrap font-medium">{r.sector}</td>
      <td className="py-1 pr-2">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-[160px] h-[26px] block" role="img" aria-label={`${r.sector} 偏离均线走势`}>
          <rect x={0} y={0} width={W} height={zero} fill="var(--color-up)" opacity={0.05} />
          <rect x={0} y={zero} width={W} height={zero} fill="var(--color-down)" opacity={0.05} />
          <line x1={0} x2={W} y1={zero} y2={zero} stroke="var(--color-line-bright)" />
          <path d={sparkPath(r.spark, W, H)} fill="none" stroke="var(--color-ink-dim)" strokeWidth={1.3} />
        </svg>
      </td>
      <td className="py-1 pr-2"><PhaseChip phase={r.phase} /></td>
      <td className="py-1 pr-2 whitespace-nowrap">
        <div className="flex items-center gap-1.5">
          <span>第 {r.days} 天</span>
          {r.typical != null && <span className="label">· 典型 {r.typical} 天</span>}
          {pr != null && (
            <span className="w-12 h-1.5 rounded-full bg-sunken overflow-hidden" title={`已走完典型时长的 ${Math.round(pr * 100)}%`}>
              <span className={`block h-full ${pr >= 1 ? "bg-brand" : "bg-ink-mute"}`} style={{ width: `${Math.min(100, pr * 100)}%` }} />
            </span>
          )}
        </div>
      </td>
      <td className="py-1 pr-2 text-right whitespace-nowrap">
        <span className={(r.osc ?? 0) >= 0 ? "text-up" : "text-down"}>{signed(r.osc, 1, "σ")}</span>
        <span className="text-ink-mute"> {(r.slope ?? 0) >= 0 ? "↗" : "↘"}</span>
      </td>
      <td className="py-1 pr-2 text-right whitespace-nowrap">
        {r.last_cross ? <>{r.last_cross.days_ago} 天前{r.last_cross.dir === "up" ? "上穿" : "下穿"}</> : "—"}
      </td>
      {basis === "rel" && <td className="py-1 text-right text-ink-mute">{fixed(r.beta, 2)}</td>}
    </tr>
  );
}

// ── ② pairs ──────────────────────────────────────────────────────────────────
function Pairs({ d, basis, pair, setPair, focus, setFocus }: {
  d: Cycle; basis: Basis; pair: [string, string] | null; setPair: (p: [string, string]) => void;
  focus: string | null; setFocus: (s: string | null) => void;
}) {
  const sectors = d.board.map((r) => r.sector);
  const list = d.pairs.filter((p) => !focus || p.a === focus || p.b === focus).slice(0, focus ? 25 : 30);
  const valid = pair && sectors.includes(pair[0]) && sectors.includes(pair[1]) ? pair : null;
  const sel: [string, string] = valid ?? (d.pairs[0] ? [d.pairs[0].a, d.pairs[0].b] : [sectors[0]!, sectors[1]!]);
  return (
    <div className="grid gap-3 xl:grid-cols-[360px_minmax(0,1fr)] items-start">
      <div className="flex flex-col gap-1.5 min-w-0">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[13px] font-semibold">最此起彼伏的组合</span>
          <select value={focus ?? ""} onChange={(e) => setFocus(e.target.value || null)}
            className="h-7 px-1.5 rounded-md bg-sunken text-[12px] outline-none">
            <option value="">全部板块</option>
            {sectors.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div className="max-h-[520px] overflow-y-auto">
          <table className="w-full text-[12px] tnum">
            <thead>
              <tr className="label text-left">
                <th className="font-normal">组合</th>
                <th className="font-normal text-right" title="两者偏离均线的相关（取 ±20 天内最负的错位）">相关</th>
                <th className="font-normal text-right" title="一上一下的天数占比 · 括号内为随机对照">对立天数</th>
                <th className="font-normal text-right" title="校正 325 对检验后的显著性；≤0.1 视为成立">q</th>
              </tr>
            </thead>
            <tbody>
              {list.map((p) => {
                const on = (p.a === sel[0] && p.b === sel[1]) || (p.a === sel[1] && p.b === sel[0]);
                const ok = (p.q ?? 1) <= 0.1;
                return (
                  <tr key={`${p.a}|${p.b}`} onClick={() => setPair([p.a, p.b])}
                    className={`border-t border-line cursor-pointer ${on ? "bg-cyan/10" : "hover:bg-elevated"}`}>
                    <td className="py-1 pr-1">{p.a} <span className="text-ink-mute">⇅</span> {p.b}
                      {p.best_lag !== 0 && <span className="label"> · {p.best_lag > 0 ? `${p.b}晚${p.best_lag}天` : `${p.a}晚${-p.best_lag}天`}</span>}</td>
                    <td className="py-1 text-right">{fixed(p.best_corr, 2)}</td>
                    <td className="py-1 text-right">{Math.round((p.opposite ?? 0) * 100)}%<span className="label"> ({Math.round((p.opposite_expected ?? 0) * 100)}%)</span></td>
                    <td className={`py-1 text-right ${ok ? "font-semibold text-cyan" : "text-ink-mute"}`}>{fixed(p.q, 2)}{ok ? "★" : ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <span className="label">★ = 校正后仍超出随机（q ≤ 0.1）。按相关从最负排起。</span>
      </div>
      <PairDetail a={sel[0]} b={sel[1]} basis={basis} />
    </div>
  );
}

function PairDetail({ a, b, basis }: { a: string; b: string; basis: Basis }) {
  const q = useQuery({
    queryKey: ["sector-cycle-pair", a, b, basis],
    queryFn: () => api.sectorCyclePair(a, b, basis),
    staleTime: 60 * 60_000,
  });
  if (q.isPending) return <p className="label py-8 text-center">计算 {a} × {b}…</p>;
  if (q.isError) return <p className="text-[12.5px] text-up">{(q.error as Error).message}</p>;
  const d = q.data!;
  return (
    <div className="flex flex-col gap-3 min-w-0">
      <div className="rounded-lg bg-sunken p-3 flex flex-col gap-1">
        <div className="flex flex-wrap items-center gap-2 text-[13px]">
          <b style={{ color: A_COL }}>{d.a}</b><PhaseChip phase={d.now.a.phase} />
          <span className="text-ink-mute">⇅</span>
          <b style={{ color: B_COL }}>{d.b}</b><PhaseChip phase={d.now.b.phase} />
        </div>
        <ul className="flex flex-col gap-1 list-disc pl-4">
          {d.conclusions.map((c, i) => <li key={i} className="text-[12.5px] leading-snug text-ink-dim">{c}</li>)}
        </ul>
      </div>
      <PairChart d={d} />
      <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
        <Stat k="反向相关" v={fixed(d.lag.best_corr, 2)} sub={`随机对照中位 ${fixed(d.lag.null_median, 2)}`}
          good={(d.lag.p ?? 1) < 0.05} />
        <Stat k="一上一下的天数" v={`${Math.round((d.opposite.share ?? 0) * 100)}%`}
          sub={`随机对照 ${Math.round((d.opposite.expected ?? 0) * 100)}%`} good={(d.opposite.p ?? 1) < 0.05} />
        <Stat k={`${d.a}转升 → ${d.b}转跌`} v={`${d.a_up_b_down.hits}/${d.a_up_b_down.events}`}
          sub={`随机约 ${Math.round((d.a_up_b_down.chance ?? 0) * d.a_up_b_down.events)} 次 · 前后 10 天`}
          good={(d.a_up_b_down.p ?? 1) < 0.05} />
        <Stat k={`${d.a}转跌 → ${d.b}转升`} v={`${d.a_down_b_up.hits}/${d.a_down_b_up.events}`}
          sub={`随机约 ${Math.round((d.a_down_b_up.chance ?? 0) * d.a_down_b_up.events)} 次 · 前后 10 天`}
          good={(d.a_down_b_up.p ?? 1) < 0.05} />
      </div>
      <LagCurve d={d} />
    </div>
  );
}

function Stat({ k, v, sub, good }: { k: string; v: string; sub: string; good: boolean }) {
  return (
    <div className="rounded-lg bg-sunken px-3 py-2">
      <div className="label truncate" title={k}>{k}</div>
      <div className={`text-[18px] font-semibold tnum ${good ? "text-cyan" : ""}`}>{v}</div>
      <div className="label">{sub}{good ? " · 超出随机" : ""}</div>
    </div>
  );
}

function PairChart({ d }: { d: CyclePairDetail }) {
  const W = 1000, H = 230;
  const box = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const n = d.dates.length;
  const vals = [...d.osc_a, ...d.osc_b].filter((v): v is number => v != null);
  const ext = Math.max(2, ...vals.map(Math.abs)) * 1.05;
  const x = (i: number) => (n < 2 ? 0 : (i / (n - 1)) * W);
  const y = (v: number) => H / 2 - (v / ext) * (H / 2);
  const path = (vs: (number | null)[]) => {
    let s = "", pen = false;
    vs.forEach((v, i) => { if (v == null) { pen = false; return; } s += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`; pen = true; });
    return s;
  };
  const bands: [number, number][] = [];
  d.opposite_mask.forEach((m, i) => {
    if (!m) return;
    const last = bands[bands.length - 1];
    if (last && last[1] === i - 1) last[1] = i; else bands.push([i, i]);
  });
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-3 text-[12px]">
        <span className="flex items-center gap-1"><span className="w-4 h-[3px] rounded-full" style={{ background: A_COL }} />{d.a}</span>
        <span className="flex items-center gap-1"><span className="w-4 h-[3px] rounded-full" style={{ background: B_COL }} />{d.b}</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-brand/20" />一个在均线上方、另一个在下方</span>
        <span className="label">▲▼ = {d.a} 上穿 / 下穿均线</span>
      </div>
      <div className="flex gap-1">
        <div className="relative w-8 shrink-0 h-[230px]">
          {[2, 0, -2].filter((t) => Math.abs(t) <= ext).map((t) => (
            <span key={t} className="absolute right-0 -translate-y-1/2 label font-mono tnum" style={{ top: `${(y(t) / H) * 100}%` }}>
              {t > 0 ? "+" : ""}{t}σ</span>
          ))}
        </div>
        <div ref={box} className="relative flex-1 min-w-0 h-[230px]"
          onMouseMove={(e) => { const r = box.current?.getBoundingClientRect(); if (r) setHover(Math.max(0, Math.min(n - 1, Math.round(((e.clientX - r.left) / r.width) * (n - 1))))); }}
          onMouseLeave={() => setHover(null)}>
          <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 w-full h-full" role="img"
            aria-label={`${d.a} 与 ${d.b} 的偏离均线走势，阴影为一上一下的时段`}>
            {bands.map(([s, e]) => <rect key={s} x={x(s)} width={Math.max(1.5, x(e) - x(s))} y={0} height={H} fill="var(--color-brand)" opacity={0.12} />)}
            <line x1={0} x2={W} y1={y(0)} y2={y(0)} stroke="var(--color-line-bright)" vectorEffect="non-scaling-stroke" />
            {d.cross_a.up.map((i) => <line key={`u${i}`} x1={x(i)} x2={x(i)} y1={0} y2={H} stroke={A_COL} strokeDasharray="2 4" opacity={0.5} vectorEffect="non-scaling-stroke" />)}
            <path d={path(d.osc_b)} fill="none" stroke={B_COL} strokeWidth={1.8} vectorEffect="non-scaling-stroke" />
            <path d={path(d.osc_a)} fill="none" stroke={A_COL} strokeWidth={1.8} vectorEffect="non-scaling-stroke" />
            {hover != null && <line x1={x(hover)} x2={x(hover)} y1={0} y2={H} stroke="var(--color-ink-mute)" vectorEffect="non-scaling-stroke" />}
          </svg>
          {d.cross_a.up.map((i) => <span key={`tu${i}`} className="absolute top-0 -translate-x-1/2 text-[10px]" style={{ left: `${(x(i) / W) * 100}%`, color: A_COL }}>▲</span>)}
          {d.cross_a.down.map((i) => <span key={`td${i}`} className="absolute bottom-0 -translate-x-1/2 text-[10px]" style={{ left: `${(x(i) / W) * 100}%`, color: A_COL }}>▼</span>)}
          {hover != null && (
            <div className="absolute top-2 card px-2 py-1 text-[12px] pointer-events-none z-10"
              style={hover > n * 0.6 ? { right: `${100 - (hover / Math.max(1, n - 1)) * 100 + 1}%` } : { left: `${(hover / Math.max(1, n - 1)) * 100 + 1}%` }}>
              <div className="font-mono">{d.dates[hover]}</div>
              <div style={{ color: A_COL }}>{d.a} {signed(d.osc_a[hover], 2, "σ")}</div>
              <div style={{ color: B_COL }}>{d.b} {signed(d.osc_b[hover], 2, "σ")}</div>
              {d.opposite_mask[hover] && <div className="text-brand-ink">一上一下</div>}
            </div>
          )}
        </div>
      </div>
      <div className="flex justify-between label font-mono tnum ml-9"><span>{d.dates[0]}</span><span>{d.dates[n - 1]}</span></div>
    </div>
  );
}

function LagCurve({ d }: { d: CyclePairDetail }) {
  const W = 420, H = 90;
  const lags = d.lag.lags;
  const x = (i: number) => (i / (lags.length - 1)) * W;
  const y = (v: number) => H / 2 - v * (H / 2);
  const pts = d.lag.curve.map((v, i) => (v == null ? "" : `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`)).join("");
  const best = lags.indexOf(d.lag.best_lag);
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[12.5px] font-semibold">谁先动？<span className="label font-normal">
        横轴：{d.b} 相对 {d.a} 错开的天数（正 = {d.b} 晚）；越往下越此消彼长；虚线 = 随机对照中位</span></span>
      <svg viewBox={`0 0 ${W} ${H + 14}`} className="w-full max-w-[460px] h-auto" role="img" aria-label="错位相关曲线">
        <line x1={0} x2={W} y1={y(0)} y2={y(0)} stroke="var(--color-line-bright)" />
        {d.lag.null_median != null && <line x1={0} x2={W} y1={y(d.lag.null_median)} y2={y(d.lag.null_median)} stroke="var(--color-ink-mute)" strokeDasharray="4 4" />}
        <line x1={x(lags.indexOf(0))} x2={x(lags.indexOf(0))} y1={0} y2={H} stroke="var(--color-line)" />
        <path d={pts} fill="none" stroke="var(--color-ink)" strokeWidth={1.6} />
        {best >= 0 && d.lag.best_corr != null && <circle cx={x(best)} cy={y(d.lag.best_corr)} r={3.5} fill="var(--color-brand)" />}
        {[-20, -10, 0, 10, 20].map((l) => (
          <text key={l} x={x(lags.indexOf(l))} y={H + 12} fontSize={10} textAnchor="middle" fill="var(--color-ink-mute)">{l > 0 ? `+${l}` : l}</text>
        ))}
      </svg>
    </div>
  );
}
