/**
 * 跷跷板 — do the stocks in this basket take turns?
 *
 * Reads top to bottom: the verdict (with the honesty checks spelled out), the
 * two camps and how far apart they have drifted, the matrix of every pair,
 * then one pair up close — scatter, rolling correlation and the period-by-
 * period strip where you can see the turns being taken.
 *
 * Defaults to 去大盘 because that is where a rotation shows: raw returns
 * carry the market, which moves both sides together and hides it.
 */

import { useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { Seesaw, SeesawRes } from "../lib/types";
import { fixed, signed } from "../lib/format";
import { usePersistentState } from "../lib/usePersistentState";
import { cellFill } from "../lib/season";
import { NEG, POS, corr, corrFill, corrInk, opposite, rollingCorr } from "../lib/seesaw";

type Res = "daily" | "weekly";
type Basis = "resid" | "raw";

export function SeesawPanel({ symbols, window, ready }: {
  symbols: string[]; window: string; ready: boolean;
}) {
  const [res, setRes] = usePersistentState<Res>("assrs.seesaw.res", "daily");
  const [basis, setBasis] = usePersistentState<Basis>("assrs.seesaw.basis", "resid");

  // Waits for the ranking: both routes read the same cached closes, so the
  // second request is instant once the first has fetched them.
  const q = useQuery({
    queryKey: ["seesaw", symbols.join(","), window],
    queryFn: () => api.seesaw(symbols, window),
    enabled: ready && symbols.length >= 2,
    staleTime: 20 * 60_000,
  });

  return (
    <section className="card p-3 flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-[14px] font-semibold">🎢 跷跷板 · 轮动检测</h2>
        <span className="label">一只涨、另一只跌？日线与周线，扣除大盘前后</span>
        <div className="ml-auto flex flex-wrap gap-2">
          <Seg value={res} set={setRes}
            options={[{ v: "daily" as Res, label: "日线" }, { v: "weekly" as Res, label: "周线" }]} />
          <Seg value={basis} set={setBasis}
            options={[
              { v: "resid" as Basis, label: "去大盘", disabled: q.data ? !q.data.daily.resid : false,
                title: "每只股票减去 β × 指数涨跌：轮动只在这一层看得见" },
              { v: "raw" as Basis, label: "原始" },
            ]} />
        </div>
      </div>
      {q.isPending && <p className="label py-6 text-center">{ready ? "正在计算…" : "等待排名数据…"}</p>}
      {q.isError && (
        <p className="text-[12.5px] text-up">
          {(q.error as Error).message}
          <button onClick={() => q.refetch()} className="ml-2 text-cyan">重试</button>
        </p>
      )}
      {q.data && <Body d={q.data} res={res} basis={q.data[res].resid ? basis : "raw"} />}
    </section>
  );
}

function Seg<T extends string>({ value, set, options }: {
  value: T; set: (v: T) => void; options: { v: T; label: string; disabled?: boolean; title?: string }[];
}) {
  return (
    <div className="flex rounded-md bg-sunken p-0.5">
      {options.map((o) => (
        <button key={o.v} disabled={o.disabled} title={o.title} onClick={() => set(o.v)}
          className={`h-6 px-2.5 rounded text-[12px] disabled:opacity-40 ${
            value === o.v ? "bg-panel shadow-sm font-semibold" : "text-ink-mute hover:text-ink"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Body({ d, res, basis }: { d: Seesaw; res: Res; basis: Basis }) {
  const r = d[res];
  const syms = d.symbols;
  const C = (basis === "resid" ? r.resid_corr : r.raw_corr) ?? r.raw_corr;
  const series = (basis === "resid" ? r.resid : r.returns) ?? r.returns;
  const name = (s: string) => d.labels[s] ?? s;

  // Default to the strongest seesaw pair, else the first two.
  const first = r.strongest ? [r.strongest.a, r.strongest.b] : [syms[0]!, syms[1]!];
  const [pair, setPair] = useState<[string, string]>([first[0]!, first[1]!]);
  const [a, b] = syms.includes(pair[0]) && syms.includes(pair[1]) ? pair : [first[0]!, first[1]!];

  const status = d.camps?.confirmed ? { t: "跷跷板成立", c: "bg-cyan/15 text-cyan" }
    : d.camps?.emerging ? { t: "最近才出现", c: "bg-brand/15 text-brand-ink" }
    : d.camps && (d.camps.out_of_sample.corr ?? 0) < 0 && d.camps.out_of_sample.same_split
      ? { t: "有迹象", c: "bg-brand/15 text-brand-ink" }
      : { t: "未发现稳定的跷跷板", c: "bg-sunken text-ink-mute" };

  return (
    <>
      <div className="rounded-lg bg-sunken p-3 flex flex-col gap-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[13px] font-semibold">结论</span>
          <span className={`px-2 py-0.5 rounded-full text-[11.5px] font-semibold ${status.c}`}>{status.t}</span>
          <span className="label">{d.from} → {d.to} · 日 {d.daily.n} / 周 {d.weekly.n} 个数据点</span>
        </div>
        <ul className="flex flex-col gap-1 list-disc pl-4">
          {d.conclusions.map((c, i) => <li key={i} className="text-[12.5px] leading-snug text-ink-dim">{c}</li>)}
        </ul>
      </div>

      {d.camps && <Camps d={d} name={name} />}

      <div className="grid gap-3 xl:grid-cols-[minmax(0,460px)_minmax(0,1fr)] items-start">
        <Matrix d={d} r={r} C={C} basis={basis} name={name} pair={[a, b]}
          onPick={(x, y) => setPair([x, y])} />
        <PairDetail d={d} r={r} series={series} basis={basis} a={a} b={b} name={name} res={res} />
      </div>

      <p className="label leading-snug">
        去大盘 = 每只股票的收益减去 β × {d.benchmark ?? "指数"}（β 按同一窗口估计），用外部指数而不是篮子平均 ——
        减篮子平均会人为制造负相关。周线取每周最后一个交易日收盘。「最明显的跷跷板」的显著线已按比较的配对数放宽；
        分组在前半段选出、在后半段检验，避免从噪音里挑出好看的分组。历史统计，非预测。
      </p>
    </>
  );
}

// ── camps ────────────────────────────────────────────────────────────────────
const VW = 1000;

function Camps({ d, name }: { d: Seesaw; name: (s: string) => string }) {
  const c = d.camps!;
  const vals = c.spread.map((v) => v ?? NaN);
  const lo = Math.min(0, ...vals.filter(Number.isFinite));
  const hi = Math.max(0, ...vals.filter(Number.isFinite));
  const H = 150;
  const pad = Math.max(1, (hi - lo) * 0.08);
  const y = (v: number) => H - ((v - (lo - pad)) / (hi + pad - (lo - pad))) * H;
  const x = (i: number) => (vals.length < 2 ? 0 : (i / (vals.length - 1)) * VW);
  const path = vals.map((v, i) => Number.isFinite(v) ? `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}` : "").join("");
  const half = c.spread_dates.indexOf(c.out_of_sample.half_from);
  const last = [...vals].reverse().find(Number.isFinite);
  const cell = (v: number | null | undefined) => (
    <td className="text-right font-mono tnum px-1.5" style={{ color: v != null && v < 0 ? NEG : undefined }}>{fixed(v, 2)}</td>
  );
  return (
    <div className="grid gap-3 lg:grid-cols-[minmax(0,320px)_minmax(0,1fr)] items-start">
      <div className="flex flex-col gap-2 text-[12.5px]">
        <div><span className="label">阵营 A</span>
          <div className="flex flex-wrap gap-1 mt-0.5">{c.a.map((s) => <Chip key={s}>{name(s)}</Chip>)}</div></div>
        <div><span className="label">阵营 B</span>
          <div className="flex flex-wrap gap-1 mt-0.5">{c.b.map((s) => <Chip key={s}>{name(s)}</Chip>)}</div></div>
        <table className="text-[12px]">
          <thead><tr className="label"><th className="text-left font-normal">去大盘相关</th>
            <th className="text-right font-normal px-1.5">日</th><th className="text-right font-normal px-1.5">周</th></tr></thead>
          <tbody>
            <tr><td className="label">A 对 B</td>{cell(c.daily.between)}{cell(c.weekly?.between)}</tr>
            <tr><td className="label">A 组内</td>{cell(c.daily.within_a)}{cell(c.weekly?.within_a)}</tr>
            <tr><td className="label">B 组内</td>{cell(c.daily.within_b)}{cell(c.weekly?.within_b)}</tr>
            <tr><td className="label">最近</td>{cell(c.daily.recent)}{cell(c.weekly?.recent)}</tr>
            <tr title="用前半段选出的分组，在后半段的组间相关"><td className="label">样本外 ⓘ</td>
              {cell(c.out_of_sample.corr)}<td /></tr>
          </tbody>
        </table>
      </div>
      <div className="flex flex-col gap-1">
        <div className="flex items-baseline gap-2 text-[12px]">
          <span className="font-semibold">A 组相对 B 组（等权，累计）</span>
          <span className="tnum font-medium">{signed(last ?? null, 1, "%")}</span>
          <span className="label">向上 = A 跑赢；竖线 = 样本外检验起点</span>
        </div>
        <svg viewBox={`0 0 ${VW} ${H}`} preserveAspectRatio="none" className="w-full h-[150px] block"
          role="img" aria-label="阵营 A 相对阵营 B 的累计收益">
          <line x1={0} x2={VW} y1={y(0)} y2={y(0)} stroke="var(--color-line-bright)" vectorEffect="non-scaling-stroke" />
          {half > 0 && <line x1={x(half)} x2={x(half)} y1={0} y2={H} stroke="var(--color-ink-mute)"
            strokeDasharray="3 4" vectorEffect="non-scaling-stroke" />}
          <path d={path} fill="none" stroke="var(--color-ink)" strokeWidth={1.6} vectorEffect="non-scaling-stroke" />
        </svg>
        <div className="flex justify-between label font-mono tnum">
          <span>{c.spread_dates[0]}</span><span>{c.spread_dates[c.spread_dates.length - 1]}</span>
        </div>
      </div>
    </div>
  );
}

function Chip({ children }: { children: React.ReactNode }) {
  return <span className="px-1.5 py-0.5 rounded bg-sunken text-[12px]">{children}</span>;
}

// ── matrix ───────────────────────────────────────────────────────────────────
function Matrix({ d, r, C, basis, name, pair, onPick }: {
  d: Seesaw; r: SeesawRes; C: (number | null)[][]; basis: Basis; name: (s: string) => string;
  pair: [string, string]; onPick: (a: string, b: string) => void;
}) {
  const syms = d.symbols;
  const opp = (basis === "resid" ? r.opposite_resid : r.opposite_raw) ?? r.opposite_raw;
  const sig = r.significant_at ?? 1;
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <span className="text-[13px] font-semibold">两两相关 <span className="label font-normal">
        {basis === "resid" ? "去大盘" : "原始"} · {r === d.daily ? "日" : "周"}线 · 点格子看这一对 · ● = 单独看显著（|r| &gt; {sig.toFixed(2)}）</span></span>
      <div className="overflow-x-auto">
        <table className="text-[12px] tnum border-separate border-spacing-[2px]">
          <thead>
            <tr><th />{syms.map((s) => (
              <th key={s} className="label font-normal px-1 max-w-[64px] truncate" title={name(s)}>{name(s).slice(0, 4)}</th>))}</tr>
          </thead>
          <tbody>
            {syms.map((s, i) => (
              <tr key={s}>
                <td className="label pr-1 whitespace-nowrap max-w-[90px] truncate" title={name(s)}>{name(s)}</td>
                {syms.map((t, j) => {
                  if (i === j) return <td key={t} className="min-w-[58px] h-8 rounded bg-sunken" />;
                  const v = C[i]?.[j] ?? null;
                  const on = (pair[0] === s && pair[1] === t) || (pair[0] === t && pair[1] === s);
                  return (
                    <td key={t} onClick={() => onPick(s, t)}
                      title={`${name(s)} × ${name(t)}：相关 ${fixed(v, 2)}，${opp[i]?.[j] != null ? `${Math.round(opp[i]![j]! * 100)}% 的${r === d.daily ? "交易日" : "周"}一涨一跌` : ""}`}
                      className={`min-w-[58px] px-1 h-8 rounded text-center cursor-pointer whitespace-nowrap ${on ? "ring-2 ring-ink" : ""}`}
                      style={{ background: corrFill(v), color: corrInk(v) }}>
                      {fixed(v, 2)}{v != null && Math.abs(v) > sig ? "●" : ""}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center gap-3 label">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm" style={{ background: POS }} />同涨同跌</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm" style={{ background: NEG }} />此消彼长</span>
      </div>
    </div>
  );
}

// ── one pair ─────────────────────────────────────────────────────────────────
function PairDetail({ d, r, series, basis, a, b, name, res }: {
  d: Seesaw; r: SeesawRes; series: Record<string, (number | null)[]>; basis: Basis;
  a: string; b: string; name: (s: string) => string; res: Res;
}) {
  const xa = series[a] ?? [];
  const xb = series[b] ?? [];
  const rc = corr(xa, xb);
  const rollRes = useMemo(() => r.resid ? rollingCorr(r.resid[a] ?? [], r.resid[b] ?? [], r.roll) : null, [r, a, b]);
  const rollRaw = useMemo(() => rollingCorr(r.returns[a] ?? [], r.returns[b] ?? [], r.roll), [r, a, b]);
  return (
    <div className="flex flex-col gap-3 min-w-0">
      <div className="flex flex-wrap items-baseline gap-2 text-[13px]">
        <span className="font-semibold">{name(a)} × {name(b)}</span>
        <span className="label">{basis === "resid" ? "去大盘" : "原始"}{res === "daily" ? "日" : "周"}相关</span>
        <b className="font-mono tnum" style={{ color: rc != null && rc < 0 ? NEG : POS }}>{fixed(rc, 2)}</b>
        <span className="label">β {fixed(r.betas[a], 2)} / {fixed(r.betas[b], 2)}</span>
      </div>
      <div className="grid gap-3 md:grid-cols-[220px_minmax(0,1fr)] items-start">
        <Scatter xa={xa} xb={xb} la={name(a)} lb={name(b)} />
        <Rolling dates={r.dates} roll={r.roll} res={rollRes} raw={rollRaw} sig={r.significant_at} unit={res === "daily" ? "日" : "周"} />
      </div>
      <Strip d={d} dates={r.dates} xa={r.returns[a] ?? []} xb={r.returns[b] ?? []}
        ra={series[a] ?? []} rb={series[b] ?? []} la={name(a)} lb={name(b)} res={res} basis={basis} />
    </div>
  );
}

function Scatter({ xa, xb, la, lb }: { xa: (number | null)[]; xb: (number | null)[]; la: string; lb: string }) {
  const S = 220;
  const pts = xa.map((v, i) => [v, xb[i]] as const).filter(([p, q]) => p != null && q != null) as [number, number][];
  // Each axis on its own scale: a 4%-a-day stock against a 1%-a-day one on a
  // shared scale squashes the quieter one into a line, and the quadrants —
  // the whole point — become unreadable.
  const ex = Math.max(0.3, ...pts.map(([p]) => Math.abs(p)));
  const ey = Math.max(0.3, ...pts.map(([, q]) => Math.abs(q)));
  const mx = (v: number) => ((v + ex) / (2 * ex)) * S;
  const my = (v: number) => ((v + ey) / (2 * ey)) * S;
  const opp = pts.filter(([p, q]) => opposite(p, q)).length;
  return (
    <div className="flex flex-col gap-1">
      <svg viewBox={`0 0 ${S} ${S}`} className="w-[220px] h-[220px] block" role="img"
        aria-label={`${la} 与 ${lb} 的收益散点`}>
        <rect x={0} y={0} width={S / 2} height={S / 2} fill={NEG} opacity={0.07} />
        <rect x={S / 2} y={S / 2} width={S / 2} height={S / 2} fill={NEG} opacity={0.07} />
        <line x1={S / 2} x2={S / 2} y1={0} y2={S} stroke="var(--color-line-bright)" />
        <line x1={0} x2={S} y1={S / 2} y2={S / 2} stroke="var(--color-line-bright)" />
        {pts.map(([p, q], i) => (
          <circle key={i} cx={mx(p)} cy={S - my(q)} r={2.2}
            fill={opposite(p, q) ? NEG : POS} opacity={0.55} />
        ))}
        <text x={S - 4} y={S / 2 - 4} textAnchor="end" fontSize={10} fill="var(--color-ink-mute)">{la} →</text>
        <text x={S / 2 + 4} y={10} fontSize={10} fill="var(--color-ink-mute)">↑ {lb}</text>
      </svg>
      <span className="label">橙色象限 = 一涨一跌：{pts.length ? Math.round((opp / pts.length) * 100) : 0}% 的点 · 横轴 ±{ex.toFixed(1)}%，纵轴 ±{ey.toFixed(1)}%</span>
    </div>
  );
}

function Rolling({ dates, roll, res, raw, sig, unit }: {
  dates: string[]; roll: number; res: (number | null)[] | null; raw: (number | null)[];
  sig: number | null; unit: string;
}) {
  const H = 180;
  const box = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const x = (i: number) => (dates.length < 2 ? 0 : (i / (dates.length - 1)) * VW);
  const y = (v: number) => H - ((v + 1) / 2) * H;
  const path = (vals: (number | null)[]) => {
    let out = "";
    let pen = false;
    vals.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      out += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`;
      pen = true;
    });
    return out;
  };
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <div className="flex flex-wrap items-center gap-3 text-[12px]">
        <span className="font-semibold">滚动 {roll} {unit}相关</span>
        {res && <span className="flex items-center gap-1"><span className="w-4 h-[2px]" style={{ background: "var(--color-ink)" }} />去大盘</span>}
        <span className="flex items-center gap-1"><span className="w-4 border-t-2 border-dashed border-ink-mute" />原始</span>
        {sig != null && <span className="label">灰带 = ±{(2 / Math.sqrt(roll)).toFixed(2)}（一个窗口内的噪音范围）</span>}
      </div>
      <div className="flex gap-1">
      <div className="relative w-6 shrink-0 h-[180px]">
        {[1, 0, -1].map((t) => (
          <span key={t} className="absolute right-0 -translate-y-1/2 label font-mono tnum" style={{ top: `${(y(t) / H) * 100}%` }}>
            {t > 0 ? "+1" : t}</span>
        ))}
      </div>
      <div ref={box} className="relative flex-1 min-w-0 h-[180px]"
        onMouseMove={(e) => {
          const rect = box.current?.getBoundingClientRect();
          if (!rect) return;
          setHover(Math.max(0, Math.min(dates.length - 1, Math.round(((e.clientX - rect.left) / rect.width) * (dates.length - 1)))));
        }}
        onMouseLeave={() => setHover(null)}>
        <svg viewBox={`0 0 ${VW} ${H}`} preserveAspectRatio="none" className="absolute inset-0 w-full h-full" role="img"
          aria-label="滚动相关系数">
          <rect x={0} width={VW} y={y(2 / Math.sqrt(roll))} height={y(-2 / Math.sqrt(roll)) - y(2 / Math.sqrt(roll))}
            fill="var(--color-sunken)" />
          {[-1, -0.5, 0, 0.5, 1].map((t) => (
            <line key={t} x1={0} x2={VW} y1={y(t)} y2={y(t)} stroke={t === 0 ? "var(--color-line-bright)" : "var(--color-line)"}
              strokeDasharray={t === 0 ? undefined : "2 4"} vectorEffect="non-scaling-stroke" />
          ))}
          <path d={path(raw)} fill="none" stroke="var(--color-ink-mute)" strokeWidth={1.3} strokeDasharray="5 4"
            vectorEffect="non-scaling-stroke" />
          {res && <path d={path(res)} fill="none" stroke="var(--color-ink)" strokeWidth={1.8} vectorEffect="non-scaling-stroke" />}
          {hover != null && <line x1={x(hover)} x2={x(hover)} y1={0} y2={H} stroke="var(--color-ink-mute)" vectorEffect="non-scaling-stroke" />}
        </svg>
        {hover != null && (
          <div className="absolute top-1 card px-2 py-1 text-[12px] pointer-events-none"
            style={hover > dates.length * 0.65 ? { right: `${100 - (hover / Math.max(1, dates.length - 1)) * 100 + 1}%` } : { left: `${(hover / Math.max(1, dates.length - 1)) * 100 + 1}%` }}>
            <div className="font-mono">{dates[hover]}</div>
            {res && <div>去大盘 <b className="tnum">{fixed(res[hover], 2)}</b></div>}
            <div className="text-ink-mute">原始 <span className="tnum">{fixed(raw[hover], 2)}</span></div>
          </div>
        )}
      </div>
      </div>
      <div className="flex justify-between label font-mono tnum ml-7"><span>{dates[0]}</span><span>{dates[dates.length - 1]}</span></div>
    </div>
  );
}

function Strip({ d, dates, xa, xb, ra, rb, la, lb, res, basis }: {
  d: Seesaw; dates: string[]; xa: (number | null)[]; xb: (number | null)[];
  ra: (number | null)[]; rb: (number | null)[]; la: string; lb: string; res: Res; basis: Basis;
}) {
  const N = res === "daily" ? 60 : 52;
  const from = Math.max(0, dates.length - N);
  const idx = Array.from({ length: dates.length - from }, (_, k) => from + k);
  const clamp = res === "daily" ? 4 : 8;
  const turns = idx.filter((i) => opposite(ra[i], rb[i])).length;
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-baseline gap-2 text-[12px]">
        <span className="font-semibold">最近 {idx.length} 个{res === "daily" ? "交易日" : "周"}</span>
        <span className="label">格子颜色 = 当期涨跌（{d.up_is_red ? "红涨绿跌" : "绿涨红跌"}）；橙点 = {basis === "resid" ? "去大盘后" : ""}一涨一跌（{turns}/{idx.length}）</span>
      </div>
      <div className="overflow-x-auto">
        <div className="grid gap-[2px] min-w-[640px]"
          style={{ gridTemplateColumns: `72px repeat(${idx.length}, minmax(8px, 1fr))` }}>
          {[{ l: la, v: xa }, { l: lb, v: xb }].map((row) => (
            <Row key={row.l} label={row.l}>
              {idx.map((i) => (
                <span key={i} className="h-5 rounded-[2px]"
                  title={`${dates[i]} ${row.l}：${signed(row.v[i], 2, "%")}`}
                  style={{ background: cellFill(row.v[i], clamp, d.up_is_red) }} />
              ))}
            </Row>
          ))}
          <Row label={basis === "resid" ? "去大盘反向" : "反向"}>
            {idx.map((i) => (
              <span key={i} className="h-3 flex items-center justify-center"
                title={opposite(ra[i], rb[i]) ? `${dates[i]}：${la} ${signed(ra[i], 2, "%")} / ${lb} ${signed(rb[i], 2, "%")}` : undefined}>
                {opposite(ra[i], rb[i]) && <span className="w-1.5 h-1.5 rounded-full" style={{ background: NEG }} />}
              </span>
            ))}
          </Row>
        </div>
      </div>
      <div className="flex justify-between label font-mono tnum"><span>{dates[from]}</span><span>{dates[dates.length - 1]}</span></div>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <span className="label text-[11px] leading-5 truncate" title={label}>{label}</span>
      {children}
    </>
  );
}
