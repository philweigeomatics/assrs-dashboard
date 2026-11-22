/**
 * ⚖️ 股债跷跷板 — stocks against bonds, total return on both sides. Where
 * money has been leaning, since when, how far the next turn is, and whether
 * the two are moving against each other at all. See stock_bond.py.
 *
 * Colours follow the market's own convention for "up": the ratio rising is
 * stocks winning, so it takes the up colour (red for A-shares, green for the
 * US) and bonds winning takes the down colour.
 */

import { useMemo, useState } from "react";
import type { StockBond, StockBondLeg, StockBondSide } from "../../lib/types";

const pct = (v: number | null | undefined, nd = 1) =>
  v == null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(nd)}%`;
const frac = (v: number | null | undefined, nd = 1) =>
  v == null ? "—" : `${v > 0 ? "+" : ""}${(v * 100).toFixed(nd)}%`;

const SIDE: Record<StockBondSide, { label: string; lean: string }> = {
  equity: { label: "股强", lean: "资金偏向股票" },
  bond: { label: "债强", lean: "资金偏向债券" },
};

export function StockBondPanel({ d }: { d: StockBond }) {
  const upVar = d.up_is_red ? "var(--color-up)" : "var(--color-down)";
  const downVar = d.up_is_red ? "var(--color-down)" : "var(--color-up)";
  const sideVar = (s: StockBondSide) => (s === "equity" ? upVar : downVar);
  const n = d.now;
  const turnTo: StockBondSide = n.side === "equity" ? "bond" : "equity";
  const corrWord = n.corr == null ? null
    : n.corr <= -0.2 ? "此消彼长，跷跷板在起作用 —— 股债比的变化里有资金在两边之间移动"
      : n.corr >= 0.2 ? "同涨同跌多于此消彼长 —— 股票和债券被同一件事推动（多半是利率），股债比的变化不全是资金换边"
        : "关系很弱 —— 股票和债券各走各的";

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <span className="px-2.5 h-8 rounded-lg flex items-center text-[15px] font-semibold text-white"
          style={{ background: sideVar(n.side) }}>
          {SIDE[n.side].lean}
        </span>
        <span className="text-[13.5px]">
          自 <b className="tnum">{n.since}</b> 起，{n.sessions} 个交易日
          {n.confirmed && <span className="text-ink-mute">（{n.confirmed} 才确认）</span>}
        </span>
      </div>

      <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="这一段的股债比" value={pct(n.ratio_pct)} hint={`${d.equity.label} ${pct(n.equity_pct)} · ${d.bond.label} ${pct(n.bond_pct)}`} />
        <Stat label="近 20 个交易日" value={pct(n.recent_pct, 2)} hint="股债比的变化" />
        <Stat label={`确认转向${SIDE[turnTo].label}还差`} value={frac(n.need)}
          hint={`${SIDE[n.side].label}这段的${n.side === "equity" ? "高点" : "低点"}在 ${n.extreme_date}，现在离它 ${frac(n.from_extreme)}；反向走 ${(d.pct * 100).toFixed(0)}% 才算转向`} />
        <Stat label="近 60 日股债相关" value={n.corr == null ? "—" : `${n.corr > 0 ? "+" : ""}${n.corr.toFixed(2)}`}
          hint={n.corr_pct == null ? "" : `十年里第 ${(n.corr_pct * 100).toFixed(0)} 百分位`} />
      </div>
      {corrWord && <p className="text-[12.5px] leading-snug text-ink-dim">相关性：{corrWord}。</p>}

      <Chart d={d} sideVar={sideVar} />

      <p className="label leading-snug">
        股债比 = {d.equity.label} ÷ {d.bond.label}，两边都是全收益（含分红、票息），{d.from} 记为 100。
        往上是股票跑赢债券，往下是债券跑赢股票。股债比从一段的高点回落 {(d.pct * 100).toFixed(0)}%
        （或从低点反弹 {(d.pct * 100).toFixed(0)}%）才算转向 —— 所以每个转折都标了两天：
        <b>转折本身</b>（圆点，事后才知道）和<b>能确认的那天</b>（竖线，当时就能看到）。
        过去股票占优的阶段中位 {d.typical.equity ?? "—"} 个交易日，债券占优 {d.typical.bond ?? "—"} 个交易日。
      </p>

      <Legs d={d} sideVar={sideVar} />
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg bg-sunken px-2.5 py-2 flex flex-col gap-0.5 min-w-0">
      <span className="label">{label}</span>
      <b className="text-[16px] tnum">{value}</b>
      {hint && <span className="text-[11.5px] text-ink-mute leading-snug">{hint}</span>}
    </div>
  );
}

const W = 960, H = 260, CH = 70, P = { l: 44, r: 10, t: 10, b: 20 };

function Chart({ d, sideVar }: { d: StockBond; sideVar: (s: StockBondSide) => string }) {
  const c = d.chart;
  const n = c.dates.length;
  const [hover, setHover] = useState<number | null>(null);
  const x = (i: number) => P.l + (i / Math.max(1, n - 1)) * (W - P.l - P.r);
  const idxOf = (date: string) => {
    let i = c.dates.findIndex((t) => t >= date);
    if (i < 0) i = n - 1;
    return i;
  };

  const g = useMemo(() => {
    const vals = c.ratio.filter((v): v is number => v != null);
    const lo = Math.log(Math.min(...vals)), hi = Math.log(Math.max(...vals));
    const pad = (hi - lo) * 0.06 || 0.05;
    const y = (v: number) => P.t + (1 - (Math.log(v) - (lo - pad)) / (hi - lo + 2 * pad)) * (H - P.t - P.b);
    let line = "";
    c.ratio.forEach((v, i) => { if (v != null) line += `${line ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`; });
    const yc = (v: number) => 6 + (1 - (v + 1) / 2) * (CH - 22);
    let cl = "";
    c.corr.forEach((v, i) => { if (v != null) cl += `${cl && c.corr[i - 1] != null ? "L" : "M"}${x(i).toFixed(1)},${yc(v).toFixed(1)}`; });
    // Ticks at round ratio levels inside the range.
    const ticks: number[] = [];
    for (const t of [25, 50, 75, 100, 125, 150, 200, 250, 300, 400]) {
      if (Math.log(t) >= lo - pad && Math.log(t) <= hi + pad) ticks.push(t);
    }
    return { y, line, yc, cl, ticks };
  }, [c]);

  const years = useMemo(() => {
    const out: { i: number; y: string }[] = [];
    c.dates.forEach((t, i) => {
      if (i === 0 || t.slice(0, 4) !== (c.dates[i - 1] ?? "").slice(0, 4)) out.push({ i, y: t.slice(0, 4) });
    });
    if (out.length > 1 && (out[1]?.i ?? 0) - (out[0]?.i ?? 0) < n / 20) out.shift();
    return out;
  }, [c.dates, n]);

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.round(((px - P.l) / (W - P.l - P.r)) * (n - 1));
    setHover(i >= 0 && i < n ? i : null);
  };
  const h = hover;
  const legAt = (date: string): StockBondLeg | undefined =>
    d.legs.find((l) => l.from <= date && date <= l.to);

  return (
    <div className="flex flex-col gap-1">
      <div className="text-[12px] tnum min-h-[18px] text-ink-dim">
        {h != null ? (
          <>
            <b>{c.dates[h]}</b> · 股债比 <b>{c.ratio[h]?.toFixed(1)}</b>
            {" "}· {d.equity.label} {c.equity[h]?.toFixed(1)} · {d.bond.label} {c.bond[h]?.toFixed(1)}
            {c.corr[h] != null && <> · 60日相关 {c.corr[h]!.toFixed(2)}</>}
            {(() => { const l = legAt(c.dates[h]!); return l ? <> · {SIDE[l.side].label}段</> : null; })()}
          </>
        ) : <span className="label">鼠标移到图上看某一周的数值</span>}
      </div>
      <svg viewBox={`0 0 ${W} ${H + CH}`} className="w-full h-auto select-none" role="img"
        aria-label="股债比与股强、债强阶段" onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {d.legs.map((l) => {
          const a = idxOf(l.from), b = idxOf(l.to);
          return <rect key={l.from} x={x(a)} y={P.t} width={Math.max(1, x(b) - x(a))}
            height={H - P.t - P.b} fill={sideVar(l.side)} opacity={0.08} />;
        })}
        {g.ticks.map((t) => (
          <g key={t}>
            <line x1={P.l} x2={W - P.r} y1={g.y(t)} y2={g.y(t)} stroke="var(--color-line)" strokeWidth={1} />
            <text x={P.l - 5} y={g.y(t) + 3} fontSize={10} textAnchor="end" fill="var(--color-ink-mute)">{t}</text>
          </g>
        ))}
        {/* Confirmation days: when the turn could first be known. */}
        {d.legs.map((l) => l.start_confirmed && (
          <line key={`c${l.from}`} x1={x(idxOf(l.start_confirmed))} x2={x(idxOf(l.start_confirmed))}
            y1={P.t} y2={H - P.b} stroke={sideVar(l.side)} strokeWidth={1} strokeDasharray="2 3" opacity={0.7} />
        ))}
        <path d={g.line} fill="none" stroke="var(--color-ink)" strokeWidth={1.5} />
        {/* The turns themselves: each leg's first day. */}
        {d.legs.slice(1).map((l) => {
          const i = idxOf(l.from);
          const v = c.ratio[i];
          return v == null ? null : (
            <circle key={`t${l.from}`} cx={x(i)} cy={g.y(v)} r={3.5} fill={sideVar(l.side)}
              stroke="var(--color-panel)" strokeWidth={1.2} />
          );
        })}
        {years.map((t) => (
          <text key={t.y} x={x(t.i)} y={H - 6} fontSize={10} fill="var(--color-ink-mute)">{t.y}</text>
        ))}

        <g transform={`translate(0 ${H})`}>
          <text x={P.l} y={4} fontSize={10} fill="var(--color-ink-mute)">股债 60 日相关（0 以下 = 此消彼长）</text>
          {[1, 0, -1].map((v) => (
            <g key={v}>
              <line x1={P.l} x2={W - P.r} y1={g.yc(v)} y2={g.yc(v)} stroke="var(--color-line)"
                strokeDasharray={v === 0 ? undefined : "2 3"} />
              <text x={P.l - 5} y={g.yc(v) + 3} fontSize={10} textAnchor="end" fill="var(--color-ink-mute)">
                {v > 0 ? "+1" : v}
              </text>
            </g>
          ))}
          <path d={g.cl} fill="none" stroke="var(--color-violet)" strokeWidth={1.2} />
        </g>
        {h != null && (
          <line x1={x(h)} x2={x(h)} y1={P.t} y2={H + CH - 4} stroke="var(--color-ink-mute)" strokeWidth={1} />
        )}
      </svg>
    </div>
  );
}

function Legs({ d, sideVar }: { d: StockBond; sideVar: (s: StockBondSide) => string }) {
  const rows = [...d.legs].reverse();
  return (
    <details>
      <summary className="text-[13px] font-semibold cursor-pointer">
        历次阶段 <span className="label font-normal">{d.legs.length} 段，新的在上</span>
      </summary>
      <div className="overflow-x-auto mt-1.5">
        <table className="w-full text-[12.5px] border-collapse min-w-[640px]">
          <thead>
            <tr className="text-ink-mute">
              <th className="text-left font-normal pb-1 pr-2">阶段</th>
              <th className="text-left font-normal pb-1 px-2">起 → 止</th>
              <th className="text-right font-normal pb-1 px-2">交易日</th>
              <th className="text-right font-normal pb-1 px-2">股债比</th>
              <th className="text-right font-normal pb-1 px-2">{d.equity.label}</th>
              <th className="text-right font-normal pb-1 px-2">{d.bond.label}</th>
              <th className="text-left font-normal pb-1 pl-2">开始被确认</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((l) => (
              <tr key={l.from} className="border-t border-line">
                <td className="py-1.5 pr-2 whitespace-nowrap">
                  <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ background: sideVar(l.side) }} />
                  {SIDE[l.side].label}{l.open && <span className="label ml-1">进行中</span>}
                </td>
                <td className="py-1.5 px-2 tnum whitespace-nowrap">{l.from} → {l.open ? "今天" : l.to}</td>
                <td className="py-1.5 px-2 text-right tnum">{l.sessions}</td>
                <td className="py-1.5 px-2 text-right tnum">{pct(l.ratio_pct)}</td>
                <td className="py-1.5 px-2 text-right tnum">{pct(l.equity_pct)}</td>
                <td className="py-1.5 px-2 text-right tnum">{pct(l.bond_pct)}</td>
                <td className="py-1.5 pl-2 tnum text-ink-mute whitespace-nowrap">{l.start_confirmed ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
