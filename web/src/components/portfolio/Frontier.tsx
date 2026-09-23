/**
 * The efficient frontier, with this portfolio and equal weight placed on it.
 *
 * The curve shows the shape of the trade-off. The two dots are what make it
 * useful: they say where the chosen allocation actually sits, which is
 * usually less impressive than the curve above it suggests.
 */

import type { PortfolioBuild } from "../../lib/types";
import { fixed } from "../../lib/format";

const VW = 420;
const VH = 220;
const PAD = 30;

export function Frontier({ d }: { d: PortfolioBuild }) {
  const pts = d.frontier;
  if (pts.length < 2) {
    return <p className="label py-6 text-center">这组股票画不出有效前沿</p>;
  }

  const mine = { vol: d.stats.ann_vol_pct, ret: d.stats.ann_return_pct };
  const eq = { vol: d.equal_stats.ann_vol_pct, ret: d.equal_stats.ann_return_pct };

  const vols = [...pts.map((p) => p.vol_pct), mine.vol ?? 0, eq.vol ?? 0];
  const rets = [...pts.map((p) => p.ret_pct), mine.ret ?? 0, eq.ret ?? 0];
  const vLo = Math.min(...vols);
  const vHi = Math.max(...vols);
  const rLo = Math.min(...rets);
  const rHi = Math.max(...rets);
  const x = (v: number) => PAD + ((v - vLo) / (vHi - vLo || 1)) * (VW - PAD * 1.5);
  const y = (r: number) => VH - PAD - ((r - rLo) / (rHi - rLo || 1)) * (VH - PAD * 1.7);

  const line = pts.map((p, i) =>
    `${i ? "L" : "M"}${x(p.vol_pct).toFixed(1)} ${y(p.ret_pct).toFixed(1)}`).join("");

  const dot = (p: { vol: number | null; ret: number | null }, colour: string,
               label: string) =>
    p.vol == null || p.ret == null ? null : (
      <g>
        <circle cx={x(p.vol)} cy={y(p.ret)} r={5} fill={colour}
          stroke="var(--color-panel)" strokeWidth={2} />
        <title>{`${label} · 波动 ${fixed(p.vol, 1)}% · 年化 ${fixed(p.ret, 1)}%`}</title>
      </g>
    );

  return (
    <div className="flex flex-col gap-1.5">
      <svg viewBox={`0 0 ${VW} ${VH}`} className="w-full block"
        style={{ maxHeight: VH }} role="img" aria-label="有效前沿">
        <line x1={PAD} y1={VH - PAD} x2={VW - 6} y2={VH - PAD}
          stroke="var(--color-line)" vectorEffect="non-scaling-stroke" />
        <line x1={PAD} y1={6} x2={PAD} y2={VH - PAD}
          stroke="var(--color-line)" vectorEffect="non-scaling-stroke" />
        <path d={line} fill="none" stroke="var(--color-line-bright)" strokeWidth={2}
          vectorEffect="non-scaling-stroke" />
        {dot(eq, "#a855f7", "等权重")}
        {dot(mine, "var(--color-cyan)", d.method_label)}
        <text x={PAD} y={VH - 10} fill="currentColor"
          className="text-ink-mute" style={{ fontSize: 10 }}>波动 →</text>
        <text x={4} y={14} fill="currentColor"
          className="text-ink-mute" style={{ fontSize: 10 }}>年化 ↑</text>
      </svg>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11.5px]">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full inline-block"
            style={{ background: "var(--color-cyan)" }} />{d.method_label}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full inline-block"
            style={{ background: "#a855f7" }} />等权重
        </span>
        <span className="text-ink-mute">灰线＝历史上每个风险水平能达到的最好收益</span>
      </div>
    </div>
  );
}
