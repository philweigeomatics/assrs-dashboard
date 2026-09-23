/**
 * Portfolio, equal weight and the index on one rebased axis.
 *
 * Three lines rather than one, because the portfolio's own line is
 * unfalsifiable on its own: it was fitted to this history. What the reader
 * needs is the distance between it and the two things it has to beat.
 */

import type { PortfolioBuild } from "../../lib/types";
import { signed } from "../../lib/format";

const VW = 1000;
const VH = 190;

export function Curves({ d }: { d: PortfolioBuild }) {
  const series = [
    { key: "opt", label: d.mode === "target" ? "目标组合" : "最大夏普", values: d.curve, colour: "var(--color-cyan)" },
    { key: "eq", label: "等权重", values: d.equal_curve, colour: "#a855f7" },
    ...(d.benchmark
      ? [{
          key: "bm", label: d.benchmark.label,
          values: d.benchmark.curve.map((v) => (v == null ? NaN : v)),
          colour: "var(--color-ink-mute)",
        }]
      : []),
  ];

  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series) {
    for (const v of s.values) {
      if (!Number.isFinite(v)) continue;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  if (!Number.isFinite(lo)) { lo = 0; hi = 0; }
  const pad = Math.max(1, (hi - lo) * 0.08);
  const top = hi + pad;
  const bottom = lo - pad;

  const n = d.dates.length;
  const x = (i: number) => (n < 2 ? 0 : (i / (n - 1)) * VW);
  const y = (v: number) => VH - ((v - bottom) / (top - bottom)) * VH;
  const path = (values: number[]) => {
    let out = "";
    let pen = false;
    values.forEach((v, i) => {
      if (!Number.isFinite(v)) { pen = false; return; }
      out += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`;
      pen = true;
    });
    return out;
  };

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px]">
        {series.map((s) => {
          const last = [...s.values].reverse().find((v) => Number.isFinite(v));
          return (
            <span key={s.key} className="flex items-center gap-1.5">
              <span className="w-4 h-[3px] rounded-full" style={{ background: s.colour }} />
              <span>{s.label}</span>
              <span className="tnum font-medium">{signed(last ?? null, 1, "%")}</span>
            </span>
          );
        })}
      </div>
      <svg viewBox={`0 0 ${VW} ${VH}`} preserveAspectRatio="none"
        className="w-full h-[190px] block" role="img"
        aria-label="组合、等权与基准的累计收益对比">
        {bottom < 0 && top > 0 && (
          <line x1={0} x2={VW} y1={y(0)} y2={y(0)} stroke="var(--color-line-bright)"
            vectorEffect="non-scaling-stroke" />
        )}
        {series.map((s) => (
          <path key={s.key} d={path(s.values)} fill="none" stroke={s.colour}
            strokeWidth={s.key === "opt" ? 1.8 : 1.3}
            vectorEffect="non-scaling-stroke" />
        ))}
      </svg>
      <div className="flex justify-between label font-mono tnum">
        <span>{d.dates[0]}</span>
        <span>{d.dates[n - 1]}</span>
      </div>
    </div>
  );
}
