/**
 * The efficient frontier — and the control that picks the weights.
 *
 * This is the part that makes the curve a menu rather than an illustration.
 * Clicking a point re-solves for the minimum-variance weights that reach
 * that return, which is what `target_return` does in the optimiser and what
 * the Streamlit page's target-return mode was for.
 *
 * The cyan dot is where the current allocation sits, computed from the same
 * annualised moments as the curve so it lands ON it rather than near it.
 * Equal weight is drawn too, and it is usually below the line — which is the
 * honest picture of what the optimisation is worth.
 */

import type { PortfolioBuild } from "../../lib/types";
import { fixed } from "../../lib/format";

const VW = 440;
const VH = 250;
const PAD = 34;

export function Frontier({ d, onPick, busy }: {
  d: PortfolioBuild;
  /** Re-solve at this annualised return (as a fraction, as the API wants). */
  onPick: (target: number | null) => void;
  busy: boolean;
}) {
  const pts = d.frontier;
  if (pts.length < 2) {
    return <p className="label py-6 text-center">这组股票画不出有效前沿</p>;
  }

  const mine = { vol: d.opt.ann_vol_pct, ret: d.opt.ann_return_pct };
  const eq = { vol: d.equal_stats.ann_vol_pct, ret: d.equal_stats.ann_return_pct };

  const vols = [...pts.map((p) => p.vol_pct), mine.vol, eq.vol ?? mine.vol];
  const rets = [...pts.map((p) => p.ret_pct), mine.ret, eq.ret ?? mine.ret];
  const vLo = Math.min(...vols);
  const vHi = Math.max(...vols);
  const rLo = Math.min(...rets);
  const rHi = Math.max(...rets);
  const x = (v: number) => PAD + ((v - vLo) / (vHi - vLo || 1)) * (VW - PAD * 1.5);
  const y = (r: number) => VH - PAD - ((r - rLo) / (rHi - rLo || 1)) * (VH - PAD * 1.8);

  const line = pts.map((p, i) =>
    `${i ? "L" : "M"}${x(p.vol_pct).toFixed(1)} ${y(p.ret_pct).toFixed(1)}`).join("");

  return (
    <div className="flex flex-col gap-2">
      <svg viewBox={`0 0 ${VW} ${VH}`} className="w-full block"
        style={{ maxHeight: VH }} role="img" aria-label="有效前沿，可点选目标收益">
        <line x1={PAD} y1={VH - PAD} x2={VW - 6} y2={VH - PAD}
          stroke="var(--color-line)" vectorEffect="non-scaling-stroke" />
        <line x1={PAD} y1={6} x2={PAD} y2={VH - PAD}
          stroke="var(--color-line)" vectorEffect="non-scaling-stroke" />
        <path d={line} fill="none" stroke="var(--color-line-bright)" strokeWidth={2}
          vectorEffect="non-scaling-stroke" />

        {/* One click target per frontier point. */}
        {pts.map((p, i) => {
          const here = d.mode === "target"
            && Math.abs((d.target_return_pct ?? -999) - p.ret_pct) < 0.05;
          return (
            <circle key={i} cx={x(p.vol_pct)} cy={y(p.ret_pct)}
              r={here ? 6 : 4}
              fill={here ? "var(--color-cyan)" : "var(--color-panel)"}
              stroke={here ? "var(--color-panel)" : "var(--color-line-bright)"}
              strokeWidth={2}
              className={busy ? "" : "cursor-pointer"}
              onClick={() => !busy && onPick(p.target)}>
              <title>{`点此按 年化 ${fixed(p.ret_pct, 1)}% 重新求解`
                + ` · 波动 ${fixed(p.vol_pct, 1)}%`}</title>
            </circle>
          );
        })}

        {eq.vol != null && eq.ret != null && (
          <g>
            <circle cx={x(eq.vol)} cy={y(eq.ret)} r={5} fill="#a855f7"
              stroke="var(--color-panel)" strokeWidth={2} />
            <title>{`等权重 · 波动 ${fixed(eq.vol, 1)}% · 年化 ${fixed(eq.ret, 1)}%`}</title>
          </g>
        )}
        {d.mode === "max_sharpe" && (
          <g>
            <circle cx={x(mine.vol)} cy={y(mine.ret)} r={6}
              fill="var(--color-cyan)" stroke="var(--color-panel)" strokeWidth={2} />
            <title>{`最大夏普 · 波动 ${fixed(mine.vol, 1)}% · 年化 ${fixed(mine.ret, 1)}%`}</title>
          </g>
        )}

        <text x={PAD} y={VH - 10} fill="currentColor" className="text-ink-mute"
          style={{ fontSize: 10 }}>年化波动 →</text>
        <text x={4} y={14} fill="currentColor" className="text-ink-mute"
          style={{ fontSize: 10 }}>年化收益 ↑</text>
      </svg>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px]">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full inline-block"
            style={{ background: "var(--color-cyan)" }} />
          {d.mode === "target" ? "当前目标" : "最大夏普"}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full inline-block"
            style={{ background: "#a855f7" }} />等权重
        </span>
        {d.mode === "target" && (
          <button onClick={() => onPick(null)} disabled={busy}
            className="text-cyan disabled:opacity-60">回到最大夏普</button>
        )}
      </div>
      <p className="label leading-snug">
        <b>点曲线上的任意一点</b>，就按那个年化收益重新求解 —— 得到的是达到它所需波动最小的权重。
        纵轴是历史均值，而历史均值不重复，所以这是在看取舍的形状，不是在挑未来的收益。
      </p>
    </div>
  );
}
