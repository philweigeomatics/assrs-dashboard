/**
 * The efficient frontier — and the control that picks the weights.
 *
 * This is the part that makes the curve a menu rather than an illustration.
 * Clicking a point re-solves for the minimum-variance weights that reach
 * that return, which is what `target_return` does in the optimiser.
 *
 * Four kinds of marker share the axes, because the curve means nothing on
 * its own:
 *   · the frontier itself, every point clickable
 *   · where the current allocation sits, from the same annualised moments
 *     as the curve so it lands ON it rather than near it
 *   · equal weight, which is the thing optimisation has to beat
 *   · each stock alone — all of them below and right of the line, and that
 *     gap IS the diversification benefit
 * Plus, when weights are being edited by hand, where those land.
 */

import type { PortfolioBuild } from "../../lib/types";
import { fixed } from "../../lib/format";

const VW = 460;
const VH = 280;
const PAD = 38;

const EQUAL = "#a855f7";
const CUSTOM = "#f59e0b";
const SINGLE = "var(--color-ink-mute)";

export type Mark = { vol_pct: number; ann_return_pct: number } | null;

export function Frontier({ d, onPick, busy, custom }: {
  d: PortfolioBuild;
  /** Re-solve at this annualised return (a fraction, as the API wants). */
  onPick: (target: number | null) => void;
  busy: boolean;
  /** Where hand-set weights land, when the editor is open. */
  custom?: Mark;
}) {
  const pts = d.frontier;
  if (pts.length < 2) {
    return <p className="label py-6 text-center">这组股票画不出有效前沿</p>;
  }

  const mine = { vol: d.opt.ann_vol_pct, ret: d.opt.ann_return_pct };
  const eq = { vol: d.equal_stats.ann_vol_pct, ret: d.equal_stats.ann_return_pct };
  const singles = d.singles ?? [];

  // The axes are set by the things you are choosing between — the frontier
  // and the portfolio markers. A single runaway stock (one A-share here at
  // 82% return on 65% volatility) would otherwise drag the scale out and
  // squash the whole curve into a corner, which is the opposite of what
  // plotting the singles was for. Singles beyond the range are pinned to the
  // edge as arrows, so they stay visible and still say where they really are.
  const vols = [...pts.map((p) => p.vol_pct), mine.vol];
  const rets = [...pts.map((p) => p.ret_pct), mine.ret];
  if (eq.vol != null) vols.push(eq.vol);
  if (eq.ret != null) rets.push(eq.ret);
  if (custom) { vols.push(custom.vol_pct); rets.push(custom.ann_return_pct); }

  const vSpan = Math.max(...vols) - Math.min(...vols);
  const rSpan = Math.max(...rets) - Math.min(...rets);
  // Room for singles, but no more than this much of the frontier's own span.
  const ROOM = 0.9;
  const vLo = Math.min(...vols, ...singles.map((s) => s.vol_pct))
    < Math.min(...vols) - vSpan * ROOM
    ? Math.min(...vols) - vSpan * ROOM
    : Math.min(...vols, ...singles.map((s) => s.vol_pct));
  const vHi = Math.max(...vols, ...singles.map((s) => s.vol_pct))
    > Math.max(...vols) + vSpan * ROOM
    ? Math.max(...vols) + vSpan * ROOM
    : Math.max(...vols, ...singles.map((s) => s.vol_pct));
  const rLo = Math.min(...rets, ...singles.map((s) => s.ret_pct))
    < Math.min(...rets) - rSpan * ROOM
    ? Math.min(...rets) - rSpan * ROOM
    : Math.min(...rets, ...singles.map((s) => s.ret_pct));
  const rHi = Math.max(...rets, ...singles.map((s) => s.ret_pct))
    > Math.max(...rets) + rSpan * ROOM
    ? Math.max(...rets) + rSpan * ROOM
    : Math.max(...rets, ...singles.map((s) => s.ret_pct));
  const x = (v: number) => PAD + ((v - vLo) / (vHi - vLo || 1)) * (VW - PAD * 1.5);
  const y = (r: number) => VH - PAD - ((r - rLo) / (rHi - rLo || 1)) * (VH - PAD * 1.8);

  const offChart = singles.filter(
    (s) => s.vol_pct > vHi || s.vol_pct < vLo
        || s.ret_pct > rHi || s.ret_pct < rLo).length;

  const line = pts.map((p, i) =>
    `${i ? "L" : "M"}${x(p.vol_pct).toFixed(1)} ${y(p.ret_pct).toFixed(1)}`).join("");

  return (
    <div className="flex flex-col gap-2">
      <svg viewBox={`0 0 ${VW} ${VH}`} className="w-full block"
        style={{ maxHeight: VH * 1.15 }} role="img"
        aria-label="有效前沿，可点选目标收益">
        {rLo < 0 && rHi > 0 && (
          <line x1={PAD} x2={VW - 6} y1={y(0)} y2={y(0)}
            stroke="var(--color-line)" strokeDasharray="3 3"
            vectorEffect="non-scaling-stroke" />
        )}
        <line x1={PAD} y1={VH - PAD} x2={VW - 6} y2={VH - PAD}
          stroke="var(--color-line)" vectorEffect="non-scaling-stroke" />
        <line x1={PAD} y1={6} x2={PAD} y2={VH - PAD}
          stroke="var(--color-line)" vectorEffect="non-scaling-stroke" />

        {/* Individual stocks first, so portfolio markers draw over them. */}
        {singles.map((s) => {
          const off = s.vol_pct > vHi || s.vol_pct < vLo
                   || s.ret_pct > rHi || s.ret_pct < rLo;
          const cx = Math.max(PAD, Math.min(VW - 8, x(s.vol_pct)));
          const cy = Math.max(8, Math.min(VH - PAD, y(s.ret_pct)));
          const label = `${s.n} ${s.t} · 单独持有 · 年化 ${fixed(s.ret_pct, 1)}%`
            + ` · 波动 ${fixed(s.vol_pct, 1)}%${off ? "（超出图外）" : ""}`;
          return off ? (
            <path key={s.t}
              d={`M${cx} ${cy - 5}L${cx + 4.5} ${cy + 3.5}L${cx - 4.5} ${cy + 3.5}Z`}
              fill="none" stroke={SINGLE} strokeWidth={1.4}
              vectorEffect="non-scaling-stroke">
              <title>{label}</title>
            </path>
          ) : (
            <circle key={s.t} cx={cx} cy={cy} r={3.5}
              fill="none" stroke={SINGLE} strokeWidth={1.4}
              vectorEffect="non-scaling-stroke">
              <title>{label}</title>
            </circle>
          );
        })}

        <path d={line} fill="none" stroke="var(--color-line-bright)" strokeWidth={2}
          vectorEffect="non-scaling-stroke" />

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
          <circle cx={x(eq.vol)} cy={y(eq.ret)} r={5} fill={EQUAL}
            stroke="var(--color-panel)" strokeWidth={2}>
            <title>{`等权重 · 年化 ${fixed(eq.ret, 1)}% · 波动 ${fixed(eq.vol, 1)}%`}</title>
          </circle>
        )}
        {d.mode === "max_sharpe" && (
          <circle cx={x(mine.vol)} cy={y(mine.ret)} r={6}
            fill="var(--color-cyan)" stroke="var(--color-panel)" strokeWidth={2}>
            <title>{`最大夏普 · 年化 ${fixed(mine.ret, 1)}% · 波动 ${fixed(mine.vol, 1)}%`}</title>
          </circle>
        )}
        {custom && (
          <g>
            <circle cx={x(custom.vol_pct)} cy={y(custom.ann_return_pct)} r={7}
              fill={CUSTOM} stroke="var(--color-panel)" strokeWidth={2}>
              <title>{`你的权重 · 年化 ${fixed(custom.ann_return_pct, 1)}%`
                + ` · 波动 ${fixed(custom.vol_pct, 1)}%`}</title>
            </circle>
          </g>
        )}

        <text x={PAD} y={VH - 10} fill="currentColor" className="text-ink-mute"
          style={{ fontSize: 10 }}>年化波动 →</text>
        <text x={4} y={14} fill="currentColor" className="text-ink-mute"
          style={{ fontSize: 10 }}>年化收益 ↑</text>
      </svg>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px]">
        <Key colour="var(--color-line-bright)" shape="line" label="有效前沿（可点选）" />
        <Key colour="var(--color-cyan)"
          label={d.mode === "target" ? "当前目标" : "最大夏普"} />
        <Key colour={EQUAL} label="等权重" />
        {custom && <Key colour={CUSTOM} label="你的权重" />}
        <Key colour={SINGLE} shape="ring" label="单只股票" />
        {offChart > 0 && (
          <span className="text-ink-mute">△ {offChart} 只在图外（悬停看数值）</span>
        )}
        {d.mode === "target" && (
          <button onClick={() => onPick(null)} disabled={busy}
            className="text-cyan disabled:opacity-60 ml-auto">回到最大夏普</button>
        )}
      </div>
      <p className="label leading-snug">
        <b>点曲线上的任意一点</b>，就按那个年化收益重新求解 —— 得到的是达到它所需波动最小的权重。
        每只股票单独持有都落在曲线的右下方，中间那段距离就是分散化换来的。
        纵轴是历史均值，而历史均值不重复。
      </p>
    </div>
  );
}

function Key({ colour, label, shape = "dot" }: {
  colour: string; label: string; shape?: "dot" | "ring" | "line";
}) {
  return (
    <span className="flex items-center gap-1.5">
      {shape === "line" ? (
        <span className="w-4 h-[3px] rounded-full" style={{ background: colour }} />
      ) : (
        <span className="w-3 h-3 rounded-full inline-block"
          style={shape === "ring"
            ? { border: `1.5px solid ${colour}` }
            : { background: colour }} />
      )}
      {label}
    </span>
  );
}
