/**
 * 📊 权重漂移 — how the weights got to where they are.
 *
 * The drift table says where each position stands today. This says how it
 * got there, which is the part that tells you whether a holding is quietly
 * compounding into a concentration or just bouncing around. The rollup has
 * been writing these rows nightly since each fund's inception.
 *
 * Live and simulated never share a line. Simulated is a retroactive "what
 * this mandate would have done before it existed"; drawing it continuous
 * with measured history would be inventing a track record.
 */

import { useState } from "react";
import type { DriftSeries } from "../../lib/types";
import { fixed } from "../../lib/format";

const VW = 1000;
const VH = 200;
const HUES = [188, 262, 32, 150, 340, 210, 96, 12, 280, 52];
const colour = (i: number) => `hsl(${HUES[i % HUES.length]} 65% 55%)`;

export function DriftChart({ real, simulated }: {
  real: DriftSeries | null; simulated: DriftSeries | null;
}) {
  const [view, setView] = useState<"real" | "sim">(real ? "real" : "sim");
  const d = view === "real" ? real : simulated;

  if (!real && !simulated) {
    return (
      <p className="label py-4">
        还没有漂移记录。每晚的净值计算会从建仓日起逐日写入。
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        {real && (
          <Tab on={view === "real"} onClick={() => setView("real")}
            label={`📡 实际记录（${real.dates.length} 天）`} />
        )}
        {simulated && (
          <Tab on={view === "sim"} onClick={() => setView("sim")}
            label={`🔬 回溯模拟（${simulated.dates.length} 天）`} />
        )}
      </div>

      {view === "sim" && (
        <p className="label leading-snug">
          这段是把当前的目标权重放回建仓之前的行情里推演出来的，不是真实发生过的持仓。
          只能用来看这组权重对行情有多敏感。
        </p>
      )}

      {d && <Plot d={d} />}
    </div>
  );
}

function Plot({ d }: { d: DriftSeries }) {
  const n = d.dates.length;
  const all = d.holdings.flatMap((h) => h.drift_pp.filter(
    (v): v is number => v != null));
  const span = Math.max(d.alert_pp, ...all.map(Math.abs), 1) * 1.12;

  const x = (i: number) => (n < 2 ? 0 : (i / (n - 1)) * VW);
  const y = (v: number) => VH / 2 - (v / span) * (VH / 2);
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
    <div className="flex flex-col gap-1.5">
      <svg viewBox={`0 0 ${VW} ${VH}`} preserveAspectRatio="none"
        className="w-full h-[200px] block" role="img"
        aria-label="每只持仓相对目标权重的偏离">
        {[d.alert_pp, -d.alert_pp].map((v) => (
          <line key={v} x1={0} x2={VW} y1={y(v)} y2={y(v)}
            stroke="var(--color-brand-ink)" strokeOpacity={0.35}
            strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
        ))}
        <line x1={0} x2={VW} y1={y(0)} y2={y(0)}
          stroke="var(--color-line-bright)" vectorEffect="non-scaling-stroke" />
        {d.holdings.map((h, i) => (
          <path key={h.t} d={path(h.drift_pp)} fill="none" stroke={colour(i)}
            strokeWidth={1.6} vectorEffect="non-scaling-stroke" />
        ))}
      </svg>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px]">
        {d.holdings.map((h, i) => {
          const last = [...h.drift_pp].reverse().find((v) => v != null) ?? null;
          return (
            <span key={h.t} className="flex items-center gap-1.5">
              <span className="w-4 h-[3px] rounded-full"
                style={{ background: colour(i) }} />
              <span>{h.n}</span>
              <span className={`tnum font-medium ${
                last != null && Math.abs(last) >= d.alert_pp
                  ? "text-brand-ink" : "text-ink-mute"}`}>
                {last == null ? "—" : `${last > 0 ? "+" : ""}${fixed(last, 1)}pp`}
              </span>
            </span>
          );
        })}
        <span className="ml-auto label font-mono tnum">
          {d.dates[0]} → {d.dates[n - 1]}
        </span>
      </div>
      <p className="label">
        零线是目标权重。虚线是 ±{d.alert_pp}pp —— 越过它，这只已经明显偏离建仓时的意图了。
      </p>
    </div>
  );
}

function Tab({ on, onClick, label }: {
  on: boolean; onClick: () => void; label: string;
}) {
  return (
    <button onClick={onClick}
      className={`h-7 px-2.5 rounded-lg text-[12.5px] transition-colors ${
        on ? "bg-elevated text-ink font-medium" : "text-ink-mute hover:text-ink"}`}>
      {label}
    </button>
  );
}
