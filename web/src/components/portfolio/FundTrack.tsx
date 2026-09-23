/**
 * A saved portfolio's NAV against its reference index.
 *
 * The fund's own curve is not the point — every A-share book was up in a
 * bull quarter. The point is the distance between it and the index over the
 * same days, so alpha is the headline and the two lines share an axis.
 *
 * Below it, drift: a portfolio is only its target weights on the day it is
 * built, and after that the winners grow into a bigger share than intended.
 */

import type { FundDetail } from "../../lib/types";
import { fixed, signed } from "../../lib/format";

const VW = 1000;
const VH = 170;

export function FundTrack({ d, onRevalue, revaluing }: {
  d: FundDetail; onRevalue: () => void; revaluing: boolean;
}) {
  const t = d.tracking;

  if (!t.valued) {
    return (
      <div className="rounded-lg bg-sunken p-3 flex flex-col items-start gap-2">
        <p className="text-[12.5px]">
          还没有估值记录。净值由每晚的 NAV 计算写入，
          {d.inception ? `这个组合建于 ${d.inception}。` : ""}
        </p>
        <button onClick={onRevalue} disabled={revaluing}
          className="h-8 px-3 rounded-lg bg-cyan text-white text-[12.5px]
            font-semibold disabled:opacity-60">
          {revaluing ? "计算中…" : "现在算一次"}
        </button>
      </div>
    );
  }

  const bench = t.benchmark;
  const series = [
    { key: "pf", label: d.name, values: t.curve, colour: "var(--color-cyan)" },
    ...(bench ? [{ key: "bm", label: bench.label,
                   values: bench.curve.map((v) => (v == null ? NaN : v)),
                   colour: "var(--color-ink-mute)" }] : []),
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
  const n = t.dates.length;
  const x = (i: number) => (n < 2 ? 0 : (i / (n - 1)) * VW);
  const y = (v: number) => VH - ((v - bottom) / (top - bottom)) * VH;
  const path = (vals: number[]) => {
    let out = "";
    let pen = false;
    vals.forEach((v, i) => {
      if (!Number.isFinite(v)) { pen = false; return; }
      out += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`;
      pen = true;
    });
    return out;
  };

  const alphaUp = (t.alpha_pct ?? 0) >= 0;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <Stat label="净值" v={t.aum == null ? "—"
          : `¥${t.aum.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
          hint={`起始 ¥${(t.inception_aum ?? 0).toLocaleString()}`} />
        <Stat label="累计收益" v={signed(t.total_return_pct, 2, "%")}
          tone={(t.total_return_pct ?? 0) >= 0 ? "up" : "down"} />
        {bench && (
          <Stat label={bench.label} v={signed(bench.curve[n - 1] ?? null, 2, "%")} />
        )}
        {t.alpha_pct != null && (
          <Stat label="超额（alpha）" v={signed(t.alpha_pct, 2, "%")}
            tone={alphaUp ? "up" : "down"} strong
            hint="组合累计收益减去同期基准累计收益" />
        )}
        <button onClick={onRevalue} disabled={revaluing}
          className="ml-auto h-7 px-2 rounded-md bg-sunken text-[12px] disabled:opacity-60">
          {revaluing ? "计算中…" : "重算净值"}
        </button>
      </div>

      <svg viewBox={`0 0 ${VW} ${VH}`} preserveAspectRatio="none"
        className="w-full h-[170px] block" role="img"
        aria-label={`${d.name} 与基准的累计收益`}>
        {bottom < 0 && top > 0 && (
          <line x1={0} x2={VW} y1={y(0)} y2={y(0)} stroke="var(--color-line-bright)"
            vectorEffect="non-scaling-stroke" />
        )}
        {series.map((s) => (
          <path key={s.key} d={path(s.values)} fill="none" stroke={s.colour}
            strokeWidth={s.key === "pf" ? 1.8 : 1.3}
            vectorEffect="non-scaling-stroke" />
        ))}
      </svg>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11.5px]">
        {series.map((s) => (
          <span key={s.key} className="flex items-center gap-1.5">
            <span className="w-4 h-[3px] rounded-full" style={{ background: s.colour }} />
            {s.label}
          </span>
        ))}
        <span className="ml-auto label font-mono tnum">
          {t.dates[0]} → {t.dates[n - 1]}
        </span>
      </div>

      {d.drift.length > 0 && (
        <div className="overflow-x-auto mt-1">
          <table className="w-full text-[12px] border-collapse">
            <thead>
              <tr className="text-ink-mute">
                <th className="text-left font-normal pb-1 pr-3">成分股</th>
                <th className="text-right font-normal pb-1 px-2">目标权重</th>
                <th className="text-right font-normal pb-1 px-2">当前权重</th>
                <th className="text-right font-normal pb-1 pl-2"
                  title="市值涨跌把权重推离目标的幅度">偏离</th>
              </tr>
            </thead>
            <tbody>
              {d.drift.map((x) => (
                <tr key={x.t} className="border-t border-line">
                  <td className="py-1 pr-3 font-mono tnum">{x.t}</td>
                  <td className="py-1 px-2 text-right tnum text-ink-dim">
                    {fixed(x.target_pct, 1)}%
                  </td>
                  <td className="py-1 px-2 text-right tnum">{fixed(x.actual_pct, 1)}%</td>
                  <td className={`py-1 pl-2 text-right tnum ${
                    Math.abs(x.drift_pct) >= 5 ? "text-brand-ink font-medium" : "text-ink-dim"}`}>
                    {signed(x.drift_pct, 1, "pp")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="label mt-1">
            截至 {d.drift[0]?.as_of}。涨得多的那只会长成比目标更大的一块 ——
            偏离超过 5pp 标成橙色。
          </p>
        </div>
      )}
    </div>
  );
}

function Stat({ label, v, hint, tone, strong }: {
  label: string; v: string; hint?: string;
  tone?: "up" | "down"; strong?: boolean;
}) {
  return (
    <span className="flex items-baseline gap-1.5" title={hint}>
      <span className="label">{label}</span>
      <span className={`tnum ${strong ? "text-[14px] font-semibold" : "text-[13px] font-medium"} ${
        tone === "up" ? "text-up" : tone === "down" ? "text-down" : ""}`}>
        {v}
      </span>
    </span>
  );
}
