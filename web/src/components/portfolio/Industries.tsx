/**
 * 🏭 行业分布 — where the money is, once tickers become sectors.
 *
 * Six names at 16% each looks diversified in the allocation bars and is not
 * diversified at all if five of them are 银行. Effective bets counts
 * positions; this counts the thing positions are standing in for, which is
 * why it gets its own verdict line rather than a footnote.
 */

import type { IndustryExposure } from "../../lib/types";
import { fixed } from "../../lib/format";

/** Distinguishable at a glance, and stable per slice index. */
const HUES = [188, 262, 32, 150, 340, 210, 96, 12, 280, 52];
const colour = (i: number) => `hsl(${HUES[i % HUES.length]} 65% 55%)`;

const TONE: Record<string, string> = {
  good: "text-up border-up/40 bg-up/5",
  warn: "text-brand-ink border-brand-ink/40 bg-brand-ink/5",
  bad: "text-up border-up/50 bg-up/10",
};

const R = 52;
const STROKE = 22;
const C = 2 * Math.PI * R;

export function Industries({ d }: { d: IndustryExposure }) {
  const slices = d.by_industry;
  if (slices.length === 0) {
    return <p className="label py-4 text-center">没有持仓</p>;
  }

  let acc = 0;
  const ring = slices.map((s, i) => {
    const frac = s.weight_pct / 100;
    const seg = { s, i, offset: acc, frac };
    acc += frac;
    return seg;
  });

  return (
    <div className="flex flex-col gap-2">
      <div className={`rounded-lg px-2.5 py-1.5 text-[12.5px] border ${TONE[d.tone] ?? ""}`}>
        {d.note}
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <svg viewBox="0 0 140 140" className="w-[140px] h-[140px] shrink-0 -rotate-90"
          role="img" aria-label="行业权重分布">
          {ring.map(({ s, i, offset, frac }) => (
            <circle key={s.industry} cx={70} cy={70} r={R} fill="none"
              stroke={colour(i)} strokeWidth={STROKE}
              strokeDasharray={`${frac * C} ${C}`}
              strokeDashoffset={-offset * C}>
              <title>{`${s.industry} · ${fixed(s.weight_pct, 1)}% · ${s.count} 只`}</title>
            </circle>
          ))}
        </svg>

        <div className="flex-1 min-w-[240px] overflow-x-auto">
          <table className="w-full text-[12px] border-collapse">
            <thead>
              <tr className="text-ink-mute">
                <th className="text-left font-normal pb-1 pr-2">行业</th>
                <th className="text-right font-normal pb-1 px-2">权重</th>
                <th className="text-right font-normal pb-1 px-2">只数</th>
                <th className="text-left font-normal pb-1 pl-2">成分股</th>
              </tr>
            </thead>
            <tbody>
              {slices.map((s, i) => (
                <tr key={s.industry} className="border-t border-line">
                  <td className="py-1 pr-2">
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm shrink-0"
                        style={{ background: colour(i) }} />
                      {s.industry}
                    </span>
                  </td>
                  <td className="py-1 px-2 text-right tnum font-medium">
                    {fixed(s.weight_pct, 1)}%
                  </td>
                  <td className="py-1 px-2 text-right tnum text-ink-mute">{s.count}</td>
                  <td className="py-1 pl-2 text-ink-dim truncate max-w-[220px]">
                    {s.holdings.slice(0, 3).join("、")}
                    {s.holdings.length > 3 && ` +${s.holdings.length - 3}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
