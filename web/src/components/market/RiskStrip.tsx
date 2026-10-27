/**
 * 🚦 市场风险 — five readings of how fragile the market is, on one row.
 *
 * Each tile is a level, where that level sits among every session since the
 * regime anchor, and sixty sessions of shape. The percentile is the point:
 * 1.5万亿 of turnover is a quiet day in this market and was a frantic one in
 * 2023, and only the comparison says which.
 *
 * The grade is RISK, not direction, so it is drawn in amber and never in
 * red/green. Under 红涨绿跌 a red tile reads "rising", and rising turnover is
 * the opposite of a warning. The sparklines are neutral for the same reason;
 * the dashed line on each is the regime median.
 */

import type { RiskBoard, RiskTile } from "../../lib/types";
import { fixed, signed, yi } from "../../lib/format";

const VW = 120;
const VH = 30;

/** Chip, gauge marker and accent, by level. */
const LEVEL = [
  { chip: "bg-panel text-ink-mute", mark: "var(--color-ink-dim)" },
  { chip: "bg-brand/15 text-brand-ink", mark: "var(--color-brand)" },
  { chip: "bg-brand text-white", mark: "var(--color-brand-ink)" },
] as const;

export function RiskStrip({ data }: { data: RiskBoard }) {
  const { have, need } = data.backfill;
  return (
    <div className="flex flex-col gap-2">
      <div className="grid gap-2 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        {data.tiles.map((t) => <Tile key={t.key} t={t} />)}
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="label">
          分位 = 自 {data.anchor} 新行情以来，当前读数在所有交易日中的位置。
          {data.flags > 0
            ? <b className="text-brand-ink"> {data.flags} 项需要留意。</b>
            : " 目前没有需要留意的项。"}
          {" "}{data.as_of} 读取。
        </span>
        {need > 0 && have < need && (
          <span className="text-[11.5px] text-brand-ink">
            涨停历史回填中 {have}/{need} 日，分位随回填逐步可用
          </span>
        )}
        {data.missing.length > 0 && (
          <span className="text-[11.5px] text-brand-ink">
            这几项没取到：{data.missing.join("、")}
          </span>
        )}
      </div>
    </div>
  );
}

function show(t: RiskTile, v: RiskTile["value"]): string {
  if (t.key === "turnover") return yi(v);
  const nd = t.key === "margin" || t.key === "valuation" ? 2 : t.key === "vol" ? 1 : 0;
  return `${fixed(v, nd)}${t.unit}`;
}

function Tile({ t }: { t: RiskTile }) {
  const lv = LEVEL[t.level] ?? LEVEL[0];
  const change = t.value != null && t.prev != null ? t.value - t.prev : null;
  const nd = t.key === "margin" || t.key === "valuation" ? 2 : t.key === "vol" ? 1 : 0;

  return (
    <div className="rounded-lg bg-sunken px-2.5 py-2 flex flex-col gap-1.5 min-w-0"
      title={`${t.note}\n最新 ${t.asof} · 新行情中位数 ${show(t, t.median)}`}>
      <div className="flex items-baseline gap-1.5">
        <span className="label truncate">{t.label}</span>
        <span className={`text-[10.5px] font-semibold px-1.5 rounded ${lv.chip}`}>
          {t.state}
        </span>
        <span className="ml-auto label font-mono tnum text-[10.5px]">{t.asof.slice(5)}</span>
      </div>

      <div className="flex items-end gap-1.5">
        <span className="text-[18px] font-semibold tnum shrink-0 leading-none">
          {show(t, t.value)}
        </span>
        {change != null && (
          <span className="text-[11.5px] tnum text-ink-mute shrink-0">
            {t.key === "turnover" ? signed(change, 0, "亿") : signed(change, nd)}
          </span>
        )}
        <Spark t={t} />
      </div>

      <Gauge t={t} mark={lv.mark} />

      {t.extra.length > 0 && (
        <div className="flex flex-wrap gap-x-3 gap-y-0.5">
          {t.extra.map((e) => (
            <span key={e.label} className="label tnum text-[11px]">
              {e.label}{" "}
              <b className="text-ink-dim font-medium">
                {e.unit === "亿" ? yi(e.value) : `${fixed(e.value, e.unit === "家" ? 0 : 1)}${e.unit}`}
              </b>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/** Where today sits in the regime, 0–100, with the quartiles marked. */
function Gauge({ t, mark }: { t: RiskTile; mark: string }) {
  if (t.pct == null) {
    return (
      <div className="label text-[11px]">
        分位 — <span className="text-ink-mute">样本 {t.n} 日，不足以分级</span>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2">
      <div className="relative flex-1 h-1.5 rounded-full bg-line"
        role="img" aria-label={`新行情分位 ${t.pct}%`}>
        {[25, 50, 75].map((q) => (
          <span key={q} className="absolute top-0 h-full w-px bg-panel" style={{ left: `${q}%` }} />
        ))}
        <span className="absolute top-1/2 h-3 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-sm"
          style={{ left: `${Math.min(100, Math.max(0, t.pct))}%`, background: mark }} />
      </div>
      <span className="label tnum text-[11px] shrink-0">
        分位 <b className="text-ink-dim">{fixed(t.pct, 0)}%</b> · {t.n} 日
      </span>
    </div>
  );
}

/** Sixty sessions of shape against the regime median. Neutral ink on purpose. */
function Spark({ t }: { t: RiskTile }) {
  const v = t.history.filter((n): n is number => n != null && Number.isFinite(n));
  if (v.length < 2) return <span className="ml-auto flex-1" />;

  const pts = t.median == null ? v : [...v, t.median];
  const lo = Math.min(...pts);
  const hi = Math.max(...pts);
  const span = hi - lo || 1;
  const x = (i: number) => (i / (v.length - 1)) * VW;
  const y = (n: number) => VH - 3 - ((n - lo) / span) * (VH - 6);
  const d = v.map((n, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(n).toFixed(1)}`).join("");
  const last = v[v.length - 1]!;

  return (
    <svg viewBox={`0 0 ${VW} ${VH}`} preserveAspectRatio="none"
      className="ml-auto flex-1 min-w-[56px] max-w-[140px] h-[30px] block"
      role="img" aria-label={`近 ${v.length} 日走势`}>
      {t.median != null && (
        <line x1={0} x2={VW} y1={y(t.median)} y2={y(t.median)}
          stroke="var(--color-line-bright)" strokeDasharray="3 3"
          vectorEffect="non-scaling-stroke" />
      )}
      <path d={d} fill="none" stroke="var(--color-ink-dim)" strokeWidth={1.3}
        vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
      <circle cx={x(v.length - 1)} cy={y(last)} r={1.8} fill="var(--color-ink-dim)" />
    </svg>
  );
}
