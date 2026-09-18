/**
 * 板块轮动 — the relative rotation map.
 *
 * Replaces the old pairwise-correlation panel. Correlation is symmetric, so it
 * could report "rotation is happening" and never which way; this plots each
 * sector's strength against the market on one axis and the CHANGE in that
 * strength on the other, which carries direction by construction. See
 * sector_rotation.py for the maths and for why the axes are standardised
 * across sectors rather than against each sector's own past.
 *
 * Sectors travel clockwise — 改善 → 领先 → 走弱 → 落后 — so the tail behind
 * each dot is not decoration: where a sector came from is most of what tells
 * you whether it is arriving or leaving. The two lists on the right are the
 * answer to the question the panel exists for, and the table under them is
 * what each quadrant has actually been worth in this market's own history.
 */

import { useMemo, useState } from "react";
import type { Quadrant, Rotation, RotationCall, RotationSector } from "../../lib/types";
import { fixed, signed } from "../../lib/format";
import { useSize } from "../../lib/useSize";

const PAD = { t: 16, r: 16, b: 28, l: 40 };
const ORDER: Quadrant[] = ["improving", "leading", "weakening", "lagging"];

/**
 * How many sectors get a trail by default.
 *
 * Drawing all 26 at once is the obvious thing and it is unreadable: 26 tails
 * of 8 points all cross in the middle, and 26 labels land on top of one
 * another. So the default is the answer to the question the panel asks — what
 * is arriving, what is leaving — with every other sector still present as a
 * plain dot, and a switch for anyone who wants the whole picture.
 */
const FOCUS_PER_SIDE = 5;

type Scope = "calls" | "all" | "none";

const SCOPES: { id: Scope; label: string }[] = [
  { id: "calls", label: "轮动中" },
  { id: "all", label: "全部" },
  { id: "none", label: "只看位置" },
];

export function RotationMap({ data, freq, onFreq }: {
  data: Rotation; freq: "w" | "d"; onFreq: (f: "w" | "d") => void;
}) {
  const [hover, setHover] = useState<string | null>(null);
  const [scope, setScope] = useState<Scope>("calls");

  const focus = useMemo(() => {
    if (scope === "all") return new Set(data.sectors.map((s) => s.name));
    if (scope === "none") return new Set<string>();
    return new Set([
      ...data.calls.into.slice(0, FOCUS_PER_SIDE),
      ...data.calls.outof.slice(0, FOCUS_PER_SIDE),
    ].map((c) => c.name));
  }, [data, scope]);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="label">
          基准 {data.benchmark} · {data.asof} · 轨迹 {data.params.tail}
          {freq === "w" ? " 周" : " 日"}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <div className="flex rounded-lg bg-sunken p-0.5" title="画出轨迹的板块">
            {SCOPES.map((o) => (
              <button key={o.id} onClick={() => setScope(o.id)}
                className={`px-2 h-6 rounded-md text-[12px] font-medium transition-colors ${
                  scope === o.id ? "bg-panel text-ink shadow-sm" : "text-ink-mute"}`}>
                {o.label}
              </button>
            ))}
          </div>
          <div className="flex rounded-lg bg-sunken p-0.5">
            {(["w", "d"] as const).map((f) => (
              <button key={f} onClick={() => onFreq(f)}
                className={`px-2.5 h-6 rounded-md text-[12px] font-medium transition-colors ${
                  freq === f ? "bg-panel text-ink shadow-sm" : "text-ink-mute"}`}>
                {f === "w" ? "周线" : "日线"}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_300px] gap-3">
        <Plot data={data} hover={hover} onHover={setHover} focus={focus} />
        <div className="flex flex-col gap-3 min-w-0">
          <Calls title="资金流入中" hint="仍落后于大盘，但相对动能已经转正 — 轮入候选"
            rows={data.calls.into} tone="in" hover={hover} onHover={setHover} />
          <Calls title="资金流出中" hint="仍跑赢大盘，但相对动能已经消退 — 减仓预警"
            rows={data.calls.outof} tone="out" hover={hover} onHover={setHover} />
          <Edge data={data} freq={freq} />
        </div>
      </div>
    </div>
  );
}

function Plot({ data, hover, onHover, focus }: {
  data: Rotation; hover: string | null; onHover: (n: string | null) => void;
  focus: Set<string>;
}) {
  const [ref, size] = useSize<HTMLDivElement>();
  const w = Math.max(size.w, 280);
  const h = Math.max(Math.round(w * 0.62), 300);

  const dom = useMemo(() => {
    const all = data.sectors.flatMap((s) => s.tail);
    const spread = Math.max(
      ...all.map((p) => Math.max(Math.abs(p.ratio - 100), Math.abs(p.mom - 100))), 1.5);
    // Square and centred on (100, 100): the quadrants are only comparable if
    // the two axes share a scale, and the crossing point must sit dead centre
    // or "just inside 领先" looks like the middle of it.
    return spread * 1.15;
  }, [data]);

  const px = (v: number) => PAD.l + ((v - 100 + dom) / (2 * dom)) * (w - PAD.l - PAD.r);
  const py = (v: number) => PAD.t + (1 - (v - 100 + dom) / (2 * dom)) * (h - PAD.t - PAD.b);
  const cx = px(100);
  const cy = py(100);

  return (
    <div ref={ref} className="min-w-0">
      {size.w > 0 && (
        <svg width={w} height={h} className="block select-none" role="img"
          aria-label="板块相对轮动图">
          {/* Quadrant grounds, tinted by what they mean. */}
          <rect x={cx} y={PAD.t} width={w - PAD.r - cx} height={cy - PAD.t}
            fill={data.quadrants.leading.color} opacity={0.07} />
          <rect x={PAD.l} y={PAD.t} width={cx - PAD.l} height={cy - PAD.t}
            fill={data.quadrants.improving.color} opacity={0.07} />
          <rect x={cx} y={cy} width={w - PAD.r - cx} height={h - PAD.b - cy}
            fill={data.quadrants.weakening.color} opacity={0.07} />
          <rect x={PAD.l} y={cy} width={cx - PAD.l} height={h - PAD.b - cy}
            fill={data.quadrants.lagging.color} opacity={0.07} />

          <line x1={PAD.l} y1={cy} x2={w - PAD.r} y2={cy}
            stroke="var(--color-line-bright)" strokeWidth={1} />
          <line x1={cx} y1={PAD.t} x2={cx} y2={h - PAD.b}
            stroke="var(--color-line-bright)" strokeWidth={1} />

          {ORDER.map((q) => {
            const right = q === "leading" || q === "weakening";
            const top = q === "leading" || q === "improving";
            return (
              <text key={q}
                x={right ? w - PAD.r - 6 : PAD.l + 6}
                y={top ? PAD.t + 13 : h - PAD.b - 6}
                textAnchor={right ? "end" : "start"}
                fontSize={11.5} fontWeight={600}
                fill={data.quadrants[q].color} opacity={0.75}>
                {data.quadrants[q].label}
              </text>
            );
          })}

          <text x={w - PAD.r} y={h - 8} textAnchor="end" fontSize={10.5}
            fill="var(--color-ink-mute)">相对强度 RS-Ratio →</text>
          <text x={12} y={PAD.t + 4} fontSize={10.5} fill="var(--color-ink-mute)"
            transform={`rotate(-90 12 ${PAD.t + 4})`} textAnchor="end">
            ← 相对动能 RS-Momentum
          </text>

          {/* Focused sectors drawn last, so their trails and labels sit over
              the unfocused dots instead of under them. */}
          {[...data.sectors]
            .sort((a, b) => Number(focus.has(a.name)) - Number(focus.has(b.name)))
            .map((s) => (
              <Trail key={s.name} s={s} px={px} py={py}
                color={data.quadrants[s.quadrant].color}
                dim={hover !== null && hover !== s.name}
                lit={hover === s.name}
                shown={focus.has(s.name) || hover === s.name}
                onHover={onHover} />
            ))}
        </svg>
      )}
    </div>
  );
}

function Trail({ s, px, py, color, dim, lit, shown, onHover }: {
  s: RotationSector; px: (v: number) => number; py: (v: number) => number;
  color: string; dim: boolean; lit: boolean; shown: boolean;
  onHover: (n: string | null) => void;
}) {
  const pts = s.tail.map((p) => `${px(p.ratio)},${py(p.mom)}`).join(" ");
  const last = s.tail[s.tail.length - 1]!;
  const x = px(last.ratio);
  const y = py(last.mom);
  return (
    <g opacity={dim ? 0.2 : 1} onMouseEnter={() => onHover(s.name)}
      onMouseLeave={() => onHover(null)} style={{ cursor: "default" }}>
      {/* Every sector keeps a dot whatever the scope: where it SITS is the
          reading, and hiding it would make half the market invisible. */}
      <circle cx={x} cy={y} r={shown ? (lit ? 6 : 4.5) : 3} fill={color}
        opacity={shown ? 1 : 0.5}
        stroke="var(--color-panel)" strokeWidth={shown ? 1.5 : 1}>
        <title>
          {`${s.name}\n${s.quadrant} · RS ${fixed(s.ratio, 1)} / 动能 ${fixed(s.mom, 1)}`}
        </title>
      </circle>
      {shown && (
        <>
          <polyline points={pts} fill="none" stroke={color}
            strokeWidth={lit ? 2 : 1.2} opacity={0.5} strokeLinejoin="round" />
          {s.heading != null && <Arrow x={x} y={y} bearing={s.heading} color={color} />}
          {/* Painted with a panel-coloured outline so a label crossing another
              sector's trail stays readable. */}
          <text x={x + 8} y={y + 4} fontSize={11} fontWeight={lit ? 700 : 500}
            fill="var(--color-ink-dim)" pointerEvents="none"
            stroke="var(--color-panel)" strokeWidth={2.5} paintOrder="stroke">
            {s.name}
          </text>
        </>
      )}
    </g>
  );
}

/** A chevron at the dot, pointing where the sector is heading. */
function Arrow({ x, y, bearing, color }: {
  x: number; y: number; bearing: number; color: string;
}) {
  // Bearing is clockwise from north in DATA space, where the momentum axis
  // grows upward; SVG's y grows downward, hence the negated dy.
  const r = (bearing * Math.PI) / 180;
  const dx = Math.sin(r);
  const dy = -Math.cos(r);
  const tip = 15;
  return (
    <line x1={x} y1={y} x2={x + dx * tip} y2={y + dy * tip}
      stroke={color} strokeWidth={2} strokeLinecap="round" opacity={0.8} />
  );
}

function Calls({ title, hint, rows, tone, hover, onHover }: {
  title: string; hint: string; rows: RotationCall[]; tone: "in" | "out";
  hover: string | null; onHover: (n: string | null) => void;
}) {
  return (
    <div className="rounded-lg bg-sunken p-2.5 flex flex-col gap-1.5">
      <div className="flex items-baseline gap-2">
        <span className={`text-[13px] font-semibold ${tone === "in" ? "text-cyan" : "text-brand-ink"}`}>
          {tone === "in" ? "↗ " : "↘ "}{title}
        </span>
      </div>
      <p className="text-[11px] text-ink-mute leading-snug">{hint}</p>

      {rows.length === 0 ? (
        <p className="text-[11.5px] text-ink-mute py-1">本期没有板块落在这个象限。</p>
      ) : rows.map((r) => (
        <div key={r.name}
          onMouseEnter={() => onHover(r.name)} onMouseLeave={() => onHover(null)}
          className={`flex items-baseline gap-2 text-[12px] rounded px-1 py-0.5 ${
            hover === r.name ? "bg-panel" : ""}`}>
          <span className="font-medium truncate flex-1 min-w-0">{r.name}</span>
          <span className="tnum text-ink-mute" title="RS-Ratio / RS-Momentum">
            {fixed(r.ratio, 1)}/{fixed(r.mom, 1)}
          </span>
          <Trend c={r} tone={tone} />
        </div>
      ))}
    </div>
  );
}

/**
 * The ABSOLUTE trend, kept visually and semantically apart from the ranking.
 *
 * Everything else on this panel is relative — the sector priced in units of
 * the market. This one number is not: it is where the sector index sits
 * against its own 20-day mean. The two disagree constantly and usefully. A
 * sector entering 改善 while still below its own mean is winning a falling
 * market, which is a defensive rotation and a different trade from a sector
 * entering 改善 on the way up.
 *
 * It is NOT breadth, despite the table it comes from being called that, and
 * it therefore cannot corroborate a cap-weighted index the way a real member
 * count would. Labelled for what it measures. See sector_rotation.load_trend.
 */
function Trend({ c, tone }: { c: RotationCall; tone: "in" | "out" }) {
  if (c.rising == null) {
    return <span className="text-[11px] text-ink-mute" title="没有该板块的趋势数据">—</span>;
  }
  const d = c.trend!.delta_pp;
  const agrees = c.rising;
  return (
    <span className={`text-[11px] tnum ${agrees ? "text-ink-dim" : "text-ink-mute"}`}
      title={`绝对趋势分 ${fixed(c.trend!.now_pct, 0)}/100（板块指数相对自身20日均线，`
        + `±5% 映射到 0–100），较上期 ${signed(d, 0)}pp。${
        agrees
          ? "绝对趋势与相对信号同向。"
          : tone === "in"
            ? "相对在改善，但绝对趋势仍在走坏 — 跌得比大盘少，属防守型轮动。"
            : "相对在走弱，但绝对价格仍在走强 — 只是跑输，不等于下跌。"}`}>
      {agrees ? "↑" : "↓"} {signed(d, 0)}pp
    </span>
  );
}

function Edge({ data, freq }: { data: Rotation; freq: "w" | "d" }) {
  const unit = freq === "w" ? "周" : "日";
  return (
    <div className="rounded-lg border border-line p-2.5 flex flex-col gap-1">
      <div className="flex items-baseline gap-2">
        <span className="text-[12.5px] font-medium">象限验证</span>
        <span className="label">未来 {data.edge.horizon}{unit}超额收益</span>
      </div>
      {data.edge.rows.map((r) => (
        <div key={r.quadrant} className="flex items-baseline gap-2 text-[11.5px]">
          <span className="flex-1 truncate" style={{ color: data.quadrants[r.quadrant].color }}>
            {data.quadrants[r.quadrant].label}
          </span>
          <span className="tnum text-ink-mute w-10 text-right">n={r.n}</span>
          <span className="tnum w-12 text-right text-ink-mute">{fixed(r.win_pct, 0)}%</span>
          <span className={`tnum w-16 text-right font-medium ${
            (r.edge_pp ?? 0) > 0 ? "text-up" : (r.edge_pp ?? 0) < 0 ? "text-down" : "text-flat"}`}>
            {r.edge_pp == null ? "—" : `${signed(r.edge_pp, 2)}pp`}
          </span>
        </div>
      ))}
      <p className="text-[11px] text-ink-mute leading-snug">
        用本市场自己的历史检验：处在该象限时，此后 {data.edge.horizon}{unit}相对基准的平均超额收益，
        已扣除全样本均值（{signed(data.edge.baseline_pct, 2)}%）。
        胜率一栏为超额为正的比例。样本互相重叠，n 不等于独立观测数。
      </p>
    </div>
  );
}
