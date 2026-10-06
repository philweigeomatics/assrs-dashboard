/**
 * Sixty sessions of shape, at row height.
 *
 * Not a chart: no axes, no scale, no interaction. The only questions it
 * answers are "which way" and "how violently", and both are legible at
 * 96×22 while a real chart is not. Click through to the analysis page for
 * anything more.
 */

type Props = {
  values: number[];
  /** Red for a rise in Shanghai, green in New York — the market's own rule. */
  upIsRed: boolean;
  width?: number;
  height?: number;
};

export function Sparkline({ values, upIsRed, width = 96, height = 22 }: Props) {
  const clean = values.filter((v) => Number.isFinite(v));
  if (clean.length < 2) {
    return <span className="label">—</span>;
  }

  const lo = Math.min(...clean);
  const hi = Math.max(...clean);
  const span = hi - lo || 1;
  // A whole-pixel inset top and bottom so the extremes are not clipped by
  // the stroke width.
  const pad = 2;
  const x = (i: number) => (i / (clean.length - 1)) * width;
  const y = (v: number) => height - pad - ((v - lo) / span) * (height - pad * 2);

  const first = clean[0]!;
  const last = clean[clean.length - 1]!;
  const rising = last >= first;
  const stroke = rising === upIsRed ? "var(--color-up)" : "var(--color-down)";

  const d = clean
    .map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`)
    .join("");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height}
      className="block shrink-0" role="img"
      aria-label={`近 ${clean.length} 日走势，${rising ? "上行" : "下行"}`}>
      {/* Where the window started, so the line has something to be above. */}
      <line x1={0} x2={width} y1={y(first)} y2={y(first)}
        stroke="var(--color-line)" strokeDasharray="2 3"
        vectorEffect="non-scaling-stroke" />
      <path d={d} fill="none" stroke={stroke} strokeWidth={1.4}
        vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
      <circle cx={x(clean.length - 1)} cy={y(last)} r={1.8} fill={stroke} />
    </svg>
  );
}
