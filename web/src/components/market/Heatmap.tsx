/**
 * 市场热力图 — the whole mapped market on one screen, by size and by move.
 *
 * Nested treemap: sectors sized by 流通市值, and inside each one its own
 * constituents sized the same way. Colour is the session's move on the A-share
 * convention — red up, green down — scaled to the largest absolute move on the
 * page, so a quiet day still has contrast and a violent one does not clip.
 *
 * Boxes are squared rather than sliced (see lib/treemap.ts). That is not a
 * cosmetic choice: sliced boxes turn every small sector into an unclickable
 * 3px sliver, and the point of the panel is that you can click the outlier you
 * just spotted.
 *
 * Labels are drawn only where they fit. A truncated name in a tiny box is
 * noise you have to read past; the box still carries its name in a tooltip and
 * still opens the stock on click.
 */

import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import type { Heatmap as HeatmapData } from "../../lib/types";
import { heatColor, heatInk, squarify, type Rect } from "../../lib/treemap";
import { signed, yi } from "../../lib/format";

const HEADER = 17;      // sector caption strip
const GAP = 3;          // between sector boxes
const INNER = 1;        // between stock boxes

/** A stock box needs this much room before its name is worth drawing. */
const NAME_W = 52;
const NAME_H = 26;

export function Heatmap({ data, width, height }: {
  data: HeatmapData; width: number; height: number;
}) {
  const nav = useNavigate();

  const scale = useMemo(() => {
    const all = data.sectors.flatMap((s) => s.stocks.map((x) => Math.abs(x.pct)));
    // 95th percentile, not the maximum: one limit-up name would otherwise set
    // the scale and flatten every other box to near-white.
    const sorted = all.sort((a, b) => a - b);
    return Math.max(sorted[Math.floor(sorted.length * 0.95)] ?? 3, 2);
  }, [data]);

  const sectors = useMemo(() => {
    if (width < 120 || height < 120) return [];
    return squarify(
      data.sectors.map((s) => ({ ...s, value: s.mcap })),
      { x: 0, y: 0, w: width, h: height });
  }, [data, width, height]);

  if (!sectors.length) return null;

  return (
    <svg width={width} height={height} className="block select-none"
      role="img" aria-label={`${data.sectors.length} 个板块的市值热力图`}>
      {sectors.map((sec) => {
        const box: Rect = {
          x: sec.x + GAP / 2, y: sec.y + GAP / 2,
          w: Math.max(sec.w - GAP, 0), h: Math.max(sec.h - GAP, 0),
        };
        const showHeader = box.h > HEADER + 14 && box.w > 44;
        const inner: Rect = showHeader
          ? { x: box.x, y: box.y + HEADER, w: box.w, h: box.h - HEADER }
          : box;
        const leaves = squarify(sec.stocks.map((s) => ({ ...s, value: s.mcap })), inner);
        const clip = `hm-${sec.name.replace(/[^\w一-龥]/g, "")}`;

        return (
          <g key={sec.name}>
            <rect {...box} rx={4} fill="var(--color-sunken)" />
            {showHeader && (
              <>
                {/* A narrow sector's caption must not run across its
                    neighbour's. SVG text has no overflow rule, so the box
                    clips it — the full name is still in the tooltip. */}
                <clipPath id={clip}>
                  <rect x={box.x} y={box.y} width={box.w} height={HEADER} />
                </clipPath>
                <text x={box.x + 5} y={box.y + 12} fontSize={11} fontWeight={600}
                  fill="var(--color-ink-dim)" clipPath={`url(#${clip})`}>
                  {sec.name}
                  <tspan fill={sec.pct >= 0 ? "var(--color-up)" : "var(--color-down)"}
                    fontWeight={500} dx={5}>{signed(sec.pct, 2)}%</tspan>
                  <title>{`${sec.name} ${signed(sec.pct, 2)}%　${yi(sec.mcap)}`}</title>
                </text>
              </>
            )}
            {leaves.map((leaf) => {
              const fill = heatColor(leaf.pct, scale);
              const ink = heatInk(fill);
              const w = Math.max(leaf.w - INNER, 0);
              const h = Math.max(leaf.h - INNER, 0);
              const named = w >= NAME_W && h >= NAME_H;
              return (
                <g key={leaf.t} onClick={() => nav(`/?t=${leaf.t}`)}
                  style={{ cursor: "pointer" }}>
                  <rect x={leaf.x} y={leaf.y} width={w} height={h} fill={fill} rx={2}>
                    <title>
                      {`${leaf.n} ${leaf.t}\n${signed(leaf.pct, 2)}%　流通市值 ${yi(leaf.mcap)}`}
                    </title>
                  </rect>
                  {named && (
                    <>
                      <text x={leaf.x + w / 2} y={leaf.y + h / 2 - 1}
                        textAnchor="middle" fontSize={11} fill={ink}
                        pointerEvents="none">{leaf.n}</text>
                      <text x={leaf.x + w / 2} y={leaf.y + h / 2 + 11}
                        textAnchor="middle" fontSize={10} fill={ink}
                        opacity={0.85} pointerEvents="none" className="tnum">
                        {signed(leaf.pct, 2)}%
                      </text>
                    </>
                  )}
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}
