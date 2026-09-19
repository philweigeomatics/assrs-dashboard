/**
 * 供应链图 — what the company makes, and which sectors buy it.
 *
 * Opens as a window over the report rather than sitting inside it: the graph
 * wants width, and the report's two-column body does not have any to spare.
 *
 * Drawn as a bipartite layout — products down the left, the macro sectors they
 * feed down the right — rather than the force-directed blob the Streamlit page
 * uses. The data IS bipartite (every link goes product → sector, never
 * product → product), so a force simulation spends its effort hiding the one
 * piece of structure that exists. Two columns make "this product reaches four
 * sectors" and "this sector depends on one product" readable at a glance, and
 * it needs no layout library.
 *
 * Hovering either side dims everything it does not touch, which is the only
 * interaction the diagram actually needs.
 */

import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";

export type ChainGraph = {
  company_name?: string;
  products?: string[];
  macro_sectors?: string[];
  links?: { source: string; target: string }[];
};

/** "Optical Fiber / 光纤光缆" → ["Optical Fiber", "光纤光缆"]. */
function split(label: string): [string, string] {
  const i = label.indexOf(" / ");
  return i < 0 ? [label, ""] : [label.slice(0, i), label.slice(i + 3)];
}

const ROW = 46;
const PAD = 18;
const COL_W = 190;

export function SupplyChainWindow({ ticker, graph, onClose }: {
  ticker: string; graph: ChainGraph; onClose: () => void;
}) {
  // Esc closes, because a window that traps you is worse than no window.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const [hover, setHover] = useState<string | null>(null);
  const products = graph.products ?? [];
  const sectors = graph.macro_sectors ?? [];
  const links = graph.links ?? [];

  const rows = Math.max(products.length, sectors.length);
  const h = Math.max(rows * ROW + PAD * 2, 220);
  const w = COL_W * 2 + 170;
  const leftX = COL_W;
  const rightX = w - COL_W;

  const yOf = (i: number, n: number) =>
    PAD + (h - PAD * 2) * (n === 1 ? 0.5 : i / (n - 1)) - (n === 1 ? 0 : 0);

  const touched = (name: string) =>
    hover === null || hover === name
    || links.some((l) => (l.source === hover && l.target === name)
      || (l.target === hover && l.source === name));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(0,0,0,0.35)" }}
      onClick={onClose} role="dialog" aria-modal="true" aria-label="供应链图">
      <div className="card max-w-[980px] w-full max-h-[88vh] overflow-auto p-4 flex flex-col gap-3"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-baseline gap-3">
          {/* Both the heading and the button must refuse to wrap: a long
              company name otherwise breaks "供应链图" and "重新生成" across
              two lines each. */}
          <h2 className="text-[15px] font-semibold whitespace-nowrap shrink-0">🔗 供应链图</h2>
          <span className="label truncate min-w-0">{graph.company_name || ticker}</span>
          <span className="ml-auto flex items-center gap-2 shrink-0">
            <Regenerate ticker={ticker} />
            <button onClick={onClose}
              className="text-[13px] text-ink-mute hover:text-ink px-2 whitespace-nowrap">
              关闭 ✕
            </button>
          </span>
        </div>

        {products.length === 0 ? (
          <p className="label py-10 text-center">这张图没有产品节点。</p>
        ) : (
          <>
            <div className="flex justify-between px-1 label">
              <span>产品 / 服务 · {products.length}</span>
              <span>下游行业 · {sectors.length}</span>
            </div>
            <svg viewBox={`0 0 ${w} ${h}`} className="w-full block"
              style={{ height: h }} role="img"
              aria-label={`${products.length} 个产品连接到 ${sectors.length} 个下游行业`}>
              {links.map((l, i) => {
                const si = products.indexOf(l.source);
                const ti = sectors.indexOf(l.target);
                if (si < 0 || ti < 0) return null;
                const y1 = yOf(si, products.length);
                const y2 = yOf(ti, sectors.length);
                const lit = hover === null || hover === l.source || hover === l.target;
                const mid = (leftX + rightX) / 2;
                return (
                  <path key={i}
                    d={`M${leftX} ${y1} C${mid} ${y1} ${mid} ${y2} ${rightX} ${y2}`}
                    fill="none" stroke="var(--color-cyan)"
                    strokeWidth={lit ? 1.8 : 1}
                    opacity={lit ? 0.55 : 0.12} />
                );
              })}

              {products.map((p, i) => (
                <Node key={p} x={leftX} y={yOf(i, products.length)} label={p}
                  align="end" lit={touched(p)}
                  onEnter={() => setHover(p)} onLeave={() => setHover(null)} />
              ))}
              {sectors.map((s, i) => (
                <Node key={s} x={rightX} y={yOf(i, sectors.length)} label={s}
                  align="start" lit={touched(s)} accent
                  onEnter={() => setHover(s)} onLeave={() => setHover(null)} />
              ))}
            </svg>
            <p className="label leading-snug">
              左侧为公司生产的产品，右侧为消化这些产品的下游行业，连线表示供货关系。
              悬停任一节点可只看它的连接。由模型生成并缓存，仅供参考。
            </p>
          </>
        )}
      </div>
    </div>
  );
}

function Node({ x, y, label, align, lit, accent = false, onEnter, onLeave }: {
  x: number; y: number; label: string; align: "start" | "end";
  lit: boolean; accent?: boolean; onEnter: () => void; onLeave: () => void;
}) {
  const [en, zh] = split(label);
  const dx = align === "end" ? -10 : 10;
  return (
    <g opacity={lit ? 1 : 0.28} onMouseEnter={onEnter} onMouseLeave={onLeave}
      style={{ cursor: "default" }}>
      <circle cx={x} cy={y} r={4.5}
        fill={accent ? "#7c3aed" : "var(--color-cyan)"} />
      <text x={x + dx} y={y - 2} textAnchor={align} fontSize={12.5}
        fill="var(--color-ink)">{zh || en}</text>
      {zh && (
        <text x={x + dx} y={y + 11} textAnchor={align} fontSize={10.5}
          fill="var(--color-ink-mute)">{en}</text>
      )}
    </g>
  );
}

function Regenerate({ ticker }: { ticker: string }) {
  const qc = useQueryClient();
  const run = useMutation({
    mutationFn: () => api.equityGenerate(ticker, "supply-chain", true),
    // Two pages open this window off two different queries — the equity brief
    // and the watchlist. Invalidating only one leaves the other showing the
    // graph it just replaced.
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["equity", ticker] });
      qc.invalidateQueries({ queryKey: ["chain", ticker] });
    },
  });
  return (
    <span className="flex items-center gap-2">
      <button onClick={() => run.mutate()} disabled={run.isPending}
        className="h-7 px-2.5 rounded-md bg-violet text-white text-[12px] font-semibold whitespace-nowrap disabled:opacity-60">
        {run.isPending ? "生成中…" : "重新生成"}
      </button>
      {run.isError && <span className="text-[12px] text-up">{(run.error as Error).message}</span>}
    </span>
  );
}
