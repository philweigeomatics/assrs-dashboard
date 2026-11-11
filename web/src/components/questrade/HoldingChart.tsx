/**
 * One holding's price over the activity window, with your own trades on it.
 *
 * The line is the RAW close (split-adjusted, not dividend-adjusted), because
 * the fills placed on it are real prices: on a dividend-adjusted series a
 * monthly payer's old buys would all float above the line.
 *
 * Each buy (▲) and sell (▼) sits at its own fill price on its own day, with a
 * vertical rule from there to today's price line. The rule is the answer to
 * "how far is this fill from now": its length is the gap, its colour whether
 * the fill looks right from today (green: bought below / sold above today's
 * price), and the label is today's price relative to the fill.
 *
 * Dividends run in a lane underneath, on the day the cash arrived — which is
 * a week or so after the ex-date, when the price actually drops. An ex-date
 * that has passed with nothing paid yet is drawn hollow.
 */

import { useLayoutEffect, useMemo, useRef, useState } from "react";
import type { QtHolding, QtPriceHistory } from "../../lib/types";
import type { ActivityRow, PendingDividend } from "../../lib/lots";
import { fixed, signed } from "../../lib/format";

const H = 300;
const M = { l: 6, r: 74, t: 14, b: 46 };
const LABEL_LIMIT = 14;

type Fill = {
  date: string; side: "buy" | "sell"; qty: number; price: number;
  account: string; restated: boolean;
};
type Paid = { date: string; net: number; account: string };

const t = (d: string) => Date.parse(d);

function money(v: number, ccy: string, nd = 2): string {
  const sym = ccy === "CAD" ? "C$" : ccy === "USD" ? "$" : "";
  return `${v < 0 ? "-" : ""}${sym}${Math.abs(v).toLocaleString(undefined, {
    minimumFractionDigits: nd, maximumFractionDigits: nd })}`;
}

function niceStep(span: number, n: number): number {
  const raw = span / n;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}

/** Fills restated into today's share terms when a split happened since. */
function fillsOf(rows: ActivityRow[], ccy: string, splits: QtPriceHistory["splits"]): Fill[] {
  const out: Fill[] = [];
  for (const r of rows) {
    if (r.type !== "Trades" || r.currency !== ccy) continue;
    const q = r.quantity ?? 0, p = r.price ?? 0;
    if (!q || !p) continue;
    const ratio = splits.filter((s) => s.date > r.date).reduce((a, s) => a * s.ratio, 1);
    out.push({ date: r.date, side: q > 0 ? "buy" : "sell", qty: Math.abs(q) * ratio,
               price: p / ratio, account: r.account, restated: ratio !== 1 });
  }
  return out.sort((a, b) => a.date.localeCompare(b.date));
}

export function HoldingChart({ px, rows, h, pending, today }: {
  px: QtPriceHistory; rows: ActivityRow[]; h: QtHolding;
  pending: PendingDividend[]; today: string;
}) {
  const box = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(640);
  const [hover, setHover] = useState<number | null>(null);
  const [focus, setFocus] = useState<number | null>(null);

  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver((es) => {
      const e = es[0];
      if (e) setW(Math.max(320, e.contentRect.width));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const ccy = h.currency;
  const now = h.price;
  const fills = useMemo(() => fillsOf(rows, ccy, px.splits), [rows, ccy, px.splits]);
  const paid: Paid[] = useMemo(() => rows
    .filter((r) => r.type === "Dividends" && (r.net ?? 0) > 0 && r.currency === ccy)
    .map((r) => ({ date: r.date, net: r.net!, account: r.account })), [rows, ccy]);

  const closes = px.closes;
  if (closes.length < 2 || now == null) {
    return <p className="label py-6 text-center">没有足够的行情画图。</p>;
  }

  // ── scales ──
  const x0 = t(closes[0]![0]), x1 = Math.max(t(today), t(closes[closes.length - 1]![0]));
  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  const x = (d: string) => M.l + (t(d) - x0) / (x1 - x0 || 1) * pw;
  const ys = [...closes.map((c) => c[1]), ...fills.map((f) => f.price), now,
              ...(h.avg_cost != null ? [h.avg_cost] : [])];
  let lo = Math.min(...ys), hi = Math.max(...ys);
  const pad = (hi - lo || hi * 0.1 || 1) * 0.06;
  lo -= pad; hi += pad;
  const y = (v: number) => M.t + (hi - v) / (hi - lo) * ph;

  const step = niceStep(hi - lo, 4);
  const yTicks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) yTicks.push(v);

  const months = (x1 - x0) / (30.4 * 86400e3);
  const every = [1, 2, 3, 6, 12].find((m) => months / m <= 7) ?? 12;
  const xTicks: string[] = [];
  const s = new Date(x0);
  for (let d = new Date(s.getFullYear(), s.getMonth() + 1, 1); d.getTime() <= x1;
       d = new Date(d.getFullYear(), d.getMonth() + 1, 1)) {
    if (d.getMonth() % every === 0) {
      xTicks.push(`${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-01`);
    }
  }

  const path = closes.map(([d, c], i) => `${i ? "L" : "M"}${x(d).toFixed(1)},${y(c).toFixed(1)}`)
    .join("");

  const yNow = y(now);
  const yAvg = h.avg_cost != null ? y(h.avg_cost) : null;
  // Keep the two right-hand labels from sitting on each other.
  const avgLabelY = yAvg == null ? null
    : Math.abs(yAvg - yNow) < 15 ? yNow + (yAvg >= yNow ? 15 : -15) : yAvg;

  const lane = H - M.b + 14;
  const maxDiv = Math.max(...paid.map((p) => p.net), 1);
  const good = "var(--color-down)", bad = "var(--color-up)";
  const showLabels = fills.length <= LABEL_LIMIT;

  // ── hover ──
  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const at = x0 + (e.clientX - r.left) / r.width * (x1 - x0);
    let a = 0, b = closes.length - 1;
    while (b - a > 1) {
      const m = (a + b) >> 1;
      if (t(closes[m]![0]) < at) a = m; else b = m;
    }
    setHover(Math.abs(t(closes[a]![0]) - at) <= Math.abs(t(closes[b]![0]) - at) ? a : b);
  };
  const hv = hover == null ? null : closes[hover] ?? null;
  const near = (d: string) => hv && Math.abs(t(d) - t(hv[0])) <= 3 * 86400e3;
  const hvFills = hv ? fills.filter((f) => near(f.date)) : [];
  const hvPaid = hv ? paid.filter((p) => near(p.date)) : [];
  const hvX = hv ? x(hv[0]) : 0;

  return (
    <div className="flex flex-col gap-1 min-w-0">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-ink-mute">
        <span><Tri side="buy" /> 买入</span>
        <span><Tri side="sell" /> 卖出</span>
        <span><Dot /> 股息到账</span>
        {pending.length > 0 && <span><Dot hollow /> 已除息、未到账</span>}
        <span className="flex items-center gap-1">
          <svg width="16" height="6"><line x1="0" x2="16" y1="3" y2="3"
            stroke="var(--color-ink)" strokeWidth="1.5" /></svg>现价</span>
        {h.avg_cost != null && (
          <span className="flex items-center gap-1">
            <svg width="16" height="6"><line x1="0" x2="16" y1="3" y2="3"
              stroke="var(--color-cyan)" strokeDasharray="3 2" /></svg>持仓均价</span>
        )}
        <span>竖线 = 成交价到现价的距离，<span className="text-down">绿</span>
          {" "}= 从今天看是对的，<span className="text-up">红</span> = 相反</span>
      </div>

      <div ref={box} className="relative w-full">
        <svg width={W} height={H} className="block select-none" role="img"
          aria-label={`${h.symbol} 价格与交易记录`}>
          {/* grid */}
          {yTicks.map((v) => (
            <g key={v}>
              <line x1={M.l} x2={M.l + pw} y1={y(v)} y2={y(v)} stroke="var(--color-line)"
                strokeOpacity={0.6} />
              <text x={M.l + pw + 6} y={y(v) + 3.5} fontSize="10"
                fill="var(--color-ink-mute)">{fixed(v, v < 10 ? 2 : 0)}</text>
            </g>
          ))}
          {xTicks.map((d) => (
            <g key={d}>
              <line x1={x(d)} x2={x(d)} y1={M.t} y2={M.t + ph} stroke="var(--color-line)"
                strokeOpacity={0.35} />
              <text x={x(d)} y={H - 4} fontSize="10" textAnchor="middle"
                fill="var(--color-ink-mute)">{d.slice(0, 7)}</text>
            </g>
          ))}

          {/* price */}
          <path d={path} fill="none" stroke="var(--color-ink-dim)" strokeWidth={1.25}
            strokeOpacity={0.75} />

          {/* average cost and today's price */}
          {yAvg != null && (
            <>
              <line x1={M.l} x2={M.l + pw} y1={yAvg} y2={yAvg} stroke="var(--color-cyan)"
                strokeDasharray="4 3" />
              <text x={M.l + pw + 6} y={avgLabelY! + 3.5} fontSize="10" fontWeight="600"
                fill="var(--color-cyan)">均 {fixed(h.avg_cost, 2)}</text>
            </>
          )}
          <line x1={M.l} x2={M.l + pw} y1={yNow} y2={yNow} stroke="var(--color-ink)"
            strokeWidth={1.5} />
          <rect x={M.l + pw + 2} y={yNow - 8} width={M.r - 4} height={16} rx={3}
            fill="var(--color-ink)" />
          <text x={M.l + pw + 6} y={yNow + 3.5} fontSize="10" fontWeight="600"
            fill="var(--color-panel)">现 {fixed(now, 2)}</text>

          {/* the gap from each fill to today */}
          {fills.map((f, i) => {
            const fx = x(f.date), fy = y(f.price);
            const ok = f.side === "buy" ? f.price <= now : f.price >= now;
            const pct = (now / f.price - 1) * 100;
            const lit = focus === i;
            return (
              <g key={`gap-${i}`} opacity={focus == null || lit ? 1 : 0.35}>
                <line x1={fx} x2={fx} y1={fy} y2={yNow} stroke={ok ? good : bad}
                  strokeWidth={lit ? 2 : 1.25} strokeDasharray="3 2" />
                {(showLabels || lit) && Math.abs(fy - yNow) > 12 && (
                  <text x={fx + 4} y={(fy + yNow) / 2 + 3.5} fontSize="10" fontWeight="600"
                    fill={ok ? good : bad}
                    stroke="var(--color-panel)" strokeWidth={3} paintOrder="stroke">
                    {signed(pct, 1)}%
                  </text>
                )}
              </g>
            );
          })}

          {/* the fills themselves */}
          {fills.map((f, i) => (
            <g key={`fill-${i}`} transform={`translate(${x(f.date)},${y(f.price)})`}
              onMouseEnter={() => setFocus(i)} onMouseLeave={() => setFocus(null)}
              style={{ cursor: "default" }}>
              <path d={f.side === "buy" ? "M0,-6 L6,5 L-6,5 Z" : "M0,6 L6,-5 L-6,-5 Z"}
                fill={f.side === "buy" ? "var(--color-cyan)" : "var(--color-brand)"}
                stroke="var(--color-panel)" strokeWidth={1.25} />
              <title>
                {`${f.date} ${f.side === "buy" ? "买入" : "卖出"} `
                  + `${fixed(f.qty, f.qty % 1 ? 4 : 0)} 股 @ ${money(f.price, ccy)}`
                  + `${f.restated ? "（按拆股后折算）" : ""} · ${f.account}`
                  + `\n现价相对 ${signed((now / f.price - 1) * 100, 1)}%`}
              </title>
            </g>
          ))}

          {/* dividend lane */}
          <line x1={M.l} x2={M.l + pw} y1={lane} y2={lane} stroke="var(--color-line)" />
          {paid.map((p, i) => (
            <circle key={`div-${i}`} cx={x(p.date)} cy={lane}
              r={2.5 + Math.sqrt(p.net / maxDiv) * 2.5}
              fill="var(--color-violet)" fillOpacity={0.8}>
              <title>{`${p.date} 股息到账 ${money(p.net, ccy)} · ${p.account}`}</title>
            </circle>
          ))}
          {pending.map((p) => (
            <circle key={`pend-${p.ex_date}`} cx={x(p.ex_date)} cy={lane} r={4.5}
              fill="var(--color-panel)" stroke="var(--color-violet)" strokeWidth={1.5}
              strokeDasharray="2 1.5">
              <title>{`${p.ex_date} 除息，每股 ${money(p.per_share, ccy, 4)} × `
                + `${fixed(p.shares, 0)} 股 ≈ ${money(p.amount, ccy)}，尚未到账`}</title>
            </circle>
          ))}
          <text x={M.l + pw + 6} y={lane + 3.5} fontSize="10"
            fill="var(--color-violet)">股息</text>

          {/* hover */}
          {hv && (
            <g pointerEvents="none">
              <line x1={hvX} x2={hvX} y1={M.t} y2={lane} stroke="var(--color-ink-mute)"
                strokeDasharray="2 2" />
              <circle cx={hvX} cy={y(hv[1])} r={3} fill="var(--color-ink)" />
            </g>
          )}
          <rect x={M.l} y={M.t} width={pw} height={lane - M.t + 6} fill="transparent"
            onMouseMove={onMove} onMouseLeave={() => setHover(null)}
            style={{ pointerEvents: focus == null ? "all" : "none" }} />
          {/* fills drawn again on top of the hover layer so they stay hoverable */}
          {fills.map((f, i) => (
            <circle key={`hit-${i}`} cx={x(f.date)} cy={y(f.price)} r={8} fill="transparent"
              onMouseEnter={() => { setFocus(i); setHover(null); }}
              onMouseLeave={() => setFocus(null)} />
          ))}
        </svg>

        {hv && (
          <div className="absolute pointer-events-none z-10 rounded-md bg-panel shadow-md
            border border-line px-2 py-1.5 text-[11px] leading-snug min-w-[150px]"
            style={{ top: 4, left: hvX > W / 2 ? undefined : hvX + 10,
                     right: hvX > W / 2 ? W - hvX + 10 : undefined }}>
            <div className="font-mono tnum text-ink-mute">{hv[0]}</div>
            <div className="tnum">收盘 <b>{money(hv[1], ccy)}</b>
              <span className="text-ink-mute"> · 现价相对 {signed((now / hv[1] - 1) * 100, 1)}%</span>
            </div>
            {hvFills.map((f, i) => (
              <div key={i} className="tnum">
                <span className={f.side === "buy" ? "text-cyan" : "text-brand-ink"}>
                  {f.side === "buy" ? "▲ 买入" : "▼ 卖出"}</span>
                {" "}{fixed(f.qty, f.qty % 1 ? 4 : 0)} 股 @ {money(f.price, ccy)}
              </div>
            ))}
            {hvPaid.map((p, i) => (
              <div key={`p${i}`} className="tnum text-violet">● 股息 {money(p.net, ccy)}</div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Tri({ side }: { side: "buy" | "sell" }) {
  return (
    <svg width="10" height="10" viewBox="-6 -6 12 12" className="inline align-[-1px]">
      <path d={side === "buy" ? "M0,-6 L6,5 L-6,5 Z" : "M0,6 L6,-5 L-6,-5 Z"}
        fill={side === "buy" ? "var(--color-cyan)" : "var(--color-brand)"} />
    </svg>
  );
}

function Dot({ hollow }: { hollow?: boolean }) {
  return (
    <svg width="10" height="10" viewBox="-5 -5 10 10" className="inline align-[-1px]">
      <circle r="3.8" fill={hollow ? "none" : "var(--color-violet)"}
        stroke="var(--color-violet)" strokeWidth={hollow ? 1.5 : 0}
        strokeDasharray={hollow ? "2 1.5" : undefined} />
    </svg>
  );
}
