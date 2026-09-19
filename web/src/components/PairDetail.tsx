/**
 * One pair, opened up — the spread, and what the two stocks were actually doing.
 *
 * A z-score chart alone cannot answer the question anyone asks of it. "The
 * spread stretched to −2.3 and came back" is the same picture whether both
 * stocks rose and the one we bought rose faster, both fell and the one we
 * bought fell less, or one genuinely went the other way. Those are three
 * completely different things to have been holding, and the z-line draws
 * them identically.
 *
 * So the legs are drawn under the spread on the same x-axis, rebased so the
 * comparison is a comparison and not a picture of which stock costs more.
 * Click a trade — a marker or a row — and both panes zoom to that window,
 * rebased at the entry, so each line ENDS at exactly the return the table
 * reports for that leg.
 *
 * The lines and the trade dots use the pair's two identity colours, not
 * 红涨绿跌: they say WHICH stock, not which direction. Direction is carried
 * by the numbers, which do follow the convention.
 *
 * Text lives in an HTML layer over the SVG rather than in it. The panes
 * stretch to the container width (preserveAspectRatio="none"), which is right
 * for a line and would squash a label.
 */

import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import type { LegPattern, PairResult, PairTrade as Trade } from "../lib/types";
import type { Span } from "../lib/pairChart";
import { extent, gapInUnits, pickLabels, placeTrades, rebase, zoomWindow }
  from "../lib/pairChart";
import { fixed, signed } from "../lib/format";
import { useSize } from "../lib/useSize";

const VW = 1000;
const ZH = 130;
const PH = 165;

const COL_A = "var(--color-cyan)";
const COL_B = "#a855f7";

const PATTERN_CN: Record<LegPattern, string> = {
  BOTH_UP: "同涨",
  BOTH_DOWN: "同跌",
  A_UP_B_DOWN: "反向",
  B_UP_A_DOWN: "反向",
  FLAT: "都没动",
};

/** Minimum gap, in real screen pixels, between two marker labels. */
const LABEL_GAP_PX = 84;


export function PairDetail({ p }: { p: PairResult }) {
  const [focus, setFocus] = useState<number | null>(null);
  const [paneRef, pane] = useSize<HTMLDivElement>();
  const n = p.dates.length;

  const spans = useMemo(() => placeTrades(p.dates, p.trades), [p.dates, p.trades]);
  const sel = spans.find((s) => s.i === focus && s.a >= 0) ?? null;

  const [lo, hi] = zoomWindow(sel, n);
  const width = Math.max(1, hi - lo);
  const x = (i: number) => ((i - lo) / width) * VW;

  // Rebased at the entry when a trade is open on screen, so each line ENDS at
  // exactly the return the table reports for that leg.
  const base = sel ? sel.a : lo;
  const ra = (i: number) => rebase(p.px_a, base, i);
  const rb = (i: number) => rebase(p.px_b, base, i);

  // What the legend reports. Zoomed in, that has to be the EXIT and not the
  // padded right edge, or the legend and the trade row beside it disagree
  // about the same trade.
  const readAt = sel ? sel.b : hi;
  const legendNote = sel
    ? `${p.dates[sel.a]} 买入日 → ${p.dates[sel.b]} 卖出日`
    : `${p.dates[lo]} → ${p.dates[hi]}`;

  const [min, max] = extent(p.px_a, p.px_b, base, lo, hi);
  const room = Math.max(0.5, (max - min) * 0.08);
  const top = max + room;
  const bottom = min - room;
  const py = (v: number) => PH - ((v - bottom) / (top - bottom)) * PH;
  const zy = (z: number) =>
    ZH / 2 - (Math.max(-3.2, Math.min(3.2, z)) / 3.2) * (ZH / 2 - 6);

  const legPath = (f: (i: number) => number) => {
    let d = "";
    let pen = false;
    for (let i = lo; i <= hi; i += 1) {
      const v = f(i);
      if (!Number.isFinite(v)) {
        pen = false;
        continue;
      }
      d += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${py(v).toFixed(1)}`;
      pen = true;
    }
    return d;
  };
  const zPath = () => {
    let d = "";
    let pen = false;
    for (let i = lo; i <= hi; i += 1) {
      const v = p.z_series[i];
      if (v == null) {
        pen = false;
        continue;
      }
      d += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${zy(v).toFixed(1)}`;
      pen = true;
    }
    return d;
  };

  // Trades with any part inside the window, and which of them get a label:
  // two labels on top of each other are less readable than one. Entry and
  // exit labels are two separate rows and are thinned separately.
  const shown = spans.filter((s) => s.a >= 0 && s.b >= lo && s.a <= hi);
  const gap = gapInUnits(LABEL_GAP_PX, pane.w, VW);
  const labelIn = pickLabels(shown.filter((s) => s.a >= lo && s.a <= hi),
                             (s) => x(s.a), focus, gap);
  const labelOut = pickLabels(shown.filter((s) => s.b >= lo && s.b <= hi),
                              (s) => x(s.b), focus, gap);

  const legsOf = (t: Trade) =>
    t.pattern
      ? `${PATTERN_CN[t.pattern]}：${p.name_a} ${signed(t.a_ret_pct, 1, "%")}，`
        + `${p.name_b} ${signed(t.b_ret_pct, 1, "%")}`
      : "期间行情缺失";

  // Every marker carries the whole trade, because a dot on a line is not a
  // fact anyone can read.
  const story = ({ t, a, b }: Span) =>
    [
      `${t.entry} 买入 ${t.buy_name}（${t.buy_code}）`,
      t.open ? "仍持有中" : `${t.exit} 卖出`,
      `Z ${signed(t.entry_z, 2)} → ${signed(t.exit_z, 2)}`,
      a >= 0 && b >= a ? `持有 ${b - a} 个交易日` : "",
      legsOf(t),
      t.pnl_pct == null ? "" : `买入腿盈亏 ${signed(t.pnl_pct, 2, "%")}`,
    ]
      .filter(Boolean)
      .join(" · ");

  const closed = p.trades.filter((t) => !t.open);

  // The one thing tying the two panes together: the same shaded window on
  // both, so "this is the stretch" and "this is what they did in it" are
  // visibly the same days.
  const bands = (h: number) =>
    shown.map((s) => {
      const x0 = x(Math.max(s.a, lo));
      const x1 = x(Math.min(s.b, hi));
      return (
        <rect key={s.i} x={x0} y={0} width={Math.max(1, x1 - x0)} height={h}
          fill={s.t.direction === "BUY_A" ? COL_A : COL_B}
          opacity={s.i === focus ? 0.16 : 0.07} />
      );
    });

  return (
    <section className="card p-3 flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="text-[13px] font-semibold">
          <Link to={`/?t=${p.code_a}`} className="hover:text-cyan">{p.name_a}</Link>
          <span className="text-ink-mute font-normal"> / </span>
          <Link to={`/?t=${p.code_b}`} className="hover:text-cyan">{p.name_b}</Link>
        </span>
        <span className="label font-mono tnum">{p.code_a} · {p.code_b}</span>
        <span className="label">
          对冲比率 β {fixed(p.beta_now, 3)} · 半衰期{" "}
          {p.half_life >= 999 ? "不收敛" : `${fixed(p.half_life, 1)} 天`}
        </span>
        {sel && (
          <button onClick={() => setFocus(null)} className="ml-auto text-[12px] text-cyan">
            ← 看全部 {p.dates[0]} 起
          </button>
        )}
      </div>

      {/* ── what the two stocks did ─────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <Swatch colour={COL_A} name={p.name_a} code={p.code_a} v={ra(readAt)}
          note={legendNote} />
        <Swatch colour={COL_B} name={p.name_b} code={p.code_b} v={rb(readAt)}
          note={legendNote} />
        <span className="label ml-auto">
          {sel
            ? `买入 ${p.dates[sel.a]} → 卖出 ${p.dates[sel.b]}，以买入日为 0%`
            : `以 ${p.dates[lo]} 为 0% · 点击任一笔交易可放大到那一段`}
        </span>
      </div>

      <div className="relative">
        <svg viewBox={`0 0 ${VW} ${PH}`} preserveAspectRatio="none"
          className="w-full h-[165px] block" role="img"
          aria-label={`${p.name_a} 与 ${p.name_b} 的同期涨跌对比`}>
          {bands(PH)}
          {bottom < 0 && top > 0 && (
            <line x1={0} x2={VW} y1={py(0)} y2={py(0)} stroke="var(--color-line-bright)"
              vectorEffect="non-scaling-stroke" />
          )}
          <path d={legPath(ra)} fill="none" stroke={COL_A} strokeWidth={1.6}
            vectorEffect="non-scaling-stroke" />
          <path d={legPath(rb)} fill="none" stroke={COL_B} strokeWidth={1.6}
            vectorEffect="non-scaling-stroke" />
        </svg>
        {/* Zoomed in, both lines end ON the numbers the trade row reports. */}
        {sel && focus != null && (
          <>
            <Pin left={x(sel.b) / VW} top={py(ra(sel.b)) / PH} colour={COL_A}
              text={`${p.name_a} ${signed(ra(sel.b), 1, "%")}`} />
            <Pin left={x(sel.b) / VW} top={py(rb(sel.b)) / PH} colour={COL_B}
              text={`${p.name_b} ${signed(rb(sel.b), 1, "%")}`} />
          </>
        )}
      </div>

      {/* ── and what the spread did ─────────────────────────────────────── */}
      <div className="relative mt-1" ref={paneRef}>
        <svg viewBox={`0 0 ${VW} ${ZH}`} preserveAspectRatio="none"
          className="w-full h-[130px] block" role="img"
          aria-label={`${p.name_a} 与 ${p.name_b} 的价差 Z 分数`}>
          {bands(ZH)}
          {[2, -2].map((v) => (
            <line key={v} x1={0} x2={VW} y1={zy(v)} y2={zy(v)} stroke="var(--color-up)"
              strokeDasharray="5 4" strokeWidth={1} opacity={0.5}
              vectorEffect="non-scaling-stroke" />
          ))}
          <line x1={0} x2={VW} y1={zy(0)} y2={zy(0)} stroke="var(--color-line-bright)"
            vectorEffect="non-scaling-stroke" />
          <path d={zPath()} fill="none" stroke="var(--color-ink-dim)" strokeWidth={1.5}
            vectorEffect="non-scaling-stroke" />
        </svg>

        {shown.map((s) => (
          <Marker key={`in-${s.i}`} show={s.a >= lo && s.a <= hi}
            left={x(s.a) / VW} top={zy(s.t.entry_z) / ZH}
            colour={s.t.direction === "BUY_A" ? COL_A : COL_B}
            label={labelIn.has(s.i) ? `买 ${s.t.buy_name}` : null}
            // Outward from the centre line. Entries sit at |z| ≥ 2 and exits
            // at z ≈ 0, so an entry label written inward lands in the band
            // the exit labels already occupy.
            below={s.t.entry_z < 0}
            title={story(s)} active={s.i === focus}
            onClick={() => setFocus(focus === s.i ? null : s.i)} />
        ))}
        {shown.map((s) => (
          <Marker key={`out-${s.i}`} show={s.b >= lo && s.b <= hi} hollow
            left={x(s.b) / VW} top={zy(s.t.exit_z) / ZH}
            colour={s.t.direction === "BUY_A" ? COL_A : COL_B}
            label={labelOut.has(s.i)
              ? (s.t.open ? "持有中" : `卖出 ${s.t.buy_name}`) : null}
            below title={story(s)} active={s.i === focus}
            onClick={() => setFocus(focus === s.i ? null : s.i)} />
        ))}
      </div>

      {/* The caption on its own line: squeezed between the two dates it
          interleaves with them at narrow widths and reads as nonsense. */}
      <div className="flex flex-col gap-0.5 label">
        <span className="flex justify-between font-mono tnum">
          <span>{p.dates[lo]}</span>
          <span>{p.dates[hi]}</span>
        </span>
        <span>价差 Z（样本外）· ±2σ 入场，回到 0 出场 · 实心＝买入，空心＝卖出</span>
      </div>

      {p.trades.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-[12px] border-collapse">
            <thead>
              <tr className="text-ink-mute">
                <th className="text-left font-normal pb-1 pr-3">入场</th>
                <th className="text-left font-normal pb-1 pr-3">出场</th>
                <th className="text-left font-normal pb-1 pr-3">买入</th>
                <th className="text-right font-normal pb-1 px-2">Z 入→出</th>
                <th className="text-left font-normal pb-1 px-2"
                  title="同一次价差回归，可能是两只都涨、两只都跌，或真的反向 —— 完全不同的持仓体验">
                  期间 {p.name_a} / {p.name_b}
                </th>
                <th className="text-right font-normal pb-1 px-2">买入价</th>
                <th className="text-right font-normal pb-1 px-2">卖出价</th>
                <th className="text-right font-normal pb-1 pl-2">盈亏</th>
              </tr>
            </thead>
            <tbody>
              {[...spans].reverse().map(({ i, t, ...sp }) => (
                <tr key={i}
                  onClick={() => setFocus(focus === i ? null : i)}
                  title={story({ i, t, ...sp })}
                  className={`border-t border-line cursor-pointer hover:bg-sunken ${
                    i === focus ? "bg-sunken" : ""}`}>
                  <td className="py-1 pr-3 font-mono tnum">{t.entry}</td>
                  <td className="py-1 pr-3 font-mono tnum">
                    {t.open ? <span className="text-brand-ink">持有中</span> : t.exit}
                  </td>
                  <td className="py-1 pr-3 whitespace-nowrap">
                    <span className="inline-block w-1.5 h-1.5 rounded-full align-middle mr-1"
                      style={{ background: t.direction === "BUY_A" ? COL_A : COL_B }} />
                    {t.buy_name}
                    <span className="text-ink-mute font-mono text-[11px]"> {t.buy_code}</span>
                  </td>
                  <td className="py-1 px-2 text-right font-mono tnum text-ink-dim">
                    {signed(t.entry_z, 2)} → {signed(t.exit_z, 2)}
                  </td>
                  <td className="py-1 px-2 whitespace-nowrap">
                    {t.pattern ? (
                      <>
                        <span className="text-ink-mute">{PATTERN_CN[t.pattern]}</span>{" "}
                        <Ret v={t.a_ret_pct} /> <span className="text-ink-mute">/</span>{" "}
                        <Ret v={t.b_ret_pct} />
                      </>
                    ) : <span className="text-ink-mute">—</span>}
                  </td>
                  <td className="py-1 px-2 text-right font-mono tnum text-ink-dim">{fixed(t.entry_price)}</td>
                  <td className="py-1 px-2 text-right font-mono tnum text-ink-dim">{fixed(t.exit_price)}</td>
                  <td className={`py-1 pl-2 text-right font-mono tnum ${
                    (t.pnl_pct ?? 0) > 0 ? "text-up" : (t.pnl_pct ?? 0) < 0 ? "text-down" : ""}`}>
                    {signed(t.pnl_pct, 2, "%")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="label leading-snug">
        盈亏只算买入的那条腿——A 股无法做空，减持另一条腿是仓位调整而非空头。
        「期间」两栏是同一段时间内两只股票各自的涨跌：同涨时价差收敛只是买入腿更强，
        和一只涨一只跌完全不是同一件事。
        已平仓 {closed.length} 笔{closed.length > 0 && `，胜率 ${fixed(p.win_rate, 0)}%`}。
        历史统计，非预测。
      </p>
    </section>
  );
}

function Ret({ v }: { v: number | null }) {
  return (
    <span className={`font-mono tnum ${
      (v ?? 0) > 0 ? "text-up" : (v ?? 0) < 0 ? "text-down" : "text-ink-dim"}`}>
      {signed(v, 1, "%")}
    </span>
  );
}

function Swatch({ colour, name, code, v, note }: {
  colour: string; name: string; code: string; v: number; note: string;
}) {
  return (
    <span className="flex items-baseline gap-1.5 text-[12.5px]" title={note}>
      <span className="w-3 h-[3px] rounded-full self-center" style={{ background: colour }} />
      <span>{name}</span>
      <span className="font-mono tnum text-[11px] text-ink-mute">{code}</span>
      <Ret v={v} />
    </span>
  );
}

/** A label pinned to a point on the pane above. */
function Pin({ left, top, colour, text }: {
  left: number; top: number; colour: string; text: string;
}) {
  return (
    <span style={{ left: `${left * 100}%`, top: `${top * 100}%` }}
      className="absolute w-0 h-0">
      <span className="absolute left-0 top-0 -translate-x-1/2 -translate-y-1/2 w-2 h-2
        rounded-full border-2 border-panel" style={{ background: colour }} />
      <span className="absolute right-1.5 top-0 -translate-y-1/2 px-1 rounded
        bg-panel/90 border border-line text-[10.5px] leading-[15px] whitespace-nowrap">
        {text}
      </span>
    </span>
  );
}

/** A trade marker: a dot on the z-line, with what it is written beside it. */
function Marker({ show, left, top, colour, label, title, hollow, below, active, onClick }: {
  show: boolean; left: number; top: number; colour: string;
  label: string | null; title: string;
  hollow?: boolean; below?: boolean; active?: boolean; onClick: () => void;
}) {
  if (!show) return null;
  return (
    <span style={{ left: `${left * 100}%`, top: `${top * 100}%` }}
      className="absolute w-0 h-0">
      <button type="button" onClick={onClick} title={title} aria-label={title}
        className="absolute left-0 top-0 -translate-x-1/2 -translate-y-1/2
          w-4 h-4 flex items-center justify-center">
        <span className={`rounded-full ${active ? "w-3 h-3" : "w-2.5 h-2.5"}`}
          style={hollow
            ? { border: `2px solid ${colour}`, background: "var(--color-panel)" }
            : { background: colour, boxShadow: "0 0 0 2px var(--color-panel)" }} />
      </button>
      {label && (
        <button type="button" onClick={onClick} title={title}
          style={below ? { top: 10 } : { bottom: 8 }}
          className={`absolute left-1/2 -translate-x-1/2 px-1 rounded border border-line
            bg-panel/90 text-[10.5px] leading-[15px] whitespace-nowrap
            ${active ? "font-semibold" : "text-ink-dim"}`}>
          {label}
        </button>
      )}
    </span>
  );
}
