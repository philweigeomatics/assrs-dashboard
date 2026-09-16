/**
 * 量化对比 — the numbers behind "why did that one go up so much more".
 *
 * Reading order is deliberate, because it is the order the question actually
 * resolves in:
 *
 *   1. 归因 — the gap split into a market-risk factor and an everything-else
 *      factor. These MULTIPLY to the full outperformance, exactly. If beta is
 *      ~0 the winner was not simply the more aggressive stock, and everyone's
 *      first instinct was wrong.
 *   2. 估值 — did the price rise because the multiple rose, or because
 *      earnings did? A gain made of re-rating is borrowed from the future;
 *      one made of earnings is not.
 *   3. 风险 — Sharpe, drawdown and capture say what the extra return cost.
 *   4. 何时拉开 — monthly relative bars, because "it doubled" and "it doubled
 *      in one month on one announcement" are different stocks.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { Num, PairStats, StockProfile } from "../lib/types";
import { fixed, moveClass, signed } from "../lib/format";

const WINDOWS: { id: string; label: string }[] = [
  { id: "60", label: "3个月" },
  { id: "120", label: "6个月" },
  { id: "252", label: "1年" },
  { id: "all", label: "全部" },
];

const A_COLOR = "var(--color-cyan)";
const B_COLOR = "#7c3aed";

export function CompareStats({ ticker, other }: { ticker: string; other: string }) {
  const [window, setWindow] = useState("252");
  const q = useQuery({
    queryKey: ["compare-stats", ticker, other, window],
    queryFn: () => api.compareStats(ticker, other, window),
    staleTime: 20 * 60_000,
  });

  return (
    <section className="card p-3 flex flex-col gap-3">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="text-[14px] font-semibold">📐 量化对比</h2>
        {q.data && (
          <span className="label">
            {q.data.from} → {q.data.to} · {q.data.bars} 个共同交易日
          </span>
        )}
        <div className="ml-auto flex items-center gap-1">
          {WINDOWS.map((w) => (
            <button key={w.id} onClick={() => setWindow(w.id)}
              className={`h-7 px-2 rounded-md text-[12.5px] border transition-colors ${
                w.id === window ? "border-cyan bg-cyan text-white" : "border-line bg-panel hover:bg-elevated"
              }`}>
              {w.label}
            </button>
          ))}
        </div>
      </div>

      {q.isPending && <p className="label py-6 text-center">正在计算…</p>}
      {q.isError && (
        <p className="text-[12.5px] text-up py-4 text-center">
          {(q.error as Error).message}
          <button onClick={() => q.refetch()} className="ml-2 text-cyan">重试</button>
        </p>
      )}
      {q.data && <Body d={q.data} />}
    </section>
  );
}

function Body({ d }: { d: PairStats }) {
  return (
    <>
      <Attribution d={d} />
      <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] items-start">
        <Metrics d={d} />
        <div className="min-w-0 flex flex-col gap-3">
          <Valuations d={d} />
          <Timing d={d} />
        </div>
      </div>
      <p className="label leading-snug">
        β/α 对 沪深300 回归（对数收益），无风险利率取 0。α 为年化。
        相关性、跟踪误差、信息比率为两只股票之间。历史统计，非预测。
      </p>
    </>
  );
}

/** The headline: the gap, split into two factors that multiply back to it. */
function Attribution({ d }: { d: PairStats }) {
  const at = d.attribution;
  const { a, b } = d;
  if (!at) {
    return (
      <div className="rounded-lg bg-sunken px-3 py-2 text-[12.5px]">
        <b>{a.label}</b> {signed(a.total_return_pct, 1, "%")} vs{" "}
        <b>{b.label}</b> {signed(b.total_return_pct, 1, "%")}
        <span className="text-ink-mute"> — 缺少沪深300数据，无法拆分 β / α。</span>
      </div>
    );
  }

  const ahead = (at.gap_ratio ?? 1) >= 1;
  const lead = ahead ? a.label : b.label;
  const lag = ahead ? b.label : a.label;
  // Stated from the winner's side whichever way round it is, so the sentence
  // never reads "A led by 0.4x".
  const ratio = ahead ? (at.gap_ratio ?? 1) : 1 / (at.gap_ratio ?? 1);
  const betaMatters = Math.abs(at.beta_factor_pct ?? 0) > 5;

  return (
    <div className="rounded-lg bg-sunken px-3 py-2 flex flex-col gap-1.5">
      <p className="text-[13px]">
        <b className="text-up">{lead}</b> 领先 <b>{lag}</b>{" "}
        <b className="font-mono tnum">{ratio.toFixed(2)}×</b>
        <span className="text-ink-mute">
          （{signed(at.gap_pct, 0, "pp")}；同期沪深300 {signed(at.market_return_pct, 1, "%")}）
        </span>
      </p>

      <div className="flex flex-wrap gap-2">
        <Factor label="β 市场风险" pct={at.beta_factor_pct}
          note={`β ${fixed(at.beta_a)} vs ${fixed(at.beta_b)}`} />
        <span className="self-center text-ink-mute text-[13px]">×</span>
        <Factor label="α 个股自身" pct={at.alpha_factor_pct}
          note={`年化 α ${signed(at.alpha_a_pct, 0, "%")} vs ${signed(at.alpha_b_pct, 0, "%")}`} />
      </div>

      <p className="text-[12px] text-ink-dim leading-snug">
        {betaMatters
          ? `两者贝塔不同，在这段行情里贡献了 ${signed(at.beta_factor_pct, 0, "%")} 的差距；其余来自个股自身。`
          : `两只股票的贝塔几乎一致（${fixed(at.beta_a)} vs ${fixed(at.beta_b)}），市场本身也几乎没动，所以这个差距与"谁更激进"无关——全部来自个股自身。`}
        {" "}两项相乘即为全部差距（残差 {fixed(at.residual_pct, 2)}%）。
      </p>
    </div>
  );
}

function Factor({ label, pct, note }: { label: string; pct: Num; note: string }) {
  return (
    <div className="rounded-md bg-panel px-2 py-1 border border-line min-w-[128px]">
      <div className="label">{label}</div>
      <div className={`font-mono tnum text-[15px] font-semibold ${moveClass(pct)}`}>
        {signed(pct, 1, "%")}
      </div>
      <div className="text-[11px] text-ink-mute">{note}</div>
    </div>
  );
}

const ROWS: { k: keyof StockProfile; label: string; nd?: number; suffix?: string; hint?: string }[] = [
  { k: "total_return_pct", label: "区间涨幅", nd: 1, suffix: "%" },
  { k: "cagr_pct", label: "年化收益", nd: 1, suffix: "%" },
  { k: "vol_annual_pct", label: "年化波动", nd: 1, suffix: "%", hint: "越高越颠簸" },
  { k: "sharpe", label: "夏普比率", nd: 2, hint: "每承担一单位波动换来的收益" },
  { k: "max_drawdown_pct", label: "最大回撤", nd: 1, suffix: "%" },
  { k: "beta", label: "β（对沪深300）", nd: 2, hint: "市场每涨1%，它涨多少" },
  { k: "alpha_annual_pct", label: "α 年化", nd: 1, suffix: "%", hint: "市场解释不了的部分" },
  { k: "r2", label: "R²", nd: 2, hint: "涨跌被大盘解释的比例" },
  { k: "up_capture_pct", label: "上涨捕获", nd: 0, suffix: "%", hint: "大盘涨时它吃到多少" },
  { k: "down_capture_pct", label: "下跌捕获", nd: 0, suffix: "%", hint: "大盘跌时它跟跌多少，越低越好" },
  { k: "positive_days_pct", label: "上涨天数占比", nd: 0, suffix: "%" },
];

function Metrics({ d }: { d: PairStats }) {
  const { a, b, pair } = d;
  return (
    <div className="min-w-0">
      <div className="grid grid-cols-[1fr_auto_auto] gap-x-2 text-[12px]">
        <span className="label">指标</span>
        <span className="label text-right font-medium" style={{ color: A_COLOR }}>{a.label}</span>
        <span className="label text-right font-medium" style={{ color: B_COLOR }}>{b.label}</span>

        {ROWS.map((r) => {
          const va = a[r.k] as Num, vb = b[r.k] as Num;
          if (va == null && vb == null) return null;
          // Lower is better for these three, so the highlight must flip.
          const lowerWins = r.k === "vol_annual_pct" || r.k === "max_drawdown_pct"
            || r.k === "down_capture_pct";
          const better = va == null || vb == null ? null
            : lowerWins
              ? (Math.abs(va) < Math.abs(vb) ? "a" : "b")
              : (va > vb ? "a" : "b");
          return (
            <Row key={String(r.k)} label={r.label} hint={r.hint}
              a={va} b={vb} nd={r.nd ?? 2} suffix={r.suffix ?? ""} better={better} />
          );
        })}
      </div>

      <div className="mt-2 pt-2 border-t border-line grid grid-cols-2 gap-x-3 gap-y-1 text-[12px]">
        <Pair label="两者相关性" v={fixed(pair.correlation, 3)}
          hint="1 = 完全同步。同板块通常很高" />
        <Pair label={`${a.label} 对 ${b.label} 的 β`} v={fixed(pair.beta_a_on_b)}
          hint="B 每涨1%，A 涨多少——配对交易的对冲比例" />
        <Pair label="跟踪误差" v={`${fixed(pair.tracking_error_pct, 1)}%`}
          hint="两者收益率之差的年化波动" />
        <Pair label="信息比率" v={fixed(pair.information_ratio)}
          hint="超额收益 ÷ 跟踪误差。>1 说明领先是稳定的，不是一两天拉开的" />
      </div>
    </div>
  );
}

function Row({ label, hint, a, b, nd, suffix, better }: {
  label: string; hint?: string; a: Num; b: Num; nd: number; suffix: string;
  better: "a" | "b" | null;
}) {
  const cell = (v: Num, mine: "a" | "b") => (
    <span className={`text-right font-mono tnum ${better === mine ? "font-semibold" : "text-ink-dim"}`}>
      {v == null ? "—" : `${fixed(v, nd)}${suffix}`}
    </span>
  );
  return (
    <>
      <span className="text-ink-dim truncate" title={hint}>
        {label}{hint && <span className="text-ink-mute"> ⓘ</span>}
      </span>
      {cell(a, "a")}
      {cell(b, "b")}
    </>
  );
}

function Pair({ label, v, hint }: { label: string; v: string; hint: string }) {
  return (
    <div className="flex items-baseline justify-between gap-1" title={hint}>
      <span className="text-ink-dim truncate">{label}</span>
      <span className="font-mono tnum">{v}</span>
    </div>
  );
}

/** Re-rating vs earnings — the part that says whether the gain is borrowed. */
function Valuations({ d }: { d: PairStats }) {
  const both = [d.a, d.b];
  if (both.every((s) => !s.valuation)) return null;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline gap-2">
        <span className="text-[12.5px] font-medium">估值拆解</span>
        <span className="label">涨幅 = 估值变化 × 盈利变化</span>
      </div>
      {both.map((s, i) => {
        const v = s.valuation;
        if (!v) {
          return (
            <p key={s.label} className="text-[11.5px] text-ink-mute">
              {s.label}：PE 缺失或为负（亏损），无法拆解。
            </p>
          );
        }
        const re = v.rerating_pct ?? 0, ea = v.earnings_pct ?? 0;
        const total = Math.abs(re) + Math.abs(ea) || 1;
        const driver = Math.abs(ea) > Math.abs(re) ? "盈利驱动" : "估值驱动";
        return (
          <div key={s.label} className="flex flex-col gap-0.5">
            <div className="flex items-baseline justify-between text-[12px]">
              <span className="font-medium" style={{ color: i === 0 ? A_COLOR : B_COLOR }}>
                {s.label}
              </span>
              <span className="label">
                PE {fixed(v.pe_start, 1)} → {fixed(v.pe_end, 1)} · {driver}
              </span>
            </div>
            <div className="flex h-4 rounded-sm overflow-hidden bg-sunken">
              <span title={`估值变化 ${signed(re, 1, "%")}`}
                style={{ width: `${(Math.abs(re) / total) * 100}%`, background: "#f59e0b" }} />
              <span title={`盈利变化 ${signed(ea, 1, "%")}`}
                style={{ width: `${(Math.abs(ea) / total) * 100}%`, background: "var(--color-up)" }} />
            </div>
            <div className="flex justify-between text-[11px]">
              <span style={{ color: "#b45309" }}>估值 {signed(re, 0, "%")}</span>
              <span className={moveClass(ea)}>盈利 {signed(ea, 0, "%")}</span>
            </div>
          </div>
        );
      })}
      <p className="label leading-snug">
        盈利为 价格 ÷ PE(TTM) 推算的滚动每股收益。靠估值抬升的涨幅是向未来借的，
        靠盈利增长的不是。
      </p>
    </div>
  );
}

/** When the gap opened: relative performance, month by month. */
function Timing({ d }: { d: PairStats }) {
  const months = d.pair.monthly;
  if (months.length < 2) return null;
  const peak = Math.max(...months.map((m) => Math.abs(m.rel_pct ?? 0)), 1);
  const biggest = months.reduce((x, m) =>
    Math.abs(m.rel_pct ?? 0) > Math.abs(x.rel_pct ?? 0) ? m : x);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline gap-2">
        <span className="text-[12.5px] font-medium">何时拉开</span>
        <span className="label">{d.a.label} 相对 {d.b.label}，逐月</span>
      </div>
      <div className="flex items-end gap-[2px] h-16">
        {months.map((m) => {
          const v = m.rel_pct ?? 0;
          const h = (Math.abs(v) / peak) * 50;
          return (
            <div key={m.month} title={`${m.month}  ${signed(v, 1, "%")}`}
              className="flex-1 flex flex-col justify-center items-stretch h-full min-w-[3px]">
              <div className="flex-1 flex flex-col justify-end">
                {v > 0 && <span style={{ height: `${h}%`, background: "var(--color-up)" }} />}
              </div>
              <div className="flex-1">
                {v < 0 && <span className="block" style={{ height: `${h}%`, background: "var(--color-down)" }} />}
              </div>
            </div>
          );
        })}
      </div>
      <p className="label">
        最大单月差距 {biggest.month} · {signed(biggest.rel_pct, 1, "%")}
      </p>
    </div>
  );
}
