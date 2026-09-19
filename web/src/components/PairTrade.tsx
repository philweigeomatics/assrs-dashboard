/**
 * 配对交易 — two stocks that usually move together, and the days they do not.
 *
 * Nothing is pre-filled. You pick every code yourself, the same way the
 * analysis page picks one, and A-shares only: the signal is read buy-only
 * because shorting A-shares is restricted, and that reading does not transfer
 * to a market where you could short the rich leg.
 *
 * The spread is walk-forward — each day's hedge ratio was fitted on the days
 * BEFORE it — so the statistics and the trade list are an assessment rather
 * than a description of the past. That is the only reason the win rate below
 * is worth reading at all.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { PairResult, PairTradeResult, StockRef } from "../lib/types";
import { useSymbolSearch } from "../lib/useSymbolSearch";
import { usePersistentState } from "../lib/usePersistentState";
import { Discover } from "./Discover";
import { fixed, signed } from "../lib/format";

const MAX = 10;

export function PairTrade() {
  // Empty by default and persisted per user — never a seeded example basket.
  const [picked, setPicked] = usePersistentState<StockRef[]>("assrs.pairs.v1", []);
  const [zWindow, setZWindow] = useState(60);
  const [olsWindow, setOlsWindow] = useState(252);

  const stocks = useQuery({ queryKey: ["stocks"], queryFn: api.stocks, staleTime: 6 * 3600_000 });
  const run = useMutation({
    mutationFn: () => api.pairTrade(picked.map((p) => p.t), zWindow, olsWindow),
  });

  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const { items } = useSymbolSearch({
    query: q, stocks: stocks.data ?? [], markets: ["CN"], limit: 8,
  });
  const chosen = new Set(picked.map((p) => p.t));
  const hits = items.filter((s) => !chosen.has(s.t));

  return (
    <>
      <section className="card p-3 flex flex-col gap-2">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h2 className="text-[14px] font-semibold">🔗 配对交易</h2>
          <span className="label">仅 A 股 · 自选 2–{MAX} 只，逐对检验协整</span>
          <div className="ml-auto flex items-center gap-2">
            <label className="label flex items-center gap-1">
              Z窗口
              <input type="number" value={zWindow} min={20} max={120} step={10}
                onChange={(e) => setZWindow(Number(e.target.value))}
                className="w-16 h-7 px-1.5 rounded-md bg-sunken text-[12.5px] font-mono tnum outline-none" />
            </label>
            <label className="label flex items-center gap-1" title="每日对冲比率由之前这么多天估计">
              OLS窗口
              <input type="number" value={olsWindow} min={60} max={504} step={21}
                onChange={(e) => setOlsWindow(Number(e.target.value))}
                className="w-16 h-7 px-1.5 rounded-md bg-sunken text-[12.5px] font-mono tnum outline-none" />
            </label>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {picked.map((s) => (
            <span key={s.t} className="h-7 pl-2 pr-1 rounded-md bg-sunken flex items-center gap-1.5 text-[12.5px]">
              <span className="truncate max-w-[130px]">{s.n}</span>
              <span className="font-mono tnum text-[11px] text-ink-mute">{s.t}</span>
              <button onClick={() => setPicked(picked.filter((p) => p.t !== s.t))}
                aria-label={`移除 ${s.n}`} className="text-ink-mute hover:text-ink px-1">✕</button>
            </span>
          ))}
          {picked.length < MAX && (
            <div className="relative">
              <input value={q} onChange={(e) => { setQ(e.target.value); setOpen(true); }}
                onFocus={() => setOpen(true)}
                onBlur={() => globalThis.setTimeout(() => setOpen(false), 150)}
                placeholder={picked.length ? "再加一只…" : "输入代码或名称添加…"}
                className="h-7 w-48 px-2 rounded-md bg-sunken text-[12.5px] outline-none focus:ring-2 focus:ring-cyan/40" />
              {open && hits.length > 0 && (
                <div className="card absolute left-0 mt-1 z-30 w-64 py-1">
                  {hits.map((s) => (
                    <button key={s.t} onMouseDown={(e) => e.preventDefault()}
                      onClick={() => { setPicked([...picked, { t: s.t, n: s.n }]); setQ(""); setOpen(false); }}
                      className="w-full flex items-baseline gap-2 px-2 py-1 text-left hover:bg-elevated">
                      <span className="font-mono tnum text-[12px] text-ink-dim shrink-0">{s.t}</span>
                      <span className="text-[12.5px] truncate">{s.n}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          {picked.length > 0 && (
            <button onClick={() => setPicked([])} className="text-[12px] text-cyan ml-1">清空</button>
          )}
          <button onClick={() => run.mutate()} disabled={picked.length < 2 || run.isPending}
            className="ml-auto h-8 px-3 rounded-lg bg-cyan text-white text-[13px] font-semibold disabled:opacity-60">
            {run.isPending ? "检验中…" : `检验 ${Math.max(0, (picked.length * (picked.length - 1)) / 2)} 个组合`}
          </button>
        </div>

        <p className="label leading-snug max-w-[78ch]">
          每日对冲比率只用之前 {olsWindow} 天估计（前推一天），所以价差、统计检验和下面的
          历史交易都是样本外的。A 股不能做空，信号一律解读为「买入便宜的那条腿、减持贵的那条」。
        </p>
      </section>

      {run.isError && (
        <div className="card p-4 text-center text-up text-[13px]">
          {(run.error as Error).message}
        </div>
      )}
      {run.isPending && (
        <div className="card p-8 flex flex-col items-center gap-2 text-ink-mute">
          <div className="h-6 w-6 rounded-full border-2 border-line border-t-cyan animate-spin" />
          <div className="label">滚动回归 + 协整检验，组合越多越慢</div>
        </div>
      )}
      {/* Which two of eighty stocks are cointegrated is not something anyone
          knows in advance, so the search sits alongside the manual picker and
          loads its findings into it. */}
      <Discover kind="pair-trade" onUse={(a, b, [na, nb]) => {
        setPicked([{ t: a, n: na }, { t: b, n: nb }]);
        globalThis.scrollTo({ top: 0, behavior: "smooth" });
      }} />

      {run.data && <Results d={run.data} />}
      {!run.data && !run.isPending && picked.length < 2 && (
        <div className="card p-10 text-center label">
          自己选两只，或用下方的「从自选股中搜索」把配对找出来
        </div>
      )}
    </>
  );
}

function Results({ d }: { d: PairTradeResult }) {
  const [openPair, setOpenPair] = useState<string | null>(null);
  return (
    <>
      <section className="card p-3 overflow-x-auto">
        <div className="flex items-baseline gap-3 mb-2">
          <span className="text-[13px] font-semibold">{d.pairs.length} 个组合</span>
          <span className="label">
            {d.from} → {d.to} · {d.bars} 个共同交易日 · Z窗口 {d.z_window} · OLS {d.ols_window}
          </span>
        </div>
        <table className="w-full text-[12.5px] border-collapse">
          <thead>
            <tr className="text-ink-mute">
              <th className="text-left font-normal pb-1 pr-3">组合</th>
              <th className="text-right font-normal pb-1 px-2">评分</th>
              <th className="text-left font-normal pb-1 px-2">信号</th>
              <th className="text-right font-normal pb-1 px-2" title="Engle-Granger 协整检验 p 值，< 0.10 通过">协整 p</th>
              <th className="text-right font-normal pb-1 px-2" title="价差平稳性 ADF p 值，< 0.10 通过">ADF p</th>
              <th className="text-right font-normal pb-1 px-2" title="< 0.45 为均值回归；此估计量在短样本上偏高">Hurst</th>
              <th className="text-right font-normal pb-1 px-2" title="价差回到一半所需天数，5–30 天最可交易">半衰期</th>
              <th className="text-right font-normal pb-1 px-2">相关性</th>
              <th className="text-right font-normal pb-1 px-2" title="当前价差偏离，|Z| ≥ 2 触发">Z</th>
              <th className="text-right font-normal pb-1 px-2" title="样本外历史交易的胜率与平均盈亏">历史</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {d.pairs.map((p) => {
              const key = `${p.code_a}/${p.code_b}`;
              const live = p.signal === "BUY_A" || p.signal === "BUY_B";
              return (
                <tr key={key} className="border-t border-line">
                  <td className="py-1 pr-3 whitespace-nowrap">
                    <Link to={`/?t=${p.code_a}`} className="hover:text-cyan">{p.name_a}</Link>
                    <span className="text-ink-mute"> / </span>
                    <Link to={`/?t=${p.code_b}`} className="hover:text-cyan">{p.name_b}</Link>
                  </td>
                  <td className="py-1 px-2 text-right font-mono tnum font-semibold">{fixed(p.score, 2)}</td>
                  <td className={`py-1 px-2 whitespace-nowrap ${live ? "text-up font-medium" : "text-ink-mute"}`}>
                    {live ? `买 ${p.buy === p.code_a ? p.name_a : p.name_b}` : p.signal_cn}
                  </td>
                  <Stat v={p.eg_p} nd={3} ok={p.coint_ok} />
                  <Stat v={p.adf_p} nd={3} ok={p.adf_ok} />
                  <Stat v={p.hurst} nd={2} ok={p.hurst_ok} />
                  <Stat v={p.half_life >= 999 ? null : p.half_life} nd={1} ok={p.hl_ok} />
                  <Stat v={p.corr} nd={2} ok={p.corr > 0.6} />
                  <td className={`py-1 px-2 text-right font-mono tnum ${live ? "font-semibold" : "text-ink-dim"}`}>
                    {signed(p.z_now, 2)}
                  </td>
                  <td className="py-1 px-2 text-right font-mono tnum text-ink-dim whitespace-nowrap">
                    {p.closed === 0 ? "—"
                      : `${fixed(p.win_rate, 0)}% · ${signed(p.avg_pnl_pct, 1, "%")}`}
                  </td>
                  <td className="py-1 pl-2 text-right">
                    <button onClick={() => setOpenPair(openPair === key ? null : key)}
                      className="text-[12px] text-cyan whitespace-nowrap">
                      {openPair === key ? "收起" : "价差图"}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {d.skipped.length > 0 && (
          <p className="label mt-2">
            {d.skipped.length} 个组合无法检验：{d.skipped.map((s) => `${s.code_a}/${s.code_b}`).join("、")}
          </p>
        )}
      </section>

      {d.pairs.filter((p) => `${p.code_a}/${p.code_b}` === openPair).map((p) => (
        <Detail key={`${p.code_a}/${p.code_b}`} p={p} />
      ))}
    </>
  );
}

function Stat({ v, nd, ok }: { v: number | null; nd: number; ok: boolean }) {
  return (
    <td className={`py-1 px-2 text-right font-mono tnum ${ok ? "text-up" : "text-ink-dim"}`}>
      {v == null ? "—" : fixed(v, nd)}
    </td>
  );
}

const VW = 900;
const VH = 150;

function Detail({ p }: { p: PairResult }) {
  const n = p.dates.length;
  const zs = p.z_series;
  const x = (i: number) => (n < 2 ? 0 : (i / (n - 1)) * VW);
  // ±3.2σ keeps the ±2 entry lines well inside the frame.
  const y = (z: number) => VH / 2 - (Math.max(-3.2, Math.min(3.2, z)) / 3.2) * (VH / 2 - 6);

  let path = "";
  let pen = false;
  zs.forEach((v, i) => {
    if (v == null) { pen = false; return; }
    path += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`;
    pen = true;
  });

  const closed = p.trades.filter((t) => !t.open);
  return (
    <section className="card p-3 flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline gap-x-3">
        <span className="text-[13px] font-semibold">{p.name_a} / {p.name_b}</span>
        <span className="label">
          价差 Z 分数（样本外）· 对冲比率 β {fixed(p.beta_now, 3)} · 半衰期{" "}
          {p.half_life >= 999 ? "不收敛" : `${fixed(p.half_life, 1)} 天`}
        </span>
      </div>

      <svg viewBox={`0 0 ${VW} ${VH}`} preserveAspectRatio="none"
        className="w-full h-[150px] block" role="img"
        aria-label={`${p.name_a} 与 ${p.name_b} 的价差 Z 分数`}>
        {[2, -2].map((v) => (
          <line key={v} x1={0} x2={VW} y1={y(v)} y2={y(v)} stroke="var(--color-up)"
            strokeDasharray="5 4" strokeWidth={1} opacity={0.5} vectorEffect="non-scaling-stroke" />
        ))}
        <line x1={0} x2={VW} y1={y(0)} y2={y(0)} stroke="var(--color-line-bright)"
          vectorEffect="non-scaling-stroke" />
        {/* Entry markers, so the trade list and the curve agree visually. */}
        {p.trades.map((t, i) => {
          const at = p.dates.indexOf(t.entry);
          if (at < 0) return null;
          return <circle key={i} cx={x(at)} cy={y(t.entry_z)} r={3}
            fill={t.direction === "BUY_A" ? "var(--color-cyan)" : "#7c3aed"} />;
        })}
        <path d={path} fill="none" stroke="var(--color-ink-dim)" strokeWidth={1.5}
          vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="flex justify-between label">
        <span>{p.dates[0]}</span>
        <span>±2σ 为入场线，回到 0 为出场</span>
        <span>{p.dates[n - 1]}</span>
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
                <th className="text-right font-normal pb-1 px-2">买入价</th>
                <th className="text-right font-normal pb-1 px-2">卖出价</th>
                <th className="text-right font-normal pb-1 pl-2">盈亏</th>
              </tr>
            </thead>
            <tbody>
              {[...p.trades].reverse().map((t, i) => (
                <tr key={i} className="border-t border-line">
                  <td className="py-1 pr-3 font-mono tnum">{t.entry}</td>
                  <td className="py-1 pr-3 font-mono tnum">
                    {t.open ? <span className="text-brand-ink">持有中</span> : t.exit}
                  </td>
                  <td className="py-1 pr-3">{t.buy_code}</td>
                  <td className="py-1 px-2 text-right font-mono tnum text-ink-dim">
                    {signed(t.entry_z, 2)} → {signed(t.exit_z, 2)}
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
        已平仓 {closed.length} 笔{closed.length > 0 && `，胜率 ${fixed(p.win_rate, 0)}%`}。
        历史统计，非预测。
      </p>
    </section>
  );
}
