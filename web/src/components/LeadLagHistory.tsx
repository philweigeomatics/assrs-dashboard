/**
 * 跟随分析 — on the days one moved, did the other follow?
 *
 * Same direction only, and measured in the follower's own volatility. Both
 * of those replace a correlation reading that got them wrong: a correlation
 * treats an inverse move as a relationship of equal standing, and it treats
 * a 1% move as a response from a stock whose ordinary day is 3%.
 *
 * The output is dated events rather than an average. An average over two
 * years cannot tell you whether the behaviour is still there; twenty dated
 * rows can.
 */

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { ApiError } from "../lib/api";
import type { FollowThrough } from "../lib/types";
import { fixed, signed } from "../lib/format";

/** Sigma at which a bar is drawn full width. */
const SCALE = 1.6;

export function LeadLagHistory({ a, b, nameA, nameB }: {
  a: string; b: string; nameA: string; nameB: string;
}) {
  const [days, setDays] = useState(504);
  const [threshold, setThreshold] = useState(1.5);
  const [flip, setFlip] = useState(false);
  const lead = flip ? b : a;
  const follow = flip ? a : b;
  const leadName = flip ? nameB : nameA;
  const followName = flip ? nameA : nameB;

  const run = useMutation({
    mutationFn: () => api.followThrough({
      a: lead, b: follow, lookback_days: days, threshold,
    }),
  });
  const d = run.data;

  return (
    <div className="flex flex-col gap-2.5 border-t border-line pt-3">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="text-[13.5px] font-semibold">🔁 跟随分析</h3>
        <span className="label">
          {leadName} 大涨大跌的那些天，{followName} 之后跟不跟
        </span>
        <div className="ml-auto flex items-center gap-2">
          <button onClick={() => { setFlip(!flip); run.reset(); }}
            className="h-7 px-2 rounded-md bg-sunken text-[12px]">⇄ 换方向</button>
          <label className="label flex items-center gap-1">
            回看
            <select value={days} onChange={(e) => setDays(Number(e.target.value))}
              className="h-7 px-1 rounded-md bg-sunken text-[12.5px] tnum outline-none">
              {[252, 504, 756].map((v) => <option key={v} value={v}>{v} 天</option>)}
            </select>
          </label>
          <label className="label flex items-center gap-1"
            title="多大的一天才算「动了」，按这只股票自己的波动衡量">
            起点
            <select value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="h-7 px-1 rounded-md bg-sunken text-[12.5px] tnum outline-none">
              {[1.2, 1.5, 2.0, 2.5].map((v) =>
                <option key={v} value={v}>{v.toFixed(1)}σ</option>)}
            </select>
          </label>
          <button onClick={() => run.mutate()} disabled={run.isPending}
            className="h-8 px-3 rounded-lg bg-cyan text-white text-[13px] font-semibold disabled:opacity-60">
            {run.isPending ? "计算中…" : d ? "重新计算" : "分析"}
          </button>
        </div>
      </div>

      {run.isError && (
        <p className="text-[12.5px] text-up">{(run.error as ApiError).message}</p>
      )}
      {d && <Report d={d} leadName={leadName} followName={followName} />}
      {!d && !run.isPending && (
        <p className="label py-3 text-center">
          点「分析」——只看同向跟随，幅度按 {followName} 自己的波动折算
        </p>
      )}
    </div>
  );
}

function Report({ d, leadName, followName }: {
  d: FollowThrough; leadName: string; followName: string;
}) {
  const byLag = Object.fromEntries(d.null.map((n) => [n.lag, n]));
  const sameDay = d.lags.find((l) => l.lag === 0);
  const best = d.best_lag ? d.lags.find((l) => l.lag === d.best_lag) : null;

  return (
    <>
      <div className="rounded-lg bg-sunken px-2.5 py-2 flex flex-col gap-1">
        <p className="text-[12.5px] leading-snug">
          {!d.enough ? (
            <>只有 <b>{d.events.length}</b> 次事件，太少，不足以下判断。
              可以把「起点」调低，或拉长回看天数。</>
          ) : best ? (
            <><b>{leadName} 领先 {d.best_lag} 天</b>：它大动之后第 {d.best_lag} 天，
              {followName} 平均同向走 <b>{signed(best.mean, 2)}σ</b>，
              {d.events.length} 次里 <b>{((best.hit ?? 0) * 100).toFixed(0)}%</b> 跟了上来
              （随机轮转只有 {signed(byLag[d.best_lag ?? 0]?.mean_hi, 2)}σ）。</>
          ) : (
            <><b>没有可用的时间差。</b>
              {sameDay && (sameDay.mean ?? 0) >= 0.5
                ? <> 两只是<b>同一天一起动</b>的（当天 {signed(sameDay.mean, 2)}σ），
                    之后几天就不再跟了 —— 想抢跑没有窗口。</>
                : <> {leadName} 大动之后，{followName} 并没有明显跟随。</>}
            </>
          )}
        </p>
        <p className="label">
          {d.from} → {d.to} · {leadName} 有 {d.events.length} 天动了超过 {d.threshold}σ
          · 幅度按 {followName} 自己前 {d.window} 天的波动折算
        </p>
      </div>

      <div className="flex flex-col gap-1">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px]">
          <span className="text-ink-mute">第几天的同向跟随（{followName} 的 σ）</span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-2 rounded-sm bg-cyan" />实际
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-2 rounded-sm bg-ink-mute/50" />随机轮转
          </span>
          <span className="text-ink-mute">虚线＝{d.follow}σ，算「跟上了」的门槛</span>
        </div>
        {d.lags.map((l) => (
          <LagBar key={l.lag} row={l} nul={byLag[l.lag]} follow={d.follow}
            best={l.lag === d.best_lag} />
        ))}
      </div>

      {d.events.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-[12px] border-collapse">
            <thead>
              <tr className="text-ink-mute">
                <th className="text-left font-normal pb-1 pr-3">{leadName} 大动那天</th>
                <th className="text-right font-normal pb-1 px-2">涨跌</th>
                <th className="text-right font-normal pb-1 px-2"
                  title="以它自己前 60 天的波动衡量">幅度</th>
                {d.lags.filter((l) => l.lag > 0).map((l) => (
                  <th key={l.lag} className="text-right font-normal pb-1 px-2">
                    +{l.lag}天
                  </th>
                ))}
                <th className="text-right font-normal pb-1 pl-2"
                  title="之后几天累计的同向幅度">累计</th>
              </tr>
            </thead>
            <tbody>
              {d.events.slice(0, 25).map((e, i) => (
                <tr key={i} className="border-t border-line">
                  <td className="py-1 pr-3 font-mono tnum whitespace-nowrap">
                    {e.date}
                    <span className={`ml-1.5 ${e.dir === "up" ? "text-up" : "text-down"}`}>
                      {e.dir === "up" ? "↑" : "↓"}
                    </span>
                  </td>
                  <td className={`py-1 px-2 text-right font-mono tnum ${
                    e.a_ret > 0 ? "text-up" : "text-down"}`}>
                    {signed(e.a_ret, 1, "%")}
                  </td>
                  <td className="py-1 px-2 text-right font-mono tnum text-ink-dim">
                    {fixed(Math.abs(e.a_z), 1)}σ
                  </td>
                  {d.lags.filter((l) => l.lag > 0).map((l) => (
                    <Cell key={l.lag} v={e.resp[l.lag] ?? null} follow={d.follow} />
                  ))}
                  <Cell v={e.cum[d.maxlag] ?? null} follow={d.follow} bold />
                </tr>
              ))}
            </tbody>
          </table>
          <p className="label mt-1.5 leading-snug">
            表里的数字是 {followName} 当天走了多少个自己的 σ，<b>正数＝和 {leadName} 同向</b>。
            所以 −1% 对一只平时 ±3% 的股票只有 −0.3σ，算不上跟随。
            {d.events.length > 25 && ` 只列出最近 25 次，共 ${d.events.length} 次。`}
          </p>
        </div>
      )}
    </>
  );
}

function LagBar({ row, nul, follow, best }: {
  row: FollowThrough["lags"][number];
  nul?: FollowThrough["null"][number];
  follow: number; best: boolean;
}) {
  const pct = (v: number) => `${Math.min(100, (Math.abs(v) / SCALE) * 100)}%`;
  const mean = row.mean ?? 0;
  return (
    <div className="flex items-center gap-2 text-[11.5px]">
      <span className={`w-16 shrink-0 tnum text-right ${
        best ? "font-semibold" : "text-ink-mute"}`}>
        {row.lag === 0 ? "当天" : `第 ${row.lag} 天`}
      </span>
      <div className="flex-1 relative h-5 rounded-sm bg-panel overflow-hidden">
        {/* The follow threshold, so a bar can be judged without arithmetic. */}
        <div className="absolute top-0 bottom-0 border-l border-dashed border-line-bright"
          style={{ left: pct(follow) }} />
        <div className={`absolute top-0.5 h-2 rounded-sm ${
          mean >= 0 ? "bg-cyan" : "bg-down"}`} style={{ width: pct(mean) }} />
        {nul && (
          <div className="absolute bottom-0.5 h-1.5 rounded-sm bg-ink-mute/50"
            style={{ width: pct(nul.mean_hi) }} />
        )}
      </div>
      <span className={`w-12 shrink-0 tnum text-right ${
        best ? "font-semibold text-up" : ""}`}>{signed(row.mean, 2)}σ</span>
      <span className="w-10 shrink-0 tnum text-right text-ink-mute">
        {row.hit == null ? "—" : `${(row.hit * 100).toFixed(0)}%`}
      </span>
    </div>
  );
}

function Cell({ v, follow, bold }: {
  v: number | null; follow: number; bold?: boolean;
}) {
  const followed = v != null && v >= follow;
  return (
    <td className={`py-1 px-2 text-right font-mono tnum ${bold ? "font-medium" : ""} ${
      v == null ? "text-ink-mute"
        : followed ? "text-up"
          : v < 0 ? "text-ink-dim" : "text-ink-mute"}`}>
      {v == null ? "—" : signed(v, 1)}
    </td>
  );
}
