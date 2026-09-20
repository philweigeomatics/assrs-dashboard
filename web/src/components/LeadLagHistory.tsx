/**
 * 领先滞后历史 — when did A lead B, by how much, and for how long.
 *
 * Not a verdict. The screen beside this one answers "is there a relationship
 * in this history"; that answer is one arrow and one q-value for two years of
 * data, and we have measured what it costs: half the pairs that cleared the
 * correction pointed the other way in the second half of the same history.
 *
 * So this draws the history instead and lets a person read it. Each column is
 * a 60-session window, each row a lag, each cell the cross-correlation. A
 * stripe that holds at one lag for months is a behaviour; a speckle is two
 * unrelated stocks.
 *
 * THE NULLS ARE NOT DECORATION. Every rolling window has a best lag, noise
 * included — and because neighbouring windows share 55 of their 60 sessions,
 * a chance correlation persists for a dozen windows by construction. On forty
 * pairs of independent random walks the longest run had a MEDIAN of 9. So
 * what rotation produces sits immediately under the panel, as bars rather
 * than a second heatmap: the same information in a fraction of the space, and
 * no risk of a reader mistaking the placebo for the real thing. Anyone
 * reading the panel alone will see relationships that are not there.
 */

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { ApiError } from "../lib/api";
import type { LeadLagHistory as History } from "../lib/types";
import { fixed, signed } from "../lib/format";

//: Big enough to read a single cell. 140 windows at this width overflow a
//: desktop column, which is what the horizontal scroll is for — shrinking
//: the cells until they all fit produces a strip nobody can interpret.
const CELL = 14;       // px per window column
const ROW = 24;        // px per lag row
const GUTTER = 88;     // px for the lag labels — enough for a 4-char name

/** Blue when A leads, violet when B does, grey under the noise band. */
function cellColour(v: number | null, band: number): string {
  if (v == null) return "transparent";
  const mag = Math.abs(v);
  if (mag < band) return `rgba(148,163,184,${0.10 + mag * 0.3})`;
  const strength = Math.min(1, (mag - band) / (0.7 - band));
  const alpha = 0.25 + strength * 0.65;
  return v > 0
    ? `rgba(6,182,212,${alpha})`
    : `rgba(168,85,247,${alpha})`;
}

export function LeadLagHistory({ a, b, nameA, nameB }: {
  a: string; b: string; nameA: string; nameB: string;
}) {
  const [days, setDays] = useState(504);
  const [window, setWindow] = useState(60);
  const run = useMutation({
    mutationFn: () => api.leadLagHistory({ a, b, lookback_days: days, window }),
  });
  const d = run.data;

  return (
    <div className="flex flex-col gap-2.5 border-t border-line pt-3">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="text-[13.5px] font-semibold">🕰 领先滞后历史</h3>
        <span className="label">
          不下结论 —— 把「什么时候、领先几天、持续多久」摊开，由你判断
        </span>
        <div className="ml-auto flex items-center gap-2">
          <label className="label flex items-center gap-1" title="总回看长度">
            回看
            <select value={days} onChange={(e) => setDays(Number(e.target.value))}
              className="h-7 px-1 rounded-md bg-sunken text-[12.5px] tnum outline-none">
              {[252, 504, 756].map((v) => <option key={v} value={v}>{v} 天</option>)}
            </select>
          </label>
          <label className="label flex items-center gap-1"
            title="每个窗口多少个交易日。短一些能更早看到关系变化，也更容易看到噪声。">
            窗口
            <select value={window} onChange={(e) => setWindow(Number(e.target.value))}
              className="h-7 px-1 rounded-md bg-sunken text-[12.5px] tnum outline-none">
              {[40, 60, 90, 120].map((v) => <option key={v} value={v}>{v} 天</option>)}
            </select>
          </label>
          <button onClick={() => run.mutate()} disabled={run.isPending}
            className="h-8 px-3 rounded-lg bg-cyan text-white text-[13px] font-semibold disabled:opacity-60">
            {run.isPending ? "计算中…" : d ? "重新计算" : "看历史"}
          </button>
        </div>
      </div>

      {run.isError && (
        <p className="text-[12.5px] text-up">{(run.error as ApiError).message}</p>
      )}

      {d && (
        <>
          <Verdict d={d} />

          <Panel panel={d.panel} nameA={nameA} nameB={nameB} />
          {/* Directly below, so the comparison is unavoidable. */}
          <div className="flex flex-col gap-1">
            <p className="label leading-snug">
              把 {nameB} 的时间轴整体转动一圈，两只股票之间真实的对应关系就没了 ——
              再跑同样的计算 {d.null.rotations} 次，得到的就是这个方法
              在「什么都没有」时会给出的数字。实际那一条要明显高过它才说明问题。
            </p>
            <NullBar d={d} />
          </div>

          {d.episodes.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-[12px] border-collapse">
                <thead>
                  <tr className="text-ink-mute">
                    <th className="text-left font-normal pb-1 pr-3">时间段</th>
                    <th className="text-left font-normal pb-1 pr-3">谁先动</th>
                    <th className="text-right font-normal pb-1 px-2">相差</th>
                    <th className="text-right font-normal pb-1 px-2"
                      title="连续多少个窗口保持同一个滞后。相邻窗口重叠很多，所以这不是独立样本数。">
                      持续（窗口）
                    </th>
                    <th className="text-right font-normal pb-1 px-2">平均 r</th>
                    <th className="text-right font-normal pb-1 pl-2">最强 r</th>
                  </tr>
                </thead>
                <tbody>
                  {d.episodes.map((e, i) => (
                    <tr key={i} className={`border-t border-line ${
                      e.beats_null ? "" : "opacity-55"}`}>
                      <td className="py-1 pr-3 font-mono tnum whitespace-nowrap">
                        {e.from} → {e.to}
                      </td>
                      <td className="py-1 pr-3 whitespace-nowrap">
                        {e.lag === 0 ? (
                          <span className="text-ink-dim">同步（没有先后）</span>
                        ) : (
                          <>{e.leads === "a" ? nameA : nameB}<span className="text-ink-mute"> 先动</span></>
                        )}
                      </td>
                      <td className="py-1 px-2 text-right tnum">
                        {e.lag === 0 ? "—" : `${Math.abs(e.lag)} 天`}
                      </td>
                      <td className="py-1 px-2 text-right tnum">
                        {e.windows}
                        {e.beats_null && (
                          <span className="text-up" title="比所有轮转出来的最长片段都长">
                            {" "}★
                          </span>
                        )}
                      </td>
                      <td className="py-1 px-2 text-right tnum">{signed(e.mean_corr, 2)}</td>
                      <td className="py-1 pl-2 text-right tnum">{signed(e.peak_corr, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="label mt-1.5 leading-snug">
                ★ 表示这段比 {d.null.rotations} 次轮转里最长的那一段还长
                （轮转最长 {d.null.longest_max} 个窗口）。没有 ★ 的段落不代表是假的，
                只代表<b>光凭长度分辨不出来</b> —— 相邻窗口重叠 {d.panel.window - d.panel.step}/
                {d.panel.window} 个交易日，连续十几个窗口保持同一个滞后，随机数据里也很常见。
              </p>
            </div>
          )}
        </>
      )}
      {!d && !run.isPending && (
        <p className="label py-3 text-center">
          点「看历史」——会画出每个窗口里两只股票在各个滞后上的相关程度
        </p>
      )}
    </div>
  );
}

/** The one-line reading, which is usually "they just move together". */
function Verdict({ d }: { d: History }) {
  const sync = d.sync_share >= 0.6;
  const quiet = d.named_share <= d.null.named_share_median + 0.1;
  return (
    <div className="rounded-lg bg-sunken px-2.5 py-2 flex flex-col gap-1">
      <p className="text-[12.5px] leading-snug">
        {quiet ? (
          <>这两只在大部分时间里<b>没有稳定的对应关系</b> ——
            有方向的窗口只占 {(d.named_share * 100).toFixed(0)}%，
            而随机轮转也有 {(d.null.named_share_median * 100).toFixed(0)}%。</>
        ) : sync ? (
          <><b>它们基本是同步的</b>：在有方向的窗口里，
            {(d.sync_share * 100).toFixed(0)}% 的主导滞后是 0 天 ——
            也就是同一天一起动，<b>没有可以抢跑的时间差</b>。
            剩下那些有先后的片段更短，也更零散。</>
        ) : (
          <>主导滞后有 {(d.sync_share * 100).toFixed(0)}% 的时间是 0 天（同步），
            其余时间出现过 {d.episodes.filter((e) => e.lag !== 0).length} 段有先后的关系
            —— 具体看下表，长度和方向都在变。</>
        )}
      </p>
      <p className="label leading-snug">
        {d.windows} 个窗口 · 每个 {d.panel.window} 个交易日 · 噪声带 |r| ≥ {d.panel.band}
        {" · "}最长连续 {d.longest_run} 个窗口（轮转出来的中位是 {d.null.longest_median}，
        最长 {d.null.longest_max}）
      </p>
    </div>
  );
}

function Panel({ panel, nameA, nameB }: {
  panel: History["panel"]; nameA: string; nameB: string;
}) {
  const w = panel.dates.length * CELL;
  // A date every ~10 columns; more than that and the labels collide.
  const tickEvery = Math.max(1, Math.ceil(120 / CELL));
  return (
    <div className="flex flex-col gap-1.5">
      <Legend band={panel.band} nameA={nameA} nameB={nameB} />
      <div className="overflow-x-auto">
        <div style={{ width: w + GUTTER, minWidth: "100%" }}>
          {panel.lags.map((lag, r) => (
            <div key={lag} className="flex items-center" style={{ height: ROW }}>
              <span className="shrink-0 text-[11px] tnum text-right pr-2
                whitespace-nowrap overflow-hidden"
                style={{ width: GUTTER }}>
                {lag === 0
                  ? <span className="text-ink-dim">同步</span>
                  : <>
                      <span className="text-ink-mute">
                        {(lag > 0 ? nameA : nameB).slice(0, 4)}
                      </span>
                      <span className="text-ink-dim"> {Math.abs(lag)}天</span>
                    </>}
              </span>
              <div className="flex">
                {panel.matrix.map((row, c) => (
                  <div key={c}
                    title={`${panel.dates[c]} · ${lag === 0 ? "同步"
                      : `${lag > 0 ? nameA : nameB} 先动 ${Math.abs(lag)} 天`}`
                      + ` · r=${fixed(row[r], 2)}`}
                    style={{ width: CELL - 1, height: ROW - 2, marginRight: 1,
                             borderRadius: 2,
                             background: cellColour(row[r] ?? null, panel.band) }} />
                ))}
              </div>
            </div>
          ))}
          <div className="flex" style={{ paddingLeft: GUTTER }}>
            {panel.dates.map((dt, c) => (
              <div key={c} style={{ width: CELL }} className="shrink-0">
                {c % tickEvery === 0 && (
                  <span className="block text-[10px] text-ink-mute font-mono whitespace-nowrap">
                    {dt.slice(2, 7)}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * What the colours mean, without hovering.
 *
 * The row says WHO moved first; the colour says whether they moved the same
 * way or opposite ways. Two different things encoded in two different
 * channels, which is unreadable unless it is spelled out.
 */
function Legend({ band, nameA, nameB }: {
  band: number; nameA: string; nameB: string;
}) {
  const swatch = (bg: string) => (
    <span className="inline-block w-4 h-3 rounded-sm align-middle" style={{ background: bg }} />
  );
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11.5px]">
      <span className="flex items-center gap-1.5">
        <span className="text-ink-mute">行＝谁先动：</span>
        <span>上半 {nameA}</span>
        <span className="text-ink-dim">·</span>
        <span>中间 同步</span>
        <span className="text-ink-dim">·</span>
        <span>下半 {nameB}</span>
      </span>
      <span className="flex items-center gap-1.5">
        <span className="text-ink-mute">色＝</span>
        {swatch("rgba(6,182,212,0.9)")}
        <span title="先动的那只涨，另一只几天后也涨；跌也一起跌">同向</span>
        {swatch("rgba(168,85,247,0.9)")}
        <span title="先动的那只涨，另一只几天后反而跌 —— 时间上仍是领先，方向相反">
          反向<span className="text-ink-mute">（一涨一跌）</span>
        </span>
        {swatch("rgba(148,163,184,0.25)")}
        <span className="text-ink-mute">|r| &lt; {band}（噪声）</span>
      </span>
      <span className="flex items-center gap-1.5 text-ink-mute">
        深浅＝强弱
        {swatch("rgba(6,182,212,0.3)")}
        {swatch("rgba(6,182,212,0.55)")}
        {swatch("rgba(6,182,212,0.9)")}
      </span>
    </div>
  );
}

/** The null as numbers rather than a second heatmap — same information, a
 *  fraction of the screen, and harder to mistake for the real thing. */
function NullBar({ d }: { d: History }) {
  const bar = (label: string, real: number, sham: number, suffix = "") => {
    const max = Math.max(real, sham, 1);
    return (
      <div className="flex items-center gap-2 text-[11.5px]">
        <span className="w-24 shrink-0 text-ink-mute">{label}</span>
        <div className="flex-1 flex flex-col gap-0.5">
          <div className="flex items-center gap-1.5">
            <div className="h-2 rounded-sm bg-cyan"
              style={{ width: `${(real / max) * 100}%` }} />
            <span className="tnum">{real}{suffix} 实际</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2 rounded-sm bg-ink-mute/50"
              style={{ width: `${(sham / max) * 100}%` }} />
            <span className="tnum text-ink-mute">{sham}{suffix} 轮转</span>
          </div>
        </div>
      </div>
    );
  };
  return (
    <div className="rounded-lg bg-sunken px-2.5 py-2 flex flex-col gap-2">
      {bar("最长连续", d.longest_run, d.null.longest_max, " 窗口")}
      {bar("有方向的窗口", Math.round(d.named_share * 100),
           Math.round(d.null.named_share_median * 100), "%")}
      {bar("片段数", d.episodes.length, Math.round(d.null.episodes_median), " 段")}
    </div>
  );
}
