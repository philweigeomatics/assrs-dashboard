/**
 * 配对搜索 — find the pairs in the watchlist, instead of guessing which to test.
 *
 * The screen you actually want, and the one that is easiest to get wrong.
 * Eighty stocks make 3,160 pairs; lead-lag tests each in both directions, so
 * 6,320 tests, of which a 5% threshold passes about 316 on data with no
 * structure in it whatsoever. Rank those by p-value, show the top ten, and
 * all ten will look compelling.
 *
 * So the funnel is the output, not a diagnostic. "3,160 → 240 → 31 → 4,
 * against 1.6 expected by chance" is what tells you whether the four mean
 * anything; the four on their own tell you nothing, and a screen that showed
 * only them would be lying by omission.
 *
 * The last stage is the one that counts: every surviving pair was chosen on
 * the first half of the history and confirmed on the second, which played no
 * part in choosing it. Rows that fail that confirmation are still listed,
 * dimmed — "we looked, and this was the closest thing" is worth seeing, and
 * hiding it would make the screen look far more productive than it is.
 *
 * A finished search must survive leaving the tab. It is two minutes and
 * eighty Tushare calls, so it is held in the query cache rather than in
 * component state, under a key made of the parameters — switching away and
 * back re-reads it instead of re-running it, and changing a parameter is a
 * different key and therefore correctly a different (empty) result.
 *
 * The server keeps it too, across restarts, keyed on the watchlist and the
 * newest published session — see discover_cache. So the first load after a
 * cold start is also instant, and a new close is the thing that expires it.
 */

import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import type { DiscoverResult, DiscoverRow } from "../lib/types";
import { usePersistentState } from "../lib/usePersistentState";
import { fixed } from "../lib/format";

export function Discover({ kind, onUse }: {
  kind: "pair-trade" | "lead-lag";
  /** Hand a found pair to the manual screen beside this one. */
  onUse?: (a: string, b: string, names: [string, string]) => void;
}) {
  const [minCorr, setMinCorr] = usePersistentState<number>("assrs.disc.corr", 0.45);
  const [within, setWithin] = usePersistentState<boolean>("assrs.disc.sector", false);
  const [days, setDays] = usePersistentState<number>("assrs.disc.days", 504);

  const qc = useQueryClient();
  // Keyed on the parameters, so the answer that comes back is unambiguously
  // the answer to what is on screen. `enabled: false` because a two-minute
  // search must never start on its own — but the cached data for this key is
  // still returned on mount, which is what makes it survive a tab switch.
  const key = ["discover", kind, days, minCorr, within] as const;
  const result = useQuery({
    queryKey: key,
    queryFn: () => api.discover({ kind, lookback_days: days,
                                  min_corr: minCorr, within_sector: within }),
    enabled: false,
    gcTime: 24 * 3600_000,
    staleTime: Infinity,
    retry: false,
  });

  // Re-running is a separate action from reading, because it costs two
  // minutes; `force` tells the server to ignore what it has stored.
  const rerun = useMutation({
    mutationFn: () => api.discover({ kind, lookback_days: days,
                                     min_corr: minCorr, within_sector: within }, true),
    onSuccess: (data) => qc.setQueryData(key, data),
  });

  const busy = result.isFetching || rerun.isPending;
  const data = result.data;
  const error = (rerun.error ?? result.error) as ApiError | null;

  return (
    <section className="card p-3 flex flex-col gap-2.5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="text-[13.5px] font-semibold">🔎 从自选股中搜索</h3>
        <span className="label">
          不用自己挑 —— 全量配对，先筛再验，最后用没参与筛选的那一半数据确认
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <label className="label flex items-center gap-1"
          title="先按同期相关性粗筛。这不是显著性检验，只是缩小范围。">
          相关性下限
          <input type="range" min={0} max={90} step={5}
            value={Math.round(minCorr * 100)}
            onChange={(e) => setMinCorr(Number(e.target.value) / 100)}
            className="w-24 accent-[var(--color-cyan)]" />
          <span className="tnum w-8">{fixed(minCorr, 2)}</span>
        </label>
        <label className="label flex items-center gap-1" title="历史长度，对半切开">
          回看
          <select value={days} onChange={(e) => setDays(Number(e.target.value))}
            className="h-7 px-1 rounded-md bg-sunken text-[12.5px] tnum outline-none">
            {[252, 504, 756].map((d) => <option key={d} value={d}>{d} 天</option>)}
          </select>
        </label>
        <label className="label flex items-center gap-1.5"
          title="只在同一板块内配对。跨行业的高相关，通常说的是大盘，不是这两只股票之间的关系。">
          <input type="checkbox" checked={within}
            onChange={(e) => setWithin(e.target.checked)}
            className="accent-[var(--color-cyan)]" />
          仅同板块
        </label>
        <button onClick={() => (data ? rerun.mutate() : result.refetch())}
          disabled={busy}
          className="h-8 px-4 rounded-lg bg-cyan text-white text-[13px] font-semibold disabled:opacity-60">
          {busy ? "搜索中…（数十只股票要逐只取行情）"
            : data ? "重新搜索" : "开始搜索"}
        </button>
        {data?.cached && (
          <span className="label"
            title={`这次结果基于 ${data.session} 收盘后的数据。只要没有新的交易日，`
              + `并且自选股、相关性下限、回看天数都没变，结论就不会变。`}>
            已缓存 · 基于 {data.session} 收盘
            {data.generated_at ? ` · 生成于 ${data.generated_at.slice(0, 16).replace("T", " ")}` : ""}
          </span>
        )}
      </div>

      {error && <p className="text-[12.5px] text-up">{error.message}</p>}
      {data && <Found data={data} onUse={onUse} />}
    </section>
  );
}

function Found({ data, onUse }: {
  data: DiscoverResult;
  onUse?: (a: string, b: string, names: [string, string]) => void;
}) {
  const f = data.funnel;
  const lead = data.kind === "lead-lag";

  return (
    <div className="flex flex-col gap-2.5">
      {/* The funnel IS the result. */}
      <div className="rounded-lg bg-sunken px-2.5 py-2 flex flex-col gap-1">
        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-[12px]">
          <Step n={f.universe} label="只股票" />
          <Arrow />
          <Step n={f.pairs_possible} label="组配对" />
          <Arrow />
          {/* When the cap bites, say so. "250 shortlisted" reads as "250
              qualified" and hides that 900 did and 650 were dropped by a
              ceiling rather than by the threshold. */}
          <Step n={f.shortlisted} strong={false}
            label={f.pairs_correlated > f.shortlisted
              ? `已测（相关性 ≥ ${fixed(f.min_corr, 2)} 的共 ${f.pairs_correlated} 组，取最相关的前 ${f.shortlisted}）`
              : `相关性 ≥ ${fixed(f.min_corr, 2)}`} />
          <Arrow />
          <Step n={f.screened} label="前半程显著" />
          <Arrow />
          <Step n={f.survivors} label="后半程仍成立" strong />
        </div>
        <p className="text-[11.5px] text-ink-mute leading-snug">
          前半程 {data.train.from} → {data.train.to}，用来<b>挑选</b>；
          后半程 {data.test.from} → {data.test.to}，完全没参与挑选，用来<b>验证</b>。
          进入验证的 {f.retested} 组里，<b>{f.retest_hits}</b> 组通过了
          p&lt;{f.alpha} —— 而纯靠运气预计就会有
          <b> {fixed(f.expected_by_chance, 1)} </b>组通过。
          经多重检验校正后真正站住的是
          <b className={f.survivors ? "text-up" : ""}> {f.survivors} </b>组。
          {f.survivors === 0 && " 这一轮没有可用的配对，这本身就是答案。"}
        </p>
        {f.pairs_correlated > f.shortlisted && (
          <p className="text-[11.5px] text-brand-ink leading-snug">
            ⚠ 相关性 ≥ {fixed(f.min_corr, 2)} 的有 {f.pairs_correlated} 组，
            但每轮最多只检验最相关的 {f.shortlisted} 组 —— 此时<b>调低</b>阈值不会有任何变化，
            只有<b>调高</b>才会真正改变被检验的集合。
          </p>
        )}
      </div>

      {data.rows.length === 0 ? (
        <p className="label py-4 text-center">
          没有一组通过前半程的筛选。可以把相关性下限调低，或拉长回看天数。
        </p>
      ) : (
        <div className="overflow-auto rounded-lg border border-line max-h-[460px]">
          <table className="w-full border-collapse text-[12px]">
            <thead className="sticky top-0 bg-panel">
              <tr className="border-b border-line text-ink-mute">
                <th className="text-left font-medium px-2 py-1.5">A</th>
                <th className="text-left font-medium px-2 py-1.5">B</th>
                <th className="text-right font-medium px-2 py-1.5">相关</th>
                <th className="text-right font-medium px-2 py-1.5"
                  title="前半程（挑选用）">p 前</th>
                <th className="text-right font-medium px-2 py-1.5"
                  title="后半程，没参与挑选">p 后</th>
                <th className="text-right font-medium px-2 py-1.5"
                  title="后半程经多重检验校正后的 q 值">q</th>
                {lead
                  ? <>
                      <th className="text-left font-medium px-2 py-1.5">方向</th>
                      <th className="text-center font-medium px-2 py-1.5"
                        title="两半是否对谁领先的判断一致">一致</th>
                    </>
                  : <>
                      <th className="text-right font-medium px-2 py-1.5">β</th>
                      <th className="text-right font-medium px-2 py-1.5"
                        title="价差回到均值所需天数">半衰期</th>
                    </>}
                {onUse && <th />}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((r) => (
                <Row key={`${r.a}-${r.b}`} r={r} lead={lead} onUse={onUse} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Step({ n, label, strong }: { n: number; label: string; strong?: boolean }) {
  return (
    <span className={`px-2 py-0.5 rounded-md ${
      strong ? "bg-cyan text-white font-semibold" : "bg-panel"}`}>
      <b className="tnum">{n.toLocaleString()}</b>
      <span className={strong ? "" : "text-ink-mute"}> {label}</span>
    </span>
  );
}

const Arrow = () => <span className="text-ink-mute">→</span>;

function Row({ r, lead, onUse }: {
  r: DiscoverRow; lead: boolean;
  onUse?: (a: string, b: string, names: [string, string]) => void;
}) {
  const leader = r.leads === "a" ? r.a : r.b;
  return (
    <tr className={`border-b border-line/60 hover:bg-sunken ${
      r.survives ? "" : "opacity-45"}`}>
      <td className="px-2 py-1 whitespace-nowrap">
        <Link to={`/?t=${r.a}`} className="text-cyan font-mono">{r.a}</Link>{" "}
        <span className="truncate">{r.name_a}</span>
      </td>
      <td className="px-2 py-1 whitespace-nowrap">
        <Link to={`/?t=${r.b}`} className="text-cyan font-mono">{r.b}</Link>{" "}
        <span className="truncate">{r.name_b}</span>
      </td>
      <td className="px-2 py-1 text-right tnum">{fixed(r.corr, 2)}</td>
      <td className="px-2 py-1 text-right tnum text-ink-mute">{fixed(r.p_train, 3)}</td>
      <td className="px-2 py-1 text-right tnum">{fixed(r.p_test, 3)}</td>
      <td className={`px-2 py-1 text-right tnum font-medium ${
        r.survives ? "text-up" : ""}`}>{fixed(r.q, 3)}</td>
      {lead
        ? <>
            <td className="px-2 py-1 whitespace-nowrap">
              <span className="font-mono">{leader}</span> 领先 {r.lag} 天
            </td>
            <td className="px-2 py-1 text-center"
              title={r.same_direction
                ? "两半都认为是同一只领先"
                : "两半对谁领先的判断相反 —— 这不是领先关系，是两次恰好显著的噪声"}>
              {r.same_direction ? "✓" : "✗"}
            </td>
          </>
        : <>
            <td className="px-2 py-1 text-right tnum">{fixed(r.beta, 2)}</td>
            <td className={`px-2 py-1 text-right tnum ${
              r.tradeable ? "" : "text-ink-mute"}`}
              title={r.tradeable ? "" : "价差收敛太慢，协整成立也做不了"}>
              {r.half_life == null ? "—" : `${fixed(r.half_life, 0)} 天`}
            </td>
          </>}
      {onUse && (
        <td className="px-2 py-1 text-right">
          <button onClick={() => onUse(r.a, r.b, [r.name_a, r.name_b])}
            className="text-[11.5px] text-cyan whitespace-nowrap">载入 ↑</button>
        </td>
      )}
    </tr>
  );
}
