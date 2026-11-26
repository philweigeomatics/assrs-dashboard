/**
 * 🩺 指数体检 — every 自建指数's stored days, checked against fresh prices.
 *
 * A day that fails to store shows up as a failed job. A day stored WRONG
 * does not — 2026-11-19 sat in two indexes for a week. This lists, per
 * index, the days that are missing, lagging, have a member stored as halted
 * while it traded, or whose level the day before's holdings and fresh prices
 * do not reproduce, and repairs from the first bad day. See index_audit.py.
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";
import type { IndexHealthRow, IndexIssue } from "../../lib/types";

const KIND: Record<IndexIssue["kind"], string> = {
  missing: "缺失交易日",
  halted: "误记停牌",
  mismatch: "数值不符",
};

const SPANS = [
  { v: 60, label: "近 60 个交易日" },
  { v: 250, label: "近一年" },
  { v: 750, label: "近三年" },
];

export function IndexHealthPanel() {
  const [sessions, setSessions] = useState(250);
  const [fresh, setFresh] = useState(false);
  const q = useQuery({
    queryKey: ["admin", "index-health", sessions],
    queryFn: () => api.indexHealth(sessions, fresh),
    staleTime: 10 * 60_000,
  });
  const rows = q.data?.indexes ?? [];
  const bad = rows.filter((r) => r.ok === false).length;
  const unknown = rows.filter((r) => r.ok === null).length;

  return (
    <section className="card p-3 flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-[14px] font-semibold">🩺 指数体检</h3>
        <select value={sessions} onChange={(e) => setSessions(Number(e.target.value))}
          aria-label="体检范围"
          className="h-7 px-2 rounded-lg bg-sunken text-[12.5px] outline-none focus:ring-2 focus:ring-cyan/40">
          {SPANS.map((s) => <option key={s.v} value={s.v}>{s.label}</option>)}
        </select>
        <button onClick={() => { setFresh(true); q.refetch().finally(() => setFresh(false)); }}
          disabled={q.isFetching}
          className="h-7 px-2.5 rounded-lg bg-sunken text-[12.5px] disabled:opacity-60">
          {q.isFetching ? "体检中…（下载行情约 10–40 秒）" : "重新体检"}
        </button>
        {q.data && (
          <span className="ml-auto label tnum">
            {rows.length - bad - unknown}/{rows.length} 无误
            {bad > 0 && <b className="text-brand-ink"> · {bad} 个有问题</b>}
            {unknown > 0 && <> · {unknown} 个没法检查</>}
            {" "}· {new Date(q.data.checked_at).toLocaleString()}
          </span>
        )}
      </div>
      <p className="label leading-snug">
        每个交易日用前一天收盘的持仓和重新下载的行情重算一遍，和存下来的数对照；同时找缺失的交易日、
        落后的天数，以及当天其实有成交、却被记成停牌（按原价不动）的成分股。每天的数据管线跑完也会核对最近
        20 个交易日，有问题就让那次运行失败（GitHub 会发邮件）。
      </p>

      {q.isPending && <p className="label py-6 text-center">体检中…（下载行情约 10–40 秒）</p>}
      {q.isError && (
        <p className="text-[12.5px] text-up">
          {(q.error as ApiError).message}
          <button onClick={() => q.refetch()} className="ml-2 text-cyan">重试</button>
        </p>
      )}
      {rows.map((r) => <IndexRow key={r.id} r={r} />)}
    </section>
  );
}

function IndexRow({ r }: { r: IndexHealthRow }) {
  const qc = useQueryClient();
  const [confirm, setConfirm] = useState(false);
  const fix = useMutation({
    mutationFn: () => api.indexRepair(r.id, r.repair_from!),
    onSuccess: () => {
      setConfirm(false);
      qc.invalidateQueries({ queryKey: ["admin", "index-health"] });
      qc.invalidateQueries({ queryKey: ["idx"] });
    },
  });
  const issues = r.issues ?? [];
  const days = [...new Set(issues.map((i) => i.d))];
  const tone = r.ok === null ? "bg-sunken" : r.ok ? "bg-sunken" : "bg-brand-ink/5 ring-1 ring-brand-ink/30";

  return (
    <div className={`rounded-xl p-2.5 flex flex-col gap-1.5 ${tone}`}>
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className="text-[13.5px] font-semibold">{r.ok ? "✅" : r.ok === null ? "❔" : "⚠️"} {r.name}</span>
        <span className="label">#{r.id} · {r.market === "CN" ? "A 股" : "美股"} · 用户 {r.user ?? "—"}</span>
        {r.last && (
          <span className="label tnum">
            存到 {r.last}{r.latest && r.latest !== r.last ? `（最新交易日 ${r.latest}）` : ""}
            {r.checked != null && ` · 核对 ${r.checked} 天`}
            {r.unverifiable ? ` · ${r.unverifiable} 天无法核对（成分股的历史行情已下架）` : ""}
          </span>
        )}
      </div>

      {r.error && <p className="text-[12.5px] text-up">{r.error}</p>}
      {r.ok && <p className="text-[12.5px] text-ink-dim">从 {r.since} 起每一天都对得上。</p>}
      {!!r.behind && (
        <p className="text-[12.5px]">
          <b>落后 {r.behind} 个交易日</b>（从 {r.behind_from} 起没有存）—— 每天的数据管线会补上；补不上的话，那次运行的日志里有原因。
        </p>
      )}

      {days.length > 0 && (
        <>
          <p className="text-[12.5px]">
            <b className="text-brand-ink">{days.length} 天不对：</b>
            <span className="tnum">{days.join("、")}</span>
          </p>
          <ul className="flex flex-col gap-0.5">
            {issues.map((i, k) => (
              <li key={k} className="text-[12px] leading-snug flex gap-2">
                <span className="tnum shrink-0 w-[78px]">{i.d}</span>
                <span className="shrink-0 w-[64px] text-brand-ink">{KIND[i.kind]}</span>
                <span className="text-ink-dim">
                  {i.detail}{i.tickers?.length ? `：${i.tickers.join("、")}` : ""}
                </span>
              </li>
            ))}
          </ul>
          {r.repair_from && (
            <div className="flex flex-wrap items-center gap-2 pt-1">
              {!confirm ? (
                <button onClick={() => setConfirm(true)}
                  className="h-7 px-2.5 rounded-lg bg-violet text-white text-[12.5px] font-semibold">
                  从 {r.repair_from} 起重算
                </button>
              ) : (
                <>
                  <span className="text-[12.5px]">
                    删除 {r.repair_from} 及以后存下的每一天，从前一天的收盘接着用最新行情重算。之前的天不动。
                  </span>
                  <button onClick={() => fix.mutate()} disabled={fix.isPending}
                    className="h-7 px-2.5 rounded-lg bg-violet text-white text-[12.5px] font-semibold disabled:opacity-60">
                    {fix.isPending ? "重算中…" : "确认重算"}
                  </button>
                  <button onClick={() => setConfirm(false)} className="text-[12px] text-ink-mute">取消</button>
                </>
              )}
              {fix.isError && <span className="text-[12px] text-up">{(fix.error as ApiError).message}</span>}
              {fix.data && (
                <span className="text-[12px] text-ink-dim">
                  删了 {fix.data.removed.length} 天，
                  {fix.data.report[0]?.ok ? `重算到 ${fix.data.report[0].as_of}` : `重算失败：${fix.data.report[0]?.error ?? ""}`}
                </span>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
