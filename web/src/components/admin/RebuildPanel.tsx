/**
 * 重建任务 — rebuild the PPIs, and watch it happen.
 *
 * A rebuild recomputes PPI and market breadth from the sector map and takes
 * 20-60 minutes, so it cannot happen inside a request. The worker writes its
 * progress to the rebuild_jobs table and this polls that table — never the
 * worker — which is what makes the bar survive a reload, a different tab, or
 * a different machine.
 *
 * Polling only runs while something is live, and stops the moment nothing
 * is: a progress bar is not worth a request every four seconds forever.
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";
import type { RebuildJob } from "../../lib/types";

/** While a job is live. Matches the Streamlit page's cadence. */
const POLL_MS = 4000;

const STATUS: Record<RebuildJob["status"], { label: string; cls: string }> = {
  pending: { label: "排队中", cls: "text-ink-dim" },
  running: { label: "进行中", cls: "text-brand-ink" },
  completed: { label: "已完成", cls: "text-up" },
  failed: { label: "失败", cls: "text-up" },
  stalled: { label: "可能已中断", cls: "text-brand-ink" },
};

export function RebuildPanel({ sectorNames }: { sectorNames: string[] }) {
  const qc = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const [err, setErr] = useState("");

  const jobs = useQuery({
    queryKey: ["admin", "rebuild"],
    queryFn: api.adminJobs,
    // Only while something is actually live.
    refetchInterval: (q) => (q.state.data?.running ? POLL_MS : false),
  });

  const start = useMutation({
    mutationFn: (sectors: string[]) => api.adminStartRebuild(sectors),
    onSuccess: () => {
      setErr("");
      setConfirming(false);
      qc.invalidateQueries({ queryKey: ["admin", "rebuild"] });
    },
    onError: (e) => setErr((e as ApiError).message),
  });

  const d = jobs.data;
  const live = d?.jobs.find((j) => j.status === "running" || j.status === "pending");

  return (
    <section className="card p-3 flex flex-col gap-2.5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="text-[13.5px] font-semibold">🔄 重建任务</h3>
        <span className="label">
          重算 PPI 和市场宽度 —— 板块成分改了之后要跑一次才会生效
        </span>
        <div className="ml-auto flex items-center gap-2">
          {live ? (
            <span className="label">任务进行中，暂时不能再开一个</span>
          ) : confirming ? (
            <>
              <button onClick={() => start.mutate([])}
                disabled={start.isPending}
                className="h-8 px-3 rounded-lg bg-up text-white text-[13px] font-semibold disabled:opacity-60">
                {start.isPending ? "启动中…" : `确认重建全部 ${sectorNames.length} 个板块`}
              </button>
              <button onClick={() => setConfirming(false)}
                className="h-8 px-3 rounded-lg bg-sunken text-[13px]">取消</button>
            </>
          ) : (
            <button onClick={() => setConfirming(true)}
              className="h-8 px-3 rounded-lg bg-sunken text-[13px] font-semibold">
              重建全部板块
            </button>
          )}
        </div>
      </div>

      {confirming && !live && (
        <p className="text-[12.5px] text-brand-ink leading-snug">
          全量重建会<b>清空并重算每个板块的 PPI</b>，按 API 限速要 20–60 分钟。
          期间看板照常可用，但读到的是正在被替换的数据。
        </p>
      )}

      {err && <p className="text-[12.5px] text-up">{err}</p>}
      {jobs.isError && (
        <p className="text-[12.5px] text-up">{(jobs.error as ApiError).message}</p>
      )}

      {d && d.jobs.length === 0 && (
        <p className="label py-3 text-center">还没有跑过重建任务</p>
      )}

      {d && d.jobs.length > 0 && (
        <div className="flex flex-col divide-y divide-line">
          {d.jobs.map((j) => <Job key={j.job_id} j={j} stale={d.stale_minutes} />)}
        </div>
      )}

      {d?.running && (
        <p className="label">每 {POLL_MS / 1000} 秒自动刷新，离开这一页也会继续跑</p>
      )}
    </section>
  );
}

function Job({ j, stale }: { j: RebuildJob; stale: number }) {
  const s = STATUS[j.status];
  const open = j.status === "running" || j.status === "pending";
  return (
    <div className="py-2 flex flex-col gap-1">
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-[12.5px]">
        <span className={`font-medium ${s.cls}`}>{s.label}</span>
        <span className="text-ink-dim">
          {j.scope === "all" ? "全部板块" : j.sectors.join("、") || "—"}
        </span>
        <span className="label font-mono tnum">{j.job_id}</span>
        <span className="ml-auto label font-mono tnum">
          {j.completed_at || j.created_at}
        </span>
      </div>

      {(open || j.status === "stalled") && (
        <div className="flex items-center gap-2">
          <div className="flex-1 h-2 rounded-full bg-sunken overflow-hidden">
            <div className={`h-full rounded-full ${
              j.status === "stalled" ? "bg-ink-mute" : "bg-cyan"}`}
              style={{ width: `${Math.max(2, Math.min(100, j.progress))}%` }} />
          </div>
          <span className="w-10 text-right tnum text-[12px]">{j.progress}%</span>
        </div>
      )}

      {j.message && (
        <p className="label leading-snug">{j.message}</p>
      )}

      {j.status === "stalled" && (
        /* A worker dies with its instance, and the row then says "running"
           for good. Saying so is better than a bar that never moves. */
        <p className="text-[12px] text-brand-ink leading-snug">
          已经 {Math.round(j.age_minutes ?? 0)} 分钟没有结束（超过 {stale} 分钟就按中断算）。
          后台进程很可能已经被回收 —— 数据没有损坏，重新跑一次即可。
        </p>
      )}

      {j.error && (
        <pre className="text-[11.5px] font-mono text-up bg-sunken rounded-md p-2
          overflow-x-auto whitespace-pre-wrap">{j.error}</pre>
      )}
    </div>
  );
}
