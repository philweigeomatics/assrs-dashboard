/**
 * 💼 组合 — build an allocation, and keep the ones worth keeping.
 *
 * Two sections rather than two pages, because they are one workflow: the
 * optimiser produces an allocation and 我的组合 is where it goes. Saving
 * writes the same `funds` / `fund_positions` rows the Streamlit page wrote,
 * so a portfolio made in either app is visible in both.
 */

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import type { SavedFund } from "../lib/types";
import { NavBar } from "../components/NavBar";
import { Optimiser } from "../components/portfolio/Optimiser";
import { fixed } from "../lib/format";

const SECTIONS = [
  { id: "build", label: "🎯 组合优化" },
  { id: "mine", label: "💼 我的组合" },
] as const;

type Section = typeof SECTIONS[number]["id"];

export function Portfolio() {
  useEffect(() => { document.title = "ASSRS · 组合"; }, []);
  const [tab, setTab] = useState<Section>("build");
  const qc = useQueryClient();
  const funds = useQuery({ queryKey: ["pf", "funds"], queryFn: api.funds });

  return (
    <div className="min-h-screen">
      <NavBar />
      <main className="max-w-[1500px] mx-auto px-3 py-3 flex flex-col gap-3">
        <section className="card p-3 flex flex-wrap items-center gap-x-3 gap-y-2">
          <h2 className="text-[14px] font-semibold">💼 组合</h2>
          <div className="flex items-center gap-1">
            {SECTIONS.map((x) => (
              <button key={x.id} onClick={() => setTab(x.id)}
                className={`h-8 px-3 rounded-lg text-[13px] font-medium transition-colors ${
                  tab === x.id ? "bg-elevated text-ink" : "text-ink-mute hover:text-ink"
                }`}>
                {x.label}
              </button>
            ))}
          </div>
          <span className="ml-auto label">
            {funds.data ? `已存 ${funds.data.funds.length} 个组合` : ""}
          </span>
        </section>

        {tab === "build" && (
          <Optimiser onSaved={() => {
            qc.invalidateQueries({ queryKey: ["pf", "funds"] });
            setTab("mine");
          }} />
        )}

        {tab === "mine" && (
          <FundList
            funds={funds.data?.funds ?? []}
            loading={funds.isPending}
            error={funds.error as ApiError | null}
            onChanged={() => qc.invalidateQueries({ queryKey: ["pf", "funds"] })}
            onBuild={() => setTab("build")} />
        )}
      </main>
    </div>
  );
}

function FundList({ funds, loading, error, onChanged, onBuild }: {
  funds: SavedFund[]; loading: boolean; error: ApiError | null;
  onChanged: () => void; onBuild: () => void;
}) {
  const [open, setOpen] = useState<number | null>(null);

  if (loading) return <div className="card p-8 text-center label">读取中…</div>;
  if (error) {
    return <div className="card p-4 text-center text-up text-[13px]">{error.message}</div>;
  }
  if (funds.length === 0) {
    return (
      <div className="card p-10 flex flex-col items-center gap-2">
        <p className="label">还没有存过组合</p>
        <button onClick={onBuild}
          className="h-8 px-3 rounded-lg bg-cyan text-white text-[13px] font-semibold">
          去做一个
        </button>
      </div>
    );
  }

  return (
    <section className="card p-3 flex flex-col divide-y divide-line">
      {funds.map((f) => (
        <Fund key={f.id} f={f} open={open === f.id}
          onToggle={() => setOpen(open === f.id ? null : f.id)}
          onChanged={onChanged} />
      ))}
    </section>
  );
}

function Fund({ f, open, onToggle, onChanged }: {
  f: SavedFund; open: boolean; onToggle: () => void; onChanged: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const detail = useQuery({
    queryKey: ["pf", "fund", f.id],
    queryFn: () => api.fundDetail(f.id),
    enabled: open,
  });
  const remove = useMutation({
    mutationFn: () => api.deleteFund(f.id),
    onSuccess: () => { setConfirming(false); onChanged(); },
  });

  return (
    <div className="py-2 flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline gap-2">
        <button onClick={onToggle} className="text-[13px] font-medium text-left">
          {f.name}
        </button>
        <span className="label tnum">{f.holdings} 只</span>
        {f.benchmark && <span className="label">对标 {f.benchmark}</span>}
        <span className="label font-mono tnum">建于 {f.inception ?? "—"}</span>
        <div className="ml-auto flex items-center gap-2">
          {confirming ? (
            <>
              <button onClick={() => remove.mutate()} disabled={remove.isPending}
                className="h-7 px-2 rounded-md bg-up text-white text-[12px] disabled:opacity-60">
                {remove.isPending ? "删除中…" : "确认删除"}
              </button>
              <button onClick={() => setConfirming(false)}
                className="h-7 px-2 rounded-md bg-sunken text-[12px]">取消</button>
            </>
          ) : (
            <>
              <button onClick={onToggle} className="text-[12px] text-cyan">
                {open ? "收起" : "展开"}
              </button>
              <button onClick={() => setConfirming(true)}
                className="text-[12px] text-ink-mute hover:text-up">删除</button>
            </>
          )}
        </div>
      </div>

      {open && (
        <div className="pl-1">
          {detail.isPending && <p className="label">读取中…</p>}
          {detail.data && (
            <div className="flex flex-col gap-1">
              {detail.data.holdings.map((h) => (
                <div key={h.t} className="flex items-center gap-2 text-[12.5px]">
                  <span className="w-20 shrink-0 font-mono tnum text-[11.5px]">{h.t}</span>
                  <div className="flex-1 h-3.5 rounded-sm bg-sunken overflow-hidden">
                    <div className="h-full rounded-sm bg-cyan"
                      style={{ width: `${Math.min(100, h.weight_pct)}%` }} />
                  </div>
                  <span className="w-14 text-right tnum">{fixed(h.weight_pct, 1)}%</span>
                </div>
              ))}
              <p className="label mt-1">
                这是建仓时的目标权重。市值会漂移，下一版会把当前实际权重和偏离画出来。
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
