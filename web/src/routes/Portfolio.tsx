/**
 * 💼 组合 — build an allocation, and manage the ones worth keeping.
 *
 * Two sections rather than two pages, because they are one workflow: the
 * optimiser produces an allocation and 我的组合 is where it goes. Saving
 * writes the same `funds` / `fund_positions` rows the Streamlit page wrote,
 * so a portfolio made in either app is visible in both.
 *
 * 我的组合 manages ONE fund at a time, picked from a dropdown. Every fund
 * expanded at once meant five NAV curves and five drift charts competing for
 * the same screen, and none of them readable.
 */

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import type { SavedFund } from "../lib/types";
import { NavBar } from "../components/NavBar";
import { Optimiser } from "../components/portfolio/Optimiser";
import { FundTrack } from "../components/portfolio/FundTrack";
import { DriftChart } from "../components/portfolio/DriftChart";
import { Rebalance } from "../components/portfolio/Rebalance";
import { Industries } from "../components/portfolio/Industries";
import { fixed, signed } from "../lib/format";

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
          <Manage
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

const PANES = [
  { id: "track", label: "📈 净值与基准" },
  { id: "alloc", label: "📋 持仓分布" },
  { id: "drift", label: "📊 权重漂移" },
  { id: "rebal", label: "⚖️ 调仓" },
] as const;

type Pane = typeof PANES[number]["id"];

function Manage({ funds, loading, error, onChanged, onBuild }: {
  funds: SavedFund[]; loading: boolean; error: ApiError | null;
  onChanged: () => void; onBuild: () => void;
}) {
  const [id, setId] = useState<number | null>(null);
  const [pane, setPane] = useState<Pane>("track");
  const [confirming, setConfirming] = useState(false);
  const qc = useQueryClient();

  // Default to the first fund once the list arrives, and recover if the
  // selected one is deleted from under us.
  const current = funds.find((f) => f.id === id) ?? funds[0] ?? null;
  useEffect(() => {
    if (current && current.id !== id) setId(current.id);
  }, [current?.id]);

  const detail = useQuery({
    queryKey: ["pf", "fund", current?.id],
    queryFn: () => api.fundDetail(current!.id),
    enabled: current != null,
    staleTime: 5 * 60_000,
  });

  const remove = useMutation({
    mutationFn: () => api.deleteFund(current!.id),
    onSuccess: () => { setConfirming(false); setId(null); onChanged(); },
  });
  const revalue = useMutation({
    mutationFn: () => api.revalueFund(current!.id),
    onSuccess: () => detail.refetch(),
  });

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

  const d = detail.data;
  const t = d?.tracking;

  return (
    <div className="flex flex-col gap-3">
      <section className="card p-3 flex flex-col gap-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <select value={current?.id ?? ""}
            onChange={(e) => { setId(Number(e.target.value)); setConfirming(false); }}
            aria-label="选择组合"
            className="h-9 px-2 rounded-lg bg-sunken text-[13.5px] font-medium
              outline-none focus:ring-2 focus:ring-cyan/40 max-w-[260px]">
            {funds.map((f) => (
              <option key={f.id} value={f.id}>{f.name}</option>
            ))}
          </select>

          {current && (
            <>
              <span className="label tnum">{current.holdings} 只</span>
              {current.benchmark && (
                <span className="label">对标 {current.benchmark}</span>
              )}
              <span className="label font-mono tnum">
                建于 {current.inception ?? "—"}
              </span>
            </>
          )}

          <div className="ml-auto flex items-center gap-2">
            {confirming ? (
              <>
                <button onClick={() => remove.mutate()} disabled={remove.isPending}
                  className="h-7 px-2 rounded-md bg-up text-white text-[12px]
                    disabled:opacity-60">
                  {remove.isPending ? "删除中…" : "确认删除"}
                </button>
                <button onClick={() => setConfirming(false)}
                  className="h-7 px-2 rounded-md bg-sunken text-[12px]">取消</button>
              </>
            ) : (
              <button onClick={() => setConfirming(true)}
                className="text-[12px] text-ink-mute hover:text-up">删除这个组合</button>
            )}
          </div>
        </div>

        {d?.risk && <RiskStrip r={d.risk} alpha={t?.alpha_pct ?? null} />}

        <div className="flex flex-wrap items-center gap-1">
          {PANES.map((p) => (
            <button key={p.id} onClick={() => setPane(p.id)}
              className={`h-8 px-3 rounded-lg text-[12.5px] font-medium transition-colors ${
                pane === p.id ? "bg-elevated text-ink" : "text-ink-mute hover:text-ink"
              }`}>
              {p.label}
            </button>
          ))}
        </div>
      </section>

      {detail.isPending && <div className="card p-8 text-center label">读取中…</div>}
      {detail.error && (
        <div className="card p-4 text-center text-up text-[13px]">
          {(detail.error as ApiError).message}
        </div>
      )}

      {d && (
        <section className="card p-3">
          {pane === "track" && (
            <FundTrack d={d} onRevalue={() => revalue.mutate()}
              revaluing={revalue.isPending} />
          )}
          {pane === "alloc" && <Allocation d={d} />}
          {pane === "drift" && (
            <DriftChart real={d.drift_history.real}
              simulated={d.drift_history.simulated} />
          )}
          {pane === "rebal" && (
            <Rebalance d={d} onDone={() => {
              detail.refetch();
              qc.invalidateQueries({ queryKey: ["pf", "funds"] });
            }} />
          )}
          {revalue.isError && (
            <p className="text-[12.5px] text-up mt-2">
              {(revalue.error as ApiError).message}
            </p>
          )}
        </section>
      )}
    </div>
  );
}

/** The five numbers the nightly rollup already computes, which nothing read. */
function RiskStrip({ r, alpha }: {
  r: NonNullable<import("../lib/types").FundDetail["risk"]>;
  alpha: number | null;
}) {
  return (
    <div className="grid gap-x-4 gap-y-1 grid-cols-2 sm:grid-cols-3 lg:grid-cols-6
      text-[12px] rounded-lg bg-sunken px-2.5 py-2">
      <Cell label="年化波动" v={r.ann_vol_pct == null ? "—" : `${fixed(r.ann_vol_pct, 1)}%`}
        hint="日收益标准差年化。" />
      <Cell label="Beta (30日)" v={fixed(r.beta_30d, 2)}
        hint="相对基准的波动倍数。大于 1 比基准更颠簸。" />
      <Cell label="最大回撤" v={`${fixed(r.max_drawdown_pct, 1)}%`}
        tone="down" hint="净值从最高点跌下来最深的一次。" />
      <Cell label="VaR 95%" v={r.var_95_pct == null ? "—" : `${fixed(r.var_95_pct, 2)}%`}
        hint="95% 的交易日里，单日亏损不会超过这个数。" />
      <Cell label="夏普" v={fixed(r.sharpe, 2)}
        hint="按已实现净值算的风险调整后收益，无风险利率 3%。" />
      <Cell label="超额" v={signed(alpha, 1, "%")}
        tone={(alpha ?? 0) >= 0 ? "up" : "down"}
        hint="组合累计收益减去同期基准累计收益。" />
      <span className="col-span-full label">
        由每晚净值计算写入，已积累 {r.days} 个估值日。
      </span>
    </div>
  );
}

function Cell({ label, v, hint, tone }: {
  label: string; v: string; hint?: string; tone?: "up" | "down";
}) {
  return (
    <span className="flex items-baseline gap-1.5" title={hint}>
      <span className="label shrink-0">{label}</span>
      <span className={`tnum font-medium ${
        tone === "up" ? "text-up" : tone === "down" ? "text-down" : ""}`}>
        {v}
      </span>
    </span>
  );
}

/** Allocation by stock and by industry, side by side. */
function Allocation({ d }: { d: import("../lib/types").FundDetail }) {
  const actual = new Map(d.drift.map((x) => [x.t, x.actual_pct]));
  const asOf = d.drift[0]?.as_of;

  return (
    <div className="grid gap-4 lg:grid-cols-2 items-start">
      <div className="flex flex-col gap-2">
        <h3 className="text-[13.5px] font-semibold">📋 按个股</h3>
        <div className="flex flex-col gap-1">
          {d.holdings.map((h) => {
            const now = actual.get(h.t);
            return (
              <div key={h.t} className="flex items-center gap-2 text-[12.5px]">
                <span className="w-20 sm:w-24 shrink-0 truncate" title={h.n}>
                  {h.n}
                </span>
                <span className="hidden md:block w-20 shrink-0 font-mono tnum
                  text-[11px] text-ink-mute">
                  {h.t}
                </span>
                <div className="flex-1 h-4 rounded-sm bg-sunken overflow-hidden relative">
                  <div className="h-full rounded-sm bg-cyan/40"
                    style={{ width: `${Math.min(100, h.weight_pct)}%` }} />
                  {now != null && (
                    <div className="absolute top-0 h-full w-[2px] bg-cyan"
                      style={{ left: `${Math.min(100, now)}%` }}
                      title={`当前实际 ${fixed(now, 1)}%`} />
                  )}
                </div>
                <span className="w-14 text-right tnum">{fixed(h.weight_pct, 1)}%</span>
                <span className="w-14 text-right tnum text-[11.5px] text-ink-mute">
                  {now == null ? "—" : `${fixed(now, 1)}%`}
                </span>
              </div>
            );
          })}
        </div>
        <p className="label">
          浅色是建仓时的目标权重，竖线是当前实际权重
          {asOf ? `（截至 ${asOf}）` : ""}。
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <h3 className="text-[13.5px] font-semibold">🏭 按行业</h3>
        <Industries d={d.industries} />
      </div>
    </div>
  );
}
