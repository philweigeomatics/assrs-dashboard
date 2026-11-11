/**
 * 候选 — save the weights on screen under a name, bring them back later.
 *
 * The point is comparison: every candidate gets its own line in 历史模拟,
 * next to equal weight and the index, so "最大夏普" and "每个行业一只" can be
 * read against each other over the same days. Loading one puts its weights
 * back in the panel and recalculates; changing it and pressing 更新 writes
 * the new weights over the old ones.
 */

import { useState } from "react";
import type { PortfolioBuild } from "../../lib/types";
import { fixed } from "../../lib/format";
import { usePersistentState } from "../../lib/usePersistentState";
import { sameWeights, sumOf } from "../../lib/weightDraft";
import {
  MAX_CANDIDATES, colourOf, heldOnly, newId, nextName, outside, type Candidate,
} from "../../lib/candidates";
import type { AllocationState } from "./Allocation";

export function useCandidates(market: "CN" | "US") {
  // Two keys rather than one keyed by market: usePersistentState reads its
  // key once, so switching markets has to switch which state is used.
  const [cn, setCn] = usePersistentState<Candidate[]>("assrs.pf.cands.CN", []);
  const [us, setUs] = usePersistentState<Candidate[]>("assrs.pf.cands.US", []);
  const raw = market === "CN" ? cn : us;
  const set = market === "CN" ? setCn : setUs;
  const list = Array.isArray(raw) ? raw : [];
  return {
    list,
    add: (c: Candidate) => set((l) => [...(Array.isArray(l) ? l : []), c].slice(-MAX_CANDIDATES)),
    update: (id: string, patch: Partial<Candidate>) =>
      set((l) => l.map((c) => (c.id === id ? { ...c, ...patch } : c))),
    remove: (id: string) => set((l) => l.filter((c) => c.id !== id)),
  };
}

export type CandidatesState = ReturnType<typeof useCandidates>;

const describe = (c: Candidate) =>
  [`${c.source} · ${c.from} → ${c.to}`,
   ...Object.entries(c.weights)
     .sort((x, y) => y[1] - x[1])
     .map(([t, v]) => `${c.names[t] ?? t} ${fixed(v, 1)}%`)].join("\n");

export function CandidateBar({ d, a, cands, activeId, setActiveId }: {
  d: PortfolioBuild;
  a: AllocationState;
  cands: CandidatesState;
  activeId: string | null;
  setActiveId: (id: string | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const { list } = cands;
  const tickers = d.holdings.map((h) => h.t);
  const active = list.find((c) => c.id === activeId) ?? null;
  const draft = heldOnly(a.w);
  const edited = active != null && !sameWeights(active.weights, draft);
  const full = list.length >= MAX_CANDIDATES;
  const label = d.mode === "target" ? "目标组合" : d.mode_label;
  const taken = (except?: string) =>
    list.some((c) => c.name === name.trim() && c.id !== except);

  const start = () => {
    setName(active ? active.name
      : !a.changed && !list.some((c) => c.name === label) ? label
      : nextName(list));
    setOpen(true);
  };
  const stamp = () => ({
    weights: draft,
    names: { ...(active?.names ?? {}), ...Object.fromEntries(d.holdings.map((h) => [h.t, h.n])) },
    from: d.from, to: d.to,
    saved: new Date().toISOString().slice(0, 10),
  });
  const create = () => {
    const c: Candidate = {
      id: newId(), name: name.trim(),
      source: a.changed ? `${label} · 手调` : label, ...stamp(),
    };
    cands.add(c);
    setActiveId(c.id);
    setOpen(false);
  };
  const update = () => {
    if (!active) return;
    cands.update(active.id, { name: name.trim(), ...stamp() });
    setOpen(false);
  };
  const sum = sumOf(draft);

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="label" title="只存在这个浏览器里。真正要留的组合用「存为组合」。">
        候选
      </span>
      {list.length === 0 && !open && (
        <span className="label">
          把现在这组权重存成候选，历史模拟里会和等权、基准画在一起比
        </span>
      )}
      {list.map((c) => {
        const miss = outside(c, tickers);
        const on = c.id === activeId;
        return (
          <span key={c.id}
            className={`h-7 pl-2 pr-0.5 rounded-md flex items-center gap-1 text-[12px] ${
              on ? "bg-elevated ring-1 ring-cyan/60" : "bg-sunken"}`}>
            <button
              onClick={() => { setActiveId(c.id); a.load(c.weights); setOpen(false); }}
              disabled={miss.length > 0}
              title={miss.length
                ? `含现在的股票里没有的：${miss.map((t) => c.names[t] ?? t).join("、")}。` +
                  `载入不了，但历史模拟里照样画。\n${describe(c)}`
                : `点一下载入这组权重\n${describe(c)}`}
              className="flex items-center gap-1.5 disabled:cursor-not-allowed">
              <span className="w-2.5 h-2.5 rounded-full shrink-0"
                style={{ background: colourOf(list, c.id) }} />
              <span className={`truncate max-w-[120px] ${miss.length ? "text-ink-mute" : ""}`}>
                {c.name}
              </span>
              {on && edited && <span className="text-brand-ink">· 已改</span>}
            </button>
            <button
              onClick={() => {
                if (!globalThis.confirm(`删除候选「${c.name}」？`)) return;
                cands.remove(c.id);
                if (on) setActiveId(null);
              }}
              aria-label={`删除候选 ${c.name}`}
              className="text-ink-mute hover:text-ink px-1">✕</button>
          </span>
        );
      })}

      {!open ? (
        <button onClick={start} disabled={sum <= 0 || (full && !active)}
          title={full && !active ? `最多 ${MAX_CANDIDATES} 个，先删掉一个` : "把面板里现在的权重存成候选"}
          className="h-7 px-2 rounded-md bg-sunken text-[12px] font-medium disabled:opacity-50">
          📌 {active && edited ? "存改动…" : "存为候选"}
        </button>
      ) : (
        <span className="flex flex-wrap items-center gap-1.5">
          <input value={name} onChange={(e) => setName(e.target.value)} autoFocus
            maxLength={24} placeholder="候选名…" aria-label="候选名"
            onKeyDown={(e) => {
              if (e.key === "Escape") setOpen(false);
              if (e.key === "Enter" && name.trim()) {
                if (active && !taken(active.id)) update();
                else if (!active && !taken() && !full) create();
              }
            }}
            className="h-7 w-32 px-2 rounded-md bg-sunken text-[12px] outline-none
              focus:ring-2 focus:ring-cyan/40" />
          {active && (
            <button onClick={update} disabled={!name.trim() || taken(active.id)}
              className="h-7 px-2 rounded-md bg-cyan text-white text-[12px] font-semibold
                disabled:opacity-50">
              更新「{active.name}」
            </button>
          )}
          <button onClick={create} disabled={!name.trim() || taken() || full}
            title={full ? `最多 ${MAX_CANDIDATES} 个`
              : taken() ? "改个名字才能另存" : undefined}
            className={`h-7 px-2 rounded-md text-[12px] font-semibold disabled:opacity-50 ${
              active ? "bg-sunken" : "bg-cyan text-white"}`}>
            {active ? "另存为新候选" : "保存"}
          </button>
          <button onClick={() => setOpen(false)}
            className="h-7 px-2 rounded-md bg-sunken text-[12px]">取消</button>
          {name.trim() && taken(active?.id) && (
            <span className="text-[12px] text-up">已有同名候选</span>
          )}
          {Math.abs(sum - 100) > 0.1 && (
            <span className="text-[12px] text-brand-ink">合计 {fixed(sum, 2)}%，差的部分按现金算</span>
          )}
        </span>
      )}
    </div>
  );
}
