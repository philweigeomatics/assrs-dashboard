/**
 * 明日预判 — write down what you think the next bar does, and get marked.
 *
 * The value is not the note, it is that the note is FALSIFIABLE and scored by
 * something other than your memory. "About to break out" can never be wrong.
 * "Closes above 68.06 on volume below the last three days' average" is either
 * right or it isn't. Hindsight quietly rewrites the first kind into whatever
 * happened; it cannot touch the second.
 *
 * So each note carries prose AND a list of checkable claims, each marked
 * separately — because the useful finding is usually that you read levels well
 * and direction badly, which a single right/wrong would hide.
 *
 * Resolution happens on its own when a session later than the note exists. A
 * Friday note stays pending through the weekend with no calendar logic: the
 * absence of a bar IS the absence of a session.
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { AlertNote, NoteClaim, NoteScorecard } from "../lib/types";
import { fixed, signed } from "../lib/format";

const KINDS: { id: string; label: string; needs: ("value" | "value2" | "lookback")[] }[] = [
  { id: "direction_up", label: "收阳（收 > 开）", needs: [] },
  { id: "direction_down", label: "收阴（收 < 开）", needs: [] },
  { id: "close_above", label: "收盘高于", needs: ["value"] },
  { id: "close_below", label: "收盘低于", needs: ["value"] },
  { id: "close_between", label: "收盘介于", needs: ["value", "value2"] },
  { id: "change_between", label: "涨跌幅介于 %", needs: ["value", "value2"] },
  { id: "volume_below_avg", label: "缩量（低于前N日均量）", needs: ["lookback"] },
  { id: "volume_above_avg", label: "放量（高于前N日均量）", needs: ["lookback"] },
];

type Draft = { kind: string; value?: string; value2?: string; lookback?: string; label?: string };

export function AlertNotes({ ticker, name, scanDate }: {
  ticker: string; name: string; scanDate: string;
}) {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["alert-notes"],
    queryFn: () => api.notes(),
    staleTime: 60_000,
  });

  // Called on mount rather than behind a button: resolving is idempotent, and
  // a scorecard you have to remember to refresh is a scorecard you stop using.
  const resolve = useMutation({
    mutationFn: () => api.notesResolve(),
    onSuccess: (d) => qc.setQueryData(["alert-notes"], { notes: d.notes, scorecard: d.scorecard }),
  });

  const all = q.data?.notes ?? [];
  const mine = all.filter((n) => n.ticker === ticker);
  const [open, setOpen] = useState(false);

  return (
    <div className="border-t border-line pt-2 flex flex-col gap-2">
      <div className="flex items-baseline gap-2">
        <span className="label">明日预判</span>
        <button onClick={() => setOpen(!open)} className="text-[12px] text-cyan">
          {open ? "收起" : "写一条"}
        </button>
        <button onClick={() => resolve.mutate()} disabled={resolve.isPending}
          className="ml-auto text-[11.5px] text-ink-mute hover:text-ink disabled:opacity-60"
          title="用最新交易日结算所有待定预判">
          {resolve.isPending ? "结算中…" : "↻ 结算"}
        </button>
      </div>

      {open && (
        <Composer ticker={ticker} name={name} scanDate={scanDate}
          onDone={() => { setOpen(false); qc.invalidateQueries({ queryKey: ["alert-notes"] }); }} />
      )}

      {mine.length === 0 && !open && (
        <p className="text-[11.5px] text-ink-mute leading-snug">
          写下明天的具体预期（方向、价位、成交量），下一个交易日自动对照。
          模糊的判断无法证伪，也就学不到东西。
        </p>
      )}

      {mine.map((n) => <NoteCard key={n.id} n={n} />)}

      {q.data?.scorecard && q.data.scorecard.total > 0 && (
        <Scorecard s={q.data.scorecard} />
      )}
    </div>
  );
}

function Composer({ ticker, name, scanDate, onDone }: {
  ticker: string; name: string; scanDate: string; onDone: () => void;
}) {
  const [text, setText] = useState("");
  const [claims, setClaims] = useState<Draft[]>([{ kind: "direction_up" }]);

  const save = useMutation({
    mutationFn: () => api.noteCreate({
      ticker, scan_date: scanDate, note: text,
      predictions: claims.map((c) => ({
        kind: c.kind,
        value: c.value === undefined || c.value === "" ? undefined : Number(c.value),
        value2: c.value2 === undefined || c.value2 === "" ? undefined : Number(c.value2),
        lookback: c.lookback ? Number(c.lookback) : undefined,
        label: c.label,
      })),
    }),
    onSuccess: onDone,
  });

  const set = (i: number, patch: Partial<Draft>) =>
    setClaims(claims.map((c, j) => (j === i ? { ...c, ...patch } : c)));

  return (
    <div className="rounded-lg bg-sunken p-2.5 flex flex-col gap-2">
      <div className="label">
        对 {name} {scanDate} 收盘后的判断 · 结算用下一个交易日
      </div>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={2}
        placeholder="为什么这么想？（筹码结构好、接近突破…）"
        className="bg-panel rounded-md p-2 text-[12.5px] outline-none focus:ring-2 focus:ring-cyan/40" />

      {claims.map((c, i) => {
        const spec = KINDS.find((k) => k.id === c.kind)!;
        return (
          <div key={i} className="flex flex-wrap items-center gap-1.5">
            <select value={c.kind} onChange={(e) => set(i, { kind: e.target.value })}
              className="h-7 px-1.5 rounded-md bg-panel text-[12.5px] outline-none">
              {KINDS.map((k) => <option key={k.id} value={k.id}>{k.label}</option>)}
            </select>
            {spec.needs.includes("value") && (
              <input type="number" step="0.01" value={c.value ?? ""}
                onChange={(e) => set(i, { value: e.target.value })} placeholder="数值"
                className="h-7 w-24 px-1.5 rounded-md bg-panel text-[12.5px] font-mono tnum outline-none" />
            )}
            {spec.needs.includes("value2") && (
              <input type="number" step="0.01" value={c.value2 ?? ""}
                onChange={(e) => set(i, { value2: e.target.value })} placeholder="至"
                className="h-7 w-24 px-1.5 rounded-md bg-panel text-[12.5px] font-mono tnum outline-none" />
            )}
            {spec.needs.includes("lookback") && (
              <input type="number" min={1} max={20} value={c.lookback ?? "3"}
                onChange={(e) => set(i, { lookback: e.target.value })}
                className="h-7 w-16 px-1.5 rounded-md bg-panel text-[12.5px] font-mono tnum outline-none"
                title="前 N 日均量" />
            )}
            <input value={c.label ?? ""} onChange={(e) => set(i, { label: e.target.value })}
              placeholder="备注（如 5日均线）"
              className="h-7 flex-1 min-w-[110px] px-1.5 rounded-md bg-panel text-[12px] outline-none" />
            {claims.length > 1 && (
              <button onClick={() => setClaims(claims.filter((_, j) => j !== i))}
                className="text-ink-mute hover:text-ink px-1" aria-label="删除这条">✕</button>
            )}
          </div>
        );
      })}

      <div className="flex items-center gap-2">
        {claims.length < 8 && (
          <button onClick={() => setClaims([...claims, { kind: "close_above" }])}
            className="text-[12px] text-cyan">+ 再加一条</button>
        )}
        <button onClick={() => save.mutate()} disabled={save.isPending || claims.length === 0}
          className="ml-auto h-7 px-3 rounded-md bg-cyan text-white text-[12.5px] font-semibold disabled:opacity-60">
          {save.isPending ? "保存中…" : "保存预判"}
        </button>
      </div>
      {save.isError && <span className="text-[12px] text-up">{(save.error as Error).message}</span>}
    </div>
  );
}

function claimText(c: NoteClaim): string {
  const k = KINDS.find((x) => x.id === c.kind);
  const base = k?.label ?? c.kind;
  if (c.value != null && c.value2 != null) return `${base} ${fixed(c.value, 2)}–${fixed(c.value2, 2)}`;
  if (c.value != null) return `${base} ${fixed(c.value, 2)}`;
  if (c.lookback != null) return `${base.replace("N", String(c.lookback))}`;
  return base;
}

function NoteCard({ n }: { n: AlertNote }) {
  const qc = useQueryClient();
  const del = useMutation({
    mutationFn: () => api.noteDelete(n.id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["alert-notes"] }),
  });
  const o = n.outcome;

  return (
    <div className="rounded-lg bg-sunken px-2.5 py-2 flex flex-col gap-1">
      <div className="flex items-baseline gap-2 text-[11.5px]">
        <span className="font-mono tnum">{n.scan_date}</span>
        {o ? (
          <span className={o.score_pct === 100 ? "text-up" : o.score_pct === 0 ? "text-down" : ""}>
            → {o.resolved ?? n.resolved_date} · {o.hits}/{o.decided} 命中
          </span>
        ) : (
          <span className="text-brand-ink">待结算</span>
        )}
        <button onClick={() => del.mutate()} className="ml-auto text-ink-mute hover:text-ink"
          aria-label="删除">✕</button>
      </div>

      {n.note && <p className="text-[12px] text-ink-dim leading-snug">{n.note}</p>}

      <div className="flex flex-col gap-0.5">
        {(o?.claims ?? n.predictions).map((c, i) => {
          const hit = (c as NoteClaim).hit;
          const mark = hit === true ? "✓" : hit === false ? "✗" : "·";
          const tone = hit === true ? "text-up" : hit === false ? "text-down" : "text-ink-mute";
          return (
            <div key={i} className="flex items-baseline gap-1.5 text-[11.5px]">
              <span className={tone}>{mark}</span>
              <span>{claimText(c as NoteClaim)}</span>
              {(c as NoteClaim).label && (
                <span className="text-ink-mute">· {(c as NoteClaim).label}</span>
              )}
              {(c as NoteClaim).actual && (
                <span className="ml-auto font-mono tnum text-ink-mute">
                  {(c as NoteClaim).actual}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Scorecard({ s }: { s: NoteScorecard }) {
  const thin = s.total < s.meaningful_at;
  return (
    <div className="rounded-lg border border-line px-2.5 py-2 flex flex-col gap-1">
      <div className="flex items-baseline gap-2">
        <span className="text-[12px] font-medium">预判记录</span>
        <span className="label">{s.hits}/{s.total} · {fixed(s.rate_pct, 0)}%</span>
      </div>
      {s.rows.map((r) => (
        <div key={r.kind} className="flex items-baseline gap-2 text-[11.5px]">
          <span className="truncate flex-1">{r.label}</span>
          <span className="font-mono tnum text-ink-mute">{r.hits}/{r.n}</span>
          <span className="font-mono tnum w-12 text-right">{fixed(r.rate_pct, 0)}%</span>
          {/* The only number that means anything: how far above the naive rule. */}
          <span className={`font-mono tnum w-16 text-right ${
            (r.edge_pp ?? 0) > 0 ? "text-up" : (r.edge_pp ?? 0) < 0 ? "text-down" : "text-ink-mute"}`}
            title={`基准 ${fixed(r.baseline_pct, 0)}%：盲猜同样的判断会有的命中率`}>
            {r.edge_pp == null ? "—" : `${signed(r.edge_pp, 0)}pp`}
          </span>
        </div>
      ))}
      <p className="text-[11px] text-ink-mute leading-snug">
        {thin
          ? `样本还太少（${s.total} 条，约 ${s.meaningful_at} 条起才看得出名堂）——现在的胜率基本是运气。`
          : "右侧为相对基准的超额命中率：基准是「盲猜同一判断」的历史频率。只有它显著为正才说明你真的读懂了什么。"}
      </p>
    </div>
  );
}
