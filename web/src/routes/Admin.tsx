/**
 * ⚙️ 板块管理 — who is in which sector.
 *
 * A sector's stock list is the input to its PPI, and the PPI drives the
 * heatmap, the breadth grid, the rotation map and the regime score that
 * every user sees. So this page is not a personal setting, and it says so:
 * each write is confirmed against a named sector, removals show what they
 * will leave behind, and nothing here pretends an edit is local.
 *
 * The nav link is hidden for non-admins, and that is a convenience. The
 * actual gate is on the server — see _require_admin — because hiding a link
 * does not stop a request.
 */

import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import type { AdminSector, NewSectorResult, StockRef } from "../lib/types";
import { NavBar } from "../components/NavBar";
import { useSymbolSearch } from "../lib/useSymbolSearch";

export function Admin() {
  const qc = useQueryClient();
  const sectors = useQuery({ queryKey: ["admin", "sectors"],
                             queryFn: api.adminSectors });
  const [open, setOpen] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const refresh = () => qc.invalidateQueries({ queryKey: ["admin", "sectors"] });
  const d = sectors.data;

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <NavBar />
      <main className="max-w-[1100px] mx-auto px-3 py-4 flex flex-col gap-3">
        <section className="card p-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h2 className="text-[14px] font-semibold">⚙️ 板块管理</h2>
          <span className="label">
            改动会影响所有人 —— 板块成分决定 PPI，PPI 决定热力图、趋势、轮动和评分
          </span>
          {d && (
            <span className="ml-auto label tnum">
              {d.sectors.length} 个板块 · {d.total_stocks} 只股票
            </span>
          )}
        </section>

        {sectors.isPending && (
          <div className="card p-8 text-center label">读取中…</div>
        )}
        {sectors.isError && (
          <div className="card p-4 text-center text-up text-[13px]">
            {(sectors.error as ApiError).message}
          </div>
        )}

        {d && (
          <>
            <section className="card p-3 flex flex-col gap-2">
              <div className="flex items-baseline gap-3">
                <h3 className="text-[13.5px] font-semibold">板块与成分股</h3>
                <button onClick={() => setCreating(!creating)}
                  className="ml-auto h-8 px-3 rounded-lg bg-cyan text-white text-[13px] font-semibold">
                  {creating ? "取消" : "＋ 新建板块"}
                </button>
              </div>

              {creating && (
                <NewSector existing={d.sectors.map((s) => s.name)}
                  minStocks={d.min_stocks}
                  onDone={() => { setCreating(false); refresh(); }} />
              )}

              <div className="flex flex-col divide-y divide-line">
                {d.sectors.map((s) => (
                  <SectorRow key={s.name} s={s} minStocks={d.min_stocks}
                    open={open === s.name}
                    onToggle={() => setOpen(open === s.name ? null : s.name)}
                    onChanged={refresh} />
                ))}
              </div>
            </section>

            <p className="label leading-snug">
              删除是软删除 —— 记录保留，可以看到什么时候移出的，也能再加回来。
              改完成分后，PPI 要等下一次夜间重建才会反映出来。
            </p>
          </>
        )}
      </main>
    </div>
  );
}

/**
 * Type, pick one, done.
 *
 * The shared StockPicker carries the analysis page's contract — recent
 * history, a current selection, keyboard navigation through both — and none
 * of that applies to adding a constituent. Same pattern as the pickers on
 * 配对交易 and 从自选股中搜索.
 */
function Pick({ exclude, placeholder, onPick }: {
  exclude: string[]; placeholder: string; onPick: (s: StockRef) => void;
}) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const stocks = useQuery({ queryKey: ["stocks"], queryFn: api.stocks,
                            staleTime: 6 * 3600_000 });
  const { items } = useSymbolSearch({
    query: q, stocks: stocks.data ?? [], markets: ["CN"], limit: 8,
  });
  const skip = new Set(exclude);
  const hits = items.filter((x) => !skip.has(x.t));

  return (
    <div className="relative">
      <input value={q} onChange={(e) => { setQ(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onBlur={() => globalThis.setTimeout(() => setOpen(false), 150)}
        placeholder={placeholder}
        className="h-7 w-48 px-2 rounded-md bg-panel text-[12.5px] outline-none
          focus:ring-2 focus:ring-cyan/40" />
      {open && hits.length > 0 && (
        <div className="card absolute left-0 mt-1 z-30 w-64 py-1">
          {hits.map((x) => (
            <button key={x.t} onMouseDown={(e) => e.preventDefault()}
              onClick={() => { onPick({ t: x.t, n: x.n }); setQ(""); setOpen(false); }}
              className="w-full flex items-baseline gap-2 px-2 py-1 text-left hover:bg-elevated">
              <span className="font-mono tnum text-[12px] text-ink-dim shrink-0">{x.t}</span>
              <span className="text-[12.5px] truncate">{x.n}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}


function SectorRow({ s, minStocks, open, onToggle, onChanged }: {
  s: AdminSector; minStocks: number; open: boolean;
  onToggle: () => void; onChanged: () => void;
}) {
  const [err, setErr] = useState("");
  const atFloor = s.stocks.length <= minStocks;

  const add = useMutation({
    mutationFn: (t: string) => api.adminAddStock(s.name, t),
    onSuccess: () => { setErr(""); onChanged(); },
    onError: (e) => setErr((e as ApiError).message),
  });
  const remove = useMutation({
    mutationFn: (t: string) => api.adminRemoveStock(s.name, t),
    onSuccess: () => { setErr(""); onChanged(); },
    onError: (e) => setErr((e as ApiError).message),
  });
  const busy = add.isPending || remove.isPending;

  return (
    <div className="py-2 flex flex-col gap-2">
      <button onClick={onToggle} className="flex items-baseline gap-2 text-left">
        <span className="text-[13px] font-medium">{s.name}</span>
        <span className={`label tnum ${atFloor ? "text-brand-ink" : ""}`}>
          {s.stocks.length} 只
          {atFloor && `（已到下限 ${minStocks}）`}
        </span>
        {s.removed.length > 0 && (
          <span className="label">· 已移出 {s.removed.length}</span>
        )}
        <span className="ml-auto text-[12px] text-cyan">{open ? "收起" : "展开"}</span>
      </button>

      {open && (
        <div className="flex flex-col gap-2 pl-1">
          <div className="flex flex-wrap gap-1.5">
            {s.stocks.map((x) => (
              <span key={x.t}
                className="h-7 pl-2 pr-1 rounded-md bg-sunken flex items-center gap-1.5 text-[12.5px]">
                <span className="truncate max-w-[120px]">{x.n || x.t}</span>
                <span className="font-mono tnum text-[11px] text-ink-mute">{x.t}</span>
                <button disabled={busy || atFloor}
                  onClick={() => remove.mutate(x.t)}
                  title={atFloor
                    ? `最少要保留 ${minStocks} 只 —— 再少，板块指数就只是单只股票`
                    : `把 ${x.n || x.t} 移出 ${s.name}`}
                  aria-label={`移出 ${x.n || x.t}`}
                  className="px-1 text-ink-mute hover:text-up disabled:opacity-30
                    disabled:hover:text-ink-mute">✕</button>
              </span>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="label">加入成分股</span>
            <Pick exclude={s.stocks.map((x) => x.t)}
              placeholder="输入代码或名称…"
              onPick={(p) => add.mutate(p.t)} />
            {busy && <span className="label">保存中…</span>}
          </div>

          {err && <p className="text-[12.5px] text-up">{err}</p>}

          {s.removed.length > 0 && (
            <details className="text-[12px]">
              <summary className="label cursor-pointer">
                已移出的 {s.removed.length} 只（可以再加回来）
              </summary>
              <div className="flex flex-wrap gap-1.5 mt-1.5">
                {s.removed.map((x) => (
                  <button key={x.t} disabled={busy}
                    onClick={() => add.mutate(x.t)}
                    title={`${x.removed_at || "—"} 移出 · 点击加回`}
                    className="h-7 px-2 rounded-md bg-sunken flex items-center gap-1.5
                      text-[12px] text-ink-dim hover:text-ink">
                    <span>{x.n || x.t}</span>
                    <span className="font-mono tnum text-[11px] text-ink-mute">{x.t}</span>
                    <span className="text-cyan">＋</span>
                  </button>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

function NewSector({ existing, minStocks, onDone }: {
  existing: string[]; minStocks: number; onDone: () => void;
}) {
  const [name, setName] = useState("");
  const [picked, setPicked] = useState<StockRef[]>([]);
  const [result, setResult] = useState<NewSectorResult | null>(null);
  const [err, setErr] = useState("");

  const taken = useMemo(() => new Set(existing), [existing]);
  const create = useMutation({
    mutationFn: () => api.adminCreateSector(name.trim(), picked.map((p) => p.t)),
    onSuccess: (r) => {
      setErr("");
      setResult(r);
      // Only a real creation clears the form; a pending SQL step means the
      // admin still needs these inputs to try again.
      if (r.created) { setName(""); setPicked([]); onDone(); }
    },
    onError: (e) => { setResult(null); setErr((e as ApiError).message); },
  });

  const nameTaken = taken.has(name.trim());
  const ready = name.trim().length > 0 && !nameTaken
    && picked.length >= minStocks && !create.isPending;

  return (
    <div className="rounded-lg bg-sunken p-2.5 flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <label className="label">板块名</label>
        <input value={name} onChange={(e) => setName(e.target.value)}
          placeholder="储能 / Energy_Storage"
          className="h-7 w-48 px-2 rounded-md bg-panel text-[12.5px] outline-none
            focus:ring-2 focus:ring-cyan/40" />
        <span className="label">
          会成为数据表名的一部分，不能有空格和标点
        </span>
      </div>
      {nameTaken && (
        <p className="text-[12.5px] text-up">「{name.trim()}」已存在</p>
      )}

      <div className="flex flex-wrap items-center gap-1.5">
        {picked.map((p) => (
          <span key={p.t}
            className="h-7 pl-2 pr-1 rounded-md bg-panel flex items-center gap-1.5 text-[12.5px]">
            <span className="truncate max-w-[110px]">{p.n}</span>
            <span className="font-mono tnum text-[11px] text-ink-mute">{p.t}</span>
            <button onClick={() => setPicked(picked.filter((x) => x.t !== p.t))}
              aria-label={`移除 ${p.n}`}
              className="px-1 text-ink-mute hover:text-ink">✕</button>
          </span>
        ))}
        <Pick exclude={picked.map((p) => p.t)}
          placeholder={picked.length ? "再加一只…" : "输入代码或名称…"}
          onPick={(p) => setPicked([...picked, p])} />
      </div>

      <div className="flex items-center gap-2">
        <button onClick={() => create.mutate()} disabled={!ready}
          className="h-8 px-3 rounded-lg bg-cyan text-white text-[13px] font-semibold disabled:opacity-60">
          {create.isPending ? "创建中…" : "创建板块"}
        </button>
        <span className="label">
          {picked.length < minStocks
            ? `至少 ${minStocks} 只股票（现在 ${picked.length} 只）`
            : `${picked.length} 只股票`}
        </span>
      </div>

      {err && <p className="text-[12.5px] text-up">{err}</p>}

      {result && !result.created && result.sql.length > 0 && (
        /* Supabase cannot create a table from the client key, so nothing was
           written. Saying "done" here would leave a sector every rebuild
           fails on. */
        <div className="flex flex-col gap-1.5">
          <p className="text-[12.5px] text-brand-ink leading-snug">
            <b>还没有创建。</b> Supabase 需要先手动建表 ——
            把下面的 SQL 贴进 Supabase SQL editor 执行，然后再点一次「创建板块」。
          </p>
          {result.sql.map((q, i) => (
            <pre key={i}
              className="text-[11.5px] font-mono bg-panel rounded-md p-2 overflow-x-auto">
              {q}
            </pre>
          ))}
          <button onClick={() => globalThis.navigator?.clipboard?.writeText(
            result.sql.join("\n"))}
            className="self-start h-7 px-2 rounded-md bg-panel text-[12px]">
            复制 SQL
          </button>
        </div>
      )}
    </div>
  );
}
