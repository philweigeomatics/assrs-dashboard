/**
 * 自选股 — the two watchlists, side by side but never mixed.
 *
 * A-shares and North America are one table in the database (the canonical
 * symbol already says which market a row is: a bare six-digit code is an
 * A-share, "US:AAPL" is not) and two lists to a person. They close nine hours
 * apart, price in different currencies, draw their candles in opposite colours
 * and are scanned by different jobs — so a single merged list would be a table
 * whose rows mean different things on every column that matters.
 *
 * Hence a toggle rather than a market column. The add box is scoped to
 * whichever list is open, so searching from the A-share tab cannot offer you
 * NVDA and then fail on submit.
 *
 * Supply-chain graphs work for either market. The graph is a model's account
 * of what a company makes and who buys it, which is answerable for a US
 * listing as readily as a Chinese one — only the filings it should consult
 * differ, and supply_chain.py picks those per market.
 */

import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Sparkline } from "../components/Sparkline";
import { fixed, signed } from "../lib/format";
import { api, ApiError } from "../lib/api";
import type { MarketCode, StockRef, WatchRef, WatchBoard, WatchRow } from "../lib/types";
import { NavBar } from "../components/NavBar";
import { SupplyChainWindow } from "../components/SupplyChainWindow";
import { useSymbolSearch, MIN_REMOTE_CHARS } from "../lib/useSymbolSearch";
import { usePersistentState } from "../lib/usePersistentState";

type Tab = "CN" | "NA";

const TABS: { id: Tab; label: string; hint: string; markets: MarketCode[] }[] = [
  { id: "CN", label: "🇨🇳 A 股", markets: ["CN"],
    hint: "每晚 20:00 北京时间扫描（Tushare）" },
  { id: "NA", label: "🇺🇸🇨🇦 美股 / 加股", markets: ["US", "CA"],
    hint: "收盘后扫描（Yahoo，已复权）" },
];

export function Watchlist() {
  useEffect(() => { document.title = "ASSRS · 自选股"; }, []);
  const qc = useQueryClient();
  const [tab, setTab] = usePersistentState<Tab>("assrs.wl.tab", "CN");
  const [chain, setChain] = useState<string | null>(null);

  const active = TABS.find((t) => t.id === tab)!;
  const list = useQuery({
    queryKey: ["watchlist", tab],
    queryFn: () => api.watchlist(tab),
    staleTime: 60_000,
  });

  // Prices, moves, shape and the nightly scan. One payload for both markets:
  // the tabs filter a list the client already holds rather than refetching.
  const board = useQuery({
    queryKey: ["watchlist", "board"],
    queryFn: api.watchlistBoard,
    staleTime: 10 * 60_000,
  });
  const [sort, setSort] = usePersistentState<SortKey>("assrs.wl.sort", "name");

  const remove = useMutation({
    mutationFn: (t: string) => api.watchlistRemove(t),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["watchlist"] }),
  });

  return (
    <div className="min-h-screen">
      <NavBar />
      <main className="max-w-[1100px] mx-auto px-3 py-3 flex flex-col gap-3">
        <section className="card p-3 flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-[14.5px] font-semibold">⭐ 自选股</h2>
            <div className="flex rounded-lg bg-sunken p-0.5">
              {TABS.map((t) => (
                <button key={t.id} onClick={() => setTab(t.id)}
                  className={`px-2.5 h-7 rounded-md text-[12.5px] font-medium transition-colors ${
                    tab === t.id ? "bg-panel text-ink shadow-sm" : "text-ink-mute hover:text-ink"}`}>
                  {t.label}
                </button>
              ))}
            </div>
            <span className="label ml-auto">{active.hint}</span>
          </div>

          <AddBox tab={tab} markets={active.markets}
            existing={new Set((list.data ?? []).map((r) => r.t))} />

          {list.isPending && <div className="py-10 text-center label">加载中…</div>}
          {list.isError && (
            <div className="py-6 text-center flex flex-col gap-1.5">
              <p className="text-[12.5px] text-up">{(list.error as ApiError).message}</p>
              <button onClick={() => list.refetch()} className="text-cyan text-[13px]">重试</button>
            </div>
          )}
          {list.data && (
            list.data.length === 0
              ? <p className="label py-8 text-center">
                  这个列表还是空的。用上面的搜索框加入第一只。
                </p>
              : <Table rows={list.data} board={board.data} sort={sort} onSort={setSort}
            upIsRed={tab === "CN"} onChain={setChain}
                  onRemove={(t) => remove.mutate(t)}
                  removing={remove.isPending ? String(remove.variables) : null} />
          )}
          {remove.isError && (
            <p className="text-[12.5px] text-up">{(remove.error as ApiError).message}</p>
          )}
        </section>
      </main>

      {chain && <ChainWindow ticker={chain} onClose={() => setChain(null)} />}
    </div>
  );
}

/**
 * The add box, scoped to the open tab's markets.
 *
 * Scoped rather than universal: offering an A-share while the North American
 * list is open produces a row the user then has to find in the other tab to
 * delete. useSymbolSearch already takes the scope — it was built for the
 * comparison box, which has the same problem for a different reason.
 */
function AddBox({ tab, markets, existing }: {
  tab: Tab; markets: MarketCode[]; existing: Set<string>;
}) {
  const qc = useQueryClient();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  const stocks = useQuery({
    queryKey: ["stocks"], queryFn: api.stocks,
    staleTime: 6 * 3600_000, enabled: tab === "CN",
  });
  const { items, searching } = useSymbolSearch({
    query, stocks: stocks.data ?? [], markets, limit: 10,
  });

  const add = useMutation({
    mutationFn: (s: StockRef) => api.watchlistAdd(s.t),
    onSuccess: () => {
      setQuery("");
      setOpen(false);
      qc.invalidateQueries({ queryKey: ["watchlist"] });
    },
  });

  const hint = tab === "CN"
    ? "输入代码或名称，例如 600519 或 贵州茅台"
    : `输入代码或公司名，例如 AAPL 或 SHOP.TO（至少 ${MIN_REMOTE_CHARS} 个字符）`;

  return (
    <div className="flex flex-col gap-1 relative">
      <input value={query} placeholder={hint}
        onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => { if (e.key === "Escape") setOpen(false); }}
        className="h-9 px-2.5 rounded-lg bg-sunken text-[13px] outline-none
                   focus:ring-2 focus:ring-cyan/40" />

      {open && query.trim() !== "" && (
        <div className="absolute top-10 left-0 right-0 z-30 card p-1 max-h-[300px] overflow-auto">
          {searching && items.length === 0 && (
            <div className="px-2 py-2 label">搜索中…</div>
          )}
          {!searching && items.length === 0 && (
            <div className="px-2 py-2 label">没有匹配的股票。</div>
          )}
          {items.map((s) => {
            const already = existing.has(s.t);
            return (
              <button key={s.t} disabled={already || add.isPending}
                onClick={() => add.mutate({ t: s.t, n: s.n })}
                className={`w-full text-left px-2 py-1.5 rounded-md text-[13px]
                  flex items-baseline gap-2 ${already ? "opacity-50" : "hover:bg-sunken"}`}>
                <span className="font-mono font-semibold">{s.t}</span>
                <span className="truncate">{s.n}</span>
                <span className="ml-auto text-[11.5px] text-ink-mute shrink-0">
                  {already ? "已在列表中" : "加入"}
                </span>
              </button>
            );
          })}
        </div>
      )}
      {add.isError && (
        <span className="text-[12.5px] text-up">{(add.error as ApiError).message}</span>
      )}
    </div>
  );
}

type SortKey = "name" | "chg" | "chg5" | "chg20" | "signals";

const SORTS: { id: SortKey; label: string }[] = [
  { id: "name", label: "名称" },
  { id: "chg", label: "今日" },
  { id: "chg5", label: "5日" },
  { id: "chg20", label: "20日" },
  { id: "signals", label: "信号" },
];

/** Biggest mover first; anything without a price sinks rather than sorting as 0. */
function compare(a: WatchRow, b: WatchRow, key: SortKey): number {
  const num = (r: WatchRow, k: keyof WatchRow) => {
    const v = r[k];
    return typeof v === "number" ? v : null;
  };
  const by = (x: number | null, y: number | null) => {
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    return y - x;
  };
  switch (key) {
    case "chg": return by(num(a, "chg_pct"), num(b, "chg_pct"));
    case "chg5": return by(num(a, "chg_5d_pct"), num(b, "chg_5d_pct"));
    case "chg20": return by(num(a, "chg_20d_pct"), num(b, "chg_20d_pct"));
    case "signals": return (b.signal_count || 0) - (a.signal_count || 0)
      || a.n.localeCompare(b.n);
    default: return a.n.localeCompare(b.n);
  }
}

function Table({ rows, board, sort, onSort, onChain, onRemove, removing, upIsRed }: {
  rows: WatchRef[];
  board: WatchBoard | undefined;
  sort: SortKey;
  onSort: (k: SortKey) => void;
  onChain: (t: string) => void;
  onRemove: (t: string) => void;
  removing: string | null;
  upIsRed: boolean;
}) {
  // The board is the whole watchlist; the tab decides which slice shows. A
  // name with no board row yet still appears, so adding a stock does not make
  // it vanish until the next fetch.
  const merged = useMemo(() => {
    const by = new Map((board?.rows ?? []).map((r) => [r.t, r]));
    const out: (WatchRef & { d?: WatchRow })[] =
      rows.map((r) => ({ ...r, d: by.get(r.t) }));
    out.sort((x, y) => {
      if (!x.d && !y.d) return x.n.localeCompare(y.n);
      if (!x.d) return 1;
      if (!y.d) return -1;
      return compare(x.d, y.d, sort);
    });
    return out;
  }, [rows, board, sort]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="label">排序</span>
        {SORTS.map((o) => (
          <button key={o.id} onClick={() => onSort(o.id)}
            className={`h-6 px-2 rounded-md text-[12px] transition-colors ${
              sort === o.id ? "bg-elevated text-ink font-medium"
                : "bg-sunken text-ink-mute hover:text-ink"}`}>
            {o.label}
          </button>
        ))}
        {board && (
          <span className="ml-auto label tnum">
            {board.counts.priced}/{board.counts.total} 有行情
            {board.scan_date && ` · 扫描 ${board.scan_date}`}
          </span>
        )}
      </div>

      <div className="overflow-auto rounded-lg border border-line">
        <table className="w-full border-collapse text-[13px]">
          <thead className="bg-panel">
            <tr className="border-b border-line text-ink-mute text-[12px]">
              <th className="text-left font-medium px-2 py-1.5">代码</th>
              <th className="text-left font-medium px-2 py-1.5">名称</th>
              <th className="text-right font-medium px-2 py-1.5">最新</th>
              <th className="text-right font-medium px-2 py-1.5">今日</th>
              <th className="text-right font-medium px-2 py-1.5">5日</th>
              <th className="text-right font-medium px-2 py-1.5">20日</th>
              <th className="text-center font-medium px-2 py-1.5">近 60 日</th>
              <th className="text-left font-medium px-2 py-1.5">信号</th>
              <th className="text-right font-medium px-2 py-1.5">操作</th>
            </tr>
          </thead>
          <tbody>
            {merged.map((r) => {
              const d = r.d;
              const live = d?.session && !d.session.complete;
              return (
                <tr key={r.t} className="border-b border-line/60 hover:bg-sunken">
                  <td className="px-2 py-1.5 whitespace-nowrap">
                    <Link to={`/?t=${encodeURIComponent(r.t)}`}
                      className="text-cyan font-mono font-semibold">{r.t}</Link>
                  </td>
                  <td className="px-2 py-1.5 max-w-[200px] truncate" title={r.n}>
                    {r.n}
                  </td>
                  <td className="px-2 py-1.5 text-right tnum whitespace-nowrap">
                    {d?.price == null ? "—" : fixed(d.price, 2)}
                    {live && (
                      <span className="ml-1 text-[10px] text-brand-ink"
                        title={`盘中 —— 当地时间 ${d!.session!.local_time}，`
                          + `这根K线走完了 ${Math.round(d!.session!.elapsed_pct)}%`}>
                        ●
                      </span>
                    )}
                  </td>
                  <Move v={d?.chg_pct} upIsRed={upIsRed} bold />
                  <Move v={d?.chg_5d_pct} upIsRed={upIsRed} />
                  <Move v={d?.chg_20d_pct} upIsRed={upIsRed} />
                  <td className="px-2 py-1 align-middle">
                    <div className="flex justify-center">
                      {d?.spark?.length
                        ? <Sparkline values={d.spark} upIsRed={upIsRed} />
                        : <span className="label">—</span>}
                    </div>
                  </td>
                  <td className="px-2 py-1.5 max-w-[220px]">
                    <div className="flex flex-wrap gap-1">
                      {(d?.signals ?? []).map((s, i) => (
                        <span key={i}
                          className={`px-1 py-px rounded text-[10.5px] whitespace-nowrap ${
                            s.dir === "bear" ? "bg-down/10 text-down"
                              : "bg-up/10 text-up"}`}>
                          {s.label}
                        </span>
                      ))}
                      {d?.rsi != null && (
                        <span className="px-1 py-px rounded text-[10.5px] bg-sunken
                          text-ink-mute whitespace-nowrap"
                          title="夜间扫描算出的 RSI(14)">
                          RSI {fixed(d.rsi, 0)}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-2 py-1.5 text-right whitespace-nowrap">
                    <button onClick={() => onChain(r.t)}
                      className="text-[12px] text-violet px-1.5">🔗 供应链</button>
                    <button onClick={() => onRemove(r.t)} disabled={removing === r.t}
                      className="text-[12px] text-ink-mute hover:text-up px-1.5 disabled:opacity-50">
                      {removing === r.t ? "移除中…" : "移除"}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Move({ v, upIsRed, bold }: {
  v: number | null | undefined; upIsRed: boolean; bold?: boolean;
}) {
  if (v == null) {
    return <td className="px-2 py-1.5 text-right label">—</td>;
  }
  const up = v > 0;
  const tone = v === 0 ? "text-ink-mute"
    : up === upIsRed ? "text-up" : "text-down";
  return (
    <td className={`px-2 py-1.5 text-right tnum whitespace-nowrap ${tone} ${
      bold ? "font-medium" : ""}`}>
      {signed(v, 2, "%")}
    </td>
  );
}

/**
 * Opens the saved graph, or offers to make one.
 *
 * "Nobody has generated this yet" is the normal state of a stock you added a
 * minute ago, so it gets a button rather than an error — and generating costs
 * a model call, which is why it is never automatic.
 */
function ChainWindow({ ticker, onClose }: { ticker: string; onClose: () => void }) {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["chain", ticker],
    queryFn: () => api.supplyChain(ticker),
    staleTime: 60 * 60_000,
  });
  const make = useMutation({
    mutationFn: () => api.equityGenerate(ticker, "supply-chain", false),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["chain", ticker] }),
  });

  const graph = useMemo(() => q.data ?? null, [q.data]);

  if (q.data?.generated) {
    return <SupplyChainWindow ticker={ticker} graph={q.data} onClose={onClose} />;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(0,0,0,0.35)" }} onClick={onClose}
      role="dialog" aria-modal="true" aria-label="供应链图">
      <div className="card max-w-[460px] w-full p-4 flex flex-col gap-3"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-baseline gap-2">
          <h2 className="text-[15px] font-semibold">🔗 供应链图</h2>
          <span className="label font-mono">{ticker}</span>
          <button onClick={onClose} className="ml-auto text-[13px] text-ink-mute">关闭 ✕</button>
        </div>

        {q.isPending && <p className="label py-4 text-center">读取中…</p>}
        {q.isError && (
          <p className="text-[12.5px] text-up">{(q.error as ApiError).message}</p>
        )}
        {graph && !graph.generated && (
          <>
            <p className="text-[12.5px] text-ink-dim leading-snug">
              还没有为这只股票生成供应链图。生成会调用一次模型（约 20–60 秒），
              结果会保存下来，下次直接打开。
            </p>
            <button onClick={() => make.mutate()} disabled={make.isPending}
              className="h-9 rounded-lg bg-violet text-white text-[13px] font-semibold
                         disabled:opacity-60">
              {make.isPending ? "生成中…（不要关闭）" : "生成供应链图"}
            </button>
            {make.isError && (
              <span className="text-[12.5px] text-up">{(make.error as ApiError).message}</span>
            )}
          </>
        )}
      </div>
    </div>
  );
}
