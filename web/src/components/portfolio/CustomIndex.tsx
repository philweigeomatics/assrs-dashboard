/**
 * 自建指数 — an index you define and watch.
 *
 * Members, a weighting method and a rebalance schedule; the level is the
 * server's ledger of units (custom_index.py), so it is continuous through
 * every rebalance, addition, removal and delisting, and each member's
 * contribution adds up exactly to the index move.
 *
 * Two honesty marks on the chart:
 *   * 回溯（假设）— the part before the rules were last set. Those members
 *     were picked knowing how they did; only what follows was watched.
 *   * The change log — every add, removal, delisting and rebalance with its
 *     turnover, the way an index provider publishes its notices.
 *
 * The rules are fixed once an index exists, as an index provider's are: a
 * rule changed afterwards would rewrite every day already watched. To see
 * the same members under another weighting, ⧉ 复制 makes a twin with the
 * same members over the same dates, and the two run side by side.
 *
 * One market per index, so one currency, one calendar and one benchmark.
 * Colours follow the market: A-shares red up, US green up.
 */

import { useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";
import type {
  CustomIndexDetail, CustomIndexList, CustomIndexRef, IndexMember, IndexMethod, IndexRebalance, IndexWindow, StockRef,
} from "../../lib/types";
import { useSymbolSearch } from "../../lib/useSymbolSearch";
import { usePersistentState } from "../../lib/usePersistentState";
import { fixed, moveClass, signed } from "../../lib/format";

const MARKETS = [{ v: "CN", label: "A 股" }, { v: "US", label: "美股" }] as const;
const WINDOWS: { v: IndexWindow; label: string }[] = [
  { v: "1d", label: "今日" }, { v: "5d", label: "5日" }, { v: "1m", label: "1月" },
  { v: "3m", label: "3月" }, { v: "all", label: "起始以来" },
];
const STATUS: Record<IndexMember["status"], { label: string; cls: string }> = {
  active: { label: "在列", cls: "bg-cyan/10 text-cyan" },
  halted: { label: "停牌", cls: "bg-brand/15 text-brand-ink" },
  pending: { label: "待加入", cls: "bg-sunken text-ink-mute" },
  nodata: { label: "无数据", cls: "bg-up/10 text-up" },
  removed: { label: "已移出", cls: "bg-sunken text-ink-mute" },
  delisted: { label: "已退市/被收购", cls: "bg-sunken text-ink-mute" },
};

const today = () => new Date().toISOString().slice(0, 10);
const defaultStart = (m: "CN" | "US") => (m === "CN" ? "2024-10-14" : `${new Date().getFullYear() - 1}-01-02`);
const bare = (t: string) => t.replace(/^US:/, "");

export function CustomIndex() {
  const qc = useQueryClient();
  const list = useQuery({ queryKey: ["idx", "list"], queryFn: api.indexes });
  const [picked, setPicked] = usePersistentState<number | null>("assrs.index.current", null);
  const [mode, setMode] = useState<"view" | "new" | "edit" | "copy">("view");
  const [confirming, setConfirming] = useState(false);

  const items = list.data?.indexes ?? [];
  const current = items.find((x) => x.id === picked) ?? items[0] ?? null;
  const remove = useMutation({
    mutationFn: () => api.deleteIndex(current!.id),
    onSuccess: () => { setConfirming(false); setPicked(null); qc.invalidateQueries({ queryKey: ["idx"] }); },
  });

  if (list.isPending) return <section className="card p-6 label text-center">读取自建指数…</section>;
  if (list.isError) {
    const e = list.error as ApiError;
    return <section className="card p-4 text-[13px] text-up">{e.message}</section>;
  }
  const meta = list.data!;
  const done = (id?: number) => {
    setMode("view");
    if (id) setPicked(id);
    qc.invalidateQueries({ queryKey: ["idx"] });
  };

  return (
    <div className="flex flex-col gap-3">
      <section className="card p-3 flex flex-wrap items-center gap-2">
        <h2 className="text-[14px] font-semibold">📊 自建指数</h2>
        {items.length > 0 && mode === "view" && (
          <select value={current?.id ?? ""} onChange={(e) => setPicked(Number(e.target.value))}
            className="h-8 px-2 rounded-md bg-sunken text-[13px] outline-none max-w-[260px]">
            {items.map((x) => (
              <option key={x.id} value={x.id}>
                {x.name} · {x.market === "CN" ? "A股" : "美股"} · {meta.methods[x.method]?.label}
              </option>
            ))}
          </select>
        )}
        <div className="ml-auto flex items-center gap-1.5">
          {mode === "view" && current && (
            <>
              <button onClick={() => setMode("edit")} title="改名或换对比指数"
                className="h-8 px-3 rounded-lg text-[12.5px] border border-line hover:bg-elevated">✎ 编辑</button>
              <button onClick={() => setMode("copy")} title="同样的成分、同样的日期，换一种加权方式"
                className="h-8 px-3 rounded-lg text-[12.5px] border border-line hover:bg-elevated">⧉ 复制</button>
              {confirming ? (
                <span className="flex items-center gap-1 text-[12.5px]">
                  删除「{current.name}」？
                  <button onClick={() => remove.mutate()} className="h-8 px-2 rounded-lg bg-up text-white">删除</button>
                  <button onClick={() => setConfirming(false)} className="h-8 px-2 rounded-lg border border-line">取消</button>
                </span>
              ) : (
                <button onClick={() => setConfirming(true)} className="h-8 px-3 rounded-lg text-[12.5px] border border-line hover:bg-elevated">删除</button>
              )}
            </>
          )}
          {mode === "view" && (
            <button onClick={() => setMode("new")} className="h-8 px-3 rounded-lg text-[12.5px] bg-cyan text-white">＋ 新建指数</button>
          )}
        </div>
      </section>

      {mode === "new" && <IndexForm meta={meta} kind="new" onDone={done} onCancel={() => setMode("view")} />}
      {(mode === "edit" || mode === "copy") && current && (
        <IndexForm key={`${mode}-${current.id}`} meta={meta} kind={mode} source={current} onDone={done} onCancel={() => setMode("view")} />
      )}
      {mode === "view" && current && <IndexView key={current.id} idx={current} />}
      {mode === "view" && !current && (
        <section className="card p-6 text-[13px] text-ink-dim leading-relaxed">
          <p className="font-semibold text-ink mb-1">把你关注的一组股票做成一条指数</p>
          <p>选成分股、加权方式（等权 / 等资金买入持有 / 市值加权 / 市值加权带上限 / 价格加权 / 波动率倒数加权）、
            调仓频率和一个对比用的行业指数，
            系统会按指数公司的做法逐日记账：调仓、增删成分、停牌、被收购退市都不会让点位跳空，
            并告诉你每只股票对涨跌贡献了多少。A 股和美股分开建。</p>
        </section>
      )}
    </div>
  );
}

// ── define / edit ────────────────────────────────────────────────────────────
type Draft = { ticker: string; name: string };

function IndexForm({ meta, kind, source, onDone, onCancel }: {
  meta: CustomIndexList; kind: "new" | "edit" | "copy"; source?: CustomIndexRef;
  onDone: (id?: number) => void; onCancel: () => void;
}) {
  // edit: only the name and the comparison. copy: a new rule, same members, same start.
  const locked = kind === "edit";
  const sameStart = kind !== "new";
  const [name, setName] = useState(kind === "copy" ? `${source!.name}·副本` : source?.name ?? "");
  const [market, setMarket] = useState<"CN" | "US">(source?.market ?? "CN");
  const [method, setMethod] = useState<IndexMethod>(source?.method ?? "equal");
  const [rebalance, setRebalance] = useState<IndexRebalance>(source?.rebalance ?? "Q");
  const [cap, setCap] = useState(String(source?.cap_pct ?? 10));
  const [start, setStart] = useState(source?.base_date ?? defaultStart("CN"));
  const [bench, setBench] = useState(source?.bench ?? "");
  const [members, setMembers] = useState<Draft[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const save = useMutation({
    mutationFn: async () => {
      const rules = { name: name.trim(), method, base_date: start, bench: bench || null,
                      rebalance: method === "equal_hold" ? "never" as const : rebalance,
                      cap_pct: method === "cap_capped" ? Number(cap) : null };
      if (kind === "edit") return api.updateIndex(source!.id, { name: rules.name, bench: rules.bench });
      if (kind === "copy") {
        const { base_date: _, ...rest } = rules;
        return api.copyIndex(source!.id, rest);
      }
      return api.createIndex({ ...rules, market, members: members.map((m) => ({ ticker: m.ticker, name: m.name })) });
    },
    onSuccess: (r) => onDone(r.id),
    onError: (e) => setErr((e as Error).message),
  });

  const hold = method === "equal_hold";
  const backfill = kind !== "edit" && start < today();

  return (
    <section className="card p-3 flex flex-col gap-3 text-[13px]">
      <div className="font-semibold">
        {kind === "edit" ? `编辑「${source!.name}」` : kind === "copy" ? `复制「${source!.name}」为新指数` : "新建指数"}
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <Field label="名称">
          <input value={name} onChange={(e) => setName(e.target.value)} maxLength={40} placeholder="例如：光模块链"
            className="h-8 w-full px-2 rounded-md bg-sunken outline-none focus:ring-2 focus:ring-cyan/40" />
        </Field>
        <Field label="市场">
          <div className="flex rounded-md bg-sunken p-0.5 w-fit">
            {MARKETS.map((m) => (
              <button key={m.v} disabled={sameStart} onClick={() => {
                setMarket(m.v); setMembers([]); setBench("");
                if (!source) setStart(defaultStart(m.v));
              }}
                className={`h-7 px-3 rounded text-[12.5px] ${market === m.v ? "bg-panel shadow-sm font-semibold" : "text-ink-mute"} disabled:opacity-60`}>
                {m.label}
              </button>
            ))}
          </div>
        </Field>
        <Field label="起始日（基点 1000）">
          <input type="date" value={start} max={today()} min="2015-01-01" disabled={sameStart} onChange={(e) => setStart(e.target.value)}
            className="h-8 px-2 rounded-md bg-sunken outline-none disabled:opacity-60" />
        </Field>
        <Field label="加权方式" hint={meta.methods[method]?.hint}>
          <select value={method} disabled={locked} onChange={(e) => setMethod(e.target.value as IndexMethod)}
            className="h-8 w-full px-2 rounded-md bg-sunken outline-none disabled:opacity-60">
            {(Object.keys(meta.methods) as IndexMethod[]).map((k) => <option key={k} value={k}>{meta.methods[k].label}</option>)}
          </select>
        </Field>
        <Field label="调仓频率" hint={hold ? "买入持有不调仓" : "每个周期的第一个交易日收盘调回目标权重"}>
          <select value={hold ? "never" : rebalance} disabled={hold || locked} onChange={(e) => setRebalance(e.target.value as IndexRebalance)}
            className="h-8 w-full px-2 rounded-md bg-sunken outline-none disabled:opacity-60">
            {(Object.keys(meta.rebalance) as IndexRebalance[]).map((k) => <option key={k} value={k}>{meta.rebalance[k]}</option>)}
          </select>
        </Field>
        <Field label="对比指数" hint="图上的对比线和「相对」读数用它；换它不会重算指数">
          <select value={bench} onChange={(e) => setBench(e.target.value)}
            className="h-8 w-full px-2 rounded-md bg-sunken outline-none">
            <option value="">大盘（{market === "US" ? "SPY" : "沪深300"}）</option>
            {[...new Set(meta.benches[market].map((b) => b.group))].map((g) => (
              <optgroup key={g} label={g}>
                {meta.benches[market].filter((b) => b.group === g).map((b) => (
                  <option key={b.code} value={b.code}>{market === "US" ? `${b.code} ${b.name}` : b.name}</option>
                ))}
              </optgroup>
            ))}
          </select>
        </Field>
        {method === "cap_capped" && (
          <Field label="单只上限 %">
            <input type="number" min={2} max={50} value={cap} disabled={locked} onChange={(e) => setCap(e.target.value)}
              className="h-8 w-24 px-2 rounded-md bg-sunken outline-none disabled:opacity-60" />
          </Field>
        )}
      </div>

      {kind === "edit" && (
        <p className="label leading-snug">
          加权方式、调仓频率、上限和起始日在建立时就固定了 —— 改了会改写你已经看过的每一天，指数就不再是同一条。
          想用别的加权方式看同一批成分，用「⧉ 复制」建一个并排比较。
        </p>
      )}
      {kind === "copy" && (
        <p className="label leading-snug">
          复制「{source!.name}」的全部成分和它们的加入 / 移出日期（含已退市的记录），起始日同为 {source!.base_date}，
          两条指数只差在加权方式。
        </p>
      )}
      {kind === "new" && (
        <Field label={`成分股（${members.length}）`}>
          <MemberPicker market={market} exclude={members.map((m) => m.ticker)}
            onAdd={(s) => setMembers([...members, { ticker: s.t, name: s.n }])} />
          <div className="flex flex-wrap gap-1.5 mt-2">
            {members.map((m, i) => (
              <span key={m.ticker} className="h-7 pl-2 pr-1 rounded-md bg-sunken flex items-center gap-1.5 text-[12.5px]">
                <span className="truncate max-w-[120px]">{m.name}</span>
                <span className="font-mono text-[11px] text-ink-mute">{bare(m.ticker)}</span>
                <button onClick={() => setMembers(members.filter((_, j) => j !== i))} aria-label={`移除 ${m.name}`}
                  className="text-ink-mute hover:text-ink px-1">✕</button>
              </span>
            ))}
          </div>
        </Field>
      )}

      {backfill && (
        <p className="label leading-snug">
          起始日早于今天：今天之前的走势是用现在选定的{kind === "copy" ? "加权方式" : "成分"}回溯计算的，
          图上标为「回溯（假设）」—— 是事后选的，这段表现会天然偏好。
        </p>
      )}
      {err && <p className="text-[12.5px] text-up">{err}</p>}
      <div className="flex gap-2">
        <button onClick={() => { setErr(null); save.mutate(); }} disabled={save.isPending || !name.trim() || (kind === "new" && members.length < 2)}
          className="h-8 px-4 rounded-lg bg-cyan text-white text-[13px] disabled:opacity-50">
          {save.isPending ? "保存中…" : kind === "edit" ? "保存" : kind === "copy" ? "建立副本" : "建立指数"}
        </button>
        <button onClick={onCancel} className="h-8 px-4 rounded-lg border border-line text-[13px]">取消</button>
        {kind === "new" && members.length < 2 && <span className="label self-center">至少 2 只</span>}
      </div>
    </section>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 min-w-0">
      <span className="label">{label}</span>
      {children}
      {hint && <span className="label leading-snug">{hint}</span>}
    </label>
  );
}

function MemberPicker({ market, exclude, onAdd, placeholder }: {
  market: "CN" | "US"; exclude: string[]; onAdd: (s: StockRef) => void; placeholder?: string;
}) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const stocks = useQuery({ queryKey: ["stocks"], queryFn: api.stocks, staleTime: 6 * 3600_000 });
  const { items, searching } = useSymbolSearch({ query: q, stocks: stocks.data ?? [], markets: [market], limit: 8 });
  const hits = items.filter((s) => !exclude.includes(s.t));
  return (
    <div className="relative w-full max-w-[280px]">
      <input value={q} onChange={(e) => { setQ(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)} onBlur={() => globalThis.setTimeout(() => setOpen(false), 150)}
        placeholder={placeholder ?? (market === "CN" ? "输入代码或名称，如 中际旭创" : "输入代码或名称，如 MRVL")}
        className="h-8 w-full px-2 rounded-md bg-sunken text-[12.5px] outline-none focus:ring-2 focus:ring-cyan/40" />
      {open && (hits.length > 0 || searching) && (
        <div className="card absolute left-0 mt-1 z-30 w-full py-1">
          {hits.map((s) => (
            <button key={s.t} type="button" onMouseDown={(e) => e.preventDefault()}
              onClick={() => { onAdd(s); setQ(""); setOpen(false); }}
              className="w-full flex items-baseline gap-2 px-2 py-1 text-left hover:bg-elevated">
              <span className="font-mono tnum text-[12px] text-ink-dim shrink-0">{bare(s.t)}</span>
              <span className="text-[12.5px] truncate">{s.n}</span>
            </button>
          ))}
          {searching && hits.length === 0 && <div className="px-2 py-1 label">正在搜索…</div>}
        </div>
      )}
    </div>
  );
}

// ── watch ────────────────────────────────────────────────────────────────────
function IndexView({ idx }: { idx: CustomIndexRef }) {
  const q = useQuery({ queryKey: ["idx", "detail", idx.id], queryFn: () => api.indexDetail(idx.id), staleTime: 10 * 60_000 });
  if (q.isPending) return <section className="card p-6 label text-center">计算「{idx.name}」… 首次需读取全部成分的历史行情</section>;
  if (q.isError) return <section className="card p-4 text-[13px] text-up">{(q.error as Error).message}</section>;
  return <Body d={q.data!} />;
}

function Body({ d }: { d: CustomIndexDetail }) {
  const red = d.index.market === "CN";
  const s = d.stats;
  const b = d.bench_ret;
  return (
    <>
      <section className="card p-3 flex flex-col gap-3">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="text-[22px] font-semibold tnum">{fixed(s.level, 2)}</span>
          <span className={`text-[15px] font-semibold tnum ${moveClass(s.ret["1d"], red)}`}>{signed(s.ret["1d"], 2, "%")}</span>
          <span className="label">
            {d.method_label} · {d.hold ? "不调仓" : `${d.rebalance_label}调仓`}
            {d.index.cap_pct ? ` · 单只上限 ${d.index.cap_pct}%` : ""} · 对比 {d.bench_label} · 起始 {d.start} = 1000 · 截至 {d.as_of}
            {d.next_rebalance ? ` · 下次调仓 ${d.next_rebalance} 后首个交易日` : ""}
          </span>
        </div>
        <div className="grid gap-2 grid-cols-2 sm:grid-cols-4 lg:grid-cols-8">
          {(["5d", "1m", "3m", "ytd", "1y", "all"] as const).map((k) => (
            <Tile key={k} k={{ "5d": "5 日", "1m": "1 月", "3m": "3 月", ytd: "今年", "1y": "1 年", all: "起始以来" }[k]}
              v={signed(s.ret[k], 1, "%")} cls={moveClass(s.ret[k], red)}
              sub={b && b[k] != null ? `${d.bench_label} ${signed(b[k], 1, "%")}` : undefined} />
          ))}
          <Tile k="最大回撤" v={signed(s.max_dd, 1, "%")} sub={`现在 ${signed(s.dd_now, 1, "%")}`} />
          <Tile k="年化波动" v={s.vol != null ? `${fixed(s.vol, 0)}%` : "—"}
            sub={s.d200 != null ? `偏离 200 日 ${signed(s.d200, 1, "%")}` : "不足 200 日"} />
        </div>
        {d.bench_note && <p className="label text-brand-ink">{d.bench_note}</p>}
        <LevelChart d={d} />
      </section>

      <Members d={d} />
      <Events d={d} />
      <p className="label leading-snug">
        全收益指数（复权价，分红再投资）。成分变化、调仓都在当日收盘生效，点位连续；停牌股按停牌前价格计入且不参与调仓；
        退市或被收购的成分以最后成交价移出，所得按比例分给其余成分，并保存其历史（雅虎会删除被收购公司的数据）。
        {d.index.market === "US" && (d.index.method === "cap" || d.index.method === "cap_capped") &&
          " 美股市值 = 当前股本 × 每日价格：期间回购与增发不计入。"}
        {d.index.market === "CN" && (d.index.method === "cap" || d.index.method === "cap_capped") && " A 股市值为流通市值。"}
        {d.index.method === "inv_vol" && " 波动率 = 调仓日收盘前 3 个月的日收益标准差，只用当时已有的数据；停牌日不算作零波动。"}
        {d.guessed_caps.length > 0 && (d.index.method === "inv_vol"
          ? ` 上市或复牌不足一个月、按其余成分的中位波动率估算：${d.guessed_caps.map(bare).join("、")}。`
          : ` 缺少市值数据、按平均市值估算：${d.guessed_caps.map(bare).join("、")}。`)}
        {" "}对比：{d.bench_label}{d.index.market === "CN" ? "（价格指数，不含分红）" : ""}。
      </p>
    </>
  );
}

function Tile({ k, v, cls = "", sub }: { k: string; v: string; cls?: string; sub?: string }) {
  return (
    <div className="rounded-lg bg-sunken px-2.5 py-1.5 min-w-0">
      <div className="label">{k}</div>
      <div className={`text-[15px] font-semibold tnum ${cls}`}>{v}</div>
      {sub && <div className="label truncate">{sub}</div>}
    </div>
  );
}

function linePath(vs: (number | null)[], x: (i: number) => number, y: (v: number) => number) {
  let s = "", pen = false;
  vs.forEach((v, i) => { if (v == null) { pen = false; return; } s += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`; pen = true; });
  return s;
}

function LevelChart({ d }: { d: CustomIndexDetail }) {
  const c = d.chart, n = c.dates.length, W = 1000, H = 260;
  const box = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const vals = [...c.level, ...(c.bench ?? []), ...c.ma200].filter((v): v is number => v != null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  // A tripled index squashes its first year flat on a linear axis; on a log
  // axis equal heights are equal percentages, which is how a level is read.
  const [log, setLog] = useState(hi / lo > 2.5);
  const tf = (v: number) => (log ? Math.log(v) : v);
  const x = (i: number) => (i / Math.max(1, n - 1)) * W;
  const y = (v: number) => H - 6 - ((tf(v) - tf(lo)) / (tf(hi) - tf(lo) || 1)) * (H - 12);
  const liveAt = d.live_from ? c.dates.findIndex((t) => t >= d.live_from!) : -1;
  const up = (d.stats.ret.all ?? 0) >= 0;
  const red = d.index.market === "CN";
  const lineColor = up === red ? "var(--color-up)" : "var(--color-down)";
  const onMove = (e: React.MouseEvent) => {
    const rc = box.current?.getBoundingClientRect();
    if (rc) setHover(Math.max(0, Math.min(n - 1, Math.round(((e.clientX - rc.left) / rc.width) * (n - 1)))));
  };
  const ticks = [hi, log ? Math.sqrt(hi * lo) : (hi + lo) / 2, lo];
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <div className="flex flex-wrap items-center gap-3 text-[12px]">
        <span className="flex items-center gap-1"><span className="w-4 h-[2px]" style={{ background: lineColor }} />{d.index.name}</span>
        {c.bench && <span className="flex items-center gap-1"><span className="w-4 border-t-2 border-dotted border-ink-mute" />{d.bench_label}（同起点）</span>}
        <span className="flex items-center gap-1"><span className="w-4 border-t-2 border-dashed" style={{ borderColor: "#b45309" }} />200 日线</span>
        {liveAt > 0 && <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-sunken border border-line" />回溯（假设）· {d.live_from} 起实盘</span>}
        <div className="ml-auto flex rounded-md bg-sunken p-0.5">
          {([[true, "对数"], [false, "线性"]] as const).map(([v, label]) => (
            <button key={label} onClick={() => setLog(v)}
              className={`h-6 px-2 rounded text-[12px] ${log === v ? "bg-panel shadow-sm font-semibold" : "text-ink-mute"}`}>{label}</button>
          ))}
        </div>
      </div>
      <div className="flex gap-1">
        <div className="flex flex-col justify-between label font-mono tnum text-right w-12 shrink-0 py-0.5" style={{ height: H }}>
          {ticks.map((t, i) => <span key={i}>{t.toFixed(0)}</span>)}
        </div>
        <div ref={box} onMouseMove={onMove} onMouseLeave={() => setHover(null)} className="relative flex-1 min-w-0" style={{ height: H }}>
          <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 w-full h-full" role="img" aria-label={`${d.index.name} 点位走势`}>
            {liveAt > 0 && <rect x={0} y={0} width={x(liveAt)} height={H} fill="var(--color-sunken)" />}
            {liveAt > 0 && <line x1={x(liveAt)} x2={x(liveAt)} y1={0} y2={H} stroke="var(--color-line-bright)" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />}
            <line x1={0} x2={W} y1={y(1000)} y2={y(1000)} stroke="var(--color-line)" vectorEffect="non-scaling-stroke" />
            {c.bench && <path d={linePath(c.bench, x, y)} fill="none" stroke="var(--color-ink-mute)" strokeDasharray="2 3" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />}
            <path d={linePath(c.ma200, x, y)} fill="none" stroke="#b45309" strokeDasharray="6 4" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
            <path d={linePath(c.level, x, y)} fill="none" stroke={lineColor} strokeWidth={1.8} vectorEffect="non-scaling-stroke" />
            {hover != null && <line x1={x(hover)} x2={x(hover)} y1={0} y2={H} stroke="var(--color-ink-mute)" vectorEffect="non-scaling-stroke" />}
          </svg>
          {liveAt > 0 && (
            <>
              <span className="absolute top-1 left-1 label">回溯（假设）</span>
              <span className="absolute top-1 label" style={{ left: `calc(${(liveAt / Math.max(1, n - 1)) * 100}% + 4px)` }}>实盘</span>
            </>
          )}
          {hover != null && (
            <div className="absolute top-6 card px-2 py-1 text-[12px] pointer-events-none z-10"
              style={hover > n * 0.6 ? { right: `${100 - (hover / Math.max(1, n - 1)) * 100 + 1}%` } : { left: `${(hover / Math.max(1, n - 1)) * 100 + 1}%` }}>
              <div className="font-mono">{c.dates[hover]}{liveAt > 0 && hover < liveAt ? " · 回溯" : ""}</div>
              <div>{d.index.name} <b className="tnum">{fixed(c.level[hover], 2)}</b></div>
              {c.bench && <div className="text-ink-mute">{d.bench_label} {fixed(c.bench[hover], 2)}</div>}
            </div>
          )}
        </div>
      </div>
      <div className="flex justify-between label font-mono tnum pl-[52px]"><span>{c.dates[0]}</span><span>{c.dates[n - 1]}</span></div>
    </div>
  );
}

// ── members ──────────────────────────────────────────────────────────────────
function Members({ d }: { d: CustomIndexDetail }) {
  const qc = useQueryClient();
  const red = d.index.market === "CN";
  const [win, setWin] = usePersistentState<IndexWindow>("assrs.index.window", "1m");
  // The first day the backfill is still being put together, so a member can be back-dated.
  // After it, a change counts from today's close and the index already watched stays as it was.
  const settling = d.index.settling;
  const [addOn, setAddOn] = useState(d.today);
  const [err, setErr] = useState<string | null>(null);
  const [showGone, setShowGone] = useState(false);
  const refresh = () => qc.invalidateQueries({ queryKey: ["idx"] });
  const fail = (e: unknown) => setErr((e as Error).message);

  const add = useMutation({
    mutationFn: (s: StockRef) => api.addIndexMember(d.index.id, { ticker: s.t, name: s.n, added_on: settling ? addOn : undefined }),
    onSuccess: refresh, onError: fail,
  });
  const drop = useMutation({
    mutationFn: ({ m, mode }: { m: IndexMember; mode: "end" | "purge" }) =>
      api.removeIndexMember(d.index.id, m.id, mode),
    onSuccess: refresh, onError: fail,
  });
  // Deleting rewrites every day the member was in — only while none of them has been watched.
  const purgeable = (m: IndexMember) => settling || m.added_on >= d.today;

  const live = d.members.filter((m) => !["removed", "delisted"].includes(m.status));
  const gone = d.members.filter((m) => ["removed", "delisted"].includes(m.status));
  const rows = showGone ? [...live, ...gone] : live;
  const maxC = Math.max(0.01, ...d.members.map((m) => Math.abs(m.contrib[win] ?? 0)));
  const sum = d.members.reduce((a, m) => a + (m.contrib[win] ?? 0), 0);

  return (
    <section className="card p-3 flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-[13.5px] font-semibold">成分与贡献</h3>
        <span className="label">贡献 = 这只股票让指数涨跌了多少个百分点，全部加起来就是指数的涨跌</span>
        <div className="ml-auto flex rounded-md bg-sunken p-0.5">
          {WINDOWS.map((w) => (
            <button key={w.v} onClick={() => setWin(w.v)}
              className={`h-6 px-2 rounded text-[12px] ${win === w.v ? "bg-panel shadow-sm font-semibold" : "text-ink-mute"}`}>{w.label}</button>
          ))}
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-[12.5px] tnum">
          <thead>
            <tr className="label text-left">
              <th className="sticky left-0 z-10 bg-panel font-normal py-1">成分</th>
              <th className="font-normal">状态</th>
              <th className="font-normal" title="今天的实际权重；竖线 = 下次调仓要调回的目标">权重 · 目标</th>
              <th className="font-normal text-right pr-2">今日</th>
              <th className="font-normal" title="对指数涨跌的贡献（百分点）">贡献 · {WINDOWS.find((w) => w.v === win)!.label}</th>
              <th className="font-normal">加入</th>
              <th className="font-normal" />
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => (
              <tr key={m.id} className={`border-t border-line ${["removed", "delisted"].includes(m.status) ? "opacity-60" : ""}`}>
                <td className="sticky left-0 z-10 bg-panel py-1.5 pr-2 whitespace-nowrap">
                  <span className="font-semibold">{m.name === m.ticker ? bare(m.ticker) : m.name}</span>
                  {m.name !== m.ticker && <span className="font-mono text-[11px] text-ink-mute ml-1">{bare(m.ticker)}</span>}
                </td>
                <td className="py-1.5 pr-2 whitespace-nowrap">
                  <span className={`px-1.5 py-0.5 rounded text-[11px] ${STATUS[m.status].cls}`}>{STATUS[m.status].label}</span>
                  {m.status === "delisted" && m.exit_price != null && <span className="label ml-1">@{fixed(m.exit_price, 2)}</span>}
                </td>
                <td className="py-1.5 pr-2">
                  <WeightBar w={m.weight} t={m.target} />
                </td>
                <td className={`py-1.5 pr-2 text-right ${moveClass(m.ret_1d, red)}`}>{signed(m.ret_1d, 2, "%")}</td>
                <td className="py-1.5 pr-2"><ContribBar v={m.contrib[win]} max={maxC} red={red} /></td>
                <td className="py-1.5 pr-2 label whitespace-nowrap">{m.added_on}{m.removed_on ? ` → ${m.removed_on}` : ""}</td>
                <td className="py-1.5 whitespace-nowrap text-right">
                  {(m.status === "active" || m.status === "halted") && (
                    <button onClick={() => drop.mutate({ m, mode: "end" })} title="今天收盘移出，保留之前的历史"
                      className="text-[12px] text-cyan mr-2">移出</button>
                  )}
                  {purgeable(m) && (
                    <button onClick={() => { if (globalThis.confirm(`彻底删除 ${m.name}？它在指数历史里的记录也会一起消失（加错了才用这个）`)) drop.mutate({ m, mode: "purge" }); }}
                      title="加错了：连同历史一起删除" className="text-[12px] text-ink-mute hover:text-up">删除</button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr className="border-t border-line-bright">
              <td className="sticky left-0 bg-panel py-1.5 label" colSpan={4}>合计（= 指数{WINDOWS.find((w) => w.v === win)!.label}涨跌）</td>
              <td className={`py-1.5 font-semibold ${moveClass(sum, red)}`}>{signed(sum, 2, " 点%")}</td>
              <td colSpan={2} />
            </tr>
          </tfoot>
        </table>
      </div>
      {gone.length > 0 && (
        <button onClick={() => setShowGone(!showGone)} className="text-[12px] text-cyan self-start">
          {showGone ? "隐藏" : "显示"}已移出 / 退市的 {gone.length} 只
        </button>
      )}
      <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-line">
        <span className="label">加入成分</span>
        <MemberPicker market={d.index.market} exclude={live.map((m) => m.ticker)} placeholder="搜索并加入…"
          onAdd={(s) => { setErr(null); add.mutate(s); }} />
        {settling && (
          <label className="label flex items-center gap-1">生效日
            <input type="date" value={addOn} min={d.index.base_date} max={d.today} onChange={(e) => setAddOn(e.target.value)}
              className="h-7 px-1.5 rounded-md bg-sunken text-[12px] outline-none" />
          </label>
        )}
        <span className="label">{settling ? "当日" : "今天"}收盘买入{d.hold ? "，按目标份额从其余成分按比例腾出资金" : "，其余成分随之重新加权"}</span>
        {(add.isPending || drop.isPending) && <span className="label">更新中…</span>}
      </div>
      <p className="label leading-snug">
        {settling
          ? "今天是建立当天：成分可以补加到更早的日期，加错的可以连同历史删除。从明天起，加入和移出都从当天收盘生效，已经走过的指数不再改动。"
          : "加入和移出都从今天收盘生效，之前的点位不变 —— 和指数公司调整成分一样。"}
      </p>
      {err && <p className="text-[12.5px] text-up">{err}</p>}
    </section>
  );
}

function WeightBar({ w, t }: { w: number | null; t: number | null }) {
  if (w == null) return <span className="label">—</span>;
  const scale = 100 / Math.max(40, (w ?? 0) * 1.2, (t ?? 0) * 1.2);
  return (
    <div className="flex items-center gap-1.5">
      <span className="relative w-24 h-2 rounded-sm bg-sunken inline-block">
        <span className="absolute inset-y-0 left-0 rounded-sm bg-cyan/60" style={{ width: `${Math.min(100, w * scale)}%` }} />
        {t != null && <span className="absolute -top-0.5 -bottom-0.5 w-0.5 bg-ink" style={{ left: `${Math.min(100, t * scale)}%` }} />}
      </span>
      <span className="w-11 text-right">{fixed(w, 1)}%</span>
      {t != null && <span className="label">/ {fixed(t, 1)}</span>}
    </div>
  );
}

function ContribBar({ v, max, red }: { v: number | null; max: number; red: boolean }) {
  const w = v == null ? 0 : (Math.abs(v) / max) * 50;
  const color = v == null ? "transparent" : (v >= 0) === red ? "var(--color-up)" : "var(--color-down)";
  return (
    <div className="flex items-center gap-1.5">
      <span className="relative w-28 h-2 rounded-sm bg-sunken inline-block">
        <span className="absolute inset-y-0 w-px bg-line-bright" style={{ left: "50%" }} />
        <span className="absolute inset-y-0 rounded-sm" style={{ left: v != null && v >= 0 ? "50%" : `${50 - w}%`, width: `${w}%`, background: color, opacity: 0.8 }} />
      </span>
      <span className={`w-14 text-right ${moveClass(v, red)}`}>{signed(v, 2)}</span>
    </div>
  );
}

// ── change log ───────────────────────────────────────────────────────────────
function Events({ d }: { d: CustomIndexDetail }) {
  const [open, setOpen] = useState(false);
  const shown = useMemo(() => (open ? d.events : d.events.slice(0, 6)), [open, d.events]);
  const word = (e: CustomIndexDetail["events"][number]) => {
    const who = e.name && e.name !== e.ticker ? e.name : bare(e.ticker ?? "");
    switch (e.kind) {
      case "start": return `指数起始 · ${e.n} 只成分 · 1000 点`;
      case "add": return `加入 ${who}`;
      case "remove": return e.reason === "delisted" ? `${who} 停止交易（退市/被收购），以最后价格移出` : `移出 ${who}`;
      case "rebalance": return `定期调仓 · 换手 ${fixed(e.turnover, 1)}%`;
    }
  };
  return (
    <section className="card p-3 flex flex-col gap-1.5">
      <h3 className="text-[13.5px] font-semibold">变动记录 <span className="label font-normal">（{d.events.length}）</span></h3>
      <ul className="flex flex-col">
        {shown.map((e, i) => (
          <li key={i} className="flex gap-3 text-[12.5px] py-0.5 border-t border-line first:border-0">
            <span className="font-mono tnum text-ink-mute w-24 shrink-0">{e.date}</span>
            <span className={e.kind === "rebalance" ? "text-ink-dim" : ""}>{word(e)}</span>
            {d.live_from && e.date < d.live_from && <span className="label">回溯</span>}
          </li>
        ))}
      </ul>
      {d.events.length > 6 && (
        <button onClick={() => setOpen(!open)} className="text-[12px] text-cyan self-start">{open ? "收起" : `显示全部 ${d.events.length} 条`}</button>
      )}
    </section>
  );
}
