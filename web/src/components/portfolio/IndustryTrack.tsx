/**
 * 🧭 行业轨迹 — which industry one stock has been trading with, and when it switched.
 *
 * A fibreglass maker can fall with property one year and rise with chips the
 * next. One row per industry, one column per session, ten years wide and on
 * the same axis as the brush above, so a switch reads as one row fading and
 * another lighting up. The market is taken out by default: left in, every
 * industry correlates 0.4–0.8 with every stock and the switch disappears.
 */

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../../lib/api";
import type { IndustryTrack as Track, StockRef } from "../../lib/types";
import { signed } from "../../lib/format";
import { usePersistentState } from "../../lib/usePersistentState";
import type { Market } from "./Optimiser";

/** Mirrors industry_track.INDUSTRIES; the server refuses anything else. */
const INDUSTRIES: Record<Market, { code: string; label: string }[]> = {
  CN: [
    { code: "931775.CSI", label: "房地产" },
    { code: "931009.CSI", label: "建筑材料" },
    { code: "399995.SZ", label: "基建工程" },
    { code: "930606.CSI", label: "钢铁" },
    { code: "000819.SH", label: "有色金属" },
    { code: "399998.SZ", label: "煤炭" },
    { code: "000813.CSI", label: "化工" },
    { code: "H30184.CSI", label: "半导体" },
    { code: "930652.CSI", label: "电子" },
    { code: "930651.CSI", label: "计算机" },
    { code: "931160.CSI", label: "通信设备" },
    { code: "399808.SZ", label: "新能源" },
    { code: "399976.SZ", label: "新能源车" },
    { code: "930607.CSI", label: "汽车" },
    { code: "000812.CSI", label: "机械" },
    { code: "399967.SZ", label: "军工" },
    { code: "H30199.CSI", label: "电力" },
    { code: "399986.SZ", label: "银行" },
    { code: "399975.SZ", label: "证券" },
    { code: "399997.SZ", label: "白酒" },
    { code: "000807.CSI", label: "食品饮料" },
    { code: "399989.SZ", label: "医疗" },
    { code: "930697.CSI", label: "家用电器" },
    { code: "399971.SZ", label: "传媒" },
    { code: "000949.CSI", label: "农业" },
  ],
  US: [
    { code: "XLK", label: "科技 XLK" },
    { code: "SMH", label: "半导体 SMH" },
    { code: "IGV", label: "软件 IGV" },
    { code: "XLC", label: "通信服务 XLC" },
    { code: "XLF", label: "金融 XLF" },
    { code: "KRE", label: "地区银行 KRE" },
    { code: "XLE", label: "能源 XLE" },
    { code: "XLB", label: "原材料 XLB" },
    { code: "XLI", label: "工业 XLI" },
    { code: "ITA", label: "航空军工 ITA" },
    { code: "XLY", label: "可选消费 XLY" },
    { code: "XLP", label: "必需消费 XLP" },
    { code: "XLV", label: "医疗 XLV" },
    { code: "XBI", label: "生物科技 XBI" },
    { code: "XLU", label: "公用事业 XLU" },
    { code: "XLRE", label: "房地产 XLRE" },
    { code: "ITB", label: "住宅建筑 ITB" },
    { code: "GDX", label: "金矿 GDX" },
  ],
};

const DEFAULTS: Record<Market, string[]> = {
  CN: ["931775.CSI", "931009.CSI", "000819.SH", "H30184.CSI", "930652.CSI", "399808.SZ"],
  US: ["XLK", "SMH", "XLF", "XLE", "XLRE", "XLV"],
};

const MAX = 8;
const ROW = 22;
const PRICE_H = 64;

type Mode = "ex" | "raw";
type Freq = "W" | "D";

/**
 * Full colour at this |r|. Ex-market numbers rarely pass 0.6; raw ones sit
 * near 0.6 all the time, so they get the whole range.
 */
const SCALE: Record<Mode, number> = { ex: 0.6, raw: 1 };

const CYAN = [6, 182, 212] as const;
const VIOLET = [168, 85, 247] as const;

/** Cyan with, violet against — the same pair as the correlation grids. */
function tint(v: number | null, scale: number): string {
  if (v == null) return "transparent";
  const a = 0.1 + Math.min(1, Math.abs(v) / scale) * 0.75;
  const [r, g, b] = v >= 0 ? CYAN : VIOLET;
  return `rgba(${r},${g},${b},${a.toFixed(2)})`;
}

const short = (t: string, picks: StockRef[]) =>
  t.startsWith("US:") ? t.slice(3) : (picks.find((p) => p.t === t)?.n || t);

function lowerBound(dates: string[], d: string) {
  let lo = 0;
  let hi = dates.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (dates[mid]! < d) lo = mid + 1;
    else hi = mid;
  }
  return Math.min(lo, dates.length - 1);
}

export function IndustryTrack({ market, index, picks, from, to }: {
  market: Market;
  /** The window index: its sessions are the axis, shared with the brush. */
  index: string;
  picks: StockRef[];
  from: string;
  to: string;
}) {
  const allowed = INDUSTRIES[market];
  const [savedSym, setSym] = usePersistentState<string>(`assrs.pf.ind.${market}.sym`, "");
  const [savedInds, setInds] = usePersistentState<string[]>(`assrs.pf.ind.${market}`, DEFAULTS[market]);
  const [freq, setFreq] = usePersistentState<Freq>("assrs.pf.ind.freq", "W");
  const [mode, setMode] = usePersistentState<Mode>("assrs.pf.ind.mode", "ex");
  const [hover, setHover] = useState<number | null>(null);

  const sym = picks.some((p) => p.t === savedSym) ? savedSym : picks[0]!.t;
  // Kept in the list's order, so the rows do not reshuffle as chips are clicked.
  const inds = allowed.map((i) => i.code).filter((c) => savedInds.includes(c));
  const toggle = (c: string) => {
    if (inds.includes(c)) {
      if (inds.length > 1) setInds(inds.filter((x) => x !== c));
    } else if (inds.length < MAX) {
      setInds([...inds, c]);
    }
  };

  const series = useQuery({
    queryKey: ["portfolio", "industry", sym, inds, index, freq],
    queryFn: () => api.portfolioIndustryTrack({ symbol: sym, industries: inds, index, freq, series: true }),
    enabled: inds.length > 0,
    staleTime: 30 * 60_000,
    retry: false,
    placeholderData: keepPreviousData,
  });
  // The window's numbers on their own: moving the window must not resend ten years.
  const win = useQuery({
    queryKey: ["portfolio", "industryWin", sym, inds, index, from, to],
    queryFn: () => api.portfolioIndustryTrack({
      symbol: sym, industries: inds, index, freq: "D", series: false, start: from, end: to,
    }),
    enabled: inds.length > 0,
    staleTime: 10 * 60_000,
    retry: false,
    placeholderData: keepPreviousData,
  });

  const d = series.data;

  return (
    <div className="flex flex-col gap-2 pt-3 border-t border-line">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <h4 className="text-[13px] font-semibold">🧭 行业轨迹</h4>
        <div className="flex items-center gap-0.5 flex-wrap">
          {picks.map((p) => (
            <button key={p.t} onClick={() => setSym(p.t)}
              className={`h-7 px-2.5 rounded-md text-[12.5px] font-medium transition-colors ${
                sym === p.t ? "bg-elevated text-ink" : "text-ink-mute hover:text-ink"}`}>
              {short(p.t, picks)}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <Toggle value={mode} onChange={setMode} options={[
            { v: "ex", label: "剔除大盘", hint: "先去掉大盘对股票和行业的共同影响，剩下的才是行业本身的联系" },
            { v: "raw", label: "原始", hint: "直接的相关系数，包含了大盘的共同涨跌" },
          ]} />
          <Toggle value={freq} onChange={setFreq} options={[
            { v: "W", label: "周", hint: "周收益，26 周滚动" },
            { v: "D", label: "日", hint: "日收益，60 个交易日滚动：反应更快，噪音也更大" },
          ]} />
        </div>
      </div>

      <div className="flex flex-wrap gap-1">
        {allowed.map((i) => {
          const on = inds.includes(i.code);
          const full = !on && inds.length >= MAX;
          return (
            <button key={i.code} onClick={() => toggle(i.code)} disabled={full}
              title={full ? `最多同时看 ${MAX} 个行业` : i.code}
              className={`h-6 px-2 rounded-md text-[12px] transition-colors ${
                on ? "bg-elevated text-ink font-medium ring-1 ring-line"
                  : full ? "text-ink-mute/50" : "text-ink-mute hover:text-ink"}`}>
              {i.label}
            </button>
          );
        })}
      </div>

      {series.isError && (
        <p className="text-[12.5px] text-up">{(series.error as ApiError).message}</p>
      )}
      {series.isPending && <p className="label py-8 text-center">读取十年行业指数…</p>}

      {d?.dates && (
        <div className={`flex flex-col gap-2 transition-opacity ${series.isFetching ? "opacity-50" : ""}`}>
          <Chart d={d} mode={mode} from={from} to={to} hover={hover} setHover={setHover}
            name={short(sym, picks)} />
          <Legend mode={mode} d={d} />
          <WindowTable d={d} w={win.data?.window} loading={win.isFetching} mode={mode}
            error={win.isError ? (win.error as ApiError).message : null} />
          <p className="label leading-snug">
            每一格是截至那一天、过去 {d.span} {d.unit}的滚动相关系数，
            <span style={{ color: "rgb(6,182,212)" }}>青色</span>是同涨同跌，
            <span style={{ color: "rgb(168,85,247)" }}>紫色</span>是反着走。
            「剔除大盘」先去掉{d.market_proxy.label}对股票和行业的共同影响：不去掉的话，几乎所有行业都和股票有 0.4–0.8 的相关，
            哪一行亮起来、哪一行暗下去就看不出来了。
            {d.failed.length > 0 && ` 读取失败、没有画出来的：${d.failed.join("、")}。`}
          </p>
        </div>
      )}
    </div>
  );
}

function Toggle<T extends string>({ value, onChange, options }: {
  value: T;
  onChange: (v: T) => void;
  options: { v: T; label: string; hint: string }[];
}) {
  return (
    <div className="flex items-center rounded-md bg-sunken p-0.5">
      {options.map((o) => (
        <button key={o.v} onClick={() => onChange(o.v)} title={o.hint}
          className={`h-6 px-2 rounded text-[12px] transition-colors ${
            value === o.v ? "bg-panel text-ink font-medium shadow-sm" : "text-ink-mute hover:text-ink"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

type Row = { key: string; label: string; values: (number | null)[]; member: boolean; market?: boolean };

function rowsOf(d: Track, mode: Mode): Row[] {
  const src = mode === "ex" ? d.ex! : d.raw!;
  const rows: Row[] = d.industries.map((i) => ({
    key: i.code, label: i.label, values: src[i.code] ?? [],
    member: typeof d.member[i.code] === "number",
  }));
  // Raw mode shows the market itself as the yardstick the others are read against.
  if (mode === "raw") {
    rows.unshift({ key: "_m", label: d.market_proxy.label, values: d.market_raw ?? [], member: false, market: true });
  }
  return rows;
}

function Chart({ d, mode, from, to, hover, setHover, name }: {
  d: Track; mode: Mode; from: string; to: string;
  hover: number | null; setHover: (k: number | null) => void; name: string;
}) {
  const dates = d.dates!;
  const n = dates.length;
  const rows = useMemo(() => rowsOf(d, mode), [d, mode]);
  const canvas = useRef<HTMLCanvasElement>(null);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const c = canvas.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    c.width = n;
    c.height = rows.length;
    const img = ctx.createImageData(n, rows.length);
    const scale = SCALE[mode];
    rows.forEach((r, y) => {
      for (let k = 0; k < n; k++) {
        const v = r.values[k];
        if (v == null) continue;
        const [cr, cg, cb] = v >= 0 ? CYAN : VIOLET;
        const o = (y * n + k) * 4;
        img.data[o] = cr;
        img.data[o + 1] = cg;
        img.data[o + 2] = cb;
        img.data[o + 3] = Math.round(255 * (0.06 + Math.min(1, Math.abs(v) / scale) * 0.84));
      }
    });
    ctx.putImageData(img, 0, 0);
  }, [rows, n, mode]);

  const price = useMemo(() => {
    const logs = (d.close ?? []).map((v) => (v != null && v > 0 ? Math.log(v) : null));
    const vals = logs.filter((v): v is number => v != null);
    if (vals.length < 2) return "";
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    let pen = "M";
    return logs.map((v, i) => {
      if (v == null) { pen = "M"; return ""; }
      const y = PRICE_H - 4 - ((v - lo) / (hi - lo || 1)) * (PRICE_H - 8);
      const s = `${pen}${((i / (n - 1)) * 1000).toFixed(1)} ${y.toFixed(1)}`;
      pen = "L";
      return s;
    }).join("");
  }, [d.close, n]);

  const years = useMemo(() => {
    const out: { y: string; i: number }[] = [];
    dates.forEach((t, i) => {
      if (i > 0 && t.slice(0, 4) !== dates[i - 1]!.slice(0, 4)) out.push({ y: t.slice(0, 4), i });
    });
    return out;
  }, [dates]);

  const pct = (i: number) => `${(i / (n - 1)) * 100}%`;
  const wi = lowerBound(dates, from);
  const wj = lowerBound(dates, to);

  const onMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const rect = box.current!.getBoundingClientRect();
    const k = Math.round(((e.clientX - rect.left) / rect.width) * (n - 1));
    setHover(Math.max(0, Math.min(n - 1, k)));
  };

  const k = hover;
  return (
    <div className="flex flex-col gap-1">
      <div className="min-h-[22px] flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[12px] tnum">
        {k == null ? (
          <span className="label">{name} · 鼠标移到图上，看那一天每个行业的数值</span>
        ) : (
          <>
            <span className="font-medium">{dates[k]}</span>
            {rows.map((r) => (
              <span key={r.key} className="px-1.5 rounded"
                style={{ background: tint(r.values[k] ?? null, SCALE[mode]) }}>
                {r.label} {signed(r.values[k] ?? null, 2)}
              </span>
            ))}
          </>
        )}
      </div>

      <div ref={box} className="relative select-none" style={{ touchAction: "pan-y" }}
        onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        <svg viewBox={`0 0 1000 ${PRICE_H}`} preserveAspectRatio="none"
          className="w-full block" style={{ height: PRICE_H }} role="img"
          aria-label={`${name} 十年价格`}>
          <rect x={(wi / (n - 1)) * 1000} y={0} width={((wj - wi) / (n - 1)) * 1000} height={PRICE_H}
            style={{ fill: "var(--color-cyan)", opacity: 0.07 }} />
          <path d={price} fill="none" stroke="var(--color-ink)" strokeWidth={1.2}
            vectorEffect="non-scaling-stroke" />
        </svg>
        <div className="relative">
          <canvas ref={canvas} className="w-full block rounded-sm"
            style={{ height: rows.length * ROW, imageRendering: "pixelated" }}
            aria-label="行业相关性热力图" />
          {rows.map((r, y) => (
            <div key={r.key} className="absolute left-0 flex items-center pointer-events-none"
              style={{ top: y * ROW, height: ROW }}>
              <span className={`ml-1 px-1 rounded text-[11px] leading-[16px] bg-panel/85 ${
                r.market ? "text-ink-mute" : "text-ink"}`}>
                {r.label}{r.member && <span className="text-ink-mute"> · 成分</span>}
              </span>
            </div>
          ))}
          {rows.slice(1).map((r, y) => (
            <div key={r.key} className="absolute inset-x-0 h-px bg-panel/70 pointer-events-none"
              style={{ top: (y + 1) * ROW }} />
          ))}
        </div>
        {[wi, wj].map((p, s) => (
          <div key={s} className="absolute top-0 bottom-0 w-0.5 -ml-px bg-cyan pointer-events-none"
            style={{ left: pct(p) }} />
        ))}
        {k != null && (
          <div className="absolute top-0 bottom-0 w-px bg-ink/60 pointer-events-none"
            style={{ left: pct(k) }} />
        )}
      </div>

      <div className="relative h-3.5 label tnum">
        {years.map((t, s) => (
          <span key={t.y} className={`absolute -translate-x-1/2 ${s % 2 ? "hidden sm:inline" : ""}`}
            style={{ left: pct(t.i) }}>
            {t.y}
          </span>
        ))}
      </div>
    </div>
  );
}

function Legend({ mode, d }: { mode: Mode; d: Track }) {
  const s = SCALE[mode];
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 label">
      <span className="flex items-center gap-1.5 tnum">
        −{s.toFixed(1)}
        <span className="inline-block h-2.5 w-28 rounded-sm" style={{
          background: `linear-gradient(to right, ${tint(-s, s)}, ${tint(0, s)}, ${tint(s, s)})`,
        }} />
        +{s.toFixed(1)}
      </span>
      <span>
        {mode === "ex" ? `剔除${d.market_proxy.label}后的偏相关` : "原始相关"} · {d.unit}收益 · {d.span}{d.unit}滚动
      </span>
      <span>{d.listed > d.dates![0]! ? `${d.listed} 上市，之前没有数据` : ""}</span>
    </div>
  );
}

function WindowTable({ d, w, loading, mode, error }: {
  d: Track;
  w: Track["window"] | undefined;
  loading: boolean;
  mode: Mode;
  error: string | null;
}) {
  if (error) return <p className="text-[12.5px] text-up">{error}</p>;
  if (!w) return loading ? <p className="label">计算窗口内的相关性…</p> : null;
  const last = (code: string) => {
    const v = (mode === "ex" ? d.ex : d.raw)?.[code];
    return v ? v[v.length - 1] ?? null : null;
  };
  const rows = d.industries
    .map((i) => ({ ...i, ...(w.rows[i.code] ?? { raw: null, ex: null }) }))
    .sort((a, b) => (b.ex ?? -9) - (a.ex ?? -9));
  return (
    <div className={`overflow-x-auto transition-opacity ${loading ? "opacity-50" : ""}`}>
      <table className="text-[12.5px] tnum">
        <caption className="label text-left pb-1">
          选中窗口内 {w.from} → {w.to}（日收益，{w.days} 天），按剔除大盘后的相关排序 ·
          和{d.market_proxy.label}本身的原始相关 {signed(w.market_raw, 2)}
        </caption>
        <thead className="label">
          <tr>
            <th className="text-left font-normal pr-4 py-1">行业</th>
            <th className="text-right font-normal px-2" title="去掉大盘的共同影响之后">剔除大盘</th>
            <th className="text-right font-normal px-2">原始</th>
            <th className="text-right font-normal px-2" title={`最近 ${d.span} ${d.unit}，跟上面图里最右边一格一样`}>
              最新（{d.span}{d.unit}）
            </th>
            <th className="text-left font-normal pl-3">成分股</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.code} className="border-t border-line">
              <td className="pr-4 py-1 whitespace-nowrap">{r.label}</td>
              <td className="text-right px-2"><Cell v={r.ex} s={SCALE.ex} /></td>
              <td className="text-right px-2"><Cell v={r.raw} s={SCALE.raw} /></td>
              <td className="text-right px-2"><Cell v={last(r.code)} s={SCALE[mode]} /></td>
              <td className="pl-3 whitespace-nowrap label"
                title={typeof d.member[r.code] === "number" ? "大成分股和自己所在的指数天然高度相关" : undefined}>
                {typeof d.member[r.code] === "number" ? `是，权重 ${d.member[r.code]}%`
                  : d.member[r.code] === false ? "否" : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Cell({ v, s }: { v: number | null; s: number }) {
  return (
    <span className="inline-block min-w-[3.2rem] px-1.5 rounded" style={{ background: tint(v, s) }}>
      {v == null ? "—" : signed(v, 2)}
    </span>
  );
}
