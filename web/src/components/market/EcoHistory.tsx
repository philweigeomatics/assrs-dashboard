/**
 * 📈 One economic series, every release: how each index moved on the session
 * it landed in, and how that compares with a normal day.
 *
 * The session is the index's own — the first one that closed after the
 * release — so a US release in the Vancouver morning is that day for 标普500
 * and the next day for 沪深300. See eco_impact for the rule.
 */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";
import type { EcoHistory as History, EcoHistoryRow, EcoIndexMeta, EcoReaction }
  from "../../lib/types";
import { localClock, localDay } from "../../lib/localDay";
import { moveClass, signed } from "../../lib/format";

const SHOW = 24;

export function EcoHistory({ country, event, codes, onClose }: {
  country: string; event: string; codes: string[]; onClose: () => void;
}) {
  const q = useQuery({
    queryKey: ["cal", "ecoHistory", country, event, codes],
    queryFn: () => api.ecoHistory(country, event, codes),
    staleTime: 30 * 60_000,
    retry: false,
  });

  return (
    <section className="rounded-lg bg-sunken p-2.5 flex flex-col gap-2.5 min-w-0">
      <div className="flex items-baseline gap-2 flex-wrap">
        <h3 className="text-[13.5px] font-semibold">
          📈 {q.data?.series ?? event} · 历次发布后的指数反应
        </h3>
        {q.data && q.data.rows.length > 0 && (
          <span className="label tnum">
            {q.data.rows.length} 次 · 自 {localDay(new Date(q.data.rows[q.data.rows.length - 1]!.at))}
          </span>
        )}
        <button onClick={onClose} className="ml-auto text-[12px] text-ink-mute hover:text-ink">
          关闭 ✕
        </button>
      </div>
      {q.isPending && <p className="label py-6 text-center">读取历史与十年指数行情…</p>}
      {q.isError && <p className="text-[12.5px] text-up">{(q.error as ApiError).message}</p>}
      {q.data && <Body d={q.data} />}
    </section>
  );
}

function Body({ d }: { d: History }) {
  const [all, setAll] = useState(false);
  if (d.rows.length === 0) {
    return <p className="label">数据库里还没有这项数据的历史记录。</p>;
  }
  const rows = all ? d.rows : d.rows.slice(0, SHOW);
  return (
    <>
      <div className="grid gap-2.5 grid-cols-1 md:grid-cols-2">
        {d.indexes.map((ix) => (
          <IndexCard key={ix.code} ix={ix} d={d} />
        ))}
      </div>
      {d.failed.length > 0 && (
        <p className="label text-up">
          {d.failed.map((f) => `${f.code}：${f.reason}`).join("；")}
        </p>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-[12px] border-collapse tnum">
          <thead>
            <tr className="text-ink-mute">
              <th className="text-left font-normal pb-1 pr-2">发布（你的时间）</th>
              <th className="text-right font-normal pb-1 px-2">实际</th>
              <th className="text-right font-normal pb-1 px-2">预期</th>
              <th className="text-right font-normal pb-1 px-2">前值</th>
              <th className="text-left font-normal pb-1 px-2">对比预期</th>
              {d.indexes.map((ix) => (
                <th key={ix.code} className="text-left font-normal pb-1 px-2" colSpan={1}>
                  {ix.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.at}|${r.event}`} className="border-t border-line">
                <td className="py-1 pr-2 font-mono whitespace-nowrap">
                  {localDay(new Date(r.at))} {localClock(r.at)}
                </td>
                <td className="py-1 px-2 text-right font-medium">{r.value ?? "—"}</td>
                <td className="py-1 px-2 text-right text-ink-mute">{r.forecast ?? "—"}</td>
                <td className="py-1 px-2 text-right text-ink-mute">{r.prev ?? "—"}</td>
                <td className="py-1 px-2 whitespace-nowrap"><Dir r={r} /></td>
                {d.indexes.map((ix) => (
                  <td key={ix.code} className="py-1 px-2 whitespace-nowrap">
                    <Cell r={r.react[ix.code] ?? null} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {d.rows.length > SHOW && (
        <button onClick={() => setAll(!all)} className="self-start text-[12px] text-cyan">
          {all ? "收起" : `显示全部 ${d.rows.length} 次`}
        </button>
      )}

      <p className="label leading-snug">
        每个指数取发布之后<b>第一个收盘</b>的交易日，日期是那个交易所自己的日期：
        美国盘前发布的数据，标普是当天，沪深300 是第二天。
        「×」是这一天的涨跌幅相对前 60 个交易日平均波动的倍数，1.0× 就是普通的一天。
        两点要注意：同一时刻发布的几项数据（比如 CPI 和核心 CPI）落在同一个交易日，
        数字会一模一样；一个交易日的涨跌也包含了当天发生的其他所有事情。
        所以看的是很多次发布的平均，单独一次说明不了什么。
        「高于/低于预期」只是方向，不代表利好或利空 —— 失业率高于预期和零售高于预期是相反的消息。
      </p>
    </>
  );
}

function Dir({ r }: { r: EcoHistoryRow }) {
  if (r.dir == null) return <span className="text-ink-mute">—</span>;
  if (r.dir === 0) return <span className="text-ink-dim">持平</span>;
  return (
    <span className="text-ink">
      {r.dir > 0 ? "▲ 高于" : "▼ 低于"}
    </span>
  );
}

function Cell({ r }: { r: EcoReaction | null }) {
  if (!r) return <span className="text-ink-mute">—</span>;
  const day = r.day.slice(5).replace("-", "/");
  if (!r.done) return <span className="text-ink-mute">{day} 待定</span>;
  return (
    <span>
      <span className="text-ink-mute font-mono">{day}</span>{" "}
      <span className={`${moveClass(r.ret)} font-medium`}>{signed(r.ret, 2, "%")}</span>
      {r.x != null && (
        <span className={`ml-1 text-[11px] ${r.x >= 1.5 ? "text-ink font-semibold" : "text-ink-mute"}`}>
          {r.x.toFixed(1)}×
        </span>
      )}
    </span>
  );
}

/** Summary plus one bar per release, oldest left, on this index's sessions. */
function IndexCard({ ix, d }: { ix: EcoIndexMeta; d: History }) {
  const s = d.summary[ix.code];
  const pts = useMemo(() => [...d.rows].reverse()
    .map((r) => ({ r, x: r.react[ix.code] }))
    .filter((p): p is { r: EcoHistoryRow; x: EcoReaction } => !!p.x && p.x.done),
  [d.rows, ix.code]);
  const [hover, setHover] = useState<number | null>(null);

  const W = 600, H = 110, PAD = 4;
  const max = Math.max(0.5, ...pts.map((p) => Math.abs(p.x.ret ?? 0)));
  const mid = H / 2;
  const step = pts.length ? (W - PAD * 2) / pts.length : 0;
  const bw = Math.max(1, step - 2);
  const y = (v: number) => (v / max) * (mid - 6);
  const years: { i: number; y: string }[] = [];
  pts.forEach((p, i) => {
    const yr = p.x.day.slice(0, 4);
    if (i === 0 || pts[i - 1]!.x.day.slice(0, 4) !== yr) years.push({ i, y: yr });
  });
  const hv = hover != null ? pts[hover] : null;

  return (
    <div className="rounded-md bg-panel p-2 flex flex-col gap-1.5 min-w-0">
      <div className="flex items-baseline gap-2 flex-wrap">
        <span className="text-[13px] font-semibold">{ix.label}</span>
        {s && s.mean_x != null && (
          <span className="text-[12.5px]">
            平均 <b className={`tnum ${s.mean_x >= 1.3 ? "text-ink" : "text-ink-dim"}`}>
              {s.mean_x.toFixed(2)}×</b> 普通一天
          </span>
        )}
        {s && <span className="label tnum">{s.n} 次</span>}
      </div>
      {s && s.n > 0 && (
        <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[12px] tnum">
          <span>平均波动 {s.mean_abs != null ? `${s.mean_abs.toFixed(2)}%` : "—"}</span>
          <span>收涨 {s.up != null ? `${Math.round(s.up * 100)}%` : "—"}</span>
          <span>
            高于预期 {s.above.n} 次，平均{" "}
            <span className={moveClass(s.above.ret)}>{signed(s.above.ret, 2, "%")}</span>
          </span>
          <span>
            低于预期 {s.below.n} 次，平均{" "}
            <span className={moveClass(s.below.ret)}>{signed(s.below.ret, 2, "%")}</span>
          </span>
        </div>
      )}
      {pts.length > 0 && (
        <>
          <svg viewBox={`0 0 ${W} ${H + 14}`} className="w-full h-auto select-none"
            onMouseLeave={() => setHover(null)} role="img"
            aria-label={`${ix.label} 每次发布当个交易日的涨跌`}>
            <line x1={0} x2={W} y1={mid} y2={mid} stroke="var(--color-line-bright)" strokeWidth={1} />
            <text x={W - 2} y={10} textAnchor="end" fontSize={10} fill="var(--color-ink-mute)">
              +{max.toFixed(1)}%
            </text>
            <text x={W - 2} y={H - 2} textAnchor="end" fontSize={10} fill="var(--color-ink-mute)">
              −{max.toFixed(1)}%
            </text>
            {pts.map((p, i) => {
              const v = p.x.ret ?? 0;
              const h = Math.max(1, Math.abs(y(v)));
              const x0 = PAD + i * step + (step - bw) / 2;
              // Faded: smaller than a normal day.
              const op = (p.x.x ?? 1) >= 1 ? 1 : 0.4;
              return (
                <g key={i} onMouseEnter={() => setHover(i)} onClick={() => setHover(i)}>
                  <rect x={PAD + i * step} y={0} width={step} height={H} fill="transparent" />
                  <rect x={x0} y={v >= 0 ? mid - h : mid} width={bw} height={h}
                    rx={Math.min(2, bw / 2)}
                    fill={v >= 0 ? "var(--color-up)" : "var(--color-down)"}
                    opacity={hover === i ? 1 : op}
                    stroke={hover === i ? "var(--color-ink)" : "none"} strokeWidth={1} />
                </g>
              );
            })}
            {years.map((t) => (
              <text key={t.y} x={PAD + t.i * step} y={H + 12} fontSize={10}
                fill="var(--color-ink-mute)">{t.y}</text>
            ))}
          </svg>
          <p className="text-[11.5px] tnum min-h-[16px] text-ink-dim">
            {hv ? (
              <>
                {localDay(new Date(hv.r.at))} 发布 · 实 {hv.r.value ?? "—"} / 预 {hv.r.forecast ?? "—"}
                {" → "}{ix.label} {hv.x.day}{" "}
                <span className={`${moveClass(hv.x.ret)} font-medium`}>{signed(hv.x.ret, 2, "%")}</span>
                {hv.x.x != null && ` · ${hv.x.x.toFixed(1)}× 普通一天`}
              </>
            ) : "每根柱子是一次发布当个交易日的涨跌；颜色浅的是比普通一天还小的。指上去看详情。"}
          </p>
        </>
      )}
    </div>
  );
}
