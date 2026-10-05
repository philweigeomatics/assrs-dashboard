/**
 * 🔎 按代码查找 — every record for one symbol, held or not.
 *
 * The point is the positions you no longer own. A picker built from current
 * holdings would hide most of the history: on a real account 27 symbols had
 * been traded in three years and only 12 were still held, so ALGM — bought
 * and sold out completely — would have been unreachable.
 *
 * It searches several years at once because that is the question people
 * actually ask ("what did I do with this?"), not "what did I do with this in
 * 2025". Each year is a cached payload underneath, so the first search is
 * slow and the rest are instant.
 */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";
import type { QtSymbolActivity } from "../../lib/types";
import { fixed, signed } from "../../lib/format";

export function SymbolHistory({ years }: { years: number }) {
  const [picked, setPicked] = useState<string | null>(null);
  const [q, setQ] = useState("");

  const list = useQuery({
    queryKey: ["qt", "symbols", years],
    queryFn: () => api.qtSymbols(years),
    staleTime: 6 * 3600_000,
  });
  const detail = useQuery({
    queryKey: ["qt", "activity", picked, years],
    queryFn: () => api.qtActivity(picked!, years),
    enabled: !!picked,
    staleTime: 6 * 3600_000,
  });

  const hits = useMemo(() => {
    const all = list.data?.symbols ?? [];
    const term = q.trim().toUpperCase();
    return term ? all.filter((s) => s.symbol.includes(term)) : all;
  }, [list.data, q]);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <input value={q} onChange={(e) => setQ(e.target.value)}
          placeholder="输入代码，如 ALGM / QMAX.TO…"
          className="h-8 w-56 px-2 rounded-lg bg-sunken text-[13px] outline-none
            focus:ring-2 focus:ring-cyan/40" />
        <span className="label tnum">
          {list.data
            ? `近 ${years} 年交易过 ${list.data.symbols.length} 个代码`
            : "读取中…"}
        </span>
        {picked && (
          <button onClick={() => setPicked(null)} className="text-[12px] text-cyan">
            清除选择
          </button>
        )}
      </div>

      {list.isPending && (
        <p className="label py-6 text-center">读取中…（首次约 30 秒）</p>
      )}
      {list.isError && (
        <p className="text-[12.5px] text-up">{(list.error as ApiError).message}</p>
      )}

      {list.data && (
        <div className="flex flex-wrap gap-1.5">
          {hits.map((s) => (
            <button key={s.symbol} onClick={() => setPicked(s.symbol)}
              title={`${s.trades} 笔买卖 · ${s.activity} 条记录 · ${s.first} → ${s.last}`
                + `\n${s.accounts.join("、")}`}
              className={`h-7 px-2 rounded-md text-[12px] font-mono tnum
                transition-colors ${
                  picked === s.symbol ? "bg-cyan text-white"
                    : "bg-sunken hover:bg-elevated"}`}>
              {s.symbol}
              <span className={`ml-1 text-[10.5px] ${
                picked === s.symbol ? "text-white/70" : "text-ink-mute"}`}>
                {s.trades}
              </span>
            </button>
          ))}
          {hits.length === 0 && (
            <p className="label">没有匹配「{q}」的代码。</p>
          )}
        </div>
      )}

      {detail.isPending && picked && (
        <p className="label py-6 text-center">读取中…</p>
      )}
      {detail.data && <Detail d={detail.data} />}
    </div>
  );
}

function Detail({ d }: { d: QtSymbolActivity }) {
  return (
    <div className="flex flex-col gap-2.5 rounded-lg bg-sunken p-2.5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h4 className="text-[14px] font-semibold font-mono tnum">{d.symbol}</h4>
        <span className="label tnum">
          {d.trade_count} 笔买卖 · {d.rows.length} 条记录
        </span>
        <span className="label font-mono tnum">{d.first} → {d.last}</span>
        <span className="label">{d.accounts.join("、")}</span>
      </div>

      {d.position.by_currency.map((p) => (
        <div key={p.currency}
          className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[12.5px]">
          <span className="label">{p.currency}</span>
          <Stat label="买入" v={`${fixed(p.bought, 0)} 股`}
            note={p.avg_buy == null ? undefined : `均价 ${fixed(p.avg_buy, 2)}`} />
          <Stat label="卖出" v={`${fixed(p.sold, 0)} 股`}
            note={p.avg_sell == null ? undefined : `均价 ${fixed(p.avg_sell, 2)}`} />
          <Stat label="净持股" v={`${fixed(p.net, 0)} 股`}
            tone={p.net > 0 ? "up" : undefined} />
          {p.avg_buy != null && p.avg_sell != null && p.sold > 0 && (
            <span className={`text-[12px] tnum ${
              p.avg_sell >= p.avg_buy ? "text-up" : "text-down"}`}
              title="卖出均价相对买入均价。只是两个简单均价之差，不是损益。">
              卖出均价 {signed((p.avg_sell / p.avg_buy - 1) * 100, 1, "%")}
            </span>
          )}
        </div>
      ))}

      <div className="overflow-x-auto">
        <table className="w-full text-[12px] border-collapse">
          <thead>
            <tr className="text-ink-mute">
              <th className="text-left font-normal pb-1 pr-2">日期</th>
              <th className="text-left font-normal pb-1 px-2">账户</th>
              <th className="text-left font-normal pb-1 px-2">类型</th>
              <th className="text-right font-normal pb-1 px-2">数量</th>
              <th className="text-right font-normal pb-1 px-2">价格</th>
              <th className="text-right font-normal pb-1 px-2">净额</th>
              <th className="text-left font-normal pb-1 pl-2">说明</th>
            </tr>
          </thead>
          <tbody>
            {d.rows.map((r, i) => (
              <tr key={`${r.date}-${i}`} className="border-t border-line">
                <td className="py-1 pr-2 font-mono tnum whitespace-nowrap">{r.date}</td>
                <td className="py-1 px-2 whitespace-nowrap">{r.account}</td>
                <td className="py-1 px-2 whitespace-nowrap">
                  {r.type_label}
                  {r.action && <span className="text-ink-mute"> {r.action}</span>}
                </td>
                <td className={`py-1 px-2 text-right tnum ${
                  (r.quantity ?? 0) > 0 ? "text-up"
                    : (r.quantity ?? 0) < 0 ? "text-down" : "text-ink-mute"}`}>
                  {r.quantity ? fixed(r.quantity, 0) : "—"}
                </td>
                <td className="py-1 px-2 text-right tnum">
                  {r.price ? fixed(r.price, 2) : "—"}
                </td>
                <td className="py-1 px-2 text-right tnum">
                  {r.net == null ? "—"
                    : `${r.currency} ${r.net.toLocaleString(undefined,
                        { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
                </td>
                <td className="py-1 pl-2 text-ink-dim max-w-[240px] truncate"
                  title={r.description}>
                  {r.description}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="label leading-snug">
        买入/卖出均价是按股数加权的<b>简单平均</b>，不是调整后成本基础（ACB），
        也不是损益 —— 跨账户的同一证券、表面亏损规则和每笔的汇率都会改变答案。
      </p>
    </div>
  );
}

function Stat({ label, v, note, tone }: {
  label: string; v: string; note?: string; tone?: "up" | "down";
}) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="label">{label}</span>
      <span className={`tnum font-medium ${
        tone === "up" ? "text-up" : tone === "down" ? "text-down" : ""}`}>
        {v}
      </span>
      {note && <span className="text-[11px] text-ink-mute tnum">{note}</span>}
    </span>
  );
}
