/**
 * 行业暴露 — where the book is, once you look through the ETFs.
 *
 * Counting each holding's own sector and stopping there gives a Canadian
 * account an answer like "60% ETF, 40% technology", which is not an answer:
 * an ETF has no sector, it has eleven. Each fund's weight is distributed
 * across its published sector weightings, so a book that is half VFV and half
 * one chip name shows up as the technology bet it is.
 *
 * Each row splits the number into what you picked and what your index funds
 * picked. Those are different facts and the decision they inform is different:
 * 25% technology you chose is a view, 25% that arrived through VFV is the
 * market's default and rebalances itself.
 *
 * The industry table underneath is stocks only, and says so. An ETF publishes
 * sector weights and never industry weights, so a portfolio-wide industry
 * breakdown cannot exist however much one would like it to.
 */

import type { QtExposure } from "../../lib/types";
import { fixed } from "../../lib/format";

const COLORS = [
  "#0062cc", "#5856d6", "#ff9500", "#1f7a35", "#d70015",
  "#0ea5e9", "#a855f7", "#f59e0b", "#14b8a6", "#ec4899", "#64748b",
];

function money(v: number, ccy: string): string {
  const sym = ccy === "CAD" ? "C$" : "$";
  return `${sym}${Math.round(v).toLocaleString()}`;
}

export function ExposurePanel({ data }: { data: QtExposure }) {
  const c = data.concentration;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <span className="label">
          {data.as_of} · 股票 {fixed(data.split.stock_pct, 1)}% ·
          ETF {fixed(data.split.etf_pct, 1)}%（已穿透到成分行业）
        </span>
        {c.top_sector && (
          <span className="text-[12.5px]">
            最大行业 <b>{c.top_sector} {fixed(c.top_pct, 1)}%</b>
            <span className="text-ink-mute"> · 前三合计 {fixed(c.top3_pct, 1)}%</span>
          </span>
        )}
        {data.unknown_pct > 0.05 && (
          <span className="text-[12px] text-brand-ink">
            {fixed(data.unknown_pct, 1)}% 无法归类
          </span>
        )}
      </div>

      <div className="flex h-6 rounded-md overflow-hidden bg-sunken">
        {data.rows.map((r, i) => (
          <div key={r.label} style={{
            width: `${r.pct}%`,
            background: r.sector ? COLORS[i % COLORS.length] : "var(--color-flat)",
          }} title={`${r.label} ${fixed(r.pct, 1)}%`} />
        ))}
      </div>

      <div className="overflow-auto rounded-lg border border-line">
        <table className="w-full border-collapse text-[12.5px]">
          <thead className="bg-panel">
            <tr className="border-b border-line text-ink-mute">
              <th className="text-left font-medium px-2 py-1.5">行业</th>
              <th className="text-right font-medium px-2 py-1.5">占比</th>
              <th className="text-right font-medium px-2 py-1.5"
                title="你直接持有的个股带来的部分">个股</th>
              <th className="text-right font-medium px-2 py-1.5"
                title="通过 ETF 间接持有的部分">ETF 穿透</th>
              <th className="text-right font-medium px-2 py-1.5">市值</th>
              <th className="text-left font-medium px-2 py-1.5">来自</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((r, i) => (
              <tr key={r.label} className="border-b border-line/60 hover:bg-sunken">
                <td className="px-2 py-1 whitespace-nowrap">
                  <i className="inline-block w-2 h-2 rounded-sm mr-1.5" style={{
                    background: r.sector ? COLORS[i % COLORS.length] : "var(--color-flat)" }} />
                  {r.label}
                </td>
                <td className="px-2 py-1 text-right tnum font-medium">{fixed(r.pct, 1)}%</td>
                <td className="px-2 py-1 text-right tnum text-ink-mute">
                  {r.from_stocks_pct ? `${fixed(r.from_stocks_pct, 1)}%` : "—"}
                </td>
                <td className="px-2 py-1 text-right tnum text-ink-mute">
                  {r.from_etfs_pct ? `${fixed(r.from_etfs_pct, 1)}%` : "—"}
                </td>
                <td className="px-2 py-1 text-right tnum">{money(r.value, data.base)}</td>
                <td className="px-2 py-1 text-ink-mute max-w-[260px] truncate"
                  title={r.holdings.map((h) => `${h.symbol} ${h.name}`).join("、")}>
                  {r.holdings.map((h) => h.symbol).join("、")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data.industries.length > 0 && (
        <div className="flex flex-col gap-1">
          <div className="flex items-baseline gap-2">
            <span className="text-[13px] font-medium">细分行业</span>
            <span className="label">
              仅个股部分（{money(data.industry_basis, data.base)}）——
              ETF 只公布行业大类，没有细分行业权重，混在一起会变成一张贴错标签的表。
            </span>
          </div>
          <div className="overflow-auto rounded-lg border border-line max-h-[300px]">
            <table className="w-full border-collapse text-[12px]">
              <tbody>
                {data.industries.map((r) => (
                  <tr key={r.name} className="border-b border-line/60">
                    <td className="px-2 py-1">{r.name}</td>
                    <td className="px-2 py-1 text-right tnum w-20">{fixed(r.pct, 1)}%</td>
                    <td className="px-2 py-1 w-40">
                      <div className="h-1.5 rounded-full bg-sunken">
                        <div className="h-1.5 rounded-full bg-cyan"
                          style={{ width: `${Math.min(r.pct ?? 0, 100)}%` }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
