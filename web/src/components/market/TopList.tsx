/**
 * 龙虎榜 — the stocks whose trading was abnormal enough for the exchange to
 * publish who bought and sold them.
 *
 * Ranked by net amount, which is the number the list exists for: a stock can
 * appear because it moved 10% and still have been net SOLD by the seats on the
 * list, and that is a different fact from the price.
 *
 * Every row links through to the full analysis rather than opening a modal —
 * the natural next question is "what does this one's chart look like", and a
 * link can be middle-clicked into a tab, which a modal cannot.
 */

import { Link } from "react-router-dom";
import type { TopList as TopListData } from "../../lib/types";
import { fixed, moveClass, signed } from "../../lib/format";

export function TopList({ data }: { data: TopListData }) {
  const buys = data.rows.filter((r) => (r.net ?? 0) > 0).length;
  const sells = data.rows.filter((r) => (r.net ?? 0) < 0).length;

  return (
    <div className="flex flex-col gap-2">
      <p className="label">
        {data.trade_date} · 共 {data.rows.length} 只 ·
        <b className="text-up"> 净买入 {buys}</b> ·
        <b className="text-down"> 净卖出 {sells}</b>
        　按当日龙虎榜净额排序。
      </p>

      <div className="overflow-auto max-h-[420px] rounded-lg border border-line">
        <table className="w-full border-collapse text-[12px]">
          <thead className="sticky top-0 z-10 bg-panel">
            <tr className="border-b border-line text-ink-mute">
              <th className="text-left font-medium px-2 py-1.5">代码</th>
              <th className="text-left font-medium px-2 py-1.5">名称</th>
              <th className="text-right font-medium px-2 py-1.5">收盘</th>
              <th className="text-right font-medium px-2 py-1.5">涨跌</th>
              <th className="text-right font-medium px-2 py-1.5" title="龙虎榜席位净买入金额">
                净额 (万)
              </th>
              <th className="text-right font-medium px-2 py-1.5"
                title="净额占当日成交额的比例">占比</th>
              <th className="text-left font-medium px-2 py-1.5">上榜理由</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((r, i) => (
              <tr key={`${r.t}-${i}`} className="border-b border-line/60 hover:bg-sunken">
                <td className="px-2 py-1">
                  <Link to={`/?t=${r.t}`} className="text-cyan font-mono font-semibold">
                    {r.t}
                  </Link>
                </td>
                <td className="px-2 py-1 whitespace-nowrap">{r.n}</td>
                <td className="px-2 py-1 text-right tnum">{fixed(r.close, 2)}</td>
                <td className={`px-2 py-1 text-right tnum font-medium ${moveClass(r.pct)}`}>
                  {signed(r.pct, 2)}%
                </td>
                <td className={`px-2 py-1 text-right tnum ${moveClass(r.net)}`}>
                  {r.net == null ? "—" : signed(r.net, 0)}
                </td>
                <td className={`px-2 py-1 text-right tnum ${moveClass(r.net_rate)}`}>
                  {signed(r.net_rate, 2)}%
                </td>
                <td className="px-2 py-1 text-ink-mute whitespace-nowrap">{r.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
