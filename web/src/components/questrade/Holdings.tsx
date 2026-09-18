/**
 * 持仓 — the whole book, one row per name.
 *
 * Not one row per account-position, which is how Questrade reports it: the
 * same name sits in a TFSA and a margin account routinely, and a table that
 * lists it twice answers neither "how much do I own" nor "what is my biggest
 * position". Rows that span accounts expand to show the split.
 *
 * Average cost is shown per account inside the expansion as well as blended on
 * the row. That is not redundancy — ACB is tracked per account, a TFSA has no
 * cost base for tax at all, and an RRSP loss cannot be harvested. The blended
 * number is for sizing; the per-account ones are the real figures.
 *
 * Every value appears twice where it matters: in the instrument's own currency
 * (which is what the Questrade screen shows) and converted into the base
 * currency (which is the only way to add the book up).
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import type { QtBook, QtHolding } from "../../lib/types";
import { fixed, moveClass, signed } from "../../lib/format";

/** North America: green up, red down — the opposite of the A-share pages. */
const NA = false;

function cash(v: number | null | undefined, ccy: string, nd = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const sym = ccy === "CAD" ? "C$" : ccy === "USD" ? "$" : "";
  return `${sym}${v.toLocaleString(undefined, {
    minimumFractionDigits: nd, maximumFractionDigits: nd })}`;
}

export function Holdings({ book }: { book: QtBook }) {
  const [open, setOpen] = useState<Set<string>>(new Set());
  const toggle = (s: string) =>
    setOpen((prev) => {
      const next = new Set(prev);
      next.has(s) ? next.delete(s) : next.add(s);
      return next;
    });

  return (
    <div className="overflow-auto rounded-lg border border-line">
      <table className="w-full border-collapse text-[12.5px]">
        <thead className="sticky top-0 z-10 bg-panel">
          <tr className="border-b border-line text-ink-mute">
            <Th>代码</Th>
            <Th>名称</Th>
            <Th>类型</Th>
            <Th right>数量</Th>
            <Th right title="按成本加权；各账户单独的成本请展开查看">均价</Th>
            <Th right>现价</Th>
            <Th right>市值（原币）</Th>
            <Th right>市值（{book.base}）</Th>
            <Th right>浮动盈亏</Th>
            <Th right>权重</Th>
          </tr>
        </thead>
        <tbody>
          {book.holdings.map((h) => (
            <Row key={h.symbol} h={h} base={book.base}
              open={open.has(h.symbol)} onToggle={() => toggle(h.symbol)} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Th({ children, right, title }: {
  children: React.ReactNode; right?: boolean; title?: string;
}) {
  return (
    <th title={title}
      className={`font-medium px-2 py-1.5 whitespace-nowrap ${right ? "text-right" : "text-left"}`}>
      {children}
    </th>
  );
}

function Row({ h, base, open, onToggle }: {
  h: QtHolding; base: string; open: boolean; onToggle: () => void;
}) {
  return (
    <>
      <tr className="border-b border-line/60 hover:bg-sunken">
        <td className="px-2 py-1 whitespace-nowrap">
          {h.yahoo ? (
            // Straight into the technical-analysis page for that instrument.
            <Link to={`/?t=${h.currency === "CAD" ? "CA" : "US"}:${h.yahoo}`}
              className="text-cyan font-mono font-semibold">{h.symbol}</Link>
          ) : (
            <span className="font-mono font-semibold text-ink-mute"
              title="无法匹配行情源，未纳入风险分析">{h.symbol}</span>
          )}
        </td>
        <td className="px-2 py-1 max-w-[220px] truncate" title={h.name}>{h.name}</td>
        <td className="px-2 py-1 whitespace-nowrap">
          <Tag>{h.kind}</Tag>
          <Tag muted>{h.currency}</Tag>
          {h.split && (
            <button onClick={onToggle}
              className="text-[11px] text-cyan ml-1"
              title="这只股票分布在多个账户">
              {open ? "收起" : `${h.accounts.length} 个账户`}
            </button>
          )}
        </td>
        <td className="px-2 py-1 text-right tnum">{fixed(h.quantity, h.quantity % 1 ? 4 : 0)}</td>
        <td className="px-2 py-1 text-right tnum">{cash(h.avg_cost, h.currency)}</td>
        <td className="px-2 py-1 text-right tnum">{cash(h.price, h.currency)}</td>
        <td className="px-2 py-1 text-right tnum">{cash(h.market_value, h.currency, 0)}</td>
        <td className="px-2 py-1 text-right tnum font-medium">
          {cash(h.market_value_base, base, 0)}
        </td>
        <td className={`px-2 py-1 text-right tnum ${moveClass(h.open_pnl, NA)}`}>
          {cash(h.open_pnl, h.currency, 0)}
          <span className="text-[11px] opacity-80"> {signed(h.open_pnl_pct, 1)}%</span>
        </td>
        <td className="px-2 py-1 text-right tnum">{fixed(h.weight_pct, 1)}%</td>
      </tr>

      {open && h.accounts.map((a) => (
        <tr key={a.id} className="border-b border-line/40 bg-sunken/60 text-[11.5px]">
          <td />
          <td className="px-2 py-1 text-ink-mute" colSpan={2}>↳ {a.label}</td>
          <td className="px-2 py-1 text-right tnum">{fixed(a.quantity, a.quantity % 1 ? 4 : 0)}</td>
          <td className="px-2 py-1 text-right tnum" title="该账户自己的平均成本">
            {cash(a.avg_cost, h.currency)}
          </td>
          <td />
          <td className="px-2 py-1 text-right tnum">{cash(a.market_value, h.currency, 0)}</td>
          <td className="px-2 py-1 text-right tnum">{cash(a.market_value_base, base, 0)}</td>
          <td className={`px-2 py-1 text-right tnum ${moveClass(a.open_pnl, NA)}`}>
            {cash(a.open_pnl, h.currency, 0)}
          </td>
          <td />
        </tr>
      ))}
    </>
  );
}

function Tag({ children, muted }: { children: React.ReactNode; muted?: boolean }) {
  return (
    <span className={`inline-block px-1.5 h-[17px] leading-[17px] rounded text-[10.5px]
      font-medium mr-1 ${muted ? "bg-sunken text-ink-mute" : "bg-cyan/10 text-cyan"}`}>
      {children}
    </span>
  );
}
