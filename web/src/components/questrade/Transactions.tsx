/**
 * 📒 交易流水 — a calendar year of activity, per account.
 *
 * Built for putting a return together, so three things drive the design:
 *
 *  · Accounts stay apart, and each says whether it is sheltered. A gain in a
 *    TFSA is not reportable and a loss in an RRSP is not claimable, so the
 *    account a row sits in changes what the row means.
 *  · CAD and USD are never added together. They are different money.
 *  · Money moved between your own accounts is marked. It appears twice —
 *    a withdrawal in one account and a deposit in another — and counting it
 *    as new money is the easiest mistake to make here.
 *
 * What this does NOT do is compute adjusted cost base or capital gains.
 * Superficial-loss rules, identical property held across accounts, and the
 * rate applying to each leg all change the answer, and a plausible wrong
 * number on a tax return is worse than no number.
 */

import { useMemo, useState } from "react";
import type { QtTransactions } from "../../lib/types";
import { fixed } from "../../lib/format";

const ALL = "__all__";

export function Transactions({ d, year, onYear, years }: {
  d: QtTransactions; year: number; onYear: (y: number) => void; years: number[];
}) {
  const [acct, setAcct] = useState<string>(ALL);
  const [type, setType] = useState<string>(ALL);
  const [hideInternal, setHideInternal] = useState(false);

  const accounts = acct === ALL ? d.accounts
    : d.accounts.filter((a) => a.id === acct);

  const rows = useMemo(() => accounts.flatMap((a) => a.rows
    .filter((r) => (type === ALL || r.type === type))
    .filter((r) => !(hideInternal && r.internal))
    .map((r) => ({ ...r, account: a.label, registered: a.registered }))
  ).sort((x, y) => (x.date < y.date ? 1 : x.date > y.date ? -1 : 0)),
  [accounts, type, hideInternal]);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <select value={year} onChange={(e) => onYear(Number(e.target.value))}
          aria-label="纳税年度"
          className="h-8 px-2 rounded-lg bg-sunken text-[13px] font-medium
            outline-none focus:ring-2 focus:ring-cyan/40">
          {years.map((y) => <option key={y} value={y}>{y} 年</option>)}
        </select>

        <select value={acct} onChange={(e) => setAcct(e.target.value)}
          aria-label="账户"
          className="h-8 px-2 rounded-lg bg-sunken text-[13px] outline-none
            focus:ring-2 focus:ring-cyan/40 max-w-[200px]">
          <option value={ALL}>全部账户</option>
          {d.accounts.map((a) => (
            <option key={a.id} value={a.id}>{a.label} ···{a.tail}</option>
          ))}
        </select>

        <select value={type} onChange={(e) => setType(e.target.value)}
          aria-label="类型"
          className="h-8 px-2 rounded-lg bg-sunken text-[13px] outline-none
            focus:ring-2 focus:ring-cyan/40">
          <option value={ALL}>全部类型</option>
          {d.types.map((t) => <option key={t.type} value={t.type}>{t.label}</option>)}
        </select>

        <label className="label flex items-center gap-1.5 cursor-pointer"
          title="账户之间的划转会在两边各出现一次，合计存入时会被重复计算">
          <input type="checkbox" checked={hideInternal}
            onChange={(e) => setHideInternal(e.target.checked)}
            className="accent-[var(--color-cyan)]" />
          隐藏内部划转
        </label>

        <span className="label tnum">{rows.length} 笔</span>

        <button onClick={() => downloadCsv(d, rows)}
          className="ml-auto h-8 px-3 rounded-lg bg-sunken text-[12.5px] font-medium">
          ⬇ 导出 CSV
        </button>
      </div>

      {d.partial && (
        <p className="label">
          {d.year} 年尚未结束 —— 数据截至 {d.through}。
        </p>
      )}

      <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3 items-start">
        {d.accounts.map((a) => <AccountCard key={a.id} a={a} />)}
      </div>

      {d.internal_transfers.length > 0 && (
        <div className="rounded-lg bg-sunken p-2.5 flex flex-col gap-1">
          <span className="label">
            账户之间的划转 · {d.internal_transfers.length} 笔 ——
            这些不是新进来的钱，两边各记了一次
          </span>
          {d.internal_transfers.map((t, i) => (
            <div key={i} className="flex flex-wrap items-baseline gap-2 text-[12px]">
              <span className="font-mono tnum text-ink-mute">{t.date}</span>
              <span>{t.from} → {t.to}</span>
              <span className="tnum font-medium">
                {t.currency} {t.amount.toLocaleString(undefined,
                  { minimumFractionDigits: 2 })}
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-[12px] border-collapse">
          <thead>
            <tr className="text-ink-mute">
              <Th>交易日</Th>
              <Th>账户</Th>
              <Th>类型</Th>
              <Th>代码</Th>
              <Th right>数量</Th>
              <Th right>价格</Th>
              <Th right>佣金</Th>
              <Th right>净额</Th>
              <Th>说明</Th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={`${r.date}-${r.type}-${r.symbol}-${i}`}
                className="border-t border-line">
                <td className="py-1 pr-2 font-mono tnum whitespace-nowrap"
                  title={r.settled && r.settled !== r.date
                    ? `交收日 ${r.settled}` : undefined}>
                  {r.date}
                </td>
                <td className="py-1 px-2 whitespace-nowrap">
                  <span className={r.registered ? "text-ink-dim" : "text-ink"}>
                    {r.account}
                  </span>
                </td>
                <td className="py-1 px-2 whitespace-nowrap">
                  {r.type_label}
                  {r.action && r.action !== r.type_label && (
                    <span className="text-ink-mute"> {r.action}</span>
                  )}
                  {r.internal && (
                    <span className="text-brand-ink" title="账户之间的划转">
                      {" "}⇄
                    </span>
                  )}
                </td>
                <td className="py-1 px-2 font-mono tnum">{r.symbol || "—"}</td>
                <td className="py-1 px-2 text-right tnum">
                  {r.quantity == null || r.quantity === 0 ? "—" : fixed(r.quantity, 0)}
                </td>
                <td className="py-1 px-2 text-right tnum">
                  {r.price == null || r.price === 0 ? "—" : fixed(r.price, 2)}
                </td>
                <td className="py-1 px-2 text-right tnum text-ink-mute">
                  {r.commission == null || r.commission === 0
                    ? "—" : fixed(r.commission, 2)}
                </td>
                <td className={`py-1 px-2 text-right tnum font-medium ${
                  (r.net ?? 0) > 0 ? "text-up" : (r.net ?? 0) < 0 ? "text-down" : ""}`}>
                  {r.net == null ? "—"
                    : `${r.currency} ${r.net.toLocaleString(undefined,
                        { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
                </td>
                <td className="py-1 pl-2 text-ink-dim max-w-[280px] truncate"
                  title={r.description}>
                  {r.description}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && (
          <p className="label py-6 text-center">这个筛选下没有记录。</p>
        )}
      </div>

      <p className="label leading-snug">
        日期是<b>交易日</b>（处置发生的那天，决定计入哪个纳税年度）；交收日在悬停提示里。
        这里只整理原始记录 —— <b>不计算调整后成本基础（ACB）或资本利得</b>：
        跨账户的同一证券、表面亏损规则、每一笔适用的汇率都会改变答案，
        这几项请交给会计师或报税软件。
      </p>
    </div>
  );
}

function AccountCard({ a }: { a: QtTransactions["accounts"][number] }) {
  return (
    <div className="rounded-lg bg-sunken p-2.5 flex flex-col gap-1.5">
      <div className="flex items-baseline gap-2">
        <span className="text-[13px] font-semibold">{a.label}</span>
        <span className="label font-mono tnum">···{a.tail}</span>
        <span className={`ml-auto text-[11px] font-medium px-1.5 py-0.5 rounded ${
          a.registered ? "text-ink-mute bg-panel" : "text-brand-ink bg-brand-ink/10"}`}
          title={a.registered
            ? "注册账户：账户内的收益不计税，亏损也不可抵扣"
            : "非注册账户：资本利得与股息需要申报"}>
          {a.registered ? "免税/递延" : "应税"}
        </span>
      </div>
      {a.summary.by_type.length === 0 ? (
        <span className="label">这一年没有记录</span>
      ) : a.summary.by_type.map((b) => (
        <div key={b.type} className="flex flex-wrap items-baseline gap-x-2 text-[12px]">
          <span className="w-20 shrink-0 text-ink-dim">{b.type_label}</span>
          <span className="label tnum w-10 shrink-0">{b.count} 笔</span>
          {b.by_currency.map((c) => (
            <span key={c.currency} className="tnum">
              <span className="text-ink-mute">{c.currency}</span>{" "}
              <span className={c.net > 0 ? "text-up" : c.net < 0 ? "text-down" : ""}>
                {c.net.toLocaleString(undefined, { minimumFractionDigits: 2,
                                                   maximumFractionDigits: 2 })}
              </span>
              {c.internal_net !== 0 && (
                <span className="text-brand-ink" title="其中属于账户间划转的部分">
                  {" "}(含划转 {c.internal_net.toLocaleString(undefined,
                    { maximumFractionDigits: 0 })})
                </span>
              )}
            </span>
          ))}
        </div>
      ))}
    </div>
  );
}

function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return (
    <th className={`font-normal pb-1 px-2 whitespace-nowrap ${
      right ? "text-right" : "text-left"}`}>
      {children}
    </th>
  );
}

type Row = QtTransactions["accounts"][number]["rows"][number]
  & { account: string; registered: boolean };

/** Whatever is on screen, in the shape a spreadsheet or an accountant wants. */
function downloadCsv(d: QtTransactions, rows: Row[]) {
  const head = ["交易日", "交收日", "账户", "账户性质", "类型", "动作", "代码",
                "数量", "价格", "佣金", "毛额", "净额", "货币", "内部划转", "说明"];
  const cell = (v: unknown) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const body = rows.map((r) => [
    r.date, r.settled, r.account, r.registered ? "注册" : "非注册",
    r.type_label, r.action, r.symbol, r.quantity, r.price, r.commission,
    r.gross, r.net, r.currency, r.internal ? "是" : "", r.description,
  ].map(cell).join(","));

  // BOM so Excel opens the Chinese headers as UTF-8 rather than mojibake.
  const blob = new Blob(["﻿" + [head.join(","), ...body].join("\r\n")],
                        { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `questrade-${d.year}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}
