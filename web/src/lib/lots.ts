/**
 * Each fill measured against today's price.
 *
 * The question this answers is a sizing one — "of the shares I bought, which
 * were bought above where it trades now, and by how much?" — so every number
 * is plain price arithmetic on one fill. It is NOT a realised gain, not an
 * adjusted cost base, and commissions are left out: those belong to the
 * ledger and the tax return, not to "should I add or trim this".
 *
 * Only buys and sells get a comparison. A dividend has no price to compare,
 * and pretending it does would put a number where there is no question.
 */

import type { Num, QtSymbolActivity } from "./types";

export type ActivityRow = QtSymbolActivity["rows"][number];

export type Lot = {
  side: "buy" | "sell";
  /** Always positive; `side` carries the direction. */
  quantity: number;
  price: number;
  /** Today's price relative to the fill: +20 means it trades 20% above it. */
  vs_now_pct: Num;
  /**
   * Buy: what that lot is up or down at today's price.
   * Sell: what selling then made versus selling now — positive means the
   * shares went for more than they fetch today.
   */
  pnl: Num;
};

/** A row is a fill when it is a trade with a quantity and a price. */
export function lotOf(r: ActivityRow, now: Num, ccy: string): Lot | null {
  if (r.type !== "Trades") return null;
  const q = r.quantity ?? 0;
  const p = r.price ?? 0;
  if (!q || !p) return null;
  const side = q > 0 ? "buy" : "sell";
  const quantity = Math.abs(q);
  // A fill in another currency (a .U listing bought with USD, say) has no
  // honest comparison with a price quoted in this one.
  const comparable = now != null && Number.isFinite(now) && r.currency === ccy;
  if (!comparable) return { side, quantity, price: p, vs_now_pct: null, pnl: null };
  const diff = side === "buy" ? now - p : p - now;
  return {
    side, quantity, price: p,
    vs_now_pct: (now / p - 1) * 100,
    pnl: diff * quantity,
  };
}

export type LotSummary = {
  buys: number;
  /** Buys whose price is below today's — the ones in the money. */
  below: number;
  above: number;
  bought: number;
  avg_buy: Num;
  avg_vs_now_pct: Num;
  /** Every buy marked to today's price, summed. Ignores what was sold since. */
  buy_pnl: Num;
  sells: number;
  sold: number;
  avg_sell: Num;
};

export function summarise(rows: ActivityRow[], now: Num, ccy: string): LotSummary {
  let buys = 0, below = 0, above = 0, bought = 0, cost = 0, pnl = 0, priced = 0;
  let sells = 0, sold = 0, proceeds = 0;
  for (const r of rows) {
    const lot = lotOf(r, now, ccy);
    if (!lot || r.currency !== ccy) continue;
    if (lot.side === "buy") {
      buys += 1;
      bought += lot.quantity;
      cost += lot.quantity * lot.price;
      if (lot.pnl != null) {
        priced += 1;
        pnl += lot.pnl;
        if (lot.price < now!) below += 1;
        else if (lot.price > now!) above += 1;
      }
    } else {
      sells += 1;
      sold += lot.quantity;
      proceeds += lot.quantity * lot.price;
    }
  }
  const avg_buy = bought ? cost / bought : null;
  return {
    buys, below, above, bought, avg_buy,
    avg_vs_now_pct: avg_buy && now != null && Number.isFinite(now)
      ? (now / avg_buy - 1) * 100 : null,
    buy_pnl: priced ? pnl : null,
    sells, sold,
    avg_sell: sold ? proceeds / sold : null,
  };
}

/** A dated cash flow, from the investor's side: money out is negative. */
export type Flow = { date: string; amount: number };

const YEAR_MS = 365 * 86400e3;

/**
 * The money-weighted return (XIRR): the one annual rate at which every cash
 * flow, discounted to its own date, nets to zero. Annual, as a fraction.
 *
 * Solved by bisection rather than Newton: a ticker's flows can be a dozen
 * buys, sells and dividends in any order, and bisection cannot wander off
 * to a nonsense root the way Newton can from a bad first guess.
 */
export function xirr(flows: Flow[]): Num {
  const fs = flows.filter((f) => f.amount !== 0 && Number.isFinite(f.amount) && f.date);
  if (!fs.some((f) => f.amount < 0) || !fs.some((f) => f.amount > 0)) return null;
  const t0 = Math.min(...fs.map((f) => Date.parse(f.date)));
  const ts = fs.map((f) => ({ t: (Date.parse(f.date) - t0) / YEAR_MS, a: f.amount }));
  if (!ts.some((f) => f.t > 0)) return null;
  const npv = (r: number) => {
    const l = Math.log1p(r);
    return ts.reduce((acc, f) => acc + f.a * Math.exp(-f.t * l), 0);
  };
  let lo = -0.9999, hi = 10;
  let flo = npv(lo), fhi = npv(hi);
  while (flo * fhi > 0 && hi < 1e6) { hi *= 10; fhi = npv(hi); }
  if (flo * fhi > 0) return null;
  for (let i = 0; i < 300 && hi - lo > 1e-12; i++) {
    const mid = (lo + hi) / 2;
    const fm = npv(mid);
    if (fm === 0) return mid;
    if (flo * fm < 0) { hi = mid; fhi = fm; } else { lo = mid; flo = fm; }
  }
  return (lo + hi) / 2;
}

/** Types whose quantity is a real change of holding with its cash on record. */
const COUNTED = new Set(["Trades", "Dividends", "Corporate actions"]);

/** How far back an unpaid ex-date is still worth looking for. */
const PENDING_DAYS = 45;
/** A payment dated a little before the ex-date still counts as that one. */
const PAY_SLACK_DAYS = 3;

const dayDiff = (a: string, b: string) => (Date.parse(b) - Date.parse(a)) / 86400e3;

export type PendingDividend = {
  ex_date: string; per_share: number; shares: number; amount: number;
};

export type ReturnContext = {
  /** Local "YYYY-MM-DD" — the date today's market value is counted on. */
  today: string;
  /** Raw closes, ascending, from the start of the activity window. */
  closes?: [string, number][];
  /** Ex-dates and per-share amounts. */
  dividends?: { ex_date: string; amount: number }[];
};

/**
 * Everything one ticker has made or lost, from the cash that actually moved.
 *
 *   total = today's market value + every cash flow on record
 *
 * Buys are cash out (commission included), sells and dividends are cash in,
 * fees and withholding are cash out — Questrade's `net` on each row already
 * carries the sign and the commission.
 *
 * Three ways it can come out, in `mode`:
 *  - "full": the records cover every share held. Exact, and split into what
 *    is still open (Questrade's own open P&L) and what the sells locked in.
 *  - "window": some shares were held before the window opened. They are
 *    counted as bought at the first close of the window, so this is the
 *    return over the window — a standard way to measure a period — not
 *    since the first purchase.
 *  - "partial": the gap cannot be priced (no price history, or more shares
 *    on record than are held). Only the recorded shares are totalled, and no
 *    rate is given.
 *
 * Dividends are counted when the cash lands, which is what a money-weighted
 * return wants. But the price drops on the ex-date, a week or so before the
 * cash arrives, so in that gap the market value has already lost the
 * dividend and the records do not have it yet. Monthly payers (QMAX, KLIP)
 * sit in that gap a quarter of the time. Any ex-date in the last 45 days
 * with no payment since is added back as 待到账, at the per-share amount
 * times the shares held going into the ex-date.
 */
export type TotalReturn = {
  mode: "full" | "window" | "partial";
  total: number;
  /** Money that went in: every buy, plus the opening position in "window". */
  invested: number;
  value: number;
  opening: { date: string; price: number; shares: number; value: number } | null;
  unrealised: Num;
  realised: Num;
  dividends: number;
  pending: PendingDividend[];
  pending_total: number;
  /** Fees, withholding and any other cash row tagged with this symbol. */
  other: number;
  /** Shares the records account for, vs shares held. */
  covered: number;
  held: number;
  complete: boolean;
  /** Rows in another currency, left out rather than mixed in. */
  skipped: number;
  /** Money-weighted return, %. Annualised only over a year or more. */
  mwr: Num;
  mwr_annualised: boolean;
  /** From the first cash flow to today. */
  days: number;
  since: string | null;
};

export function totalReturn(rows: ActivityRow[], h: {
  price: Num; quantity: number; currency: string; open_pnl: Num;
}, ctx: ReturnContext): TotalReturn {
  const flows: Flow[] = [];
  let trading = 0, dividends = 0, other = 0, invested = 0;
  let covered = 0, skipped = 0;
  const mine = rows.filter((r) => {
    if (r.currency === h.currency) return true;
    skipped += 1;
    return false;
  });
  for (const r of mine) {
    const net = r.net ?? 0;
    if (net) flows.push({ date: r.date, amount: net });
    if (r.type === "Trades") {
      trading += net;
      if ((r.quantity ?? 0) > 0) invested += -net;
    } else if (r.type === "Dividends") {
      dividends += net;
    } else {
      other += net;
    }
    if (COUNTED.has(r.type)) covered += r.quantity ?? 0;
  }

  /** Shares held going into `day`: today's, less every change since. */
  const heldBefore = (day: string) => h.quantity - mine
    .filter((r) => COUNTED.has(r.type) && r.date >= day)
    .reduce((acc, r) => acc + (r.quantity ?? 0), 0);

  const price = h.price != null && Number.isFinite(h.price) ? h.price : null;
  const gap = h.quantity - covered;
  const complete = Math.abs(gap) < 1e-6;
  const first = ctx.closes?.[0];
  const mode: TotalReturn["mode"] = complete ? "full"
    : gap > 0 && first && price != null ? "window" : "partial";

  let opening: TotalReturn["opening"] = null;
  if (mode === "window") {
    const [date, close] = first!;
    opening = { date, price: close, shares: gap, value: gap * close };
    flows.push({ date, amount: -opening.value });
    invested += opening.value;
  }

  const pending: PendingDividend[] = [];
  for (const d of ctx.dividends ?? []) {
    const age = dayDiff(d.ex_date, ctx.today);
    if (age < 0 || age > PENDING_DAYS) continue;
    const paid = mine.some((r) => r.type === "Dividends" && (r.net ?? 0) > 0
      && dayDiff(d.ex_date, r.date) >= -PAY_SLACK_DAYS);
    if (paid) continue;
    const shares = heldBefore(d.ex_date);
    if (shares > 1e-9) {
      pending.push({ ex_date: d.ex_date, per_share: d.amount, shares,
                     amount: shares * d.amount });
    }
  }
  const pending_total = pending.reduce((a, p) => a + p.amount, 0);

  const shares = mode === "partial" ? Math.max(0, Math.min(covered, h.quantity)) : h.quantity;
  const value = price != null ? shares * price : 0;
  const recorded = flows.reduce((a, f) => a + f.amount, 0);
  const total = value + pending_total + recorded;

  const unrealised = mode === "full" && h.open_pnl != null ? h.open_pnl : null;

  let mwr: Num = null, annualised = false, days = 0;
  const since = flows.length ? flows.map((f) => f.date).sort()[0] ?? null : null;
  if (since) days = Math.max(0, dayDiff(since, ctx.today));
  if (mode !== "partial" && since && days > 0) {
    const r = xirr([...flows, { date: ctx.today, amount: value + pending_total }]);
    if (r != null) {
      annualised = days >= 365;
      mwr = (annualised ? r : Math.pow(1 + r, days / 365) - 1) * 100;
    }
  }

  return {
    mode, total, invested, value, opening,
    unrealised,
    realised: unrealised == null ? null : value + trading - unrealised,
    dividends, pending, pending_total, other,
    covered, held: h.quantity, complete, skipped,
    mwr, mwr_annualised: annualised, days, since,
  };
}
