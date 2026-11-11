import { describe, expect, it } from "vitest";
import { lotOf, summarise, totalReturn, xirr, type ActivityRow } from "./lots";

const row = (over: Partial<ActivityRow>): ActivityRow => ({
  date: "2026-01-02", settled: "2026-01-03", type: "Trades", type_label: "买卖",
  action: "Buy", symbol: "VFV.TO", description: "", quantity: 10, price: 100,
  gross: null, commission: null, net: null, currency: "CAD", internal: false,
  account: "TFSA", account_id: "1", registered: true, ...over,
});

describe("lotOf", () => {
  it("marks a buy to today's price", () => {
    const lot = lotOf(row({ quantity: 10, price: 100 }), 120, "CAD")!;
    expect(lot.side).toBe("buy");
    expect(lot.vs_now_pct).toBeCloseTo(20);
    expect(lot.pnl).toBeCloseTo(200);
  });

  it("scores a sell by what it fetched against today", () => {
    // Sold at 100, now 80: selling then was worth 20 a share more.
    const lot = lotOf(row({ quantity: -5, price: 100, action: "Sell" }), 80, "CAD")!;
    expect(lot.side).toBe("sell");
    expect(lot.quantity).toBe(5);
    expect(lot.pnl).toBeCloseTo(100);
    expect(lot.vs_now_pct).toBeCloseTo(-20);
  });

  it("gives no comparison for a dividend", () => {
    expect(lotOf(row({ type: "Dividends", quantity: 0, price: 0, net: 12 }), 120, "CAD"))
      .toBeNull();
  });

  it("refuses to compare across currencies", () => {
    const lot = lotOf(row({ currency: "USD" }), 120, "CAD")!;
    expect(lot.pnl).toBeNull();
    expect(lot.vs_now_pct).toBeNull();
  });
});

describe("summarise", () => {
  it("counts buys above and below today's price and averages them", () => {
    const s = summarise([
      row({ quantity: 10, price: 90 }),
      row({ quantity: 10, price: 110 }),
      row({ quantity: 20, price: 95 }),
      row({ quantity: -5, price: 130, action: "Sell" }),
      row({ type: "Dividends", quantity: 0, price: 0, net: 4 }),
    ], 100, "CAD");
    expect(s.buys).toBe(3);
    expect(s.below).toBe(2);
    expect(s.above).toBe(1);
    expect(s.bought).toBe(40);
    expect(s.avg_buy).toBeCloseTo(97.5);
    expect(s.avg_vs_now_pct).toBeCloseTo((100 / 97.5 - 1) * 100);
    expect(s.buy_pnl).toBeCloseTo(100 + -100 + 100);
    expect(s.sells).toBe(1);
    expect(s.avg_sell).toBeCloseTo(130);
  });

  it("has no averages when nothing was traded", () => {
    const s = summarise([row({ type: "Dividends", quantity: 0, price: 0 })], 100, "CAD");
    expect(s.avg_buy).toBeNull();
    expect(s.buy_pnl).toBeNull();
  });
});

describe("xirr", () => {
  it("is the plain rate for one flow in and one out a year apart", () => {
    expect(xirr([{ date: "2025-01-01", amount: -1000 },
                 { date: "2026-01-01", amount: 1100 }])).toBeCloseTo(0.1, 6);
  });

  it("weights money by how long it was at work", () => {
    // Same $100 gain on $2,000 in, but half the money arrived for the last
    // half-year only: the rate is higher than 100/2000 = 5%.
    const r = xirr([{ date: "2025-01-01", amount: -1000 },
                    { date: "2025-07-02", amount: -1000 },
                    { date: "2026-01-01", amount: 2100 }])!;
    expect(r).toBeGreaterThan(0.06);
    expect(r).toBeLessThan(0.075);
  });

  it("has no answer without money both going in and coming out", () => {
    expect(xirr([{ date: "2025-01-01", amount: -1000 }])).toBeNull();
    expect(xirr([{ date: "2025-01-01", amount: -1000 },
                 { date: "2025-01-01", amount: 1100 }])).toBeNull();
  });
});

describe("totalReturn", () => {
  const h = { price: 120, quantity: 15, currency: "CAD", open_pnl: 280 };
  const ctx = { today: "2026-11-05" };
  const rows = [
    row({ date: "2025-01-02", quantity: 10, price: 100, net: -1005 }),
    row({ date: "2025-06-02", quantity: 10, price: 105, net: -1055 }),
    row({ date: "2026-03-02", quantity: -5, price: 130, net: 645, action: "Sell" }),
    row({ date: "2026-04-07", type: "Dividends", quantity: 0, price: 0, net: 30 }),
    row({ date: "2026-04-07", type: "Fees and rebates", quantity: 0, price: 0, net: -2 }),
  ];

  it("adds market value to every cash flow", () => {
    const t = totalReturn(rows, h, ctx);
    expect(t.mode).toBe("full");
    expect(t.value).toBeCloseTo(1800);
    // 1800 - 1005 - 1055 + 645 + 30 - 2
    expect(t.total).toBeCloseTo(413);
    expect(t.invested).toBeCloseTo(2060);
    expect(t.dividends).toBeCloseTo(30);
    expect(t.other).toBeCloseTo(-2);
  });

  it("splits trading into what is open and what the sells locked in", () => {
    const t = totalReturn(rows, h, ctx);
    expect(t.unrealised).toBeCloseTo(280);
    // trading = 1800 - 1005 - 1055 + 645 = 385; realised = 385 - 280
    expect(t.realised).toBeCloseTo(105);
    expect(t.unrealised! + t.realised! + t.dividends + t.other).toBeCloseTo(t.total);
  });

  it("gives an annualised money-weighted return over a year and more", () => {
    const t = totalReturn(rows, h, ctx);
    expect(t.mwr_annualised).toBe(true);
    expect(t.since).toBe("2025-01-02");
    const r = xirr([...rows.map((x) => ({ date: x.date, amount: x.net! })),
                    { date: ctx.today, amount: 1800 }])!;
    expect(t.mwr).toBeCloseTo(r * 100, 6);
  });

  it("does not annualise a holding younger than a year", () => {
    const t = totalReturn([row({ date: "2026-08-07", quantity: 15, price: 100, net: -1500 })],
      h, ctx);
    expect(t.mwr_annualised).toBe(false);
    // 1500 → 1800 in under a year is +20% over the period, not annualised.
    expect(t.mwr).toBeCloseTo(20, 6);
  });

  it("counts shares held before the window as bought at its first close", () => {
    const t = totalReturn(rows, { ...h, quantity: 25 },
      { ...ctx, closes: [["2024-01-02", 90], ["2024-01-03", 91]] });
    expect(t.mode).toBe("window");
    expect(t.opening).toEqual({ date: "2024-01-02", price: 90, shares: 10, value: 900 });
    expect(t.value).toBeCloseTo(3000);
    // 3000 - 1387 of recorded flows - 900 opening
    expect(t.total).toBeCloseTo(713);
    expect(t.invested).toBeCloseTo(2960);
    expect(t.mwr).not.toBeNull();
    expect(t.unrealised).toBeNull();
  });

  it("gives no rate when the gap cannot be priced", () => {
    const t = totalReturn(rows, { ...h, quantity: 25 }, ctx);
    expect(t.mode).toBe("partial");
    expect(t.value).toBeCloseTo(1800);
    expect(t.mwr).toBeNull();
  });

  it("does not count a transfer-in as covered", () => {
    const t = totalReturn([...rows,
      row({ type: "Other", quantity: 10, price: 0, net: 0 })], { ...h, quantity: 25 }, ctx);
    expect(t.complete).toBe(false);
  });

  it("adds back a dividend that has gone ex but not been paid", () => {
    const t = totalReturn(rows, h,
      { ...ctx, dividends: [{ ex_date: "2026-10-30", amount: 0.2 }] });
    expect(t.pending).toEqual([{ ex_date: "2026-10-30", per_share: 0.2, shares: 15,
                                 amount: 3 }]);
    expect(t.total).toBeCloseTo(416);
  });

  it("does not add it back once the cash has landed", () => {
    const t = totalReturn([...rows,
      row({ date: "2026-11-04", type: "Dividends", quantity: 0, price: 0, net: 3 })], h,
      { ...ctx, dividends: [{ ex_date: "2026-10-30", amount: 0.2 }] });
    expect(t.pending).toEqual([]);
  });

  it("prices a pending dividend on the shares held going into the ex-date", () => {
    // 5 of the 15 were bought on the ex-date itself: no dividend on those.
    const t = totalReturn([...rows,
      row({ date: "2026-10-30", quantity: 5, price: 118, net: -590 })], { ...h, quantity: 20 },
      { ...ctx, dividends: [{ ex_date: "2026-10-30", amount: 0.2 }] });
    expect(t.pending[0]!.shares).toBe(15);
  });

  it("ignores old ex-dates", () => {
    const t = totalReturn(rows, h,
      { ...ctx, dividends: [{ ex_date: "2026-08-29", amount: 0.2 }] });
    expect(t.pending).toEqual([]);
  });

  it("leaves other-currency rows out", () => {
    const t = totalReturn([...rows, row({ type: "Dividends", quantity: 0, price: 0,
      net: 9, currency: "USD" })], h, ctx);
    expect(t.skipped).toBe(1);
    expect(t.dividends).toBeCloseTo(30);
  });
});
