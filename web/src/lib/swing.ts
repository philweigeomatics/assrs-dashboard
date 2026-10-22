/**
 * The 波动与反弹 scenario: what a market move of `move`% over `days`
 * sessions would roughly do to each stock.
 *
 * Centre: the up-day multiple for a rise, the down-day multiple for a fall —
 * that asymmetry is the whole point of the panel. Band: the stock's own
 * (non-market) daily volatility scaled to the horizon, one standard
 * deviation. Linear, so a guide to who amplifies more, not a forecast.
 */
import type { Swing } from "./types";

export type ScenarioRow = { s: string; mid: number; band: number };

export function scenario(d: Swing, move: number, days: number): ScenarioRow[] {
  const rows: ScenarioRow[] = [{ s: "__market__", mid: move, band: 0 }];
  for (const s of d.symbols) {
    const st = d.stats[s];
    const mult = move >= 0 ? st?.beta_up : st?.beta_down;
    if (mult == null) continue;
    const resid = st?.resid_vol_daily_pct ?? 0;
    rows.push({ s, mid: mult * move, band: resid * Math.sqrt(days) });
  }
  return rows;
}
