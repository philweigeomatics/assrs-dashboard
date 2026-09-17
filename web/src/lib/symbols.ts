/**
 * Which market a symbol belongs to, on the client.
 *
 * Mirrors markets.split() on the server: a bare six-digit code is an A-share,
 * because every ticker already stored is bare and unprefixed; everything else
 * carries its market. Kept deliberately small — the server remains the
 * authority, and this exists only so the UI can scope a search or label a
 * chip without a round trip.
 */

import type { MarketCode } from "./types";

const PREFIXED = /^(CN|US|CA):(.+)$/i;

/**
 * An unprefixed symbol is an A-share — that is the whole convention, and the
 * server refuses anything unprefixed that is not six digits, so there is no
 * third case to fall through to here.
 */
export function splitMarket(symbol: string): MarketCode {
  const m = PREFIXED.exec(symbol.trim());
  return m ? (m[1]!.toUpperCase() as MarketCode) : "CN";
}

/** The part after the prefix — what the exchange itself calls it. */
export function bareCode(symbol: string): string {
  const m = PREFIXED.exec(symbol.trim());
  return m ? m[2]! : symbol.trim();
}
