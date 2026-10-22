/**
 * A chain being built: ordered legs, upstream first, each a set of A-share
 * codes. Pure functions so the drag-and-drop handlers stay trivial and the
 * rules — a stock lives in one leg, at most 12 per leg, at most 6 legs, names
 * unique — are tested in one place and match the server's.
 */

export const MAX_LEGS = 6;
export const MAX_PER_LEG = 12;

/**
 * Leg colours, upstream to downstream. Validated for colour-blind separation
 * in this order; one adjacent pair sits in the 6–8 ΔE band, which is why
 * every line also carries its name at its end.
 */
export const LEG_COLOURS = ["#0062cc", "#b45309", "#0d9488", "#be185d", "#7c3aed", "#65a30d"];

export type Leg = { id: string; name: string; tickers: string[] };
export type Chain = { name: string; legs: Leg[] };

let seq = 0;
export const newLegId = () => `leg${Date.now().toString(36)}${(seq++).toString(36)}`;

export function emptyChain(name = "新产业链"): Chain {
  return { name, legs: [{ id: newLegId(), name: "", tickers: [] }, { id: newLegId(), name: "", tickers: [] }] };
}

/**
 * Put `tickers` into one leg. A stock already in another leg moves rather
 * than appearing twice — one stock in two legs would make them correlate by
 * construction. An unnamed leg takes the 板块 name it was filled from.
 */
export function addToLeg(chain: Chain, legId: string, tickers: string[], sector?: string): Chain {
  const target = chain.legs.find((l) => l.id === legId);
  if (!target) return chain;
  const room = MAX_PER_LEG - target.tickers.filter((t) => !tickers.includes(t)).length;
  const incoming = [...new Set(tickers)].slice(0, Math.max(0, room));
  const legs = chain.legs.map((l) => {
    if (l.id === legId) {
      const merged = [...l.tickers.filter((t) => !incoming.includes(t)), ...incoming];
      return { ...l, tickers: merged, name: l.name || uniqueName(chain, sector ?? "", l.id) };
    }
    return { ...l, tickers: l.tickers.filter((t) => !incoming.includes(t)) };
  });
  return { ...chain, legs };
}

function uniqueName(chain: Chain, wanted: string, selfId: string): string {
  if (!wanted) return "";
  const taken = new Set(chain.legs.filter((l) => l.id !== selfId).map((l) => l.name));
  if (!taken.has(wanted)) return wanted;
  let i = 2;
  while (taken.has(`${wanted}${i}`)) i++;
  return `${wanted}${i}`;
}

export function removeTicker(chain: Chain, legId: string, t: string): Chain {
  return { ...chain, legs: chain.legs.map((l) => l.id === legId ? { ...l, tickers: l.tickers.filter((x) => x !== t) } : l) };
}

export function renameLeg(chain: Chain, legId: string, name: string): Chain {
  return { ...chain, legs: chain.legs.map((l) => l.id === legId ? { ...l, name: name.slice(0, 20) } : l) };
}

export function addLeg(chain: Chain): Chain {
  if (chain.legs.length >= MAX_LEGS) return chain;
  return { ...chain, legs: [...chain.legs, { id: newLegId(), name: "", tickers: [] }] };
}

export function removeLeg(chain: Chain, legId: string): Chain {
  return { ...chain, legs: chain.legs.filter((l) => l.id !== legId) };
}

/** Swap a leg with its neighbour: −1 = further upstream, +1 = further downstream. */
export function moveLeg(chain: Chain, legId: string, dir: -1 | 1): Chain {
  const i = chain.legs.findIndex((l) => l.id === legId);
  const j = i + dir;
  if (i < 0 || j < 0 || j >= chain.legs.length) return chain;
  const legs = [...chain.legs];
  [legs[i], legs[j]] = [legs[j]!, legs[i]!];
  return { ...chain, legs };
}

/** What stops this chain being analysed, or null if nothing does. */
export function problem(chain: Chain): string | null {
  const filled = chain.legs.filter((l) => l.tickers.length);
  if (filled.length < 2) return "至少需要两个有股票的环节";
  if (filled.length > MAX_LEGS) return `最多 ${MAX_LEGS} 个环节`;
  const unnamed = filled.filter((l) => !l.name.trim());
  if (unnamed.length) return "每个环节都需要一个名字";
  const names = filled.map((l) => l.name.trim());
  if (new Set(names).size !== names.length) return "环节名称不能重复";
  return null;
}

/** The request body: filled legs only, upstream first. */
export function toRequest(chain: Chain) {
  return chain.legs.filter((l) => l.tickers.length).map((l) => ({ name: l.name.trim(), tickers: l.tickers }));
}
