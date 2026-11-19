/**
 * How the AI company profile reads: Porter's five forces as the diagram they
 * are always drawn as, PESTEL as six labelled cards, the overview as a
 * headline and a paragraph.
 *
 * Each view checks the shape it was given and returns null when it is not
 * the expected one, so the caller can fall back to a plain dump — a model
 * that renames a key should cost the layout, never the content.
 */

const FORCES = {
  rivalry: { zh: "行业内竞争", en: "Competitive rivalry" },
  new_entrants: { zh: "潜在进入者威胁", en: "Threat of new entrants" },
  substitutes: { zh: "替代品威胁", en: "Threat of substitutes" },
  suppliers: { zh: "供应商议价能力", en: "Supplier power" },
  buyers: { zh: "买方议价能力", en: "Buyer power" },
} as const;
type ForceKey = keyof typeof FORCES;
type Force = { score: number; summary: string };

const LEVEL = ["", "很弱", "较弱", "中等", "较强", "很强"];

/**
 * 1–2 favourable, 3 middling, 4–5 a threat. Not red and green: here those
 * mean a price going up or down.
 */
const tone = (s: number) =>
  s >= 4 ? { bar: "var(--color-brand-ink)", chip: "bg-brand-ink/10 text-brand-ink" }
    : s <= 2 ? { bar: "var(--color-cyan)", chip: "bg-cyan/10 text-ink" }
      : { bar: "var(--color-ink-mute)", chip: "bg-sunken text-ink-dim" };

function asForces(v: unknown): Record<ForceKey, Force> | null {
  if (!v || typeof v !== "object") return null;
  const o = v as Record<string, unknown>;
  const out = {} as Record<ForceKey, Force>;
  for (const k of Object.keys(FORCES) as ForceKey[]) {
    const f = o[k] as { score?: unknown; summary?: unknown } | undefined;
    const score = Number(f?.score);
    if (!f || !Number.isFinite(score) || typeof f.summary !== "string") return null;
    out[k] = { score: Math.min(5, Math.max(1, Math.round(score))), summary: f.summary };
  }
  return out;
}

function Pips({ score }: { score: number }) {
  const t = tone(score);
  return (
    <span className="inline-flex items-center gap-[3px]" aria-label={`强度 ${score} / 5`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <span key={i} className="w-3 h-1.5 rounded-full"
          style={{ background: i <= score ? t.bar : "var(--color-line)" }} />
      ))}
    </span>
  );
}

function ForceCard({ k, f, centre = false }: { k: ForceKey; f: Force; centre?: boolean }) {
  const t = tone(f.score);
  return (
    <div className={`rounded-xl p-2.5 flex flex-col gap-1.5 h-full ${
      centre ? "bg-elevated ring-2 ring-line-bright" : "bg-sunken"}`}>
      <div className="flex items-baseline gap-2 flex-wrap">
        <span className={`font-semibold ${centre ? "text-[14px]" : "text-[13px]"}`}>{FORCES[k].zh}</span>
        <span className="text-[10.5px] text-ink-mute">{FORCES[k].en}</span>
      </div>
      <div className="flex items-center gap-2">
        <Pips score={f.score} />
        <span className={`text-[11.5px] px-1.5 rounded ${t.chip}`}>{LEVEL[f.score]} · {f.score}/5</span>
      </div>
      <p className="text-[12.5px] leading-relaxed text-ink-dim">{f.summary}</p>
    </div>
  );
}

const Arrow = ({ dir }: { dir: "down" | "up" | "right" | "left" }) => (
  <span aria-hidden className="text-ink-mute text-[18px] leading-none select-none self-center justify-self-center">
    {{ down: "↓", up: "↑", right: "→", left: "←" }[dir]}
  </span>
);

/** The five forces, rivalry in the middle and the four pressures on it around it. */
export function PortersView({ value }: { value: unknown }) {
  const f = asForces(value);
  if (!f) return null;
  const mean = (Object.values(f).reduce((a, x) => a + x.score, 0) / 5);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-[12.5px]">
        <span className="text-ink-mute">整体竞争压力</span>
        <Pips score={Math.round(mean)} />
        <b className="tnum">{mean.toFixed(1)} / 5</b>
        <span className="label">1 = 对公司有利，5 = 威胁很大</span>
      </div>

      {/* Phones: one column, rivalry first. */}
      <div className="flex flex-col gap-2 md:hidden">
        {(["rivalry", "new_entrants", "suppliers", "buyers", "substitutes"] as ForceKey[]).map((k) => (
          <ForceCard key={k} k={k} f={f[k]} centre={k === "rivalry"} />
        ))}
      </div>

      {/* Wider: the classic diagram — entrants above, substitutes below,
          suppliers and buyers either side, all pressing on the centre. */}
      <div className="hidden md:grid gap-1.5 items-stretch"
        style={{ gridTemplateColumns: "1fr 20px 1.15fr 20px 1fr",
                 gridTemplateRows: "auto 20px auto 20px auto" }}>
        <div style={{ gridColumn: 3, gridRow: 1 }}><ForceCard k="new_entrants" f={f.new_entrants} /></div>
        <div style={{ gridColumn: 3, gridRow: 2 }} className="grid"><Arrow dir="down" /></div>
        <div style={{ gridColumn: 1, gridRow: 3 }}><ForceCard k="suppliers" f={f.suppliers} /></div>
        <div style={{ gridColumn: 2, gridRow: 3 }} className="grid"><Arrow dir="right" /></div>
        <div style={{ gridColumn: 3, gridRow: 3 }}><ForceCard k="rivalry" f={f.rivalry} centre /></div>
        <div style={{ gridColumn: 4, gridRow: 3 }} className="grid"><Arrow dir="left" /></div>
        <div style={{ gridColumn: 5, gridRow: 3 }}><ForceCard k="buyers" f={f.buyers} /></div>
        <div style={{ gridColumn: 3, gridRow: 4 }} className="grid"><Arrow dir="up" /></div>
        <div style={{ gridColumn: 3, gridRow: 5 }}><ForceCard k="substitutes" f={f.substitutes} /></div>
      </div>
    </div>
  );
}

const PESTEL = [
  { k: "political", letter: "P", zh: "政治", en: "Political", color: "#6366f1" },
  { k: "economic", letter: "E", zh: "经济", en: "Economic", color: "#0ea5e9" },
  { k: "social", letter: "S", zh: "社会", en: "Social", color: "#a855f7" },
  { k: "technological", letter: "T", zh: "技术", en: "Technological", color: "#f59e0b" },
  { k: "environmental", letter: "E", zh: "环境", en: "Environmental", color: "#14b8a6" },
  { k: "legal", letter: "L", zh: "法律", en: "Legal", color: "#64748b" },
] as const;

/** Six cards, each dimension's letter, name and its points. */
export function PestelView({ value }: { value: unknown }) {
  if (!value || typeof value !== "object") return null;
  const o = value as Record<string, unknown>;
  const lists = PESTEL.map((d) => {
    const v = o[d.k];
    const items = Array.isArray(v) ? v.map(String) : typeof v === "string" ? [v] : null;
    return { ...d, items };
  });
  if (lists.some((l) => l.items == null)) return null;
  return (
    <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
      {lists.map((d) => (
        <div key={d.k} className="rounded-xl bg-sunken p-2.5 flex flex-col gap-1.5"
          style={{ borderTop: `3px solid ${d.color}` }}>
          <div className="flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg grid place-items-center text-white text-[14px] font-bold shrink-0"
              style={{ background: d.color }}>{d.letter}</span>
            <span className="text-[13.5px] font-semibold">{d.zh}</span>
            <span className="text-[10.5px] text-ink-mute">{d.en}</span>
          </div>
          <ul className="flex flex-col gap-1">
            {d.items!.map((t, i) => (
              <li key={i} className="flex gap-2 text-[12.5px] leading-relaxed text-ink-dim">
                <span className="mt-[7px] w-1.5 h-1.5 rounded-full shrink-0" style={{ background: d.color }} />
                <span>{t}</span>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

/** A headline and a paragraph. */
export function OverviewView({ value }: { value: unknown }) {
  if (!value || typeof value !== "object") return null;
  const o = value as { tagline?: unknown; summary?: unknown };
  if (typeof o.summary !== "string") return null;
  return (
    <div className="rounded-xl bg-sunken p-3 flex flex-col gap-1.5">
      {typeof o.tagline === "string" && o.tagline && (
        <p className="text-[14.5px] font-semibold leading-snug">{o.tagline}</p>
      )}
      <p className="text-[13px] leading-relaxed text-ink-dim">{o.summary}</p>
    </div>
  );
}
