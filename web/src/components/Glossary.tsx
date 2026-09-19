/**
 * 指标怎么读 — the explanation panel, and the ⓘ affordance beside a header.
 *
 * Collapsed by default: someone who already knows what an ADF test is should
 * not have to scroll past a wall of text to reach the table. Open once and it
 * stays open, per user, because the reader who needs it needs it every time.
 *
 * The content comes from lib/indicators, which also supplies the tooltips, so
 * the short line on a column and the card in here cannot disagree.
 */

import type { Indicator } from "../lib/indicators";
import { usePersistentState } from "../lib/usePersistentState";

export function Glossary({ title, items, note }: {
  title: string; items: Indicator[]; note?: string;
}) {
  const [open, setOpen] = usePersistentState(`assrs.glossary.${title}`, false);
  return (
    <section className="card p-3 flex flex-col gap-2">
      <button onClick={() => setOpen(!open)}
        className="flex items-baseline gap-2 text-left">
        <span className="text-[13px] font-semibold">📖 {title}</span>
        {!open && <span className="label">每一栏是什么意思、该怎么看</span>}
        <span className="ml-auto text-[12px] text-cyan">{open ? "收起" : "展开"}</span>
      </button>

      {open && (
        <>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {items.map((it) => (
              <div key={it.label}
                className="rounded-lg bg-sunken p-2.5 flex flex-col gap-1">
                <div className="flex items-baseline gap-2">
                  <span className="text-[12.5px] font-semibold">{it.label}</span>
                  {it.pass && (
                    <span className="text-[11px] font-mono tnum text-up">{it.pass}</span>
                  )}
                </div>
                <p className="text-[12px] leading-snug">{it.what}</p>
                <p className="text-[12px] leading-snug text-ink-dim">{it.reads}</p>
                {it.caveat && (
                  <p className="text-[12px] leading-snug text-ink-mute">
                    <span className="text-ink-dim">注意 · </span>{it.caveat}
                  </p>
                )}
              </div>
            ))}
          </div>
          {note && <p className="label leading-snug max-w-[86ch]">{note}</p>}
        </>
      )}
    </section>
  );
}

/**
 * A column header that says it can be explained.
 *
 * A tooltip nobody knows is there is not an explanation. The dotted underline
 * and the mark are the whole point — they are what make the reader hover.
 */
export function Hint({ children, tip, className = "" }: {
  children: React.ReactNode; tip: string; className?: string;
}) {
  return (
    <span title={tip}
      className={`underline decoration-dotted decoration-line-bright underline-offset-2
        cursor-help ${className}`}>
      {children}<span className="text-ink-mute no-underline"> ⓘ</span>
    </span>
  );
}
