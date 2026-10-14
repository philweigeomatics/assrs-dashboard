/**
 * 🤖 一组经济数据的合并解读.
 *
 * One call for the whole selection, not one per event. That is the cheaper
 * request and also the better question: a single macro print says very
 * little, and the differences between a batch of them are the substance.
 *
 * The part worth keeping honest is `coherent`. Asked to find a theme in
 * twelve unrelated releases a model will happily invent one, so it is told
 * to say when a selection does not add up — and when it says so, that answer
 * leads rather than hiding under a manufactured narrative.
 */

import type { EcoRead as EcoReadPayload } from "../../lib/types";

export function EcoRead({ d, onClose }: {
  d: EcoReadPayload; onClose: () => void;
}) {
  const r = d.read ?? {};
  const incoherent = r.coherent === false;

  return (
    <div className="rounded-lg border border-violet/40 bg-violet/5 p-3
      flex flex-col gap-2.5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="text-[14px] font-semibold">
          🤖 {r.headline || "解读"}
        </span>
        <span className="label tnum">
          {d.count} 条 · 已公布 {d.released} · 未公布 {d.upcoming}
          {d.span && ` · ${d.span[0]} ~ ${d.span[1]}`}
        </span>
        {d.cached && <span className="label">（缓存）</span>}
        <button onClick={onClose}
          className="ml-auto text-[12px] text-ink-mute hover:text-ink">收起</button>
      </div>

      {incoherent && (
        <p className="text-[12.5px] leading-snug rounded-md bg-brand-ink/10
          text-brand-ink px-2.5 py-1.5">
          ⚠️ 这组事件读不出统一主题 —— {r.coherence_note}
        </p>
      )}

      {r.overall && (
        <p className="text-[13px] leading-relaxed">{r.overall}</p>
      )}

      {(r.themes?.length ?? 0) > 0 && (
        <div className="grid gap-2 sm:grid-cols-2">
          {r.themes!.map((t, i) => (
            <div key={i} className="rounded-md bg-panel p-2 flex flex-col gap-1">
              <span className="text-[12.5px] font-semibold">{t.theme}</span>
              <p className="text-[12px] leading-snug text-ink-dim">{t.read}</p>
              {(t.events?.length ?? 0) > 0 && (
                <span className="label truncate" title={t.events!.join("、")}>
                  {t.events!.join("、")}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {(r.most_important?.length ?? 0) > 0 && (
        <div className="flex flex-col gap-1">
          <span className="label">最值得注意</span>
          {r.most_important!.map((m, i) => (
            <p key={i} className="text-[12.5px] leading-snug">
              <b>{m.event}</b> —— {m.why}
            </p>
          ))}
        </div>
      )}

      {(r.watch_next?.length ?? 0) > 0 && (
        <div className="flex flex-col gap-1">
          <span className="label">接下来看什么</span>
          {r.watch_next!.map((w, i) => (
            <p key={i} className="text-[12.5px] leading-snug text-ink-dim">
              · {w}
            </p>
          ))}
        </div>
      )}

      {r.what_would_change_it && (
        <p className="text-[12px] leading-snug text-ink-mute">
          <b>什么情况会推翻它：</b>{r.what_would_change_it}
        </p>
      )}

      <p className="label">
        宏观与技术面解读，不构成投资建议。差值由程序算出后交给模型，模型负责权衡。
      </p>
    </div>
  );
}
