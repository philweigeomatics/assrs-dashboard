/**
 * One box for both jobs: pick from history, or search everything.
 *
 * Focus it empty and your recent stocks are listed immediately. Type a code
 * or a name and the same list becomes search results, with stocks you have
 * looked at before marked ⟲ and ranked first within their match tier. The
 * full stock list lives in the browser (see /stocks), so filtering happens
 * per keystroke without a request.
 *
 * Keyboard: ↑/↓ move, Enter picks, Esc closes. A 6-digit code + Enter picks
 * that code even before the list has loaded.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { suggest } from "../lib/search";
import type { HistoryRef, StockRef } from "../lib/types";

export function StockPicker({
  stocks,
  history,
  current,
  onPick,
}: {
  stocks: StockRef[];
  history: HistoryRef[];
  current: StockRef | null;
  onPick: (s: StockRef) => void;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const box = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);

  const items = useMemo(() => suggest(query, stocks, history, 12), [query, stocks, history]);
  useEffect(() => setActive(0), [query]);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  function pick(s: StockRef) {
    onPick(s);
    setQuery("");
    setOpen(false);
    input.current?.blur();
  }

  function onKey(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      setActive((a) => Math.min(a + 1, Math.max(items.length - 1, 0)));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const chosen = items[active];
      if (chosen) pick(chosen);
      else if (/^\d{6}$/.test(query.trim())) pick({ t: query.trim(), n: query.trim() });
    } else if (e.key === "Escape") {
      setOpen(false);
      input.current?.blur();
    }
  }

  const heading = query.trim() ? `匹配 ${items.length} 只` : "最近搜索";

  return (
    <div ref={box} className="relative w-full max-w-md">
      <div className="flex items-center gap-2 bg-sunken rounded-[10px] px-3 h-10 focus-within:ring-2 focus-within:ring-cyan/40">
        <span aria-hidden className="text-ink-mute">⌕</span>
        <input
          ref={input}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKey}
          placeholder={current ? `${current.n} ${current.t} · 输入代码或名称切换` : "输入代码或名称，或从最近搜索中选择"}
          className="flex-1 bg-transparent outline-none text-[15px] placeholder:text-ink-mute"
          role="combobox"
          aria-expanded={open}
          aria-controls="stock-picker-list"
          aria-autocomplete="list"
        />
      </div>

      {open && (
        <div
          id="stock-picker-list"
          role="listbox"
          className="card absolute z-30 mt-1.5 w-full py-1.5 max-h-[420px] overflow-auto"
        >
          <div className="px-3 pb-1 label">{heading}</div>
          {items.length === 0 && (
            <div className="px-3 py-2 text-ink-mute text-[13px]">
              {query.trim() ? "没有匹配的股票" : "还没有搜索记录——输入代码或名称开始"}
            </div>
          )}
          {items.map((s, i) => (
            <button
              key={s.t}
              type="button"
              role="option"
              aria-selected={i === active}
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => pick(s)}
              className={`w-full flex items-center gap-3 px-3 py-1.5 text-left ${
                i === active ? "bg-elevated" : ""
              }`}
            >
              <span className="font-mono tnum text-[13px] text-ink-dim w-[58px] shrink-0">{s.t}</span>
              <span className="flex-1 truncate text-[14px]">{s.n}</span>
              {s.fromHistory && <span className="text-[12px] text-ink-mute" title="搜索过">⟲</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
