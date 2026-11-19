/**
 * The app's language, in one place.
 *
 * Every API request carries it (`X-Lang`, see api.ts), and the server writes
 * AI prose in it — i18n.py holds the server's half. Today that is all it
 * changes: the interface itself is Chinese. When it is translated, the
 * strings will key off this same value, so nothing else needs to learn that
 * there is more than one language.
 *
 * Remembered per browser. Held outside React so `api.ts`, which is not a
 * component, reads the same value the toggle shows.
 */

import { useSyncExternalStore } from "react";

export type Lang = "zh" | "en";

export const LANGS: { id: Lang; label: string; short: string }[] = [
  { id: "zh", label: "中文", short: "中" },
  { id: "en", label: "English", short: "EN" },
];

export const DEFAULT_LANG: Lang = "zh";
const KEY = "assrs.lang";

const isLang = (v: unknown): v is Lang => LANGS.some((l) => l.id === v);

let current: Lang = (() => {
  try {
    const v = localStorage.getItem(KEY);
    return isLang(v) ? v : DEFAULT_LANG;
  } catch {
    return DEFAULT_LANG;
  }
})();
const listeners = new Set<() => void>();

export function getLang(): Lang {
  return current;
}

export function setLang(v: Lang) {
  if (!isLang(v) || v === current) return;
  current = v;
  try { localStorage.setItem(KEY, v); } catch { /* this session only */ }
  listeners.forEach((f) => f());
}

export function useLang(): Lang {
  return useSyncExternalStore(
    (f) => { listeners.add(f); return () => { listeners.delete(f); }; },
    getLang);
}
