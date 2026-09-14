import { useEffect, useState } from "react";

/**
 * useState that survives reloads via localStorage. Every read and write is
 * guarded: private windows and some browser settings make the storage
 * accessor itself throw, and a chart preference is never worth a crash.
 */
export function usePersistentState<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw == null ? initial : (JSON.parse(raw) as T);
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* storage unavailable: keep the in-memory value */
    }
  }, [key, value]);
  return [value, setValue] as const;
}
