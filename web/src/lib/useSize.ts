/**
 * The pixel size of an element, kept current through resizes.
 *
 * The treemap and the rotation map are laid out in the browser (see
 * treemap.ts), which means they need real pixels rather than a CSS percentage.
 * A ResizeObserver rather than a window listener: these panels also change
 * width when a sibling collapses, which no window event reports.
 */

import { useEffect, useRef, useState } from "react";

export function useSize<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const box = entry?.contentRect;
      if (box) setSize({ w: Math.round(box.width), h: Math.round(box.height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return [ref, size] as const;
}
