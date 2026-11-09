import { useEffect, useRef, useState } from "react";

/**
 * A number box you can type into like a text box.
 *
 * What the box shows is its own text, not the number: half-typed states —
 * empty, "-", "-0.", "1." — are not numbers yet, and a box bound straight to
 * a number turned each of them back into the last valid value, so a minus
 * sign could not be typed and a field could not be emptied. The number is
 * passed up only once the text is one; a box left invalid goes back to the
 * last number when it loses focus. ↑ / ↓ still step it.
 */
export function NumberInput({
  value, set, after, step = 0.01, decimals = 2, signed = false, max, label, className,
}: {
  value: number;
  set: (v: number) => void;
  after?: () => void;
  step?: number;
  decimals?: number;
  /** Accepts a minus sign. Without it the box cannot go below zero. */
  signed?: boolean;
  /** A keystroke that would go past this is ignored. */
  max?: number;
  label?: string;
  className?: string;
}) {
  const show = (v: number) => (Number.isFinite(v) ? String(Number(v.toFixed(decimals))) : "");
  const [text, setText] = useState(() => show(value));
  // The last number this box sent up. A different value arriving means
  // something else changed it (Δ% re-seeds 开/高/低, a slider moved), so the
  // box shows it; its own echo leaves the text as typed ("1." stays "1.").
  const sent = useRef(value);
  useEffect(() => {
    if (value !== sent.current) {
      sent.current = value;
      setText(show(value));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  const commit = (v: number) => {
    sent.current = v;
    set(v);
    after?.();
  };
  const pattern = signed ? /^-?\d*\.?\d*$/ : /^\d*\.?\d*$/;

  return (
    <input
      type="text"
      inputMode={signed ? "text" : "decimal"}
      autoComplete="off"
      aria-label={label}
      value={text}
      onChange={(e) => {
        const t = e.target.value.trim().replace(/[−–]/g, "-").replace(/,/g, "");
        if (!pattern.test(t)) return;               // ignore letters, a second dot
        const v = parseFloat(t);
        if (max != null && v > max) return;
        setText(t);
        if (Number.isFinite(v)) commit(v);
      }}
      onBlur={() => {
        if (!Number.isFinite(parseFloat(text))) setText(show(value));
      }}
      onKeyDown={(e) => {
        if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
        e.preventDefault();
        const base = Number.isFinite(parseFloat(text)) ? parseFloat(text) : value;
        let v = Number((base + (e.key === "ArrowUp" ? step : -step)).toFixed(decimals));
        if (!signed) v = Math.max(0, v);
        if (max != null) v = Math.min(max, v);
        setText(show(v));
        commit(v);
      }}
      onFocus={(e) => e.currentTarget.select()}
      className={className}
    />
  );
}
