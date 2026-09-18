/**
 * Squarified treemap layout (Bruls, Huizing & van Wijk, 2000).
 *
 * The naive layout — slice the box, then slice the slice — is trivial to write
 * and useless to read: a small sector becomes a 400×3 sliver you cannot click,
 * let alone label. Squarified packs each row only while adding the next box
 * keeps the row's worst aspect ratio from getting worse, then starts a new row.
 * The result is boxes near square, which is what makes "this one is twice that
 * one" legible by eye.
 *
 * Done in the browser rather than on the server because rectangles depend on
 * the pixel size of the container: the server would have to know the viewport,
 * and every resize would cost a request. The API sends the tree; this turns it
 * into rectangles at whatever size the panel happens to be.
 */

export type Rect = { x: number; y: number; w: number; h: number };
export type Sized = { value: number };
export type Placed<T> = T & Rect;

type Cell<T> = { item: T; area: number };

/**
 * The worst (largest) aspect ratio in `row` if it is laid along a side of
 * length `side`. Lower is squarer; 1 would be perfect squares.
 */
function worst<T>(row: Cell<T>[], side: number): number {
  if (row.length === 0) return Infinity;
  let sum = 0;
  let max = -Infinity;
  let min = Infinity;
  for (const c of row) {
    sum += c.area;
    if (c.area > max) max = c.area;
    if (c.area < min) min = c.area;
  }
  if (sum <= 0 || side <= 0 || min <= 0) return Infinity;
  const s2 = sum * sum;
  const w2 = side * side;
  return Math.max((w2 * max) / s2, s2 / (w2 * min));
}

function place<T>(row: Cell<T>[], free: Rect, out: Placed<T>[]): Rect {
  const sum = row.reduce((s, c) => s + c.area, 0);
  if (sum <= 0) return free;

  if (free.w >= free.h) {
    // Row runs down the left edge as a vertical strip.
    const w = sum / free.h;
    let y = free.y;
    for (const c of row) {
      const h = c.area / w;
      out.push({ ...c.item, x: free.x, y, w, h });
      y += h;
    }
    return { x: free.x + w, y: free.y, w: free.w - w, h: free.h };
  }

  // Row runs across the top edge as a horizontal band.
  const h = sum / free.w;
  let x = free.x;
  for (const c of row) {
    const w = c.area / h;
    out.push({ ...c.item, x, y: free.y, w, h });
    x += w;
  }
  return { x: free.x, y: free.y + h, w: free.w, h: free.h - h };
}

/**
 * Lay `items` out inside `box`, each rectangle's area proportional to `value`.
 *
 * Items with a non-positive or non-finite value are dropped rather than given
 * a zero-area rectangle: a zero-width box is invisible but still catches
 * clicks and still renders a label at the wrong place.
 *
 * Input order is irrelevant — the algorithm sorts by area, as it must, so two
 * callers passing the same items in different orders get the same picture.
 */
export function squarify<T extends Sized>(items: readonly T[], box: Rect): Placed<T>[] {
  if (!(box.w > 0) || !(box.h > 0)) return [];

  const usable = items.filter((i) => Number.isFinite(i.value) && i.value > 0);
  const total = usable.reduce((s, i) => s + i.value, 0);
  if (total <= 0) return [];

  const scale = (box.w * box.h) / total;
  const queue: Cell<T>[] = usable
    .map((item) => ({ item, area: item.value * scale }))
    .sort((a, b) => b.area - a.area);

  const out: Placed<T>[] = [];
  let free: Rect = { ...box };
  let row: Cell<T>[] = [];

  for (const cell of queue) {
    const side = Math.min(free.w, free.h);
    // Adding this box either squares the row up or starts making it worse.
    // The moment it makes it worse, the row is finished.
    if (row.length > 0 && worst([...row, cell], side) > worst(row, side)) {
      free = place(row, free, out);
      row = [];
    }
    row.push(cell);
  }
  if (row.length > 0) place(row, free, out);

  return out;
}

/**
 * Colour for a percentage move, on the A-share convention: red up, green down.
 *
 * Interpolated from the panel ground rather than between red and green, so a
 * flat box reads as flat instead of as a muddy midpoint, and `scale` (the
 * largest absolute move on the page) keeps the strongest box saturated on a
 * quiet day without blowing out on a violent one.
 */
export function heatColor(pct: number, scale: number): string {
  const t = Math.min(Math.abs(pct) / Math.max(scale, 0.5), 1);
  const [r, g, b] = pct >= 0 ? [215, 0, 21] : [31, 122, 53];
  // Ease so small moves stay visible instead of washing out to white.
  const k = 0.12 + 0.88 * Math.sqrt(t);
  const mix = (c: number) => Math.round(248 + (c - 248) * k);
  return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
}

/** Black or white text, whichever survives on `rgb(...)`. */
export function heatInk(background: string): string {
  const m = background.match(/\d+/g);
  if (!m || m.length < 3) return "#000";
  const [r, g, b] = m.map(Number) as [number, number, number];
  // Rec. 601 luma — good enough to pick between two inks.
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? "#111" : "#fff";
}
