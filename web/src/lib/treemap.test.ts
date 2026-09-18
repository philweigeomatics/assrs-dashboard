import { describe, expect, it } from "vitest";
import { heatColor, heatInk, squarify, type Rect } from "./treemap";

const BOX: Rect = { x: 0, y: 0, w: 800, h: 500 };
const sized = (values: number[]) => values.map((value, i) => ({ name: `s${i}`, value }));

function overlaps(a: Rect, b: Rect): boolean {
  const e = 1e-6;
  return a.x < b.x + b.w - e && b.x < a.x + a.w - e
    && a.y < b.y + b.h - e && b.y < a.y + a.h - e;
}

describe("squarify", () => {
  it("gives every box an area proportional to its value", () => {
    const values = [40, 25, 15, 10, 6, 4];
    const out = squarify(sized(values), BOX);
    const total = values.reduce((s, v) => s + v, 0);

    for (const box of out) {
      const share = box.value / total;
      expect(box.w * box.h).toBeCloseTo(BOX.w * BOX.h * share, 4);
    }
  });

  it("fills the box exactly, with nothing overlapping", () => {
    const out = squarify(sized([50, 30, 20, 12, 8, 5, 3, 2, 1]), BOX);

    const area = out.reduce((s, b) => s + b.w * b.h, 0);
    expect(area).toBeCloseTo(BOX.w * BOX.h, 3);

    for (const b of out) {
      expect(b.x).toBeGreaterThanOrEqual(-1e-6);
      expect(b.y).toBeGreaterThanOrEqual(-1e-6);
      expect(b.x + b.w).toBeLessThanOrEqual(BOX.x + BOX.w + 1e-6);
      expect(b.y + b.h).toBeLessThanOrEqual(BOX.y + BOX.h + 1e-6);
    }
    for (let i = 0; i < out.length; i++) {
      for (let j = i + 1; j < out.length; j++) {
        expect(overlaps(out[i]!, out[j]!), `${out[i]!.name} overlaps ${out[j]!.name}`)
          .toBe(false);
      }
    }
  });

  it("keeps boxes close to square, in a panel of any shape", () => {
    // The bound is tight on purpose. Packing a row against the LONGER side of
    // the free space instead of the shorter one still produces a valid,
    // non-overlapping, area-correct treemap — it just produces slivers, up to
    // 50:1 in a letterbox panel. A loose bound would pass that happily, and
    // slivers are the entire failure mode this algorithm exists to avoid.
    const shapes: Rect[] = [
      { x: 0, y: 0, w: 800, h: 500 },
      { x: 0, y: 0, w: 1400, h: 300 },   // letterbox
      { x: 0, y: 0, w: 300, h: 900 },    // column
    ];
    const inputs = [
      [500, 250, 120, 60, 30, 15, 8, 4, 2, 1],
      Array.from({ length: 24 }, () => 10),
      [1000, 400, 300, 200, 150, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 2, 1],
    ];
    for (const shape of shapes) {
      for (const values of inputs) {
        const out = squarify(sized(values), shape);
        const worst = Math.max(...out.map((b) => Math.max(b.w / b.h, b.h / b.w)));
        expect(worst, `${shape.w}×${shape.h} with ${values.length} boxes`)
          .toBeLessThan(3);
      }
    }
  });

  it("does not depend on the order the items arrive in", () => {
    const values = [12, 40, 3, 25, 8];
    const a = squarify(sized(values), BOX);
    const b = squarify([...sized(values)].reverse(), BOX);
    const key = (r: (typeof a)[number]) => `${r.name}:${r.x.toFixed(6)}:${r.y.toFixed(6)}`;
    expect(a.map(key).sort()).toEqual(b.map(key).sort());
  });

  it("drops values that cannot be drawn instead of placing empty boxes", () => {
    const out = squarify(
      [{ name: "a", value: 10 }, { name: "zero", value: 0 },
       { name: "neg", value: -5 }, { name: "nan", value: NaN }],
      BOX);
    expect(out.map((b) => b.name)).toEqual(["a"]);
    expect(out[0]!.w * out[0]!.h).toBeCloseTo(BOX.w * BOX.h, 6);
  });

  it("returns nothing for an empty list or a box with no room", () => {
    expect(squarify([], BOX)).toEqual([]);
    expect(squarify(sized([1, 2]), { x: 0, y: 0, w: 0, h: 100 })).toEqual([]);
    expect(squarify(sized([1, 2]), { x: 0, y: 0, w: 100, h: -4 })).toEqual([]);
  });

  it("lays out inside an offset box, not at the origin", () => {
    const out = squarify(sized([3, 1]), { x: 40, y: 12, w: 200, h: 100 });
    expect(Math.min(...out.map((b) => b.x))).toBeCloseTo(40, 6);
    expect(Math.min(...out.map((b) => b.y))).toBeCloseTo(12, 6);
  });

  it("carries the item's own fields through to the rectangle", () => {
    const out = squarify([{ name: "a", value: 1, ticker: "600519" }], BOX);
    expect(out[0]!.ticker).toBe("600519");
  });
});

describe("heatColor", () => {
  it("paints a rise red and a fall green — the A-share convention", () => {
    const red = heatColor(5, 5).match(/\d+/g)!.map(Number);
    const green = heatColor(-5, 5).match(/\d+/g)!.map(Number);
    expect(red[0]).toBeGreaterThan(red[1]!);
    expect(green[1]).toBeGreaterThan(green[0]!);
  });

  it("gets stronger with the size of the move", () => {
    const lum = (c: string) => c.match(/\d+/g)!.map(Number).reduce((s, v) => s + v, 0);
    expect(lum(heatColor(8, 10))).toBeLessThan(lum(heatColor(2, 10)));
  });

  it("leaves a tiny move visibly tinted, not a shade off white", () => {
    // A box that moved at all has to be distinguishable from one that did
    // not, on a screen, at 14px. "Not the same string" is not that test.
    const flat = heatColor(0, 10).match(/\d+/g)!.map(Number);
    const tiny = heatColor(0.1, 10).match(/\d+/g)!.map(Number);
    const delta = Math.max(...tiny.map((v, i) => Math.abs(v - flat[i]!)));
    expect(delta).toBeGreaterThan(20);
  });

  it("clamps at the scale instead of overflowing past it", () => {
    expect(heatColor(50, 10)).toBe(heatColor(10, 10));
  });

  it("picks ink that can be read on the box it sits on", () => {
    expect(heatInk(heatColor(0, 10))).toBe("#111");
    expect(heatInk(heatColor(10, 10))).toBe("#fff");
  });
});
