// Tests of the stress ramp and its range mapping in src/colormap.ts: the ends are blue and red, the
// values in between are continuous, and a clamp corner saturates at the top colour instead of
// stretching the scale.

import { describe, expect, test } from 'bun:test';
import { fillColors, rampColor, rampCssGradient, rampPosition } from './colormap.ts';

describe('rampColor', () => {
  test('low is blue and high is red', () => {
    const [r0, , b0] = rampColor(0);
    const [r1, , b1] = rampColor(1);
    expect(b0).toBeGreaterThan(r0);
    expect(r1).toBeGreaterThan(b1);
  });

  test('values outside 0..1 take the nearest end, and NaN takes the low end', () => {
    expect(rampColor(-1)).toEqual(rampColor(0));
    expect(rampColor(2)).toEqual(rampColor(1));
    expect(rampColor(Number.NaN)).toEqual(rampColor(0));
  });

  test('the ramp has no jumps, so stress bands do not get false edges', () => {
    for (let i = 0; i < 1000; i++) {
      const a = rampColor(i / 1000);
      const b = rampColor((i + 1) / 1000);
      for (let channel = 0; channel < 3; channel++) {
        expect(Math.abs(a[channel] - b[channel])).toBeLessThan(0.01);
      }
    }
  });

  test('each channel stays displayable', () => {
    for (let i = 0; i <= 200; i++) {
      for (const channel of rampColor(i / 200)) {
        expect(channel).toBeGreaterThanOrEqual(0);
        expect(channel).toBeLessThanOrEqual(1);
      }
    }
  });
});

describe('rampPosition', () => {
  test('the peak outside the clamps is the top of the scale', () => {
    expect(rampPosition(250, 10, 250)).toBe(1);
    expect(rampPosition(10, 10, 250)).toBe(0);
    expect(rampPosition(130, 10, 250)).toBeCloseTo(0.5, 12);
  });

  test('a clamp corner above the peak draws the top colour', () => {
    const out = new Float32Array(3);
    fillColors(new Float32Array([900]), 0, 100, out);
    const [r, g, b] = rampColor(1);
    expect(out[0]).toBeCloseTo(r, 5);
    expect(out[1]).toBeCloseTo(g, 5);
    expect(out[2]).toBeCloseTo(b, 5);
  });

  test('a flat field sits at the bottom instead of dividing by zero', () => {
    expect(rampPosition(5, 5, 5)).toBe(0);
    expect(rampPosition(0, 0, 0)).toBe(0);
  });
});

test('the legend gradient runs from the low colour at the bottom to the high colour at the top', () => {
  const css = rampCssGradient();
  expect(css.startsWith('linear-gradient(to top, ')).toBe(true);
  const first = rampColor(0).map(c => Math.round(c * 255));
  const last = rampColor(1).map(c => Math.round(c * 255));
  expect(css).toContain(`rgb(${first[0]} ${first[1]} ${first[2]})`);
  expect(css.endsWith(`rgb(${last[0]} ${last[1]} ${last[2]}))`)).toBe(true);
});
