// Tests of the exaggeration and displacement scaling in src/animation.ts: the drawn motion is the
// same at any load, the label's factor is exactly the one applied, and the part returns to rest.

import { describe, expect, test } from 'bun:test';
import { ANIMATION_PERIOD_MS, DRAWN_SHARE_OF_LENGTH, animationPhase, displacePositions, exaggerationScale } from './animation.ts';

describe('exaggerationScale', () => {
  test('draws the largest displacement at about the target share of the length', () => {
    const length = 155;
    for (const maxDisplacement of [1e-5, 0.003, 0.2, 7, 40]) {
      const drawn = exaggerationScale(maxDisplacement, length) * maxDisplacement;
      const target = DRAWN_SHARE_OF_LENGTH * length;
      // Two-significant-figure rounding moves the factor by at most 5%.
      expect(Math.abs(drawn - target) / target).toBeLessThan(0.0501);
    }
  });

  test('motion is visible at a tiny load and a large one', () => {
    expect(exaggerationScale(1e-6, 100) * 1e-6).toBeGreaterThan(0.1 * 100);
    expect(exaggerationScale(80, 100) * 80).toBeGreaterThan(0.1 * 100);
  });

  test('a big deflection is scaled down rather than beyond the view', () => {
    expect(exaggerationScale(1000, 100)).toBeCloseTo(0.012, 10);
  });

  test('no displacement yet gives no exaggeration, not infinity', () => {
    expect(exaggerationScale(0, 100)).toBe(0);
    expect(exaggerationScale(Number.NaN, 100)).toBe(0);
  });

  test('the factor is a round number the label can state exactly', () => {
    const scale = exaggerationScale(0.0123, 155);
    expect(Number(scale.toPrecision(2))).toBe(scale);
  });
});

describe('animationPhase', () => {
  test('starts at rest, peaks at half the period, and returns', () => {
    expect(animationPhase(0)).toBeCloseTo(0, 12);
    expect(animationPhase(ANIMATION_PERIOD_MS / 2)).toBeCloseTo(1, 12);
    expect(animationPhase(ANIMATION_PERIOD_MS)).toBeCloseTo(0, 12);
  });

  test('stays within rest and full bend', () => {
    for (let t = 0; t < 5000; t += 37) {
      const phase = animationPhase(t);
      expect(phase).toBeGreaterThanOrEqual(0);
      expect(phase).toBeLessThanOrEqual(1);
    }
  });
});

describe('displacePositions', () => {
  const base = new Float32Array([1, 2, 3, 4, 5, 6]);
  const displacement = new Float32Array([0.5, 0, -1, 0, 2, 0]);

  test('scales the displacement by the factor and the phase', () => {
    const out = new Float32Array(6);
    displacePositions(base, displacement, 10, 0.5, out);
    expect(Array.from(out)).toEqual([3.5, 2, -2, 4, 15, 6]);
  });

  test('at phase zero the part is exactly its undeformed shape', () => {
    const out = new Float32Array(6);
    displacePositions(base, displacement, 1000, 0, out);
    expect(Array.from(out)).toEqual(Array.from(base));
  });
});
