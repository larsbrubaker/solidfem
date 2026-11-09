// Tests of the exaggeration and displacement scaling in src/animation.ts: the drawn motion is the
// same at any load, the label's factor is exactly the one applied, the part returns to rest, and a
// rotation stays a rotation however far it is exaggerated.

import { describe, expect, test } from 'bun:test';
import {
  ANIMATION_PERIOD_MS, DRAWN_SHARE_OF_LENGTH, MAX_DRAWN_ROTATION, animationPhase, displacePositions, exaggerationScale,
} from './animation.ts';

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

describe('exaggerationScale with a rotation cap', () => {
  const length = 300;
  // Far too small a rotation to reach the cap, so the share-of-length rule decides.
  const negligible = 1e-9;

  test('a rotation too small to matter leaves the share-of-length rule untouched', () => {
    for (const maxDisplacement of [1e-5, 0.003, 0.2, 7, 40]) {
      expect(exaggerationScale(maxDisplacement, length, negligible)).toBe(exaggerationScale(maxDisplacement, length));
    }
  });

  test('never draws a rotation past the cap', () => {
    for (let rotation = 1e-7; rotation < 1e3; rotation *= 1.37) {
      for (const maxDisplacement of [1e-4, 0.5, 40]) {
        const scale = exaggerationScale(maxDisplacement, length, rotation);
        expect(scale * rotation).toBeLessThanOrEqual(MAX_DRAWN_ROTATION);
      }
    }
  });

  test('when rotation limits it, it draws at least 90% of the cap', () => {
    // Rounding down to two significant figures loses at most one unit in eleven.
    for (let rotation = 1e-4; rotation < 10; rotation *= 1.73) {
      const scale = exaggerationScale(1e-6, length, rotation);
      expect(scale * rotation).toBeGreaterThan(0.9 * MAX_DRAWN_ROTATION);
    }
  });

  test('a slender twisted bar is limited by rotation, and a short one is not', () => {
    const maxDisplacement = 0.25;
    const slender = exaggerationScale(maxDisplacement, length, 0.05);
    expect(slender * 0.05).toBeLessThanOrEqual(MAX_DRAWN_ROTATION);
    expect(slender).toBeLessThan(exaggerationScale(maxDisplacement, length));
    expect(exaggerationScale(maxDisplacement, length, 1e-5)).toBe(exaggerationScale(maxDisplacement, length));
  });

  test('the factor is a round number the label can state exactly when rotation limits it', () => {
    for (let rotation = 1e-4; rotation < 10; rotation *= 1.61) {
      const scale = exaggerationScale(1e-6, length, rotation);
      expect(Number(scale.toPrecision(2))).toBe(scale);
    }
  });

  test('no rotation yet gives the share-of-length factor', () => {
    expect(exaggerationScale(7, length, 0)).toBe(exaggerationScale(7, length));
    expect(exaggerationScale(7, length, Number.NaN)).toBe(exaggerationScale(7, length));
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

  test('zero rotation is exactly the straight line', () => {
    const straight = new Float32Array(6);
    const arc = new Float32Array(6);
    displacePositions(base, displacement, 37.5, 0.3, straight);
    displacePositions(base, displacement, 37.5, 0.3, arc, new Float32Array(6));
    expect(Array.from(arc)).toEqual(Array.from(straight));
  });

  test('a rotation at phase zero moves nothing', () => {
    const rotation = new Float32Array([0.3, -0.1, 0.2, 0, 0, 0.5]);
    const out = new Float32Array(6);
    displacePositions(base, displacement, 1000, 0, out, rotation);
    expect(Array.from(out)).toEqual(Array.from(base));
  });
});

describe('displacePositions along an arc', () => {
  /** A rigid turn at `rate` about the z axis, with the solver's tangent displacement at each point. */
  function rigidTurn(points: number[], rate: number) {
    const base = new Float32Array(points);
    const displacement = new Float32Array(points.length);
    const rotation = new Float32Array(points.length);
    for (let i = 0; i + 2 < points.length; i += 3) {
      displacement[i] = -rate * points[i + 1];
      displacement[i + 1] = rate * points[i];
      rotation[i + 2] = rate;
    }
    return { base, displacement, rotation };
  }

  const POINTS = [10, 0, 5, 3, 4, -2, -6, 8, 40, 0.5, -0.25, 1];
  // The arrays are Float32, whose rounding is 6e-8 relative to a coordinate of size up to 10;
  // 1e-5 is that rounding with a margin of about 15.
  const FLOAT32_TOLERANCE = 1e-5;

  for (const degrees of [60, 120]) {
    test(`a pure rotation keeps every point's distance from the axis at ${degrees} degrees`, () => {
      const rate = 2e-3;
      const { base, displacement, rotation } = rigidTurn(POINTS, rate);
      const angle = (degrees * Math.PI) / 180;
      const out = new Float32Array(base.length);
      displacePositions(base, displacement, angle / rate, 1, out, rotation);
      for (let i = 0; i + 2 < base.length; i += 3) {
        const before = Math.hypot(base[i], base[i + 1]);
        const after = Math.hypot(out[i], out[i + 1]);
        expect(Math.abs(after - before)).toBeLessThan(FLOAT32_TOLERANCE);
      }
    });

    test(`a pure rotation lands where the rotation puts it at ${degrees} degrees`, () => {
      const rate = 2e-3;
      const { base, displacement, rotation } = rigidTurn(POINTS, rate);
      const angle = (degrees * Math.PI) / 180;
      const out = new Float32Array(base.length);
      displacePositions(base, displacement, angle / rate, 1, out, rotation);
      for (let i = 0; i + 2 < base.length; i += 3) {
        const x = base[i] * Math.cos(angle) - base[i + 1] * Math.sin(angle);
        const y = base[i] * Math.sin(angle) + base[i + 1] * Math.cos(angle);
        expect(Math.abs(out[i] - x)).toBeLessThan(FLOAT32_TOLERANCE);
        expect(Math.abs(out[i + 1] - y)).toBeLessThan(FLOAT32_TOLERANCE);
        expect(out[i + 2]).toBe(base[i + 2]);
      }
    });
  }

  test('a small rotation matches the straight line', () => {
    const rate = 1e-6;
    const { base, displacement, rotation } = rigidTurn(POINTS, rate);
    const out = new Float32Array(base.length);
    const amount = 100; // A drawn angle of 1e-4 rad.
    displacePositions(base, displacement, amount, 1, out, rotation);
    for (let i = 0; i < base.length; i++) {
      const straight = base[i] + displacement[i] * amount;
      // The arc leaves the straight line by half the drawn angle times the drawn movement, here
      // under 1e-4 * 10 * 1e-4 = 1e-7, and Float32 rounding of a coordinate near 40 adds 2.4e-6;
      // 5e-6 is those with a margin of two.
      expect(Math.abs(out[i] - straight)).toBeLessThan(5e-6);
    }
  });

  test('the arc has no jump where its series hands over to its closed form', () => {
    // The series is used below a drawn angle of 1e-2 rad.
    const { base, displacement, rotation } = rigidTurn([1, 1, 1], 1e-3);
    const below = new Float32Array(3);
    const above = new Float32Array(3);
    displacePositions(base, displacement, (1e-2 * 0.999) / 1e-3, 1, below, rotation);
    displacePositions(base, displacement, (1e-2 * 1.001) / 1e-3, 1, above, rotation);
    // The point moves by the angle difference (2e-5 rad) times its radius (1.42), 2.8e-5 of
    // travel; Float32 rounding of a coordinate near 1 adds 1.2e-7 on each axis. A jump at the
    // changeover would show on top of that.
    expect(Math.hypot(below[0] - above[0], below[1] - above[1])).toBeLessThan(3e-5);
  });

  test('a point on the axis of a turning part stays where it is', () => {
    const base = new Float32Array([0, 0, 7]);
    const displacement = new Float32Array([0, 0, 0]);
    const rotation = new Float32Array([0, 0, 0.02]);
    const out = new Float32Array(3);
    displacePositions(base, displacement, 50, 1, out, rotation);
    expect(Array.from(out)).toEqual(Array.from(base));
  });
});
