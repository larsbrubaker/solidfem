// The example chooser and status lines against the real table (src/testdata/catalog.json, kept
// identical to catalogJson() by the wasm tests): every part is offered with a friendly name and a
// line of what it tests, a failing part says so before anything else, and a pass limited to mesh
// stability does not claim accuracy.

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'fs';
import { join } from 'path';
import { parseCatalog } from './catalog.ts';
import {
  LOAD_SCALE_RANGE, chooserOptions, largestExtent, loadScaleAt, loadScalePosition, loadScaleText,
  meshingText, statusView,
} from './examples.ts';

const catalog = parseCatalog(readFileSync(join(import.meta.dir, 'testdata/catalog.json'), 'utf-8'));
const parts = catalog.examples;

describe('the example chooser', () => {
  test('offers every part by a friendly name with what it tests', () => {
    const options = chooserOptions(parts);
    expect(options.length).toBe(parts.length);
    expect(options.length).toBeGreaterThanOrEqual(11);
    for (const part of parts) {
      expect(part.name.length).toBeGreaterThan(0);
      expect(part.tests.length).toBeGreaterThan(0);
      expect(part.held.length).toBeGreaterThan(0);
      expect(part.loaded.length).toBeGreaterThan(0);
    }
  });

  test('names NAFEMS benchmarks in words a newcomer can read', () => {
    const le10 = parts.find(part => part.id === 'le10');
    expect(le10?.name).toContain('Thick plate');
    expect(le10?.name).toContain('NAFEMS');
  });
});

describe('the status line', () => {
  test('a failing part says so first', () => {
    for (const part of parts.filter(p => p.verdict === 'fail')) {
      const { text } = statusView(part);
      expect(text.startsWith('Not trustworthy yet') || text.startsWith('Not checked')).toBe(true);
    }
  });

  test('a pass is a match against a reference, and a limited pass does not claim that', () => {
    for (const part of parts) {
      const { text, tone } = statusView(part);
      if (tone === 'pass') expect(text.startsWith('Matches')).toBe(true);
      if (tone === 'limited') expect(text).not.toMatch(/^Matches/);
    }
  });
});

describe('the load size', () => {
  test('the slider covers a tenth to ten times the tested load and round-trips', () => {
    expect(loadScaleAt(0)).toBeCloseTo(LOAD_SCALE_RANGE.min, 6);
    expect(loadScaleAt(1000)).toBeCloseTo(LOAD_SCALE_RANGE.max, 6);
    for (const scale of [0.2, 1, 7]) {
      expect(loadScaleAt(loadScalePosition(scale))).toBeCloseTo(scale, 1);
    }
  });

  test('states what the benchmark applied', () => {
    expect(loadScaleText(parts[0], 1)).toContain(parts[0].load);
  });
});

describe('the deflection scale', () => {
  test('is the part bounding box’s largest side', () => {
    expect(largestExtent(new Float32Array([0, 0, 0, 10, 2, 3]))).toBe(10);
    expect(largestExtent(new Float32Array(0))).toBe(0);
  });
});

describe('the meshing progress', () => {
  test('shows a percentage while meshing and nothing past it', () => {
    expect(meshingText({ state: 'meshing', meshDone: 1, meshTotal: 4 })).toBe('Meshing the part: 25%');
    expect(meshingText({ state: 'refining', meshDone: 0, meshTotal: 0 })).toBe('Refining the mesh…');
    expect(meshingText({ state: 'preparing' })).toBe('Placing the holds and loads…');
    expect(meshingText({ state: 'solving' })).toBeNull();
  });
});
