// Tests of src/catalog.ts against the real catalog (src/testdata/catalog.json, which the wasm
// tests keep identical to catalogJson()): every combination at every corner of its ranges and at
// every magnitude slider position produces start arguments the engine's rule accepts, and
// changing combination re-clamps. catalog.engine.test.ts runs the same corners through the wasm.

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'fs';
import { join } from 'path';
import {
  LOG_SLIDER_POSITIONS, clampToRange, combinationOf, defaultChoice, logSliderPosition,
  logSliderValue, parseCatalog, reclamp, startArguments,
  type Catalog, type Choice, type Combination, type Range,
} from './catalog.ts';

const catalog: Catalog = parseCatalog(readFileSync(join(import.meta.dir, 'testdata/catalog.json'), 'utf-8'));

function inRange(value: number, range: Range): boolean {
  return Number.isFinite(value) && value >= range.min && value <= range.max;
}

function argsInRange(combination: Combination, args: ReturnType<typeof startArguments>) {
  const [section, length, width, height, wall, material, restraint, load, magnitude] = args;
  expect(section).toBe(combination.section);
  expect(restraint).toBe(combination.restraint);
  expect(load).toBe(combination.load);
  expect(inRange(length, combination.length)).toBe(true);
  expect(inRange(width, combination.width)).toBe(true);
  expect(inRange(height, combination.height)).toBe(true);
  expect(inRange(magnitude, combination.magnitude)).toBe(true);
  expect(Number.isFinite(wall)).toBe(true);
  if (combination.wallFraction) expect(inRange(wall, combination.wallFraction)).toBe(true);
  expect(Number.isInteger(material) && material >= 0 && material < catalog.materials.length).toBe(true);
}

function choiceIn(combination: Combination, pick: (r: Range) => number): Choice {
  const base = defaultChoice(catalog);
  return {
    ...base,
    section: combination.section,
    restraint: combination.restraint,
    load: combination.load,
    length: pick(combination.length),
    width: pick(combination.width),
    height: pick(combination.height),
    wallFraction: combination.wallFraction ? pick(combination.wallFraction) : base.wallFraction,
    magnitudes: { ...base.magnitudes, [combination.load]: pick(combination.magnitude) },
  };
}

describe('the catalog fixture', () => {
  test('offers every pairing the page has buttons for', () => {
    for (const section of ['solid', 'tube'] as const) {
      for (const restraint of ['one_end', 'both_ends'] as const) {
        for (const load of ['push', 'twist'] as const) {
          expect(combinationOf(catalog, section, restraint, load)).toBeDefined();
        }
      }
    }
  });

  test('a tube has a wall range and a solid does not', () => {
    for (const combination of catalog.combinations) {
      expect(combination.wallFraction !== null).toBe(combination.section === 'tube');
    }
  });
});

describe('start arguments', () => {
  test('every combination is in range at its minimum, middle and maximum', () => {
    for (const combination of catalog.combinations) {
      for (const pick of [(r: Range) => r.min, (r: Range) => 0.5 * (r.min + r.max), (r: Range) => r.max]) {
        argsInRange(combination, startArguments(catalog, choiceIn(combination, pick)));
      }
    }
  });

  test('every combination stays in range when the visitor has asked for too much', () => {
    for (const combination of catalog.combinations) {
      argsInRange(combination, startArguments(catalog, choiceIn(combination, r => r.max * 1.7)));
      argsInRange(combination, startArguments(catalog, choiceIn(combination, r => r.min * 0.1)));
      argsInRange(combination, startArguments(catalog, choiceIn(combination, () => Number.NaN)));
    }
  });

  test('every magnitude slider position of every combination is in range', () => {
    for (const combination of catalog.combinations) {
      for (let position = 0; position <= LOG_SLIDER_POSITIONS; position++) {
        const value = logSliderValue(position, combination.magnitude);
        expect(inRange(value, combination.magnitude)).toBe(true);
      }
    }
  });

  test('every linear slider position of every combination is in range', () => {
    const steps: [keyof Combination, number][] = [['length', 1], ['width', 0.5], ['height', 0.5], ['wallFraction', 0.01]];
    for (const combination of catalog.combinations) {
      for (const [key, step] of steps) {
        const range = combination[key] as Range | null;
        if (!range) continue;
        for (let value = range.min; value <= range.max + step; value += step) {
          // A browser's stepped value, however it rounds, is clamped before it reaches the engine.
          const clamped = clampToRange(value, range);
          expect(inRange(clamped, range)).toBe(true);
        }
      }
    }
  });

  test('the magnitude slider reaches both ends of the range', () => {
    for (const combination of catalog.combinations) {
      expect(logSliderValue(0, combination.magnitude)).toBe(combination.magnitude.min);
      expect(logSliderValue(LOG_SLIDER_POSITIONS, combination.magnitude)).toBe(combination.magnitude.max);
    }
  });

  test('a slider position survives the round trip through its value', () => {
    const combination = combinationOf(catalog, 'solid', 'one_end', 'push');
    for (const position of [0, 137, 500, 871, LOG_SLIDER_POSITIONS]) {
      const value = logSliderValue(position, combination.magnitude);
      expect(Math.abs(logSliderPosition(value, combination.magnitude) - position)).toBeLessThanOrEqual(2);
    }
  });
});

describe('re-clamping when the combination changes', () => {
  test('moving from a long slim both-ends solid to a both-ends tube pulls the values in', () => {
    const start = reclamp(catalog, {
      ...defaultChoice(catalog), restraint: 'both_ends', length: 320, width: 14, height: 14,
    });
    expect(start.length).toBe(320);
    expect(start.width).toBe(14);
    const tube = reclamp(catalog, { ...start, section: 'tube' });
    const ranges = combinationOf(catalog, 'tube', 'both_ends', 'push');
    expect(tube.length).toBe(ranges.length.max);
    expect(tube.width).toBe(ranges.width.min);
    expect(tube.height).toBe(ranges.height.min);
  });

  test('each load kind keeps its own magnitude across a change', () => {
    let choice = reclamp(catalog, defaultChoice(catalog));
    choice = reclamp(catalog, { ...choice, magnitudes: { push: 500, twist: choice.magnitudes.twist } });
    const twist = reclamp(catalog, { ...choice, load: 'twist' });
    expect(twist.magnitudes.push).toBe(500);
    expect(startArguments(catalog, twist)[8]).toBe(twist.magnitudes.twist);
    expect(reclamp(catalog, { ...twist, load: 'push' }).magnitudes.push).toBe(500);
  });

  test('a solid keeps the wall value for the next tube', () => {
    const tube = reclamp(catalog, { ...defaultChoice(catalog), section: 'tube', wallFraction: 0.3 });
    const solid = reclamp(catalog, { ...tube, section: 'solid' });
    expect(reclamp(catalog, { ...solid, section: 'tube' }).wallFraction).toBeCloseTo(0.3, 12);
  });

  test('the opening choice is in range and starts well below the 2000 N maximum', () => {
    const choice = defaultChoice(catalog);
    const combination = combinationOf(catalog, choice.section, choice.restraint, choice.load);
    argsInRange(combination, startArguments(catalog, choice));
    expect(choice.magnitudes.push).toBe(10);
    expect(choice.magnitudes.twist).toBe(1000);
  });

  test('a material index past the table falls back inside it', () => {
    const choice = reclamp(catalog, { ...defaultChoice(catalog), material: 99 });
    expect(choice.material).toBe(catalog.materials.length - 1);
  });
});
