// Tests of the results panel's wording in src/results-view.ts: a percentage reads as a result only
// at done and inside the model, while solving it is marked in progress (never "-100.0 %"), and a
// result outside the model is marked not meaningful with a warning naming why.

import { describe, expect, test } from 'bun:test';
import { modelWarning, safetyMessage, tableView, type Summary, type Validity } from './results-view.ts';

const inside: Validity = {
  withinModel: true, exceedsYield: false, excessiveDeflection: false, excessiveTwist: false,
  deflectionLimit: 12.9, twistRadians: null, twistLimitRadians: null,
};
const row = {
  quantity: 'Deflection', unit: 'mm', place: 'mean vertical displacement of the free end section',
  fea: 2.45, textbook: 2.5, percentDifference: -2,
};

function summary(state: string, validity: Validity, extra: Partial<Summary> = {}): Summary {
  return { state, textbook: 'compared', rows: [row], validity, safetyFactor: 3, maxDisplacement: 2.45, ...extra };
}

describe('the percentage', () => {
  test('is a result only at done and inside the model', () => {
    const rows = tableView(summary('done', inside)).rows;
    expect(rows[0].difference).toBe('-2.0 %');
    expect(rows[0].status).toBe('result');
  });

  test('never reads as a verdict while the solve runs, even at -100 %', () => {
    for (const state of ['assembling', 'solving', 'recovering']) {
      const rows = tableView(summary(state, inside, { rows: [{ ...row, fea: 0, percentDifference: -100 }] })).rows;
      expect(rows[0].difference).toBe('solving…');
      expect(rows[0].status).toBe('solving');
      expect(rows[0].difference).not.toContain('100');
    }
  });

  test('keeps the live FEA value during the solve', () => {
    expect(tableView(summary('solving', inside)).rows[0].fea).toBe('2.45');
  });

  test('is marked not meaningful outside the model, even when it would read as agreement', () => {
    const outside: Validity = { ...inside, withinModel: false, exceedsYield: true, excessiveDeflection: true };
    const rows = tableView(summary('done', outside, { rows: [{ ...row, percentDifference: -1.8 }] })).rows;
    expect(rows[0].difference).toBe('not meaningful');
    expect(rows[0].status).toBe('not-meaningful');
    expect(rows[0].difference).not.toContain('1.8');
  });

  test('says there is no textbook answer in the engine’s words', () => {
    const view = tableView({ state: 'done', textbook: 'none', explanation: 'A closed tube has no simple formula.', validity: inside });
    expect(view.rows).toEqual([]);
    expect(view.note).toContain('No textbook answer');
    expect(view.note).toContain('A closed tube has no simple formula.');
  });
});

describe('the warning', () => {
  test('is absent inside the model and during a solve', () => {
    expect(modelWarning(summary('done', inside))).toBeNull();
    const outside = { ...inside, withinModel: false, exceedsYield: true };
    expect(modelWarning(summary('solving', outside))).toBeNull();
  });

  test('names yield in plain language and says the numbers do not apply', () => {
    const text = modelWarning(summary('done', { ...inside, withinModel: false, exceedsYield: true }))!;
    expect(text).toContain('permanently bend or break');
    expect(text).toContain('do not apply');
    expect(text).toContain('textbook comparison');
  });

  test('names a deflection too large for the model, with both values', () => {
    const text = modelWarning(summary('done', { ...inside, withinModel: false, excessiveDeflection: true }, { maxDisplacement: 166 }))!;
    expect(text).toContain('bend 166 mm');
    expect(text).toContain('12.9 mm');
  });

  test('names a twist too large for the model in degrees', () => {
    const twist: Validity = {
      ...inside, withinModel: false, excessiveTwist: true, deflectionLimit: null,
      twistRadians: 0.5, twistLimitRadians: 0.1,
    };
    const text = modelWarning(summary('done', twist))!;
    expect(text).toContain('28.6°');
    expect(text).toContain('5.73°');
  });

  test('gives every reason when more than one limit is passed', () => {
    const both = { ...inside, withinModel: false, exceedsYield: true, excessiveDeflection: true };
    const text = modelWarning(summary('done', both, { maxDisplacement: 166 }))!;
    expect(text).toContain('permanently bend or break');
    expect(text).toContain('and it would bend 166 mm');
  });
});

describe('the safety factor message', () => {
  test('is an estimate during the solve', () => {
    expect(safetyMessage(summary('solving', inside))).toContain('Estimate');
  });

  test('agrees with the warning past yield', () => {
    const text = safetyMessage(summary('done', { ...inside, withinModel: false, exceedsYield: true }, { safetyFactor: 0.1 }));
    expect(text).toContain('would yield');
    expect(text).toContain('does not apply');
  });

  test('does not call a too-flexible part comfortable', () => {
    const text = safetyMessage(summary('done', { ...inside, withinModel: false, excessiveDeflection: true }, { safetyFactor: 4 }));
    expect(text).toContain('not reliable');
    expect(text).not.toContain('Comfortably');
  });

  test('reads normally inside the model', () => {
    expect(safetyMessage(summary('done', inside, { safetyFactor: 5 }))).toContain('below');
  });
});
