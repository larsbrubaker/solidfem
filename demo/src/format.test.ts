// Tests of the on-screen number text in src/format.ts, including the percentage difference the
// textbook table shows.

import { describe, expect, test } from 'bun:test';
import {
  exaggerationLabel, formatDuration, formatNumber, formatPercent, formatSafetyFactor, safetyVerdict, stateLabel,
} from './format.ts';

describe('formatPercent', () => {
  test('gives one decimal and an explicit sign', () => {
    expect(formatPercent(1.234)).toBe('+1.2 %');
    expect(formatPercent(-0.44)).toBe('-0.4 %');
    expect(formatPercent(12)).toBe('+12.0 %');
  });

  test('never shows a negative zero', () => {
    expect(formatPercent(0)).toBe('0.0 %');
    expect(formatPercent(-0.04)).toBe('0.0 %');
    expect(formatPercent(0.04)).toBe('0.0 %');
  });

  test('drops decimals once the number is huge', () => {
    expect(formatPercent(1234.56)).toBe('+1235 %');
    expect(formatPercent(-1000)).toBe('-1000 %');
  });

  test('a missing or non-finite value shows a dash, not NaN', () => {
    expect(formatPercent(null)).toBe('—');
    expect(formatPercent(undefined)).toBe('—');
    expect(formatPercent(Number.NaN)).toBe('—');
    expect(formatPercent(Infinity)).toBe('—');
  });
});

describe('formatNumber', () => {
  test('gives three significant figures by default', () => {
    expect(formatNumber(0.12345)).toBe('0.123');
    expect(formatNumber(12.345)).toBe('12.3');
    expect(formatNumber(1234.5)).toBe('1,235');
  });

  test('zero stays plain and extremes use exponent form', () => {
    expect(formatNumber(0)).toBe('0');
    expect(formatNumber(2.5e7)).toBe('2.50e+7');
    expect(formatNumber(1.5e-5)).toBe('1.50e-5');
  });

  test('a missing value shows a dash', () => {
    expect(formatNumber(null)).toBe('—');
  });
});

describe('the safety factor', () => {
  test('the text says what the number means', () => {
    expect(formatSafetyFactor(2.5)).toBe('2.5');
    expect(formatSafetyFactor(250)).toBe('over 100');
    expect(formatSafetyFactor(null)).toBe('—');
    expect(safetyVerdict(0.5)).toContain('would yield');
    expect(safetyVerdict(1.4)).toContain('close to yielding');
    expect(safetyVerdict(5)).toContain('below');
    expect(safetyVerdict(null)).toContain('Waiting');
  });
});

describe('exaggerationLabel', () => {
  test('states the factor and the true deflection', () => {
    expect(exaggerationLabel(1200, 0.0155)).toBe('Shape exaggerated ×1,200. True deflection: 0.0155 mm.');
  });

  test('says so when the shape is drawn smaller than true', () => {
    expect(exaggerationLabel(0.012, 1560)).toContain('smaller than true deflection');
    expect(exaggerationLabel(0.012, 1560)).toContain('1,560 mm');
  });

  test('no deflection yet is stated plainly', () => {
    expect(exaggerationLabel(0, 0)).toContain('undeformed');
  });
});

test('durations read naturally', () => {
  expect(formatDuration(340)).toBe('340 ms');
  expect(formatDuration(1500)).toBe('1.5 s');
});

test('every engine state has words', () => {
  for (const state of ['assembling', 'solving', 'recovering', 'done', 'failed', 'too_large', 'cancelled']) {
    expect(stateLabel(state).length).toBeGreaterThan(0);
  }
});
