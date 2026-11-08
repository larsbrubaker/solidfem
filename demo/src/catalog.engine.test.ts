// Runs src/catalog.ts through the real engine: the wasm built by `bun run build:wasm` is loaded in
// Bun, and every corner of every combination must start a study, and one small study must solve to
// the end through `Demo.step`. Skipped only when public/pkg has not been built yet.

import { describe, expect, test } from 'bun:test';
import { existsSync, readFileSync } from 'fs';
import { join } from 'path';
import { defaultChoice, parseCatalog, startArguments, type Catalog, type Combination, type Range } from './catalog.ts';
import { modelWarning, tableView } from './results-view.ts';

const pkg = join(import.meta.dir, '../public/pkg');
const built = existsSync(join(pkg, 'solidfem_wasm.js'));

async function loadWasm() {
  const wasm = await import('../public/pkg/solidfem_wasm.js' as string);
  await wasm.default({ module_or_path: readFileSync(join(pkg, 'solidfem_wasm_bg.wasm')) });
  return wasm;
}

describe.skipIf(!built)('the real engine', () => {
  test('accepts every combination at every corner of its ranges', async () => {
    const wasm = await loadWasm();
    const catalog: Catalog = parseCatalog(wasm.catalogJson());
    const demo = new wasm.Demo();
    const base = defaultChoice(catalog);
    const pickers = [(r: Range) => r.min, (r: Range) => r.max];
    for (const combination of catalog.combinations as Combination[]) {
      for (const pickLength of pickers) {
        for (const pickSide of pickers) {
          for (const pickMagnitude of pickers) {
            const args = startArguments(catalog, {
              ...base,
              section: combination.section,
              restraint: combination.restraint,
              load: combination.load,
              length: pickLength(combination.length),
              width: pickSide(combination.width),
              height: pickSide(combination.height),
              wallFraction: combination.wallFraction ? pickSide(combination.wallFraction) : base.wallFraction,
              magnitudes: { ...base.magnitudes, [combination.load]: pickMagnitude(combination.magnitude) },
            });
            expect(demo.start(...args)).toBe('assembling');
          }
        }
      }
    }
  }, 60000);

  test('the opening choice solves to the end and matches the textbook', async () => {
    const wasm = await loadWasm();
    const catalog: Catalog = parseCatalog(wasm.catalogJson());
    const demo = new wasm.Demo();
    demo.start(...startArguments(catalog, defaultChoice(catalog)));
    let state = 'assembling';
    for (let i = 0; i < 100000 && state !== 'done' && state !== 'failed'; i++) {
      state = demo.step(50);
    }
    expect(state).toBe('done');
    const summary = JSON.parse(demo.summaryJson());
    expect(summary.textbook).toBe('compared');
    expect(summary.rows.length).toBeGreaterThan(0);
    for (const row of summary.rows) {
      expect(Number.isFinite(row.percentDifference)).toBe(true);
      expect(Math.abs(row.percentDifference)).toBeLessThan(10);
    }
    expect(summary.safetyFactor).toBeGreaterThan(0);
    expect(summary.validity.withinModel).toBe(true);
    expect(demo.vonMises().length).toBe(demo.positions().length / 3);
  }, 120000);

  test('the top of the load slider is reported outside the model and marked not meaningful', async () => {
    const wasm = await loadWasm();
    const catalog: Catalog = parseCatalog(wasm.catalogJson());
    const demo = new wasm.Demo();
    const base = defaultChoice(catalog);
    demo.start(...startArguments(catalog, { ...base, magnitudes: { ...base.magnitudes, push: 2000 } }));
    let state = 'assembling';
    for (let i = 0; i < 100000 && state !== 'done' && state !== 'failed'; i++) {
      state = demo.step(50);
    }
    expect(state).toBe('done');
    const summary = JSON.parse(demo.summaryJson());
    expect(summary.validity.withinModel).toBe(false);
    expect(summary.validity.exceedsYield).toBe(true);
    expect(summary.validity.excessiveDeflection).toBe(true);
    expect(tableView(summary).rows[0].difference).toBe('not meaningful');
    expect(modelWarning(summary)).toContain('do not apply');
  }, 120000);
});
