// What the page offers, from the engine's catalogJson(): the catalog's shape, the visitor's
// choice, and the clamping that keeps a choice inside the ranges of the current combination, so
// no slider position can ever be refused by Demo.start. Everything here is plain data and pure
// functions (tested in src/catalog.test.ts against a real catalogJson() fixture); src/main.ts
// holds the DOM wiring, and the wasm's validation (demo/wasm/src/catalog.rs) stays the authority.
//
// The magnitude slider is logarithmic, because a push of 1 N and one of 2000 N are both real
// choices and a linear slider would leave the small loads in the first pixels. Each load kind
// remembers its own magnitude: a push of 500 N and a twist of 500 N mm mean different things.

export type Section = 'solid' | 'tube';
export type Restraint = 'one_end' | 'both_ends';
export type Load = 'push' | 'twist';

export interface Range {
  min: number;
  max: number;
}

export interface Material {
  name: string;
  youngsModulus: number;
  poissonRatio: number;
  yieldStrength: number;
}

/** One (section, restraint, load) combination and its allowed ranges. */
export interface Combination {
  section: Section;
  restraint: Restraint;
  load: Load;
  hasTextbookAnswer: boolean;
  length: Range;
  width: Range;
  height: Range;
  /** Null for a solid section. */
  wallFraction: Range | null;
  magnitude: Range;
}

export interface Catalog {
  displayName: string;
  maxElements: number;
  materials: Material[];
  combinations: Combination[];
}

/** What the visitor has chosen; `magnitudes` keeps one value per load kind. */
export interface Choice {
  section: Section;
  restraint: Restraint;
  load: Load;
  material: number;
  length: number;
  width: number;
  height: number;
  wallFraction: number;
  magnitudes: Record<Load, number>;
}

/** The arguments of `Demo.start`, in its order. */
export type StartArguments = [
  section: Section,
  length: number,
  width: number,
  height: number,
  wallFraction: number,
  material: number,
  restraint: Restraint,
  load: Load,
  magnitude: number,
];

/** Positions of the logarithmic magnitude slider. */
export const LOG_SLIDER_POSITIONS = 1000;
/** Where on the slider a new load starts: low enough that the default part is safe. */
const DEFAULT_LOG_FRACTION = 0.3;

/** Slider steps: 1 mm of length, 0.5 mm across the section, 0.01 of wall fraction. */
export const STEP_LENGTH = 1;
export const STEP_SIDE = 0.5;
export const STEP_WALL = 0.01;

export function parseCatalog(json: string): Catalog {
  return JSON.parse(json) as Catalog;
}

/** The combination a choice is in. The engine offers every pairing, so this always finds one. */
export function combinationOf(catalog: Catalog, section: Section, restraint: Restraint, load: Load): Combination {
  const found = catalog.combinations.find(
    c => c.section === section && c.restraint === restraint && c.load === load,
  );
  if (!found) throw new Error(`The engine does not offer ${section} / ${restraint} / ${load}.`);
  return found;
}

/** A value inside `range`; a value that is not a number becomes the range's minimum. */
export function clampToRange(value: number, range: Range): number {
  if (!Number.isFinite(value)) return range.min;
  return Math.min(range.max, Math.max(range.min, value));
}

/** `value` rounded to `digits` significant figures. */
export function roundToSignificant(value: number, digits: number): number {
  if (value === 0 || !Number.isFinite(value)) return 0;
  const magnitude = Math.floor(Math.log10(Math.abs(value)));
  const factor = Math.pow(10, digits - 1 - magnitude);
  return Math.round(value * factor) / factor;
}

/** The value at a position of the logarithmic slider, kept inside `range`. */
export function logSliderValue(position: number, range: Range): number {
  const fraction = Math.min(1, Math.max(0, position / LOG_SLIDER_POSITIONS));
  const raw = range.min * Math.pow(range.max / range.min, fraction);
  return clampToRange(roundToSignificant(raw, 3), range);
}

/** The slider position nearest to `value`. */
export function logSliderPosition(value: number, range: Range): number {
  const clamped = clampToRange(value, range);
  const fraction = Math.log(clamped / range.min) / Math.log(range.max / range.min);
  return Math.round(fraction * LOG_SLIDER_POSITIONS);
}

/** The opening choice: a solid bar held at one end, at the middle of its ranges, under a small push. */
export function defaultChoice(catalog: Catalog): Choice {
  const first = combinationOf(catalog, 'solid', 'one_end', 'push');
  const twist = combinationOf(catalog, 'solid', 'one_end', 'twist');
  const middle = (r: Range) => 0.5 * (r.min + r.max);
  const start = (r: Range) =>
    clampToRange(roundToSignificant(r.min * Math.pow(r.max / r.min, DEFAULT_LOG_FRACTION), 1), r);
  const tube = catalog.combinations.find(c => c.wallFraction !== null);
  return {
    section: first.section,
    restraint: first.restraint,
    load: first.load,
    material: 0,
    length: middle(first.length),
    width: middle(first.width),
    height: middle(first.height),
    wallFraction: tube && tube.wallFraction ? middle(tube.wallFraction) : 0,
    magnitudes: { push: start(first.magnitude), twist: start(twist.magnitude) },
  };
}

/**
 * The choice with every number inside the ranges of its combination. A solid has no wall range,
 * so its wall value is kept for when the visitor returns to a tube.
 */
export function reclamp(catalog: Catalog, choice: Choice): Choice {
  const combination = combinationOf(catalog, choice.section, choice.restraint, choice.load);
  const magnitudes = { ...choice.magnitudes };
  for (const load of ['push', 'twist'] as const) {
    const ranges = combinationOf(catalog, choice.section, choice.restraint, load);
    magnitudes[load] = clampToRange(magnitudes[load], ranges.magnitude);
  }
  return {
    ...choice,
    material: Math.min(Math.max(0, Math.trunc(choice.material)), catalog.materials.length - 1),
    length: clampToRange(choice.length, combination.length),
    width: clampToRange(choice.width, combination.width),
    height: clampToRange(choice.height, combination.height),
    wallFraction: combination.wallFraction
      ? clampToRange(choice.wallFraction, combination.wallFraction)
      : choice.wallFraction,
    magnitudes,
  };
}

/** The arguments to start a study of `choice`, re-clamped so they are always accepted. */
export function startArguments(catalog: Catalog, choice: Choice): StartArguments {
  const c = reclamp(catalog, choice);
  return [c.section, c.length, c.width, c.height, c.wallFraction, c.material, c.restraint, c.load,
    c.magnitudes[c.load]];
}
