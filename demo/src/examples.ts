// What the page shows about an example part, as plain data and pure functions so `bun test` can
// hold it: the chooser's rows and the status line come from the engine's table
// (demo/wasm/src/examples.rs, which holds the strings in one place), the load-size slider's words,
// and the size that scales a part's deflection drawing. src/main.ts connects these to the DOM.

import { LOG_SLIDER_POSITIONS, logSliderPosition, logSliderValue, type Range } from './catalog.ts';
import { formatNumber } from './format.ts';

/** How the benchmark judged a part: the status line's tone. */
export type ExampleVerdict = 'pass' | 'limited' | 'fail';

/** One row of the engine's example table, from `catalogJson().examples`. */
export interface ExamplePart {
  id: string;
  name: string;
  tests: string;
  held: string;
  loaded: string;
  asset: string;
  load: string;
  verdict: ExampleVerdict;
  status: string;
  /** From the part towards the camera on opening, chosen to show its shape. */
  view: [number, number, number];
}

/** The load-size slider's range: from a tenth to ten times what the benchmark applied. */
export const LOAD_SCALE_RANGE: Range = { min: 0.1, max: 10 };

/** The material dropdown's value for the benchmark's own material. */
export const AS_TESTED = -1;

/** A chooser row: the part's friendly name, and its id to pick it by. */
export function chooserOptions(parts: ExamplePart[]): { value: string; text: string }[] {
  return parts.map(part => ({ value: part.id, text: part.name }));
}

/** The status line of a part, with the tone that says how to read it. */
export function statusView(part: ExamplePart): { tone: ExampleVerdict; text: string } {
  return { tone: part.verdict, text: part.status };
}

/** The part's description, as shown beside the chooser. */
export function aboutText(part: ExamplePart): string {
  return `${part.tests} Held: ${part.held} Load: ${part.loaded}`;
}

/** The slider position of a load size. */
export function loadScalePosition(scale: number): number {
  return logSliderPosition(scale, LOAD_SCALE_RANGE);
}

/** The load size at a slider position. */
export function loadScaleAt(position: number): number {
  return logSliderValue(position, LOAD_SCALE_RANGE);
}

/** The slider's words for a load size, naming what the benchmark applied. */
export function loadScaleText(part: ExamplePart, scale: number): string {
  return `${formatNumber(scale)} × the tested load (${part.load})`;
}

export { LOG_SLIDER_POSITIONS };

/** The largest side of the part's bounding box, in the positions' units: the deflection scale. */
export function largestExtent(positions: Float32Array): number {
  if (positions.length < 3) return 0;
  const low = [Infinity, Infinity, Infinity];
  const high = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i + 2 < positions.length; i += 3) {
    for (let axis = 0; axis < 3; axis++) {
      low[axis] = Math.min(low[axis], positions[i + axis]);
      high[axis] = Math.max(high[axis], positions[i + axis]);
    }
  }
  return Math.max(high[0] - low[0], high[1] - low[1], high[2] - low[2]);
}

/** The line under the safety factor for an example part. */
export function exampleSafetyMessage(hasYield: boolean, factor: number | null | undefined, verdict: string): string {
  if (hasYield) return verdict;
  if (factor === null || factor === undefined) {
    return 'The benchmark gives this material no yield stress, so there is no safety factor. Pick a material to see one.';
  }
  return verdict;
}

/** The progress words while a part is being meshed, or null when it is past meshing. */
export function meshingText(summary: { state: string; meshDone?: number; meshTotal?: number }): string | null {
  const percent = (summary.meshTotal ?? 0) > 0 ? Math.round((100 * (summary.meshDone ?? 0)) / (summary.meshTotal ?? 1)) : null;
  switch (summary.state) {
    case 'meshing': return percent === null ? 'Meshing the part…' : `Meshing the part: ${percent}%`;
    case 'refining': return percent === null ? 'Refining the mesh…' : `Refining the mesh: ${percent}%`;
    case 'preparing': return 'Placing the holds and loads…';
    default: return null;
  }
}
