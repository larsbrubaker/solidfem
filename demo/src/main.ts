// The page's entry point: loads the wasm package, builds the controls from catalogJson(), runs the
// study through src/scheduler.ts and shows each result in the viewer, the legend and the tables.
// The rules live with their tests: ranges and clamping in src/catalog.ts, slices and restarts in
// src/scheduler.ts, the ramp in src/colormap.ts, the exaggeration in src/animation.ts, the number
// text in src/format.ts, what the results panel says in src/results-view.ts; this file only
// connects them to the DOM. The display name comes from the wasm's displayName() and nowhere else.
//
// There is no worker: the solve runs on the main thread in small slices (src/scheduler.ts) between
// frames, so a control change takes effect at once and the page never freezes.

import { wasmModuleUrl } from './asset-url.ts';
import { Viewer } from './viewer.ts';
import { createDropdown, createSegmented, createSlider, type Slider, type SliderSpec } from './controls.ts';
import {
  LOG_SLIDER_POSITIONS, STEP_LENGTH, STEP_SIDE, STEP_WALL, combinationOf, defaultChoice,
  logSliderPosition, logSliderValue, parseCatalog, reclamp, startArguments,
  type Catalog, type Choice, type Load, type Range, type Restraint, type Section,
} from './catalog.ts';
import { exaggerationScale } from './animation.ts';
import { rampCssGradient } from './colormap.ts';
import { exaggerationLabel, formatDuration, formatNumber, stateLabel } from './format.ts';
import { modelWarning, safetyMessage, safetyText, tableView, type Summary } from './results-view.ts';
import { StudyDriver, type Clock } from './scheduler.ts';

interface DemoInstance {
  start(...args: any[]): string;
  step(budget: number): string;
  cancel(): void;
  summaryJson(): string;
  refusal(): string | undefined;
  positions(): Float32Array;
  indices(): Uint32Array;
  vonMises(): Float32Array;
  displacements(): Float32Array;
  rotations(): Float32Array;
}

interface WasmModule {
  default: () => Promise<unknown>;
  displayName(): string;
  catalogJson(): string;
  Demo: new () => DemoInstance;
}

const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

function linearSpec(range: Range, step: number, format: (v: number) => string): SliderSpec {
  return {
    min: range.min, max: range.max, step,
    toValue: position => position,
    toPosition: value => value,
    format,
  };
}

function magnitudeSpec(load: Load, range: Range): SliderSpec {
  return {
    min: 0, max: LOG_SLIDER_POSITIONS, step: 1,
    toValue: position => logSliderValue(position, range),
    toPosition: value => logSliderPosition(value, range),
    format: value => loadText(load, value),
  };
}

function loadText(load: Load, value: number): string {
  return load === 'push' ? `${formatNumber(value)} N` : `${formatNumber(value)} N·mm`;
}

const mm = (v: number) => `${formatNumber(v, 4)} mm`;

function loadLabelFor(choice: Choice): string {
  const verb = choice.load === 'push' ? 'Push' : 'Twist';
  return `${verb} ${loadText(choice.load, choice.magnitudes[choice.load])}`;
}

async function loadWasm(): Promise<WasmModule> {
  // By URL at run time, so the bundler leaves wasm-pack's module and its .wasm beside it alone.
  const url = wasmModuleUrl(window.location.href);
  const wasm = (await import(/* webpackIgnore: true */ url)) as WasmModule;
  await wasm.default();
  return wasm;
}

function renderTable(summary: Summary) {
  const table = el('comparison');
  const view = tableView(summary);
  el('comparison-note').textContent = view.note;
  if (view.note) {
    table.innerHTML = '';
    return;
  }
  const body = view.rows.map(row => `<tr class="row-${row.status}">
      <td>${row.quantity} <small>${row.place}</small></td>
      <td class="num">${row.fea} <small>${row.unit}</small></td>
      <td class="num">${row.textbook} <small>${row.unit}</small></td>
      <td class="num difference">${row.difference}</td>
    </tr>`).join('');
  table.innerHTML = `<table>
    <thead><tr><th>Quantity and where</th><th class="num">FEA</th><th class="num">Textbook</th><th class="num">Difference</th></tr></thead>
    <tbody>${body}</tbody></table>`;
}

function setStaleness(stale: boolean) {
  el('results').classList.toggle('stale', stale);
}

async function main() {
  const wasm = await loadWasm();
  const catalog: Catalog = parseCatalog(wasm.catalogJson());
  const name = wasm.displayName();
  document.title = `${name}: Strength of a bar in your browser`;
  el('logo').textContent = name;

  const viewer = new Viewer(el('viewer'));
  const engine = new wasm.Demo();
  let choice: Choice = reclamp(catalog, defaultChoice(catalog));
  let startedAt = 0;
  let hasResult = false;
  let lastSliceMs = 0;
  let lastSummary: Summary | null = null;

  // Honest on a reduced-motion request: no animation until the visitor asks for it.
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  viewer.paused = reducedMotion;

  const sliders: Record<'length' | 'width' | 'height' | 'wall' | 'magnitude', Slider> = {
    length: createSlider('Length', linearSpec({ min: 0, max: 1 }, STEP_LENGTH, mm), choice.length, v => change(c => { c.length = v; })),
    width: createSlider('Width', linearSpec({ min: 0, max: 1 }, STEP_SIDE, mm), choice.width, v => change(c => { c.width = v; })),
    height: createSlider('Height', linearSpec({ min: 0, max: 1 }, STEP_SIDE, mm), choice.height, v => change(c => { c.height = v; })),
    wall: createSlider('Wall (share of the smaller side)', linearSpec({ min: 0, max: 1 }, STEP_WALL, formatWall), choice.wallFraction, v => change(c => { c.wallFraction = v; })),
    magnitude: createSlider('Load size', magnitudeSpec(choice.load, { min: 1, max: 2 }), choice.magnitudes[choice.load], v => change(c => { c.magnitudes[c.load] = v; })),
  };

  function formatWall(fraction: number): string {
    const thickness = fraction * Math.min(choice.width, choice.height);
    return `${Math.round(fraction * 100)} % (${formatNumber(thickness)} mm)`;
  }

  const controls = el('controls');
  controls.append(
    createSegmented('Part', [{ value: 'solid', text: 'Solid bar' }, { value: 'tube', text: 'Thick-walled tube' }], choice.section,
      v => change(c => { c.section = v as Section; }, true)).element,
    sliders.length.element, sliders.width.element, sliders.height.element, sliders.wall.element,
    createDropdown('Material', catalog.materials.map((m, i) => ({ value: String(i), text: m.name })), '0',
      v => change(c => { c.material = Number(v); }, true)),
    createSegmented('Held', [{ value: 'one_end', text: 'Fixed at one end' }, { value: 'both_ends', text: 'Fixed at both ends' }], choice.restraint,
      v => change(c => { c.restraint = v as Restraint; }, true)).element,
    createSegmented('Load', [{ value: 'push', text: 'Push' }, { value: 'twist', text: 'Twist' }], choice.load,
      v => change(c => { c.load = v as Load; }, true)).element,
    sliders.magnitude.element,
  );

  function refreshSliders() {
    const c = combinationOf(catalog, choice.section, choice.restraint, choice.load);
    sliders.length.update(linearSpec(c.length, STEP_LENGTH, mm), choice.length);
    sliders.width.update(linearSpec(c.width, STEP_SIDE, mm), choice.width);
    sliders.height.update(linearSpec(c.height, STEP_SIDE, mm), choice.height);
    if (c.wallFraction) sliders.wall.update(linearSpec(c.wallFraction, STEP_WALL, formatWall), choice.wallFraction);
    sliders.wall.element.style.display = c.wallFraction ? '' : 'none';
    sliders.magnitude.update(magnitudeSpec(choice.load, c.magnitude), choice.magnitudes[choice.load]);
  }

  /** Applies one control's change and restarts the study; `combination` re-ranges the sliders. */
  function change(apply: (c: Choice) => void, combination = false) {
    const next = { ...choice, magnitudes: { ...choice.magnitudes } };
    apply(next);
    choice = reclamp(catalog, next);
    if (combination) refreshSliders();
    else if (choice.section === 'tube') {
      // A change of width or height moves the wall's mm readout.
      const c = combinationOf(catalog, choice.section, choice.restraint, choice.load);
      if (c.wallFraction) sliders.wall.update(linearSpec(c.wallFraction, STEP_WALL, formatWall), choice.wallFraction);
    }
    setStaleness(true);
    // The warning belongs to the finished result, which this change has just left behind.
    showWarning(null);
    showState('cancelled');
    driver.request(startArguments(catalog, choice));
  }

  const clock: Clock = {
    now: () => performance.now(),
    setTimer: (fn, ms) => window.setTimeout(fn, ms),
    clearTimer: handle => window.clearTimeout(handle),
    requestFrame: fn => window.requestAnimationFrame(fn),
  };

  const driver = new StudyDriver(engine, clock, {
    onStart(state) {
      startedAt = performance.now();
      showState(state);
      if (state === 'too_large') {
        el('message').textContent = engine.refusal() ?? 'That part is too large for the demo.';
        return;
      }
      el('message').textContent = '';
      // Before there is any result, show the undeformed part in grey rather than an empty view.
      if (!hasResult) showPart(readSummary(), false);
    },
    onSlice(state, refresh, sliceMs) {
      lastSliceMs = sliceMs;
      if (refresh) {
        const summary = readSummary();
        renderResults(summary);
        if (state !== 'assembling') setStaleness(false);
        if (state === 'done' || (summary.maxStress ?? 0) > 0) {
          showPart(summary, true);
          hasResult = true;
        }
        if (state === 'failed') el('message').textContent = summary.failure ?? 'The solve failed.';
      }
      showState(state);
    },
    onError(message) {
      el('message').textContent = message;
      showState('failed');
    },
  });

  function readSummary(): Summary {
    const summary = JSON.parse(engine.summaryJson()) as Summary;
    lastSummary = summary;
    return summary;
  }

  /** Copies the engine's arrays into the viewer; a study with no stress yet draws grey. */
  function showPart(summary: Summary, colored: boolean) {
    const low = summary.minStress ?? 0;
    const peak = summary.peakStressOutsideClamps ?? 0;
    const maxStress = summary.maxStress ?? 0;
    const high = peak > low ? peak : Math.max(maxStress, low);
    const scale = exaggerationScale(summary.maxDisplacement ?? 0, choice.length, summary.maxRotation ?? 0);
    viewer.showPart({
      positions: engine.positions(),
      indices: engine.indices(),
      displacements: engine.displacements(),
      rotations: engine.rotations(),
      vonMises: colored ? engine.vonMises() : new Float32Array(0),
      rampLow: low,
      rampHigh: high,
      scale,
    }, {
      length: choice.length, width: choice.width, height: choice.height,
      restraint: choice.restraint, load: choice.load, loadLabel: loadLabelFor(choice),
    });
    el('legend-low').textContent = colored ? `${formatNumber(low)} MPa` : '—';
    el('legend-high').textContent = colored ? `${formatNumber(high)} MPa` : '—';
    el('legend').classList.toggle('muted', !colored);
    el('exaggeration').textContent = exaggerationLabel(scale, summary.maxDisplacement);
  }

  function renderResults(summary: Summary) {
    if (summary.state === 'assembling') return;
    renderTable(summary);
    el('peak-stress').textContent = `${formatNumber(summary.peakStressOutsideClamps)} MPa`;
    el('safety-factor').textContent = safetyText(summary);
    el('safety-verdict').textContent = safetyMessage(summary);
    el('results').classList.toggle('solving', summary.state !== 'done');
    showWarning(modelWarning(summary));
  }

  function showWarning(text: string | null) {
    const box = el('model-warning');
    box.textContent = text ?? '';
    box.classList.toggle('visible', text !== null);
    el('results').classList.toggle('outside-model', text !== null);
  }

  function showState(state: string) {
    const label = stateLabel(state);
    el('state').textContent = label;
    const running = state === 'assembling' || state === 'solving' || state === 'recovering' || state === 'cancelled';
    el('viewer-state').textContent = running && hasResult ? `${label}: showing the last result` : running ? `${label}…` : '';
    el('viewer-state').classList.toggle('visible', running);
    const summary = lastSummary;
    const iterations = summary && summary.iterations !== undefined && state !== 'cancelled' ? summary.iterations : 0;
    const done = state === 'done' ? `Finished in ${formatDuration(performance.now() - startedAt)}.` : '';
    el('progress').textContent = state === 'cancelled'
      ? 'Waiting for the controls to settle…'
      : `Iterations: ${iterations}. Last slice: ${lastSliceMs.toFixed(1)} ms. ${done}`.trim();
  }

  const pause = el<HTMLButtonElement>('pause');
  const showPause = () => {
    pause.textContent = viewer.paused ? 'Play the animation' : 'Pause the animation';
    pause.setAttribute('aria-pressed', String(viewer.paused));
  };
  pause.addEventListener('click', () => {
    viewer.paused = !viewer.paused;
    showPause();
  });
  showPause();

  el('legend-bar').style.background = rampCssGradient();
  refreshSliders();
  driver.request(startArguments(catalog, choice), true);
}

main().catch(e => {
  console.error(e);
  const box = document.getElementById('message');
  if (box) box.textContent = `Could not start the demo: ${String(e)}`;
});
