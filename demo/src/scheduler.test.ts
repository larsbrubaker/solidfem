// Tests of src/scheduler.ts with a fake clock and engine: the budget adapts to hold a slice near
// the target and recovers from an overrun at once, and a control change cancels the running
// solve immediately, coalesces while dragging and restarts with fresh budgets.

import { describe, expect, test } from 'bun:test';
import {
  BudgetController, CEILING_SLICE_MS, DEBOUNCE_MS, INITIAL_BUDGET, StudyDriver, TARGET_SLICE_MS,
  VIEW_REFRESH_MS, type Clock, type Engine,
} from './scheduler.ts';

describe('BudgetController', () => {
  test('starts small, at a budget of 5 or less', () => {
    expect(INITIAL_BUDGET).toBeLessThanOrEqual(5);
    expect(new BudgetController().budgetFor('solving')).toBe(INITIAL_BUDGET);
  });

  test('converges so a slice takes about the target time', () => {
    const controller = new BudgetController();
    const costPerUnit = 0.05;
    let elapsed = 0;
    for (let i = 0; i < 40; i++) {
      const budget = controller.budgetFor('solving');
      elapsed = budget * costPerUnit;
      controller.record('solving', budget, elapsed);
    }
    expect(elapsed).toBeGreaterThan(TARGET_SLICE_MS * 0.8);
    expect(elapsed).toBeLessThanOrEqual(TARGET_SLICE_MS * 1.05);
  });

  test('an overrun cuts straight to the budget that would have met the target', () => {
    const controller = new BudgetController();
    controller.record('solving', 200, 50);
    expect(controller.budgetFor('solving')).toBe(32);
    expect(32 * (50 / 200)).toBeLessThanOrEqual(TARGET_SLICE_MS);
  });

  test('a very slow machine still gets one unit of progress per frame', () => {
    const controller = new BudgetController();
    controller.record('assembling', 1, 500);
    expect(controller.budgetFor('assembling')).toBe(1);
  });

  test('growth is limited per slice, so a quick burst cannot jump past the ceiling', () => {
    const controller = new BudgetController();
    controller.record('solving', 10, 0.5);
    expect(controller.budgetFor('solving')).toBe(15);
  });

  test('a clock too coarse to see the slice still grows the budget', () => {
    const controller = new BudgetController();
    controller.record('solving', 10, 0);
    expect(controller.budgetFor('solving')).toBe(15);
  });

  test('each engine phase learns its own budget', () => {
    const controller = new BudgetController();
    controller.record('assembling', 3, CEILING_SLICE_MS * 3);
    controller.record('solving', 3, 1);
    expect(controller.budgetFor('assembling')).toBeLessThan(controller.budgetFor('solving'));
    expect(controller.budgetFor('recovering')).toBe(INITIAL_BUDGET);
  });

  test('reset forgets what the old mesh taught it', () => {
    const controller = new BudgetController();
    controller.record('solving', 3, 1);
    controller.reset();
    expect(controller.budgetFor('solving')).toBe(INITIAL_BUDGET);
  });
});

class FakeClock implements Clock {
  time = 0;
  private timers = new Map<number, { at: number; fn: () => void }>();
  private frames: (() => void)[] = [];
  private nextId = 1;

  now(): number {
    return this.time;
  }

  setTimer(fn: () => void, ms: number): number {
    const id = this.nextId++;
    this.timers.set(id, { at: this.time + ms, fn });
    return id;
  }

  clearTimer(handle: number): void {
    this.timers.delete(handle);
  }

  requestFrame(fn: () => void): number {
    this.frames.push(fn);
    return 0;
  }

  /** Moves time forward and runs timers that come due. */
  advance(ms: number): void {
    this.time += ms;
    for (const [id, timer] of [...this.timers]) {
      if (timer.at <= this.time) {
        this.timers.delete(id);
        timer.fn();
      }
    }
  }

  /** Runs the frames scheduled so far, as one animation frame. */
  runFrame(): void {
    const frames = this.frames;
    this.frames = [];
    for (const frame of frames) frame();
  }

  get pendingFrames(): number {
    return this.frames.length;
  }
}

class FakeEngine implements Engine {
  starts: unknown[][] = [];
  cancels = 0;
  steps: number[] = [];
  private script: string[] = [];
  private clock: FakeClock;
  sliceMs = 2;

  constructor(clock: FakeClock, script: string[]) {
    this.clock = clock;
    this.script = script;
  }

  start(...args: unknown[]): string {
    this.starts.push(args);
    return 'assembling';
  }

  step(budget: number): string {
    this.steps.push(budget);
    this.clock.time += this.sliceMs;
    return this.script.shift() ?? 'solving';
  }

  cancel(): void {
    this.cancels++;
  }
}

function setup(script: string[] = []) {
  const clock = new FakeClock();
  const engine = new FakeEngine(clock, script);
  const log: { slices: { state: string; refresh: boolean }[]; starts: string[]; errors: string[] } = {
    slices: [], starts: [], errors: [],
  };
  const driver = new StudyDriver(engine, clock, {
    onStart: state => log.starts.push(state),
    onSlice: (state, refresh) => log.slices.push({ state, refresh }),
    onError: message => log.errors.push(message),
  });
  return { clock, engine, driver, log };
}

describe('StudyDriver', () => {
  test('the first study starts at once and steps one slice per frame', () => {
    const { clock, engine, driver } = setup();
    driver.request(['first'], true);
    expect(engine.starts).toEqual([['first']]);
    expect(engine.steps.length).toBe(0);
    clock.runFrame();
    expect(engine.steps.length).toBe(1);
    expect(engine.steps[0]).toBeLessThanOrEqual(5);
    clock.runFrame();
    expect(engine.steps.length).toBe(2);
  });

  test('a control change cancels the running solve at once and no slice runs in the debounce', () => {
    const { clock, engine, driver } = setup();
    driver.request(['first'], true);
    clock.runFrame();
    driver.request(['second']);
    expect(engine.cancels).toBeGreaterThan(0);
    const stepsAtChange = engine.steps.length;
    clock.runFrame();
    clock.advance(DEBOUNCE_MS - 1);
    clock.runFrame();
    expect(engine.steps.length).toBe(stepsAtChange);
    expect(engine.starts.length).toBe(1);
  });

  test('dragging a slider restarts only once, with the last value, after it settles', () => {
    const { clock, engine, driver } = setup();
    driver.request(['first'], true);
    for (let i = 0; i < 10; i++) {
      driver.request(['drag', i]);
      clock.advance(DEBOUNCE_MS / 4);
    }
    expect(engine.starts.length).toBe(1);
    clock.advance(DEBOUNCE_MS);
    expect(engine.starts.length).toBe(2);
    expect(engine.starts[1]).toEqual(['drag', 9]);
  });

  test('a restart begins with fresh, small budgets', () => {
    const { clock, engine, driver } = setup();
    engine.sliceMs = 1;
    driver.request(['first'], true);
    for (let i = 0; i < 12; i++) clock.runFrame();
    expect(engine.steps[engine.steps.length - 1]).toBeGreaterThan(INITIAL_BUDGET);
    driver.request(['second']);
    clock.advance(DEBOUNCE_MS);
    clock.runFrame();
    expect(engine.steps[engine.steps.length - 1]).toBe(INITIAL_BUDGET);
  });

  test('the view refresh is throttled, and the final state always refreshes', () => {
    const { clock, driver, log } = setup(['solving', 'solving', 'solving', 'done']);
    driver.request(['first'], true);
    clock.runFrame();
    clock.time += VIEW_REFRESH_MS / 10;
    clock.runFrame();
    clock.time += VIEW_REFRESH_MS / 10;
    clock.runFrame();
    expect(log.slices.map(s => s.refresh)).toEqual([true, false, false]);
    clock.time += VIEW_REFRESH_MS / 10;
    clock.runFrame();
    expect(log.slices[log.slices.length - 1]).toEqual({ state: 'done', refresh: true });
  });

  test('a finished study stops asking for frames', () => {
    const { clock, engine, driver } = setup(['done']);
    driver.request(['first'], true);
    clock.runFrame();
    expect(engine.steps.length).toBe(1);
    expect(clock.pendingFrames).toBe(0);
  });

  test('a start that throws reports the message and does not step', () => {
    const { clock, engine, driver, log } = setup();
    engine.start = () => {
      throw new Error('length 9000 is outside the allowed range');
    };
    driver.request(['bad'], true);
    clock.runFrame();
    expect(log.errors.length).toBe(1);
    expect(log.errors[0]).toContain('outside the allowed range');
    expect(engine.steps.length).toBe(0);
  });

  test('a too-large refusal does not step', () => {
    const { clock, engine, driver, log } = setup();
    engine.start = () => 'too_large';
    driver.request(['big'], true);
    clock.runFrame();
    expect(log.starts).toEqual(['too_large']);
    expect(engine.steps.length).toBe(0);
  });
});

describe('stopping a study for a part change', () => {
  test('halt stops the slices and never updates the view again', () => {
    const calls: string[] = [];
    let frame: (() => void) | null = null;
    const clock = {
      now: () => 0,
      setTimer: () => 1,
      clearTimer: () => {},
      requestFrame: (fn: () => void) => { frame = fn; return 1; },
    };
    const engine = {
      start: () => 'meshing',
      step: () => { calls.push('step'); return 'meshing'; },
      cancel: () => { calls.push('cancel'); },
    };
    const driver = new StudyDriver(engine, clock, {
      onStart: () => {},
      onSlice: () => { calls.push('slice'); },
      onError: () => {},
    });
    driver.request([], true);
    driver.halt();
    expect(driver.currentState).toBe('cancelled');
    (frame as unknown as () => void)();
    expect(calls).toEqual(['cancel', 'cancel']);
  });
});
