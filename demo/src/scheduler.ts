// The time-aware scheduler that keeps the page responsive with the solve on the main thread:
// `BudgetController` picks how much work each animation frame may give to `Demo.step`, and
// `StudyDriver` runs the study (start, restart on change, one slice per frame). Neither touches
// the DOM or the wasm directly (an engine and a clock are injected), so src/scheduler.test.ts
// runs the same code with a fake clock.
//
// Why a budget per engine phase: one unit means different work in each (elements while assembling
// or recovering, iterations while solving), so a budget learned in one phase says nothing about
// the next. Each phase starts low and grows only while slices stay under target. A slice that
// runs long cuts the budget at once, straight to what would have hit the target, so one overlong
// frame never repeats.
//
// A control change cancels the running study at once and restarts after a short debounce: while
// a slider is dragged, starts coalesce and no solve time is wasted on positions already left
// behind.

/** The slice length the controller steers towards, in ms. */
export const TARGET_SLICE_MS = 8;
/** A slice longer than this is an overrun and cuts the budget at once. */
export const CEILING_SLICE_MS = 16;
/** Every phase's first slice uses this budget, so a slow machine's first frame stays short. */
export const INITIAL_BUDGET = 3;
/** The largest change upward per slice: a burst of quick slices must not jump the budget. */
const MAX_GROWTH = 1.5;
/** Far above any study's iteration count, so a timer glitch cannot produce a runaway budget. */
const MAX_BUDGET = 20_000;
/** How long after the last control change a restart waits, in ms. */
export const DEBOUNCE_MS = 120;
/** How often a running study's result is pushed to the view, in ms. */
export const VIEW_REFRESH_MS = 150;

export class BudgetController {
  private budgets = new Map<string, number>();

  /** The budget for the next slice of `phase`. */
  budgetFor(phase: string): number {
    return this.budgets.get(phase) ?? INITIAL_BUDGET;
  }

  /** Learns from a finished slice of `phase` that used `budget` and took `elapsedMs`. */
  record(phase: string, budget: number, elapsedMs: number): void {
    const costPerUnit = elapsedMs / Math.max(budget, 1);
    let next: number;
    if (costPerUnit <= 0 || !Number.isFinite(costPerUnit)) {
      // The clock's resolution saw nothing: the slice was tiny, so grow.
      next = budget * MAX_GROWTH;
    } else {
      const toTarget = TARGET_SLICE_MS / costPerUnit;
      // An overrun takes the target directly; otherwise growth is limited per slice.
      next = elapsedMs > CEILING_SLICE_MS ? toTarget : Math.min(toTarget, budget * MAX_GROWTH);
    }
    this.budgets.set(phase, Math.max(1, Math.min(MAX_BUDGET, Math.floor(next))));
  }

  /** Forgets everything learned: a new study has its own mesh and costs. */
  reset(): void {
    this.budgets.clear();
  }
}

/** The engine calls the driver makes; `Demo` from the wasm package satisfies this. */
export interface Engine {
  start(...args: any[]): string;
  step(budget: number): string;
  cancel(): void;
}

export interface Clock {
  now(): number;
  setTimer(fn: () => void, ms: number): number;
  clearTimer(handle: number): void;
  requestFrame(fn: () => void): number;
}

export interface DriverHooks {
  /** A study began: state is what `start` returned. */
  onStart(state: string): void;
  /** Called after a slice with the new state; `refresh` says the view should be updated. */
  onSlice(state: string, refresh: boolean, sliceMs: number): void;
  /** A start was refused or threw. */
  onError(message: string): void;
}

const RUNNING = new Set(['assembling', 'solving', 'recovering']);

export class StudyDriver {
  readonly controller = new BudgetController();
  private state = 'idle';
  private frameRequested = false;
  private timer: number | null = null;
  private lastRefresh = -Infinity;
  private generation = 0;

  constructor(private engine: Engine, private clock: Clock, private hooks: DriverHooks) {}

  /** The latest state of the study. */
  get currentState(): string {
    return this.state;
  }

  /** Whether a study is mid-solve. */
  get isRunning(): boolean {
    return RUNNING.has(this.state);
  }

  /**
   * A control changed: stops the running study now and starts `args` after the debounce, or at
   * once with `immediate` (the first study on load).
   */
  request(args: unknown[], immediate = false): void {
    this.generation += 1;
    this.engine.cancel();
    this.state = 'cancelled';
    if (this.timer !== null) {
      this.clock.clearTimer(this.timer);
      this.timer = null;
    }
    if (immediate) {
      this.begin(args);
      return;
    }
    const generation = this.generation;
    this.timer = this.clock.setTimer(() => {
      this.timer = null;
      if (generation === this.generation) this.begin(args);
    }, DEBOUNCE_MS);
  }

  private begin(args: unknown[]): void {
    this.controller.reset();
    this.lastRefresh = -Infinity;
    try {
      this.state = this.engine.start(...args);
    } catch (e) {
      this.state = 'failed';
      this.hooks.onError(String(e));
      return;
    }
    this.hooks.onStart(this.state);
    this.schedule();
  }

  private schedule(): void {
    if (!RUNNING.has(this.state) || this.frameRequested) return;
    this.frameRequested = true;
    this.clock.requestFrame(() => {
      this.frameRequested = false;
      this.frame();
    });
  }

  /** One animation frame: one slice, then the next frame if the study is still going. */
  private frame(): void {
    if (!RUNNING.has(this.state)) return;
    const phase = this.state;
    const budget = this.controller.budgetFor(phase);
    const started = this.clock.now();
    const state = this.engine.step(budget);
    const elapsed = this.clock.now() - started;
    this.controller.record(phase, budget, elapsed);
    this.state = state;
    const finished = !RUNNING.has(state);
    const refresh = finished || this.clock.now() - this.lastRefresh >= VIEW_REFRESH_MS;
    if (refresh) this.lastRefresh = this.clock.now();
    this.hooks.onSlice(state, refresh, elapsed);
    this.schedule();
  }
}
