// Touch handling for the sliders src/controls.ts builds, so that scrolling a page on a phone with
// a finger that happens to land on a slider scrolls the page instead of changing a setting.
//
// On a touch screen (`pointer: coarse`, styles/main.css) the native range input takes no pointer
// input at all and its wrapper, `touch-action: pan-y`, takes it instead: the browser keeps
// vertical panning, and `attachTouchDrag` watches each touch from where it went down. Until the
// finger has moved `DRAG_START_PX` sideways, at least `DRAG_RATIO` times as far as it moved up or
// down, nothing changes; a gesture that goes vertical first is a scroll (the browser takes it and
// cancels the pointer) and a lift before either is a tap, which changes nothing. Once it is a
// drag the value follows the finger, and the wrapper raises the same `input` and `change` events
// on the input that a native drag would, so the pages' listeners need nothing new. A mouse on a
// touch-screen device drags at once, as a native slider would. With a fine pointer (a desktop)
// the input keeps its native mouse and keyboard handling and the wrapper does nothing.
//
// The decision (`gestureOf`) and the value under the finger (`valueAt`) are plain functions,
// tested in src/slider-touch.test.ts.

/** Sideways movement, in CSS px, after which a touch can become a drag. */
export const DRAG_START_PX = 10;
/** How many times its vertical movement a touch must have moved sideways to become a drag. */
export const DRAG_RATIO = 2;
/** The thumb's width in CSS px (styles/main.css), so the ends of the track match a native one. */
const THUMB_PX = 16;

export type Gesture = 'undecided' | 'drag' | 'scroll';

/** What a touch that has moved (dx, dy) px from where it went down is. */
export function gestureOf(dx: number, dy: number): Gesture {
  const ax = Math.abs(dx);
  const ay = Math.abs(dy);
  if (ax >= DRAG_START_PX && ax >= DRAG_RATIO * ay) return 'drag';
  if (ay >= DRAG_START_PX && ay > ax / DRAG_RATIO) return 'scroll';
  return 'undecided';
}

/**
 * The slider value under a pointer at `x` over a track from `left`, `width` px wide: the thumb's
 * centre runs from half a thumb in from either end, as on a native slider, and the value snaps to
 * `step` from `min`.
 */
export function valueAt(x: number, left: number, width: number, min: number, max: number,
  step: number): number {
  const run = Math.max(width - THUMB_PX, 1);
  const t = Math.min(Math.max((x - left - THUMB_PX / 2) / run, 0), 1);
  const raw = min + t * (max - min);
  const snapped = step > 0 ? min + Math.round((raw - min) / step) * step : raw;
  // a step like 0.25 leaves no float noise on the label; the clamp keeps rounding inside the range
  const decimals = (String(step).split('.')[1] ?? '').length;
  return Math.min(Math.max(Number(snapped.toFixed(decimals)), min), max);
}

/** Lets `hold` (the wrapper of `input`) drag the input by touch only once a touch is a drag. */
export function attachTouchDrag(hold: HTMLElement, input: HTMLInputElement): void {
  let start: { id: number; x: number; y: number } | null = null;
  let gesture: Gesture = 'undecided';
  let changed = false;

  const setFrom = (x: number) => {
    const rect = input.getBoundingClientRect();
    const v = valueAt(x, rect.left, rect.width, Number(input.min), Number(input.max),
      Number(input.step));
    if (String(v) === input.value) return;
    input.value = String(v);
    changed = true;
    input.dispatchEvent(new Event('input', { bubbles: true }));
  };
  // keeps the drag when the finger leaves the track; a pointer the browser no longer tracks
  // cannot be captured, which must not stop the value following it
  const capture = (id: number) => {
    try { hold.setPointerCapture(id); } catch { /* not an active pointer */ }
  };
  const end = () => {
    if (changed) input.dispatchEvent(new Event('change', { bubbles: true }));
    start = null;
    gesture = 'undecided';
    changed = false;
  };

  hold.addEventListener('pointerdown', e => {
    // the input itself took the event: a fine pointer, handled natively
    if (e.target === input || start) return;
    start = { id: e.pointerId, x: e.clientX, y: e.clientY };
    gesture = e.pointerType === 'mouse' ? 'drag' : 'undecided';
    if (gesture === 'drag') {
      capture(e.pointerId);
      setFrom(e.clientX);
    }
  });
  hold.addEventListener('pointermove', e => {
    if (!start || e.pointerId !== start.id || gesture === 'scroll') return;
    if (gesture === 'undecided') {
      gesture = gestureOf(e.clientX - start.x, e.clientY - start.y);
      if (gesture !== 'drag') return;
      capture(e.pointerId);
      input.focus({ preventScroll: true });
    }
    setFrom(e.clientX);
  });
  hold.addEventListener('pointerup', e => {
    if (start && e.pointerId === start.id) end();
  });
  hold.addEventListener('pointercancel', e => {
    if (start && e.pointerId === start.id) end();
  });
}
