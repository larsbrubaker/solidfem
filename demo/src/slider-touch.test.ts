// Tests for src/slider-touch.ts's touch rule: a touch on a slider changes nothing until it is
// clearly a sideways drag, a mostly vertical one is left to scroll the page, a tap changes
// nothing, and the value under the finger matches a native slider's track. Run with `bun test`
// from demo/; the functions are pure, so no DOM is involved.

import { describe, expect, test } from 'bun:test';
import { gestureOf, valueAt } from './slider-touch.ts';

/** What a touch that went through `points` (offsets from where it went down) ends up as. */
function gestureAlong(points: [number, number][]): string {
  let gesture = 'undecided';
  for (const [dx, dy] of points) {
    if (gesture !== 'undecided') break;
    gesture = gestureOf(dx, dy);
  }
  return gesture;
}

describe('gestureOf', () => {
  test('a tap, or a small wobble, is undecided and changes nothing', () => {
    expect(gestureAlong([])).toBe('undecided');
    expect(gestureAlong([[2, 1], [-3, 4], [6, -5]])).toBe('undecided');
  });

  test('a mostly vertical swipe is a scroll, even with some sideways drift', () => {
    expect(gestureAlong([[1, 4], [3, 9], [5, 16], [8, 40]])).toBe('scroll');
    expect(gestureAlong([[0, -12]])).toBe('scroll');
  });

  test('a clearly sideways move is a drag once it passes the threshold', () => {
    expect(gestureAlong([[4, 0], [8, 1]])).toBe('undecided');
    expect(gestureAlong([[4, 0], [8, 1], [12, 2]])).toBe('drag');
    expect(gestureAlong([[-15, 3]])).toBe('drag');
  });

  test('a diagonal that is not clearly sideways never becomes a drag', () => {
    expect(gestureAlong([[12, 8], [20, 14], [40, 30]])).toBe('scroll');
  });

  test('a swipe that went vertical first stays a scroll when it turns sideways', () => {
    expect(gestureAlong([[0, 12], [60, 12]])).toBe('scroll');
  });
});

describe('valueAt', () => {
  // a 216 px track at x = 100: the thumb centre runs from 108 to 308
  test('the ends of the track are the ends of the range', () => {
    expect(valueAt(100, 100, 216, 0, 10, 1)).toBe(0);
    expect(valueAt(108, 100, 216, 0, 10, 1)).toBe(0);
    expect(valueAt(308, 100, 216, 0, 10, 1)).toBe(10);
    expect(valueAt(400, 100, 216, 0, 10, 1)).toBe(10);
  });

  test('the value snaps to the step, without float noise', () => {
    expect(valueAt(208, 100, 216, 0, 10, 1)).toBe(5);
    expect(valueAt(208, 100, 216, 0.25, 10, 0.25)).toBe(5.25);
    expect(valueAt(150, 100, 216, 0.1, 0.9, 0.1)).toBe(0.3);
  });
});
