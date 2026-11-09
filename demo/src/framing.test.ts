// Tests of the camera decisions in src/framing.ts, in the orders a page load can produce them: the
// part stays the normal size on screen however late the view's real size arrives, and resizing
// never undoes what the visitor did with the camera.

import { describe, expect, test } from 'bun:test';
import { CameraFraming, MIN_VIEWPORT_PIXELS, fitDistance, boundsOfPositions, clearRect, partBounds, viewShift, type CameraPose, type Vec3 } from './framing.ts';

const FOV = 36;
const SETUP = { length: 100, width: 10, height: 20, restraint: 'one_end' };
const BOUNDS = partBounds(SETUP);
const START: CameraPose = { position: [0, 0, 0], target: [0, 0, 0] };
// A distance worked out here from the field of view alone, so the expected value does not come
// from the code under test: a sphere fits when its radius subtends half the vertical field.
const NORMAL_DISTANCE = (1.02 * BOUNDS.radius) / Math.sin((18 * Math.PI) / 180);
// Doubles carry 15 to 16 significant digits; these distances are products of a few of them.
const DIGITS = 9;

function distance(pose: CameraPose): number {
  return Math.hypot(pose.position[0] - pose.target[0], pose.position[1] - pose.target[1], pose.position[2] - pose.target[2]);
}

function direction(pose: CameraPose): Vec3 {
  const d = distance(pose);
  return [(pose.position[0] - pose.target[0]) / d, (pose.position[1] - pose.target[1]) / d, (pose.position[2] - pose.target[2]) / d];
}

function expectSameDirection(a: CameraPose, b: CameraPose) {
  const [ax, ay, az] = direction(a);
  const [bx, by, bz] = direction(b);
  expect(ax).toBeCloseTo(bx, DIGITS);
  expect(ay).toBeCloseTo(by, DIGITS);
  expect(az).toBeCloseTo(bz, DIGITS);
}

/** A framing that has placed the part in a real landscape view, and the resulting pose. */
function framedInLandscape(): { framing: CameraFraming; pose: CameraPose } {
  const framing = new CameraFraming(FOV);
  framing.viewportChanged(1200, 700, START);
  const pose = framing.showPart(BOUNDS, START);
  if (!pose) throw new Error('the first part in a real view is placed at once');
  return { framing, pose };
}

/** The same pose seen from a different side, as after a drag: same target, same distance. */
function orbited(pose: CameraPose): CameraPose {
  const d = distance(pose);
  const unit = Math.hypot(-0.6, 0.7, 0.2);
  return {
    target: pose.target,
    position: [pose.target[0] + (-0.6 / unit) * d, pose.target[1] + (0.7 / unit) * d, pose.target[2] + (0.2 / unit) * d],
  };
}

describe('fitDistance', () => {
  test('a landscape view fits the part up its height, at any width', () => {
    expect(fitDistance(BOUNDS.radius, FOV, 1)).toBeCloseTo(NORMAL_DISTANCE, DIGITS);
    expect(fitDistance(BOUNDS.radius, FOV, 3.5)).toBeCloseTo(NORMAL_DISTANCE, DIGITS);
  });

  test('a portrait view backs away in proportion, to fit the part across', () => {
    expect(fitDistance(BOUNDS.radius, FOV, 0.5)).toBeCloseTo(2 * NORMAL_DISTANCE, DIGITS);
  });
});

describe('a part arriving before the view has a real size', () => {
  test('a sliver of a container followed by the real size ends at the normal distance', () => {
    const framing = new CameraFraming(FOV);
    expect(framing.viewportChanged(1, 800, START)).toEqual({ viewport: null, placement: null });
    expect(framing.showPart(BOUNDS, START)).toBeNull();
    expect(framing.waitingForViewport).toBe(true);

    expect(framing.viewportChanged(1, 800, START).placement).toBeNull();
    const { viewport, placement } = framing.viewportChanged(1200, 700, START);
    expect(viewport).toEqual({ width: 1200, height: 700 });
    expect(framing.waitingForViewport).toBe(false);
    expect(distance(placement!)).toBeCloseTo(NORMAL_DISTANCE, DIGITS);
    expect(placement!.target).toEqual(BOUNDS.centre);
    expect(placement!.near).toBeCloseTo(NORMAL_DISTANCE / 100, DIGITS);
    expect(placement!.far).toBeCloseTo(NORMAL_DISTANCE * 10, DIGITS);
  });

  test('a size just big enough to count is used, and one just below is not', () => {
    const framing = new CameraFraming(FOV);
    expect(framing.viewportChanged(MIN_VIEWPORT_PIXELS - 1, 500, START).viewport).toBeNull();
    expect(framing.viewportChanged(500, MIN_VIEWPORT_PIXELS - 1, START).viewport).toBeNull();
    expect(framing.viewportChanged(MIN_VIEWPORT_PIXELS, MIN_VIEWPORT_PIXELS, START).viewport).not.toBeNull();
  });

  test('a part placed against a passing tall narrow layout returns to the normal distance', () => {
    const framing = new CameraFraming(FOV);
    framing.viewportChanged(100, 3000, START);
    const far = framing.showPart(BOUNDS, START)!;
    expect(distance(far)).toBeCloseTo(30 * NORMAL_DISTANCE, DIGITS);

    const settled = framing.viewportChanged(1200, 700, far).placement!;
    expect(distance(settled)).toBeCloseTo(NORMAL_DISTANCE, DIGITS);
    expect(settled.near).toBeCloseTo(NORMAL_DISTANCE / 100, DIGITS);
    expectSameDirection(settled, far);
  });
});

describe('resizing before the visitor has touched the camera', () => {
  test('going portrait re-frames to fit, and going back returns to the normal distance', () => {
    const { framing, pose } = framedInLandscape();
    const portrait = framing.viewportChanged(400, 800, pose).placement!;
    expect(distance(portrait)).toBeCloseTo(2 * NORMAL_DISTANCE, DIGITS);
    expectSameDirection(portrait, pose);

    const landscape = framing.viewportChanged(1200, 700, portrait).placement!;
    expect(distance(landscape)).toBeCloseTo(NORMAL_DISTANCE, DIGITS);
    expectSameDirection(landscape, pose);
  });

  test('a landscape resize leaves the camera exactly where it is', () => {
    const { framing, pose } = framedInLandscape();
    expect(framing.viewportChanged(900, 600, pose)).toEqual({ viewport: { width: 900, height: 600 }, placement: null });
  });
});

describe('resizing after the visitor has moved the camera', () => {
  test('a landscape resize moves nothing the visitor set', () => {
    const { framing, pose } = framedInLandscape();
    expect(framing.viewportChanged(900, 600, orbited(pose)).placement).toBeNull();
  });

  test('going portrait keeps the direction and only backs away enough to fit', () => {
    const { framing, pose } = framedInLandscape();
    const moved = orbited(pose);
    const portrait = framing.viewportChanged(400, 800, moved).placement!;
    expectSameDirection(portrait, moved);
    expect(portrait.target).toEqual(moved.target);
    expect(distance(portrait)).toBeCloseTo(2 * distance(moved), DIGITS);
  });

  test('a zoomed and panned view keeps its pan and its zoom relative to the fit', () => {
    const { framing, pose } = framedInLandscape();
    const seen = orbited(pose);
    const target: Vec3 = [seen.target[0] + 5, seen.target[1] - 3, seen.target[2] + 2];
    const [dx, dy, dz] = direction(seen);
    const half = 0.5 * distance(seen);
    const zoomed: CameraPose = { target, position: [target[0] + dx * half, target[1] + dy * half, target[2] + dz * half] };

    const portrait = framing.viewportChanged(400, 800, zoomed).placement!;
    expect(portrait.target).toEqual(target);
    expectSameDirection(portrait, zoomed);
    expect(distance(portrait)).toBeCloseTo(NORMAL_DISTANCE, DIGITS);
  });

  test('a view hidden for a while comes back unchanged', () => {
    const { framing, pose } = framedInLandscape();
    const moved = orbited(pose);
    expect(framing.viewportChanged(0, 0, moved)).toEqual({ viewport: null, placement: null });
    expect(framing.viewportChanged(1200, 700, moved).placement).toBeNull();
  });
});

describe('showing parts', () => {
  test('the same size again never moves the camera', () => {
    const { framing, pose } = framedInLandscape();
    expect(framing.showPart(partBounds(SETUP), orbited(pose))).toBeNull();
  });

  test('a new size keeps the visitor\'s direction and fits the new part', () => {
    const { framing, pose } = framedInLandscape();
    const moved = orbited(pose);
    const bigger = partBounds({ ...SETUP, width: 30 });
    const placed = framing.showPart(bigger, { position: moved.position, target: bigger.centre })!;
    expect(placed.target).toEqual(bigger.centre);
    expect(distance(placed)).toBeCloseTo((1.02 * bigger.radius) / Math.sin((18 * Math.PI) / 180), DIGITS);
    expectSameDirection(placed, { position: moved.position, target: bigger.centre });
  });

  test('a bar\'s bounds cover its diagonal and the room for its plates and arrow', () => {
    expect(BOUNDS.centre).toEqual([50, 5, 10]);
    expect(BOUNDS.radius).toBeCloseTo(0.5 * Math.sqrt(100 * 100 + 10 * 10 + 20 * 20) + 2 * 20, DIGITS);
  });
});

describe('the bounds of an example part', () => {
  test('are centred on its surface and enclose it', () => {
    const bounds = boundsOfPositions('hook', new Float32Array([0, 0, 0, 2, 4, 6]));
    expect(bounds.key).toBe('hook');
    expect(bounds.centre).toEqual([1, 2, 3]);
    expect(bounds.radius).toBeGreaterThanOrEqual(0.5 * Math.hypot(2, 4, 6));
  });
});

describe('a part beside the legend', () => {
  const FOV = 36;
  const legend = { left: 10, top: 10, right: 200, bottom: 190 };
  const strip = (w: number, h: number) => ({ left: 10, top: h - 50, right: w - 10, bottom: h - 10 });
  const pose = { position: [0, -10, 0] as const, target: [0, 0, 0] as const };
  /** The part's sphere on screen, in the clear rectangle: its centre in pixels and its radius. */
  function onScreen(w: number, h: number, obstructions: { left: number; top: number; right: number; bottom: number }[], radius: number) {
    const framing = new CameraFraming(FOV);
    framing.viewportChanged(w, h, pose, obstructions);
    const placement = framing.showPart({ key: 'k', centre: [0, 0, 0], radius }, pose)!;
    const distance = Math.hypot(...placement.position);
    const pixels = (radius / (distance * Math.tan((FOV * Math.PI) / 360))) * (h / 2);
    return { x: w / 2 - framing.viewOffset.x, y: h / 2 - framing.viewOffset.y, pixels };
  }
  function clearOf(circle: { x: number; y: number; pixels: number }, r: { left: number; top: number; right: number; bottom: number }): boolean {
    return circle.x + circle.pixels <= r.left || circle.x - circle.pixels >= r.right
      || circle.y + circle.pixels <= r.top || circle.y - circle.pixels >= r.bottom;
  }
  // Non-bar proportions: a long flat plate, a compact block, a small clip.
  const parts = [
    new Float32Array([0, 0, 0, 300, 100, 2]),
    new Float32Array([-10, 24, 0, 10, 44, 20]),
    new Float32Array([0, 0, 0, 20, 5, 2]),
  ];

  for (const [name, w, h] of [['a wide desktop view', 1100, 620], ['a phone', 360, 320], ['a portrait phone', 360, 640]] as const) {
    test(`stays clear of the legend and the strip on ${name}`, () => {
      const obstructions = [legend, strip(w, h)];
      for (const positions of parts) {
        const bounds = boundsOfPositions('p', positions);
        const circle = onScreen(w, h, obstructions, bounds.radius);
        expect(clearOf(circle, legend)).toBe(true);
        expect(clearOf(circle, strip(w, h))).toBe(true);
        expect(circle.x - circle.pixels).toBeGreaterThanOrEqual(0);
        expect(circle.x + circle.pixels).toBeLessThanOrEqual(w);
        expect(circle.y - circle.pixels).toBeGreaterThanOrEqual(0);
        expect(circle.y + circle.pixels).toBeLessThanOrEqual(h);
      }
    });
  }

  test('an obstruction outside the view changes nothing', () => {
    const viewport = { width: 800, height: 600 };
    const clear = clearRect(viewport, [{ left: 900, top: 0, right: 950, bottom: 40 }]);
    expect(clear).toEqual({ left: 0, top: 0, right: 800, bottom: 600 });
    expect(viewShift(viewport, clear)).toEqual({ x: 0, y: 0 });
  });

  test('a part with its own direction opens from it, even after the visitor has orbited another', () => {
    const framing = new CameraFraming(FOV);
    framing.viewportChanged(800, 600, pose);
    framing.showPart({ key: 'a', centre: [0, 0, 0], radius: 5 }, pose);
    const placed = framing.showPart({ key: 'b', centre: [0, 0, 0], radius: 5, direction: [0, 0, 1] }, pose)!;
    expect(placed.position[0]).toBeCloseTo(0, 9);
    expect(placed.position[1]).toBeCloseTo(0, 9);
    expect(placed.position[2]).toBeGreaterThan(0);
  });
});
