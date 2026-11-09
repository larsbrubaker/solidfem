// Where the viewer's camera goes, as plain numbers so `bun test` can run it without a browser or a
// GPU; src/viewer.ts applies what this decides to the Three.js camera and controls.
//
// The view's real size is not known when the page starts: the container can report a sliver, or a
// shape it holds only until the layout settles, and the first result can arrive during that time.
// Two rules keep the part filling the view whatever order those things happen in:
//
// - A size too small to be a real view is ignored, and a part that arrives before any real size
//   waits for one instead of being placed against a meaningless shape.
// - When the view's shape changes, the camera's distance is multiplied by the change in the
//   distance that fits the part. An untouched camera sits exactly at the fitting distance, so it
//   lands exactly on the new one; one the visitor has orbited, panned or zoomed keeps its direction,
//   its target and its zoom relative to the fit. A change that leaves the fitting distance alone,
//   like any resize of a landscape view, moves nothing.
//
// The legend and the exaggeration strip sit on top of the view, so the part is fitted to what they
// leave clear: the largest rectangle beside them (`clearRect`), with the view shifted so the part's
// centre lands in the middle of it (`viewShift`). The part is held in a sphere, so this holds for
// any shape of part, and a phone's narrow view gives up the same room as a wide one.

export type Vec3 = readonly [number, number, number];

/** Where the camera is and what it orbits around. */
export interface CameraPose {
  position: Vec3;
  target: Vec3;
}

/** A pose to move the camera to, with the clip planes that suit it. */
export interface CameraPlacement extends CameraPose {
  near: number;
  far: number;
}

/** The sphere a part and its markings need to stay inside the view. */
export interface PartBounds {
  /** Equal for two parts of the same size, which share a placement. */
  key: string;
  centre: Vec3;
  radius: number;
  /** From the part towards the camera for this part, overriding the visitor's last direction. */
  direction?: Vec3;
}

export interface Viewport {
  width: number;
  height: number;
}

/** A rectangle of the view in CSS pixels from its top-left corner. */
export interface Rect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/** How far to shift the drawn view so the part's centre lands in the clear rectangle's middle. */
export interface ViewShift {
  x: number;
  y: number;
}

/** What a size report changes: the size to draw at (null to ignore it) and any camera move. */
export interface ViewportChange {
  viewport: Viewport | null;
  placement: CameraPlacement | null;
}

/** Below this in either direction a container is still being laid out, or is hidden. */
export const MIN_VIEWPORT_PIXELS = 16;

/** Slight margin, so the markings at the edge of the bounds do not touch the frame. */
const FIT_MARGIN = 1.02;

/** From the part towards the camera on the first view: in front, a little to the right, above. */
const OPENING_DIRECTION: Vec3 = [0.3, -0.85, 0.45];

/** The unit vector along `v`; a camera exactly on its target has no direction, so it gets the opening one. */
function normalized(v: Vec3): Vec3 {
  const [x, y, z] = Math.hypot(v[0], v[1], v[2]) > 0 ? v : OPENING_DIRECTION;
  const length = Math.hypot(x, y, z);
  return [x / length, y / length, z / length];
}

function subtract(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function along(origin: Vec3, direction: Vec3, distance: number): Vec3 {
  return [
    origin[0] + direction[0] * distance,
    origin[1] + direction[1] * distance,
    origin[2] + direction[2] * distance,
  ];
}

/** The bounds of a bar: its diagonal, plus room for the plates and the load arrow around it. */
export function partBounds(setup: { length: number; width: number; height: number; restraint: string }): PartBounds {
  const { length, width, height, restraint } = setup;
  return {
    key: `${length}|${width}|${height}|${restraint}`,
    centre: [0.5 * length, 0.5 * width, 0.5 * height],
    radius: 0.5 * Math.hypot(length, width, height) + Math.max(width, height) * 2,
  };
}

/** The bounds of an example part, from its surface: a sphere round it with a margin for the arrows. */
export function boundsOfPositions(key: string, positions: Float32Array, direction?: Vec3): PartBounds {
  const low: [number, number, number] = [Infinity, Infinity, Infinity];
  const high: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i + 2 < positions.length; i += 3) {
    for (let axis = 0; axis < 3; axis++) {
      low[axis] = Math.min(low[axis], positions[i + axis]);
      high[axis] = Math.max(high[axis], positions[i + axis]);
    }
  }
  const radius = 0.5 * Math.hypot(high[0] - low[0], high[1] - low[1], high[2] - low[2]);
  return {
    key,
    centre: [0.5 * (low[0] + high[0]), 0.5 * (low[1] + high[1]), 0.5 * (low[2] + high[2])],
    radius: radius * 1.25,
    direction,
  };
}

/** The largest rectangle of the view beside `obstructions`, each cut off in turn. */
export function clearRect(viewport: Viewport, obstructions: Rect[]): Rect {
  let clear: Rect = { left: 0, top: 0, right: viewport.width, bottom: viewport.height };
  for (const o of obstructions) {
    const sides: Rect[] = [
      { ...clear, right: Math.min(clear.right, o.left) },
      { ...clear, left: Math.max(clear.left, o.right) },
      { ...clear, bottom: Math.min(clear.bottom, o.top) },
      { ...clear, top: Math.max(clear.top, o.bottom) },
    ];
    const room = (r: Rect) => Math.min(r.right - r.left, r.bottom - r.top);
    const overlaps = o.left < clear.right && o.right > clear.left && o.top < clear.bottom && o.bottom > clear.top;
    if (overlaps) clear = sides.reduce((best, side) => (room(side) > room(best) ? side : best));
  }
  return clear;
}

/** The view shift that puts the view's centre at the middle of `clear`. */
export function viewShift(viewport: Viewport, clear: Rect): ViewShift {
  return {
    x: 0.5 * viewport.width - 0.5 * (clear.left + clear.right),
    y: 0.5 * viewport.height - 0.5 * (clear.top + clear.bottom),
  };
}

/** The clear rectangle's smaller side as a share of the view's height. */
export function clearShare(viewport: Viewport, clear: Rect): number {
  return Math.min(clear.right - clear.left, clear.bottom - clear.top) / viewport.height;
}

/**
 * How far back a camera must be for a sphere to fit. `share` is the room the sphere has as a share
 * of the view's height; without it, the whole view: across and up.
 */
export function fitDistance(radius: number, fovDegrees: number, aspect: number, share?: number): number {
  const fov = (fovDegrees * Math.PI) / 180;
  const fitHeight = radius / Math.sin(fov / 2);
  // The field of view is vertical, so a portrait view has to back away to fit the part across.
  return (fitHeight / (share ?? Math.min(aspect, 1))) * FIT_MARGIN;
}

/** The camera decisions for one viewer; see the file header for the rules. */
export class CameraFraming {
  private aspect: number | null = null;
  /** The clear share of the view's height, once obstructions are known. */
  private share: number | undefined = undefined;
  /** Where the part's centre is drawn relative to the view's centre. */
  private shift: ViewShift = { x: 0, y: 0 };
  private bounds: PartBounds | null = null;
  /** Whether `bounds` has its placement yet; false while a part waits for a real size. */
  private placed = false;
  /** A part has been placed before, so the visitor's viewing direction is the one to keep. */
  private hasPlacedOnce = false;

  constructor(private fovDegrees: number) {}

  /** The view shift that keeps the part clear of the panels, for the last reported size. */
  get viewOffset(): ViewShift {
    return this.shift;
  }

  /** A part is known but the view has no real size to place it against yet. */
  get waitingForViewport(): boolean {
    return this.bounds !== null && !this.placed;
  }

  /** Records the part being shown; a new part size moves the camera, the same size never does. */
  showPart(bounds: PartBounds, pose: CameraPose): CameraPlacement | null {
    if (this.bounds !== null && bounds.key === this.bounds.key) return null;
    this.bounds = bounds;
    this.placed = false;
    return this.aspect === null ? null : this.place(this.aspect, bounds, pose);
  }

  /** Takes the container's size, as reported, whenever it may have changed. */
  viewportChanged(clientWidth: number, clientHeight: number, pose: CameraPose, obstructions: Rect[] = []): ViewportChange {
    if (clientWidth < MIN_VIEWPORT_PIXELS || clientHeight < MIN_VIEWPORT_PIXELS) {
      return { viewport: null, placement: null };
    }
    const viewport = { width: clientWidth, height: clientHeight };
    const clear = clearRect(viewport, obstructions);
    const aspect = clientWidth / clientHeight;
    const previous = this.aspect;
    const previousShare = this.share;
    this.aspect = aspect;
    this.share = obstructions.length > 0 ? clearShare(viewport, clear) : undefined;
    this.shift = viewShift(viewport, clear);
    return {
      viewport,
      placement: this.bounds === null ? null : this.replace(previous, previousShare, aspect, this.bounds, pose),
    };
  }

  /** The move a new view shape calls for, if any. */
  private replace(
    previous: number | null, previousShare: number | undefined, aspect: number, bounds: PartBounds, pose: CameraPose,
  ): CameraPlacement | null {
    if (!this.placed || previous === null) return this.place(aspect, bounds, pose);
    const before = fitDistance(bounds.radius, this.fovDegrees, previous, previousShare);
    const after = fitDistance(bounds.radius, this.fovDegrees, aspect, this.share);
    if (before === after) return null;
    const offset = subtract(pose.position, pose.target);
    const distance = Math.hypot(offset[0], offset[1], offset[2]) * (after / before);
    return {
      position: along(pose.target, normalized(offset), distance),
      target: pose.target,
      near: after / 100,
      far: after * 10,
    };
  }

  /** Aims the camera at a part from the opening view, or from the visitor's direction after that. */
  private place(aspect: number, bounds: PartBounds, pose: CameraPose): CameraPlacement {
    const distance = fitDistance(bounds.radius, this.fovDegrees, aspect, this.share);
    const direction = normalized(
      bounds.direction ?? (this.hasPlacedOnce ? subtract(pose.position, bounds.centre) : OPENING_DIRECTION),
    );
    this.placed = true;
    this.hasPlacedOnce = true;
    return {
      position: along(bounds.centre, direction, distance),
      target: bounds.centre,
      near: distance / 100,
      far: distance * 10,
    };
  }
}
