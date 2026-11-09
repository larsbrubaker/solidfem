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
}

export interface Viewport {
  width: number;
  height: number;
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

/** How far back a camera must be for a sphere to fit both across and up the view. */
export function fitDistance(radius: number, fovDegrees: number, aspect: number): number {
  const fov = (fovDegrees * Math.PI) / 180;
  const fitHeight = radius / Math.sin(fov / 2);
  // The field of view is vertical, so a portrait view has to back away to fit the part across.
  return (fitHeight / Math.min(aspect, 1)) * FIT_MARGIN;
}

/** The camera decisions for one viewer; see the file header for the rules. */
export class CameraFraming {
  private aspect: number | null = null;
  private bounds: PartBounds | null = null;
  /** Whether `bounds` has its placement yet; false while a part waits for a real size. */
  private placed = false;
  /** A part has been placed before, so the visitor's viewing direction is the one to keep. */
  private hasPlacedOnce = false;

  constructor(private fovDegrees: number) {}

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
  viewportChanged(clientWidth: number, clientHeight: number, pose: CameraPose): ViewportChange {
    if (clientWidth < MIN_VIEWPORT_PIXELS || clientHeight < MIN_VIEWPORT_PIXELS) {
      return { viewport: null, placement: null };
    }
    const aspect = clientWidth / clientHeight;
    const previous = this.aspect;
    this.aspect = aspect;
    return {
      viewport: { width: clientWidth, height: clientHeight },
      placement: this.bounds === null ? null : this.replace(previous, aspect, this.bounds, pose),
    };
  }

  /** The move a new view shape calls for, if any. */
  private replace(previous: number | null, aspect: number, bounds: PartBounds, pose: CameraPose): CameraPlacement | null {
    if (!this.placed || previous === null) return this.place(aspect, bounds, pose);
    const before = fitDistance(bounds.radius, this.fovDegrees, previous);
    const after = fitDistance(bounds.radius, this.fovDegrees, aspect);
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
    const distance = fitDistance(bounds.radius, this.fovDegrees, aspect);
    const direction = normalized(this.hasPlacedOnce ? subtract(pose.position, bounds.centre) : OPENING_DIRECTION);
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
