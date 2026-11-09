// The deflection animation: how much the displacement is exaggerated, where in its cycle the part
// is, and the displaced vertex positions. The exaggeration is chosen from the result, not the
// load, so the same visible motion shows at a 1 N push and a 2000 N one: the largest displacement
// is drawn as a fixed share of the bar's length, unless that would turn some part of the surface
// by more than MAX_DRAWN_ROTATION, in which case the rotation sets it. The scale is rounded so the
// on-screen label can state it exactly. Pure functions, tested in src/animation.test.ts.
//
// Each vertex is drawn along an arc rather than a straight line. A linear solve's displacement for
// a rotation is the tangent of that rotation, so a straight-line extrapolation carries a point off
// the circle it should stay on: a twisted bar swells and a bent one stretches, more with every bit
// of exaggeration. Taking the engine's local rotation at the vertex as the arc's rate keeps
// rotations rigid at any exaggeration, and reduces exactly to the straight line where the rotation
// is zero. Displacement along the rotation axis is scaled by the same factor as the rest of it,
// sin(φ)/φ of the straight-line amount: under 10 % short at the 45 degree cap, and the parts this
// page offers displace almost entirely across their rotation axis.

/** The largest drawn displacement, as a share of the bar's length. */
export const DRAWN_SHARE_OF_LENGTH = 0.12;
/** The largest local rotation the drawing may show, in radians: 45 degrees. */
export const MAX_DRAWN_ROTATION = Math.PI / 4;
/** One full bend and return. */
export const ANIMATION_PERIOD_MS = 2400;
/**
 * Below this drawn angle, in radians, the arc uses its series. Above it the closed forms are exact
 * to rounding; below it they lose digits to `1 - cos(φ)`, while the series' first dropped term is
 * under 3e-16 of the value.
 */
const SERIES_ANGLE = 1e-2;

/** Rounds `raw` to the two-significant-figure factor the label states, nearest. */
function roundedFactor(raw: number): number {
  const factor = Math.pow(10, Math.floor(Math.log10(raw)) - 1);
  return Number((Math.round(raw / factor) * factor).toPrecision(2));
}

/**
 * The largest two-significant-figure factor at or under `limit`. The decimal rounding can land a
 * hair above the bound, and the bound must hold, so it steps down one unit when it does.
 */
function roundedDown(limit: number): number {
  const factor = Math.pow(10, Math.floor(Math.log10(limit)) - 1);
  let units = Math.floor(limit / factor);
  let value = Number((units * factor).toPrecision(2));
  while (value > limit && units > 1) {
    units -= 1;
    value = Number((units * factor).toPrecision(2));
  }
  return value;
}

/**
 * The exaggeration: the rounded factor that draws `maxDisplacement` at the target share of `length`,
 * lowered if needed so the largest drawn local rotation stays within MAX_DRAWN_ROTATION.
 * `maxRotation` is the largest local rotation over the surface in radians; none is treated as no
 * limit from rotation.
 */
export function exaggerationScale(maxDisplacement: number, length: number, maxRotation = 0): number {
  if (!(maxDisplacement > 0) || !(length > 0)) return 0;
  const byShare = roundedFactor((DRAWN_SHARE_OF_LENGTH * length) / maxDisplacement);
  if (!(maxRotation > 0)) return byShare;
  return Math.min(byShare, roundedDown(MAX_DRAWN_ROTATION / maxRotation));
}

/** The animation's position at `timeMs`: eases from unloaded (0) to fully bent (1) and back. */
export function animationPhase(timeMs: number, periodMs = ANIMATION_PERIOD_MS): number {
  return 0.5 * (1 - Math.cos((2 * Math.PI * timeMs) / periodMs));
}

/**
 * Writes the displaced positions into `out`, three numbers per vertex. At `a = scale * phase` a
 * vertex at `x` with displacement `u` and local rotation `w` is drawn at
 * `x + (sin φ / |w|) u + ((1 - cos φ) / |w|²) w × u` with `φ = a |w|`, which is `x + a u` at zero
 * rotation. Without `rotation` every vertex takes the straight line.
 */
export function displacePositions(
  base: Float32Array, displacement: Float32Array, scale: number, phase: number, out: Float32Array, rotation?: Float32Array,
): void {
  const amount = scale * phase;
  const count = base.length;
  for (let i = 0; i + 2 < count; i += 3) {
    const ux = displacement[i];
    const uy = displacement[i + 1];
    const uz = displacement[i + 2];
    const wx = rotation ? rotation[i] : 0;
    const wy = rotation ? rotation[i + 1] : 0;
    const wz = rotation ? rotation[i + 2] : 0;
    const rate = Math.sqrt(wx * wx + wy * wy + wz * wz);
    const angle = amount * rate;
    let along: number;
    let across: number;
    if (angle < SERIES_ANGLE) {
      const angle2 = angle * angle;
      along = amount * (1 - (angle2 / 6) * (1 - angle2 / 20));
      across = amount * amount * (0.5 - (angle2 / 24) * (1 - angle2 / 30));
    } else {
      along = Math.sin(angle) / rate;
      across = (1 - Math.cos(angle)) / (rate * rate);
    }
    out[i] = base[i] + along * ux + across * (wy * uz - wz * uy);
    out[i + 1] = base[i + 1] + along * uy + across * (wz * ux - wx * uz);
    out[i + 2] = base[i + 2] + along * uz + across * (wx * uy - wy * ux);
  }
}
