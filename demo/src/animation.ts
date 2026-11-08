// The deflection animation: how much the displacement is exaggerated, where in its cycle the part
// is, and the displaced vertex positions. The exaggeration is chosen from the result, not the
// load, so the same visible motion shows at a 1 N push and a 2000 N one: the largest displacement
// is drawn as a fixed share of the bar's length. The scale is rounded so the on-screen label can
// state it exactly. Pure functions, tested in src/animation.test.ts.

/** The largest drawn displacement, as a share of the bar's length. */
export const DRAWN_SHARE_OF_LENGTH = 0.12;
/** One full bend and return. */
export const ANIMATION_PERIOD_MS = 2400;

/** The exaggeration: the rounded factor that draws `maxDisplacement` at the target share of `length`. */
export function exaggerationScale(maxDisplacement: number, length: number): number {
  if (!(maxDisplacement > 0) || !(length > 0)) return 0;
  const raw = (DRAWN_SHARE_OF_LENGTH * length) / maxDisplacement;
  const magnitude = Math.floor(Math.log10(raw));
  const factor = Math.pow(10, magnitude - 1);
  return Math.round(raw / factor) * factor;
}

/** The animation's position at `timeMs`: eases from unloaded (0) to fully bent (1) and back. */
export function animationPhase(timeMs: number, periodMs = ANIMATION_PERIOD_MS): number {
  return 0.5 * (1 - Math.cos((2 * Math.PI * timeMs) / periodMs));
}

/** Writes `base + displacement * scale * phase` into `out`, three numbers per vertex. */
export function displacePositions(
  base: Float32Array, displacement: Float32Array, scale: number, phase: number, out: Float32Array,
): void {
  const amount = scale * phase;
  for (let i = 0; i < base.length; i++) {
    out[i] = base[i] + displacement[i] * amount;
  }
}
