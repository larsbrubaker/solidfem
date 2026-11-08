// The stress colours: the rainbow ramp every stress plot uses (blue is low, red is high) as a pure
// function of a value in 0..1, the range mapping that puts the ramp's top at the peak stress
// outside the clamps, and the CSS gradient of the on-screen legend, so the legend and the part can
// never disagree. Tested in src/colormap.test.ts.
//
// The ramp's range is deliberately not the highest stress in the part. A clamp's corners are a
// stress singularity, whose value grows as the mesh gets finer and means nothing; with the ramp
// topped there, the whole part would read as blue. Stress above the peak outside the clamps
// saturates at the top colour.

export type Rgb = [r: number, g: number, b: number];

/** Five stops, spaced evenly, in 0..1 sRGB. */
const STOPS: Rgb[] = [
  [0.12, 0.22, 0.75],
  [0.1, 0.65, 0.85],
  [0.2, 0.75, 0.3],
  [0.97, 0.8, 0.15],
  [0.86, 0.15, 0.13],
];

/** The ramp's colour at `t` in 0..1; values outside (or NaN) take the nearest end. */
export function rampColor(t: number): Rgb {
  const clamped = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0;
  const scaled = clamped * (STOPS.length - 1);
  const index = Math.min(STOPS.length - 2, Math.floor(scaled));
  const fraction = scaled - index;
  const a = STOPS[index];
  const b = STOPS[index + 1];
  return [a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction, a[2] + (b[2] - a[2]) * fraction];
}

/** The ramp position of a stress within [low, high]; a flat range sits at the bottom. */
export function rampPosition(stress: number, low: number, high: number): number {
  if (!(high > low)) return 0;
  return (stress - low) / (high - low);
}

/** Fills `out` with three floats per stress value in `stresses`. */
export function fillColors(stresses: Float32Array, low: number, high: number, out: Float32Array): void {
  for (let i = 0; i < stresses.length; i++) {
    const [r, g, b] = rampColor(rampPosition(stresses[i], low, high));
    out[i * 3] = r;
    out[i * 3 + 1] = g;
    out[i * 3 + 2] = b;
  }
}

/** The CSS linear-gradient (bottom to top) matching the ramp, for the legend bar. */
export function rampCssGradient(): string {
  const steps: string[] = [];
  for (let i = 0; i <= 8; i++) {
    const [r, g, b] = rampColor(i / 8);
    steps.push(`rgb(${Math.round(r * 255)} ${Math.round(g * 255)} ${Math.round(b * 255)})`);
  }
  return `linear-gradient(to top, ${steps.join(', ')})`;
}
