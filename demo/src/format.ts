// Number text for the page: values with three significant figures and thousands separators, the
// percentage difference shown in the textbook table, the stress and safety-factor readouts, and
// the state line. Pure functions, tested in src/format.test.ts.

/** A number with `digits` significant figures, without exponent notation in the usual range. */
export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  if (value === 0) return '0';
  const abs = Math.abs(value);
  if (abs >= 1e6 || abs < 1e-3) return value.toExponential(digits - 1);
  const decimals = Math.max(0, digits - 1 - Math.floor(Math.log10(abs)));
  return value.toLocaleString('en-US', { maximumFractionDigits: decimals, minimumFractionDigits: 0 });
}

/**
 * The percent difference for the table, with an explicit sign: "+1.2 %", "-0.4 %", and "0.0 %"
 * when it rounds to zero (never "-0.0 %"). Values at or past 1000 % drop the decimals.
 */
export function formatPercent(percent: number | null | undefined): string {
  if (percent === null || percent === undefined || !Number.isFinite(percent)) return '—';
  const decimals = Math.abs(percent) >= 1000 ? 0 : 1;
  const text = Math.abs(percent).toFixed(decimals);
  if (Number(text) === 0) return `${text} %`;
  return `${percent < 0 ? '-' : '+'}${text} %`;
}

/** The safety factor: yield strength over the peak stress outside the clamps. */
export function formatSafetyFactor(factor: number | null | undefined): string {
  if (factor === null || factor === undefined || !Number.isFinite(factor)) return '—';
  return factor >= 100 ? 'over 100' : formatNumber(factor, 3);
}

/** What the safety factor means for the part, in plain words. */
export function safetyVerdict(factor: number | null | undefined): string {
  if (factor === null || factor === undefined || !Number.isFinite(factor)) return 'Waiting for the first stress estimate.';
  if (factor < 1) return 'Below 1: the part would yield under this load.';
  if (factor < 2) return 'Above 1 but thin: the part is close to yielding.';
  return 'Comfortably below the material’s yield stress.';
}

/**
 * The on-screen statement of how the shape is scaled, so nobody reads the drawing as true size.
 * Very large deflections are scaled down rather than up, and it says so.
 */
export function exaggerationLabel(scale: number, maxDisplacement: number | null | undefined): string {
  if (!(scale > 0)) return 'Shape shown undeformed until the solve has a deflection.';
  const factor = scale.toLocaleString('en-US', { maximumFractionDigits: 4 });
  const verb = scale < 1 ? 'Shape drawn smaller than true deflection' : 'Shape exaggerated';
  return `${verb} ×${factor}. True deflection: ${formatNumber(maxDisplacement)} mm.`;
}

/** A duration given in milliseconds: "340 ms" or "1.2 s". */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

/** The label for an engine state. */
export function stateLabel(state: string): string {
  switch (state) {
    case 'meshing': return 'Meshing the part';
    case 'refining': return 'Refining the mesh';
    case 'preparing': return 'Placing the holds and loads';
    case 'assembling': return 'Building the model';
    case 'solving': return 'Solving';
    case 'recovering': return 'Computing stresses';
    case 'done': return 'Done';
    case 'failed': return 'Could not solve';
    case 'too_large': return 'Too large';
    case 'cancelled': return 'Restarting';
    default: return 'Starting';
  }
}
