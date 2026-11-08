// What the results panel says for a summary from the engine: pure functions from `summaryJson()`
// to text and row states, tested in src/results-view.test.ts; src/main.ts only puts them in the
// DOM. A percentage is presented as a result only when the solve is done and the result is inside
// the linear model. While the solve runs the estimate is still moving (its first iterations read
// -100 %), and past the model's limits the textbook is the same linear theory, so close agreement
// there would mean nothing. Which limits apply is the engine's (demo/wasm/src/validity.rs).

import { formatNumber, formatPercent, formatSafetyFactor, safetyVerdict } from './format.ts';

export interface Validity {
  withinModel: boolean;
  exceedsYield: boolean;
  excessiveDeflection: boolean;
  excessiveTwist: boolean;
  deflectionLimit: number | null;
  twistRadians: number | null;
  twistLimitRadians: number | null;
}

export interface ComparisonRow {
  quantity: string;
  unit: string;
  place: string;
  fea: number | null;
  textbook: number | null;
  percentDifference: number | null;
}

export interface Summary {
  state: string;
  iterations?: number;
  residual?: number | null;
  elements?: number;
  nodes?: number;
  minStress?: number | null;
  maxStress?: number | null;
  maxDisplacement?: number | null;
  peakStressOutsideClamps?: number | null;
  safetyFactor?: number | null;
  validity?: Validity;
  failure?: string | null;
  textbook?: 'compared' | 'none';
  rows?: ComparisonRow[];
  explanation?: string;
}

/** How a row's difference column reads. */
export type RowStatus = 'solving' | 'result' | 'not-meaningful';

export interface RowView {
  quantity: string;
  place: string;
  fea: string;
  textbook: string;
  unit: string;
  difference: string;
  status: RowStatus;
}

export interface TableView {
  /** Sentence shown instead of the table when there is no textbook answer. */
  note: string;
  rows: RowView[];
}

/** The model's limits were exceeded, as far as a finished result can say. */
export function outsideModel(summary: Summary): boolean {
  return summary.state === 'done' && summary.validity !== undefined && !summary.validity.withinModel;
}

function rowStatus(summary: Summary): RowStatus {
  if (summary.state !== 'done') return 'solving';
  return outsideModel(summary) ? 'not-meaningful' : 'result';
}

const DIFFERENCE_TEXT: Record<Exclude<RowStatus, 'result'>, string> = {
  solving: 'solving…',
  'not-meaningful': 'not meaningful',
};

/** The table rows and notes for a summary. */
export function tableView(summary: Summary): TableView {
  if (summary.textbook === 'none') {
    return { note: `No textbook answer for this case. ${summary.explanation ?? ''}`.trim(), rows: [] };
  }
  const status = rowStatus(summary);
  return {
    note: '',
    rows: (summary.rows ?? []).map(row => ({
      quantity: row.quantity,
      place: row.place,
      fea: formatNumber(row.fea),
      textbook: formatNumber(row.textbook),
      unit: row.unit,
      difference: status === 'result' ? formatPercent(row.percentDifference) : DIFFERENCE_TEXT[status],
      status,
    })),
  };
}

/** The calm warning shown beside a finished result outside the model, or null when it is inside. */
export function modelWarning(summary: Summary): string | null {
  if (!outsideModel(summary) || !summary.validity) return null;
  const v = summary.validity;
  const reasons: string[] = [];
  if (v.exceedsYield) {
    reasons.push('the peak stress is above the material’s yield stress, so the part would permanently bend or break');
  }
  if (v.excessiveDeflection) {
    reasons.push(`it would bend ${formatNumber(summary.maxDisplacement)} mm, far more than the roughly ${formatNumber(v.deflectionLimit)} mm a linear model can describe`);
  }
  if (v.excessiveTwist) {
    reasons.push(`it would twist ${formatNumber(radiansToDegrees(v.twistRadians))}°, beyond the roughly ${formatNumber(radiansToDegrees(v.twistLimitRadians))}° a linear model can describe`);
  }
  const why = reasons.map((reason, i) => (i === 0 ? reason : `and ${reason}`)).join(', ');
  const sentence = why.charAt(0).toUpperCase() + why.slice(1);
  return `Outside this model. ${sentence}. These numbers and the textbook comparison do not apply here: they come from the same linear theory, so their agreement means nothing. Reduce the load to see a valid result.`;
}

function radiansToDegrees(radians: number | null): number | null {
  return radians === null ? null : (radians * 180) / Math.PI;
}

/** The line under the safety factor, consistent with the table and the warning. */
export function safetyMessage(summary: Summary): string {
  if (summary.state !== 'done') return 'Estimate while the solve runs.';
  if (summary.validity && !summary.validity.withinModel) {
    if (summary.validity.exceedsYield) {
      return 'Below 1: the part would yield under this load, and this linear model does not apply past yield.';
    }
    return 'Below yield, but the part bends too far for this model, so this figure is not reliable.';
  }
  return safetyVerdict(summary.safetyFactor);
}

/** The safety factor text. */
export function safetyText(summary: Summary): string {
  return formatSafetyFactor(summary.safetyFactor);
}
