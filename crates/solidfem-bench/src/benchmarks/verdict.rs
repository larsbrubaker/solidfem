//! The judgments of the real-part benchmark runner, free of any meshing or solving so they can be
//! tested alone: the verdicts a part may get, the mesh-convergence and stress-singularity reading
//! of a series of results at finer and finer meshes, the limit check every number goes through,
//! and the load-balance check that every case must pass. The case files in `cases/` apply them to
//! what `pipeline.rs` measured.
//!
//! **Non-finite values never pass.** Every limit goes through [`within_limit`] and every running
//! worst value through [`worst_of`], so a `NaN` or infinite measurement fails a case instead of
//! slipping through a comparison that is false for `NaN`, or being dropped by `f64::max`.
//!
//! **Convergence versus singularity.** A peak stress over refinement is read through the growth
//! exponent `s` in `peak ~ h^-s`, measured between consecutive densities as
//! `ln(p[i+1] / p[i]) / ln(h[i] / h[i+1])`. At a smooth feature the peak's error falls like `h^2`
//! for 10-node elements, so `s` shrinks towards zero, by about a factor of four per halving of
//! `h`. A re-entrant corner of opening angle `omega` has a stress field `~ r^(lambda-1)` with
//! `lambda` the Williams eigenvalue (0.5445 at the 270 degree corner of an L), so a sampled peak
//! grows like `h^(lambda-1)`: `s` stays near `1 - lambda`, 0.456 for the L. A peak is called
//! [`PeakBehaviour::Singular`] when it grew at each step, its last exponent is at least
//! [`SINGULAR_EXPONENT`] and that exponent has not fallen below [`SINGULAR_PERSISTENCE`] of the one
//! before it. The threshold 0.2 is under half the theoretical exponent, because the peak is read
//! from a recovered field and integration points a finite distance from the corner, and those
//! lag the true growth on any given mesh; the persistence ratio sits between 1.0 (the
//! singular behaviour) and 0.25 (the smooth one), so a converging series cannot pass it.

use std::fmt;

/// The smallest last-step growth exponent a singular peak is allowed. Derived in the file header.
pub const SINGULAR_EXPONENT: f64 = 0.2;
/// The fraction of the previous step's exponent the last must reach to count as not decaying.
pub const SINGULAR_PERSISTENCE: f64 = 0.6;

/// A part's result: PASS, FAIL, NOT ASSESSED or OUT OF SCOPE.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Everything measured is within its stated limits.
    Pass,
    /// Every stated limit was met, but those limits test only what the text says; shown as
    /// `PASS (text)`, never as a bare PASS.
    PassLimited(String),
    /// A limit was broken, or a stage refused the part, with the named reason.
    Fail(String),
    /// A solve hit its time cap, so no error can be quoted.
    NotAssessed(String),
    /// The physics this part needs is not modelled, naming what is missing.
    OutOfScope(String),
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Verdict::Pass => f.write_str("PASS"),
            Verdict::PassLimited(limits) => write!(f, "PASS ({limits})"),
            Verdict::Fail(why) => write!(f, "FAIL: {why}"),
            Verdict::NotAssessed(why) => write!(f, "NOT ASSESSED: {why}"),
            Verdict::OutOfScope(why) => write!(f, "OUT OF SCOPE: {why}"),
        }
    }
}

/// How a peak stress behaved as the mesh was refined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PeakBehaviour {
    /// The last step changed the peak by at most the allowed fraction, and it is not singular.
    Converged { last_change: f64 },
    /// The peak keeps growing like a stress singularity: not a usable number at any mesh.
    Singular { exponent: f64 },
    /// Neither: the peak moved more than allowed but does not grow like a singularity.
    Unsettled { last_change: f64 },
}

/// The growth exponents of `peaks` over the mesh sizes `sizes`, one per consecutive pair, or `None`
/// when a size or peak is not positive and finite, the sizes are not strictly decreasing, or the
/// lengths do not match.
pub fn growth_exponents(sizes: &[f64], peaks: &[f64]) -> Option<Vec<f64>> {
    if sizes.len() != peaks.len() || sizes.len() < 2 {
        return None;
    }
    if sizes
        .iter()
        .chain(peaks)
        .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return None;
    }
    sizes
        .windows(2)
        .zip(peaks.windows(2))
        .map(|(h, p)| {
            if h[1] >= h[0] {
                return None;
            }
            Some((p[1] / p[0]).ln() / (h[0] / h[1]).ln())
        })
        .collect()
}

/// The fractional change of the last value in `values` from the one before it.
pub fn last_change(values: &[f64]) -> Option<f64> {
    let (last, previous) = (*values.last()?, *values.get(values.len().checked_sub(2)?)?);
    if last == 0.0 {
        return None;
    }
    Some(((last - previous) / last).abs())
}

/// Reads a peak-stress series coarse to fine: singular, converged within `allowed_change`, or
/// unsettled. Needs at least three densities to tell a persisting exponent from a falling one;
/// with fewer it is never called singular.
pub fn classify_peak(sizes: &[f64], peaks: &[f64], allowed_change: f64) -> Option<PeakBehaviour> {
    let change = last_change(peaks)?;
    let exponents = growth_exponents(sizes, peaks)?;
    let growing = peaks.windows(2).all(|p| p[1] > p[0]);
    if growing && exponents.len() >= 2 {
        let last = exponents[exponents.len() - 1];
        let previous = exponents[exponents.len() - 2];
        if last >= SINGULAR_EXPONENT && last >= SINGULAR_PERSISTENCE * previous {
            return Some(PeakBehaviour::Singular { exponent: last });
        }
    }
    Some(if change <= allowed_change {
        PeakBehaviour::Converged {
            last_change: change,
        }
    } else {
        PeakBehaviour::Unsettled {
            last_change: change,
        }
    })
}

/// How well the supports balance the applied load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadBalance {
    /// Sum of the applied loads, from each loaded face's own area.
    pub applied: [f64; 3],
    /// Sum of the support reactions, from the recovered stresses.
    pub reaction: [f64; 3],
    /// The sum of the applied load magnitudes without cancellation, the scale of the check.
    pub gross: f64,
    /// `|applied + reaction| / gross`.
    pub error: f64,
}

impl LoadBalance {
    /// Compares `applied` with `reaction` against the gross load `gross`.
    pub fn new(applied: [f64; 3], reaction: [f64; 3], gross: f64) -> Self {
        let sum = [
            applied[0] + reaction[0],
            applied[1] + reaction[1],
            applied[2] + reaction[2],
        ];
        let error = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt() / gross;
        LoadBalance {
            applied,
            reaction,
            gross,
            error,
        }
    }
}

/// The largest allowed `|applied + reaction| / gross`. Each support reaction is an integral of the
/// recovered stresses against the same 4-point rule the assembly used (exact on straight-edged
/// 10-node elements), so the imbalance is the solve residual summed over the free values, at most
/// about `sqrt(n)` of the attained relative residual against the gross load for `n` values. The
/// solver stops at a relative residual of 1e-9 or, when that is below what `f64` can resolve for
/// the product `A x`, at the rounding floor of that product (`solidfem-solve`'s `pcg.rs`); a
/// slender strip stops by that path at 2.6e-7 to 4.1e-7, not at 1e-9. So 1e-6 is a limit checked
/// against the attained residual on every row, not a consequence of the 1e-9 tolerance.
pub const LOAD_BALANCE_LIMIT: f64 = 1e-6;

/// A plain interval check with a one-sided error reported: `value` must lie in `[low, high]`.
pub fn within(value: f64, low: f64, high: f64) -> bool {
    value >= low && value <= high
}

/// True when `value` is finite and at most `limit`: every error and last-change limit goes through
/// this, so a `NaN` or infinite value fails it where a bare `>` would let it pass.
pub fn within_limit(value: f64, limit: f64) -> bool {
    value.is_finite() && value <= limit
}

/// The larger of two values, `NaN` when either is: a running worst or peak must not lose a `NaN`
/// the way `f64::max` does.
pub fn worst_of(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}
