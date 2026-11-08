//! Whether a result is inside the range this model can describe. The solver is linear elastic with
//! small-deflection geometry, so at a large enough load it still returns a number and that number
//! still agrees with the textbook formulas (which are the same linear theory), although neither
//! means anything any more. [`assess`] names the cases, so the page can say so.
//!
//! Each limit is where the effect linear theory leaves out reaches [`NEGLECTED_EFFECT_SHARE`] of
//! what it keeps:
//!
//! - **Yield.** A linear material has no yield, so past the yield stress the model does not apply
//!   at all. The limit is the material's yield strength, compared with the peak stress outside
//!   the clamps (the clamps' corners are a singularity, `view.rs`).
//! - **A push on a bar fixed at one end.** Linear theory ignores that the tip also moves inward
//!   as the bar bends, by `integral of w'^2 / 2` along it, which is `3 d^2 / (5 L)` for the
//!   cantilever's cubic shape. As a share of the deflection `d` that is `0.6 d / L`, so the limit
//!   is `d = L * share / 0.6`: one twelfth of the length.
//! - **A push on a bar fixed at both ends.** Held ends cannot move inward, so the bar must stretch
//!   instead, and that stiffens it long before the slope is large. Taking the clamped shape
//!   `w (1 - cos(2 pi x / L)) / 2`, the axial strain is `pi^2 w^2 / (4 L^2)`, its energy gives a
//!   force of `pi^4 E A w^3 / (8 L^3)`, and the bending stiffness is `192 E I / L^3`. The share
//!   is therefore `pi^4 w^2 / (1536 r^2)` with `r^2 = I / A`, so the limit is
//!   `w = r * sqrt(1536 * share) / pi^2`, about `0.89 r`. It depends on the section, not the
//!   length.
//! - **Twist.** Linear theory moves a point along the tangent by `r phi`, where the true path also
//!   moves inward by `r phi^2 / 2`, a share of `phi / 2`, so the limit is `phi = 2 * share`:
//!   0.1 radian. The angle is the largest displacement over the distance from the axis to the
//!   section's corner, which is where a twist moves a point furthest.

use crate::catalog::{LoadKind, Restraint, StudyRequest};
use crate::view::ViewData;

/// How large the neglected effect may be, as a share of what linear theory keeps.
pub const NEGLECTED_EFFECT_SHARE: f64 = 0.05;

/// The twist angle limit in radians.
pub const TWIST_LIMIT_RADIANS: f64 = 2.0 * NEGLECTED_EFFECT_SHARE;

/// The largest deflection of a bar fixed at one end that stays inside the model.
pub fn one_end_deflection_limit(length: f64) -> f64 {
    length * NEGLECTED_EFFECT_SHARE / 0.6
}

/// The largest deflection of a bar fixed at both ends that stays inside the model, for a section
/// with this radius of gyration.
pub fn both_ends_deflection_limit(radius_of_gyration: f64) -> f64 {
    let pi_squared = std::f64::consts::PI * std::f64::consts::PI;
    radius_of_gyration * (1536.0 * NEGLECTED_EFFECT_SHARE).sqrt() / pi_squared
}

/// What the current estimate says about the model's validity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Validity {
    pub exceeds_yield: bool,
    pub excessive_deflection: bool,
    pub excessive_twist: bool,
    /// The deflection limit in mm, for a push.
    pub deflection_limit: Option<f64>,
    /// The twist angle in radians, for a twist.
    pub twist_radians: Option<f64>,
    /// The twist limit in radians, for a twist.
    pub twist_limit_radians: Option<f64>,
}

impl Validity {
    /// True when no limit is exceeded.
    pub fn within_model(&self) -> bool {
        !(self.exceeds_yield || self.excessive_deflection || self.excessive_twist)
    }
}

/// Assesses a result of `request`. A value exactly at a limit is inside it.
pub fn assess(request: &StudyRequest, yield_strength: f64, view: &ViewData) -> Validity {
    let exceeds_yield = view.peak_stress_outside_clamps > yield_strength;
    match request.load {
        LoadKind::Push => {
            let limit = match request.restraint {
                Restraint::FixedOneEnd => one_end_deflection_limit(request.length),
                Restraint::FixedBothEnds => {
                    let section = request.section_shape();
                    both_ends_deflection_limit((section.inertia() / section.area()).sqrt())
                }
            };
            Validity {
                exceeds_yield,
                excessive_deflection: view.max_displacement > limit,
                excessive_twist: false,
                deflection_limit: Some(limit),
                twist_radians: None,
                twist_limit_radians: None,
            }
        }
        LoadKind::Twist => {
            let corner =
                0.5 * (request.width * request.width + request.height * request.height).sqrt();
            let radians = view.max_displacement / corner;
            Validity {
                exceeds_yield,
                excessive_deflection: false,
                excessive_twist: radians > TWIST_LIMIT_RADIANS,
                deflection_limit: None,
                twist_radians: Some(radians),
                twist_limit_radians: Some(TWIST_LIMIT_RADIANS),
            }
        }
    }
}
