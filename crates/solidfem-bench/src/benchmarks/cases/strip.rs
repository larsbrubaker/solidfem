//! Part 1: a thin cantilever strip with a tip load, at nu = 0 and nu = 0.3.

use super::*;

const STRIP_L: f64 = 100.0;
const STRIP_W: f64 = 10.0;
const STRIP_T: f64 = 1.0;
const STRIP_F: f64 = 1.0;

fn strip_apply(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
    let (faces, area) = plane_faces(mesh, 0, STRIP_L);
    check_area(area, STRIP_W * STRIP_T, "tip face")?;
    Ok(Supports {
        fixed: plane_nodes(mesh, 0, 0.0),
        tractions: faces
            .into_iter()
            .map(|f| (f, [0.0, 0.0, -STRIP_F / area]))
            .collect(),
        ..Supports::default()
    })
}

fn strip_measure(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
) -> Result<Vec<(String, f64)>, String> {
    let tip = mean_deflection(mesh, solution, 0, STRIP_L, 2)?;
    Ok(vec![("tip".to_string(), tip)])
}

/// Bending and shear parts of the Timoshenko tip deflection for the strip.
fn strip_reference(nu: f64) -> (f64, f64) {
    let inertia = STRIP_W * STRIP_T * STRIP_T * STRIP_T / 12.0;
    let area = STRIP_W * STRIP_T;
    let g = E / (2.0 * (1.0 + nu));
    (
        STRIP_F * STRIP_L * STRIP_L * STRIP_L / (3.0 * E * inertia),
        STRIP_F * STRIP_L / ((5.0 / 6.0) * g * area),
    )
}

/// **Limit, nu = 0.** Tip deflection against `F L^3/(3 E I) + F L/(k G A)` with `k = 5/6`, within
/// 1% at the finest mesh, and the last refinement moving it by at most 1%. With nu = 0 there is no
/// anticlastic restraint at the clamp, so beam theory is exact to the shear term, which is itself
/// about 0.6 (t/L)^2 of the bending term (6e-5 here): the reference's own uncertainty is under
/// 0.01%, and the whole 1% is what a two-layer 10-node strip is allowed for discretization.
fn strip_nu0_judge(rows: &[DensityResult]) -> Verdict {
    verdict_of((|| {
        let last = rows.last().ok_or("no rows")?;
        let (bending, shear) = strip_reference(0.0);
        check_error(last.value("tip"), bending + shear, 0.01, "tip deflection")?;
        stable(rows, "tip", 0.01, "tip deflection")
    })())
}

/// The fraction by which the clamp stiffens the strip relative to the narrow beam, for an
/// effective fully restrained length `length`. Derived on [`strip_nu03_judge`].
fn clamp_stiffening(nu: f64, length: f64) -> f64 {
    3.0 * nu * nu * length / STRIP_L
}

/// **Limit, nu = 0.3.** In linear statics the width-over-thickness ratio has no effect on a free
/// strip: the anticlastic curvature is free to form, so away from the clamp the section bends with
/// the narrow-beam stiffness `E I`, whatever `b / t` is. The clamp alone restrains the transverse
/// strain. Over a length `l` near the root the section is stiffened from `E I` by up to
/// `1 / (1 - nu^2)`, removing a fraction `nu^2` of the curvature there; with moment `F L` and arm
/// `L` that takes `nu^2 (F L / E I) l L` off the tip, a fraction `3 nu^2 l / L` of the bending
/// deflection. The restraint decays away from the clamp like a self-equilibrated end disturbance
/// of a strip with free edges, whose decay length is about a quarter of the width `b`; a full
/// restraint is at most about half the width long, so the effective `l` lies between `b / 4` and
/// `b / 2`, a clamp reduction of 0.68% to 1.35% of the bending term here, about 1%. That is an
/// estimate of a local effect, not a closed form. The tip deflection is compared with the narrow
/// beam `d = d_b + d_s` (Timoshenko shear added): it must lie below `d` by the clamp reduction,
/// give or take 1% of `d_b` for discretization as in the nu = 0 case, so the band is
/// `[d - 2.35% d_b, d + 0.32% d_b]`, the narrow beam itself just inside the upper limit, with the
/// last refinement moving it by at most 1%.
fn strip_nu03_judge(rows: &[DensityResult]) -> Verdict {
    verdict_of((|| {
        let last = rows.last().ok_or("no rows")?;
        let nu = 0.3;
        let (bending, shear) = strip_reference(nu);
        let narrow = bending + shear;
        let (low, high) = (
            narrow - (clamp_stiffening(nu, 0.5 * STRIP_W) + 0.01) * bending,
            narrow - (clamp_stiffening(nu, 0.25 * STRIP_W) - 0.01) * bending,
        );
        let tip = last.value("tip");
        if !within(tip, low, high) {
            return Err(format!(
                "tip deflection {tip:.6e} outside the clamp-restrained narrow-beam band [{low:.6e}, {high:.6e}]"
            ));
        }
        stable(rows, "tip", 0.01, "tip deflection")
    })())
}

fn strip_densities() -> Vec<Density> {
    vec![
        density("h=2.0", "strip_h2.0", Sizing::Uniform(2.0), 2.0),
        density("h=1.4", "strip_h1.4", Sizing::Uniform(1.4), 1.4),
        density("h=1.0", "strip_h1.0", Sizing::Uniform(1.0), 1.0),
    ]
}

pub(super) fn cases() -> Vec<Case> {
    vec![
        Case {
            id: "strip-nu0",
            title: "Part 1: thin cantilever strip 100 x 10 x 1, tip load, nu = 0",
            material: steel(0.0),
            densities: strip_densities(),
            layers: 2,
            apply: strip_apply,
            measure: strip_measure,
            iteration_budget: None,
            judge: strip_nu0_judge,
        },
        Case {
            id: "strip-nu03",
            title: "Part 1: thin cantilever strip 100 x 10 x 1, tip load, nu = 0.3",
            material: steel(0.3),
            densities: strip_densities(),
            layers: 2,
            apply: strip_apply,
            measure: strip_measure,
            iteration_budget: None,
            judge: strip_nu03_judge,
        },
    ]
}
