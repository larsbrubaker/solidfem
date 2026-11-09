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

/// **Limit, nu = 0.3.** For a strip of width b = 10 t the anticlastic curvature is partly
/// restrained and the root clamp restrains the transverse strain, so the true stiffness lies
/// between the narrow beam (no restraint, `E I`) and the plate strip restrained everywhere
/// (`E I / (1 - nu^2)`): the bending part of the deflection is in
/// `[(1 - nu^2) d_b, d_b]`, with the Timoshenko shear term added to each bound. This is a
/// bracket from the two limiting beam theories, not a single closed form, so the only claim the
/// reference makes is that interval. The measurement is allowed 1% of `d_b` outside each bound
/// for discretization, as the nu = 0 case is, and the last refinement moving it by at most 1%.
fn strip_nu03_judge(rows: &[DensityResult]) -> Verdict {
    verdict_of((|| {
        let last = rows.last().ok_or("no rows")?;
        let nu = 0.3;
        let (bending, shear) = strip_reference(nu);
        let tol = 0.01 * bending;
        let (low, high) = (
            (1.0 - nu * nu) * bending + shear - tol,
            bending + shear + tol,
        );
        let tip = last.value("tip");
        if !within(tip, low, high) {
            return Err(format!(
                "tip deflection {tip:.6e} outside the bracket [{low:.6e}, {high:.6e}]"
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
