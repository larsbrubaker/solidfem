// A stress field the elements represent exactly reads as converged, and an estimate that cannot be
// formed says why instead of reading as converged.

use solidfem_elastic::{Material, ModelError};

use super::support::{box_mesh, held_study, solve_and_estimate, unsolved_held_study};

const SIZE: [f64; 3] = [3.0, 2.0, 2.0];
const CURVATURE: f64 = 1e-3;
const NU: f64 = 0.3;
const STEEL: Material = Material {
    youngs_modulus: 2100.0,
    poisson_ratio: NU,
};

/// Pure bending about z: `σ_xx = E κ y` and every other component zero, which is in equilibrium
/// with no body force. Its displacement is quadratic, so 10-node elements hold it exactly.
fn bending(p: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = p;
    [
        CURVATURE * x * y,
        -0.5 * CURVATURE * (x * x + NU * (y * y - z * z)),
        -NU * CURVATURE * y * z,
    ]
}

#[test]
fn an_exactly_represented_bending_field_reads_as_converged() {
    let mesh = box_mesh(SIZE, [3, 2, 2]);
    let mut study = held_study(mesh, SIZE, STEEL, bending);
    let residual = study.report().relative_residual;
    let (_, estimate) = solve_and_estimate(&mut study);
    // In exact arithmetic the solution is the bending field itself, its stress is linear, and a
    // linear patch fit and the 10-node interpolation of it reproduce a linear field, so the
    // estimate is zero. What is left is rounding from two sources. The solve stops at relative
    // residual `ρ`, which leaves a relative energy-norm error of at most `sqrt(κ) ρ` in the
    // displacements, where `κ`, the condition number of this coarse mesh's free stiffness, is
    // below 1e4 (it grows as 1/h² from about 1e2 on a unit element); the stress error and so `η`
    // inherit it, at most 1e2 ρ. The patch fits solve 4 x 4 normal equations on offsets scaled to
    // one, condition below about 1e2, which with the interpolation leaves about 1e2 x 2.2e-16 of
    // the stress, so 1e-13 of `η`. Ten times both bounds is the tolerance.
    let tolerance = 10.0 * (1e2 * residual + 1e-13);
    eprintln!(
        "η = {:e}, residual {residual:e}, tolerance {tolerance:e}",
        estimate.relative_error
    );
    assert!(
        estimate.relative_error <= tolerance,
        "η = {} with residual {residual}, tolerance {tolerance}",
        estimate.relative_error
    );
    assert!(estimate.solution_norm_squared > 0.0);
    // Every element's own indicator is at rounding too, relative to the whole solution's norm.
    let scale = estimate.solution_norm_squared.sqrt();
    for (tet, error) in estimate.element_errors.iter().enumerate() {
        assert!(*error <= tolerance * scale, "element {tet}: {error}");
    }
}

#[test]
fn a_patch_holding_two_materials_refuses_an_estimate() {
    let mesh = box_mesh(SIZE, [3, 2, 2]);
    let soft = Material {
        youngs_modulus: 700.0,
        poisson_ratio: NU,
    };
    let two_materials = move |p: [f64; 3]| if p[0] < 1.0 { STEEL } else { soft };
    let mut study = held_study(mesh, SIZE, two_materials, bending);
    let error = study
        .solution_with_error_estimate(&mut || false)
        .expect_err("a two-material patch has no recovered field to compare with");
    assert!(
        matches!(error, ModelError::MixedMaterialPatch { .. }),
        "{error:?}"
    );
}

#[test]
fn a_preview_of_an_unconverged_solve_refuses_an_estimate() {
    // A preview recovery serves a host drawing a solve in progress; its displacements are not an
    // answer yet, so an estimate of them would read a partial solve as converged.
    let mut study = unsolved_held_study(box_mesh(SIZE, [3, 2, 2]), SIZE, STEEL, bending);
    study.step(2, &mut || false).expect("two iterations run");
    let mut recovery = study.begin_recovery();
    study
        .continue_recovery(&mut recovery, usize::MAX, &mut || false)
        .expect("a preview recovers");
    let mut estimation = study.begin_error_estimate(recovery);
    let error = study
        .continue_error_estimate(&mut estimation, usize::MAX, &mut || false)
        .expect_err("a preview is not an answer to estimate");
    assert_eq!(error, ModelError::NotFinished);
    assert_eq!(
        study.finish_error_estimate(&estimation),
        Err(ModelError::NotFinished)
    );
}
