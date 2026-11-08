// Cantilever tip deflection against Timoshenko beam theory, and the slender case that exposes
// locking. A traction over the free end carries the shear; the root is fully clamped.

use solidfem::elastic::{Material, Model, Solution};

use crate::support::{bar, faces_at, fix_root, mean_displacement, report_case, run};

const LOAD: f64 = 1.0;

/// Runs the cantilever and returns (tip deflection under the load, solution).
fn run_cantilever(
    length: f64,
    depth: f64,
    material: Material,
    divisions: [usize; 3],
) -> (f64, Solution) {
    let area = depth * depth;
    let mesh = bar([length, depth, depth], divisions);
    let mut model = Model::new(mesh.clone(), material);
    fix_root(&mut model);
    let traction = LOAD / area;
    for face in faces_at(&mesh, length) {
        model.add_traction(face, [0.0, 0.0, -traction]);
    }
    let solution = run(model);
    (-mean_displacement(&mesh, &solution, length, 2), solution)
}

fn timoshenko(length: f64, depth: f64, material: Material) -> f64 {
    let area = depth * depth;
    let inertia = depth.powi(4) / 12.0;
    let e = material.youngs_modulus;
    let g = e / (2.0 * (1.0 + material.poisson_ratio));
    // Shear coefficient of a rectangular section at nu = 0.
    let kappa = 5.0 / 6.0;
    LOAD * length.powi(3) / (3.0 * e * inertia) + LOAD * length / (kappa * g * area)
}

#[test]
fn tip_deflection_matches_timoshenko_and_the_error_falls_as_the_mesh_refines() {
    // nu = 0 removes the clamp's Poisson restraint, which beam theory does not model. The
    // remaining modelling gap is the clamp's warping restraint and the uniform end traction,
    // each a fraction of the 0.8% shear term at L/h = 10, so under 0.5% together; quadratic
    // elements add a discretization error of a few tenths of a percent at four cells through the
    // depth. Limit at [40, 4, 4]: 1%.
    //
    // The error against the formula stops being a measure of convergence once it reaches the
    // modelling gap: the solid's true deflection sits slightly above the beam formula, so finer
    // meshes than these eventually cross it. These meshes are nested, and a conforming element is
    // too stiff, so the deflection itself must also rise with every refinement. That check holds
    // at any mesh; the error check holds while discretization error dominates, as it does here.
    let material = Material {
        youngs_modulus: 1000.0,
        poisson_ratio: 0.0,
    };
    let (length, depth) = (10.0, 1.0);
    let expected = timoshenko(length, depth, material);
    let mut previous = f64::INFINITY;
    let mut previous_tip = f64::NEG_INFINITY;
    for (name, divisions) in [
        ("cantilever_10x1x1", [10, 1, 1]),
        ("cantilever_20x2x2", [20, 2, 2]),
        ("cantilever_40x4x4", [40, 4, 4]),
    ] {
        let (tip, solution) = run_cantilever(length, depth, material, divisions);
        let error = ((tip - expected) / expected).abs();
        report_case(name, error * 100.0, &solution);
        assert!(
            error < previous,
            "{name}: error {error} did not fall from {previous}"
        );
        assert!(
            tip > previous_tip,
            "{name}: tip {tip} did not rise from {previous_tip}: a refinement stiffened the part"
        );
        previous = error;
        previous_tip = tip;
    }
    assert!(previous < 0.01, "finest mesh error {previous}");
}

#[test]
fn a_slender_cantilever_does_not_lock_at_a_coarse_mesh() {
    // At L/h = 50 and element aspect 2, beam theory's shear term is 0.03% of the bending term.
    // A locking element would report 50% of the deflection or less; a quadratic element carries
    // the bending strain exactly per element and should be within a few percent. Limit: 5%.
    let material = Material {
        youngs_modulus: 2100.0,
        poisson_ratio: 0.3,
    };
    let (length, depth) = (50.0, 1.0);
    let expected = timoshenko(length, depth, material);
    let (tip, solution) = run_cantilever(length, depth, material, [25, 1, 1]);
    let error = ((tip - expected) / expected).abs();
    report_case("slender_cantilever_25x1x1", error * 100.0, &solution);
    assert!(
        error < 0.05,
        "slender tip {tip}, expected {expected}, error {error}"
    );
}
