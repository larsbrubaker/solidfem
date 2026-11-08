// A bar in uniaxial tension. Uniform strain is inside the quadratic element's displacement space,
// so these measure solver accuracy and load application, not discretization.

use solidfem::elastic::{Material, Model};

use crate::support::{bar, faces_at, fix_root, mean_displacement, report_case, run};

const LENGTH: f64 = 20.0;
const FORCE: f64 = 1000.0;
const YOUNGS: f64 = 2100.0;

#[test]
fn a_zero_poisson_bar_is_exact_to_the_solver_tolerance() {
    // With nu = 0 the clamp restrains nothing, so u = F x / (A E), sigma = F / A exactly. The
    // default relative residual is 1e-9; with the Jacobi-scaled condition number of this bar
    // below 1e4 the displacement error is bounded by about 1e-5, so 1e-5 is the derived limit.
    let material = Material {
        youngs_modulus: YOUNGS,
        poisson_ratio: 0.0,
    };
    let mesh = bar([LENGTH, 1.0, 1.0], [20, 2, 2]);
    let mut model = Model::new(mesh.clone(), material);
    fix_root(&mut model);
    for face in faces_at(&mesh, LENGTH) {
        model.add_traction(face, [FORCE, 0.0, 0.0]);
    }
    let solution = run(model);
    let elongation = mean_displacement(&mesh, &solution, LENGTH, 0);
    let expected = FORCE * LENGTH / YOUNGS;
    let error = ((elongation - expected) / expected).abs();
    for stress in &solution.point_stresses {
        let relative = ((stress.stress[0] - FORCE) / FORCE).abs();
        assert!(
            relative < 1e-5,
            "sigma_xx {} at {:?}",
            stress.stress[0],
            stress.position
        );
    }
    // A uniaxial state: von Mises equals the axial stress, at integration points and in the
    // per-node display average, with the same solver-tolerance limit as the stress itself.
    for stress in &solution.point_stresses {
        assert!(
            ((stress.von_mises - FORCE) / FORCE).abs() < 1e-5,
            "{}",
            stress.von_mises
        );
    }
    for (node, value) in solution.nodal_von_mises.iter().enumerate() {
        assert!(
            ((value - FORCE) / FORCE).abs() < 1e-5,
            "node {node}: {value}"
        );
    }
    report_case("tension_nu0", error * 100.0, &solution);
    assert!(error < 1e-5, "elongation error {error}");
}

#[test]
fn a_poisson_bar_matches_f_over_a_away_from_the_clamp() {
    // The clamp's Poisson restraint decays like exp(-pi x / b): at 5 widths from each end that is
    // about 1e-7, so a 1e-4 limit on mid-span stress and the gauge-length strain leaves the
    // solver tolerance and rounding the only other contributors.
    let material = Material {
        youngs_modulus: YOUNGS,
        poisson_ratio: 0.3,
    };
    let mesh = bar([LENGTH, 1.0, 1.0], [20, 2, 2]);
    let mut model = Model::new(mesh.clone(), material);
    fix_root(&mut model);
    for face in faces_at(&mesh, LENGTH) {
        model.add_traction(face, [FORCE, 0.0, 0.0]);
    }
    let solution = run(model);
    let mut peak: f64 = 0.0;
    for stress in &solution.point_stresses {
        if stress.position[0] > 5.0 && stress.position[0] < 15.0 {
            peak = peak.max(((stress.stress[0] - FORCE) / FORCE).abs());
        }
    }
    let gauge =
        mean_displacement(&mesh, &solution, 15.0, 0) - mean_displacement(&mesh, &solution, 5.0, 0);
    let expected = FORCE * 10.0 / YOUNGS;
    let error = ((gauge - expected) / expected).abs();
    report_case("tension_nu03", error.max(peak) * 100.0, &solution);
    assert!(peak < 1e-4, "stress error {peak}");
    assert!(error < 1e-4, "gauge error {error}");
}
