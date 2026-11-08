// A two-material bar in series: stiff half and soft half along the axis, under axial load. With
// nu = 0 the closed form is the sum of each part's elongation. Run with the interface on an
// element boundary, then inside elements (the mesh not following the material), to report what
// the second costs.

use solidfem::elastic::{Material, Model, Quadrature};

use crate::support::{bar, faces_at, fix_root, mean_displacement, report_case, run};

const LENGTH: f64 = 10.0;
const FORCE: f64 = 1.0;
const STIFF: f64 = 1000.0;
const SOFT: f64 = 250.0;

fn material_field(interface: f64) -> impl Fn([f64; 3]) -> Material {
    move |position| Material {
        youngs_modulus: if position[0] < interface { STIFF } else { SOFT },
        poisson_ratio: 0.0,
    }
}

/// Runs the bar and returns (error fraction, report data for printing).
fn relative_error(name: &str, interface: f64, rule: Quadrature) -> f64 {
    let mesh = bar([LENGTH, 1.0, 1.0], [20, 2, 2]);
    let mut model = Model::new(mesh.clone(), material_field(interface));
    model.set_quadrature(rule);
    fix_root(&mut model);
    for face in faces_at(&mesh, LENGTH) {
        model.add_traction(face, [FORCE, 0.0, 0.0]);
    }
    let solution = run(model);
    let elongation = mean_displacement(&mesh, &solution, LENGTH, 0);
    let expected = FORCE * interface / STIFF + FORCE * (LENGTH - interface) / SOFT;
    let error = ((elongation - expected) / expected).abs();
    report_case(name, error * 100.0, &solution);
    error
}

#[test]
fn an_interface_on_an_element_boundary_matches_the_series_formula() {
    // The exact solution is piecewise linear with its kink on an element boundary, so it is in the
    // element space: the error is the solver's, bounded by 1e-5 as in the tension case.
    let error = relative_error("two_material_aligned", 5.0, Quadrature::FourPoint);
    assert!(error < 1e-5, "error {error}");
}

#[test]
fn an_interface_inside_elements_stays_within_the_uniform_strain_bound() {
    // Interface at x = 5.25, mid-cell of a 0.5-long cell. Exact compliance of the cut cell is
    // 0.25 / STIFF + 0.25 / SOFT = 1.25e-3; a uniform-strain cell at the mixed stiffness has
    // 0.5 / 625 = 0.8e-3. That gap of 0.45e-3 out of the bar's 24.25e-3 total compliance is 1.9%,
    // the most stiffness the mis-averaging of one cell can add. Limit 2% for the 14-point rule.
    let fourteen = relative_error("two_material_inside_14pt", 5.25, Quadrature::FourteenPoint);
    assert!(fourteen < 0.02, "14-point error {fourteen}");
    // Reported, not asserted: four samples per element are not an integration of the jump.
    relative_error("two_material_inside_4pt", 5.25, Quadrature::FourPoint);
}
