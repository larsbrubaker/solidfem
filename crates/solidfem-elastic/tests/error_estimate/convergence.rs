// The estimate on a smooth field the elements cannot represent: it falls at the rate quadratic
// elements converge at, it tracks the true energy-norm error, and slicing never changes it.
//
// The field is `u = ∇φ` with `φ = e^x cos(y/√2) cos(z/√2)`, harmonic, so `div u = 0` and
// `Δu = ∇Δφ = 0`: it satisfies the equations of elasticity with no body force for any material,
// and its strain is the Hessian of `φ`. Every boundary node of the unit cube is held at it.
// Because the boundary values are interpolated rather than exact, the energy identity
// `‖u - u_h‖² = ‖u‖² - ‖u_h‖²` of a purely loaded problem does not hold here, so the true error
// is integrated directly: `∫ (ε - ε_h) : D : (ε - ε_h)` with the 14-point rule (degree five),
// whose own error on this smooth integrand is orders below the discretization error measured.

use solidfem_elastic::{
    point_geometry, strain_at, stress_at, Material, ModelError, Quadrature, StaticStudy,
};

use super::support::{box_mesh, held_study, solve_and_estimate};

const SIZE: [f64; 3] = [1.0; 3];
const MATERIAL: Material = Material {
    youngs_modulus: 1.0,
    poisson_ratio: 0.3,
};

fn wave() -> f64 {
    std::f64::consts::FRAC_1_SQRT_2
}

fn displacement(p: [f64; 3]) -> [f64; 3] {
    let a = wave();
    let e = p[0].exp();
    let (sy, cy) = (a * p[1]).sin_cos();
    let (sz, cz) = (a * p[2]).sin_cos();
    [e * cy * cz, -a * e * sy * cz, -a * e * cy * sz]
}

/// The exact strain `[xx, yy, zz, 2xy, 2yz, 2zx]`: the Hessian of `φ`.
fn strain(p: [f64; 3]) -> [f64; 6] {
    let a = wave();
    let e = p[0].exp();
    let (sy, cy) = (a * p[1]).sin_cos();
    let (sz, cz) = (a * p[2]).sin_cos();
    [
        e * cy * cz,
        -a * a * e * cy * cz,
        -a * a * e * cy * cz,
        2.0 * (-a * e * sy * cz),
        2.0 * (a * a * e * sy * sz),
        2.0 * (-a * e * cy * sz),
    ]
}

/// The true `‖u - u_h‖`, integrated element by element.
fn true_error(study: &StaticStudy<Material>, displacements: &[[f64; 3]]) -> f64 {
    let mesh = study.mesh();
    let mut sum = 0.0;
    for tet in mesh.tets() {
        let nodes = tet.map(|n| mesh.nodes()[n as usize]);
        let u = tet.map(|n| displacements[n as usize]);
        for (index, q) in Quadrature::FourteenPoint.points().iter().enumerate() {
            let geometry = point_geometry(&nodes, q.natural, index).expect("integrates");
            let element = strain_at(&geometry, &u);
            let exact = strain(geometry.position);
            let mut difference = [0.0; 6];
            for k in 0..6 {
                difference[k] = exact[k] - element[k];
            }
            let stress = stress_at(&MATERIAL, &difference);
            let density: f64 = stress.iter().zip(&difference).map(|(s, e)| s * e).sum();
            sum += q.weight * geometry.det_j / 6.0 * density;
        }
    }
    sum.sqrt()
}

#[test]
fn the_estimate_falls_at_the_quadratic_rate_and_tracks_the_true_error() {
    let mut rows = Vec::new();
    for cells in [2usize, 4, 8] {
        let mut study = held_study(box_mesh(SIZE, [cells; 3]), SIZE, MATERIAL, displacement);
        let (solution, estimate) = solve_and_estimate(&mut study);
        let truth = true_error(&study, &solution.displacements);
        let estimated = estimate.error_norm_squared.sqrt();
        rows.push((cells, estimate.relative_error, estimated, truth));
    }
    eprintln!("cells  eta        estimated   true        effectivity");
    for (cells, eta, estimated, truth) in &rows {
        eprintln!(
            "{cells:>5}  {eta:.4e}  {estimated:.4e}  {truth:.4e}  {:.3}",
            estimated / truth
        );
    }
    // Quadratic elements on a smooth solution converge as h² in the energy norm, and the
    // recovered stress is fitted to the element stress, so the estimate falls as h² too: a rate
    // of 2 per halving. The next term of the error's expansion in h is smaller by a factor of
    // order h, which at the finest pair (h = 1/4 to 1/8) can move the observed rate by
    // log2(1 + h) ≈ 0.3 either way, so the rates are accepted in [1.7, 2.3].
    let (fine, finer) = (rows[1], rows[2]);
    for (name, coarse_value, fine_value) in [
        ("estimate", fine.2, finer.2),
        ("true error", fine.3, finer.3),
    ] {
        let rate = (coarse_value / fine_value).log2();
        assert!(
            (1.7..=2.3).contains(&rate),
            "{name} rate {rate} from {coarse_value} to {fine_value}"
        );
    }
    // A linear patch fit reproduces the stress only to O(h²), the same order as the element
    // stress, so the estimate is not asymptotically exact: theory gives an effectivity that
    // settles to a constant of order one fixed by the solution and the mesh pattern, not one.
    // An estimate within a factor of two either way is what a user can read as "about this
    // converged"; the band says that and no more.
    for (cells, _, estimated, truth) in &rows[1..] {
        let effectivity = estimated / truth;
        assert!(
            (0.5..=2.0).contains(&effectivity),
            "{cells} cells: effectivity {effectivity}"
        );
    }
}

#[test]
fn an_estimate_in_slices_gives_the_bits_of_one_run() {
    let mut study = held_study(box_mesh(SIZE, [2; 3]), SIZE, MATERIAL, displacement);
    let (_, whole) = solve_and_estimate(&mut study);
    let mut recovery = study.begin_recovery();
    while !study
        .continue_recovery(&mut recovery, 7, &mut || false)
        .expect("recovers")
    {}
    let mut estimation = study.begin_error_estimate(recovery);
    let mut slices = 0;
    while !study
        .continue_error_estimate(&mut estimation, 5, &mut || false)
        .expect("estimates")
    {
        slices += 1;
    }
    assert!(slices > 1, "the estimate ran in more than one slice");
    let sliced = study.finish_error_estimate(&estimation).expect("finished");
    assert_eq!(sliced, whole);
    // An unfinished estimation refuses rather than reporting a partial sum.
    let mut partial = study.begin_error_estimate(estimation.into_recovery());
    assert_eq!(
        study.finish_error_estimate(&partial),
        Err(ModelError::NotFinished)
    );
    // Cancelled part-way, it keeps the elements already done and resumes to the same bits.
    let mut calls = 0;
    let mut cancel_on_fourth = || {
        calls += 1;
        calls == 4
    };
    assert_eq!(
        study.continue_error_estimate(&mut partial, usize::MAX, &mut cancel_on_fourth),
        Err(ModelError::Cancelled)
    );
    assert_eq!(partial.elements_done(), 3);
    let done = study
        .continue_error_estimate(&mut partial, usize::MAX, &mut || false)
        .expect("resumes");
    assert!(done);
    assert_eq!(study.finish_error_estimate(&partial), Ok(whole));
}
