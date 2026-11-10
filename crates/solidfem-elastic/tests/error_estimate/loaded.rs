// The estimate on a loaded part with free surfaces: a cantilever clamped at one end and sheared
// at the other, where the elements cannot represent the solution (the clamp's edges carry a weak
// singularity). It must fall as the mesh refines and stay within a factor of two of the true
// energy-norm error.
//
// The true error comes from the energy identity. The clamp holds every displacement at zero and
// the uniform end traction's nodal loads are integrated exactly (a quadratic face's shape
// functions on a flat face, by the degree-4 triangle rule), so the solution space is a subspace
// of the exact one and the solution is its energy projection: `‖u - u_h‖² = ‖u‖² - ‖u_h‖²`. The
// exact `‖u‖²` is extrapolated from four nested meshes by the rate the three finest show, and
// `‖u_h‖²` is integrated here from the element strains with the 14-point rule.

use solidfem_elastic::{
    point_geometry, strain_at, stress_at, Material, Model, MultigridOptions, PcgOptions,
    PreconditionerChoice, Quadrature, StaticStudy, StudyOptions,
};
use solidfem_mesh::QuadraticTetMesh;

use super::support::box_mesh;

const LENGTH: f64 = 4.0;
const SIZE: [f64; 3] = [LENGTH, 1.0, 1.0];
const MATERIAL: Material = Material {
    youngs_modulus: 1.0,
    poisson_ratio: 0.3,
};

fn cantilever(cells: usize) -> StaticStudy<Material> {
    let mesh: QuadraticTetMesh = box_mesh(SIZE, [4 * cells, cells, cells]);
    let end_faces: Vec<_> = mesh
        .boundary_faces()
        .into_iter()
        .filter(|face| {
            face.nodes
                .iter()
                .all(|n| mesh.nodes()[*n as usize][0] == LENGTH)
        })
        .collect();
    let clamped: Vec<u32> = (0..mesh.nodes().len() as u32)
        .filter(|n| mesh.nodes()[*n as usize][0] == 0.0)
        .collect();
    let mut model = Model::new(mesh, MATERIAL);
    for node in clamped {
        model.fix_node(node);
    }
    for face in end_faces {
        model.add_traction(face, [0.0, 0.0, -1.0]);
    }
    let options = StudyOptions {
        pcg: PcgOptions {
            tolerance: 1e-12,
            max_iterations: None,
        },
        preconditioner: PreconditionerChoice::Multigrid(MultigridOptions::default()),
        clock: None,
    };
    let mut study = StaticStudy::new(model, options, &mut || false).expect("study builds");
    study.step(usize::MAX, &mut || false).expect("solves");
    study
}

/// `‖u_h‖²`, integrated from the element strains.
fn energy_norm_squared(study: &StaticStudy<Material>, displacements: &[[f64; 3]]) -> f64 {
    let mesh = study.mesh();
    let mut sum = 0.0;
    for tet in mesh.tets() {
        let nodes = tet.map(|n| mesh.nodes()[n as usize]);
        let u = tet.map(|n| displacements[n as usize]);
        for (index, q) in Quadrature::FourteenPoint.points().iter().enumerate() {
            let geometry = point_geometry(&nodes, q.natural, index).expect("integrates");
            let strain = strain_at(&geometry, &u);
            let stress = stress_at(&MATERIAL, &strain);
            let density: f64 = stress.iter().zip(&strain).map(|(s, e)| s * e).sum();
            sum += q.weight * geometry.det_j / 6.0 * density;
        }
    }
    sum
}

#[test]
fn a_clamped_cantilever_estimate_falls_and_stays_within_a_factor_of_two() {
    let mut rows = Vec::new();
    for cells in [1usize, 2, 4, 8] {
        let mut study = cantilever(cells);
        let (solution, estimate) = study
            .solution_with_error_estimate(&mut || false)
            .expect("estimates");
        let energy = energy_norm_squared(&study, &solution.displacements);
        rows.push((cells, energy, estimate));
    }
    // Aitken extrapolation from the three finest energies: the rate they show, then the limit.
    let (e2, e3, e4) = (rows[1].1, rows[2].1, rows[3].1);
    let ratio = (e3 - e2) / (e4 - e3);
    assert!(
        ratio > 1.0,
        "the energies converge monotonically: ratio {ratio}"
    );
    let exact = e4 + (e4 - e3) / (ratio - 1.0);
    eprintln!(
        "energy rate {:.3} (2^rate per halving), extrapolated ‖u‖² {exact:.6e}",
        ratio.log2()
    );
    eprintln!("cells deep  eta        estimated   true        effectivity");
    let mut previous_eta = f64::INFINITY;
    for (cells, energy, estimate) in &rows[..3] {
        let truth = (exact - energy).sqrt();
        let estimated = estimate.error_norm_squared.sqrt();
        let effectivity = estimated / truth;
        eprintln!(
            "{cells:>10}  {:.4e}  {estimated:.4e}  {truth:.4e}  {effectivity:.3}",
            estimate.relative_error
        );
        assert!(
            estimate.relative_error < previous_eta,
            "{cells} cells deep: η {} did not fall",
            estimate.relative_error
        );
        previous_eta = estimate.relative_error;
        // A linear patch fit is the same order as the element stress, so theory gives the
        // estimate only equivalence to the true error, not exactness, and the one-sided patches
        // along the clamp's singular edges read their stress low. What a user may take from `η`
        // is the order of the error, so the band is a factor of two either way. The finest mesh
        // is left out: its true error is the extrapolation's own error.
        assert!(
            (0.5..=2.0).contains(&effectivity),
            "{cells} cells deep: effectivity {effectivity}"
        );
    }
}
