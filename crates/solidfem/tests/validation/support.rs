// Shared by the validation cases: building a bar, selecting faces and stations, running a study to
// the end through the public `step` API, and printing the measured numbers.

use std::time::Instant;

use solidfem::elastic::{
    Clock, MaterialField, Model, Solution, StaticStudy, StepOutcome, StudyOptions,
};
use solidfem::mesh::{box_tet_mesh, BoundaryFace, QuadraticTetMesh};

/// Distance within which a node counts as on a plane.
const ON_PLANE: f64 = 1e-9;

/// A box bar along x from the origin.
pub fn bar(size: [f64; 3], divisions: [usize; 3]) -> QuadraticTetMesh {
    QuadraticTetMesh::from_linear(&box_tet_mesh([0.0; 3], size, divisions).expect("box"))
}

/// Fixes every node on the plane x = 0.
pub fn fix_root<M: MaterialField>(model: &mut Model<M>) {
    let nodes: Vec<u32> = (0..model.mesh().nodes().len() as u32)
        .filter(|n| model.mesh().nodes()[*n as usize][0].abs() < ON_PLANE)
        .collect();
    for node in nodes {
        model.fix_node(node);
    }
}

/// The boundary faces lying entirely on the plane x = `x`.
pub fn faces_at(mesh: &QuadraticTetMesh, x: f64) -> Vec<BoundaryFace> {
    mesh.boundary_faces()
        .into_iter()
        .filter(|f| {
            f.nodes[..3]
                .iter()
                .all(|n| (mesh.nodes()[*n as usize][0] - x).abs() < ON_PLANE)
        })
        .collect()
}

/// Node indices on the plane x = `x`.
pub fn nodes_at(mesh: &QuadraticTetMesh, x: f64) -> Vec<usize> {
    mesh.nodes()
        .iter()
        .enumerate()
        .filter(|(_, p)| (p[0] - x).abs() < ON_PLANE)
        .map(|(i, _)| i)
        .collect()
}

/// The mean of displacement component `axis` over the plane x = `x`.
pub fn mean_displacement(mesh: &QuadraticTetMesh, solution: &Solution, x: f64, axis: usize) -> f64 {
    let nodes = nodes_at(mesh, x);
    assert!(!nodes.is_empty(), "no nodes at x = {x}");
    let sum: f64 = nodes.iter().map(|n| solution.displacements[*n][axis]).sum();
    sum / nodes.len() as f64
}

/// Runs a study to convergence in slices of 1000 iterations with a wall-clock.
pub fn run<M: MaterialField>(model: Model<M>) -> Solution {
    let start = Instant::now();
    let clock: Clock = Box::new(move || start.elapsed().as_secs_f64());
    let options = StudyOptions {
        clock: Some(clock),
        ..StudyOptions::default()
    };
    let mut study = StaticStudy::new(model, options, &mut || false).expect("a valid model");
    loop {
        match study.step(1000, &mut || false).expect("the solve succeeds") {
            StepOutcome::Converged(_) => break,
            StepOutcome::Running(_) => {}
            StepOutcome::Cancelled(_) => panic!("nothing cancelled this solve"),
        }
    }
    study.solution(&mut || false).expect("a converged solve")
}

/// Prints one case's numbers on a line a run with --no-capture can collect.
pub fn report_case(name: &str, error: f64, solution: &Solution) {
    let r = solution.report;
    println!(
        "VALIDATION {name}: error={error:.3e}% total_dofs={} free_dofs={} iterations={} residual={:.2e} floor={} assembly_s={:.4} solve_s={:.4} recovery_s={:.4}",
        r.total_dofs, r.free_dofs, r.iterations, r.relative_residual, r.at_rounding_floor, r.assembly_seconds, r.solve_seconds, r.recovery_seconds
    );
}
