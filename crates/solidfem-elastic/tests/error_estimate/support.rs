// Shared set-up for the error estimate tests: a structured box whose every boundary node is held
// at an exact displacement field, solved to the rounding floor, so the finite element solution
// differs from the exact field only by the discretization the estimate is meant to measure.

use solidfem_elastic::{
    ErrorEstimate, MaterialField, Model, PcgOptions, Solution, StaticStudy, StudyOptions,
};
use solidfem_mesh::{box_tet_mesh, QuadraticTetMesh};

/// A box from the origin of `size`, `cells` per axis, as 10-node elements.
pub fn box_mesh(size: [f64; 3], cells: [usize; 3]) -> QuadraticTetMesh {
    QuadraticTetMesh::from_linear(&box_tet_mesh([0.0; 3], size, cells).expect("box"))
}

/// A study of `mesh` with every boundary node prescribed to `exact` and no other load, solved.
/// The solve asks for a relative residual of 1e-14, below what f64 resolves on these models, so
/// it stops at the rounding floor.
pub fn held_study<M: MaterialField>(
    mesh: QuadraticTetMesh,
    size: [f64; 3],
    material: M,
    exact: impl Fn([f64; 3]) -> [f64; 3],
) -> StaticStudy<M> {
    let mut study = unsolved_held_study(mesh, size, material, exact);
    study.step(usize::MAX, &mut || false).expect("solves");
    study
}

/// [`held_study`] before any solver iteration.
pub fn unsolved_held_study<M: MaterialField>(
    mesh: QuadraticTetMesh,
    size: [f64; 3],
    material: M,
    exact: impl Fn([f64; 3]) -> [f64; 3],
) -> StaticStudy<M> {
    // Box coordinates on a face are exactly 0 or the box length (`box_tet_mesh`).
    let on_boundary = |p: [f64; 3]| (0..3).any(|axis| p[axis] == 0.0 || p[axis] == size[axis]);
    let boundary: Vec<(u32, [f64; 3])> = mesh
        .nodes()
        .iter()
        .enumerate()
        .filter(|(_, p)| on_boundary(**p))
        .map(|(node, p)| (node as u32, exact(*p)))
        .collect();
    let mut model = Model::new(mesh, material);
    for (node, value) in boundary {
        for (axis, component) in value.iter().enumerate() {
            model.prescribe_displacement(node, axis, *component);
        }
    }
    let options = StudyOptions {
        pcg: PcgOptions {
            tolerance: 1e-14,
            max_iterations: None,
        },
        ..StudyOptions::default()
    };
    StaticStudy::new(model, options, &mut || false).expect("study builds")
}

/// The solution and its error estimate, in one call.
pub fn solve_and_estimate<M: MaterialField>(
    study: &mut StaticStudy<M>,
) -> (Solution, ErrorEstimate) {
    study
        .solution_with_error_estimate(&mut || false)
        .expect("estimates")
}
