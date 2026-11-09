//! Prescribed displacement components: an enforced stretch of a cube gives the exact uniform
//! stress, rollers leave the free components free, a held value is exact, and a model the
//! prescribed components do not hold is refused by name.

use solidfem::elastic::{
    Material, Model, ModelError, Quadrature, StaticStudy, StepOutcome, StudyOptions,
};
use solidfem::mesh::{QuadraticTetMesh, TetMesh};

const STEEL: Material = Material {
    youngs_modulus: 200_000.0,
    poisson_ratio: 0.3,
};
const STRETCH: f64 = 0.001;

/// A unit cube of six linear tetrahedra with positive volume, upgraded to 10 nodes.
fn cube() -> QuadraticTetMesh {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let tets = vec![
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ];
    QuadraticTetMesh::from_linear(&TetMesh::new(points, tets).expect("the cube is valid"))
}

/// The held components of a uniaxial stretch along x: every node of the x = 0 face held in x, the
/// x = 1 face at the stretch in x, and just enough of the x = 0 face held in y and z.
fn stretched(mesh: QuadraticTetMesh) -> Model<Material> {
    let mut model = Model::new(mesh, STEEL);
    model.set_quadrature(Quadrature::FourPoint);
    let nodes: Vec<[f64; 3]> = model.mesh().nodes().to_vec();
    let at = |target: [f64; 3]| {
        nodes
            .iter()
            .position(|p| *p == target)
            .expect("a corner of the cube") as u32
    };
    for (node, p) in nodes.iter().enumerate() {
        if p[0] == 0.0 {
            model.prescribe_displacement(node as u32, 0, 0.0);
        }
        if p[0] == 1.0 {
            model.prescribe_displacement(node as u32, 0, STRETCH);
        }
    }
    model.prescribe_displacement(at([0.0, 0.0, 0.0]), 1, 0.0);
    model.prescribe_displacement(at([0.0, 0.0, 0.0]), 2, 0.0);
    model.prescribe_displacement(at([0.0, 1.0, 0.0]), 2, 0.0);
    model
}

fn solve(model: Model<Material>) -> solidfem::elastic::Solution {
    let mut study =
        StaticStudy::new(model, StudyOptions::default(), &mut || false).expect("the study builds");
    loop {
        match study.step(1000, &mut || false).expect("the solve runs") {
            StepOutcome::Converged(_) => break,
            StepOutcome::Running(_) => {}
            StepOutcome::Cancelled(_) => panic!("nothing cancels this solve"),
        }
    }
    study.solution(&mut || false).expect("a solution")
}

#[test]
fn an_enforced_stretch_gives_the_exact_uniform_stress() {
    let solution = solve(stretched(cube()));
    let expected = STEEL.youngs_modulus * STRETCH;
    for point in &solution.point_stresses {
        // A uniaxial state: E times the stretch along x and nothing else, to the solver's digits.
        assert!(
            (point.stress[0] - expected).abs() < 1e-5 * expected,
            "{:?} against {expected}",
            point.stress
        );
        for component in &point.stress[1..] {
            assert!(component.abs() < 1e-5 * expected, "{:?}", point.stress);
        }
    }
}

#[test]
fn a_prescribed_value_is_reproduced_exactly_and_the_free_components_stay_free() {
    let model = stretched(cube());
    let nodes = model.mesh().nodes().to_vec();
    let solution = solve(model);
    for (node, p) in nodes.iter().enumerate() {
        if p[0] == 1.0 {
            assert_eq!(solution.displacements[node][0], STRETCH);
        }
        if p[0] == 0.0 {
            assert_eq!(solution.displacements[node][0], 0.0);
        }
    }
    // The transverse contraction is uniform: u_y = -nu * STRETCH * y at the free far corner.
    let corner = nodes
        .iter()
        .position(|p| *p == [1.0, 1.0, 1.0])
        .expect("a corner");
    let expected = -STEEL.poisson_ratio * STRETCH;
    assert!((solution.displacements[corner][1] - expected).abs() < 1e-6 * STRETCH);
}

#[test]
fn prescribed_components_that_leave_a_rigid_motion_are_refused_by_name() {
    let mut model = Model::new(cube(), STEEL);
    let nodes = model.mesh().nodes().to_vec();
    for (node, p) in nodes.iter().enumerate() {
        if p[0] == 0.0 {
            model.prescribe_displacement(node as u32, 0, 0.0);
        }
    }
    // Only the x direction is held: y, z and the rotations about x remain free.
    let error = StaticStudy::new(model, StudyOptions::default(), &mut || false)
        .err()
        .expect("the study is refused");
    assert!(
        matches!(error, ModelError::DofsDoNotHold { rank, .. } if rank < 6),
        "{error}"
    );
}

#[test]
fn bad_prescribed_components_are_refused() {
    let mut model = stretched(cube());
    model.prescribe_displacement(0, 3, 0.0);
    assert!(matches!(
        StaticStudy::new(model, StudyOptions::default(), &mut || false),
        Err(ModelError::AxisOutOfRange { axis: 3 })
    ));

    let mut model = stretched(cube());
    model.prescribe_displacement(0, 1, f64::NAN);
    assert!(matches!(
        StaticStudy::new(model, StudyOptions::default(), &mut || false),
        Err(ModelError::NonFiniteLoad)
    ));

    let mut model = stretched(cube());
    model.fix_node(0);
    model.prescribe_displacement(0, 1, 0.0);
    assert!(matches!(
        StaticStudy::new(model, StudyOptions::default(), &mut || false),
        Err(ModelError::PrescribedOnFixedNode { node: 0 })
    ));
}
