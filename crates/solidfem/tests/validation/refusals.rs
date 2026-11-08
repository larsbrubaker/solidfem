// What a study must refuse rather than answer, and cancellation. A model that cannot be solved
// names what is missing before any solving; it never returns numbers.

use solidfem::elastic::{
    Freedom, Material, Model, ModelError, StaticStudy, StepOutcome, StudyOptions,
};
use solidfem::mesh::{QuadraticTetMesh, TetMesh};

use crate::support::{bar, faces_at, fix_root};

const MATERIAL: Material = Material {
    youngs_modulus: 2100.0,
    poisson_ratio: 0.3,
};

fn study(model: Model<Material>) -> Result<StaticStudy<Material>, ModelError> {
    StaticStudy::new(model, StudyOptions::default(), &mut || false)
}

fn loaded_bar() -> Model<Material> {
    let mesh = bar([4.0, 1.0, 1.0], [4, 1, 1]);
    let mut model = Model::new(mesh.clone(), MATERIAL);
    for face in faces_at(&mesh, 4.0) {
        model.add_traction(face, [1.0, 0.0, 0.0]);
    }
    model
}

#[test]
fn no_fixed_nodes_is_refused() {
    assert_eq!(study(loaded_bar()).err(), Some(ModelError::NoFixedNodes));
}

#[test]
fn one_fixed_node_leaves_a_rotation_about_it() {
    let mut model = loaded_bar();
    model.fix_node(0);
    match study(model).err() {
        Some(ModelError::BodyNotHeld {
            freedom,
            held_nodes,
            ..
        }) => {
            assert_eq!(freedom, Freedom::RotationAboutPoint);
            assert_eq!(held_nodes, 1);
        }
        other => panic!("expected BodyNotHeld, got {other:?}"),
    }
}

#[test]
fn fixed_nodes_along_one_line_leave_a_rotation_about_it() {
    let mut model = loaded_bar();
    let line: Vec<u32> = (0..model.mesh().nodes().len() as u32)
        .filter(|n| {
            let p = model.mesh().nodes()[*n as usize];
            p[1].abs() < 1e-9 && p[2].abs() < 1e-9
        })
        .collect();
    assert!(line.len() >= 3);
    for node in line {
        model.fix_node(node);
    }
    match study(model).err() {
        Some(ModelError::BodyNotHeld { freedom, .. }) => {
            assert_eq!(freedom, Freedom::RotationAboutLine);
        }
        other => panic!("expected BodyNotHeld, got {other:?}"),
    }
}

#[test]
fn a_second_piece_with_nothing_holding_it_is_refused() {
    // Two separate tetrahedra; only the first is fixed.
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [5.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [5.0, 1.0, 0.0],
        [5.0, 0.0, 1.0],
    ];
    let linear = TetMesh::new(positions, vec![[0, 1, 2, 3], [4, 5, 6, 7]]).expect("mesh");
    let mesh = QuadraticTetMesh::from_linear(&linear);
    let mut model = Model::new(mesh, MATERIAL);
    for node in 0..3 {
        model.fix_node(node);
    }
    model.add_nodal_force(5, [0.0, 0.0, 1.0]);
    match study(model).err() {
        Some(ModelError::BodyNotHeld {
            first_tet, freedom, ..
        }) => {
            assert_eq!(first_tet, 1);
            assert_eq!(freedom, Freedom::Free);
        }
        other => panic!("expected BodyNotHeld, got {other:?}"),
    }
}

#[test]
fn a_model_with_no_load_is_refused() {
    let mesh = bar([4.0, 1.0, 1.0], [4, 1, 1]);
    let mut model = Model::new(mesh, MATERIAL);
    fix_root(&mut model);
    assert_eq!(study(model).err(), Some(ModelError::NoEffectiveLoad));
}

#[test]
fn a_load_only_on_fixed_nodes_is_refused() {
    let mesh = bar([4.0, 1.0, 1.0], [4, 1, 1]);
    let mut model = Model::new(mesh, MATERIAL);
    fix_root(&mut model);
    model.add_nodal_force(0, [0.0, 0.0, 5.0]);
    assert_eq!(study(model).err(), Some(ModelError::NoEffectiveLoad));
}

#[test]
fn cancel_stops_assembly() {
    let mut model = loaded_bar();
    fix_root(&mut model);
    let result = StaticStudy::new(model, StudyOptions::default(), &mut || true);
    assert_eq!(result.err(), Some(ModelError::Cancelled));
}

#[test]
fn cancel_stops_the_solve_and_leaves_no_result() {
    let mut model = loaded_bar();
    fix_root(&mut model);
    let mut study = study(model).expect("valid");
    study.step(2, &mut || false).expect("runs");
    let outcome = study.step(100, &mut || true).expect("cancels");
    assert!(matches!(outcome, StepOutcome::Cancelled(_)));
    assert_eq!(
        study.solution(&mut || false).err(),
        Some(ModelError::NotFinished)
    );
}
