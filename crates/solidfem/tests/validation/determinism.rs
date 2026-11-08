// Results must be the same bits across runs and however a solve is sliced: saved results and
// comparisons depend on it, and the web build solves in small slices between frames.

use solidfem::elastic::{Material, Model, Solution, StaticStudy, StepOutcome, StudyOptions};

use crate::support::{bar, faces_at, fix_root};

fn model() -> Model<Material> {
    let material = Material {
        youngs_modulus: 2100.0,
        poisson_ratio: 0.3,
    };
    let mesh = bar([6.0, 1.0, 1.0], [6, 2, 2]);
    let mut model = Model::new(mesh.clone(), material);
    fix_root(&mut model);
    for face in faces_at(&mesh, 6.0) {
        model.add_traction(face, [0.0, 3.0, -7.0]);
    }
    model
}

fn solve_in_slices(slice: usize) -> Solution {
    let mut study =
        StaticStudy::new(model(), StudyOptions::default(), &mut || false).expect("valid");
    loop {
        match study.step(slice, &mut || false).expect("solves") {
            StepOutcome::Converged(_) => break,
            StepOutcome::Running(progress) => assert!(progress.iteration > 0),
            StepOutcome::Cancelled(_) => panic!("nothing cancelled"),
        }
    }
    study.solution(&mut || false).expect("converged")
}

fn assert_identical(a: &Solution, b: &Solution) {
    assert_eq!(a.displacements.len(), b.displacements.len());
    for (p, q) in a.displacements.iter().zip(&b.displacements) {
        for axis in 0..3 {
            assert_eq!(p[axis].to_bits(), q[axis].to_bits());
        }
    }
    assert_eq!(a.point_stresses.len(), b.point_stresses.len());
    for (p, q) in a.point_stresses.iter().zip(&b.point_stresses) {
        assert_eq!(p.von_mises.to_bits(), q.von_mises.to_bits());
        for (s, t) in p.stress.iter().zip(&q.stress) {
            assert_eq!(s.to_bits(), t.to_bits());
        }
    }
    for (p, q) in a.nodal_von_mises.iter().zip(&b.nodal_von_mises) {
        assert_eq!(p.to_bits(), q.to_bits());
    }
}

#[test]
fn two_runs_give_identical_bits() {
    assert_identical(&solve_in_slices(100_000), &solve_in_slices(100_000));
}

#[test]
fn many_small_steps_give_identical_bits_to_one_long_solve() {
    assert_identical(&solve_in_slices(100_000), &solve_in_slices(7));
}
