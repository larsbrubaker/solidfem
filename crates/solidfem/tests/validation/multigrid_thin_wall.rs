// Multigrid on a wall two element layers thick. The through-wall pairing that gives some coarse
// rows a closest neighbour half a wall away must not narrow what their prolongator smoothing
// keeps, or the wall's bending modes stay in the error. Measured against a stocky bar solved the
// same way, because the plate's iteration count means nothing alone.

use solidfem::elastic::{
    Material, Model, MultigridOptions, PreconditionerChoice, StaticStudy, StepOutcome, StudyOptions,
};

use crate::support::{bar, faces_at, fix_root};

/// The most iterations a two-layer thin wall may take, as a multiple of a stocky bar's. This
/// number has no derivation: a V-cycle's iteration count depends on the hierarchy, the mesh and
/// where the plate's rounding floor stops it, and nothing gives it in closed form. It is a
/// regression guard chosen between two measured ratios: 4.3 with the coarse-level filter (150
/// and 35 iterations) and 8.9 with a filter set by each row's own closest neighbour (312 and 35).
/// Six is 1.4 times the first and the second is 1.5 times six. A deliberate change to the cycle
/// may move it, and should measure both ratios again.
const THIN_WALL_MULTIPLE: u64 = 6;

fn steel() -> Material {
    Material {
        youngs_modulus: 200_000.0,
        poisson_ratio: 0.3,
    }
}

/// The iterations a solve of a clamped, end-loaded box takes and the residual it stopped at: the
/// plate stops at its rounding floor short of the 1e-9 tolerance, the bar reaches it.
fn iterations(size: [f64; 3], divisions: [usize; 3]) -> (u64, f64) {
    let mesh = bar(size, divisions);
    let mut model = Model::new(mesh.clone(), steel());
    fix_root(&mut model);
    for face in faces_at(&mesh, size[0]) {
        model.add_traction(face, [0.0, 0.0, -1.0]);
    }
    let options = StudyOptions {
        preconditioner: PreconditionerChoice::Multigrid(MultigridOptions {
            threads: 1,
            ..MultigridOptions::default()
        }),
        ..StudyOptions::default()
    };
    let mut study = StaticStudy::new(model, options, &mut || false).expect("the study builds");
    loop {
        match study
            .step(usize::MAX, &mut || false)
            .expect("the solve runs")
        {
            StepOutcome::Converged(_) => break,
            StepOutcome::Running(_) => {}
            StepOutcome::Cancelled(_) => panic!("nothing cancelled"),
        }
    }
    let report = study.report();
    (report.iterations, report.relative_residual)
}

#[test]
fn a_two_layer_thin_plate_stays_within_a_stated_multiple_of_a_stocky_bar() {
    let (plate, plate_residual) = iterations([50.0, 50.0, 0.8], [20, 20, 2]);
    let (stocky, stocky_residual) = iterations([10.0, 1.0, 1.0], [20, 4, 4]);
    println!(
        "VALIDATION multigrid thin wall: plate {plate} iterations (residual {plate_residual:.2e}), stocky bar {stocky} (residual {stocky_residual:.2e})"
    );
    assert!(
        plate <= THIN_WALL_MULTIPLE * stocky,
        "the thin plate took {plate} iterations, more than {THIN_WALL_MULTIPLE} times the stocky bar's {stocky}"
    );
}
