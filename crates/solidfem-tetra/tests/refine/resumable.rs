// Determinism and control: the same bits across runs and slice sizes, cancelling and resuming,
// and the named budget error.

use crate::shapes::plate_with_holes;
use crate::support::{constant_size, options_with_layers, recovered};
use solidfem_tetra::{RefineOptions, RefinePhase, Refiner, SolidMesh, StepStatus, TetraError};

fn solid() -> SolidMesh {
    recovered(&plate_with_holes(10.0, 8.0, 0.8, 5, 4, 1, &[(2, 2, true)]))
}

fn run(step_size: usize) -> SolidMesh {
    let mut refiner =
        Refiner::new(solid(), constant_size(2.5), options_with_layers(2)).expect("valid");
    loop {
        match refiner.step(step_size, &mut || false).expect("refines") {
            StepStatus::Done => return refiner.result().expect("finished"),
            StepStatus::Running => {}
            StepStatus::Cancelled => panic!("nothing asked to cancel"),
        }
    }
}

#[test]
fn two_runs_give_the_same_bits() {
    assert_eq!(run(usize::MAX), run(usize::MAX));
}

#[test]
fn output_does_not_depend_on_the_slice_size() {
    let whole = run(usize::MAX);
    for step_size in [1, 3, 17] {
        assert_eq!(run(step_size), whole, "sliced by {step_size}");
    }
}

#[test]
fn cancelling_mid_run_stops_and_a_later_step_finishes_with_the_same_result() {
    let mut refiner =
        Refiner::new(solid(), constant_size(2.5), options_with_layers(2)).expect("valid");
    let mut asked = 0;
    let status = refiner
        .step(usize::MAX, &mut || {
            asked += 1;
            asked > 50
        })
        .expect("refines");
    assert_eq!(status, StepStatus::Cancelled);
    assert!(
        matches!(refiner.result(), Err(TetraError::RefinementNotFinished)),
        "no result before it finishes"
    );
    assert_eq!(
        refiner.step(usize::MAX, &mut || false),
        Ok(StepStatus::Done)
    );
    assert_eq!(refiner.result().expect("finished"), run(usize::MAX));
}

#[test]
fn phases_run_in_order() {
    let mut refiner =
        Refiner::new(solid(), constant_size(2.5), options_with_layers(2)).expect("valid");
    let order = [
        RefinePhase::Layers,
        RefinePhase::Sizing,
        RefinePhase::Quality,
        RefinePhase::Done,
    ];
    let mut furthest = 0;
    loop {
        let at = order
            .iter()
            .position(|&p| p == refiner.progress().phase)
            .expect("known phase");
        assert!(at >= furthest, "phases must not go backwards");
        furthest = at;
        if refiner.step(1, &mut || false).expect("refines") == StepStatus::Done {
            break;
        }
    }
    assert_eq!(refiner.progress().phase, RefinePhase::Done);
}

#[test]
fn a_budget_too_small_for_the_work_is_the_named_error() {
    let options = RefineOptions {
        layers_through_walls: 2,
        max_added_points: 3,
        ..RefineOptions::default()
    };
    let mut refiner = Refiner::new(solid(), constant_size(2.5), options).expect("valid");
    let error = refiner
        .step(usize::MAX, &mut || false)
        .expect_err("the budget runs out");
    assert_eq!(
        error,
        TetraError::RefinementBudgetExhausted { added: 3, limit: 3 }
    );
}

#[test]
fn a_bad_size_function_is_named_rather_than_looping() {
    let mut refiner =
        Refiner::new(solid(), Box::new(|_| f64::NAN), RefineOptions::default()).expect("valid");
    let error = refiner
        .step(usize::MAX, &mut || false)
        .expect_err("a NaN size is refused");
    assert!(matches!(error, TetraError::InvalidSizeFunction { .. }));
}
