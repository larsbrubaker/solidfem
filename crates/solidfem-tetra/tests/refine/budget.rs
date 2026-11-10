// The tetrahedron budget as a promise: a run that finishes has at most `max_tets` tetrahedra, the
// size stage says when it stopped there, and a solid already over the budget before or after the
// layers it must have is refused by name with its count. The same bits however a run is sliced.

use crate::shapes::{prism, subdivided_box, Surface};
use crate::support::{assert_contracts, constant_size, min_dihedral, recovered, run_refine};
use solidfem_tetra::{RefineOptions, RefinePhase, Refiner, SolidMesh, StepStatus, TetraError};

/// A 10 x 10 x 2 plate with a coarse surface: a fine interior size fills it with far more
/// tetrahedra than the surface alone needs.
fn plate() -> Surface {
    subdivided_box([0.0, 0.0, 0.0], [10.0, 10.0, 2.0], 4)
}

/// Thin enough for layers through the plate's thickness, fine enough inside to overshoot.
const FILL_SIZE: f64 = 0.6;

fn options(max_tets: usize) -> RefineOptions {
    RefineOptions {
        layers_through_walls: 2,
        max_tets,
        ..RefineOptions::default()
    }
}

fn worst_dihedral(mesh: &SolidMesh) -> f64 {
    mesh.tets
        .iter()
        .map(|tet| min_dihedral(mesh, tet))
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn a_fill_that_would_overshoot_stops_at_the_budget() {
    let surface = plate();
    let solid = recovered(&surface);
    let (free, free_report) =
        run_refine(solid.clone(), constant_size(FILL_SIZE), options(usize::MAX)).expect("refines");
    let budget = free.tets.len() / 2;
    assert!(
        !free_report.size_stopped_at_budget,
        "an unlimited run never stops at a budget"
    );
    let (capped, report) =
        run_refine(solid.clone(), constant_size(FILL_SIZE), options(budget)).expect("refines");
    assert!(
        capped.tets.len() <= budget,
        "{} tetrahedra over a budget of {budget}",
        capped.tets.len()
    );
    assert!(report.size_stopped_at_budget, "the stop is reported");
    assert_contracts(&capped, &solid, &surface);
    let (free_worst, capped_worst) = (worst_dihedral(&free), worst_dihedral(&capped));
    eprintln!(
        "unbudgeted {} tets worst {free_worst:.3} deg; budget {budget}: {} tets worst {capped_worst:.3} deg",
        free.tets.len(),
        capped.tets.len()
    );
    assert!(capped_worst > 0.0, "every tetrahedron stays positive");
}

#[test]
fn a_surface_already_over_the_budget_is_refused_before_starting() {
    let solid = recovered(&subdivided_box([0.0; 3], [10.0, 10.0, 2.0], 12));
    let tets = solid.tets.len();
    let budget = tets - 1;
    let error = Refiner::new(solid, constant_size(FILL_SIZE), options(budget))
        .err()
        .expect("refused");
    assert_eq!(
        error,
        TetraError::TetBudgetExceeded {
            tets,
            budget,
            phase: RefinePhase::Layers,
        }
    );
}

#[test]
fn layers_that_cross_the_budget_are_refused_after_the_layer_stage() {
    let solid = recovered(&plate());
    let budget = solid.tets.len();
    let mut refiner =
        Refiner::new(solid, constant_size(FILL_SIZE), options(budget)).expect("not over yet");
    let error = refiner
        .step(usize::MAX, &mut || false)
        .expect_err("the layers cross it");
    let TetraError::TetBudgetExceeded {
        tets,
        budget: named,
        phase,
    } = error
    else {
        panic!("expected the budget refusal, got {error:?}");
    };
    assert!(tets > budget, "{tets} is not over {budget}");
    assert_eq!((named, phase), (budget, RefinePhase::Sizing));
}

fn sliced(step_size: usize, budget: usize) -> SolidMesh {
    let mut refiner = Refiner::new(
        recovered(&plate()),
        constant_size(FILL_SIZE),
        options(budget),
    )
    .expect("valid");
    loop {
        match refiner.step(step_size, &mut || false).expect("refines") {
            StepStatus::Done => return refiner.result().expect("finished"),
            StepStatus::Running => {}
            StepStatus::Cancelled => panic!("nothing asked to cancel"),
        }
    }
}

#[test]
fn a_budgeted_run_does_not_depend_on_the_slice_size() {
    let free = sliced(usize::MAX, usize::MAX);
    let budget = free.tets.len() / 2;
    let whole = sliced(usize::MAX, budget);
    assert!(whole.tets.len() <= budget);
    for step_size in [1, 3, 17] {
        assert_eq!(sliced(step_size, budget), whole, "sliced by {step_size}");
    }
}

#[test]
fn the_refusal_names_the_count_in_plain_words() {
    let error = TetraError::TetBudgetExceeded {
        tets: 120,
        budget: 100,
        phase: RefinePhase::Sizing,
    };
    let message = error.to_string();
    assert!(
        message.contains("120") && message.contains("20 over") && message.contains("100"),
        "{message}"
    );
    assert!(!message.contains("  "), "a run of spaces in: {message}");
}

fn options_without_layers(max_tets: usize) -> RefineOptions {
    RefineOptions {
        max_tets,
        ..RefineOptions::default()
    }
}

#[test]
fn without_layers_the_size_stage_keeps_the_budget() {
    let surface = plate();
    let solid = recovered(&surface);
    let (free, _) = run_refine(
        solid.clone(),
        constant_size(FILL_SIZE),
        options_without_layers(usize::MAX),
    )
    .expect("refines");
    let budget = free.tets.len() / 2;
    let (capped, report) = run_refine(
        solid.clone(),
        constant_size(FILL_SIZE),
        options_without_layers(budget),
    )
    .expect("refines");
    assert!(
        capped.tets.len() <= budget,
        "{} over {budget}",
        capped.tets.len()
    );
    assert!(report.size_stopped_at_budget);
    assert_contracts(&capped, &solid, &surface);
}

#[test]
fn without_layers_a_solid_over_the_budget_is_refused_before_sizing() {
    let solid = recovered(&plate());
    let tets = solid.tets.len();
    let error = Refiner::new(
        solid,
        constant_size(FILL_SIZE),
        options_without_layers(tets - 1),
    )
    .err()
    .expect("refused");
    assert_eq!(
        error,
        TetraError::TetBudgetExceeded {
            tets,
            budget: tets - 1,
            phase: RefinePhase::Sizing,
        }
    );
}

#[test]
fn the_quality_pass_refuses_moves_that_would_cross_the_budget() {
    // No layers and a size no edge exceeds: the size stage adds nothing, so with the budget at
    // the recovered count it fits. The curved wall's slivers need quality points (as
    // `slivers.rs` measures), each adding tetrahedra, so the quality pass meets the limit.
    let solid = recovered(&prism(128, 50.0, 5.0, false));
    let budget = solid.tets.len();
    let (capped, report) =
        run_refine(solid, constant_size(1.0e6), options_without_layers(budget)).expect("refines");
    assert!(!report.size_stopped_at_budget, "the size stage fits");
    assert!(
        report.moves_refused_for_budget > 0,
        "the quality pass met the limit"
    );
    assert!(capped.tets.len() <= budget);
}
