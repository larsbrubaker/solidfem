// Determinism and slicing: the output is a pure function of the input, whatever the slice sizes,
// and a run can be cancelled and resumed between slices.

use crate::support::{triangulate, Rng};
use solidfem_tetra::{StepStatus, Tetrahedralization, Triangulator};

fn input() -> Vec<[f64; 3]> {
    // Mostly random, with a lattice of cospherical corners so the tie rule's order is exercised.
    let mut rng = Rng(5);
    let mut points: Vec<[f64; 3]> = (0..1500).map(|_| rng.point()).collect();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                points.push([f64::from(i) * 2.0, f64::from(j) * 2.0, f64::from(k) * 2.0]);
            }
        }
    }
    points
}

/// Runs to completion stepping `step_size` points at a time.
fn run_sliced(points: Vec<[f64; 3]>, step_size: usize) -> Tetrahedralization {
    let mut triangulator = Triangulator::new(points).expect("the input is valid");
    loop {
        match triangulator
            .step(step_size, &mut || false)
            .expect("every point inserts")
        {
            StepStatus::Done => return triangulator.result().expect("the run finished"),
            StepStatus::Running => {}
            StepStatus::Cancelled => panic!("nothing asked to cancel"),
        }
    }
}

#[test]
fn two_runs_give_the_same_bits() {
    let first = triangulate(input());
    let second = triangulate(input());
    assert_eq!(first, second);
}

#[test]
fn output_does_not_depend_on_the_step_size() {
    let whole = triangulate(input());
    for step_size in [1, 2, 7, 113] {
        assert_eq!(
            run_sliced(input(), step_size),
            whole,
            "stepping by {step_size} gave a different result"
        );
    }
}

#[test]
fn cancellation_stops_the_run_and_a_later_step_finishes_it() {
    let mut triangulator = Triangulator::new(input()).expect("the input is valid");
    let mut asked = 0;
    let status = triangulator
        .step(usize::MAX, &mut || {
            asked += 1;
            asked > 50
        })
        .expect("every point inserts");
    assert_eq!(status, StepStatus::Cancelled);
    assert_eq!(asked, 51, "the cancel check runs before each point");
    assert_eq!(
        triangulator.inserted(),
        4 + 50,
        "exactly the points before the cancellation are in"
    );
    assert!(!triangulator.is_done());

    assert_eq!(
        triangulator.step(usize::MAX, &mut || false),
        Ok(StepStatus::Done)
    );
    assert_eq!(
        triangulator.result().expect("finished"),
        triangulate(input())
    );
}

#[test]
fn a_cancelled_step_inserts_nothing_when_cancel_is_already_set() {
    let mut triangulator = Triangulator::new(input()).expect("the input is valid");
    let before = triangulator.inserted();
    assert_eq!(
        triangulator.step(usize::MAX, &mut || true),
        Ok(StepStatus::Cancelled)
    );
    assert_eq!(triangulator.inserted(), before);
}
