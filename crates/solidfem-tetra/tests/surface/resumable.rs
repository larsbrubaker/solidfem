// Determinism and slicing for surface recovery: a run is the same bits however it is sliced, and
// can be cancelled and resumed, with the phases reported as they go.

use crate::shapes::{hollow_box, rotated};
use solidfem_tetra::{SolidMesh, StepStatus, SurfaceMesher, SurfacePhase};

fn input() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    rotated(&hollow_box(10.0, 1.0, 4, 3))
}

fn run(step_size: usize) -> SolidMesh {
    let (points, triangles) = input();
    let mut mesher = SurfaceMesher::new(points, triangles).expect("valid");
    loop {
        match mesher.step(step_size, &mut || false).expect("recovers") {
            StepStatus::Done => return mesher.result().expect("finished"),
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
    let (points, triangles) = input();
    let mut mesher = SurfaceMesher::new(points, triangles).expect("valid");
    let mut asked = 0;
    let status = mesher
        .step(usize::MAX, &mut || {
            asked += 1;
            asked > 20
        })
        .expect("recovers");
    assert_eq!(status, StepStatus::Cancelled);
    assert!(mesher.result().is_err(), "no result before it finishes");
    assert_eq!(mesher.step(usize::MAX, &mut || false), Ok(StepStatus::Done));
    assert_eq!(mesher.result().expect("finished"), run(usize::MAX));
}

#[test]
fn phases_are_reported_in_order_and_the_report_counts_what_was_present() {
    let (points, triangles) = input();
    let total_triangles = triangles.len();
    let mut mesher = SurfaceMesher::new(points, triangles).expect("valid");
    assert_eq!(mesher.progress().phase, SurfacePhase::Delaunay);
    let order = [
        SurfacePhase::Delaunay,
        SurfacePhase::RecoveringEdges,
        SurfacePhase::RecoveringFaces,
        SurfacePhase::Classifying,
        SurfacePhase::Done,
    ];
    let mut furthest = 0;
    loop {
        let phase = mesher.progress().phase;
        let at = order.iter().position(|&p| p == phase).expect("known phase");
        assert!(at >= furthest, "phases must not go backwards");
        furthest = at;
        if mesher.step(1, &mut || false).expect("recovers") == StepStatus::Done {
            break;
        }
    }
    assert_eq!(mesher.progress().phase, SurfacePhase::Done);
    let report = mesher.report();
    assert_eq!(report.faces_total, total_triangles);
    assert!(report.edges_present_after_delaunay <= report.edges_total);
    assert!(report.faces_present_after_delaunay <= report.faces_total);
    assert_eq!(report.edges_failed + report.faces_failed, 0);
}
