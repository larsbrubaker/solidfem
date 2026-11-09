//! The hollow-box case against its production code: the exact overlap of a triangle with the load
//! disk on overlaps known in closed form, the load summing to 10 N on any mesh, the judge's
//! refinement and absolute limits, and the independent structured reference.
//!
//! The reference test is ignored because its solves take minutes; run it in release:
//! `cargo test --release --workspace --test benchmarks structured_reference -- --ignored --nocapture`

use std::f64::consts::PI;

use solidfem::elastic::{Report, Solution};
use solidfem::mesh::{box_tet_mesh, QuadraticTetMesh};
use solidfem_bench::benchmarks::cases::{
    all_cases, disk_triangle_overlap, structured_reference, Case, PATCH_W_REFERENCE,
};
use solidfem_bench::benchmarks::pipeline::{face_area, DensityResult};
use solidfem_bench::benchmarks::verdict::{LoadBalance, Verdict};

/// The relative tolerance on a closed-form overlap: a handful of `f64` products and inverse
/// trigonometric calls lose no more than a few units in the last place.
const AREA_TOL: f64 = 1e-12;

fn assert_area(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= AREA_TOL * expected.max(1.0),
        "{what}: {actual} against {expected}"
    );
}

#[test]
fn a_triangle_inside_the_disk_overlaps_by_its_own_area() {
    let triangle = [[-1.0, -1.0], [2.0, -1.0], [-1.0, 2.0]];
    assert_area(
        disk_triangle_overlap([0.0, 0.0], 10.0, triangle),
        4.5,
        "inside",
    );
}

#[test]
fn a_triangle_around_the_disk_overlaps_by_the_whole_disk() {
    let triangle = [[-100.0, -100.0], [300.0, -100.0], [-100.0, 300.0]];
    assert_area(
        disk_triangle_overlap([2.0, 3.0], 5.0, triangle),
        PI * 25.0,
        "around",
    );
}

#[test]
fn a_triangle_far_away_overlaps_by_nothing() {
    let triangle = [[100.0, 100.0], [110.0, 100.0], [100.0, 110.0]];
    assert_eq!(disk_triangle_overlap([0.0, 0.0], 3.0, triangle), 0.0);
}

#[test]
fn a_right_corner_at_the_centre_with_long_legs_overlaps_by_a_quarter_disk() {
    let triangle = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
    assert_area(
        disk_triangle_overlap([0.0, 0.0], 3.0, triangle),
        0.25 * PI * 9.0,
        "quarter",
    );
}

#[test]
fn a_straight_edge_through_the_centre_overlaps_by_a_half_disk_and_winding_does_not_matter() {
    let triangle = [[-20.0, 0.0], [20.0, 0.0], [0.0, 20.0]];
    let reversed = [triangle[2], triangle[1], triangle[0]];
    let half = 0.5 * PI * 9.0;
    assert_area(
        disk_triangle_overlap([0.0, 0.0], 3.0, triangle),
        half,
        "half",
    );
    assert_area(
        disk_triangle_overlap([0.0, 0.0], 3.0, reversed),
        half,
        "reversed",
    );
}

#[test]
fn an_edge_cutting_a_chord_overlaps_by_the_circular_segment() {
    // The disk's cap above y = d: r^2 acos(d/r) - d sqrt(r^2 - d^2).
    let (r, d) = (3.0_f64, 1.0_f64);
    let segment = r * r * (d / r).acos() - d * (r * r - d * d).sqrt();
    let triangle = [[-100.0, d], [100.0, d], [0.0, 100.0]];
    assert_area(
        disk_triangle_overlap([0.0, 0.0], r, triangle),
        segment,
        "segment",
    );
}

#[test]
fn triangles_of_a_grid_share_the_disk_with_nothing_lost_or_doubled() {
    // A grid whose lines cut the disk on every side, each square split along a diagonal.
    let (radius, centre) = (3.0, [20.1, 19.7]);
    let lines: Vec<f64> = (0..=50).map(|i| f64::from(i) * 0.9).collect();
    let mut total = 0.0;
    for x in lines.windows(2) {
        for y in lines.windows(2) {
            let (a, b, c, d) = ([x[0], y[0]], [x[1], y[0]], [x[1], y[1]], [x[0], y[1]]);
            total += disk_triangle_overlap(centre, radius, [a, b, c]);
            total += disk_triangle_overlap(centre, radius, [a, c, d]);
        }
    }
    assert!(
        (total - PI * radius * radius).abs() <= 1e-10,
        "the overlaps add to {total}"
    );
}

fn hollow_box_case() -> Case {
    all_cases()
        .into_iter()
        .find(|case| case.id == "hollow-box")
        .expect("the hollow-box case exists")
}

fn top_loaded_mesh(n: usize) -> QuadraticTetMesh {
    QuadraticTetMesh::from_linear(
        &box_tet_mesh([0.0; 3], [40.0, 40.0, 20.0], [n, n, 4]).expect("box"),
    )
}

#[test]
fn the_load_totals_ten_newtons_exactly_on_any_mesh() {
    let case = hollow_box_case();
    // Grids that place faces differently against the disk, so a selection by face centroids
    // would load a different area on each.
    for n in [7, 12, 23] {
        let mesh = top_loaded_mesh(n);
        let supports = (case.apply)(&mesh).expect("the disk lies on the top face");
        let force: f64 = supports
            .tractions
            .iter()
            .map(|(face, traction)| {
                assert_eq!((traction[0], traction[1]), (0.0, 0.0), "only downward");
                traction[2] * face_area(&mesh, face)
            })
            .sum();
        assert!(
            (force + 10.0).abs() <= 1e-9,
            "the {n} mesh is loaded with {force} N"
        );
    }
}

fn solution_of(mesh: &QuadraticTetMesh, displacement: [f64; 3]) -> Solution {
    Solution {
        displacements: vec![displacement; mesh.nodes().len()],
        point_stresses: Vec::new(),
        nodal_von_mises: Vec::new(),
        report: Report {
            total_dofs: 0,
            free_dofs: 0,
            nonzeros: 0,
            at_rounding_floor: false,
            iterations: 0,
            relative_residual: 0.0,
            assembly_seconds: 0.0,
            preconditioner_seconds: 0.0,
            multigrid: None,
            solve_seconds: 0.0,
            recovery_seconds: 0.0,
        },
    }
}

#[test]
fn a_rigid_downward_shift_reads_as_that_deflection() {
    let case = hollow_box_case();
    let mesh = top_loaded_mesh(12);
    let solution = solution_of(&mesh, [0.0, 0.0, -4.25e-3]);
    let values = (case.measure)(&mesh, &solution).expect("measured");
    assert_eq!(values[0].0, "patch_w");
    assert!(
        (values[0].1 - 4.25e-3).abs() <= 1e-15,
        "patch_w {} against 4.25e-3",
        values[0].1
    );
}

fn rows(patch_w: [f64; 3]) -> Vec<DensityResult> {
    patch_w
        .iter()
        .zip([2.0, 1.5, 1.0])
        .map(|(w, size)| DensityResult {
            label: format!("h={size}"),
            size,
            values: vec![("patch_w".to_string(), *w)],
            balance: LoadBalance::new([0.0; 3], [0.0; 3], 1.0),
            nodes: 0,
            elements: 0,
            free_dofs: 0,
            iterations: 0,
            residual: 0.0,
            min_dihedral: 0.0,
            seconds: 0.0,
        })
        .collect()
}

#[test]
fn the_judge_passes_a_converged_series_at_the_reference() {
    let case = hollow_box_case();
    let verdict = (case.judge)(&rows([5.30e-3, 5.38e-3, 5.45e-3]));
    assert_eq!(verdict, Verdict::Pass);
}

#[test]
fn the_judge_fails_a_series_that_stands_still_away_from_the_reference() {
    let case = hollow_box_case();
    match (case.judge)(&rows([5.0e-3, 5.05e-3, 5.1e-3])) {
        Verdict::Fail(reason) => {
            assert!(reason.contains("structured reference"), "{reason}");
        }
        other => panic!("expected a failure against the reference, got {other:?}"),
    }
}

#[test]
fn the_judge_fails_a_series_still_moving_near_the_reference() {
    let case = hollow_box_case();
    match (case.judge)(&rows([4.8e-3, 5.1e-3, 5.5e-3])) {
        Verdict::Fail(reason) => assert!(reason.contains("last refinement"), "{reason}"),
        other => panic!("expected a failure of the refinement limit, got {other:?}"),
    }
}

#[test]
#[ignore = "solves of about 140k, 500k and 1.8M free values, minutes in release; see this file's header"]
fn structured_reference_extrapolates_to_the_judged_value() {
    let coarse = structured_reference(1.0).expect("the coarse solve");
    let middle = structured_reference(0.5).expect("the middle solve");
    let fine = structured_reference(0.25).expect("the fine solve");
    eprintln!("{coarse:?}\n{middle:?}\n{fine:?}");
    let (w0, w1, w2) = (coarse.patch_w, middle.patch_w, fine.patch_w);
    let rate = ((w1 - w0) / (w2 - w1)).ln() / 2.0_f64.ln();
    let extrapolated = w2 + (w2 - w1) / (2.0_f64.powf(rate) - 1.0);
    eprintln!("observed rate {rate:.3}, extrapolated {extrapolated:.6e}");
    // The limit: the constant is stated to 3 figures, so rounds by up to 0.1%, and the rate
    // moving between 1.5 and 2.0 shifts this extrapolation by under 0.1% (the last change is
    // under 0.2% of the value and enters divided by 2^rate - 1): their sum.
    let error = ((extrapolated - PATCH_W_REFERENCE) / PATCH_W_REFERENCE).abs();
    assert!(
        error <= 0.002,
        "extrapolated {extrapolated:.6e} against {PATCH_W_REFERENCE:.6e}, error {:.3}%",
        100.0 * error
    );
}
