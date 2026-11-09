//! The gear-tooth case against its production code: the structured mesh fits the tooth and
//! carries the case's own load; the section stresses of a solved tooth follow beam theory, the
//! one check that does not rest on the elements and solver; the judge's reference, its limit and
//! its refinement check; and a real structured refinement extrapolating to the judged value.
//!
//! The reference test is ignored because its solves take minutes; run it in release:
//! `cargo test --release --workspace --test benchmarks gear_reference -- --ignored --nocapture`

use solidfem_bench::benchmarks::cases::{
    all_cases, beam_section_ratios, gear_structured_reference, half_model_supports,
    lewis_kf_information, solve_structured, structured_tooth, Case, ToothGrid, ROOT_PEAK_REFERENCE,
    ROOT_PEAK_REFERENCE_GRIDS,
};
use solidfem_bench::benchmarks::pipeline::{face_area, DensityResult};
use solidfem_bench::benchmarks::verdict::{LoadBalance, Verdict};

/// A coarse grid: seconds to solve, only the mesh, the load and the section stresses read from it.
const COARSE: ToothGrid = ToothGrid {
    across: 24,
    side_cells: [6, 8, 24, 30],
    half_layers: 2,
};

/// The half tooth carries half the case's force, tangentially.
#[test]
fn the_half_tooth_carries_half_the_benchmark_load() {
    let (mesh, _) = structured_tooth(COARSE).expect("the coarse grid");
    let supports = half_model_supports(&mesh).expect("the tip is found");
    let mut force = [0.0; 3];
    for (face, traction) in &supports.tractions {
        for axis in 0..3 {
            force[axis] += traction[axis] * face_area(&mesh, face);
        }
    }
    assert!(
        (force[0] - 500.0).abs() <= 1e-6 * 500.0 && force[1].abs() <= 1e-6 * 500.0,
        "half-tooth resultant {force:?} N"
    );
    assert!(!supports.fixed.is_empty(), "no clamped node");
    assert!(!supports.prescribed.is_empty(), "no symmetry node");
}

/// The bending stress across the flank above the fillet follows the beam section formula. It is
/// the one check on this case that does not use the elements, solver or stress recovery to set
/// its expected value: the section's moment comes from the applied load and its thickness from
/// the outline. The ratio is a fit of `sigma_yy` across the section against `6 F l / (b t^2)`.
///
/// The limit: Equilibrium fixes the moment of any section exactly, so a departure is the stress
/// profile not being the straight line the beam formula draws. The worst case is the steepest
/// flank, y = 42, a half angle of about 0.41 rad. The exact wedge in bending, whose surface stress
/// is `2 F sin(a) / (b r (2 a - sin 2a))` along the flank, is 5.4% off the beam formula at that
/// angle when resolved along y; 6% is that plus the band's averaging.
#[test]
fn the_flank_section_stresses_follow_beam_theory() {
    let (mesh, solution, _, _) = solve_structured(COARSE).expect("the coarse solve");
    for (y, ratio) in beam_section_ratios(&mesh, &solution, &[37.0, 38.0, 39.0, 40.0, 41.0, 42.0])
        .expect("the sections are found")
    {
        eprintln!("y = {y}: ratio {ratio:.4}");
        assert!(
            (ratio - 1.0).abs() <= 0.06,
            "at y = {y} the section bending stress is {ratio} of beam theory"
        );
    }
}

fn gear_case() -> Case {
    all_cases()
        .into_iter()
        .find(|case| case.id == "gear-tooth")
        .expect("the case exists")
}

fn rows(peaks: [f64; 2]) -> Vec<DensityResult> {
    peaks
        .iter()
        .zip([0.9, 0.6])
        .map(|(peak, size)| DensityResult {
            label: format!("near={size}"),
            size,
            values: vec![("root_peak".to_string(), *peak)],
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
fn the_judge_passes_a_stable_series_at_the_reference() {
    let case = gear_case();
    let peaks = [0.99 * ROOT_PEAK_REFERENCE, ROOT_PEAK_REFERENCE];
    assert_eq!((case.judge)(&rows(peaks)), Verdict::Pass);
}

#[test]
fn the_judge_fails_the_lewis_figure_it_only_reports() {
    let case = gear_case();
    let information = lewis_kf_information();
    match (case.judge)(&rows([1.01 * information, information])) {
        Verdict::Fail(reason) => assert!(reason.contains("structured reference"), "{reason}"),
        other => panic!("expected a failure against the reference, got {other:?}"),
    }
}

#[test]
fn the_judge_fails_a_peak_more_than_five_percent_off() {
    let case = gear_case();
    let off = 1.06 * ROOT_PEAK_REFERENCE;
    match (case.judge)(&rows([off, off])) {
        Verdict::Fail(reason) => assert!(reason.contains("structured reference"), "{reason}"),
        other => panic!("expected a failure against the reference, got {other:?}"),
    }
}

#[test]
fn the_judge_fails_a_series_that_moves_too_much() {
    let case = gear_case();
    match (case.judge)(&rows([0.92 * ROOT_PEAK_REFERENCE, ROOT_PEAK_REFERENCE])) {
        Verdict::Fail(reason) => assert!(reason.contains("last refinement"), "{reason}"),
        other => panic!("expected a failure of the refinement limit, got {other:?}"),
    }
}

/// A real refinement of the structured tooth: the observed rate and the extrapolation to zero
/// size give the judged value.
#[test]
#[ignore = "structured solves up to a few million free values, minutes in release; see this file's header"]
fn gear_reference_extrapolates_to_the_judged_value() {
    let [coarse, middle, fine] = ROOT_PEAK_REFERENCE_GRIDS;
    let coarse = gear_structured_reference(coarse).expect("the coarse solve");
    let middle = gear_structured_reference(middle).expect("the middle solve");
    let fine = gear_structured_reference(fine).expect("the fine solve");
    eprintln!("{coarse:?}\n{middle:?}\n{fine:?}");
    let (p0, p1, p2) = (coarse.root_peak, middle.root_peak, fine.root_peak);
    // Each grid is 1.5 times the last in every direction.
    let rate = ((p1 - p0) / (p2 - p1)).ln() / 1.5_f64.ln();
    let extrapolated = p2 + (p2 - p1) / (1.5_f64.powf(rate) - 1.0);
    eprintln!("observed rate {rate:.3}, extrapolated {extrapolated:.4}");
    // The limit: the constant is stated to 3 figures, so rounds by up to 0.05%, and the rate
    // moving between 1.5 and 2 shifts this extrapolation by 0.15% (the last change is 0.4% of the
    // value and enters divided by 1.5^rate - 1): their sum.
    let error = ((extrapolated - ROOT_PEAK_REFERENCE) / ROOT_PEAK_REFERENCE).abs();
    assert!(
        error <= 0.002,
        "extrapolated {extrapolated:.4} against {ROOT_PEAK_REFERENCE:.4}, error {:.3}%",
        100.0 * error
    );
}
