//! What the runner claims, on rows built by hand and fed to the production judges from
//! `all_cases`: a non-finite measurement can never pass, and each judge's limit and wording says
//! exactly what was tested.

use solidfem_bench::benchmarks::cases::{all_cases, check_error, winkler_inner_stress};
use solidfem_bench::benchmarks::pipeline::{mesher_refused, DensityResult};
use solidfem_bench::benchmarks::verdict::{growth_exponents, worst_of, LoadBalance, Verdict};

fn row(size: f64, values: &[(&str, f64)]) -> DensityResult {
    DensityResult {
        label: format!("h={size}"),
        size,
        values: values
            .iter()
            .map(|(key, value)| ((*key).to_string(), *value))
            .collect(),
        balance: LoadBalance::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0),
        nodes: 1,
        elements: 1,
        free_dofs: 3,
        iterations: 1,
        residual: 0.0,
        min_dihedral: 30.0,
        seconds: 0.0,
    }
}

fn judge(id: &str, rows: &[DensityResult]) -> Verdict {
    let case = all_cases()
        .into_iter()
        .find(|case| case.id == id)
        .expect("the case exists");
    (case.judge)(rows)
}

/// `u_r` at the bore of the benchmark ring, worked by hand: `A = 10 * 25 / 75`, `B = 100 A`,
/// `((1 - nu) A r + (1 + nu) B / r) / E` at `r = 5`, `E = 200000`, `nu = 0.3`.
const RING_BORE: f64 = 4.916_67e-4;

fn ring_rows(hoop: f64, radial: f64) -> Vec<DensityResult> {
    vec![
        row(
            1.0,
            &[
                ("bore_ur", RING_BORE),
                ("hoop_err", 0.01),
                ("radial_err", 0.01),
            ],
        ),
        row(
            0.75,
            &[
                ("bore_ur", RING_BORE),
                ("hoop_err", hoop),
                ("radial_err", radial),
            ],
        ),
    ]
}

#[test]
fn a_non_finite_error_never_passes_the_reference_check() {
    assert!(check_error(f64::NAN, 1.0, 0.01, "x").is_err());
    assert!(check_error(1.0, f64::NAN, 0.01, "x").is_err());
    assert!(check_error(f64::INFINITY, 1.0, 0.01, "x").is_err());
    assert!(check_error(1.0, f64::INFINITY, 0.01, "x").is_err());
    assert!(check_error(0.0, 0.0, 0.01, "x").is_err());
}

#[test]
fn a_nan_at_the_finest_mesh_fails_the_stability_check_too() {
    // The narrow beam for the strip at nu = 0.3, with the last refinement's value lost.
    let tip = 1.98;
    let rows = vec![row(1.4, &[("tip", tip)]), row(1.0, &[("tip", f64::NAN)])];
    assert!(matches!(judge("strip-nu03", &rows), Verdict::Fail(_)));
    // The same series with the NaN on the coarse row: the last refinement cannot be read.
    let rows = vec![row(1.4, &[("tip", f64::NAN)]), row(1.0, &[("tip", tip)])];
    assert!(matches!(judge("strip-nu03", &rows), Verdict::Fail(_)));
}

#[test]
fn a_nan_stress_error_fails_the_ring_and_a_good_one_passes() {
    assert!(matches!(
        judge("ring", &ring_rows(f64::NAN, 0.01)),
        Verdict::Fail(_)
    ));
    assert!(matches!(
        judge("ring", &ring_rows(0.01, f64::NAN)),
        Verdict::Fail(_)
    ));
    assert_eq!(judge("ring", &ring_rows(0.01, 0.01)), Verdict::Pass);
}

#[test]
fn an_infinite_peak_is_refused_rather_than_called_a_singularity() {
    assert_eq!(
        growth_exponents(&[1.0, 0.5, 0.25], &[100.0, 200.0, f64::INFINITY]),
        None
    );
}

#[test]
fn a_running_worst_keeps_a_nan_that_max_would_drop() {
    assert!(worst_of(1.0, f64::NAN).is_nan());
    assert!(worst_of(f64::NAN, 1.0).is_nan());
    assert_eq!(worst_of(1.0, 2.0), 2.0);
}

fn strip_rows(tip: f64) -> Vec<DensityResult> {
    vec![row(1.4, &[("tip", tip)]), row(1.0, &[("tip", tip)])]
}

/// The strip at nu = 0.3: bending `F L^3 / (3 E I) = 2.0` for `I = 10 / 12`, plus the shear term
/// `F L / ((5/6) G A)` with `G = E / 2.6`, `A = 10`.
const STRIP_NARROW: f64 = 2.0 + 100.0 / (5.0 / 6.0 * (200_000.0 / 2.6) * 10.0);

#[test]
fn the_strip_at_nu_03_is_held_to_the_clamp_restrained_narrow_beam() {
    // The measured value passes.
    assert_eq!(judge("strip-nu03", &strip_rows(1.98013)), Verdict::Pass);
    // Each of these sat inside the plate-to-beam bracket of 1.800 to 2.020 and is now a failure:
    // too stiff for a clamp-only restraint, and above the narrow beam.
    assert!(matches!(
        judge("strip-nu03", &strip_rows(0.97 * STRIP_NARROW)),
        Verdict::Fail(_)
    ));
    assert!(matches!(
        judge("strip-nu03", &strip_rows(1.0035 * STRIP_NARROW)),
        Verdict::Fail(_)
    ));
}

fn hook_rows(inner: f64) -> Vec<DensityResult> {
    vec![
        row(2.2, &[("inner_sigma", inner), ("tip_uy", 1.0)]),
        row(1.6, &[("inner_sigma", inner), ("tip_uy", 1.0)]),
    ]
}

#[test]
fn the_hook_is_held_to_the_tighter_winkler_bach_limit() {
    let reference = winkler_inner_stress(1000.0, 20.0, 40.0, 10.0);
    // The measured +2.1% passes the 2.4% limit.
    assert_eq!(judge("hook", &hook_rows(1.021 * reference)), Verdict::Pass);
    // 3% would have passed the old 6% limit and does not pass this one.
    assert!(matches!(
        judge("hook", &hook_rows(1.03 * reference)),
        Verdict::Fail(_)
    ));
}

#[test]
fn the_sharp_corner_says_singularity_and_no_accuracy_reference() {
    // A corner peak growing like h^-0.456 and a tip deflection that has settled.
    let rows = vec![
        row(1.0, &[("tip", 1.34), ("peak", 100.0)]),
        row(0.5, &[("tip", 1.343), ("peak", 137.1)]),
        row(0.25, &[("tip", 1.344), ("peak", 188.0)]),
    ];
    let verdict = judge("l-sharp", &rows);
    let text = verdict.to_string();
    assert!(text.starts_with("PASS ("), "{text}");
    assert!(text.contains("singularity detected"), "{text}");
    assert!(text.contains("no accuracy reference"), "{text}");
    assert_ne!(verdict, Verdict::Pass);
}

#[test]
fn the_fillet_says_mesh_stable_and_no_accuracy_reference() {
    let rows = vec![
        row(1.2, &[("tip", 1.29), ("peak", 220.0)]),
        row(0.8, &[("tip", 1.2946), ("peak", 242.5)]),
        row(0.5, &[("tip", 1.2946), ("peak", 248.1)]),
    ];
    assert_eq!(
        judge("l-fillet", &rows).to_string(),
        "PASS (mesh-stable, no accuracy reference)"
    );
}

#[test]
fn a_bracket_that_loses_the_singularity_still_fails() {
    // Peaks that settle: the sharp corner must not pass as if it were smooth.
    let rows = vec![
        row(1.0, &[("tip", 1.34), ("peak", 100.0)]),
        row(0.5, &[("tip", 1.343), ("peak", 101.0)]),
        row(0.25, &[("tip", 1.344), ("peak", 101.2)]),
    ];
    assert!(matches!(judge("l-sharp", &rows), Verdict::Fail(_)));
}

#[test]
fn the_snap_fit_pass_says_nominal_bending_strain_and_no_allowable() {
    // 1.5 delta T / L^2 = 7.5e-3, and the same strain a coarser mesh gave.
    let strain = 7.56e-3;
    let rows = vec![
        row(0.7, &[("nominal_bending_strain", strain)]),
        row(0.5, &[("nominal_bending_strain", strain)]),
    ];
    let text = judge("snapfit", &rows).to_string();
    assert!(text.starts_with("PASS ("), "{text}");
    assert!(text.contains("nominal bending strain"), "{text}");
    assert!(text.contains("allowable strain"), "{text}");
    assert!(!text.contains("permissible"), "{text}");
}

#[test]
fn a_mesher_refusal_names_the_density_and_says_no_surface_defect_was_shown() {
    let text = mesher_refused("near=60", "a recovery fault");
    assert!(text.contains("near=60"), "{text}");
    assert!(text.contains("mesher refused"), "{text}");
    assert!(text.contains("this workspace's surface recovery"), "{text}");
    assert!(text.contains("no surface defect shown"), "{text}");
    assert!(text.contains("a recovery fault"), "{text}");
}
