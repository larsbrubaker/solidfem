//! The convergence and singularity reading on series the theory fixes: a re-entrant corner's peak
//! must be reported as singular, a smooth feature's as converged, and neither mistaken for the
//! other. The load-balance check is exercised on hand-built numbers.

use solidfem_bench::benchmarks::cases::{check_error, ktn_finite_width};
use solidfem_bench::benchmarks::verdict::{
    classify_peak, growth_exponents, last_change, LoadBalance, PeakBehaviour, Verdict,
    LOAD_BALANCE_LIMIT, SINGULAR_EXPONENT,
};

/// What a re-entrant 270 degree corner does: the sampled peak rises like `h^(lambda - 1)` with
/// `lambda` the Williams eigenvalue, so each halving multiplies it by `2^0.456`.
fn singular_series() -> (Vec<f64>, Vec<f64>) {
    let sizes = vec![1.0, 0.5, 0.25];
    let peaks = vec![
        100.0,
        100.0 * 2.0_f64.powf(0.4555),
        100.0 * 4.0_f64.powf(0.4555),
    ];
    (sizes, peaks)
}

#[test]
fn a_reentrant_corners_growing_peak_is_detected_as_a_singularity() {
    let (sizes, peaks) = singular_series();
    match classify_peak(&sizes, &peaks, 0.03) {
        Some(PeakBehaviour::Singular { exponent }) => {
            assert!((exponent - 0.4555).abs() < 1e-9);
        }
        other => panic!("expected Singular, got {other:?}"),
    }
}

#[test]
fn a_smoothly_converging_peak_is_not_called_singular() {
    // Error falling as h^2 from a 250 MPa limit: 220, 242.5, 248.1.
    let sizes = vec![1.2, 0.8, 0.5];
    let peaks = vec![220.0, 242.5, 248.1];
    match classify_peak(&sizes, &peaks, 0.03) {
        Some(PeakBehaviour::Converged { last_change }) => {
            assert!(last_change < 0.03, "last change {last_change}");
        }
        other => panic!("expected Converged, got {other:?}"),
    }
}

#[test]
fn a_singular_series_at_a_slower_rate_than_the_exponent_threshold_is_not_singular() {
    // Growing, but with an exponent below the 0.2 the singularity reading needs.
    let sizes = vec![1.0, 0.5, 0.25];
    let peaks = vec![100.0, 110.0, 121.0];
    let exponents = growth_exponents(&sizes, &peaks).expect("series valid");
    assert!(exponents[1] < SINGULAR_EXPONENT);
    assert!(matches!(
        classify_peak(&sizes, &peaks, 0.03),
        Some(PeakBehaviour::Unsettled { .. })
    ));
}

#[test]
fn a_peak_still_moving_but_not_singular_is_unsettled_not_converged() {
    // Falling back down: not singular, and moving more than the 3% allowed.
    let sizes = vec![1.0, 0.5, 0.25];
    let peaks = vec![120.0, 112.0, 104.0];
    assert!(matches!(
        classify_peak(&sizes, &peaks, 0.03),
        Some(PeakBehaviour::Unsettled { .. })
    ));
}

#[test]
fn a_series_of_two_can_never_be_called_singular() {
    let sizes = vec![1.0, 0.5];
    let peaks = vec![100.0, 200.0];
    assert!(!matches!(
        classify_peak(&sizes, &peaks, 0.03),
        Some(PeakBehaviour::Singular { .. })
    ));
}

#[test]
fn a_bad_series_is_refused_rather_than_misread() {
    assert_eq!(growth_exponents(&[1.0, 0.5], &[-1.0, 2.0]), None);
    assert_eq!(growth_exponents(&[0.5, 1.0], &[1.0, 2.0]), None);
    assert_eq!(last_change(&[1.0]), None);
    assert_eq!(classify_peak(&[1.0], &[1.0], 0.03), None);
}

#[test]
fn load_balance_error_is_the_imbalance_over_the_gross_load() {
    let balanced = LoadBalance::new([0.0, -100.0, 0.0], [0.0, 100.0, 0.0], 100.0);
    assert_eq!(balanced.error, 0.0);
    let off = LoadBalance::new([0.0, -100.0, 0.0], [0.0, 99.0, 0.0], 100.0);
    assert!((off.error - 0.01).abs() < 1e-12);
    assert!(off.error > LOAD_BALANCE_LIMIT);
}

#[test]
fn check_error_passes_within_the_limit_and_names_a_failure_beyond_it() {
    assert!(check_error(1.005, 1.0, 0.01, "x").is_ok());
    let error = check_error(1.02, 1.0, 0.01, "tip deflection").unwrap_err();
    assert!(error.contains("tip deflection"), "{error}");
    assert!(error.contains("over the"), "{error}");
}

#[test]
fn the_finite_width_chart_reaches_the_infinite_plate_value_and_its_quoted_points() {
    // The curve fit is 3.004 at d/W -> 0 (0.13% above the exact 3), and 2.732 at d/W = 0.1.
    assert!((ktn_finite_width(0.0) - 3.004).abs() < 1e-9);
    assert!((ktn_finite_width(0.1) - 2.73188).abs() < 1e-4);
}

#[test]
fn a_verdict_says_which_of_the_four_it_is() {
    assert_eq!(Verdict::Pass.to_string(), "PASS");
    assert!(Verdict::Fail("why".to_string())
        .to_string()
        .starts_with("FAIL"));
    assert!(Verdict::NotAssessed("capped".to_string())
        .to_string()
        .starts_with("NOT ASSESSED"));
    assert!(Verdict::OutOfScope("no contact".to_string())
        .to_string()
        .starts_with("OUT OF SCOPE"));
}
