// The model-validity limits (`validity.rs`): a result is judged correctly on each side of every
// limit, and a real study at the top of the load slider is reported as outside the model while
// the default one is not, so the page can never present an overloaded part as agreement.

use solidfem_wasm::catalog::{LoadKind, Restraint, SectionKind, StudyRequest};
use solidfem_wasm::engine::DemoStudy;
use solidfem_wasm::validity::{
    assess, both_ends_deflection_limit, one_end_deflection_limit, TWIST_LIMIT_RADIANS,
};
use solidfem_wasm::view::ViewData;

use crate::support::run_to_done;

/// One percent either side of a limit: far from rounding, close enough to prove where it sits.
const BELOW: f64 = 0.99;
const ABOVE: f64 = 1.01;

fn request(restraint: Restraint, load: LoadKind) -> StudyRequest {
    StudyRequest {
        section: SectionKind::Solid,
        length: 155.0,
        width: 15.0,
        height: 15.0,
        wall_fraction: 0.0,
        material: 0,
        restraint,
        load,
        magnitude: 10.0,
    }
}

fn view(peak: f64, max_displacement: f64) -> ViewData {
    ViewData {
        positions: Vec::new(),
        indices: Vec::new(),
        von_mises: Vec::new(),
        displacement: Vec::new(),
        rotation: Vec::new(),
        min_stress: 0.0,
        max_stress: peak,
        max_displacement,
        max_rotation: 0.0,
        peak_stress_outside_clamps: peak,
        safety_factor: None,
    }
}

#[test]
fn stress_on_each_side_of_yield() {
    let r = request(Restraint::FixedOneEnd, LoadKind::Push);
    assert!(!assess(&r, 50.0, &view(50.0 * BELOW, 0.0)).exceeds_yield);
    assert!(assess(&r, 50.0, &view(50.0 * ABOVE, 0.0)).exceeds_yield);
    assert!(
        !assess(&r, 50.0, &view(50.0, 0.0)).exceeds_yield,
        "at yield is inside"
    );
}

#[test]
fn a_push_held_at_one_end_is_limited_by_a_twelfth_of_the_length() {
    let r = request(Restraint::FixedOneEnd, LoadKind::Push);
    let limit = one_end_deflection_limit(r.length);
    assert!(
        (limit - r.length / 12.0).abs() < 1e-9,
        "0.05 / 0.6 is one twelfth"
    );
    let inside = assess(&r, 1e9, &view(1.0, limit * BELOW));
    let outside = assess(&r, 1e9, &view(1.0, limit * ABOVE));
    assert!(inside.within_model() && !inside.excessive_deflection);
    assert!(!outside.within_model() && outside.excessive_deflection);
    assert_eq!(inside.deflection_limit, Some(limit));
    assert_eq!(inside.twist_radians, None);
}

#[test]
fn a_push_held_at_both_ends_is_limited_by_the_section_not_the_length() {
    let r = request(Restraint::FixedBothEnds, LoadKind::Push);
    // A square section's radius of gyration is its side over the square root of twelve.
    let gyration = r.height / 12.0f64.sqrt();
    let limit = both_ends_deflection_limit(gyration);
    assert!(
        ((limit / gyration) - 0.888).abs() < 0.001,
        "the stated 0.89 r"
    );
    assert!(!assess(&r, 1e9, &view(1.0, limit * BELOW)).excessive_deflection);
    assert!(assess(&r, 1e9, &view(1.0, limit * ABOVE)).excessive_deflection);
    // The same bar, longer, has the same limit: stretching depends on the section.
    let longer = StudyRequest { length: 320.0, ..r };
    assert_eq!(
        assess(&longer, 1e9, &view(1.0, 0.0)).deflection_limit,
        Some(limit)
    );
}

#[test]
fn a_twist_is_limited_by_the_angle() {
    let r = request(Restraint::FixedOneEnd, LoadKind::Twist);
    let corner = 0.5 * (r.width * r.width + r.height * r.height).sqrt();
    let at = |radians: f64| assess(&r, 1e9, &view(1.0, radians * corner));
    assert!(!at(TWIST_LIMIT_RADIANS * BELOW).excessive_twist);
    assert!(at(TWIST_LIMIT_RADIANS * ABOVE).excessive_twist);
    assert_eq!(TWIST_LIMIT_RADIANS, 0.1);
    let reading = at(0.05);
    assert!((reading.twist_radians.expect("a twist has an angle") - 0.05).abs() < 1e-12);
    assert_eq!(reading.twist_limit_radians, Some(TWIST_LIMIT_RADIANS));
    assert_eq!(reading.deflection_limit, None);
}

#[test]
fn a_real_study_reports_the_default_part_inside_and_the_top_of_the_slider_outside() {
    let default = StudyRequest {
        magnitude: 10.0,
        ..request(Restraint::FixedOneEnd, LoadKind::Push)
    };
    let mut gentle = DemoStudy::start(default).expect("offered");
    run_to_done(&mut gentle, 1000);
    let validity = gentle.validity(&gentle.view());
    assert!(validity.within_model(), "{validity:?}");

    let overloaded = StudyRequest {
        magnitude: 2000.0,
        ..default
    };
    let mut hard = DemoStudy::start(overloaded).expect("offered");
    run_to_done(&mut hard, 1000);
    let validity = hard.validity(&hard.view());
    assert!(!validity.within_model());
    assert!(validity.exceeds_yield, "{validity:?}");
    assert!(validity.excessive_deflection, "{validity:?}");
}
