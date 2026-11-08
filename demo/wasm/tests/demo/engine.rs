// The engine's contract with the page: slices never change the result, a new request cancels at
// once, oversized and bad requests return a named state or error, and the render arrays stay
// consistent during the solve.

use solidfem_wasm::catalog::{
    offered_combinations, ranges, Combination, LoadKind, Range, RequestError, Restraint,
    SectionKind, StudyRequest,
};
use solidfem_wasm::engine::{DemoStudy, StudyState};
use solidfem_wasm::plan::{plan_study_within, planned_elements, PlanError, MAX_ELEMENTS};

use crate::support::{request_at, run_to_done};

fn solid_cantilever() -> StudyRequest {
    request_at(
        Combination {
            section: SectionKind::Solid,
            restraint: Restraint::FixedOneEnd,
            load: LoadKind::Push,
        },
        0,
        0,
    )
}

fn view_bits(study: &DemoStudy) -> Vec<u32> {
    let v = study.view();
    v.von_mises
        .iter()
        .chain(&v.displacement)
        .map(|x| x.to_bits())
        .collect()
}

#[test]
fn results_are_bit_identical_however_the_solve_is_sliced() {
    let run = |budget: u32| {
        let mut study = DemoStudy::start(solid_cantilever()).expect("offered");
        run_to_done(&mut study, budget);
        (view_bits(&study), study.rows().to_vec())
    };
    let (one_call, rows_one) = run(u32::MAX);
    let (small, rows_small) = run(3);
    assert_eq!(one_call, small, "render arrays differ by slicing");
    assert_eq!(rows_one.len(), rows_small.len());
    for (a, b) in rows_one.iter().zip(&rows_small) {
        assert_eq!(a.fea.to_bits(), b.fea.to_bits(), "{}", a.quantity);
        assert_eq!(a.textbook.to_bits(), b.textbook.to_bits(), "{}", a.quantity);
        assert_eq!(
            a.percent_difference.to_bits(),
            b.percent_difference.to_bits(),
            "{}",
            a.quantity
        );
    }
}

#[test]
fn rows_update_as_a_live_estimate_during_the_solve() {
    let mut study = DemoStudy::start(solid_cantilever()).expect("offered");
    let mut solving_rows_changed = false;
    let mut previous = None;
    loop {
        let state = study.step(5);
        if state == StudyState::Solving {
            let deflection = study.rows()[0].fea;
            if previous.is_some_and(|p: f64| p != deflection) {
                solving_rows_changed = true;
            }
            previous = Some(deflection);
        }
        if state == StudyState::Done {
            break;
        }
        assert_ne!(state, StudyState::Failed, "{:?}", study.failure());
    }
    assert!(
        solving_rows_changed,
        "the textbook comparison stayed still while solving"
    );
}

#[test]
fn starting_a_new_study_replaces_a_running_one_at_once() {
    // The page drops the old study by starting a new one; here a held study is cancelled and a
    // new one runs from the start.
    let mut first = DemoStudy::start(solid_cantilever()).expect("offered");
    for _ in 0..3 {
        first.step(5);
    }
    first.cancel();
    assert_eq!(first.step(5), StudyState::Cancelled);
    assert_eq!(first.state(), StudyState::Cancelled);

    let mut second = DemoStudy::start(solid_cantilever()).expect("offered");
    run_to_done(&mut second, 1000);
    assert_eq!(second.state(), StudyState::Done);
    assert!(second.rows()[0].fea > 0.0);
}

#[test]
fn a_request_past_the_size_limit_returns_too_large() {
    let request = solid_cantilever();
    let elements = planned_elements(&request);
    match plan_study_within(&request, elements - 1) {
        Err(PlanError::TooLarge { elements: e, limit }) => {
            assert_eq!(e, elements);
            assert_eq!(limit, elements - 1);
        }
        Err(other) => panic!("wrong error: {other}"),
        Ok(_) => panic!("expected too large"),
    }
}

#[test]
fn no_position_the_page_offers_is_refused_as_too_large() {
    // A dense walk of every slider, ends included, for every offered combination. The element
    // count steps with ceilings of length over side and of the hollow over the wall, so the walk
    // has to visit the interior as well as the corners.
    const STEPS: usize = 12;
    let along = |r: Range, i: usize| r.min + (r.max - r.min) * i as f64 / (STEPS - 1) as f64;
    let mut largest = 0;
    for combination in offered_combinations() {
        let r = ranges(combination.section, combination.restraint, combination.load);
        let walls = r.wall_fraction.map_or(1, |_| STEPS);
        let mut request = request_at(combination, 0, 1);
        for l in 0..STEPS {
            request.length = along(r.length, l);
            for w in 0..STEPS {
                request.width = along(r.width, w);
                for h in 0..STEPS {
                    request.height = along(r.height, h);
                    for t in 0..walls {
                        if let Some(range) = r.wall_fraction {
                            request.wall_fraction = along(range, t);
                        }
                        request.validate().expect("a position inside the ranges");
                        largest = largest.max(planned_elements(&request));
                    }
                }
            }
        }
    }
    assert_eq!(
        largest, MAX_ELEMENTS,
        "MAX_ELEMENTS is documented as the largest mesh an offered position needs: keep the two equal"
    );
}

#[test]
fn bad_inputs_are_errors_not_panics() {
    let mut nan_length = solid_cantilever();
    nan_length.length = f64::NAN;
    assert!(matches!(
        DemoStudy::start(nan_length),
        Err(PlanError::Request(RequestError::NotANumber("length")))
    ));

    let mut infinite = solid_cantilever();
    infinite.magnitude = f64::INFINITY;
    assert!(matches!(
        DemoStudy::start(infinite),
        Err(PlanError::Request(RequestError::NotANumber("magnitude")))
    ));

    let mut too_long = solid_cantilever();
    too_long.length = 1e6;
    assert!(matches!(
        DemoStudy::start(too_long),
        Err(PlanError::Request(RequestError::OutOfRange {
            parameter: "length",
            ..
        }))
    ));

    let mut negative = solid_cantilever();
    negative.width = -3.0;
    assert!(matches!(
        DemoStudy::start(negative),
        Err(PlanError::Request(RequestError::OutOfRange {
            parameter: "width",
            ..
        }))
    ));

    let mut no_such_material = solid_cantilever();
    no_such_material.material = 99;
    assert!(matches!(
        DemoStudy::start(no_such_material),
        Err(PlanError::Request(RequestError::UnknownMaterial(99)))
    ));

    let combination = Combination {
        section: SectionKind::Tube,
        restraint: Restraint::FixedOneEnd,
        load: LoadKind::Push,
    };
    let mut bad_wall = request_at(combination, 0, 1);
    bad_wall.wall_fraction = 0.9;
    assert!(matches!(
        DemoStudy::start(bad_wall),
        Err(PlanError::Request(RequestError::OutOfRange {
            parameter: "wall_fraction",
            ..
        }))
    ));
}

#[test]
fn render_arrays_have_consistent_lengths_and_finite_values_during_and_after_the_solve() {
    let mut study = DemoStudy::start(solid_cantilever()).expect("offered");
    let check = |study: &DemoStudy| {
        let v = study.view();
        let vertices = v.positions.len() / 3;
        assert_eq!(v.positions.len() % 3, 0);
        assert_eq!(v.displacement.len(), vertices * 3);
        assert_eq!(v.von_mises.len(), vertices);
        assert_eq!(v.indices.len() % 3, 0);
        assert!(!v.indices.is_empty());
        assert!(v.indices.iter().all(|i| (*i as usize) < vertices));
        assert!(v.positions.iter().all(|x| x.is_finite()));
        assert!(v.displacement.iter().all(|x| x.is_finite()));
        assert!(v.von_mises.iter().all(|x| x.is_finite()));
        assert!(v.max_stress.is_finite());
        assert!(v.max_displacement.is_finite());
        assert!(v.peak_stress_outside_clamps.is_finite());
    };
    check(&study);
    loop {
        let state = study.step(20);
        check(&study);
        if state == StudyState::Done {
            break;
        }
        assert_ne!(state, StudyState::Failed, "{:?}", study.failure());
    }
    let done = study.view();
    assert!(done.max_stress > 0.0);
    assert!(done.max_displacement > 0.0);
    let safety = done.safety_factor.expect("stress exists");
    assert!(safety.is_finite() && safety > 0.0);
    // The peak outside the clamps never exceeds the overall maximum.
    assert!(done.peak_stress_outside_clamps <= done.max_stress + 1e-9);
}

#[test]
fn a_rectangle_shear_coefficient_is_five_sixths() {
    // Derived by hand in section.rs from the first-moment shear stress: exactly 5/6.
    let section = solidfem_wasm::section::Section::Solid {
        width: 17.0,
        height: 23.0,
    };
    assert!((section.shear_coefficient() - 5.0 / 6.0).abs() < 1e-12);
}

#[test]
fn the_saint_venant_constant_of_a_square_matches_the_tabulated_value() {
    // The classical tabulated torsion constant of a square section: J = 0.1406 a^4.
    let section = solidfem_wasm::section::Section::Solid {
        width: 10.0,
        height: 10.0,
    };
    let j = section.torsion_constant().expect("solid");
    assert!(((j / 1e4) - 0.1406).abs() < 1e-3, "J/a^4 = {}", j / 1e4);
}
