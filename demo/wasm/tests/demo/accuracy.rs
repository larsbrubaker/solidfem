// The demo's central promise: every combination it offers agrees with the textbook, enumerated
// from the catalog so an inaccurate combination cannot be offered without failing here.
//
// The full sweep is ignored in the default run because it solves a few hundred studies; run it,
// in the release profile, whenever a mesh rule, range or formula changes:
// `cargo nextest run --workspace --release --test demo --run-ignored only --no-capture accuracy`
// The default run keeps one middle-of-the-ranges study per offered combination.
//
// Limits, derived before any run:
//  - Deflection of a solid bar: 5%. The end-loaded bar is at least 6 depths long (shear term up to
//    about 2.3% of the bending term); clamp warping and Poisson restraint are a fraction of a
//    percent to one percent over the clamp zone, and quadratic elements at 4 cells across add
//    about a percent of discretization error. The fixed-fixed solid bar is at least 12 depths long
//    with a band load near mid-span whose local effect averages over the section, leaving about
//    a percent more; 5% holds those sources together.
//  - Deflection of a tube: 7%. Same sources, plus the shear coefficient from the classical
//    first-moment stress distribution, which for a chunky tube can be some 10 to 20% off the
//    exact 3D value, on a shear term of a fifth to a half of the bending term at the shortest
//    fixed-fixed tubes (8 to 12 depths long, the shear term growing as the square of depth over
//    length): up to 2 to 4%, on top of the solid bar's sources, within 7%.
//  - Bending stress at a station at least about a depth from clamp and load: 5%. Saint-Venant's
//    principle has decayed the end disturbances below a couple of percent there, and the
//    straight-line fit spans the slab one cell either side of the station.
//  - Twist rate between sections at least 1.5 widths from the ends: 3%. End effects decay as
//    exp(-pi x / b), under 1% at 1.5 widths, and quadratic elements at four cells across capture
//    the Saint-Venant warping field to about a percent (solidfem's validation reaches 1% on a
//    square bar; width-to-height up to 2 here adds some).

use std::sync::Mutex;

use solidfem_wasm::catalog::{
    offered_combinations, ranges, Combination, LoadKind, Restraint, SectionKind, StudyRequest,
    MATERIALS,
};
use solidfem_wasm::engine::DemoStudy;
use solidfem_wasm::plan::Textbook;

use crate::support::{case_name, material_indices, request_at, run_to_done};

const DEFLECTION_LIMIT_SOLID_PERCENT: f64 = 5.0;
const DEFLECTION_LIMIT_TUBE_PERCENT: f64 = 7.0;
const STRESS_LIMIT_PERCENT: f64 = 5.0;
const TWIST_LIMIT_PERCENT: f64 = 3.0;

/// One accuracy case: a full request and its printable name.
struct Case {
    name: String,
    section: SectionKind,
    request: StudyRequest,
}

/// Whether the catalog promises a textbook comparison for this combination.
fn has_textbook_answer(combination: Combination) -> bool {
    combination.section == SectionKind::Solid || combination.load == LoadKind::Push
}

/// The combinations with a textbook column, straight from what the catalog offers.
fn compared_combinations() -> Vec<Combination> {
    offered_combinations()
        .into_iter()
        .filter(|c| has_textbook_answer(*c))
        .collect()
}

/// The limit for one quantity of a case.
fn limit(section: SectionKind, quantity: &str) -> f64 {
    match quantity {
        "Deflection" if section == SectionKind::Tube => DEFLECTION_LIMIT_TUBE_PERCENT,
        "Deflection" => DEFLECTION_LIMIT_SOLID_PERCENT,
        "Bending stress" => STRESS_LIMIT_PERCENT,
        "Twist rate" => TWIST_LIMIT_PERCENT,
        other => panic!("unexpected quantity {other}"),
    }
}

/// Solves one case, returning its report lines, or the lines and the failures.
fn run_case(case: &Case) -> Result<Vec<String>, String> {
    let name = &case.name;
    let mut study = DemoStudy::start(case.request).map_err(|e| format!("{name}: refused: {e}"))?;
    run_to_done(&mut study, u32::MAX);
    let (elements, _) = study.mesh_size();
    let mut lines = Vec::new();
    let mut failures = Vec::new();
    for row in study.rows() {
        let allowed = limit(case.section, row.quantity);
        lines.push(format!(
            "{name}: {} FEA {:.5} textbook {:.5} diff {:+.3}% (limit {allowed}%, {elements} elements)",
            row.quantity, row.fea, row.textbook, row.percent_difference
        ));
        let within = row.percent_difference.abs() <= allowed;
        if !within {
            failures.push(format!(
                "{name}: {} differs by {:+.3}%, limit {allowed}%",
                row.quantity, row.percent_difference
            ));
        }
    }
    if study.rows().is_empty() {
        failures.push(format!(
            "{name}: promised a textbook answer but has no rows"
        ));
    }
    if failures.is_empty() {
        Ok(lines)
    } else {
        Err(format!("{}\n{}", lines.join("\n"), failures.join("\n")))
    }
}

/// Runs the cases on a few threads, printing every line and failing with the failures listed.
fn run_all(cases: Vec<Case>) {
    let queue = Mutex::new(cases.into_iter());
    let lines = Mutex::new(Vec::new());
    let failures = Mutex::new(Vec::new());
    let workers = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(12);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let Some(case) = queue.lock().expect("no panic holding lock").next() else {
                    return;
                };
                match run_case(&case) {
                    Ok(found) => lines.lock().expect("no panic holding lock").extend(found),
                    Err(error) => failures.lock().expect("no panic holding lock").push(error),
                }
            });
        }
    });
    for line in lines.into_inner().expect("lock") {
        println!("{line}");
    }
    let failures = failures.into_inner().expect("lock");
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The Poisson-ratio extremes of the table: a material enters the answer only through E, which
/// scales the displacement exactly, and nu, so the extremes bound every material's nu.
fn extreme_poisson_materials() -> [usize; 2] {
    let by_poisson = |a: &usize, b: &usize| {
        MATERIALS[*a]
            .poisson_ratio
            .total_cmp(&MATERIALS[*b].poisson_ratio)
    };
    [
        (0..MATERIALS.len()).min_by(by_poisson).expect("materials"),
        (0..MATERIALS.len()).max_by(by_poisson).expect("materials"),
    ]
}

/// The middle of the ranges, once per offered combination: the default run's coverage.
#[test]
fn every_combination_agrees_with_the_textbook_at_the_middle_of_its_ranges() {
    run_all(
        compared_combinations()
            .into_iter()
            .map(|combination| Case {
                name: case_name(combination, 0, "middle"),
                section: combination.section,
                request: request_at(combination, 0, 1),
            })
            .collect(),
    );
}

/// Every offered combination, every material, at the bottom, middle and top of its ranges.
#[test]
#[ignore = "the full sweep: run it in the release profile (this file's header has the command)"]
fn accuracy_every_offered_combination_agrees_with_the_textbook() {
    let mut cases = Vec::new();
    for combination in compared_combinations() {
        for material in material_indices() {
            for (level, label) in ["bottom", "middle", "top"].into_iter().enumerate() {
                cases.push(Case {
                    name: case_name(combination, material, label),
                    section: combination.section,
                    request: request_at(combination, material, level),
                });
            }
        }
    }
    run_all(cases);
}

/// The range ends set independently: shortest and longest bars with every pairing of the smallest
/// and largest width and height, each wall limit, at both extreme Poisson ratios. A flat or a tall
/// section at the ends of the length range is where a beam formula and the mesh are worst off.
#[test]
#[ignore = "the full sweep: run it in the release profile (this file's header has the command)"]
fn accuracy_the_corners_of_every_range_agree_with_the_textbook() {
    let mut cases = Vec::new();
    for combination in compared_combinations() {
        let r = ranges(combination.section, combination.restraint, combination.load);
        let walls = r.wall_fraction.map_or(vec![0.0], |w| vec![w.min, w.max]);
        for material in extreme_poisson_materials() {
            for length in [r.length.min, r.length.max] {
                for width in [r.width.min, r.width.max] {
                    for height in [r.height.min, r.height.max] {
                        for wall_fraction in &walls {
                            let mut request = request_at(combination, material, 1);
                            request.length = length;
                            request.width = width;
                            request.height = height;
                            request.wall_fraction = *wall_fraction;
                            cases.push(Case {
                                name: case_name(
                                    combination,
                                    material,
                                    &format!("L={length} w={width} h={height} t={wall_fraction}"),
                                ),
                                section: combination.section,
                                request,
                            });
                        }
                    }
                }
            }
        }
    }
    run_all(cases);
}

#[test]
fn a_tube_in_twist_reports_no_textbook_answer_and_no_rows() {
    for restraint in [Restraint::FixedOneEnd, Restraint::FixedBothEnds] {
        let combination = Combination {
            section: SectionKind::Tube,
            restraint,
            load: LoadKind::Twist,
        };
        assert!(!has_textbook_answer(combination));
        let mut study = DemoStudy::start(request_at(combination, 0, 1)).expect("offered");
        assert!(matches!(study.textbook(), Textbook::NoTextbook(_)));
        assert!(study.rows().is_empty());
        run_to_done(&mut study, u32::MAX);
        assert!(
            study.rows().is_empty(),
            "a combination with no textbook answer must not invent a number"
        );
        assert!(matches!(study.textbook(), Textbook::NoTextbook(_)));
        // The simulation still ran: the page shows its stress.
        assert!(study.view().max_stress > 0.0);
    }
}
