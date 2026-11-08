// The demo's central promise: every combination it offers agrees with the textbook, enumerated
// from the catalog so an inaccurate combination cannot be offered without failing here.
//
// Limits, derived before any run:
//  - Deflection of a solid bar: 5%. The end-loaded bar is at least 6 depths long (shear term up to
//    about 2.3% of the bending term); clamp warping and Poisson restraint are a fraction of a
//    percent to one percent over the clamp zone, and quadratic elements at 4 cells across, up to
//    2.5 cells along, add about a percent of discretization error. The fixed-fixed bar is at least
//    12 depths long with a band load near mid-span whose local effect averages over the section,
//    leaving about a percent more; 5% holds those sources together.
//  - Deflection of a tube: 7%. Same sources, plus the shear coefficient from the classical
//    first-moment stress distribution, which for a chunky tube can be some 10 to 20% off the
//    exact 3D value, on a shear term of up to roughly 20% of the bending term at the shortest
//    fixed-fixed tubes: up to 2 to 4%, on top of the solid bar's 5%-class sources, within 7%.
//  - Bending stress at a station at least 1.5 depths from clamp and load: 5%. Saint-Venant's
//    principle has decayed the end disturbances below a couple of percent there, and the
//    straight-line fit spans the slab one cell either side of the station.
//  - Twist rate between sections at least 1.5 widths from the ends: 3%. End effects decay as
//    exp(-pi x / b), under 1% at 1.5 widths, and quadratic elements at four cells across capture
//    the Saint-Venant warping field to about a percent (solidfem's validation reaches 1% on a
//    square bar; width-to-height up to 2 here adds some).

use solidfem_wasm::catalog::{Combination, Detail, LoadKind, Restraint, SectionKind, MATERIALS};
use solidfem_wasm::engine::DemoStudy;
use solidfem_wasm::plan::{PlanError, Textbook};

use crate::support::{case_name, material_indices, request_at, run_to_done};

const DEFLECTION_LIMIT_SOLID_PERCENT: f64 = 5.0;
const DEFLECTION_LIMIT_TUBE_PERCENT: f64 = 7.0;
const STRESS_LIMIT_PERCENT: f64 = 5.0;
const TWIST_LIMIT_PERCENT: f64 = 3.0;

/// One accuracy case: a combination at a range level of every parameter for one material.
#[derive(Clone, Copy)]
struct Case {
    combination: Combination,
    material: usize,
    level: usize,
    detail: Detail,
}

/// The Poisson-ratio extremes of the table: a material enters the answer only through E, which
/// scales the displacement exactly, and nu, which reaches the comparison through the shear
/// modulus and the clamp's Poisson restraint, so the extremes bound every material's nu.
fn extreme_poisson_materials() -> [usize; 2] {
    let min = (0..MATERIALS.len())
        .min_by(|a, b| {
            MATERIALS[*a]
                .poisson_ratio
                .total_cmp(&MATERIALS[*b].poisson_ratio)
        })
        .expect("materials");
    let max = (0..MATERIALS.len())
        .max_by(|a, b| {
            MATERIALS[*a]
                .poisson_ratio
                .total_cmp(&MATERIALS[*b].poisson_ratio)
        })
        .expect("materials");
    [min, max]
}

/// The cases enumerated from the catalog: each combination with each material at the middle of
/// its ranges and with the two extreme-Poisson materials at the bottom and top, plus the fine
/// detail level at the middle (fine meshes that exceed the size cap are refused, not offered).
fn all_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for combination in solidfem_wasm::catalog::offered_combinations() {
        if combination.section == SectionKind::Tube && combination.load == LoadKind::Twist {
            continue; // No textbook answer: `a_tube_in_twist_reports_no_textbook_answer` tests it.
        }
        for material in material_indices() {
            cases.push(Case {
                combination,
                material,
                level: 1,
                detail: Detail::Standard,
            });
        }
        for material in extreme_poisson_materials() {
            for level in [0, 2] {
                cases.push(Case {
                    combination,
                    material,
                    level,
                    detail: Detail::Standard,
                });
            }
        }
        cases.push(Case {
            combination,
            material: 0,
            level: 1,
            detail: Detail::Fine,
        });
    }
    cases
}

/// The limit for one quantity of a case.
fn limit(case: &Case, quantity: &str) -> f64 {
    match quantity {
        "Deflection" => {
            if case.combination.section == SectionKind::Tube {
                DEFLECTION_LIMIT_TUBE_PERCENT
            } else {
                DEFLECTION_LIMIT_SOLID_PERCENT
            }
        }
        "Bending stress" => STRESS_LIMIT_PERCENT,
        "Twist rate" => TWIST_LIMIT_PERCENT,
        other => panic!("unexpected quantity {other}"),
    }
}

fn run_case(case: Case) -> Result<Vec<String>, String> {
    let request = request_at(case.combination, case.material, case.level, case.detail);
    let name = case_name(
        case.combination,
        case.material,
        &format!("level{}/{:?}", case.level, case.detail),
    );
    let mut study = match DemoStudy::start(request) {
        Ok(study) => study,
        Err(PlanError::TooLarge { elements, limit }) if case.detail == Detail::Fine => {
            return Ok(vec![format!(
                "{name}: too large at fine detail ({elements} > {limit}), not offered"
            )]);
        }
        Err(error) => return Err(format!("{name}: refused: {error}")),
    };
    run_to_done(&mut study, 100_000);
    let (elements, _) = study.mesh_size();
    let mut lines = Vec::new();
    let mut failures = Vec::new();
    for row in study.rows() {
        let allowed = limit(&case, row.quantity);
        lines.push(format!(
            "{name}: {} FEA {:.5} textbook {:.5} diff {:+.3}% (limit {}%, {} elements)",
            row.quantity, row.fea, row.textbook, row.percent_difference, allowed, elements
        ));
        let within = row.percent_difference.abs() <= allowed;
        if !within {
            failures.push(format!(
                "{name}: {} differs by {:+.3}%, limit {allowed}%",
                row.quantity, row.percent_difference
            ));
        }
    }
    if failures.is_empty() {
        Ok(lines)
    } else {
        Err(format!("{}\n{}", lines.join("\n"), failures.join("\n")))
    }
}

/// Runs the cases on a few threads, printing every line and failing with the failures listed.
fn run_all(cases: Vec<Case>) {
    let cases = std::sync::Mutex::new(cases.into_iter());
    let lines = std::sync::Mutex::new(Vec::new());
    let failures = std::sync::Mutex::new(Vec::new());
    let workers = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(12);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let Some(case) = cases.lock().expect("no panic holding lock").next() else {
                    return;
                };
                match run_case(case) {
                    Ok(found) => lines.lock().expect("no panic holding lock").extend(found),
                    Err(error) => failures.lock().expect("no panic holding lock").push(error),
                }
            });
        }
    });
    let lines = lines.into_inner().expect("lock");
    let failures = failures.into_inner().expect("lock");
    for line in lines {
        println!("{line}");
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn every_offered_combination_agrees_with_the_textbook() {
    run_all(all_cases());
}

#[test]
fn the_slenderest_and_stubbiest_corners_agree_with_the_textbook() {
    // The range ends set independently, shortest with tallest and longest with thinnest, at the
    // largest Poisson ratio, whose clamp restraint is strongest.
    let mut cases: Vec<(Combination, usize, f64, f64)> = Vec::new();
    let max_nu = extreme_poisson_materials()[1];
    for combination in solidfem_wasm::catalog::offered_combinations() {
        if combination.section == SectionKind::Tube && combination.load == LoadKind::Twist {
            continue;
        }
        let r = solidfem_wasm::catalog::ranges(
            combination.section,
            combination.restraint,
            combination.load,
        );
        cases.push((combination, max_nu, r.length.min, r.width.max));
        cases.push((combination, max_nu, r.length.max, r.width.min));
    }
    let lines = std::sync::Mutex::new(Vec::new());
    let failures = std::sync::Mutex::new(Vec::new());
    let queue = std::sync::Mutex::new(cases.into_iter());
    let workers = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(12);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let Some((combination, material, length, side)) =
                    queue.lock().expect("no panic holding lock").next()
                else {
                    return;
                };
                let mut request = request_at(combination, material, 1, Detail::Standard);
                request.length = length;
                request.width = side;
                request.height = side;
                let name = case_name(
                    combination,
                    material,
                    &format!("corner L={length} side={side}"),
                );
                let result = (|| -> Result<Vec<String>, String> {
                    let mut study =
                        DemoStudy::start(request).map_err(|e| format!("{name}: {e}"))?;
                    run_to_done(&mut study, 100_000);
                    let case = Case {
                        combination,
                        material,
                        level: 1,
                        detail: Detail::Standard,
                    };
                    let mut out = Vec::new();
                    let mut bad = Vec::new();
                    for row in study.rows() {
                        let allowed = limit(&case, row.quantity);
                        out.push(format!(
                            "{name}: {} diff {:+.3}% (limit {allowed}%)",
                            row.quantity, row.percent_difference
                        ));
                        let within = row.percent_difference.abs() <= allowed;
                        if !within {
                            bad.push(format!(
                                "{name}: {} differs by {:+.3}%, limit {allowed}%",
                                row.quantity, row.percent_difference
                            ));
                        }
                    }
                    if bad.is_empty() {
                        Ok(out)
                    } else {
                        Err(format!("{}\n{}", out.join("\n"), bad.join("\n")))
                    }
                })();
                match result {
                    Ok(found) => lines.lock().expect("no panic holding lock").extend(found),
                    Err(error) => failures.lock().expect("no panic holding lock").push(error),
                }
            });
        }
    });
    let lines = lines.into_inner().expect("lock");
    let failures = failures.into_inner().expect("lock");
    for line in lines {
        println!("{line}");
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn a_tube_in_twist_reports_no_textbook_answer_and_no_rows() {
    for restraint in [Restraint::FixedOneEnd, Restraint::FixedBothEnds] {
        let combination = Combination {
            section: SectionKind::Tube,
            restraint,
            load: LoadKind::Twist,
        };
        let request = request_at(combination, 0, 1, Detail::Standard);
        let mut study = DemoStudy::start(request).expect("offered");
        assert!(matches!(study.textbook(), Textbook::NoTextbook(_)));
        assert!(study.rows().is_empty());
        run_to_done(&mut study, 100_000);
        assert!(
            study.rows().is_empty(),
            "a combination with no textbook answer must not invent a number"
        );
        assert!(matches!(study.textbook(), Textbook::NoTextbook(_)));
        // The simulation still ran: the page shows its stress.
        assert!(study.view().max_stress > 0.0);
    }
}
