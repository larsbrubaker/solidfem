// Every example part the page offers runs end to end from its committed asset through the same
// `ExampleStudy` the page drives: meshed, held and loaded by its benchmark case, solved in small
// slices, with the load the part carries balanced by its reactions. What the page says about a
// part comes from one table, and that table cannot drift from the benchmarks or its assets.

use std::path::{Path, PathBuf};

use solidfem::elastic::{Quadrature, Report, Solution};
use solidfem_bench::benchmarks::cases::{
    ktn_finite_width, LOAD_DISK, PATCH_W_REFERENCE, ROOT_PEAK_REFERENCE,
};
use solidfem_bench::benchmarks::pipeline::load_balance;
use solidfem_bench::benchmarks::verdict::LOAD_BALANCE_LIMIT;
use solidfem_wasm::engine::StudyState;
use solidfem_wasm::example_study::{
    spread_evenly, ExampleMaterial, ExampleState, ExampleStudy, MAX_HELD_MARKERS, MAX_LOAD_ARROWS,
};
use solidfem_wasm::examples::{find, ExamplePart, EXAMPLES};

/// The asset a part's entry names.
pub fn asset_path(part: &ExamplePart) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../public/parts")
        .join(part.surface_file)
}

pub fn read_asset(part: &ExamplePart) -> String {
    std::fs::read_to_string(asset_path(part)).expect("the part's committed surface")
}

pub fn benchmark_material(part: &ExamplePart) -> ExampleMaterial {
    ExampleMaterial {
        material: part.benchmark_material(),
        yield_strength: None,
    }
}

/// Runs a started study to the end with `budget` per step, returning the finished study.
fn run_to_end(mut study: ExampleStudy, budget: u32) -> ExampleStudy {
    loop {
        match study.step(budget) {
            ExampleState::Study(StudyState::Done) => return study,
            ExampleState::Failed | ExampleState::Study(StudyState::Failed) => {
                panic!("{} failed: {:?}", study.part().id, study.failure())
            }
            ExampleState::Cancelled => panic!("{} was cancelled", study.part().id),
            _ => {}
        }
    }
}

/// Steps a started study until its mesh is ready and its markers exist.
fn run_to_prepared(mut study: ExampleStudy) -> ExampleStudy {
    while study.study().is_none() {
        match study.step(200) {
            ExampleState::Failed => panic!("{} failed: {:?}", study.part().id, study.failure()),
            ExampleState::Cancelled => panic!("{} was cancelled", study.part().id),
            _ => {}
        }
    }
    study
}

#[test]
fn every_part_runs_from_its_asset_to_a_converged_balanced_result() {
    for part in EXAMPLES.iter() {
        let study =
            ExampleStudy::start(part, &read_asset(part), benchmark_material(part), 1.0, None);
        let study = run_to_end(study, 25);
        let solve = study.study().expect("the part reached the solver");
        assert!(
            solve.residual() <= 1e-6,
            "{}: residual {}",
            part.id,
            solve.residual()
        );
        let meshed = study.meshed().expect("the part stayed meshed");
        let balance = load_balance(
            &meshed.mesh,
            &meshed.supports,
            solve.point_stresses(),
            Quadrature::FourPoint,
        )
        .expect("every stress point has a valid element");
        assert!(
            balance.error <= LOAD_BALANCE_LIMIT,
            "{}: load balance error {:.3e} of {} gross",
            part.id,
            balance.error,
            balance.gross
        );
        let view = solve.view();
        assert!(
            view.max_stress > 0.0 && view.max_displacement > 0.0,
            "{} has no result",
            part.id
        );
        assert_eq!(
            view.von_mises.len(),
            view.positions.len() / 3,
            "{}",
            part.id
        );
    }
}

#[test]
fn every_part_names_a_real_case_and_density_and_has_its_asset() {
    for part in EXAMPLES.iter() {
        let (case, density) = part.case_and_density().unwrap_or_else(|| {
            panic!(
                "{}: no benchmark case {} with surface {}",
                part.id, part.case_id, part.surface_file
            )
        });
        assert_eq!(case.id, part.case_id);
        assert_eq!(density.surface_file, part.surface_file);
        assert!(
            asset_path(part).is_file(),
            "{}: no committed asset {}",
            part.id,
            part.surface_file
        );
    }
}

/// The named quantity a real solve of `part`'s own page mesh measures, through the benchmark's
/// own `measure`.
fn measured_on_page_mesh(part: &'static ExamplePart, name: &str) -> f64 {
    let study = ExampleStudy::start(part, &read_asset(part), benchmark_material(part), 1.0, None);
    let study = run_to_end(study, 200);
    let solve = study.study().expect("the part reached the solver");
    let solution = Solution {
        displacements: solve.displacements().to_vec(),
        point_stresses: solve.point_stresses().to_vec(),
        nodal_stress: solve.nodal_stress().to_vec(),
        nodal_von_mises: solve.nodal_von_mises().to_vec(),
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
    };
    let (case, _) = part.case_and_density().expect("a real case");
    let meshed = study.meshed().expect("meshed");
    let values = (case.measure)(&meshed.mesh, &solution).expect("the case measures its part");
    values
        .iter()
        .find(|(key, _)| key == name)
        .unwrap_or_else(|| panic!("{}: no {name}", part.id))
        .1
}

/// Each demo-only surface's status line quotes the error of that very mesh, so the percentage on
/// the page is a measurement and goes stale loudly. The status gives one decimal, so the measured
/// value must round to the quoted text.
#[test]
fn a_demo_only_surfaces_status_quotes_its_own_measured_error() {
    let cases: [(&str, &str, f64); 3] = [
        ("plate", "ktn", ktn_finite_width(0.1)),
        ("gear", "root_peak", ROOT_PEAK_REFERENCE),
        ("hollow", "patch_w", PATCH_W_REFERENCE),
    ];
    let mut wrong = Vec::new();
    for (id, name, reference) in cases {
        let part = find(id).expect("offered");
        let measured = measured_on_page_mesh(part, name);
        let percent = 100.0 * (measured - reference).abs() / reference.abs();
        let quoted = format!("{percent:.1}%");
        println!("MEASURED {id}: {measured:.6e}, reference {reference:.6e}, error {quoted}");
        if !part.status.contains(&quoted) {
            wrong.push(format!(
                "{id}: measures {quoted}, status says: {}",
                part.status
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_demo_only_surface_states_the_generators_sizing() {
    for part in EXAMPLES.iter() {
        let Some(sizing) = part.demo_sizing else {
            continue;
        };
        let stem = part.surface_file.strip_suffix(".txt").expect("a text file");
        let generated = solidfem_bench::surfaces::demo_only_parts()
            .into_iter()
            .flat_map(|p| p.densities)
            .find(|d| d.stem == stem)
            .unwrap_or_else(|| panic!("{stem} is not a demo-only surface of the generator"));
        assert_eq!(generated.size, sizing, "{stem}");
    }
}

#[test]
fn held_markers_and_load_arrows_are_capped_and_spread_evenly() {
    // Hundreds of points on a plane, numbered row by row as a mesher might: the first N would be
    // one corner, so spread markers must reach all four corners' quarters.
    let points: Vec<[f64; 3]> = (0..600)
        .map(|i| [f64::from(i % 30), f64::from(i / 30), 0.0])
        .collect();
    let chosen = spread_evenly(&points, 40);
    assert!(!chosen.is_empty() && chosen.len() <= 40, "{}", chosen.len());
    for (low_x, low_y) in [(true, true), (true, false), (false, true), (false, false)] {
        assert!(
            chosen.iter().any(|i| {
                let p = points[*i];
                (p[0] < 15.0) == low_x && (p[1] < 10.0) == low_y
            }),
            "no marker in quarter ({low_x}, {low_y}) of {chosen:?}"
        );
    }
    for part in EXAMPLES.iter() {
        let study =
            ExampleStudy::start(part, &read_asset(part), benchmark_material(part), 1.0, None);
        let study = run_to_prepared(study);
        let held = study.held_points().len() / 3;
        let arrows = study.load_arrows().len() / 7;
        assert!(
            held > 0 && held <= MAX_HELD_MARKERS,
            "{}: {held} held",
            part.id
        );
        assert!(
            arrows > 0 && arrows <= MAX_LOAD_ARROWS,
            "{}: {arrows} arrows",
            part.id
        );
    }
}

#[test]
fn the_hollow_box_load_is_drawn_only_inside_its_disk() {
    let part = find("hollow").expect("offered");
    let study = ExampleStudy::start(part, &read_asset(part), benchmark_material(part), 1.0, None);
    let study = run_to_prepared(study);
    let (centre, radius) = LOAD_DISK;
    let arrows = study.load_arrows();
    assert!(arrows.len() >= 7, "only {} arrows", arrows.len() / 7);
    for arrow in arrows.as_chunks::<7>().0 {
        // A press lands where tail + direction * length meets the surface.
        let tip: Vec<f64> = (0..3)
            .map(|axis| f64::from(arrow[axis]) + f64::from(arrow[axis + 3]) * f64::from(arrow[6]))
            .collect();
        let distance = ((tip[0] - centre[0]).powi(2) + (tip[1] - centre[1]).powi(2)).sqrt();
        assert!(distance <= radius + 1e-4, "arrow lands at {tip:?}");
        assert!(
            (tip[2] - 20.0).abs() <= 1e-4,
            "arrow lands off the top: {tip:?}"
        );
        assert!(arrow[5] < 0.0, "a press points down: {arrow:?}");
    }
}

#[test]
fn each_part_opens_looking_along_the_direction_that_shows_its_shape() {
    // Parts whose outline is in the x-y plane open looking mostly along z, with some tilt so the
    // part still reads as a solid; the ones that show their bending open from the side.
    for id in [
        "ring", "plate", "l-sharp", "l-fillet", "le1", "le10", "hook", "gear",
    ] {
        let part = find(id).expect("offered");
        let [x, y, z] = part.view;
        let length = (x * x + y * y + z * z).sqrt();
        assert!(
            z / length >= 0.65,
            "{id} does not look down its thin axis: {:?}",
            part.view
        );
        assert!(z / length <= 0.97, "{id} has no tilt: {:?}", part.view);
    }
    for id in ["strip", "snap"] {
        let [x, y, z] = find(id).expect("offered").view;
        assert!(
            y.abs() > x.abs() && y.abs() > z.abs(),
            "{id} is not viewed from the side"
        );
    }
}

#[test]
fn a_changed_load_size_scales_the_results_without_solving_again() {
    let part = find("snap").expect("the snap-fit is offered");
    let study = ExampleStudy::start(part, &read_asset(part), benchmark_material(part), 1.0, None);
    let mut study = run_to_end(study, 200);
    let once = study.study().expect("solved").view();
    let iterations = study.study().expect("solved").iterations();
    study.set_load_scale(3.0);
    assert_eq!(study.study().expect("solved").iterations(), iterations);
    let thrice = study.study().expect("solved").view();
    for (a, b) in once.von_mises.iter().zip(&thrice.von_mises) {
        assert!((b - 3.0 * a).abs() <= 1e-5 * a.abs().max(1.0));
    }
    assert!((thrice.max_displacement - 3.0 * once.max_displacement).abs() <= 1e-9);
}

#[test]
fn a_changed_material_keeps_the_mesh_of_the_part() {
    let part = find("snap").expect("the snap-fit is offered");
    let first = ExampleStudy::start(part, &read_asset(part), benchmark_material(part), 1.0, None);
    let first = run_to_end(first, 200);
    let elements = first.meshed().expect("meshed").mesh.tets().len();
    let meshed = first.take_meshed();
    let second = ExampleStudy::start(part, "not read", benchmark_material(part), 1.0, meshed);
    assert_eq!(
        second.state(),
        ExampleState::Preparing,
        "it does not mesh again"
    );
    let second = run_to_end(second, 200);
    assert_eq!(second.meshed().expect("meshed").mesh.tets().len(), elements);
}

#[test]
fn a_part_the_mesher_cannot_take_says_so_in_words() {
    let part = find("snap").expect("the snap-fit is offered");
    let study = ExampleStudy::start(part, "0 0\n", benchmark_material(part), 1.0, None);
    assert_eq!(study.state(), ExampleState::Failed);
    assert!(
        study.failure().is_some_and(|s| s.contains("refused")),
        "{:?}",
        study.failure()
    );
}

#[test]
fn a_failing_part_says_so_before_the_numbers() {
    for part in EXAMPLES.iter() {
        match part.verdict {
            solidfem_wasm::examples::Verdict::Fail => assert!(
                part.status.starts_with("Not trustworthy yet")
                    || part.status.starts_with("Not checked"),
                "{}: {:?}",
                part.id,
                part.status
            ),
            solidfem_wasm::examples::Verdict::Limited => assert!(
                !part.status.starts_with("Matches"),
                "{}: {:?}",
                part.id,
                part.status
            ),
            solidfem_wasm::examples::Verdict::Pass => {
                assert!(part.status.starts_with("Matches"), "{}", part.id)
            }
        }
    }
}
