// Measures the demo's size cap: every offered section, restraint and load at the top of its ranges
// (longest bar, smallest section, largest Poisson ratio, both wall limits), then the longest single
// slice of each phase at the page's budget. Ignored in normal runs, since it is a measurement and
// not a behaviour test; on an otherwise idle machine, in the release profile that stands in for a
// browser run:
// `cargo nextest run --workspace --release --test demo --run-ignored only --no-capture timing`

use std::time::Instant;

use solidfem_wasm::catalog::{offered_combinations, ranges, SectionKind, StudyRequest, MATERIALS};
use solidfem_wasm::engine::{DemoStudy, StudyState};
use solidfem_wasm::plan::planned_elements;

use crate::support::{case_name, request_at};

/// The budget the page is expected to pass to `step` on each frame.
const PAGE_BUDGET: u32 = 10;

/// Runs of each case; the fastest is reported.
const REPEATS: usize = 3;

/// A one-shot budget: the phases' own cost, with no per-frame preview.
const WHOLE_STUDY: u32 = u32::MAX;

/// Runs a study, returning milliseconds per phase, the longest slice of each and the start cost.
fn time_study(request: StudyRequest, budget: u32) -> (DemoStudy, [f64; 3], [f64; 3], f64) {
    let start = Instant::now();
    let mut study = DemoStudy::start(request).expect("every offered position starts");
    let start_ms = start.elapsed().as_secs_f64() * 1e3;
    let (mut totals, mut longest) = ([0.0f64; 3], [0.0f64; 3]);
    loop {
        let phase = match study.state() {
            StudyState::Assembling => 0,
            StudyState::Solving => 1,
            StudyState::Recovering => 2,
            StudyState::Done => break,
            other => panic!("{other:?}: {:?}", study.failure()),
        };
        let slice = Instant::now();
        study.step(budget);
        let ms = slice.elapsed().as_secs_f64() * 1e3;
        totals[phase] += ms;
        longest[phase] = longest[phase].max(ms);
    }
    (study, totals, longest, start_ms)
}

/// The whole cost of one timed run, in milliseconds.
fn phase_total(run: &(DemoStudy, [f64; 3], [f64; 3], f64)) -> f64 {
    run.3 + run.1[0] + run.1[1] + run.1[2]
}

#[test]
#[ignore = "measures the cap in the release profile; not a behaviour test"]
fn timing_every_offered_study_at_the_top_of_its_ranges() {
    let highest_poisson = (0..MATERIALS.len())
        .max_by(|a, b| {
            MATERIALS[*a]
                .poisson_ratio
                .total_cmp(&MATERIALS[*b].poisson_ratio)
        })
        .expect("materials");
    let mut slowest: Option<(f64, String, StudyRequest)> = None;
    for combination in offered_combinations() {
        let r = ranges(combination.section, combination.restraint, combination.load);
        let walls = match combination.section {
            SectionKind::Solid => vec![0.0],
            SectionKind::Tube => {
                let w = r.wall_fraction.expect("a tube has a wall");
                vec![w.min, w.max]
            }
        };
        for wall in walls {
            let mut request = request_at(combination, highest_poisson, 2);
            request.length = r.length.max;
            request.width = r.width.min;
            request.height = r.height.min;
            request.wall_fraction = wall;
            let name = case_name(combination, highest_poisson, &format!("wall {wall}"));
            // The fastest of a few runs: other work on the machine only ever adds time, so the
            // minimum is the closest a shared machine gets to an idle one.
            let (study, totals, _, start_ms) = (0..REPEATS)
                .map(|_| time_study(request, WHOLE_STUDY))
                .min_by(|a, b| phase_total(a).total_cmp(&phase_total(b)))
                .expect("at least one run");
            let (elements, nodes) = study.mesh_size();
            let total = start_ms + totals[0] + totals[1] + totals[2];
            println!(
                "TIMING {name}: elements={elements} nodes={nodes} iterations={} \
                 start_ms={start_ms:.0} assembly_ms={:.0} solve_ms={:.0} recovery_ms={:.0} \
                 total_s={:.2}",
                study.iterations(),
                totals[0],
                totals[1],
                totals[2],
                total / 1e3
            );
            assert_eq!(elements, planned_elements(&request));
            if slowest.as_ref().is_none_or(|(t, _, _)| total > *t) {
                slowest = Some((total, name, request));
            }
        }
    }
    let (_, name, request) = slowest.expect("something is offered");
    let (study, totals, longest, start_ms) = time_study(request, PAGE_BUDGET);
    let view = Instant::now();
    let summary = study.view();
    let view_ms = view.elapsed().as_secs_f64() * 1e3;
    println!(
        "SLICES {name} at budget {PAGE_BUDGET}: start_ms={start_ms:.1} longest assembly={:.1}ms \
         solve={:.1}ms recovery={:.1}ms; totals assembly={:.0}ms solve={:.0}ms recovery={:.0}ms; \
         view_ms={view_ms:.1} ({} vertices)",
        longest[0],
        longest[1],
        longest[2],
        totals[0],
        totals[1],
        totals[2],
        summary.von_mises.len()
    );
}
