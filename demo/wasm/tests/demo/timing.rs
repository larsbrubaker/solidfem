// Measures the demo's size cap, ignored in normal runs: on this machine's native release build,
// as the proxy for a browser run, printing phase totals and longest slices. Run with
// `cargo nextest run --workspace --release --test demo --run-ignored only --no-capture timing`.

use std::time::Instant;

use solidfem_wasm::catalog::{Combination, Detail, LoadKind, Restraint, SectionKind};
use solidfem_wasm::engine::{DemoStudy, StudyState, ELEMENTS_PER_BUDGET_UNIT};

use crate::support::request_at;

/// Times one request through the phases at the page's slice size; prints phase totals and the
/// longest single slice of each.
fn time_request(name: &str, combination: Combination, level: usize, detail: Detail, budget: u32) {
    let mut request = request_at(combination, 0, level, detail);
    request.width = 10.0;
    request.height = 10.0;
    let start = Instant::now();
    let mut study = match DemoStudy::start(request) {
        Ok(study) => study,
        Err(error) => {
            println!("TIMING {name}: refused: {error}");
            return;
        }
    };
    let plan_ms = start.elapsed().as_secs_f64() * 1e3;
    let mut totals = [0.0f64; 3];
    let mut longest = [0.0f64; 3];
    loop {
        let phase = match study.state() {
            StudyState::Assembling => 0,
            StudyState::Solving => 1,
            StudyState::Recovering => 2,
            StudyState::Done => break,
            StudyState::Failed => panic!("failed: {:?}", study.failure()),
            StudyState::Cancelled => panic!("cancelled"),
        };
        let slice = Instant::now();
        study.step(budget);
        let ms = slice.elapsed().as_secs_f64() * 1e3;
        totals[phase] += ms;
        longest[phase] = longest[phase].max(ms);
    }
    let (elements, nodes) = study.mesh_size();
    println!(
        "TIMING {name}: elements={elements} nodes={nodes} iterations={} plan_ms={plan_ms:.0} \
         assembly_ms={:.0} (max slice {:.1}) solve_ms={:.0} (max slice {:.1}) recovery_ms={:.0} \
         (max slice {:.1}) [budget {budget}, {} elements per unit]",
        study.iterations(),
        totals[0],
        longest[0],
        totals[1],
        longest[1],
        totals[2],
        longest[2],
        ELEMENTS_PER_BUDGET_UNIT,
    );
}

#[test]
#[ignore = "measures the cap on the native release build; not a behaviour test"]
fn timing_at_the_cap() {
    let solid = Combination {
        section: SectionKind::Solid,
        restraint: Restraint::FixedBothEnds,
        load: LoadKind::Push,
    };
    let tube = Combination {
        section: SectionKind::Tube,
        restraint: Restraint::FixedBothEnds,
        load: LoadKind::Push,
    };
    // One-shot budgets: phase totals without the per-frame preview, for the cap itself.
    time_request("solid fine largest", solid, 2, Detail::Fine, 1_000_000);
    time_request(
        "tube standard largest",
        tube,
        0,
        Detail::Standard,
        1_000_000,
    );
}
