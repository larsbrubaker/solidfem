// Measures what each example part costs at its density on the page: mesh, setup and solve time,
// iterations and the longest single slice at the page's budget, for both preconditioners. Ignored
// in normal runs; in the release profile that stands in for a browser run:
// `cargo nextest run --workspace --release --test demo --run-ignored only --no-capture parts_timing`
// with `SOLIDFEM_PARTS=plate,gear` set to time only those.

use std::time::Instant;

use solidfem_wasm::engine::StudyState;
use solidfem_wasm::example_study::{ExampleState, ExampleStudy};
use solidfem_wasm::examples::EXAMPLES;

use crate::parts::{benchmark_material, read_asset};

/// The budget the page starts its slices at and grows from; the longest slice is the worst frame.
const PAGE_BUDGET: u32 = 10;

/// Iterations past which a solve is reported as not worth waiting for on a page.
const GIVE_UP_ITERATIONS: u64 = 3_000;

#[derive(Default)]
struct Times {
    meshing: f64,
    longest_meshing: f64,
    preparing: f64,
    assembly: f64,
    longest_assembly: f64,
    solve: f64,
    longest_solve: f64,
    recovery: f64,
}

fn run(part: &'static solidfem_wasm::examples::ExamplePart, multigrid: bool) -> String {
    let text = read_asset(part);
    let parse = Instant::now();
    let mut study = ExampleStudy::start_with_preconditioner(
        part,
        &text,
        benchmark_material(part),
        1.0,
        None,
        multigrid,
    );
    let mut times = Times {
        meshing: parse.elapsed().as_secs_f64() * 1e3,
        ..Times::default()
    };
    let mut gave_up = false;
    loop {
        let state = study.state();
        match state {
            ExampleState::Study(StudyState::Done) => break,
            ExampleState::Failed | ExampleState::Study(StudyState::Failed) => {
                return format!("{}: REFUSED: {:?}", part.id, study.failure());
            }
            _ => {}
        }
        let slice = Instant::now();
        study.step(PAGE_BUDGET);
        let ms = slice.elapsed().as_secs_f64() * 1e3;
        match state {
            ExampleState::Meshing | ExampleState::Refining => {
                times.meshing += ms;
                times.longest_meshing = times.longest_meshing.max(ms);
            }
            ExampleState::Preparing => times.preparing += ms,
            ExampleState::Study(StudyState::Assembling) => {
                times.assembly += ms;
                times.longest_assembly = times.longest_assembly.max(ms);
            }
            ExampleState::Study(StudyState::Solving) => {
                times.solve += ms;
                times.longest_solve = times.longest_solve.max(ms);
            }
            ExampleState::Study(StudyState::Recovering) => times.recovery += ms,
            _ => {}
        }
        if study
            .study()
            .is_some_and(|s| s.iterations() > GIVE_UP_ITERATIONS)
        {
            gave_up = true;
            break;
        }
    }
    let solve = study.study().expect("reached the solver");
    let (elements, nodes) = solve.mesh_size();
    format!(
        "{:<10} {:<9} elements={elements:<6} nodes={nodes:<7} mesh_ms={:<7.0} longest_mesh_ms={:<6.1} \
         prepare_ms={:<5.1} assemble_ms={:<7.1} longest_assemble_ms={:<7.1} solve_ms={:<8.0} \
         longest_solve_ms={:<6.1} iters={:<6} recover_ms={:<6.1}{}",
        part.id,
        if multigrid { "multigrid" } else { "jacobi" },
        times.meshing,
        times.longest_meshing,
        times.preparing,
        times.assembly,
        times.longest_assembly,
        times.solve,
        times.longest_solve,
        solve.iterations(),
        times.recovery,
        if gave_up { " (gave up)" } else { "" },
    )
}

#[test]
#[ignore = "measures the example parts in the release profile; not a behaviour test"]
fn timing_every_example_part_with_multigrid() {
    let wanted = std::env::var("SOLIDFEM_PARTS").unwrap_or_default();
    for part in EXAMPLES.iter() {
        if wanted.is_empty() || wanted.split(',').any(|id| id == part.id) {
            println!("{}", run(part, true));
        }
    }
}

#[test]
#[ignore = "measures the compact example parts with Jacobi; not a behaviour test"]
fn timing_compact_example_parts_with_jacobi() {
    for id in ["ring", "hook"] {
        let part = solidfem_wasm::examples::find(id).expect("offered");
        println!("{}", run(part, false));
    }
}
