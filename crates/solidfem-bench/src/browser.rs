//! The browser benchmark's wasm exports: the page calls `bench_advance` repeatedly with a timeout
//! between calls, so the run proceeds in slices and the page stays responsive. Each call does one
//! bounded piece of work: meshing and assembling one case, a slice of the solve, or recovering
//! results. Time comes from the page's `performance.now` through the `env.now` import. Plain
//! exports, no generated glue: the page's script is the only binding this needs.

use std::cell::RefCell;

use solidfem_elastic::Clock;

use crate::cases::{block_case, hollow_box_case, small_hollow_box_case};
use crate::known_answer::{
    cantilever_case, evaluate, mid_span_bending_stress, timoshenko_deflection, EndSelection,
};
use crate::pipeline::{advance, finish, prepare, Case, Ending, Prepared, Refusal};
use crate::table::{parts_header, parts_row, refusal_row};

/// wasm32 addresses 4 GiB; refuse past 3.5 GiB so the page reports instead of trapping.
const BYTE_LIMIT: u64 = 3 * (1 << 30) + (1 << 29);
/// The solve cap, matching the native runs.
const CAP_SECONDS: f64 = 120.0;
/// Iterations per slice between returns to the page.
const SLICE: usize = 25;

#[link(wasm_import_module = "env")]
extern "C" {
    fn now() -> f64;
    fn log(pointer: *const u8, length: usize);
}

fn read_now() -> f64 {
    // SAFETY: `now` is the page's import, taking nothing and returning a number; it touches no
    // Rust state.
    unsafe { now() }
}

fn write_line(line: &str) {
    // SAFETY: the page's `log` reads `length` bytes at `pointer` inside the module's memory and
    // copies them; the slice is valid for the duration of the call.
    unsafe { log(line.as_ptr(), line.len()) }
}

struct Running {
    prepared: Prepared,
    name: String,
    known_answer: bool,
    solve_start: f64,
}

#[derive(Default)]
struct State {
    next_case: usize,
    header_written: bool,
    running: Option<Running>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// The browser cases: the middle known-answer grid, then a 2 mm hollow box and a smaller one.
fn case_at(index: usize) -> Option<(Case, bool)> {
    match index {
        0 => Some((block_case(), false)),
        1 => Some((cantilever_case(6), true)),
        2 => Some((hollow_box_case(2.0), false)),
        3 => Some((small_hollow_box_case(), false)),
        _ => None,
    }
}

fn new_clock() -> Clock {
    Box::new(read_now)
}

/// Runs the next bounded piece of work; returns 1 while there is more to do, 0 when finished.
#[no_mangle]
pub extern "C" fn bench_advance() -> i32 {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if !state.header_written {
            state.header_written = true;
            write_line(parts_header());
            return 1;
        }
        if state.running.is_none() {
            return begin_next(&mut state);
        }
        step_running(&mut state);
        1
    })
}

fn begin_next(state: &mut State) -> i32 {
    let Some((case, known_answer)) = case_at(state.next_case) else {
        return 0;
    };
    state.next_case += 1;
    match prepare(&case, &mut new_clock, BYTE_LIMIT) {
        Ok(prepared) => {
            state.running = Some(Running {
                prepared,
                name: case.name,
                known_answer,
                solve_start: read_now(),
            });
        }
        Err(refusal) => write_line(&refusal_row(&case.name, &refusal_text(&refusal))),
    }
    1
}

fn step_running(state: &mut State) {
    let Some(mut running) = state.running.take() else {
        return;
    };
    let ended = match advance(&mut running.prepared, SLICE) {
        Ok(true) if read_now() - running.solve_start > CAP_SECONDS => Some(Ending::Capped),
        Ok(true) => None,
        Ok(false) => Some(Ending::Converged),
        Err(refusal) => {
            write_line(&refusal_row(&running.name, &refusal_text(&refusal)));
            return;
        }
    };
    match ended {
        None => state.running = Some(running),
        Some(ending) => match finish(&mut running.prepared, ending) {
            Ok(measured) => {
                write_line(&parts_row(&running.name, &running.prepared, &measured));
                if running.known_answer {
                    if let Some(solution) = &measured.solution {
                        let (tip, stress) = evaluate(
                            &running.prepared.mesh,
                            solution,
                            running.prepared.dx,
                            EndSelection::Plane,
                        );
                        write_line(&format!(
                            "  tip error {:+.3}%  peak stress error {:+.3}%",
                            100.0 * (tip / timoshenko_deflection() - 1.0),
                            100.0 * (stress / mid_span_bending_stress() - 1.0)
                        ));
                    }
                }
            }
            Err(refusal) => write_line(&refusal_row(&running.name, &refusal_text(&refusal))),
        },
    }
}

fn refusal_text(refusal: &Refusal) -> String {
    match refusal {
        Refusal::TooMuchMemory { estimated_gib } => {
            format!("refused: estimate {estimated_gib:.2} GiB exceeds the wasm32 limit")
        }
        Refusal::Failed(why) => format!("failed: {why}"),
    }
}
