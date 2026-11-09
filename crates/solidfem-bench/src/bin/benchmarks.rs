//! The real-part benchmark runner: `solidfem-benchmarks <surface-dir> [case-id ...]` runs the
//! named cases (every case when none are named) with each solve capped at `SOLVE_CAP_SECONDS` (10
//! minutes unless the variable overrides it), prints a table per part and a final verdict for
//! each: PASS, FAIL, NOT ASSESSED (a solve hit its cap, with no error quoted) or OUT OF SCOPE. A
//! PASS that tests less than an accuracy reference says in brackets what it did test.
//! `<surface-dir>` holds the remeshed surfaces in the text format `solidfem-gate file` reads, one
//! file per part and density, named in `benchmarks/cases/`. Run in release.

use std::io::Write as _;
use std::path::Path;

use solidfem_bench::benchmarks::cases::{all_cases, Case};
use solidfem_bench::benchmarks::pipeline::{read_surface, run_case, CaseResult};

/// The cap on one solve, seconds.
const DEFAULT_CAP_SECONDS: f64 = 600.0;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(directory) = args.first().cloned() else {
        eprintln!("usage: solidfem-benchmarks <surface-dir> [case-id ...]");
        eprintln!(
            "cases: {}",
            all_cases()
                .iter()
                .map(|case| case.id)
                .collect::<Vec<_>>()
                .join(" ")
        );
        std::process::exit(2);
    };
    let cap = std::env::var("SOLVE_CAP_SECONDS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(DEFAULT_CAP_SECONDS);
    let wanted = &args[1..];
    let cases: Vec<Case> = all_cases()
        .into_iter()
        .filter(|case| wanted.is_empty() || wanted.iter().any(|id| id == case.id))
        .collect();
    let cases: Vec<Case> = if std::env::var("BENCH_FINEST_ONLY").is_ok() {
        // A diagnostic run of only the finest density of each case; its verdict means nothing.
        cases
            .into_iter()
            .map(|mut case| {
                case.densities = case
                    .densities
                    .split_off(case.densities.len().saturating_sub(1));
                case
            })
            .collect()
    } else {
        cases
    };
    let mut summary = Vec::new();
    for case in &cases {
        println!("=== {} ({})", case.title, case.id);
        let dir = directory.clone();
        let mut surface_of = |density: &solidfem_bench::benchmarks::cases::Density| {
            let path = Path::new(&dir).join(&density.surface_file);
            read_surface(&path.to_string_lossy())
        };
        let mut progress = |line: &str| {
            println!("  ... {line}");
            std::io::stdout().flush().ok();
        };
        let result = run_case(case, &mut surface_of, cap, &mut progress);
        print_case(case, &result);
        summary.push((case.id, result.verdict.to_string()));
        std::io::stdout().flush().ok();
    }
    println!("=== Verdicts");
    for (id, verdict) in summary {
        println!("{id:<12} {verdict}");
    }
}

fn print_case(case: &Case, result: &CaseResult) {
    for row in &result.rows {
        let values: Vec<String> = row
            .values
            .iter()
            .map(|(name, value)| format!("{name}={value:.6e}"))
            .collect();
        println!(
            "  {:<10} {:>8} nodes {:>8} elements {:>9} dofs, min dihedral {:>5.2} deg, {:>5} iterations, residual {:.2e}, {:>7.1} s, balance applied {:.4e} reaction {:.4e} error {:.2e}",
            row.label,
            row.nodes,
            row.elements,
            row.free_dofs,
            row.min_dihedral,
            row.iterations,
            row.residual,
            row.seconds,
            magnitude(row.balance.applied),
            magnitude(row.balance.reaction),
            row.balance.error
        );
        println!("             {}", values.join("  "));
    }
    println!("  VERDICT {}: {}", case.id, result.verdict);
}

fn magnitude(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
