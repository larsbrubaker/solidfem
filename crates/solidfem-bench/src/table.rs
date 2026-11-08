//! Text tables of what was measured: one line per case, the same on native and in the browser, so
//! the two are compared directly.

use crate::pipeline::{Ending, Measured, Prepared};

/// The header of the parts table.
pub fn parts_header() -> &'static str {
    "case                          | cells     | tets     | nodes    | free DOF  | mesh s  | assembly s | CG iters | solve s  | recovery s | est. GiB | residual | status"
}

/// One row of the parts table.
pub fn parts_row(name: &str, prepared: &Prepared, measured: &Measured) -> String {
    let r = measured.report;
    let status = match measured.ending {
        Ending::Converged => "converged",
        Ending::Capped => "capped",
    };
    format!(
        "{name:<29} | {:>9} | {:>8} | {:>8} | {:>9} | {:>7.2} | {:>10.3} | {:>8} | {:>8.2} | {:>10.3} | {:>8.2} | {:>8.2e} | {status}",
        prepared.cells,
        prepared.tets,
        prepared.nodes,
        r.free_dofs,
        prepared.mesh_seconds,
        r.assembly_seconds,
        r.iterations,
        r.solve_seconds,
        r.recovery_seconds,
        prepared.estimated_bytes as f64 / (1u64 << 30) as f64,
        r.relative_residual,
    )
}

/// A case refused or failed, on one line.
pub fn refusal_row(name: &str, what: &str) -> String {
    format!("{name:<29} | {what}")
}
