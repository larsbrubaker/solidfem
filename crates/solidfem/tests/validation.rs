// The validation area's test binary: end-to-end studies against closed-form physics. Each case
// prints a `VALIDATION` line (run with --no-capture in release for measured numbers).

#[path = "validation/support.rs"]
mod support;

#[path = "validation/cantilever.rs"]
mod cantilever;
#[path = "validation/determinism.rs"]
mod determinism;
#[path = "validation/refusals.rs"]
mod refusals;
#[path = "validation/tension.rs"]
mod tension;
#[path = "validation/torsion.rs"]
mod torsion;
#[path = "validation/two_material.rs"]
mod two_material;
