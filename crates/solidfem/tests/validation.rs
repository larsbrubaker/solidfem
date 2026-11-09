// The validation area's test binary: end-to-end studies against closed-form physics. Each case
// prints a `VALIDATION` line (run with --no-capture in release for measured numbers).

#[path = "validation/support.rs"]
mod support;

#[path = "validation/cantilever.rs"]
mod cantilever;
#[path = "validation/determinism.rs"]
mod determinism;
#[path = "validation/multigrid.rs"]
mod multigrid;
#[path = "validation/pipeline.rs"]
mod pipeline;
#[path = "validation/prescribed.rs"]
mod prescribed;
#[path = "validation/refusals.rs"]
mod refusals;
// The mesher's own test shapes, so a test here meshes the same input bit for bit.
#[allow(dead_code)]
#[path = "../../solidfem-tetra/tests/surface/shapes.rs"]
mod shapes;
#[path = "validation/shell_core.rs"]
mod shell_core;
#[path = "validation/tension.rs"]
mod tension;
#[path = "validation/torsion.rs"]
mod torsion;
#[path = "validation/two_material.rs"]
mod two_material;
