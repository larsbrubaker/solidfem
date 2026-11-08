//! solidfem-bench: measures the pipeline from a signed-distance field through the Quartet mesher
//! into the solver. `pipeline.rs` runs one [`Case`] (a shape, supports and a load), `shapes.rs`
//! holds the signed-distance functions, `cases.rs` the benchmark cases, `known_answer.rs` the
//! cantilever whose result is known in closed form, and `table.rs` prints what was measured.
//! `browser.rs` drives the same pipeline from plain wasm exports, in slices, with the page's
//! clock; the native binary (`src/bin/bench.rs`) prints the tables and writes the browser page.
//! This crate depends on the mesher and is never published; the library crates do not.

pub mod cases;
pub mod known_answer;
pub mod pipeline;
pub mod shapes;
pub mod table;

#[cfg(target_arch = "wasm32")]
mod browser;
