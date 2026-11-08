//! solidfem-bench: measures the pipeline from a signed-distance field through the Quartet mesher
//! into the solver. `pipeline.rs` runs one [`Case`] (a shape, supports and a load), `shapes.rs`
//! holds the signed-distance functions, `cases.rs` the benchmark cases, `known_answer.rs` the
//! cantilever whose result is known in closed form, and `table.rs` prints what was measured.
//! `browser.rs` drives the same pipeline from plain wasm exports, in slices, with the page's
//! clock; the native binary (`src/bin/bench.rs`) prints the tables and writes the browser page.
//! `corpus` is the separate native-only measurement of real triangle meshes and the surface
//! remesher (`src/bin/corpus.rs` runs it). `sliver` is the native-only measurement of what bad
//! boundary triangles cost a stress result (`src/bin/sliver.rs` runs it). This crate depends on the
//! mesher and is never published; the library crates do not.

#[cfg(not(target_arch = "wasm32"))]
pub mod corpus;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use corpus::g3_mount::Mesh;
#[cfg(not(target_arch = "wasm32"))]
pub mod sliver;

pub mod cases;
pub mod known_answer;
pub mod pipeline;
pub mod shapes;
pub mod table;

#[cfg(target_arch = "wasm32")]
mod browser;
