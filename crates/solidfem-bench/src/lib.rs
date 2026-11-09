//! solidfem-bench: measurements of what real inputs do to the mesher and the solver, native only,
//! never published. `corpus` measures real triangle meshes, solidmesh's `remesh_for_analysis` on
//! them and the volume mesher on what it produces (`src/bin/corpus.rs` runs it); `sliver`
//! measures what bad boundary triangles cost a stress result (`src/bin/sliver.rs` runs it). The
//! end-of-mesher gate, this workspace's mesher into the solver on a thin plate, is
//! `src/bin/gate.rs`; what it prints on request about the multigrid is in `gate_diagnostics.rs`.
//! `benchmarks` runs real parts through the whole pipeline against known references
//! (`src/bin/benchmarks.rs` runs it). `surfaces` builds and remeshes the surfaces those
//! benchmarks read (`src/bin/surfaces.rs` runs it) and holds the remesh options the corpus uses
//! too. The library crates depend on nothing here.

pub mod benchmarks;
#[cfg(not(target_arch = "wasm32"))]
pub mod corpus;
pub mod gate_diagnostics;
#[cfg(not(target_arch = "wasm32"))]
pub mod sliver;
#[cfg(not(target_arch = "wasm32"))]
pub mod surfaces;
