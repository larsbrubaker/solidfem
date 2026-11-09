//! solidfem-bench: measurements of what real inputs do to the mesher and the solver, native only,
//! never published. `corpus` measures real triangle meshes and the solidmesh surface remesher
//! (`src/bin/corpus.rs` runs it); `sliver` measures what bad boundary triangles cost a stress
//! result (`src/bin/sliver.rs` runs it). The end-of-mesher gate, this workspace's mesher into the
//! solver on a thin plate, is `src/bin/gate.rs`. The library crates depend on nothing here.

#[cfg(not(target_arch = "wasm32"))]
pub mod corpus;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use corpus::g3_mount::Mesh;
#[cfg(not(target_arch = "wasm32"))]
pub mod sliver;
