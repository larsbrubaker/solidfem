//! The corpus measurement: what closed triangle meshes look like to a tetrahedral mesher, and
//! whether solidmesh's `remesh_for_analysis`, the remesh a user's part goes through, can prepare
//! them. `soup.rs` holds the triangle soup, the vector helpers and the point-to-triangle distance
//! every other file uses, `bvh.rs` a triangle tree for rays and nearest points, `measure.rs` the
//! topology, quality and wall-thickness readings and which parts are solids to remesh at all,
//! `self_intersect.rs` the penetrating-pair count, `parts.rs` the synthetic parts built through
//! solidmesh's real operations, `record.rs` the one-line key=value records and the tables built
//! from them, `remesh_trial.rs` one remesh attempt with what it did to the part, read by the
//! corpus's own measures rather than the remesher's report, `volume_trial.rs` the volume mesh of
//! a remeshed part with its outcome named by failure class, `solve_trial.rs` a linear static
//! bending solve of that volume mesh checked against its reactions and its size, `diagnose.rs` what blocks surface
//! recovery on a part at several remesh densities. The binary is `src/bin/corpus.rs`.

pub mod bvh;
pub mod diagnose;
pub mod measure;
pub mod parts;
pub mod record;
pub mod remesh_trial;
pub mod self_intersect;
pub mod solve_trial;
pub mod soup;
pub mod volume_trial;
