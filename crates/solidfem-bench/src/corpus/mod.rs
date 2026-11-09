//! The corpus measurement: what closed triangle meshes look like to a tetrahedral mesher, and
//! whether the surface remesher can prepare them. `soup.rs` holds the triangle soup and vector
//! helpers every other file uses, `bvh.rs` a triangle tree for rays and nearest points,
//! `measure.rs` the topology, quality and wall-thickness readings, `self_intersect.rs` the
//! penetrating-pair count, `parts.rs` the synthetic parts built through solidmesh's real
//! operations, `g3_mount.rs` the existing surface remesher (compiled from solidmesh's source),
//! `record.rs` the one-line key=value records and the tables built from them,
//! `remesh_trial.rs` one remesh attempt with what it did to the part, and `volume_trial.rs` the
//! volume mesh of a remeshed part with its outcome named by failure class. The binary is
//! `src/bin/corpus.rs`.

pub mod bvh;
pub mod g3_mount;
pub mod measure;
pub mod parts;
pub mod record;
pub mod remesh_trial;
pub mod self_intersect;
pub mod soup;
pub mod volume_trial;
