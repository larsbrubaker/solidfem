//! The existing surface remesher, compiled here from solidmesh's own source files so the corpus
//! measures the production remesher rather than a copy of it.
//!
//! solidmesh workaround: `g3`'s remesher and its `DMesh3` are `pub(crate)` in solidmesh-mesh, with
//! no public entry point for an isotropic remesh, so this module mounts those exact files with
//! `#[path]`. The upstream fix is a public `Mesh::remeshed(options)` in solidmesh-mesh; when it
//! exists this mount goes away. This crate is `publish = false`, and the published solidfem
//! crates carry none of this.
//!
//! The mounted files reach their siblings through `super::` and the host crate through
//! `crate::Mesh`, so the sibling set below mirrors `g3/mod.rs` for the files the remesher needs,
//! and [`Mesh`] is the small type the conversion helpers in `dmesh3.rs` compile against. It is
//! re-exported at this crate's root for exactly that reason; nothing else uses it, the corpus
//! converts through [`remesh`] instead.
#![allow(dead_code, unused_imports, unused_variables, clippy::all)]

use solidmesh_math::Vector3Float;
use solidmesh_mesh::FaceList;

#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/dmesh3.rs"]
pub(crate) mod dmesh3;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/dmesh3_collapse.rs"]
pub(crate) mod dmesh3_collapse;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/dmesh3_edge_ops.rs"]
pub(crate) mod dmesh3_edge_ops;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/dmesh3_queries.rs"]
pub(crate) mod dmesh3_queries;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/dmesh3_set_triangle.rs"]
pub(crate) mod dmesh3_set_triangle;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/dmesh_aabb_tree3.rs"]
pub(crate) mod dmesh_aabb_tree3;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/mesh_constraints.rs"]
pub(crate) mod mesh_constraints;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/mesh_projection_target.rs"]
pub(crate) mod mesh_projection_target;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/mesh_refiner_base.rs"]
pub(crate) mod mesh_refiner_base;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/mesh_util.rs"]
pub(crate) mod mesh_util;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/ray_intersection.rs"]
pub(crate) mod ray_intersection;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/ref_count_vector.rs"]
pub(crate) mod ref_count_vector;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/remesher.rs"]
pub(crate) mod remesher;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/remesher_pro.rs"]
pub(crate) mod remesher_pro;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/small_list_set.rs"]
pub(crate) mod small_list_set;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/tri_distance.rs"]
pub(crate) mod tri_distance;
#[path = "../../../../../solidmesh/crates/polygon-mesh/src/g3/vector3d.rs"]
pub(crate) mod vector3d;

pub mod remesh;

/// The `crate::Mesh` the mounted `dmesh3.rs` compiles against: same field names as solidmesh's.
#[derive(Default)]
pub struct Mesh {
    pub vertices: Vec<Vector3Float>,
    pub faces: FaceList,
}

impl Mesh {
    /// An empty mesh.
    pub fn new() -> Self {
        Self::default()
    }
}
