//! solidfem-mesh: the tetrahedral meshes structural analysis runs on.
//!
//! [`TetMesh`] is the linear input a mesher or a caller supplies as plain arrays;
//! [`QuadraticTetMesh`] upgrades it to 10-node elements by adding one node per shared edge, the
//! element the solver needs because linear tetrahedra lock and under-report bending.
//! [`box_tet_mesh`] builds structured boxes for tests and demos.
//!
//! Node order inside a 10-node tet (the same order `solidfem-elastic` expects for its shape
//! functions): corners 0, 1, 2, 3 (positive volume), then edge midpoints 4 = 0-1, 5 = 1-2,
//! 6 = 2-0, 7 = 0-3, 8 = 1-3, 9 = 2-3. Every ordering decision is made from explicit sorts on
//! node indices, so the same input builds the same mesh, node for node, on every platform.

mod boundary;
mod box_mesh;
mod error;
mod quadratic;
mod tet;

pub use boundary::{BoundaryFace, FACE_NODES};
pub use box_mesh::box_tet_mesh;
pub use error::MeshError;
pub use quadratic::QuadraticTetMesh;
pub use tet::{tet_volume, TetMesh};
