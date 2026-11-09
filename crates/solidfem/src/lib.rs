//! solidfem: linear-static structural analysis on tetrahedral meshes, in pure Rust.
//!
//! The facade over the workspace's crates: [`mesh`] holds the tetrahedral meshes and [`solve`]
//! the sparse matrix and the resumable conjugate gradient solver, and [`elastic`] the 10-node
//! element, materials, boundary conditions, static studies and stress results, and [`tetra`] the
//! exact Delaunay tetrahedralization of a point set and the mesher built on it. [`meshing`] joins
//! the mesher to the solver: a refined solid becomes the validated 10-node mesh a study runs on.

pub mod meshing;

pub use solidfem_elastic as elastic;
pub use solidfem_mesh as mesh;
pub use solidfem_solve as solve;
pub use solidfem_tetra as tetra;
