//! solidfem: linear-static structural analysis on tetrahedral meshes, in pure Rust.
//!
//! The facade over the workspace's crates: [`mesh`] holds the tetrahedral meshes and [`solve`]
//! the sparse matrix and the resumable conjugate gradient solver.

pub use solidfem_mesh as mesh;
pub use solidfem_solve as solve;
