//! solidfem-tetra: exact incremental Delaunay tetrahedralization, the core of the tetrahedral
//! mesher.
//!
//! [`Triangulator`] takes a set of `[f64; 3]` points and builds the Delaunay tetrahedralization of
//! their convex hull by Bowyer-Watson cavity insertion: walk to a tetrahedron the new point is in
//! the sphere of, carve out the connected region it conflicts with, and star the region's boundary
//! to the point. Run it a slice at a time with [`Triangulator::step`], cancel between slices, and
//! add single points to a finished triangulation with [`Triangulator::insert_point`].
//!
//! Boundary recovery and refinement build on this crate and are not part of it.
//!
//! The rules that span its files:
//! - **Exact decisions.** Every geometric decision is an exact predicate (`predicates.rs`), so a
//!   degenerate input (a grid, points on a sphere, a thin plate) needs no tolerance, and a
//!   coordinate is never moved: the output's `points` are the input's bits.
//! - **One tie rule.** Cospherical points are resolved by symbolic perturbation of the lifting,
//!   ordered by point index (`predicates.rs`). Which tetrahedra exist depends on the points and
//!   their indices, never on the insertion order or on how a run was sliced.
//! - **No bounding box.** A vertex at infinity (`core.rs`) closes the triangulation, so hull
//!   tetrahedra are exact and nothing artificial has to be removed.
//! - **Duplicates are refused.** Tetrahedra index the caller's points one for one, and welding is
//!   a decision about the caller's surface, so a repeated point is
//!   [`TetraError::DuplicatePoint`] rather than a silent merge.
//!
//! Layout: `triangulator.rs` is the public interface and the input checks, `order.rs` the
//! insertion order, `core.rs` the tetrahedra, walk and cavity, `predicates.rs` the exact tests and
//! the tie rule, `error.rs` the failures.

mod core;
mod error;
mod order;
mod predicates;
mod triangulator;

pub use error::TetraError;
pub use triangulator::{StepStatus, Tetrahedralization, Triangulator, HULL};
