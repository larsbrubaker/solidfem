//! solidfem-tetra: exact incremental Delaunay tetrahedralization, the core of the tetrahedral
//! mesher.
//!
//! [`Triangulator`] takes a set of `[f64; 3]` points and builds the Delaunay tetrahedralization of
//! their convex hull by Bowyer-Watson cavity insertion: walk to a tetrahedron the new point is in
//! the sphere of, carve out the connected region it conflicts with, and star the region's boundary
//! to the point. Run it a slice at a time with [`Triangulator::step`], cancel between slices, and
//! add single points to a finished triangulation with [`Triangulator::insert_point`].
//!
//! [`SurfaceMesher`] takes a closed surface and builds a tetrahedralization of the solid it
//! bounds, its boundary the same surface: the Delaunay tetrahedralization of the surface's
//! vertices, then the surface's real edges and flat regions recovered by flips (`flips.rs`,
//! `recover.rs`) and a point inside the solid where no flip can finish (`steiner.rs`), and every
//! exterior tetrahedron removed by a flood fill. Triangles inside a flat region may change;
//! nothing about the surface's shape does.
//!
//! [`Refiner`] takes that finished solid and adds points strictly inside it: layers through thin
//! walls, points driven by a caller's size function, and a pass that improves the worst
//! tetrahedra, leaving the surface exactly as it was (`refine.rs`, `layers.rs`, `skins.rs`,
//! `interior.rs`, `quality.rs`). Requesting layers also removes every interior edge the rule in
//! `skins.rs` calls through a thin wall, or ends the run in [`TetraError::ThroughEdgeNotSplit`].
//! That rule leans towards "through": opposed surface facings at the two ends, and across them no
//! more than the thin-wall ratio of local surface edge length. It promises nothing at a rim, where
//! surface edges joining the skins stay fixed and a tetrahedron can bridge the wall through one;
//! a wall face too wide for vertex rays is also filled with points through its middle
//! (`wall_fill.rs`). `tests/refine/sweep.rs` measures the rule over shapes whose skins are known
//! from how they were built, and pins how many runs still succeed with an edge across a wall.
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
//! Layout: `refine.rs` is the refinement interface, `refine_mesh.rs` reads a solid in and writes
//! the refined one out, `layers.rs` the through-wall points from
//! surface vertices, `wall_fill.rs` those from the middle of large wall faces, `skins.rs`
//! the opposite skins of a wall and the splitting of the edges joining them, `interior.rs` the
//! insertion, `quality.rs` the improvement, `surface.rs` is the surface-recovery interface and
//! classifies the solid, `patches.rs`
//! finds the flat regions, `recover.rs` orders the recovery of each constraint, `plane_delaunay.rs`
//! chooses each flat region's triangulation, `flips.rs` holds the flips, `steiner.rs` adds points inside the solid, `triangulator.rs` the
//! point-set interface and input checks, `order.rs` the insertion order, `core.rs` the tetrahedra,
//! walk and cavity, `predicates.rs` the exact tests and the tie rule, `error.rs` the failures.

mod core;
mod error;
mod flips;
mod interior;
mod layers;
mod order;
mod patches;
mod plane_delaunay;
mod predicates;
mod quality;
mod recover;
mod refine;
mod refine_mesh;
mod skins;
mod steiner;
mod surface;
mod triangulator;
mod wall_fill;

pub use error::TetraError;
pub use quality::DihedralHistogram;
pub use refine::{
    RefineOptions, RefinePhase, RefineProgress, RefineReport, Refiner, SizeFn, SIZE_RATIO,
};
pub use surface::{RecoveryReport, SolidMesh, SurfaceMesher, SurfacePhase, SurfaceProgress};
pub use triangulator::{StepStatus, Tetrahedralization, Triangulator, HULL};
