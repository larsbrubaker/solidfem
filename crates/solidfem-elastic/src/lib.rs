//! solidfem-elastic: linear-static elasticity on 10-node quadratic tetrahedra.
//!
//! The quadratic element is the point of this crate: linear tetrahedra lock in bending and
//! under-report it, and a wrong "this part is fine" is the worst failure the library can have.
//!
//! A [`Model`] holds a mesh, a [`MaterialField`] evaluated at every integration point, fixed
//! nodes, nodal forces and face tractions. [`StaticStudy::new`] validates it (refusing a model
//! that cannot be held or has no effective load, before any solving), eliminates the fixed nodes
//! and assembles; [`StaticStudy::step`] then runs the solve in bounded slices with a cancel check,
//! and [`StaticStudy::solution`] recovers displacements and von Mises stress.
//!
//! Exactness: every loop runs in node or element order on one thread and the solver's iterations
//! depend only on its input, so results are bit-identical across runs and however a solve is
//! sliced. Wall time comes from a caller-supplied clock, because `std::time::Instant` panics on
//! `wasm32-unknown-unknown`.

mod assembly;
mod constraints;
mod element;
mod error;
mod loads;
mod material;
mod quadrature;
mod results;
mod shape;
mod study;

pub use assembly::assemble_stiffness;
pub use constraints::Freedom;
pub use element::{
    element_stiffness, point_geometry, strain_at, stress_at, von_mises, PointGeometry,
};
pub use error::{ElementFault, ModelError};
pub use material::{Material, MaterialField};
pub use quadrature::{Quadrature, QuadraturePoint, TRIANGLE_POINTS};
pub use results::{PointStress, Solution};
pub use shape::{shape_derivatives, shape_values, NODE_COUNT, NODE_NATURAL};
pub use solidfem_solve::{PcgOptions, Progress, StepOutcome};
pub use study::{Clock, Model, Report, StaticStudy, StudyOptions};
