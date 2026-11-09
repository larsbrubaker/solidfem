//! The errors a study can return. Each names what is wrong in terms a caller can act on, and every
//! refusal happens before a result exists, so a model that cannot be solved never yields numbers.

use std::fmt;

use solidfem_solve::{JacobiError, MultigridError, PcgError};

use crate::constraints::Freedom;
use crate::material::Material;

/// What is wrong at one integration point of one element.
#[derive(Debug, Clone, PartialEq)]
pub enum ElementFault {
    /// The element is inside out or collapsed at this point.
    InvertedJacobian { point: usize, determinant: f64 },
    /// The material field returned properties no solid can have.
    InvalidMaterial { point: usize, material: Material },
}

/// Why a study could not run or has no result.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    /// No node is fixed, so the whole part is free to fly away.
    NoFixedNodes,
    /// One piece of the part is not held firmly enough.
    BodyNotHeld {
        /// The lowest tetrahedron index in the piece.
        first_tet: u32,
        /// How many of its nodes are held.
        held_nodes: usize,
        /// The motion still free.
        freedom: Freedom,
    },
    /// A fixed node or a force names a node that does not exist.
    NodeOutOfRange { node: u32 },
    /// A force or traction has a NaN or infinite component.
    NonFiniteLoad,
    /// A traction names a face that is not on the mesh boundary.
    FaceNotOnBoundary { tet: u32, face: u8 },
    /// No load reaches a free node, so there is nothing to solve and an all-zero answer would
    /// wrongly read as a part under no stress.
    NoEffectiveLoad,
    /// An element could not be integrated.
    Element { tet: u32, fault: ElementFault },
    /// The stiffness diagonal could not be inverted.
    Preconditioner(JacobiError),
    /// The multigrid preconditioner could not be built.
    Multigrid(MultigridError),
    /// The rigid-body modes were asked for with node and fixed-flag lists of different lengths.
    RigidBodyData {
        /// Nodes supplied.
        nodes: usize,
        /// Fixed flags supplied.
        flags: usize,
    },
    /// The solver failed; no result is reported.
    Solver(PcgError),
    /// The cancel check stopped assembly or recovery.
    Cancelled,
    /// A result was asked for before the solve converged.
    NotFinished,
    /// A prescribed displacement names an axis other than 0, 1 or 2.
    AxisOutOfRange { axis: usize },
    /// A prescribed displacement names a node that is also fixed outright.
    PrescribedOnFixedNode { node: u32 },
    /// With prescribed components, a body is not held against all six rigid motions.
    DofsDoNotHold {
        /// The lowest tetrahedron index in the piece.
        first_tet: u32,
        /// The rigid motions the held components do remove, out of six.
        rank: usize,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::NoFixedNodes => write!(
                f,
                "nothing holds the part: fix at least three nodes that are not in a line"
            ),
            ModelError::BodyNotHeld {
                first_tet,
                held_nodes,
                freedom,
            } => write!(
                f,
                "the piece containing tetrahedron {first_tet} has {held_nodes} held node(s) and \
                 can still {freedom}: fix at least three nodes of it that are not in a line"
            ),
            ModelError::NodeOutOfRange { node } => write!(f, "node {node} does not exist"),
            ModelError::NonFiniteLoad => write!(f, "a load has a NaN or infinite component"),
            ModelError::FaceNotOnBoundary { tet, face } => {
                write!(f, "face {face} of tetrahedron {tet} is not on the boundary")
            }
            ModelError::NoEffectiveLoad => write!(
                f,
                "no load reaches a free node: add a force or pressure away from the fixed nodes"
            ),
            ModelError::Element { tet, fault } => write!(f, "tetrahedron {tet}: {fault:?}"),
            ModelError::Preconditioner(error) => write!(f, "{error}"),
            ModelError::Multigrid(error) => write!(f, "{error}"),
            ModelError::RigidBodyData { nodes, flags } => write!(
                f,
                "the rigid-body modes need one fixed flag per node: {flags} flags for {nodes} nodes"
            ),
            ModelError::Solver(error) => write!(f, "{error}"),
            ModelError::Cancelled => write!(f, "the study was cancelled"),
            ModelError::NotFinished => write!(f, "the solve has not converged yet"),
            ModelError::AxisOutOfRange { axis } => {
                write!(
                    f,
                    "a prescribed displacement names axis {axis}: use 0, 1 or 2"
                )
            }
            ModelError::PrescribedOnFixedNode { node } => write!(
                f,
                "node {node} is both fixed and given a prescribed displacement: use one"
            ),
            ModelError::DofsDoNotHold { first_tet, rank } => write!(
                f,
                "the piece containing tetrahedron {first_tet} is held against only {rank} of its \
                 six rigid motions: prescribe or fix more components so it cannot move freely"
            ),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<PcgError> for ModelError {
    fn from(error: PcgError) -> Self {
        ModelError::Solver(error)
    }
}

impl From<MultigridError> for ModelError {
    fn from(error: MultigridError) -> Self {
        match error {
            MultigridError::Cancelled => ModelError::Cancelled,
            other => ModelError::Multigrid(other),
        }
    }
}

impl From<JacobiError> for ModelError {
    fn from(error: JacobiError) -> Self {
        ModelError::Preconditioner(error)
    }
}
