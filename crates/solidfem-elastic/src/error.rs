//! The errors a study can return. Each names what is wrong in terms a caller can act on, and every
//! refusal happens before a result exists, so a model that cannot be solved never yields numbers,
//! and a mesh comparison or safety factor that would mean nothing is refused rather than computed.

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
    /// A displacement field was supplied that does not have one vector per mesh node.
    DisplacementFieldLength {
        /// Nodes in the mesh.
        nodes: usize,
        /// Vectors supplied.
        values: usize,
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
    /// An error estimate met a vertex whose elements hold more than one material, where stress
    /// jumps and no smoothed stress is fitted; estimates are made for parts of one material.
    MixedMaterialPatch { vertex: u32 },
    /// An error estimate met a vertex whose elements' integration points all lie in one plane, so
    /// no smoothed stress could be fitted: the elements there are flat.
    CoplanarPatch { vertex: u32 },
    /// An error estimate was asked of a solution with no strain energy, which has no scale for a
    /// relative error.
    ZeroStrainEnergy,
    /// An error estimate summed to NaN or infinity: the displacements are not finite numbers.
    NonFiniteEnergy,
    /// An error estimate was given a recovery begun for a mesh with other node, vertex or element
    /// counts than the study's.
    RecoveryMeshMismatch,
    /// A mesh comparison was given a "fine" mesh with no more elements in the region read than
    /// the coarse one, so the comparison would say nothing about refinement there.
    FineMeshNotFiner {
        /// Elements of the coarse mesh with an integration point inside the region.
        coarse_elements: usize,
        /// Elements of the fine mesh with an integration point inside the region.
        fine_elements: usize,
    },
    /// A mesh comparison found no integration point inside the region on one mesh: nothing of
    /// its elements' volume is there to read.
    NothingInsideRegion { mesh: ComparedMesh },
    /// A mesh comparison found a peak stress that is NaN or infinite on one mesh: that solution
    /// is broken, not a result.
    NonFinitePeak { mesh: ComparedMesh, von_mises: f64 },
    /// A mesh comparison found exactly zero stress everywhere in the region on one mesh, which
    /// has no safety factor. A region at rounding-level stress is not caught here: it passes and
    /// reads a huge safety factor.
    ZeroPeak { mesh: ComparedMesh },
    /// A mesh comparison was given a solution whose arrays do not have one entry per node of its
    /// mesh, or whose integration points do not cover its elements.
    SolutionMeshMismatch { mesh: ComparedMesh },
    /// A safety factor was asked for with a yield strength that is not a finite stress above
    /// zero.
    InvalidYieldStrength { yield_strength: f64 },
}

/// Which of a mesh comparison's two solves an error is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparedMesh {
    /// The solve passed as coarse.
    Coarse,
    /// The solve passed as fine.
    Fine,
}

impl fmt::Display for ComparedMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComparedMesh::Coarse => write!(f, "coarse"),
            ComparedMesh::Fine => write!(f, "fine"),
        }
    }
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
            ModelError::DisplacementFieldLength { nodes, values } => write!(
                f,
                "a displacement field needs one vector per node: {values} vectors for {nodes} \
                 nodes"
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
            ModelError::MixedMaterialPatch { vertex } => write!(
                f,
                "the error cannot be estimated: the elements at vertex {vertex} are of more than \
                 one material, and estimates are made only for parts of a single material; solve \
                 the part with one material to see how converged it is"
            ),
            ModelError::CoplanarPatch { vertex } => write!(
                f,
                "the error cannot be estimated: the elements at vertex {vertex} are flat, so no \
                 smoothed stress can be fitted there; remesh the part to remove the flat elements"
            ),
            ModelError::NonFiniteEnergy => write!(
                f,
                "the error cannot be estimated: the solution's energy is not a finite number, so \
                 its displacements are not a usable result; check the model's loads and material"
            ),
            ModelError::RecoveryMeshMismatch => write!(
                f,
                "the error estimate was given a recovery of a different mesh: begin the recovery \
                 from this study"
            ),
            ModelError::ZeroStrainEnergy => write!(
                f,
                "the solution stores no strain energy, so its relative error has no scale: check \
                 that the loads or prescribed displacements deform the part"
            ),
            ModelError::FineMeshNotFiner {
                coarse_elements,
                fine_elements,
            } => write!(
                f,
                "the meshes cannot be compared: in the region read the fine mesh has \
                 {fine_elements} elements and the coarse one {coarse_elements}; mesh the fine \
                 solve with smaller elements there"
            ),
            ModelError::NothingInsideRegion { mesh } => write!(
                f,
                "the meshes cannot be compared: no element of the {mesh} mesh reaches inside the \
                 region read, and a face or line alone cannot be read however fine the mesh; \
                 widen the region so it holds part of the solid"
            ),
            ModelError::NonFinitePeak { mesh, von_mises } => write!(
                f,
                "the meshes cannot be compared: the {mesh} solution's peak stress is {von_mises}, \
                 so that solution is broken; solve the part on that mesh again and check its \
                 loads and material"
            ),
            ModelError::ZeroPeak { mesh } => write!(
                f,
                "the meshes cannot be compared: the {mesh} solution has no stress in the region \
                 read, which has no safety factor; check that the loads stress this region"
            ),
            ModelError::SolutionMeshMismatch { mesh } => write!(
                f,
                "the meshes cannot be compared: the {mesh} solution was not solved on the {mesh} \
                 mesh; pass each solution with the mesh its study ran on"
            ),
            ModelError::InvalidYieldStrength { yield_strength } => write!(
                f,
                "a safety factor needs a finite yield strength above zero, not \
                 {yield_strength}; give the material's yield strength"
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
