//! Errors a tet mesh construction can return: each names the offending element or node, so a
//! caller's message can say what to fix.

use std::fmt;

/// Why a mesh could not be built.
#[derive(Debug, Clone, PartialEq)]
pub enum MeshError {
    /// The mesh has no tetrahedra.
    Empty,
    /// A position has a NaN or infinite coordinate.
    NonFinitePosition { node: usize },
    /// A tetrahedron names a node beyond the position array.
    NodeOutOfRange { tet: usize, node: u32 },
    /// A node belongs to no tetrahedron, so it could move freely and the stiffness would be
    /// singular.
    UnusedNode { node: usize },
    /// A tetrahedron has zero or negative volume.
    DegenerateTet { tet: usize, volume: f64 },
    /// A tetrahedron's volume is positive only by rounding: `relative_volume`, its determinant
    /// over its longest edge cubed, is below [`crate::tet::FLAT_RELATIVE_VOLUME`], so its sign
    /// means nothing and the solver would integrate noise.
    FlatTet { tet: usize, relative_volume: f64 },
    /// Three or more tetrahedra share one face, so they overlap.
    NonManifoldFace { nodes: [u32; 3] },
    /// A box dimension or division is not positive.
    InvalidBox { reason: &'static str },
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeshError::Empty => write!(f, "the mesh has no tetrahedra"),
            MeshError::NonFinitePosition { node } => {
                write!(f, "node {node} has a NaN or infinite coordinate")
            }
            MeshError::NodeOutOfRange { tet, node } => {
                write!(
                    f,
                    "tetrahedron {tet} names node {node}, which does not exist"
                )
            }
            MeshError::UnusedNode { node } => {
                write!(f, "node {node} belongs to no tetrahedron")
            }
            MeshError::DegenerateTet { tet, volume } => write!(
                f,
                "tetrahedron {tet} has volume {volume}; every tetrahedron needs positive volume, \
                 with corners ordered so (v1-v0)x(v2-v0)*(v3-v0) > 0"
            ),
            MeshError::FlatTet {
                tet,
                relative_volume,
            } => write!(
                f,
                "tetrahedron {tet} is flat to rounding (relative volume {relative_volume}); its \
                 volume has no reliable sign, so it cannot be analysed"
            ),
            MeshError::NonManifoldFace { nodes } => write!(
                f,
                "the face with nodes {nodes:?} is shared by more than two tetrahedra"
            ),
            MeshError::InvalidBox { reason } => write!(f, "invalid box: {reason}"),
        }
    }
}

impl std::error::Error for MeshError {}
