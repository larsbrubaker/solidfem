//! From the tetrahedral mesher to the solver: [`quadratic_from_solid`] upgrades a finished
//! [`SolidMesh`](crate::tetra::SolidMesh) to the validated 10-node mesh `solidfem-elastic` runs on.
//!
//! The mesher decides positivity by exact predicates, the solver by the floating-point Jacobian
//! it integrates with, and those can disagree on a sliver near the rounding limit. So the upgrade
//! checks the determinant at every integration point of the requested rule, the same computation
//! the solver will do, and a tetrahedron that fails it is a named error here, before assembly.
//! The midside nodes are the edge midpoints, so the elements have straight edges and each
//! element's Jacobian is constant; the check is still the solver's own.

use std::fmt;

use crate::elastic::{point_geometry, ElementFault, Quadrature, NODE_COUNT};
use crate::mesh::{MeshError, QuadraticTetMesh, TetMesh};
use crate::tetra::SolidMesh;

/// Why a finished solid could not become a 10-node mesh.
#[derive(Debug, Clone, PartialEq)]
pub enum MeshingError {
    /// The linear mesh's own validation refused it, naming the tetrahedron or node.
    Mesh(MeshError),
    /// Tetrahedron `tet` has a Jacobian that is not positive at integration point `point`.
    NonPositiveJacobian {
        tet: usize,
        point: usize,
        determinant: f64,
    },
}

impl fmt::Display for MeshingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mesh(error) => write!(f, "the mesh was refused: {error}"),
            Self::NonPositiveJacobian {
                tet,
                point,
                determinant,
            } => write!(
                f,
                "tetrahedron {tet} has Jacobian determinant {determinant} at integration point \
                 {point}, which must be positive; the element is too flat to analyse"
            ),
        }
    }
}

impl std::error::Error for MeshingError {}

/// Upgrades `solid` to 10-node elements with straight edges, checking every element's Jacobian
/// at the integration points of `quadrature`.
///
/// # Errors
/// [`MeshingError::Mesh`] when the linear mesh is refused and
/// [`MeshingError::NonPositiveJacobian`] for an element the solver could not integrate.
pub fn quadratic_from_solid(
    solid: &SolidMesh,
    quadrature: Quadrature,
) -> Result<QuadraticTetMesh, MeshingError> {
    let linear =
        TetMesh::new(solid.points.clone(), solid.tets.clone()).map_err(MeshingError::Mesh)?;
    let mesh = QuadraticTetMesh::from_linear(&linear);
    let points = quadrature.points();
    for (tet, element) in mesh.tets().iter().enumerate() {
        let nodes: [[f64; 3]; NODE_COUNT] = element.map(|n| mesh.nodes()[n as usize]);
        for (point, sample) in points.iter().enumerate() {
            if let Err(ElementFault::InvertedJacobian { point, determinant }) =
                point_geometry(&nodes, sample.natural, point)
            {
                return Err(MeshingError::NonPositiveJacobian {
                    tet,
                    point,
                    determinant,
                });
            }
        }
    }
    Ok(mesh)
}
