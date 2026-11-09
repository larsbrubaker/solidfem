//! Recovery after a solve. Integration point values are the raw results; the per-node display
//! value is a volume-weighted average of von Mises stress evaluated at the node in each incident
//! element. Averaging the scalar rather than the tensor cannot cancel opposing stresses across
//! elements, and evaluating at the node rather than copying the nearest integration point does not
//! understate a peak. [`Recovery`] is the resumable form: a host with no threads runs it in bounded
//! slices, in element order, so slicing never changes a value. [`nodal_rotations`] gives the local
//! rotation of a displacement field the same way, for drawing a deformed shape that keeps its
//! rotations rigid.

use solidfem_mesh::QuadraticTetMesh;

use crate::element::{
    checked_material, point_geometry, rotation_at, strain_at, stress_at, von_mises,
};
use crate::error::ModelError;
use crate::material::MaterialField;
use crate::quadrature::Quadrature;
use crate::shape::{NODE_COUNT, NODE_NATURAL};
use crate::study::Report;

/// Stress at one integration point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointStress {
    /// The element.
    pub tet: u32,
    /// The integration point within it.
    pub point: u8,
    /// Physical position.
    pub position: [f64; 3],
    /// `[xx, yy, zz, xy, yz, zx]`.
    pub stress: [f64; 6],
    /// Von Mises equivalent stress.
    pub von_mises: f64,
}

/// A solved study's results.
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// Displacement of every mesh node, zero at fixed nodes.
    pub displacements: Vec<[f64; 3]>,
    /// Stress at every integration point, in (element, point) order.
    pub point_stresses: Vec<PointStress>,
    /// Volume-weighted average von Mises stress per mesh node, for display.
    pub nodal_von_mises: Vec<f64>,
    /// Sizes, iteration counts and wall times.
    pub report: Report,
}

/// A resumable stress recovery over a snapshot of nodal displacements. It runs in element order,
/// so a recovery resumed in slices gives exactly the bits of one run; [`StaticStudy`] drives it
/// because it owns the mesh and material it reads.
///
/// [`StaticStudy`]: crate::study::StaticStudy
#[derive(Debug, Clone)]
pub struct Recovery {
    displacements: Vec<[f64; 3]>,
    point_stresses: Vec<PointStress>,
    weighted: Vec<f64>,
    weights: Vec<f64>,
    next: usize,
}

/// Displacements and the stresses recovered from them: a live estimate while a solve runs, the
/// answer once it has converged.
#[derive(Debug, Clone, PartialEq)]
pub struct RecoveredField {
    /// Displacement of every mesh node, zero at fixed nodes.
    pub displacements: Vec<[f64; 3]>,
    /// Stress at every integration point, in (element, point) order.
    pub point_stresses: Vec<PointStress>,
    /// Volume-weighted average von Mises stress per mesh node, for display.
    pub nodal_von_mises: Vec<f64>,
}

/// The local rotation vector (half the curl of the displacement), in radians, at every mesh node
/// of `displacements`.
///
/// Each element contributes the 10-node shape-function gradients evaluated at the node, and a node
/// shared by several elements takes the average weighted by each element's Jacobian determinant at
/// that node: on this mesh's straight-edged elements that is six times the element's volume, the
/// same volume weighting as the nodal von Mises stress. Gradients are discontinuous between
/// elements, so unlike the displacement itself this is a smoothed value: exact on a rigid
/// rotation, an average where the rotation varies across the node. `displacements` must hold one
/// vector per mesh node.
pub fn nodal_rotations(
    mesh: &QuadraticTetMesh,
    displacements: &[[f64; 3]],
) -> Result<Vec<[f64; 3]>, ModelError> {
    let node_count = mesh.nodes().len();
    if displacements.len() != node_count {
        return Err(ModelError::DisplacementFieldLength {
            nodes: node_count,
            values: displacements.len(),
        });
    }
    let mut weighted = vec![[0.0f64; 3]; node_count];
    let mut weights = vec![0.0f64; node_count];
    for (tet_index, tet) in mesh.tets().iter().enumerate() {
        let nodes = tet.map(|n| mesh.nodes()[n as usize]);
        let u = tet.map(|n| displacements[n as usize]);
        for node in 0..NODE_COUNT {
            let geometry = point_geometry(&nodes, NODE_NATURAL[node], node).map_err(|fault| {
                ModelError::Element {
                    tet: tet_index as u32,
                    fault,
                }
            })?;
            let rotation = rotation_at(&geometry, &u);
            let global = tet[node] as usize;
            for axis in 0..3 {
                weighted[global][axis] += geometry.det_j * rotation[axis];
            }
            weights[global] += geometry.det_j;
        }
    }
    // Every mesh node belongs to a tetrahedron (`TetMesh::new`), so every weight is positive.
    Ok(weighted
        .iter()
        .zip(&weights)
        .map(|(sum, weight)| sum.map(|c| c / weight))
        .collect())
}

impl Recovery {
    /// A recovery of `displacements` over a mesh of `element_count` elements, none done yet.
    pub(crate) fn new(
        displacements: Vec<[f64; 3]>,
        node_count: usize,
        element_count: usize,
        points_per_element: usize,
    ) -> Self {
        Recovery {
            displacements,
            point_stresses: Vec::with_capacity(element_count * points_per_element),
            weighted: vec![0.0; node_count],
            weights: vec![0.0; node_count],
            next: 0,
        }
    }

    /// How many elements have been processed.
    pub fn elements_done(&self) -> usize {
        self.next
    }

    /// The displacements being recovered.
    pub fn displacements(&self) -> &[[f64; 3]] {
        &self.displacements
    }

    /// The finished field. Every mesh node belongs to a tetrahedron (`TetMesh::new`), so once the
    /// last element is done every weight is positive.
    pub(crate) fn into_field(self) -> RecoveredField {
        let nodal_von_mises = self
            .weighted
            .iter()
            .zip(&self.weights)
            .map(|(s, w)| s / w)
            .collect();
        RecoveredField {
            displacements: self.displacements,
            point_stresses: self.point_stresses,
            nodal_von_mises,
        }
    }

    /// Recovers the next `max_elements` elements, reading the cancel check once per element.
    pub(crate) fn advance(
        &mut self,
        mesh: &QuadraticTetMesh,
        field: &impl MaterialField,
        rule: Quadrature,
        max_elements: usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<(), ModelError> {
        let element_error = |tet: usize, fault| ModelError::Element {
            tet: tet as u32,
            fault,
        };
        let points = rule.points();
        let end = self
            .next
            .saturating_add(max_elements)
            .min(mesh.tets().len());
        while self.next < end {
            let tet_index = self.next;
            if cancel() {
                return Err(ModelError::Cancelled);
            }
            let tet = &mesh.tets()[tet_index];
            let nodes = tet.map(|n| mesh.nodes()[n as usize]);
            let u = tet.map(|n| self.displacements[n as usize]);
            let mut volume = 0.0;
            for (index, q) in points.iter().enumerate() {
                let geometry = point_geometry(&nodes, q.natural, index)
                    .map_err(|f| element_error(tet_index, f))?;
                volume += q.weight * geometry.det_j / 6.0;
                let material = checked_material(field, geometry.position, index)
                    .map_err(|f| element_error(tet_index, f))?;
                let stress = stress_at(&material, &strain_at(&geometry, &u));
                self.point_stresses.push(PointStress {
                    tet: tet_index as u32,
                    point: index as u8,
                    position: geometry.position,
                    stress,
                    von_mises: von_mises(&stress),
                });
            }
            for node in 0..NODE_COUNT {
                let geometry = point_geometry(&nodes, NODE_NATURAL[node], node)
                    .map_err(|f| element_error(tet_index, f))?;
                let material = checked_material(field, geometry.position, node)
                    .map_err(|f| element_error(tet_index, f))?;
                let stress = stress_at(&material, &strain_at(&geometry, &u));
                let global = tet[node] as usize;
                self.weighted[global] += volume * von_mises(&stress);
                self.weights[global] += volume;
            }
            self.next += 1;
        }
        Ok(())
    }
}
