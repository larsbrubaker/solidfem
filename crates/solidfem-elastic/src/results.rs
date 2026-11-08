//! Stress recovery after a solve. Integration point values are the raw results; the per-node
//! display value is a volume-weighted average of von Mises stress evaluated at the node in each
//! incident element. Averaging the scalar rather than the tensor cannot cancel opposing stresses
//! across elements, and evaluating at the node rather than copying the nearest integration point
//! does not understate a peak.

use solidfem_mesh::QuadraticTetMesh;

use crate::element::{checked_material, point_geometry, strain_at, stress_at, von_mises};
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

/// Recovers stresses from nodal displacements, reading the cancel check once per element.
pub(crate) fn recover(
    mesh: &QuadraticTetMesh,
    field: &impl MaterialField,
    rule: Quadrature,
    displacements: &[[f64; 3]],
    cancel: &mut dyn FnMut() -> bool,
) -> Result<(Vec<PointStress>, Vec<f64>), ModelError> {
    let element_error = |tet: usize, fault| ModelError::Element {
        tet: tet as u32,
        fault,
    };
    let points = rule.points();
    let mut point_stresses = Vec::with_capacity(mesh.tets().len() * points.len());
    let mut weighted = vec![0.0; mesh.nodes().len()];
    let mut weights = vec![0.0; mesh.nodes().len()];
    for (tet_index, tet) in mesh.tets().iter().enumerate() {
        if cancel() {
            return Err(ModelError::Cancelled);
        }
        let nodes = tet.map(|n| mesh.nodes()[n as usize]);
        let u = tet.map(|n| displacements[n as usize]);
        let mut volume = 0.0;
        for (index, q) in points.iter().enumerate() {
            let geometry = point_geometry(&nodes, q.natural, index)
                .map_err(|f| element_error(tet_index, f))?;
            volume += q.weight * geometry.det_j / 6.0;
            let material = checked_material(field, geometry.position, index)
                .map_err(|f| element_error(tet_index, f))?;
            let stress = stress_at(&material, &strain_at(&geometry, &u));
            point_stresses.push(PointStress {
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
            weighted[global] += volume * von_mises(&stress);
            weights[global] += volume;
        }
    }
    // Every mesh node belongs to a tetrahedron (`TetMesh::new`), so every weight is positive.
    let nodal_von_mises = weighted.iter().zip(&weights).map(|(s, w)| s / w).collect();
    Ok((point_stresses, nodal_von_mises))
}
