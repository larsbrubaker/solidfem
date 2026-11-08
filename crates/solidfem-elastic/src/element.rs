//! The 10-node element: geometry at a point, the stiffness integral and stress recovery. The
//! element is isoparametric, so it also holds for curved elements a mesher may produce later.
//! Stiffness is built from the closed form of `B^T D B` for isotropic material rather than from
//! explicit matrices, and each pair is written once and mirrored, so the assembled matrix is
//! exactly symmetric.

use crate::error::ElementFault;
use crate::material::{Material, MaterialField};
use crate::quadrature::Quadrature;
use crate::shape::{shape_derivatives, shape_values, NODE_COUNT};

/// Element geometry at one natural point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointGeometry {
    /// Each shape function's gradient in physical coordinates.
    pub gradients: [[f64; 3]; NODE_COUNT],
    /// The Jacobian determinant: six times the local volume ratio.
    pub det_j: f64,
    /// The physical position of the point.
    pub position: [f64; 3],
}

/// Computes the element's geometry at `natural`, refusing a collapsed or inside-out point.
/// `point` identifies the integration point in the error.
pub fn point_geometry(
    nodes: &[[f64; 3]; NODE_COUNT],
    natural: [f64; 3],
    point: usize,
) -> Result<PointGeometry, ElementFault> {
    let derivatives = shape_derivatives(natural);
    let values = shape_values(natural);
    let mut j = [[0.0; 3]; 3];
    let mut position = [0.0; 3];
    for a in 0..NODE_COUNT {
        for i in 0..3 {
            for (k, entry) in j[i].iter_mut().enumerate() {
                *entry += derivatives[a][i] * nodes[a][k];
            }
            position[i] += values[a] * nodes[a][i];
        }
    }
    let cof = [
        [
            j[1][1] * j[2][2] - j[1][2] * j[2][1],
            j[0][2] * j[2][1] - j[0][1] * j[2][2],
            j[0][1] * j[1][2] - j[0][2] * j[1][1],
        ],
        [
            j[1][2] * j[2][0] - j[1][0] * j[2][2],
            j[0][0] * j[2][2] - j[0][2] * j[2][0],
            j[0][2] * j[1][0] - j[0][0] * j[1][2],
        ],
        [
            j[1][0] * j[2][1] - j[1][1] * j[2][0],
            j[0][1] * j[2][0] - j[0][0] * j[2][1],
            j[0][0] * j[1][1] - j[0][1] * j[1][0],
        ],
    ];
    let det_j = j[0][0] * cof[0][0] + j[0][1] * cof[1][0] + j[0][2] * cof[2][0];
    if det_j.is_nan() || det_j <= 0.0 {
        return Err(ElementFault::InvertedJacobian {
            point,
            determinant: det_j,
        });
    }
    let mut gradients = [[0.0; 3]; NODE_COUNT];
    for a in 0..NODE_COUNT {
        for (k, gradient) in gradients[a].iter_mut().enumerate() {
            let sum = cof[k][0] * derivatives[a][0]
                + cof[k][1] * derivatives[a][1]
                + cof[k][2] * derivatives[a][2];
            *gradient = sum / det_j;
        }
    }
    Ok(PointGeometry {
        gradients,
        det_j,
        position,
    })
}

/// The field's material at a point, refused when it is not a stable solid.
pub(crate) fn checked_material(
    field: &impl MaterialField,
    position: [f64; 3],
    point: usize,
) -> Result<Material, ElementFault> {
    let material = field.at(position);
    if material.is_valid() {
        Ok(material)
    } else {
        Err(ElementFault::InvalidMaterial { point, material })
    }
}

/// Writes the 30 x 30 element stiffness into `ke`, row-major, degrees of freedom ordered node by
/// node (x, y, z). The material is evaluated at each integration point.
pub fn element_stiffness(
    nodes: &[[f64; 3]; NODE_COUNT],
    field: &impl MaterialField,
    rule: Quadrature,
    ke: &mut [f64; 900],
) -> Result<(), ElementFault> {
    ke.fill(0.0);
    for (index, q) in rule.points().iter().enumerate() {
        let geometry = point_geometry(nodes, q.natural, index)?;
        let (lambda, mu) = checked_material(field, geometry.position, index)?.lame();
        let measure = q.weight * geometry.det_j / 6.0;
        for a in 0..NODE_COUNT {
            for b in a..NODE_COUNT {
                let ga = geometry.gradients[a];
                let gb = geometry.gradients[b];
                let dot = ga[0] * gb[0] + ga[1] * gb[1] + ga[2] * gb[2];
                for i in 0..3 {
                    for k in 0..3 {
                        // Multiplying the gradients first makes the a == b block's (i, k) and
                        // (k, i) entries the same bits.
                        let mut value = lambda * (ga[i] * gb[k]) + mu * (ga[k] * gb[i]);
                        if i == k {
                            value += mu * dot;
                        }
                        value *= measure;
                        ke[(3 * a + i) * 30 + (3 * b + k)] += value;
                        if a != b {
                            ke[(3 * b + k) * 30 + (3 * a + i)] += value;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Strain `[xx, yy, zz, 2xy, 2yz, 2zx]` (engineering shears) from nodal displacements.
pub fn strain_at(geometry: &PointGeometry, u: &[[f64; 3]; NODE_COUNT]) -> [f64; 6] {
    let mut h = [[0.0; 3]; 3];
    for (ua, ga) in u.iter().zip(&geometry.gradients) {
        for (ui, row) in ua.iter().zip(h.iter_mut()) {
            for (gj, entry) in ga.iter().zip(row.iter_mut()) {
                *entry += ui * gj;
            }
        }
    }
    [
        h[0][0],
        h[1][1],
        h[2][2],
        h[0][1] + h[1][0],
        h[1][2] + h[2][1],
        h[2][0] + h[0][2],
    ]
}

/// Stress `[xx, yy, zz, xy, yz, zx]` from strain in engineering shear form.
pub fn stress_at(material: &Material, strain: &[f64; 6]) -> [f64; 6] {
    let (lambda, mu) = material.lame();
    let trace = strain[0] + strain[1] + strain[2];
    [
        lambda * trace + 2.0 * mu * strain[0],
        lambda * trace + 2.0 * mu * strain[1],
        lambda * trace + 2.0 * mu * strain[2],
        mu * strain[3],
        mu * strain[4],
        mu * strain[5],
    ]
}

/// Von Mises equivalent stress of a `[xx, yy, zz, xy, yz, zx]` tensor.
pub fn von_mises(s: &[f64; 6]) -> f64 {
    let (a, b, c) = (s[0] - s[1], s[1] - s[2], s[2] - s[0]);
    (0.5 * (a * a + b * b + c * c) + 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5])).sqrt()
}
