//! Refuses a model that is not held firmly enough, before any solving. Each face-connected body
//! needs three non-collinear held nodes; a node shared with an already held body counts as held,
//! so a piece pinned to a fixed piece is found through that shared node. Three non-collinear
//! points remove every rigid motion, while fewer leave a rotation about a line or a point and a
//! singular stiffness that would otherwise surface as garbage.

use std::collections::BTreeMap;
use std::fmt;

use solidfem_mesh::QuadraticTetMesh;

use crate::error::ModelError;

/// The rigid motion still free in an insufficiently held body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freedom {
    /// Nothing holds the body.
    Free,
    /// Rotation about the single held point.
    RotationAboutPoint,
    /// Rotation about the line through the held points.
    RotationAboutLine,
}

impl fmt::Display for Freedom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Freedom::Free => write!(f, "move freely"),
            Freedom::RotationAboutPoint => write!(f, "rotate about the held point"),
            Freedom::RotationAboutLine => write!(f, "rotate about the line through its held nodes"),
        }
    }
}

/// Points closer to a line than this fraction of the body's size count as on the line. Far above
/// coordinate rounding (about 1e-16 of the size) and far below any real feature.
const COLLINEAR_FRACTION: f64 = 1e-9;

/// The motion left free by these held points, or `None` when they hold the body.
fn remaining_freedom(points: &[[f64; 3]], size: f64) -> Option<Freedom> {
    let Some(first) = points.first() else {
        return Some(Freedom::Free);
    };
    let limit = COLLINEAR_FRACTION * size;
    let (mut far, mut far_distance) = (*first, 0.0);
    for p in points {
        let d = distance(*first, *p);
        if d > far_distance {
            (far, far_distance) = (*p, d);
        }
    }
    if far_distance <= limit {
        return Some(Freedom::RotationAboutPoint);
    }
    let axis = [
        (far[0] - first[0]) / far_distance,
        (far[1] - first[1]) / far_distance,
        (far[2] - first[2]) / far_distance,
    ];
    for p in points {
        let v = [p[0] - first[0], p[1] - first[1], p[2] - first[2]];
        let along = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
        let off = [
            v[0] - along * axis[0],
            v[1] - along * axis[1],
            v[2] - along * axis[2],
        ];
        if (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]).sqrt() > limit {
            return None;
        }
    }
    Some(Freedom::RotationAboutLine)
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Checks that every body is held, given which nodes are fixed.
pub(crate) fn check_constraints(mesh: &QuadraticTetMesh, fixed: &[bool]) -> Result<(), ModelError> {
    if !fixed.iter().any(|f| *f) {
        return Err(ModelError::NoFixedNodes);
    }
    let bodies = mesh.face_connected_bodies();
    let body_nodes: Vec<Vec<u32>> = bodies
        .iter()
        .map(|tets| {
            let mut nodes: Vec<u32> = tets.iter().flat_map(|t| mesh.tets()[*t as usize]).collect();
            nodes.sort_unstable();
            nodes.dedup();
            nodes
        })
        .collect();
    let positions = mesh.nodes();
    let sizes: Vec<f64> = body_nodes
        .iter()
        .map(|n| bounding_size(positions, n))
        .collect();

    let mut held = fixed.to_vec();
    let mut anchored = vec![false; bodies.len()];
    loop {
        let mut changed = false;
        for (index, nodes) in body_nodes.iter().enumerate() {
            if anchored[index] {
                continue;
            }
            let points = held_points(positions, nodes, &held);
            if remaining_freedom(&points, sizes[index]).is_none() {
                anchored[index] = true;
                changed = true;
                for node in nodes {
                    held[*node as usize] = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    for (index, nodes) in body_nodes.iter().enumerate() {
        if !anchored[index] {
            let points = held_points(positions, nodes, &held);
            // A body stays unanchored only when `remaining_freedom` reported a motion.
            let freedom = remaining_freedom(&points, sizes[index]).unwrap_or(Freedom::Free);
            return Err(ModelError::BodyNotHeld {
                first_tet: bodies[index][0],
                held_nodes: points.len(),
                freedom,
            });
        }
    }
    Ok(())
}

fn held_points(positions: &[[f64; 3]], nodes: &[u32], held: &[bool]) -> Vec<[f64; 3]> {
    nodes
        .iter()
        .filter(|n| held[**n as usize])
        .map(|n| positions[*n as usize])
        .collect()
}

fn bounding_size(positions: &[[f64; 3]], nodes: &[u32]) -> f64 {
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for n in nodes {
        let p = positions[*n as usize];
        for axis in 0..3 {
            low[axis] = low[axis].min(p[axis]);
            high[axis] = high[axis].max(p[axis]);
        }
    }
    distance(low, high)
}

/// Checks that every body is held when some components are prescribed individually: the fixed
/// nodes' components and the prescribed components together must span all six rigid motions of
/// each body. A pivoted elimination of the rigid-motion values at the held components gives the
/// rank, the same near-collinearity threshold as the node check deciding when a pivot is gone.
pub(crate) fn check_held_dofs(
    mesh: &QuadraticTetMesh,
    fixed: &[bool],
    prescribed: &BTreeMap<(u32, usize), f64>,
) -> Result<(), ModelError> {
    let positions = mesh.nodes();
    for body in mesh.face_connected_bodies() {
        let mut in_body = vec![false; positions.len()];
        let mut nodes = Vec::new();
        for t in &body {
            for n in mesh.tets()[*t as usize] {
                if !in_body[n as usize] {
                    in_body[n as usize] = true;
                    nodes.push(n);
                }
            }
        }
        nodes.sort_unstable();
        let size = bounding_size(positions, &nodes).max(f64::MIN_POSITIVE);
        let mut center = [0.0; 3];
        for n in &nodes {
            for (axis, value) in center.iter_mut().enumerate() {
                *value += positions[*n as usize][axis];
            }
        }
        for value in &mut center {
            *value /= nodes.len() as f64;
        }
        let mut rows: Vec<[f64; 6]> = Vec::new();
        let mut add = |node: u32, axis: usize| {
            let p = positions[node as usize];
            let (x, y, z) = (
                (p[0] - center[0]) / size,
                (p[1] - center[1]) / size,
                (p[2] - center[2]) / size,
            );
            rows.push(match axis {
                0 => [1.0, 0.0, 0.0, 0.0, z, -y],
                1 => [0.0, 1.0, 0.0, -z, 0.0, x],
                _ => [0.0, 0.0, 1.0, y, -x, 0.0],
            });
        };
        for n in &nodes {
            if fixed[*n as usize] {
                for axis in 0..3 {
                    add(*n, axis);
                }
            }
        }
        for (node, axis) in prescribed.keys() {
            if (*node as usize) < positions.len() && in_body[*node as usize] && *axis < 3 {
                add(*node, *axis);
            }
        }
        let rank = pivot_rank(&mut rows);
        if rank < 6 {
            return Err(ModelError::DofsDoNotHold {
                first_tet: body[0],
                rank,
            });
        }
    }
    Ok(())
}

/// The rank of six-column rows by elimination with column pivoting among the rows, a pivot counted
/// when it is above `COLLINEAR_FRACTION`, the node check's notion of a real offset.
fn pivot_rank(rows: &mut [[f64; 6]]) -> usize {
    let mut rank = 0;
    for column in 0..6 {
        let Some(best) = (rank..rows.len()).max_by(|a, b| {
            rows[*a][column]
                .abs()
                .total_cmp(&rows[*b][column].abs())
                .then(b.cmp(a))
        }) else {
            break;
        };
        if rows[best][column].abs() <= COLLINEAR_FRACTION {
            continue;
        }
        rows.swap(rank, best);
        let pivot = rows[rank];
        for row in rows[rank + 1..].iter_mut() {
            let factor = row[column] / pivot[column];
            if factor != 0.0 {
                for k in column..6 {
                    row[k] -= factor * pivot[k];
                }
            }
        }
        rank += 1;
    }
    rank
}
