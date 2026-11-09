//! The six rigid-body modes of a mesh, the near-null space the multigrid preconditioner must
//! keep on its coarse levels. A mode has a value at every free unknown of every node, corner and
//! midside alike, because a mode missing at a midside node is simply not a rigid-body motion of
//! the 10-node element. Fixed nodes have no unknowns, so they have no row here.
//!
//! Rotations are taken about the centroid of the free nodes and scaled by the largest distance
//! from it, so the rotation columns are the same size as the translations and the
//! orthonormalization on an aggregate keeps its digits.

use crate::assembly::FIXED;
use crate::error::ModelError;

/// The six rigid-body modes at the free unknowns: three translations then rotations about x, y
/// and z, row-major with six values per unknown (`3 * free_nodes * 6` values). Refuses a
/// `is_fixed` list that does not have one flag per node.
pub fn rigid_body_modes(nodes: &[[f64; 3]], is_fixed: &[bool]) -> Result<Vec<f64>, ModelError> {
    if nodes.len() != is_fixed.len() {
        return Err(ModelError::RigidBodyData {
            nodes: nodes.len(),
            flags: is_fixed.len(),
        });
    }
    let mut free = vec![FIXED; nodes.len()];
    let mut free_count = 0usize;
    for (node, fixed) in is_fixed.iter().enumerate() {
        if !fixed {
            // Fits in u32: a mesh addresses nodes with u32.
            free[node] = free_count as u32;
            free_count += 1;
        }
    }
    Ok(free_rigid_body_modes(nodes, &free, free_count))
}

/// The modes for an already-numbered free set: `free[node]` is the free index or [`FIXED`].
pub(crate) fn free_rigid_body_modes(
    nodes: &[[f64; 3]],
    free: &[u32],
    free_count: usize,
) -> Vec<f64> {
    let mut center = [0.0; 3];
    for (node, index) in free.iter().enumerate() {
        if *index != FIXED {
            for axis in 0..3 {
                center[axis] += nodes[node][axis];
            }
        }
    }
    // A mesh with no free node has no rows, and the division below would only produce unused
    // values, so the zero count needs no special case beyond avoiding a zero divisor.
    let divisor = free_count.max(1) as f64;
    for value in &mut center {
        *value /= divisor;
    }
    let mut reach = 0.0_f64;
    for (node, index) in free.iter().enumerate() {
        if *index != FIXED {
            for axis in 0..3 {
                reach = reach.max((nodes[node][axis] - center[axis]).abs());
            }
        }
    }
    let scale = if reach > 0.0 { 1.0 / reach } else { 1.0 };
    let mut modes = vec![0.0; 3 * free_count * 6];
    for (node, index) in free.iter().enumerate() {
        if *index == FIXED {
            continue;
        }
        let at = 3 * *index as usize * 6;
        let x = (nodes[node][0] - center[0]) * scale;
        let y = (nodes[node][1] - center[1]) * scale;
        let z = (nodes[node][2] - center[2]) * scale;
        let rows: [[f64; 6]; 3] = [
            [1.0, 0.0, 0.0, 0.0, z, -y],
            [0.0, 1.0, 0.0, -z, 0.0, x],
            [0.0, 0.0, 1.0, y, -x, 0.0],
        ];
        for (r, row) in rows.iter().enumerate() {
            modes[at + r * 6..at + (r + 1) * 6].copy_from_slice(row);
        }
    }
    modes
}
