//! Reading a finished [`SolidMesh`] into refinement and back out, and the per-tetrahedron geometry
//! the size stage asks for. `refine.rs` holds the stages; this file is what they read and write.

use crate::core::{DEAD, FACE, GHOST};
use crate::error::TetraError;
use crate::flips::Flipper;
use crate::quality::{cross, dot, subtract};
use crate::refine::SizeFn;
use crate::surface::SolidMesh;
use crate::triangulator::HULL;

pub(crate) fn checked_size(size: &SizeFn, position: [f64; 3]) -> Result<f64, TetraError> {
    let value = size(position);
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(TetraError::InvalidSizeFunction { position, value })
    }
}

pub(crate) fn centroid_of(points: &[[f64; 3]], tet: &[u32; 4]) -> [f64; 3] {
    let mut sum = [0.0; 3];
    for &v in tet {
        let p = points[v as usize];
        for k in 0..3 {
            sum[k] += p[k];
        }
    }
    [sum[0] / 4.0, sum[1] / 4.0, sum[2] / 4.0]
}

/// The tetrahedron's circumcentre, or `None` when it is too flat for the formula to be finite.
pub(crate) fn circumcentre_of(points: &[[f64; 3]], tet: &[u32; 4]) -> Option<[f64; 3]> {
    let [a, b, c, d] = tet.map(|v| points[v as usize]);
    let (u, v, w) = (subtract(b, a), subtract(c, a), subtract(d, a));
    let denominator = 2.0 * dot(u, cross(v, w));
    if denominator == 0.0 {
        return None;
    }
    let (lu, lv, lw) = (dot(u, u), dot(v, v), dot(w, w));
    let (vw, wu, uv) = (cross(v, w), cross(w, u), cross(u, v));
    Some([
        a[0] + (lu * vw[0] + lv * wu[0] + lw * uv[0]) / denominator,
        a[1] + (lu * vw[1] + lv * wu[1] + lw * uv[1]) / denominator,
        a[2] + (lu * vw[2] + lv * wu[2] + lw * uv[2]) / denominator,
    ])
}

/// Each boundary face of the solid, wound counterclockwise from outside.
pub(crate) fn boundary_faces(solid: &SolidMesh) -> Vec<[u32; 3]> {
    let mut faces = Vec::new();
    for (t, tet) in solid.tets.iter().enumerate() {
        // The output tet is right-handed; the internal tet swaps its first two vertices, so
        // FACE lists faces with the opposite vertex on their positive (inner) side.
        let internal = [tet[1], tet[0], tet[2], tet[3]];
        for (slot, neighbor) in solid.neighbors[t].iter().enumerate() {
            if *neighbor != HULL {
                continue;
            }
            let i = [1, 0, 2, 3][slot];
            let [p, q, r] = FACE[i].map(|position| internal[position]);
            // `orient` is positive when the fourth point is below the plane: the opposite
            // vertex is on that side, so `p q r` is counterclockwise from outside already.
            faces.push([p, q, r]);
        }
    }
    faces
}

/// The current mesh, real tetrahedra only, in the public orientation and order.
pub(crate) fn extract(flipper: &Flipper) -> SolidMesh {
    let core = &flipper.core;
    let is_real = |slot: usize| core.tets[slot].iter().all(|&v| v != GHOST && v != DEAD);
    let mut output_index = vec![HULL; core.tets.len()];
    let mut count = 0u32;
    for (slot, index) in output_index.iter_mut().enumerate() {
        if is_real(slot) {
            *index = count;
            count += 1;
        }
    }
    let mut tets = Vec::with_capacity(count as usize);
    let mut neighbors = Vec::with_capacity(count as usize);
    for slot in 0..core.tets.len() {
        if !is_real(slot) {
            continue;
        }
        let tet = core.tets[slot];
        let adjacent = core.neighbors[slot];
        tets.push([tet[1], tet[0], tet[2], tet[3]]);
        neighbors.push([
            output_index[adjacent[1] as usize],
            output_index[adjacent[0] as usize],
            output_index[adjacent[2] as usize],
            output_index[adjacent[3] as usize],
        ]);
    }
    SolidMesh {
        points: core.points.clone(),
        tets,
        neighbors,
    }
}
