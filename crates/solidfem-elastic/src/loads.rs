//! The load vector: nodal forces plus consistent nodal loads for tractions over 6-node boundary
//! faces, integrated with the triangle rule so a curved face is handled too. A consistent load
//! reproduces the resultant force and moment of the traction exactly on a straight face.

use solidfem_mesh::{BoundaryFace, QuadraticTetMesh};

use crate::quadrature::TRIANGLE_POINTS;

/// Adds a uniform `traction` over `face` to the full `3 * node_count` load vector.
pub(crate) fn add_traction(
    mesh: &QuadraticTetMesh,
    face: &BoundaryFace,
    traction: [f64; 3],
    load: &mut [f64],
) {
    let positions = face.nodes.map(|n| mesh.nodes()[n as usize]);
    for ([r, s], weight) in TRIANGLE_POINTS {
        let (values, dr, ds) = triangle_shape(r, s);
        let mut tangent_r = [0.0; 3];
        let mut tangent_s = [0.0; 3];
        for (node, p) in positions.iter().enumerate() {
            for axis in 0..3 {
                tangent_r[axis] += dr[node] * p[axis];
                tangent_s[axis] += ds[node] * p[axis];
            }
        }
        let normal = [
            tangent_r[1] * tangent_s[2] - tangent_r[2] * tangent_s[1],
            tangent_r[2] * tangent_s[0] - tangent_r[0] * tangent_s[2],
            tangent_r[0] * tangent_s[1] - tangent_r[1] * tangent_s[0],
        ];
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        // The reference triangle has area one half and the rule's weights sum to one.
        let measure = 0.5 * length * weight;
        for (node, global) in face.nodes.iter().enumerate() {
            for axis in 0..3 {
                load[3 * *global as usize + axis] += traction[axis] * values[node] * measure;
            }
        }
    }
}

/// The 6-node triangle's shape values and (r, s) derivatives, nodes in `FACE_NODES` order.
fn triangle_shape(r: f64, s: f64) -> ([f64; 6], [f64; 6], [f64; 6]) {
    let l1 = 1.0 - r - s;
    let values = [
        l1 * (2.0 * l1 - 1.0),
        r * (2.0 * r - 1.0),
        s * (2.0 * s - 1.0),
        4.0 * l1 * r,
        4.0 * r * s,
        4.0 * s * l1,
    ];
    let dr = [
        -(4.0 * l1 - 1.0),
        4.0 * r - 1.0,
        0.0,
        4.0 * (l1 - r),
        4.0 * s,
        -4.0 * s,
    ];
    let ds = [
        -(4.0 * l1 - 1.0),
        0.0,
        4.0 * s - 1.0,
        -4.0 * r,
        4.0 * r,
        4.0 * (l1 - s),
    ];
    (values, dr, ds)
}
