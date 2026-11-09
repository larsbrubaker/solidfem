//! The 10-node tetrahedron's quadratic shape functions on the reference element, in the node
//! order `solidfem-mesh` documents: corners 0-3, then midpoints of edges 0-1, 1-2, 2-0, 0-3, 1-3
//! and 2-3. Natural coordinates (r, s, t) are the barycentrics of corners 1, 2 and 3.

/// Nodes per element.
pub const NODE_COUNT: usize = 10;

/// The natural coordinates of each element node, used to evaluate fields at the nodes.
pub const NODE_NATURAL: [[f64; 3]; NODE_COUNT] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.5, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
];

/// Each midpoint node's two corners, in node order.
pub(crate) const EDGES: [(usize, usize); 6] = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)];

fn barycentrics(natural: [f64; 3]) -> [f64; 4] {
    [
        1.0 - natural[0] - natural[1] - natural[2],
        natural[0],
        natural[1],
        natural[2],
    ]
}

/// The ten shape function values at a natural point.
pub fn shape_values(natural: [f64; 3]) -> [f64; NODE_COUNT] {
    let l = barycentrics(natural);
    let mut n = [0.0; NODE_COUNT];
    for i in 0..4 {
        n[i] = l[i] * (2.0 * l[i] - 1.0);
    }
    for (slot, (a, b)) in EDGES.iter().enumerate() {
        n[4 + slot] = 4.0 * l[*a] * l[*b];
    }
    n
}

/// Each shape function's derivatives with respect to (r, s, t).
pub fn shape_derivatives(natural: [f64; 3]) -> [[f64; 3]; NODE_COUNT] {
    let l = barycentrics(natural);
    // dL_i / d(r, s, t).
    let d_l: [[f64; 3]; 4] = [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let mut d = [[0.0; 3]; NODE_COUNT];
    for axis in 0..3 {
        for i in 0..4 {
            d[i][axis] = (4.0 * l[i] - 1.0) * d_l[i][axis];
        }
        for (slot, (a, b)) in EDGES.iter().enumerate() {
            d[4 + slot][axis] = 4.0 * (d_l[*a][axis] * l[*b] + l[*a] * d_l[*b][axis]);
        }
    }
    d
}
