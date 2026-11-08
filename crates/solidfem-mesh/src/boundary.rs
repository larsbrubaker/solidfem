//! Boundary faces of a tetrahedral mesh: the faces only one tetrahedron has. Tractions are
//! applied over these 6-node triangles. A face's corners run so its outward normal points away
//! from its tetrahedron, computed from the corner order `tet.rs` guarantees (positive volume).

/// Local node indices of each face of a 10-node tet: three corners, then the midpoints of
/// corner edges 0-1, 1-2 and 2-0, listed so the right-hand normal of the corners is outward for
/// a positive-volume element. Face `n` is the face opposite corner `n`.
pub const FACE_NODES: [[usize; 6]; 4] = [
    [1, 2, 3, 5, 9, 8],
    [0, 3, 2, 7, 9, 6],
    [0, 1, 3, 4, 8, 7],
    [0, 2, 1, 6, 5, 4],
];

/// A face on the mesh boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundaryFace {
    /// The tetrahedron the face belongs to.
    pub tet: u32,
    /// The local face index: which corner it is opposite.
    pub face: u8,
    /// Its six mesh nodes, in `FACE_NODES` order.
    pub nodes: [u32; 6],
}

/// The three corner nodes of a linear tet's face `face`.
pub(crate) fn face_nodes_from_corners(tet: &[u32; 4], face: usize) -> [u32; 3] {
    let local = FACE_NODES[face];
    [tet[local[0]], tet[local[1]], tet[local[2]]]
}

/// A face's corners sorted ascending: the key two tetrahedra agree on whatever their local order.
pub(crate) fn sorted_key(mut corners: [u32; 3]) -> [u32; 3] {
    corners.sort_unstable();
    corners
}
