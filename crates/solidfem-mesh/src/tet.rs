//! The linear tetrahedral input mesh: positions and four-node tetrahedra from plain arrays,
//! validated on construction so the solver can trust them. `quadratic.rs` upgrades it to 10-node
//! elements and `boundary.rs` finds its faces.

use crate::boundary::{face_nodes_from_corners, sorted_key};
use crate::error::MeshError;

/// Signed volume of a tetrahedron, positive when `(b-a) x (c-a) * (d-a) > 0`.
pub fn tet_volume(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    let det = u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0]);
    det / 6.0
}

/// A validated mesh of four-node tetrahedra.
#[derive(Debug, Clone, PartialEq)]
pub struct TetMesh {
    positions: Vec<[f64; 3]>,
    tets: Vec<[u32; 4]>,
}

impl TetMesh {
    /// Builds a mesh from plain arrays. Refused: no tetrahedra, non-finite positions, indices out
    /// of range, a tetrahedron without positive volume, an unused node (it would make the
    /// stiffness singular) and a face shared by more than two tetrahedra (overlapping elements).
    pub fn new(positions: Vec<[f64; 3]>, tets: Vec<[u32; 4]>) -> Result<Self, MeshError> {
        if tets.is_empty() {
            return Err(MeshError::Empty);
        }
        if let Some(node) = positions
            .iter()
            .position(|p| p.iter().any(|c| !c.is_finite()))
        {
            return Err(MeshError::NonFinitePosition { node });
        }
        let mut used = vec![false; positions.len()];
        for (tet_index, tet) in tets.iter().enumerate() {
            for &node in tet {
                let Some(flag) = used.get_mut(node as usize) else {
                    return Err(MeshError::NodeOutOfRange {
                        tet: tet_index,
                        node,
                    });
                };
                *flag = true;
            }
            let [a, b, c, d] = tet.map(|n| positions[n as usize]);
            let volume = tet_volume(a, b, c, d);
            // A NaN from extreme coordinates is refused too.
            if volume.is_nan() || volume <= 0.0 {
                return Err(MeshError::DegenerateTet {
                    tet: tet_index,
                    volume,
                });
            }
        }
        if let Some(node) = used.iter().position(|used| !used) {
            return Err(MeshError::UnusedNode { node });
        }
        let mut faces: Vec<[u32; 3]> = tets
            .iter()
            .flat_map(|tet| (0..4).map(|face| sorted_key(face_nodes_from_corners(tet, face))))
            .collect();
        faces.sort_unstable();
        if let Some(window) = faces.windows(3).find(|w| w[0] == w[2]) {
            return Err(MeshError::NonManifoldFace { nodes: window[0] });
        }
        Ok(TetMesh { positions, tets })
    }

    /// Node positions.
    pub fn positions(&self) -> &[[f64; 3]] {
        &self.positions
    }

    /// Tetrahedra as four node indices each, positive volume.
    pub fn tets(&self) -> &[[u32; 4]] {
        &self.tets
    }

    /// The mesh with every position moved through `map` and revalidated, used for distorted
    /// meshes; a map that turns any tetrahedron inside out is refused.
    pub fn with_mapped_positions(
        &self,
        map: impl Fn([f64; 3]) -> [f64; 3],
    ) -> Result<Self, MeshError> {
        let positions = self.positions.iter().map(|p| map(*p)).collect();
        TetMesh::new(positions, self.tets.clone())
    }
}
