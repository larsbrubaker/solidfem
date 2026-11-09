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

/// Below this ratio of a tetrahedron's determinant to its longest edge cubed, the sign of its
/// volume is rounding and nothing else. The positions themselves carry rounding of `u * M`, with
/// `u` the unit roundoff (1.1e-16) and `M` the largest coordinate magnitude, so the three edge
/// vectors are each `2 u M` uncertain and the determinant `6 u M L^2`; relative to `L^3` that is
/// `6 u (M / L)`, which reaches 1e-12 for an element about 1500 times smaller than its coordinates.
/// A shape below this limit is therefore not resolved by the numbers, and its angles are far under
/// a degree, so no real element is lost. The ratio is scale invariant: scaling a part leaves it.
pub const FLAT_RELATIVE_VOLUME: f64 = 1e-12;

/// The determinant of a tetrahedron over its longest edge cubed: its shape, independent of size.
fn relative_volume(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3], volume: f64) -> f64 {
    let length_squared = |p: [f64; 3], q: [f64; 3]| {
        (p[0] - q[0]) * (p[0] - q[0])
            + (p[1] - q[1]) * (p[1] - q[1])
            + (p[2] - q[2]) * (p[2] - q[2])
    };
    let longest_squared = [
        length_squared(a, b),
        length_squared(a, c),
        length_squared(a, d),
        length_squared(b, c),
        length_squared(b, d),
        length_squared(c, d),
    ]
    .into_iter()
    .fold(0.0_f64, f64::max);
    6.0 * volume / (longest_squared * longest_squared.sqrt())
}

/// A validated mesh of four-node tetrahedra.
#[derive(Debug, Clone, PartialEq)]
pub struct TetMesh {
    positions: Vec<[f64; 3]>,
    tets: Vec<[u32; 4]>,
}

impl TetMesh {
    /// Builds a mesh from plain arrays. Refused: no tetrahedra, non-finite positions, indices out
    /// of range, a tetrahedron without positive volume or flat to rounding (positive only because
    /// of how its determinant rounded, [`FLAT_RELATIVE_VOLUME`]), an unused node (it would make
    /// the stiffness singular) and a face shared by more than two tetrahedra (overlapping elements).
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
            let relative = relative_volume(a, b, c, d, volume);
            if relative.is_nan() || relative < FLAT_RELATIVE_VOLUME {
                return Err(MeshError::FlatTet {
                    tet: tet_index,
                    relative_volume: relative,
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
