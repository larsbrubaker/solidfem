//! A structured box of tetrahedra for tests and demos. Each hexahedral cell is split into six
//! tets around its body diagonal (lowest corner to highest corner), which meshes conformingly
//! across neighbouring cells because every cell cuts its faces along the same diagonal.

use crate::error::MeshError;
use crate::tet::TetMesh;

/// The six axis orders of the walk from a cell's lowest corner to its highest; each gives one
/// tetrahedron. Odd permutations produce negative volume unless their walk is reversed, handled
/// where the tet is built.
const AXIS_ORDERS: [([usize; 3], bool); 6] = [
    ([0, 1, 2], true),
    ([1, 2, 0], true),
    ([2, 0, 1], true),
    ([0, 2, 1], false),
    ([2, 1, 0], false),
    ([1, 0, 2], false),
];

/// Builds a box with `size` lengths from `origin`, cut into `divisions` cells per axis and six
/// tetrahedra per cell. Node order is x-then-y-then-z ascending.
pub fn box_tet_mesh(
    origin: [f64; 3],
    size: [f64; 3],
    divisions: [usize; 3],
) -> Result<TetMesh, MeshError> {
    if divisions.contains(&0) {
        return Err(MeshError::InvalidBox {
            reason: "each axis needs at least one division",
        });
    }
    // A NaN size is refused as well as a non-positive one.
    if size.iter().any(|l| l.is_nan() || *l <= 0.0) {
        return Err(MeshError::InvalidBox {
            reason: "each box length must be positive",
        });
    }
    let dims = divisions.map(|d| d + 1);
    let node_index = |i: usize, j: usize, k: usize| -> u32 {
        // Fits in u32 unless a box asks for over four billion nodes, which the host could not
        // hold anyway; the cast is checked by the tet's node range in `TetMesh::new`.
        (i + dims[0] * (j + dims[1] * k)) as u32
    };
    let mut positions = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                positions.push([
                    coord(origin[0], size[0], i, divisions[0]),
                    coord(origin[1], size[1], j, divisions[1]),
                    coord(origin[2], size[2], k, divisions[2]),
                ]);
            }
        }
    }
    let mut tets = Vec::with_capacity(6 * divisions[0] * divisions[1] * divisions[2]);
    for k in 0..divisions[2] {
        for j in 0..divisions[1] {
            for i in 0..divisions[0] {
                let start = [i, j, k];
                for (order, even) in AXIS_ORDERS {
                    let mut at = start;
                    let v0 = node_index(at[0], at[1], at[2]);
                    at[order[0]] += 1;
                    let v1 = node_index(at[0], at[1], at[2]);
                    at[order[1]] += 1;
                    let v2 = node_index(at[0], at[1], at[2]);
                    at[order[2]] += 1;
                    let v3 = node_index(at[0], at[1], at[2]);
                    tets.push(if even {
                        [v0, v1, v2, v3]
                    } else {
                        [v0, v2, v1, v3]
                    });
                }
            }
        }
    }
    TetMesh::new(positions, tets)
}

/// Coordinate of grid line `index` of `divisions` along an axis; the far end is exactly
/// `origin + size` so faces can be selected by exact comparison.
fn coord(origin: f64, size: f64, index: usize, divisions: usize) -> f64 {
    if index == divisions {
        origin + size
    } else {
        origin + size * (index as f64 / divisions as f64)
    }
}
