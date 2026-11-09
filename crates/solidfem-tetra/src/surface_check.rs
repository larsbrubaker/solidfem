//! What makes a list of triangles a surface [`crate::SurfaceMesher`] can work on. `surface.rs`
//! calls this before anything else, so a surface that is not closed, not consistently wound or
//! has a flat triangle gets a named error instead of a mesh.

use crate::error::TetraError;

/// Refuses anything not a closed, manifold, consistently wound surface using every vertex and
/// with no flat triangle, and returns its undirected edges in a fixed order.
pub(crate) fn check_surface(
    points: &[[f64; 3]],
    triangles: &[[u32; 3]],
) -> Result<Vec<(u32, u32)>, TetraError> {
    let mut directed: Vec<(u32, u32)> = Vec::with_capacity(triangles.len() * 3);
    let mut used = vec![false; points.len()];
    for (index, triangle) in triangles.iter().enumerate() {
        for &vertex in triangle {
            if vertex as usize >= points.len() {
                return Err(TetraError::SurfaceIndexOutOfRange {
                    triangle: index,
                    index: vertex,
                });
            }
            used[vertex as usize] = true;
        }
        let [a, b, c] = *triangle;
        if a == b || b == c || a == c || collinear(points, *triangle) {
            return Err(TetraError::DegenerateTriangle { triangle: index });
        }
        directed.extend_from_slice(&[(a, b), (b, c), (c, a)]);
    }
    if let Some(index) = used.iter().position(|&u| !u) {
        // `index` fits u32: `check_surface` runs before `Triangulator::new` refuses larger inputs,
        // and a set that large fails that test next, so a clamp here is only cosmetic.
        return Err(TetraError::UnusedVertex {
            index: u32::try_from(index).unwrap_or(u32::MAX),
        });
    }
    directed.sort_unstable();
    for pair in directed.windows(2) {
        if pair[0] == pair[1] {
            return Err(TetraError::SurfaceNotClosed {
                edge: [pair[0].0, pair[0].1],
            });
        }
    }
    let mut edges = Vec::with_capacity(directed.len() / 2);
    for &(a, b) in &directed {
        if directed.binary_search(&(b, a)).is_err() {
            return Err(TetraError::SurfaceNotClosed { edge: [a, b] });
        }
        if a < b {
            edges.push((a, b));
        }
    }
    Ok(edges)
}

/// Whether a triangle's corners are exactly on one line: all three 2D projections vanish.
fn collinear(points: &[[f64; 3]], triangle: [u32; 3]) -> bool {
    use robust::{orient2d, Coord};
    let [a, b, c] = triangle.map(|v| points[v as usize]);
    let area = |i: usize, j: usize| {
        let project = |p: &[f64; 3]| Coord { x: p[i], y: p[j] };
        orient2d(project(&a), project(&b), project(&c))
    };
    area(0, 1) == 0.0 && area(1, 2) == 0.0 && area(0, 2) == 0.0
}
