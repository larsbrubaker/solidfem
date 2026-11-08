//! The exact geometric tests the triangulation is decided by: orientation and in-sphere from the
//! `robust` crate's adaptive exact arithmetic, and the one tie rule. `core.rs` calls these and no
//! floating-point test of its own, so every insertion decision is exact.
//!
//! **Tie rule (symbolic perturbation of the lifting).** A Delaunay triangulation is the lower
//! convex hull of the points lifted to the paraboloid `w = |p|^2`. Five exactly cospherical points
//! leave the choice open, so each point's lifted height is imagined raised by a distinct
//! infinitesimal `eps_i`, with `eps_i >> eps_j` when point `i` has the lower index. The five-point
//! in-sphere determinant is linear in the `w` column, so its sign is the sign of the first
//! non-zero cofactor taken in priority order (lowest index first), and each cofactor is an exact
//! orientation of the other four points. It is computed only on an exact tie, moves no coordinate,
//! and is one total consistent rule, so the result does not depend on the insertion order.

use robust::{insphere, orient3d, Coord3D};

/// A point in the predicate crate's type.
fn coord(point: &[f64; 3]) -> Coord3D<f64> {
    Coord3D {
        x: point[0],
        y: point[1],
        z: point[2],
    }
}

/// Exact orientation of four points: positive when `d` lies below the plane of `a b c`, that is,
/// when `a b c` appear counterclockwise seen from above; zero when coplanar.
pub(crate) fn orient(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3], d: &[f64; 3]) -> f64 {
    orient3d(coord(a), coord(b), coord(c), coord(d))
}

/// Whether point `e` is strictly inside the circumsphere of `tet`, whose four vertices must have
/// positive [`orient`], under the tie rule in this file's header.
pub(crate) fn in_sphere(points: &[[f64; 3]], tet: [u32; 4], e: u32) -> bool {
    let [a, b, c, d] = tet.map(|v| &points[v as usize]);
    let p = &points[e as usize];
    let exact = insphere(coord(a), coord(b), coord(c), coord(d), coord(p));
    if exact != 0.0 {
        return exact > 0.0;
    }
    perturbed_in_sphere(points, [tet[0], tet[1], tet[2], tet[3], e])
}

/// The in-sphere answer once the exact determinant is zero: the first non-zero cofactor, taken
/// from the lowest point index up. Each cofactor is [`orient`] of the other four in their order,
/// with the alternating sign from expanding the five-by-five determinant along its lifted column.
fn perturbed_in_sphere(points: &[[f64; 3]], five: [u32; 5]) -> bool {
    let mut by_priority = [0usize, 1, 2, 3, 4];
    by_priority.sort_unstable_by_key(|&position| five[position]);
    for position in by_priority {
        let mut others = [0usize; 4];
        let mut count = 0;
        for other in 0..5 {
            if other != position {
                others[count] = other;
                count += 1;
            }
        }
        let [a, b, c, d] = others.map(|other| &points[five[other] as usize]);
        let cofactor = orient(a, b, c, d);
        if cofactor != 0.0 {
            // Row `position` (counted from zero) has cofactor sign (-1)^(position + 1 + 4)
            // along the lifted column: negative at even positions.
            return if position % 2 == 0 {
                cofactor < 0.0
            } else {
                cofactor > 0.0
            };
        }
    }
    // Unreachable for a tetrahedron: the last cofactor is the positive orientation of its own
    // four vertices, which never vanishes.
    false
}
