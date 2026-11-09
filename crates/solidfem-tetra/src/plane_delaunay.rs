//! Choosing a flat patch's triangulation. Once nothing pierces a patch, the mesh faces inside its
//! boundary are a triangulation of the region, but only by accident of how the patch was recovered:
//! lattice coordinates rounded to `f64` can leave three points nearly on a line, and a triangle on
//! them has an area of rounding size that is exactly positive and still ruins every tetrahedron
//! standing on it. `recover.rs` calls [`make_patch_delaunay`] on each flat patch before locking it.
//!
//! **What the result is.** The patch's triangulation is flipped towards the constrained Delaunay
//! triangulation of the region in the patch's own plane: an edge inside the region is flipped when
//! the far apex lies strictly inside the circumcircle of the triangle on the other side. The test
//! is exact and needs no projection: for points in one plane, the circumsphere of three of them and
//! a vertex off the plane cuts the plane in exactly their circumcircle. Dropping an axis would
//! also be exact, but it shears a tilted plane, so it would test the Delaunay property of the
//! sheared points rather than of the patch.
//!
//! **Tie rule.** An exactly cocircular quadrilateral keeps the diagonal it has: only a strict
//! in-circle answer flips.
//!
//! **Why it ends.** Lift each patch vertex to the paraboloid over the plane. A flip of an edge that
//! fails the test replaces two lifted triangles by two strictly below them, so the volume under the
//! lifted triangulation strictly falls with each flip; a patch has finitely many triangulations, so
//! none can come round again. The test is a strict inequality, so a tie can never feed a cycle.
//!
//! **When a flip is not legal.** The in-plane quadrilateral of a failing edge is always strictly
//! convex, but the tetrahedra around the edge need not admit the flip right now. Such an edge is
//! set aside and retried once another flip in the patch has reshaped things; what remains after a
//! round with no flip keeps the diagonal it has, and is counted in
//! [`crate::RecoveryReport::in_plane_flips_refused`]. The triangulation stays valid either way.
//!
//! **What is guaranteed.** The inner patch edges that are not locally Delaunay are exactly the
//! ones that report counts, no more and no fewer; a report of zero means every flat patch is
//! Delaunay. A refused edge can be the long edge of a triangle a rounding away from a line, so a
//! non-zero count is the one case where such a triangle can still be on the surface.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::error::TetraError;
use crate::flips::{edge_key, face_key, Flipper, Target};
use crate::predicates::strictly_in_circle;

/// A patch's triangulation held as, for each edge `(low, high)`, the opposite corners of the
/// faces on it. A sorted map, because the flip order must not depend on hashing.
type Corners = BTreeMap<(u32, u32), Vec<u32>>;

fn edge(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

fn corners_of(covering: &[[u32; 3]]) -> Corners {
    let mut corners = Corners::new();
    for &[a, b, c] in covering {
        for (u, v, opposite) in [(a, b, c), (a, c, b), (b, c, a)] {
            corners.entry(edge(u, v)).or_default().push(opposite);
        }
    }
    corners
}

/// Replaces corner `old` of edge `(u, v)` by `new`.
fn replace_corner(corners: &mut Corners, u: u32, v: u32, old: u32, new: u32) {
    if let Some(found) = corners.get_mut(&edge(u, v)) {
        if let Some(slot) = found.iter_mut().find(|corner| **corner == old) {
            *slot = new;
        }
    }
}

/// The opposite corners `(b, d)` of the faces `a c b` and `a c d`, when `(a, c)` is inside the
/// patch.
fn inner_corners(corners: &Corners, boundary: &HashSet<u64>, a: u32, c: u32) -> Option<(u32, u32)> {
    if boundary.contains(&edge_key(a, c)) {
        return None;
    }
    match corners.get(&edge(a, c))?.as_slice() {
        [b, d] => Some((*b, *d)),
        _ => None,
    }
}

/// Whether the flip of inner edge `(a, c)` lowers the lifted triangulation.
fn fails_in_circle(flipper: &Flipper, a: u32, c: u32, b: u32, d: u32, off: u32) -> bool {
    let at = |v: u32| &flipper.core.points[v as usize];
    strictly_in_circle(at(a), at(c), at(b), at(off), at(d))
}

/// Flips the inner edges of patch `id`, currently triangulated as `covering`, towards the
/// patch's constrained Delaunay triangulation, returning the faces it ends up with, sorted.
/// `boundary` holds the patch's boundary edges and `off` is any vertex off its plane.
pub(crate) fn make_patch_delaunay(
    flipper: &mut Flipper,
    id: usize,
    boundary: &HashSet<u64>,
    off: u32,
    covering: &[[u32; 3]],
) -> Result<Vec<[u32; 3]>, TetraError> {
    let mut corners = corners_of(covering);
    let mut work: Vec<(u32, u32)> = corners.keys().copied().collect();
    let mut set_aside: Vec<(u32, u32)> = Vec::new();
    let mut flipped_this_round = false;
    loop {
        while let Some((a, c)) = work.pop() {
            let Some((b, d)) = inner_corners(&corners, boundary, a, c) else {
                continue;
            };
            if !fails_in_circle(flipper, a, c, b, d, off) {
                continue;
            }
            if flipper.flip_patch_edge(a, c, b, d, Target::Patch(id as u32))? {
                flipper.counts.in_plane_flips += 1;
                flipped_this_round = true;
                corners.remove(&edge(a, c));
                corners.insert(edge(b, d), vec![a, c]);
                replace_corner(&mut corners, a, b, c, d);
                replace_corner(&mut corners, b, c, a, d);
                replace_corner(&mut corners, c, d, a, b);
                replace_corner(&mut corners, a, d, c, b);
                work.extend_from_slice(&[edge(a, b), edge(b, c), edge(c, d), edge(a, d)]);
            } else {
                set_aside.push((a, c));
            }
        }
        if !flipped_this_round || set_aside.is_empty() {
            break;
        }
        flipped_this_round = false;
        work.append(&mut set_aside);
    }
    set_aside.sort_unstable();
    set_aside.dedup();
    flipper.counts.in_plane_flips_refused += set_aside
        .iter()
        .filter(|&&(a, c)| {
            inner_corners(&corners, boundary, a, c)
                .is_some_and(|(b, d)| fails_in_circle(flipper, a, c, b, d, off))
        })
        .count();
    let faces: BTreeSet<[u32; 3]> = corners
        .iter()
        .flat_map(|(&(u, v), opposite)| opposite.iter().map(move |&w| face_key(u, v, w)))
        .collect();
    Ok(faces.into_iter().collect())
}
