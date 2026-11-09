//! Counts triangle pairs whose interiors penetrate each other, the defect that makes a surface
//! unusable as a tetrahedral mesher's boundary. A pair counts only when it overlaps by more than
//! a small tolerance on every separating axis, so triangles that merely touch along a shared edge
//! or at a T-junction do not. Pairs sharing a vertex are skipped, so the count is a lower bound.
//!
//! That skip makes the pair count blind to a fold: two triangles sharing an edge and lying back
//! on each other, exactly coplanar with their opposite corners on the same side of the edge. A
//! tetrahedral mesher cannot separate inside from outside there, so [`count_folds`] counts them
//! apart, with exact orientation predicates: a fold is a fact about the coordinates, never a
//! tolerance. The predicates are exact while their products stay out of the subnormal range,
//! which holds for coordinates read from `f32` and for a remesh of them.

use robust::{orient2d, orient3d, Coord, Coord3D};

use super::bvh::Bvh;
use super::soup::{cross, dot, length, scale, sub, TriSoup, P3};

/// Above this the count is reported as "at least": the exact number no longer matters to a mesher
/// that refuses a model with one such pair as surely as with a thousand.
pub const PAIR_CAP: u64 = 500_000;

/// How many penetrating pairs were found, and whether the count stopped at [`PAIR_CAP`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairCount {
    pub pairs: u64,
    pub capped: bool,
}

/// Counts the penetrating triangle pairs of `soup`.
pub fn count_penetrating_pairs(soup: &TriSoup, bvh: &Bvh<'_>) -> PairCount {
    let tolerance = length_scale(soup) * 1e-9;
    let mut pairs = 0_u64;
    let mut capped = false;
    bvh.visit_overlapping_pairs(&mut |a, b| {
        let (ta, tb) = (soup.triangles[a as usize], soup.triangles[b as usize]);
        if ta.iter().any(|v| tb.contains(v)) {
            return true;
        }
        if penetrates(
            &soup.corners(a as usize),
            &soup.corners(b as usize),
            tolerance,
        ) {
            pairs += 1;
            if pairs >= PAIR_CAP {
                capped = true;
                return false;
            }
        }
        true
    });
    PairCount { pairs, capped }
}

/// The separation tolerance scale: the largest coordinate magnitude, floored at 1.
fn length_scale(soup: &TriSoup) -> f64 {
    soup.vertices
        .iter()
        .flat_map(|v| v.iter().map(|c| c.abs()))
        .fold(1.0, f64::max)
}

/// Whether two triangles overlap by more than `tolerance` on every separating axis: the two face
/// normals and the nine edge-pair directions when they are not parallel, or the six in-plane edge
/// normals when they are coplanar. Parallel triangles on different planes cannot penetrate.
fn penetrates(a: &[P3; 3], b: &[P3; 3], tolerance: f64) -> bool {
    let (na, nb) = (tri_normal(a), tri_normal(b));
    let parallel = length(cross(unit_or_zero(na), unit_or_zero(nb))) < 1e-9;
    if parallel {
        // On different planes they cannot touch, and on the same plane the normal axis is a
        // single point for both, so only the in-plane edge normals can tell them apart.
        if dot(unit_or_zero(na), sub(b[0], a[0])).abs() > tolerance {
            return false;
        }
        return (0..3).all(|i| {
            overlaps_on(cross(na, sub(a[(i + 1) % 3], a[i])), a, b, tolerance)
                && overlaps_on(cross(nb, sub(b[(i + 1) % 3], b[i])), a, b, tolerance)
        });
    }
    if !overlaps_on(na, a, b, tolerance) || !overlaps_on(nb, a, b, tolerance) {
        return false;
    }
    (0..3).all(|i| {
        let ea = sub(a[(i + 1) % 3], a[i]);
        (0..3).all(|j| {
            let eb = sub(b[(j + 1) % 3], b[j]);
            overlaps_on(cross(ea, eb), a, b, tolerance)
        })
    })
}

fn tri_normal(t: &[P3; 3]) -> P3 {
    cross(sub(t[1], t[0]), sub(t[2], t[0]))
}

fn unit_or_zero(v: P3) -> P3 {
    let len = length(v);
    if len > 0.0 {
        scale(v, 1.0 / len)
    } else {
        [0.0; 3]
    }
}

/// Whether the two projections on `axis` share an interior: each span reaches past the other's
/// start by more than `tolerance`. A triangle projected on its own normal is a single point, and
/// a point strictly inside the other span still counts, while a span that only touches does not.
/// A vanishing
/// axis cannot separate anything, so it reports overlap.
fn overlaps_on(axis: P3, a: &[P3; 3], b: &[P3; 3], tolerance: f64) -> bool {
    let len = length(axis);
    if len < 1e-30 {
        return true;
    }
    let unit = scale(axis, 1.0 / len);
    let (lo_a, hi_a) = span(unit, a);
    let (lo_b, hi_b) = span(unit, b);
    lo_a + tolerance < hi_b && lo_b + tolerance < hi_a
}

fn span(axis: P3, tri: &[P3; 3]) -> (f64, f64) {
    tri.iter()
        .map(|&p| dot(axis, p))
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), value| {
            (lo.min(value), hi.max(value))
        })
}

/// Counts the folds of `soup`: pairs of triangles on one edge whose opposite corners are exactly
/// coplanar with the edge and on the same side of it.
pub fn count_folds(soup: &TriSoup) -> u64 {
    let mut edges: Vec<(u32, u32, usize)> = Vec::with_capacity(soup.triangles.len() * 3);
    for (face, t) in soup.triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            edges.push((a.min(b), a.max(b), face));
        }
    }
    edges.sort_unstable();
    let mut folds = 0;
    for run in edges.chunk_by(|x, y| (x.0, x.1) == (y.0, y.1)) {
        let (a, b) = (run[0].0, run[0].1);
        for (i, first) in run.iter().enumerate() {
            for second in &run[i + 1..] {
                let (Some(c), Some(d)) = (
                    opposite(soup, first.2, a, b),
                    opposite(soup, second.2, a, b),
                ) else {
                    continue;
                };
                let corners = [a, b, c, d].map(|v| soup.vertices[v as usize]);
                if folded(corners) {
                    folds += 1;
                }
            }
        }
    }
    folds
}

/// The corner of triangle `face` that is neither `a` nor `b`, if it has exactly one.
fn opposite(soup: &TriSoup, face: usize, a: u32, b: u32) -> Option<u32> {
    let mut rest = soup.triangles[face]
        .iter()
        .copied()
        .filter(|&v| v != a && v != b);
    let corner = rest.next()?;
    rest.next().is_none().then_some(corner)
}

/// Whether `c` and `d` lie exactly in one plane with the edge `a..b` and on the same side of it.
fn folded([a, b, c, d]: [P3; 4]) -> bool {
    let p3 = |p: P3| Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    };
    if orient3d(p3(a), p3(b), p3(c), p3(d)) != 0.0 {
        return false;
    }
    // Coplanar, so the two triangles' normals are parallel and every axis projection orders the
    // corners alike up to one sign per projection; the first projection where `c` is off the
    // edge's line decides. A `c` on the line in every projection is a triangle with no side.
    for (i, j) in [(0, 1), (1, 2), (2, 0)] {
        let p2 = |p: P3| Coord { x: p[i], y: p[j] };
        let side_c = orient2d(p2(a), p2(b), p2(c));
        if side_c != 0.0 {
            let side_d = orient2d(p2(a), p2(b), p2(d));
            return side_c.signum() * side_d.signum() > 0.0 && side_d != 0.0;
        }
    }
    false
}
