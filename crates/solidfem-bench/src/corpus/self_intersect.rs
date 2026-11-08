//! Counts triangle pairs whose interiors penetrate each other, the defect that makes a surface
//! unusable as a tetrahedral mesher's boundary. A pair counts only when it overlaps by more than
//! a small tolerance on every separating axis, so triangles that merely touch along a shared edge
//! or at a T-junction do not. Pairs sharing a vertex are skipped, so the count is a lower bound.

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
