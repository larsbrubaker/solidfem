//! The interior edges between surface vertices that run along the surface rather than through the
//! solid, so that `skins.rs` does not call them through a wall: a point placed on one would lie in
//! the surface, and with a surface triangle make an element flat to rounding. The commonest is the
//! other diagonal of a flat quad whose corners were rounded.
//!
//! **What "along the surface" means.** A chord u-v is excluded only when it can be walked from one
//! end to the other across triangles of one flat patch: from a vertex, into the first triangle in
//! surface order whose corner there holds the chord's direction strictly (seen along that
//! triangle's normal) and which holds both ends within the tolerance below. At a crease vertex the
//! direction seen along the other face's normal can fall inside that face's corner too, so the
//! tolerance is what picks the face the chord runs along, whatever the triangle order. Then
//! across each edge the chord leaves by into the patch's triangle beyond it, until the other end
//! is a corner, or on from a vertex the chord passes within `2e` of (three points meant to be in a
//! line, each rounded by at most `e`) strictly further along it. Each triangle crossed must have a
//! plane and hold both ends within the tolerance below of it, so the chord's part over it is
//! within that of the triangle in height and over it in plan. So every point of an excluded chord
//! is within the coordinates' rounding of the surface, and a point on it is inside the solid by no
//! more than the coordinates can resolve. Being in a plane both ends share is not enough: a rib
//! standing in a plate has the root vertices of both its skins on the plate's top patch, and the
//! chord between them lies in that plane yet passes through the rib; the walk finds no patch
//! triangle across the rib's foot, nor across any hole of an annular or L-shaped patch a chord
//! leaves and comes back to. Any chord that cannot be walked is left to the wall test, which leans
//! towards calling it through. So is a chord leaving a vertex exactly along a surface edge, which
//! no corner holds strictly: the walk does not open, which is the safe side.
//!
//! **The tolerance.** Surfaces arrive with coordinates stored in single precision, rounded by at
//! most `2^-24` of the value, so a point is within `e = sqrt(3) 2^-24 M` of where it was meant
//! (`M` the surface's largest coordinate magnitude), and a rounded flat quad is two triangles
//! folded by a rounding-sized angle. Rounding can give a triangle's computed plane a slope of at
//! most `4e / h` against the intended one (see [`rounding_tilt`]), so a point of the intended plane
//! within `|uv|` of the triangle is within `2e + |uv| 4e / h` of the computed plane, with no small
//! angle taken. A triangle whose slope bound exceeds the one [`max_tilt`] gives has no plane: a
//! plane rounding can turn further than two faces of one flat patch may differ is not known to be
//! the patch's plane. The chord is walked from both ends, so the answer is the same in either
//! order.

use std::collections::HashMap;

use crate::quality::{cross, dot, length, subtract};
use crate::skins::SAME_FACING_COSINE;

/// The relative rounding of one coordinate as surfaces arrive: mesh files and modelling kernels
/// store positions in single precision, whose rounding is at most half a unit in the last place,
/// `2^-24` of the value.
const COORDINATE_ROUNDING: f64 = 1.0 / 16_777_216.0;

/// The largest rounding slope a triangle may have and still give a plane, from the angle `t` two
/// faces of one flat patch may differ by ([`SAME_FACING_COSINE`]): `sqrt(2 (1 - cos t))` is
/// `2 sin(t / 2)`, just under `t` and under `tan t`, so a plane accepted here is turned by rounding
/// less than one patch's faces may differ. Computed from the constant so the two cannot drift.
fn max_tilt() -> f64 {
    (2.0 * (1.0 - SAME_FACING_COSINE)).sqrt()
}

/// One surface triangle as the chord test reads it.
#[derive(Clone, Copy)]
struct Facet {
    corners: [u32; 3],
    normal: [f64; 3],
    patch: u32,
    /// The rounding tilt bound, when the triangle gives a plane.
    tilt: Option<f64>,
}

/// The surface triangles, by vertex and by edge, for [`QuadChords::along_surface`].
#[derive(Default)]
pub(crate) struct QuadChords {
    facets: Vec<Facet>,
    /// Each surface vertex's triangles, in surface order.
    at_vertex: Vec<Vec<u32>>,
    /// Each surface edge's triangles, read only by key.
    at_edge: HashMap<(u32, u32), Vec<u32>>,
    /// `2e`: how far rounding can put two points, each off the intended plane, apart across it.
    offset: f64,
}

impl QuadChords {
    /// The chord data of a surface, with each triangle's unit normal and flat patch.
    pub(crate) fn new(
        points: &[[f64; 3]],
        boundary: &[[u32; 3]],
        normals: &[[f64; 3]],
        patches: &[u32],
        count: usize,
    ) -> Self {
        let magnitude = boundary
            .iter()
            .flatten()
            .flat_map(|&v| points[v as usize])
            .fold(0.0_f64, |m, c| m.max(c.abs()));
        let rounding = 3.0_f64.sqrt() * COORDINATE_ROUNDING * magnitude;
        let max_tilt = max_tilt();
        let mut at_vertex: Vec<Vec<u32>> = vec![Vec::new(); count];
        let mut at_edge: HashMap<(u32, u32), Vec<u32>> = HashMap::with_capacity(boundary.len() * 2);
        let mut facets = Vec::with_capacity(boundary.len());
        for (index, face) in boundary.iter().enumerate() {
            // `as` saturates; a surface has far fewer than `u32::MAX` triangles.
            let id = index as u32;
            for (k, &v) in face.iter().enumerate() {
                at_vertex[v as usize].push(id);
                let w = face[(k + 1) % 3];
                at_edge.entry((v.min(w), v.max(w))).or_default().push(id);
            }
            let tilt = rounding_tilt(face.map(|v| points[v as usize]), rounding);
            facets.push(Facet {
                corners: *face,
                normal: normals[index],
                patch: patches[index],
                tilt: (tilt <= max_tilt).then_some(tilt),
            });
        }
        Self {
            facets,
            at_vertex,
            at_edge,
            offset: 2.0 * rounding,
        }
    }

    /// Whether the chord from `u` to `v` runs along one flat patch (the module header's rule),
    /// walked from either end so the answer does not depend on the order the ends are passed in.
    pub(crate) fn along_surface(&self, points: &[[f64; 3]], u: u32, v: u32) -> bool {
        self.walk(points, u, v) || self.walk(points, v, u)
    }

    /// The walk from `u` towards `v` across the triangles of one flat patch, each within rounding
    /// of the chord's ends (`near`): from a vertex, the triangle whose corner there holds the
    /// chord's direction strictly; then across edges, keeping `a` right of the chord and `b` left,
    /// until `v` is a corner, or on through a vertex within `2e` of the chord's line strictly
    /// further along it. Every step moves along the chord or through a triangle, and there are
    /// finitely many, so the walk is capped by the triangle count.
    fn walk(&self, points: &[[f64; 3]], u: u32, v: u32) -> bool {
        let (pu, pv) = (points[u as usize], points[v as usize]);
        let along = subtract(pv, pu);
        let squared = dot(along, along);
        let reach = length(along);
        if squared == 0.0 {
            return false;
        }
        let p = |i: u32| points[i as usize];
        let near = |t: &Facet| {
            t.tilt.is_some_and(|tilt| {
                let corner = p(t.corners[0]);
                let limit = self.offset + reach * tilt;
                dot(subtract(pu, corner), t.normal).abs() <= limit
                    && dot(subtract(pv, corner), t.normal).abs() <= limit
            })
        };
        let left = |at: [f64; 3], t: &Facet| dot(cross(along, subtract(at, pu)), t.normal);
        let (mut start, mut progress) = (u, 0.0_f64);
        let mut budget = self.facets.len() + 1;
        'vertex: loop {
            let opening = self.at_vertex[start as usize].iter().find_map(|&f| {
                let t = &self.facets[f as usize];
                let k = t.corners.iter().position(|&c| c == start)?;
                let (a, b) = (t.corners[(k + 1) % 3], t.corners[(k + 2) % 3]);
                let ps = p(start);
                let to_v = subtract(pv, ps);
                let holds = dot(cross(subtract(p(a), ps), to_v), t.normal) > 0.0
                    && dot(cross(to_v, subtract(p(b), ps)), t.normal) > 0.0;
                // Only a triangle the chord is near can open the walk: at a crease vertex the
                // chord's direction seen along the other face's normal can fall inside that
                // face's corner too, and taking it would make the answer depend on triangle order.
                ((holds || a == v || b == v) && near(t)).then_some((f, a, b))
            });
            let Some((mut tri, mut a, mut b)) = opening else {
                return false;
            };
            loop {
                budget = match budget.checked_sub(1) {
                    Some(left_over) => left_over,
                    None => return false,
                };
                let t = self.facets[tri as usize];
                if !near(&t) {
                    return false;
                }
                if a == v || b == v {
                    return true;
                }
                let next = self.at_edge.get(&(a.min(b), a.max(b))).and_then(|sharing| {
                    sharing
                        .iter()
                        .copied()
                        .find(|&g| g != tri && self.facets[g as usize].patch == t.patch)
                });
                let Some(next) = next else {
                    return false;
                };
                let t2 = self.facets[next as usize];
                let Some(c) = t2.corners.into_iter().find(|&x| x != a && x != b) else {
                    return false;
                };
                if !near(&t2) {
                    return false;
                }
                if c == v {
                    return true;
                }
                let offset = subtract(p(c), pu);
                let along_c = dot(offset, along) / squared;
                let off_line = length(subtract(offset, scaled(along, along_c)));
                if off_line <= self.offset {
                    if along_c <= progress || along_c >= 1.0 {
                        return false;
                    }
                    (start, progress) = (c, along_c);
                    continue 'vertex;
                }
                if left(p(c), &t2) > 0.0 {
                    b = c;
                } else {
                    a = c;
                }
                tri = next;
            }
        }
    }
}

fn scaled(v: [f64; 3], by: f64) -> [f64; 3] {
    [v[0] * by, v[1] * by, v[2] * by]
}

/// How far rounding of `rounding` on each corner can tilt a triangle's plane, as a slope. Each
/// corner's offset from the intended plane is in `[-e, e]`; the slope is the gradient of the linear
/// function taking those offsets, `sum (d_i - d_0) grad(lambda_i)` over the two other corners, at
/// most `2e / h_1 + 2e / h_2 <= 4e / h` with `|grad(lambda_i)| = 1 / h_i` and `h` the triangle's
/// smallest height. Infinite for a triangle with no area.
fn rounding_tilt(corners: [[f64; 3]; 3], rounding: f64) -> f64 {
    let [a, b, c] = corners;
    let doubled_area = length(cross(subtract(b, a), subtract(c, a)));
    let longest = [
        length(subtract(b, a)),
        length(subtract(c, b)),
        length(subtract(a, c)),
    ]
    .into_iter()
    .fold(0.0_f64, f64::max);
    let height = doubled_area / longest;
    if height > 0.0 {
        4.0 * rounding / height
    } else {
        f64::INFINITY
    }
}
