//! Adding a point strictly inside the solid when no flip can make a constraint appear. Stuck
//! constraints are stuck because the tetrahedra around them have coplanar apexes and locked sides;
//! a new point in general position changes that geometry where flips cannot. `surface.rs` calls
//! this between recovery attempts, a bounded number of times.
//!
//! **Where the point goes.** The obstruction is a mesh edge, or the face a missing surface edge
//! passes through. The union of the tetrahedra around it is star-shaped from every point on the
//! edge or face itself, so a point just off it on the solid side sees that whole region and a
//! star of new tetrahedra replaces it, removing the obstruction. A hull edge folded over a stuck
//! edge cannot be removed at all, so there the point takes every real tetrahedron around that hull
//! edge, leaving a ring the fold flip in `flips.rs` can close. The offset is along the inward
//! normal of the surface at the blockage, as far as the room allows: the search tries the
//! largest offset first (the shapeliest elements) and halves it until the point fits.
//!
//! **Which obstruction.** A face or edge in the walk's way may be held there by a mesh edge lying
//! across the segment whose ring is too long to triangulate: the walk returns to it however many
//! points are placed around one face. Such an edge has both segment ends on its ring, so
//! [`edge_obstructions`] offers first the tetrahedra of one arc of that ring (the longer arc
//! first). The point takes them, the ring shrinks to the other arc plus the ends and the point,
//! and the edge can then be flipped away with the segment created.
//!
//! **What may never happen.** The point must be strictly inside the solid: an exact parity count
//! along a ray tells inside from outside, and a point exactly on the surface is rejected. The
//! cavity is the obstruction's tetrahedra plus anything needed to make it star-shaped around the
//! point, each exact; it may not take a locked face or remove a locked edge, so every constraint
//! recovered so far survives. A surface vertex always keeps a tetrahedron outside the cavity; a
//! point added earlier may be swallowed by a later cavity, and extraction drops any added point
//! left in no tetrahedron (`surface.rs`). A point that cannot be placed is reported rather than
//! forced.

use crate::core::{FACE, GHOST};
use crate::error::TetraError;
use crate::flips::{edge_key, face_key, Flipper};
use crate::predicates::orient;
use crate::recover::Entry;

/// How many times one constraint may have a point added on its behalf.
pub(crate) const MAX_STEINER_PER_CONSTRAINT: usize = 8;

/// The largest number of halvings of the offset tried before giving up on a position.
const OFFSET_HALVINGS: i32 = 16;

/// The tetrahedra a point must replace to clear an obstruction.
#[derive(Clone, Copy)]
pub(crate) enum Obstruction {
    /// All tetrahedra around a mesh edge.
    Edge(u32, u32),
    /// The two tetrahedra sharing the face of tetrahedron `t` opposite slot `slot`.
    Face(u32, usize),
    /// The real tetrahedra around a hull edge that folds over a stuck surface edge: a point
    /// seeing all of them shortens the fold's ring to just that edge's ends, so the fold flip
    /// (`Flipper::cap_hull_fold`) can finish.
    HullFold(u32, u32),
    /// The tetrahedra around mesh edge `(a, b)` on the arc of its ring from `from` round to `to`:
    /// the ring is left holding `from`, `to`, the point and the other arc, so the edge can be
    /// flipped away with `(from, to)` created.
    RingArc { a: u32, b: u32, from: u32, to: u32 },
}

/// The surface as the inside test needs it: its triangles and their bounding boxes.
pub(crate) struct SolidRegion {
    triangles: Vec<[u32; 3]>,
    boxes: Vec<([f64; 3], [f64; 3])>,
    high: [f64; 3],
    low: [f64; 3],
}

/// What is known about a position relative to the solid.
enum Side {
    Inside,
    Outside,
    OnSurface,
}

impl SolidRegion {
    pub(crate) fn new(points: &[[f64; 3]], triangles: &[[u32; 3]]) -> Self {
        let mut boxes = Vec::with_capacity(triangles.len());
        let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for t in triangles {
            let [a, b, c] = t.map(|v| points[v as usize]);
            let mut bmin = [0.0; 3];
            let mut bmax = [0.0; 3];
            for k in 0..3 {
                bmin[k] = a[k].min(b[k]).min(c[k]);
                bmax[k] = a[k].max(b[k]).max(c[k]);
                low[k] = low[k].min(bmin[k]);
                high[k] = high[k].max(bmax[k]);
            }
            boxes.push((bmin, bmax));
        }
        Self {
            triangles: triangles.to_vec(),
            boxes,
            high,
            low,
        }
    }

    /// Exact parity along a ray from `point` along a coordinate axis. A ray that grazes a
    /// triangle's edge or plane is not decided by parity, so a blocked direction is abandoned
    /// for the next; if all six are blocked the point is treated as not placeable.
    fn classify(&self, points: &[[f64; 3]], point: [f64; 3]) -> Option<Side> {
        for (t, bounds) in self.triangles.iter().zip(&self.boxes) {
            if (0..3).all(|k| point[k] >= bounds.0[k] && point[k] <= bounds.1[k])
                && on_triangle(points, *t, point)
            {
                return Some(Side::OnSurface);
            }
        }
        'directions: for (axis, positive) in [
            (0, true),
            (0, false),
            (1, true),
            (1, false),
            (2, true),
            (2, false),
        ] {
            let mut end = point;
            end[axis] = if positive {
                self.high[axis] + 1.0
            } else {
                self.low[axis] - 1.0
            };
            let others = [(axis + 1) % 3, (axis + 2) % 3];
            let mut crossings = 0usize;
            for (t, bounds) in self.triangles.iter().zip(&self.boxes) {
                if others
                    .iter()
                    .any(|&k| point[k] < bounds.0[k] || point[k] > bounds.1[k])
                {
                    continue;
                }
                let along_low = if positive { point[axis] } else { end[axis] };
                let along_high = if positive { end[axis] } else { point[axis] };
                if bounds.1[axis] < along_low || bounds.0[axis] > along_high {
                    continue;
                }
                match ray_hits(points, *t, point, end) {
                    Hit::Through => crossings += 1,
                    Hit::Grazes => continue 'directions,
                    Hit::Misses => {}
                }
            }
            return Some(if crossings % 2 == 1 {
                Side::Inside
            } else {
                Side::Outside
            });
        }
        None
    }
}

enum Hit {
    Through,
    Grazes,
    Misses,
}

/// Exact test of the segment from `start` to `end` against a triangle.
fn ray_hits(points: &[[f64; 3]], t: [u32; 3], start: [f64; 3], end: [f64; 3]) -> Hit {
    let [a, b, c] = t.map(|v| &points[v as usize]);
    let (s1, s2) = (orient(a, b, c, &start), orient(a, b, c, &end));
    if s1 * s2 > 0.0 {
        return Hit::Misses;
    }
    let sides = [
        orient(&start, &end, a, b),
        orient(&start, &end, b, c),
        orient(&start, &end, c, a),
    ];
    let positive = sides.iter().all(|&s| s > 0.0);
    let negative = sides.iter().all(|&s| s < 0.0);
    if (positive || negative) && s1 != 0.0 && s2 != 0.0 {
        return Hit::Through;
    }
    // A zero is only a graze if the ray can actually reach the triangle; opposite strict signs
    // among the three sides mean it passes clear beside it.
    let any_positive = sides.iter().any(|&s| s > 0.0);
    let any_negative = sides.iter().any(|&s| s < 0.0);
    if any_positive && any_negative {
        return Hit::Misses;
    }
    Hit::Grazes
}

/// Whether `point` lies exactly on triangle `t`, by exact tests in the plane.
fn on_triangle(points: &[[f64; 3]], t: [u32; 3], point: [f64; 3]) -> bool {
    use robust::{orient2d, Coord};
    let [a, b, c] = t.map(|v| &points[v as usize]);
    if orient(a, b, c, &point) != 0.0 {
        return false;
    }
    for axes in [(0, 1), (1, 2), (0, 2)] {
        let at = |q: &[f64; 3]| Coord {
            x: q[axes.0],
            y: q[axes.1],
        };
        if orient2d(at(a), at(b), at(c)) == 0.0 {
            continue;
        }
        let sides = [
            orient2d(at(a), at(b), at(&point)),
            orient2d(at(b), at(c), at(&point)),
            orient2d(at(c), at(a), at(&point)),
        ];
        return sides.iter().all(|&s| s >= 0.0) || sides.iter().all(|&s| s <= 0.0);
    }
    false
}

/// Outcome of trying to insert a point for an obstruction.
pub(crate) enum Placed {
    /// A point was added; the stuck constraint should be retried.
    Added,
    /// Nothing could be placed. `no_room` means every candidate in the search was outside the
    /// solid or had no constraint-respecting cavity, so the solid has no room here for a point.
    NotPlaced { no_room: bool },
}

impl Flipper {
    /// Tries to add a point near `base`, offset by `directions` scaled by `scale`, clearing
    /// `obstruction`.
    pub(crate) fn insert_steiner(
        &mut self,
        region: &SolidRegion,
        obstruction: Obstruction,
        base: [f64; 3],
        directions: &[[f64; 3]],
        scale: f64,
    ) -> Result<Placed, TetraError> {
        let seeds = match obstruction {
            Obstruction::RingArc { a, b, from, to } => {
                let Some((ring, tets)) = self.ring_public(a, b) else {
                    return Ok(Placed::NotPlaced { no_room: false });
                };
                let (Some(start), Some(end)) = (
                    ring.iter().position(|&w| w == from),
                    ring.iter().position(|&w| w == to),
                ) else {
                    return Ok(Placed::NotPlaced { no_room: false });
                };
                let n = ring.len();
                let mut arc = Vec::new();
                let mut i = start;
                while i != end {
                    arc.push(tets[i]);
                    i = (i + 1) % n;
                }
                arc
            }
            Obstruction::HullFold(a, b) => {
                let Some((_, tets)) = self.ring_public(a, b) else {
                    return Ok(Placed::NotPlaced { no_room: false });
                };
                tets.into_iter()
                    .filter(|&t| !self.core.tets[t as usize].contains(&GHOST))
                    .collect()
            }
            Obstruction::Edge(a, b) => {
                let Some((_, tets)) = self.ring_public(a, b) else {
                    return Ok(Placed::NotPlaced { no_room: false });
                };
                tets
            }
            Obstruction::Face(t, slot) => {
                let n = self.core.neighbors[t as usize][slot];
                let face = FACE[slot].map(|p| self.core.tets[t as usize][p]);
                if self
                    .locked_faces
                    .contains(&face_key(face[0], face[1], face[2]))
                {
                    return Ok(Placed::NotPlaced { no_room: false });
                }
                vec![t, n]
            }
        };
        if seeds
            .iter()
            .any(|&t| self.core.tets[t as usize].contains(&GHOST))
        {
            return Ok(Placed::NotPlaced { no_room: false });
        }
        for direction in directions {
            for halvings in 0..OFFSET_HALVINGS {
                let offset = scale * 2.0_f64.powi(-halvings);
                let candidate = [
                    base[0] + direction[0] * offset,
                    base[1] + direction[1] * offset,
                    base[2] + direction[2] * offset,
                ];
                if candidate.iter().any(|value| !value.is_finite()) {
                    continue;
                }
                let Some(cavity) = self.steiner_cavity(&seeds, candidate) else {
                    continue;
                };
                match region.classify(&self.core.points, candidate) {
                    Some(Side::Inside) => {}
                    _ => continue,
                }
                self.add_steiner(&cavity, candidate)?;
                return Ok(Placed::Added);
            }
        }
        Ok(Placed::NotPlaced { no_room: true })
    }

    /// The cavity for a point at `candidate`, or `None` when no such cavity respects the
    /// constraints. Grows from `seeds` across any boundary face the point cannot see, then
    /// checks no locked face or edge would be destroyed. A surface vertex must keep a
    /// tetrahedron outside the cavity; a point added earlier may be swallowed (see `add_steiner`).
    fn steiner_cavity(&mut self, seeds: &[u32], candidate: [f64; 3]) -> Option<Vec<u32>> {
        let mut cavity: Vec<u32> = seeds.to_vec();
        let mut grew = true;
        while grew {
            grew = false;
            for index in 0..cavity.len() {
                let t = cavity[index];
                let tet = self.core.tets[t as usize];
                if tet.contains(&GHOST) {
                    return None;
                }
                for (i, positions) in FACE.iter().enumerate() {
                    let n = self.core.neighbors[t as usize][i];
                    if cavity.contains(&n) {
                        continue;
                    }
                    let [a, b, c] =
                        positions.map(|position| &self.core.points[tet[position] as usize]);
                    if orient(a, b, c, &candidate) > 0.0 {
                        continue;
                    }
                    let face = positions.map(|position| tet[position]);
                    if self
                        .locked_faces
                        .contains(&face_key(face[0], face[1], face[2]))
                    {
                        return None;
                    }
                    if self.core.tets[n as usize].contains(&GHOST) {
                        return None;
                    }
                    cavity.push(n);
                    grew = true;
                }
                if cavity.len() > 512 {
                    return None;
                }
            }
        }
        // A locked edge may stay only if some tetrahedron around it survives; a locked face
        // shared by two cavity tetrahedra would be destroyed.
        let inside = cavity.clone();
        // Only the surface vertices have to keep a tetrahedron; an added point is free to be
        // replaced, and extraction drops any left in no tetrahedron.
        let surface_count = self.core.points.len() - self.steiner_points;
        for &t in &inside {
            let tet = self.core.tets[t as usize];
            if tet
                .iter()
                .any(|&v| (v as usize) < surface_count && !self.vertex_survives(v, &inside))
            {
                return None;
            }
            for (i, positions) in FACE.iter().enumerate() {
                let n = self.core.neighbors[t as usize][i];
                if inside.contains(&n) {
                    let face = positions.map(|position| tet[position]);
                    if self
                        .locked_faces
                        .contains(&face_key(face[0], face[1], face[2]))
                    {
                        return None;
                    }
                }
            }
            for a in 0..4 {
                for b in a + 1..4 {
                    if self.locked_edges.contains(&edge_key(tet[a], tet[b]))
                        && !self.edge_survives(tet[a], tet[b], &inside)
                    {
                        return None;
                    }
                }
            }
        }
        Some(cavity)
    }

    /// The edges of the face shared by `t` and its neighbour across `slot` that make its 2 to 3 flip
    /// illegal, in the order of the face edges. Empty when the flip is legal, or impossible because
    /// the face is on the hull or locked.
    pub(crate) fn face_blockers(&self, t: u32, slot: usize) -> Vec<(u32, u32)> {
        let tet = self.core.tets[t as usize];
        let u = tet[slot];
        let other = self.core.tets[self.core.neighbors[t as usize][slot] as usize];
        let Some(y) = other.iter().copied().find(|w| !tet.contains(w)) else {
            return Vec::new();
        };
        let face = FACE[slot].map(|position| tet[position]);
        let (a, b, c) = (face[1], face[0], face[2]);
        if u == GHOST
            || y == GHOST
            || [a, b, c].contains(&GHOST)
            || self.locked_faces.contains(&face_key(a, b, c))
        {
            return Vec::new();
        }
        [(a, b), (b, c), (c, a)]
            .into_iter()
            .filter(|&(x, z)| !self.tet_positive(&[u, y, x, z]))
            .collect()
    }

    /// Whether vertex `v` keeps a tetrahedron outside `cavity`.
    fn vertex_survives(&mut self, v: u32, cavity: &[u32]) -> bool {
        v == GHOST
            || self
                .tets_around(v)
                .into_iter()
                .any(|t| !cavity.contains(&t))
    }

    /// Whether some tetrahedron containing edge `(a, b)` lies outside `cavity`.
    fn edge_survives(&mut self, a: u32, b: u32, cavity: &[u32]) -> bool {
        self.tets_around(a).into_iter().any(|t| {
            let tet = self.core.tets[t as usize];
            tet.contains(&b) && !cavity.contains(&t)
        })
    }
}

/// What a point added for a stuck edge `(u, v)` can clear, best first, with the position to offset
/// from. What stops the walk is what must be cleared, not the face it stopped at: a blocked face
/// keeps its blocking edge after a point replaces only the two tetrahedra around it, so the walk
/// comes straight back to the same edge and a row of points piles up against it. A mesh edge
/// lying across the segment comes first: taking one arc of its ring (the longer first, the
/// tetrahedra that keep its ring from triangulating) leaves a ring it can be flipped through.
pub(crate) fn edge_obstructions(
    flipper: &mut Flipper,
    entry: Entry,
    u: u32,
    v: u32,
) -> Vec<(Obstruction, [f64; 3])> {
    let point = |flipper: &Flipper, w: u32| flipper.core.points[w as usize];
    let (pu, pv) = (point(flipper, u), point(flipper, v));
    match entry {
        Entry::Face(t, slot) => {
            let face = FACE[slot].map(|p| flipper.core.tets[t as usize][p]);
            if let Some((a, b)) = flipper.hull_fold_over(face, u, v) {
                return vec![(Obstruction::HullFold(a, b), midpoint(pu, pv))];
            }
            let [a, b, c] = face.map(|f| point(flipper, f));
            let crossing = segment_plane_crossing(pu, pv, a, b, c);
            let mut candidates = Vec::new();
            if let Some((p, q)) = chord_across(flipper, u, v) {
                let base = midpoint(point(flipper, p), point(flipper, q));
                if let Some((ring, _)) = flipper.ring_public(p, q) {
                    let n = ring.len();
                    let (from_u, from_v) = (
                        ring.iter().position(|&w| w == u),
                        ring.iter().position(|&w| w == v),
                    );
                    if let (Some(i), Some(j)) = (from_u, from_v) {
                        // Ring arc lengths from u round to v, and the other way; equal lengths keep the
                        // arc starting at `u` first (the sort is stable).
                        let mut arcs = [((j + n - i) % n, u, v), ((i + n - j) % n, v, u)];
                        arcs.sort_by_key(|x| std::cmp::Reverse(x.0));
                        for (_, from, to) in arcs {
                            candidates.push((
                                Obstruction::RingArc {
                                    a: p,
                                    b: q,
                                    from,
                                    to,
                                },
                                base,
                            ));
                        }
                    }
                }
                candidates.push((Obstruction::Edge(p, q), base));
            }
            for (x, y) in flipper.face_blockers(t, slot) {
                if !flipper.locked_edges.contains(&edge_key(x, y)) {
                    let (px, py) = (point(flipper, x), point(flipper, y));
                    candidates.push((Obstruction::Edge(x, y), midpoint(px, py)));
                }
            }
            candidates.push((Obstruction::Face(t, slot), crossing));
            candidates
        }
        Entry::Edge(a, b) => {
            let (pa, pb) = (point(flipper, a), point(flipper, b));
            vec![(Obstruction::Edge(a, b), midpoint(pa, pb))]
        }
    }
}

/// The unlocked mesh edge whose ring holds both `u` and `v` and which passes closest to the
/// segment between them: that edge lies across the segment and has to go for the edge to exist.
/// Found among the tetrahedra around `u`, which hold every edge that can have `u` on its ring.
/// Only a choice of where to try; everything after it is decided exactly.
fn chord_across(flipper: &mut Flipper, u: u32, v: u32) -> Option<(u32, u32)> {
    let mut chords: Vec<(u32, u32)> = Vec::new();
    for t in flipper.tets_around(u) {
        let tet = flipper.core.tets[t as usize];
        for i in 0..4 {
            for j in i + 1..4 {
                let (a, b) = (tet[i].min(tet[j]), tet[i].max(tet[j]));
                if a != GHOST
                    && b != GHOST
                    && a != u
                    && b != u
                    && !chords.contains(&(a, b))
                    && !flipper.locked_edges.contains(&edge_key(a, b))
                {
                    chords.push((a, b));
                }
            }
        }
    }
    chords.sort_unstable();
    let (pu, pv) = (
        flipper.core.points[u as usize],
        flipper.core.points[v as usize],
    );
    let mut best: Option<(f64, (u32, u32))> = None;
    for (a, b) in chords {
        let (pa, pb) = (
            flipper.core.points[a as usize],
            flipper.core.points[b as usize],
        );
        let distance = segment_distance(pu, pv, pa, pb);
        if best.is_none_or(|(d, _)| distance < d) {
            let Some((ring, _)) = flipper.ring_public(a, b) else {
                continue;
            };
            if ring.contains(&u) && ring.contains(&v) {
                best = Some((distance, (a, b)));
            }
        }
    }
    best.map(|(_, chord)| chord)
}

/// The closest approach of segments `a b` and `c d`.
fn segment_distance(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let between = |x: [f64; 3], y: [f64; 3]| [y[0] - x[0], y[1] - x[1], y[2] - x[2]];
    let (e, f, r) = (between(a, b), between(c, d), between(c, a));
    let (ee, ef, ff, er, fr) = (dot(e, e), dot(e, f), dot(f, f), dot(e, r), dot(f, r));
    let det = ee * ff - ef * ef;
    let (s, t) = if det > 0.0 {
        (
            ((ef * fr - ff * er) / det).clamp(0.0, 1.0),
            ((ee * fr - ef * er) / det).clamp(0.0, 1.0),
        )
    } else {
        (0.5, 0.5)
    };
    let at = |p: [f64; 3], dir: [f64; 3], k: f64| {
        [p[0] + dir[0] * k, p[1] + dir[1] * k, p[2] + dir[2] * k]
    };
    let gap = between(at(a, e, s), at(c, f, t));
    dot(gap, gap).sqrt()
}

pub(crate) fn triangle_normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    normalized([
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ])
}

pub(crate) fn normalized(v: [f64; 3]) -> [f64; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if length > 0.0 {
        [v[0] / length, v[1] / length, v[2] / length]
    } else {
        [0.0; 3]
    }
}

pub(crate) fn midpoint(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ]
}

/// Where the segment `p q` meets the plane of `a b c`, to within rounding: only a position to
/// offset from, every decision about it is made exactly afterwards.
pub(crate) fn segment_plane_crossing(
    p: [f64; 3],
    q: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> [f64; 3] {
    let n = triangle_normal(a, b, c);
    let dot = |v: [f64; 3]| n[0] * v[0] + n[1] * v[1] + n[2] * v[2];
    let (dp, dq) = (
        dot([p[0] - a[0], p[1] - a[1], p[2] - a[2]]),
        dot([q[0] - a[0], q[1] - a[1], q[2] - a[2]]),
    );
    let denominator = dq - dp;
    let t = if denominator == 0.0 {
        0.5
    } else {
        (-dp / denominator).clamp(0.0, 1.0)
    };
    [
        p[0] + (q[0] - p[0]) * t,
        p[1] + (q[1] - p[1]) * t,
        p[2] + (q[2] - p[2]) * t,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangulator::Triangulator;

    /// The corners of a cube and a point near its centre, as a finished triangulation.
    fn cube_with_centre() -> Flipper {
        let mut points = Vec::new();
        for x in [0.0, 1.0] {
            for y in [0.0, 1.0] {
                for z in [0.0, 1.0] {
                    points.push([x, y, z]);
                }
            }
        }
        points.push([0.5, 0.5, 0.5]);
        let mut triangulator = Triangulator::new(points).expect("valid points");
        triangulator
            .step(usize::MAX, &mut || false)
            .expect("triangulates");
        let core = triangulator.into_core();
        Flipper::new(core, Default::default(), Default::default(), Vec::new())
    }

    /// A point whose whole star another point's cavity took is in no tetrahedron. Nothing may
    /// reach it through the slot it used to name, which has been freed or reused by then.
    #[test]
    fn a_swallowed_point_has_no_tetrahedra_and_no_edges() {
        let mut flipper = cube_with_centre();
        let centre = 8;
        let star = flipper.tets_around(centre);
        assert!(!star.is_empty(), "the centre is in the triangulation");
        flipper
            .add_steiner(&star, [0.5, 0.5, 0.25])
            .expect("the star of the centre is star-shaped around a point near it");
        assert!(flipper.tets_around(centre).is_empty());
        assert_eq!(flipper.find_edge(centre, 0), None);
        assert!(flipper.ring_public(centre, 0).is_none());
        assert_eq!(flipper.swallowed_points, 1);
        assert_eq!(flipper.kept_steiner_points(), 0);
    }
}
