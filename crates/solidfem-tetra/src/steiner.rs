//! Adding a point strictly inside the solid when no flip can make a constraint appear. Stuck
//! constraints are stuck because the tetrahedra around them have coplanar apexes and locked sides;
//! a new point in general position changes that geometry where flips cannot. `surface.rs` calls
//! this between recovery attempts, a bounded number of times.
//!
//! **Where the point goes.** The obstruction is a mesh edge, or the face a missing surface edge
//! passes through. The union of the tetrahedra around it is star-shaped from every point on the
//! edge or face itself, so a point just off it on the solid side sees that whole region and a
//! star of new tetrahedra replaces it, removing the obstruction. The offset is along the inward
//! normal of the surface at the blockage, as far as the room allows: the search tries the
//! largest offset first (the shapeliest elements) and halves it until the point fits.
//!
//! **What may never happen.** The point must be strictly inside the solid: an exact parity count
//! along a ray tells inside from outside, and a point exactly on the surface is rejected. The
//! cavity is the obstruction's tetrahedra plus anything needed to make it star-shaped around the
//! point, each exact; it may not take a locked face or remove a locked edge, so every constraint
//! recovered so far survives. A point that cannot be placed is reported rather than forced.

use crate::core::{FACE, GHOST};
use crate::error::TetraError;
use crate::flips::{edge_key, face_key, Flipper};
use crate::predicates::orient;

/// How many times one constraint may have a point added on its behalf.
pub(crate) const MAX_STEINER_PER_CONSTRAINT: usize = 8;

/// The largest number of halvings of the offset tried before giving up on a position.
const OFFSET_HALVINGS: i32 = 16;

/// The tetrahedra a point must replace to clear an obstruction.
pub(crate) enum Obstruction {
    /// All tetrahedra around a mesh edge.
    Edge(u32, u32),
    /// The two tetrahedra sharing the face of tetrahedron `t` opposite slot `slot`.
    Face(u32, usize),
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
    /// checks no locked face or edge would be destroyed.
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
        for &t in &inside {
            let tet = self.core.tets[t as usize];
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

    /// Whether some tetrahedron containing edge `(a, b)` lies outside `cavity`.
    fn edge_survives(&mut self, a: u32, b: u32, cavity: &[u32]) -> bool {
        self.tets_around(a).into_iter().any(|t| {
            let tet = self.core.tets[t as usize];
            tet.contains(&b) && !cavity.contains(&t)
        })
    }
}
