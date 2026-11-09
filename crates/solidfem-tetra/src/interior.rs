//! Adding a point strictly inside a finished solid, the one way refinement creates vertices:
//! find the tetrahedra the point is in, remove those it conflicts with, and star the hole to it.
//! `refine.rs` and `layers.rs` choose where points go; this file says whether one can go there.
//!
//! **Why a point in a tetrahedron is inside the solid.** The mesh is the solid, so location is
//! exact containment: a walk across real faces only, ending at a tetrahedron with no negative
//! orientation. The point is then strictly inside it, or on a face or edge shared only by real
//! tetrahedra, because a face or edge on the surface has a ghost and refuses. Being in a real
//! tetrahedron is the proof of being inside; no parity count is needed.
//!
//! **The cavity.** Tetrahedra the point conflicts with (exact in-sphere), flooded from the seed
//! across real faces only, so the fixed surface is never crossed. The mesh is not Delaunay near
//! the surface or after flips, so conflict alone need not be star-shaped: a tetrahedron with an
//! outer face the point does not see strictly is removed from the cavity, the part still
//! joined to the seeds is kept, and that is repeated. Only a seed needing removal refuses the
//! point. Every remaining outer face is strictly visible, so each new tetrahedron is exactly
//! positive, and surface faces cannot be removed because the flood never crosses a ghost.

use crate::core::{FACE, GHOST};
use crate::error::TetraError;
use crate::flips::Flipper;
use crate::predicates::{in_sphere, orient};

/// A cavity larger than this is left uncarved past it; real cavities are a few dozen tetrahedra.
const MAX_CAVITY: usize = 512;

/// How far off the requested point an edge-ring split may land before it gives up: the exact
/// tests can refuse a point rounded just outside a tetrahedron that is nearly flat, and a point
/// a little along the edge is on the edge just the same.
const SPLIT_OFFSETS: [f64; 5] = [1.0, 0.875, 1.125, 0.75, 1.25];

impl Flipper {
    /// Splits edge `(p, q)` at `fraction` of its length from `p` by replacing the ring of
    /// tetrahedra around it, so nothing away from the edge changes. Only an edge with no ghost
    /// in its ring can be split, so the new point is strictly inside the solid; every new
    /// tetrahedron is checked exactly positive first. Returns the new vertex, or `None` when the
    /// ring refuses every offset of the point.
    pub(crate) fn split_edge(
        &mut self,
        p: u32,
        q: u32,
        fraction: f64,
    ) -> Result<Option<u32>, TetraError> {
        let Some((ring, cavity)) = self.ring_public(p, q) else {
            return Ok(None);
        };
        if ring.contains(&GHOST) {
            return Ok(None);
        }
        let (from, to) = (self.core.points[p as usize], self.core.points[q as usize]);
        for scale in SPLIT_OFFSETS {
            let t = fraction * scale;
            if t <= 0.0 || t >= 1.0 {
                continue;
            }
            let point = [
                from[0] + (to[0] - from[0]) * t,
                from[1] + (to[1] - from[1]) * t,
                from[2] + (to[2] - from[2]) * t,
            ];
            if point == from || point == to || self.first_unseen(&cavity, &point).is_some() {
                continue;
            }
            self.add_steiner(&cavity, point)?;
            // The vertex count stays below `u32::MAX`, as the input is capped far below it.
            return Ok(Some(self.core.points.len() as u32 - 1));
        }
        Ok(None)
    }

    /// The tetrahedra that must be replaced for `target`, by a walk from `start`: the containing
    /// one, the two sharing a face, or the ring of an edge. `None` when the walk is stopped by
    /// the surface or `target` is on the surface or at a vertex.
    pub(crate) fn locate_interior(&mut self, start: u32, target: [f64; 3]) -> Option<Vec<u32>> {
        let mut t = start;
        let mut state = 0x2545_F491_u32;
        for _ in 0..self.core.tets.len() + 16 {
            let tet = self.core.tets[t as usize];
            if tet.contains(&GHOST) {
                return None;
            }
            let sides: [f64; 4] = std::array::from_fn(|i| {
                let [a, b, c] = FACE[i].map(|position| &self.core.points[tet[position] as usize]);
                orient(a, b, c, &target)
            });
            state = state.wrapping_mul(0x9E37_79B1).wrapping_add(0x7F4A);
            let first = (state >> 30) as usize;
            let next = (0..4)
                .map(|k| (first + k) % 4)
                .find(|&i| sides[i] < 0.0 && !self.is_ghost(self.core.neighbors[t as usize][i]));
            if let Some(i) = next {
                t = self.core.neighbors[t as usize][i];
                continue;
            }
            if sides.iter().any(|&side| side < 0.0) {
                return None;
            }
            let zeros: Vec<usize> = (0..4).filter(|&i| sides[i] == 0.0).collect();
            return match zeros.as_slice() {
                [] => Some(vec![t]),
                &[i] => {
                    let n = self.core.neighbors[t as usize][i];
                    (!self.is_ghost(n)).then(|| vec![t, n])
                }
                &[i, j] => {
                    let mut ends = (0..4).filter(|&k| k != i && k != j).map(|k| tet[k]);
                    let (p, q) = (ends.next()?, ends.next()?);
                    let (ring, tets) = self.ring_public(p, q)?;
                    (!ring.contains(&GHOST)).then_some(tets)
                }
                _ => None,
            };
        }
        None
    }

    fn is_ghost(&self, t: u32) -> bool {
        self.core.tets[t as usize].contains(&GHOST)
    }

    /// Adds `point` by replacing a cavity grown from `seeds`, unless a vertex of the cavity lies
    /// within `gap`, or the point cannot see the cavity's boundary. Returns whether it was added.
    pub(crate) fn insert_interior(
        &mut self,
        seeds: &[u32],
        point: [f64; 3],
        gap: f64,
    ) -> Result<bool, TetraError> {
        let Some(cavity) = self.interior_cavity(seeds, point) else {
            return Ok(false);
        };
        if gap > 0.0 {
            for &t in &cavity {
                for &v in &self.core.tets[t as usize] {
                    let q = self.core.points[v as usize];
                    let d = [q[0] - point[0], q[1] - point[1], q[2] - point[2]];
                    if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < gap * gap {
                        return Ok(false);
                    }
                }
            }
        }
        self.add_steiner(&cavity, point)?;
        Ok(true)
    }

    /// The conflict region of `point` around `seeds`, cut down until `point` sees every outer
    /// face. `None` when it would have to give up a seed.
    fn interior_cavity(&mut self, seeds: &[u32], point: [f64; 3]) -> Option<Vec<u32>> {
        // `in_sphere` names points by index; the point is kept in the list only for the flood.
        // The count stays below `u32::MAX`, as the input is capped far below it.
        let index = self.core.points.len() as u32;
        self.core.points.push(point);
        let mut cavity = seeds.to_vec();
        let mut next = 0;
        while next < cavity.len() {
            let t = cavity[next];
            next += 1;
            for slot in 0..4 {
                let n = self.core.neighbors[t as usize][slot];
                if cavity.contains(&n) || self.is_ghost(n) || cavity.len() >= MAX_CAVITY {
                    continue;
                }
                if in_sphere(&self.core.points, self.core.tets[n as usize], index) {
                    cavity.push(n);
                }
            }
        }
        self.core.points.pop();
        while let Some(t) = self.first_unseen(&cavity, &point) {
            if seeds.contains(&t) {
                return None;
            }
            cavity.retain(|&c| c != t);
            cavity = self.connected_to_seeds(&cavity, seeds);
        }
        Some(cavity)
    }

    /// The first cavity tetrahedron with an outer face `point` does not see strictly.
    fn first_unseen(&self, cavity: &[u32], point: &[f64; 3]) -> Option<u32> {
        cavity.iter().copied().find(|&t| {
            let tet = self.core.tets[t as usize];
            FACE.iter().enumerate().any(|(i, positions)| {
                if cavity.contains(&self.core.neighbors[t as usize][i]) {
                    return false;
                }
                let [a, b, c] = positions.map(|position| &self.core.points[tet[position] as usize]);
                orient(a, b, c, point) <= 0.0
            })
        })
    }

    /// The part of `cavity` reachable from `seeds` through cavity faces.
    fn connected_to_seeds(&self, cavity: &[u32], seeds: &[u32]) -> Vec<u32> {
        let mut kept: Vec<u32> = seeds
            .iter()
            .copied()
            .filter(|s| cavity.contains(s))
            .collect();
        let mut next = 0;
        while next < kept.len() {
            let t = kept[next];
            next += 1;
            for slot in 0..4 {
                let n = self.core.neighbors[t as usize][slot];
                if cavity.contains(&n) && !kept.contains(&n) {
                    kept.push(n);
                }
            }
        }
        kept
    }
}
