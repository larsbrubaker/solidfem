//! The flips at a fold of the hull, split from `flips.rs` (which holds the ordinary face and edge
//! removals they reuse): where a hull edge's two hull faces are coplanar it can be removed like any
//! edge ([`Flipper::ghost_ring_is_flat`]), and where they fold, a surface edge that runs just
//! inside the fold is made by closing the ring through it and keeping one sliver under the fold
//! ([`Flipper::cap_hull_fold`]). `recover.rs` asks for these when a walk to a constraint is
//! blocked at the hull.

use crate::core::GHOST;
use crate::error::TetraError;
use crate::flips::{edge_key, Closure, Flipper, Target};
use crate::predicates::orient;

impl Flipper {
    /// The flip that makes target edge `(u, v)` under a fold of the hull. `(p, q)` is a hull edge
    /// whose two hull faces are not coplanar, `ring` is its ring rotated so the point at infinity
    /// is last, and the surface edge `(u, v)` joins the ring's two ends: it runs just inside the
    /// fold, and no sequence of ordinary flips can make it, because a hull edge that is not flat
    /// can never be removed and points added inside the solid leave the fold where it is. The ring's
    /// real tetrahedra are replaced by the chain closed through `(u, v)` and one sliver
    /// `(p, q, u, v)` that keeps the hull's fold, leaving the hull faces untouched; exact
    /// positive volume of every new tetrahedron makes the region the same.
    pub(crate) fn cap_hull_fold(
        &mut self,
        p: u32,
        q: u32,
        ring: &[u32],
        tets: &[u32],
        target: Target,
    ) -> Result<bool, TetraError> {
        let Target::Edge(u, v) = target else {
            return Ok(false);
        };
        let n = ring.len();
        let (first, last) = (ring[0], ring[n - 2]);
        if n < 4 || !((first == u && last == v) || (first == v && last == u)) {
            return Ok(false);
        }
        let sliver = [p, q, first, last];
        if !self.tet_positive(&sliver) {
            return Ok(false);
        }
        let Some(new) = self.triangulate_ring(p, q, &ring[..n - 1], Closure::Sliver, target) else {
            return Ok(false);
        };
        self.apply(&tets[..n - 2], &new)?;
        self.counts.larger += 1;
        Ok(true)
    }

    /// The hull edge of `face` that folds over surface edge `(u, v)`, its ring running between the
    /// two ends of that edge, if there is one.
    pub(crate) fn hull_fold_over(&mut self, face: [u32; 3], u: u32, v: u32) -> Option<(u32, u32)> {
        let edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])];
        edges.into_iter().find(|&(a, b)| {
            if self.locked_edges.contains(&edge_key(a, b)) {
                return false;
            }
            let Some((mut ring, _)) = self.ring(a, b) else {
                return false;
            };
            let Some(position) = ring.iter().position(|&w| w == GHOST) else {
                return false;
            };
            ring.rotate_left(position + 1);
            let n = ring.len();
            n >= 4
                && !self.ghost_ring_is_flat(a, b, &ring)
                && ((ring[0] == u && ring[n - 2] == v) || (ring[0] == v && ring[n - 2] == u))
        })
    }

    /// Tries [`Flipper::cap_hull_fold`] on the hull edge `(a, b)`, for a walk blocked by a face
    /// that contains it. Returns whether the edge `(u, v)` now exists under the fold.
    pub(crate) fn cap_fold_of(
        &mut self,
        a: u32,
        b: u32,
        u: u32,
        v: u32,
    ) -> Result<bool, TetraError> {
        if self.locked_edges.contains(&edge_key(a, b)) {
            return Ok(false);
        }
        let Some((mut ring, mut tets)) = self.ring(a, b) else {
            return Ok(false);
        };
        let Some(position) = ring.iter().position(|&w| w == GHOST) else {
            return Ok(false);
        };
        ring.rotate_left(position + 1);
        tets.rotate_left(position + 1);
        if self.ghost_ring_is_flat(a, b, &ring) {
            return Ok(false);
        }
        self.cap_hull_fold(a, b, &ring, &tets, Target::Edge(u, v))
    }

    /// Whether a hull edge can be removed: `ring` ends with the point at infinity, the two hull
    /// faces at the edge lie exactly in one plane, and the two new hull faces keep the mesh on
    /// their inner side.
    pub(crate) fn ghost_ring_is_flat(&self, p: u32, q: u32, ring: &[u32]) -> bool {
        let n = ring.len();
        if n < 4 {
            return false;
        }
        let (first, last) = (ring[0], ring[n - 2]);
        let point = |v: u32| &self.core.points[v as usize];
        if orient(point(p), point(q), point(first), point(last)) != 0.0 {
            return false;
        }
        // A real vertex of the ring off that plane tells which side the mesh is on.
        let reference = ring[1..n - 2]
            .iter()
            .copied()
            .find(|&v| orient(point(p), point(q), point(first), point(v)) != 0.0);
        let Some(reference) = reference else {
            return false;
        };
        orient(point(first), point(last), point(p), point(reference)) < 0.0
            && orient(point(last), point(first), point(q), point(reference)) < 0.0
    }
}
