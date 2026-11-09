//! The flips surface recovery is made of: a tetrahedron-to-tetrahedra replacement that keeps every
//! tetrahedron strictly positive by exact test, in two shapes. A *face removal* replaces the two
//! tetrahedra sharing a face with three around the edge joining their apexes (2 to 3). An *edge
//! removal* replaces the ring of `n` tetrahedra around an edge with `2(n - 2)` tetrahedra, one
//! pair per triangle of some triangulation of the ring of vertices around the edge (3 to 2,
//! 4 to 4 and larger). `recover.rs` decides what to flip; this file decides whether a flip is
//! legal and carries it out through `Core::replace`.
//!
//! A flip is legal only when every new tetrahedron has strictly positive exact orientation, so
//! the region removed and the region created coincide and no flat tetrahedron ever exists. An edge
//! on the convex hull is the one special case: its ring contains the vertex at infinity, and it
//! can be removed only if the two hull faces at the edge lie exactly in one plane (a flip within a
//! flat hull facet). Then the hull is unchanged and the ring's real part is triangulated like any
//! other ring.
//!
//! The ring triangulation is chosen by dynamic programming over intervals of the ring, so every
//! legal triangulation is considered; among those the cost picks the one best for the current
//! constraint (fewest new edges through a target triangle, or the target edge itself).
//!
//! When no triangulation of a ring is legal, the edges around it are flipped first, up to
//! [`RELIEF_DEPTH`] levels, because a ring is shaped by its neighbours. Those flips stand even if
//! the edge then stays: each is legal, so the mesh is valid whichever way it ends.
//!
//! One more flip keeps a hull edge whose two hull faces fold ([`Flipper::cap_hull_fold`]): the
//! ring's real tetrahedra are replaced by the chain closed through the wanted surface edge plus one
//! sliver under the fold, leaving the hull exactly as it is.
//!
//! Edges and faces that belong to the input surface are locked: no flip may destroy them. Edges
//! are removed only by an edge removal and faces only by a face removal or by the removal of one
//! of their edges, so those are the only places locks are checked.

use std::collections::HashSet;

use crate::core::{Core, FACE, GHOST};
use crate::error::TetraError;
use crate::patches::{project, Patch};
use crate::predicates::orient;

/// Rings larger than this are not attempted: the interval search is cubic in the ring, and rings
/// this large do not occur around an edge of a surface triangulation's Delaunay mesh in practice.
const MAX_RING: usize = 32;

/// `vertex_tet` of a point a later point's cavity swallowed: it is in no tetrahedron.
const NO_TET: u32 = u32::MAX;

/// A cost no legal choice reaches, meaning no triangulation of an interval exists.
const NO_TRIANGULATION: i32 = i32::MAX / 4;

/// How many levels of neighbouring edges an edge removal will flip first to open its ring.
const RELIEF_DEPTH: u32 = 3;

/// How the ring triangulation is closed along the chord joining the chain's two ends.
enum Closure {
    /// The chain goes all the way round the edge; that chord is a side of the ring.
    Ring,
    /// The chain ends at a hull edge whose two hull faces are coplanar, split along the chord.
    FlatHull,
    /// The chain ends at a hull edge whose two hull faces fold, which a sliver tetrahedron keeps.
    Sliver,
}

/// What an edge removal should favour.
#[derive(Clone, Copy)]
pub(crate) enum Target {
    /// Prefer creating this exact edge.
    Edge(u32, u32),
    /// Refuse new edges through the interior of this triangle.
    Face([u32; 3]),
    /// Refuse new edges through the region of this flat patch.
    Patch(u32),
}

/// How many flips of each kind have been made.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FlipCounts {
    pub(crate) two_three: usize,
    pub(crate) three_two: usize,
    pub(crate) four_four: usize,
    pub(crate) larger: usize,
    /// Edges inside a flat patch flipped towards its Delaunay triangulation (`plane_delaunay.rs`).
    pub(crate) in_plane_flips: usize,
    /// Inner patch edges that fail the in-circle test but had no legal flip in the end.
    pub(crate) in_plane_flips_refused: usize,
}

/// The key of an undirected edge.
pub(crate) fn edge_key(a: u32, b: u32) -> u64 {
    (u64::from(a.min(b)) << 32) | u64::from(a.max(b))
}

/// The key of a face, independent of its vertex order.
pub(crate) fn face_key(a: u32, b: u32, c: u32) -> [u32; 3] {
    let mut face = [a, b, c];
    face.sort_unstable();
    face
}

/// The triangulation being flipped, with what the flips need to answer "what is around here".
pub(crate) struct Flipper {
    pub(crate) core: Core,
    /// One real or ghost tetrahedron incident to each vertex.
    vertex_tet: Vec<u32>,
    visited: Vec<u32>,
    generation: u32,
    pub(crate) locked_edges: HashSet<u64>,
    pub(crate) locked_faces: HashSet<[u32; 3]>,
    pub(crate) counts: FlipCounts,
    /// The flat patches of the surface, and the mesh faces each ended up covered by.
    pub(crate) patches: Vec<Patch>,
    pub(crate) patch_faces: Vec<Vec<[u32; 3]>>,
    /// How many levels of neighbouring-edge relief an edge removal may use. Set to zero for a stuck
    /// patch's retry after a point and for the flip of the chord a point was placed for, where
    /// relief could flip the new point's star away again; a stuck edge's walks use it in full.
    pub(crate) relief_depth: u32,
    /// How many points have ever been added inside the solid, swallowed ones included: the
    /// input's vertices are the points before these, and the limit on points counts them.
    pub(crate) steiner_points: usize,
    /// Added points a later point's cavity took whole, now in no tetrahedron.
    pub(crate) swallowed_points: usize,
}

/// A face removal found to be illegal, with the edges of the shared face that block it.
pub(crate) struct Blocked {
    pub(crate) edges: Vec<(u32, u32)>,
}

impl Flipper {
    pub(crate) fn new(
        core: Core,
        locked_edges: HashSet<u64>,
        locked_faces: HashSet<[u32; 3]>,
        patches: Vec<Patch>,
    ) -> Self {
        let mut vertex_tet = vec![0; core.points.len()];
        for (slot, tet) in core.tets.iter().enumerate() {
            for &vertex in tet {
                if (vertex as usize) < vertex_tet.len() {
                    // `slot` fits u32: the triangulation counts its tetrahedra in u32.
                    vertex_tet[vertex as usize] = slot as u32;
                }
            }
        }
        Self {
            core,
            vertex_tet,
            visited: Vec::new(),
            generation: 0,
            locked_edges,
            locked_faces,
            counts: FlipCounts::default(),
            patch_faces: vec![Vec::new(); patches.len()],
            patches,
            relief_depth: RELIEF_DEPTH,
            steiner_points: 0,
            swallowed_points: 0,
        }
    }

    /// Every tetrahedron containing vertex `v`, including ghosts, in a fixed order; none for a
    /// point a later cavity swallowed.
    pub(crate) fn tets_around(&mut self, v: u32) -> Vec<u32> {
        if self.vertex_tet[v as usize] == NO_TET {
            return Vec::new();
        }
        if self.visited.len() < self.core.tets.len() {
            self.visited.resize(self.core.tets.len(), 0);
        }
        self.generation += 1;
        let mut found = vec![self.vertex_tet[v as usize]];
        self.visited[found[0] as usize] = self.generation;
        let mut next = 0;
        while next < found.len() {
            let t = found[next];
            next += 1;
            for i in 0..4 {
                if self.core.tets[t as usize][i] == v {
                    continue;
                }
                let n = self.core.neighbors[t as usize][i];
                if self.visited[n as usize] != self.generation {
                    self.visited[n as usize] = self.generation;
                    found.push(n);
                }
            }
        }
        found
    }

    /// A tetrahedron containing edge `(p, q)`, if the edge exists.
    pub(crate) fn find_edge(&mut self, p: u32, q: u32) -> Option<u32> {
        self.tets_around(p)
            .into_iter()
            .find(|&t| self.core.tets[t as usize].contains(&q))
    }

    /// Whether triangle `(a, b, c)` is a face of the mesh.
    pub(crate) fn face_exists(&mut self, a: u32, b: u32, c: u32) -> bool {
        let around = self.tets_around(a);
        around.iter().any(|&t| {
            let tet = &self.core.tets[t as usize];
            tet.contains(&b) && tet.contains(&c)
        })
    }

    /// The vertices of the ring around edge `(p, q)` and the tetrahedra between them, with
    /// `(p, q, w[i], w[i + 1])` an even permutation of tetrahedron `i`'s vertices, so each is
    /// positive when real. `None` when the edge is absent.
    pub(crate) fn ring_public(&mut self, p: u32, q: u32) -> Option<(Vec<u32>, Vec<u32>)> {
        self.ring(p, q)
    }

    fn ring(&mut self, p: u32, q: u32) -> Option<(Vec<u32>, Vec<u32>)> {
        let start = self.find_edge(p, q)?;
        let mut ring = Vec::new();
        let mut tets = Vec::new();
        let mut t = start;
        loop {
            let (x, y) = frame(&self.core.tets[t as usize], p, q);
            ring.push(x);
            tets.push(t);
            let tet = self.core.tets[t as usize];
            let opposite = tet.iter().position(|&v| v == x)?;
            t = self.core.neighbors[t as usize][opposite];
            let (nx, _) = frame(&self.core.tets[t as usize], p, q);
            if nx != y {
                return None;
            }
            if t == start {
                return Some((ring, tets));
            }
        }
    }

    pub(crate) fn apply(&mut self, old: &[u32], new: &[[u32; 4]]) -> Result<(), TetraError> {
        let slots = self.core.replace(old, new)?;
        for (slot, tet) in slots.iter().zip(new) {
            for &vertex in tet {
                if vertex != GHOST {
                    self.vertex_tet[vertex as usize] = *slot;
                }
            }
        }
        Ok(())
    }

    /// Removes the face shared by `t` and its neighbour across slot `slot` (a 2 to 3 flip),
    /// joining the two apexes. Returns the blocking edges of the face when illegal.
    pub(crate) fn remove_face(
        &mut self,
        t: u32,
        slot: usize,
    ) -> Result<Result<(), Blocked>, TetraError> {
        let tet = self.core.tets[t as usize];
        let u = tet[slot];
        let n = self.core.neighbors[t as usize][slot];
        let other = self.core.tets[n as usize];
        let Some(y_slot) = (0..4).find(|&j| !tet.contains(&other[j])) else {
            return Err(TetraError::CavityInconsistent { point: u });
        };
        let y = other[y_slot];
        // The face wound so its vertices appear counterclockwise seen from `u`, so the three new
        // tetrahedra read `(u, y, a, b)` around the new edge.
        let face = FACE[slot].map(|position| tet[position]);
        let (a, b, c) = (face[1], face[0], face[2]);
        if u == GHOST
            || y == GHOST
            || self.locked_faces.contains(&face_key(a, b, c))
            || [a, b, c].contains(&GHOST)
        {
            return Ok(Err(Blocked { edges: Vec::new() }));
        }
        let new = [[u, y, a, b], [u, y, b, c], [u, y, c, a]];
        let blocked: Vec<(u32, u32)> = [(a, b), (b, c), (c, a)]
            .into_iter()
            .zip(new.iter())
            .filter(|(_, tet)| !self.tet_positive(tet))
            .map(|(edge, _)| edge)
            .collect();
        if !blocked.is_empty() {
            return Ok(Err(Blocked { edges: blocked }));
        }
        self.apply(&[t, n], &new)?;
        self.counts.two_three += 1;
        Ok(Ok(()))
    }

    /// Strictly positive by exact test; a tetrahedron containing the point at infinity is not.
    pub(crate) fn tet_positive(&self, tet: &[u32; 4]) -> bool {
        if tet.contains(&GHOST) {
            return false;
        }
        let [a, b, c, d] = tet.map(|v| &self.core.points[v as usize]);
        orient(a, b, c, d) > 0.0
    }

    /// Removes edge `(p, q)` if it is not locked and a legal retriangulation of its ring exists
    /// with no new edge through a target triangle, first flipping neighbouring edges when that is
    /// what opens a legal one. Returns whether it was removed.
    pub(crate) fn remove_edge(
        &mut self,
        p: u32,
        q: u32,
        target: Target,
    ) -> Result<bool, TetraError> {
        self.remove_edge_within(p, q, target, self.relief_depth)
    }

    /// One edge removal attempt, then, for up to `depth` levels, attempts on the edges from the
    /// two ends to the ring: a ring is shaped by the tetrahedra around it, and flipping a
    /// neighbouring edge reshapes the ring without moving a vertex.
    fn remove_edge_within(
        &mut self,
        p: u32,
        q: u32,
        target: Target,
        depth: u32,
    ) -> Result<bool, TetraError> {
        if self.remove_edge_once(p, q, target)? {
            return Ok(true);
        }
        if depth == 0 {
            return Ok(false);
        }
        let Some((ring, _)) = self.ring(p, q) else {
            return Ok(false);
        };
        if ring.len() > MAX_RING {
            return Ok(false);
        }
        let mut relief: Vec<(u32, u32)> = Vec::with_capacity(3 * ring.len());
        for (i, &w) in ring.iter().enumerate() {
            if w == GHOST {
                continue;
            }
            // The ring side first: it is the edge a ring triangle has to be built across.
            let next = ring[(i + 1) % ring.len()];
            if next != GHOST {
                relief.push((w, next));
            }
            relief.push((p, w));
            relief.push((q, w));
        }
        for (a, b) in relief {
            if self.remove_edge_within(a, b, target, depth - 1)?
                && self.remove_edge_once(p, q, target)?
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn remove_edge_once(&mut self, p: u32, q: u32, target: Target) -> Result<bool, TetraError> {
        if self.locked_edges.contains(&edge_key(p, q)) {
            return Ok(false);
        }
        let Some((mut ring, mut tets)) = self.ring(p, q) else {
            return Ok(false);
        };
        let n = ring.len();
        if n > MAX_RING {
            return Ok(false);
        }
        let ghost = ring.iter().position(|&v| v == GHOST);
        if let Some(position) = ghost {
            // Rotate the ring so the point at infinity is last.
            ring.rotate_left(position + 1);
            tets.rotate_left(position + 1);
            if !self.ghost_ring_is_flat(p, q, &ring) {
                return self.cap_hull_fold(p, q, &ring, &tets, target);
            }
        }
        let chain = if ghost.is_some() { n - 1 } else { n };
        if chain < 2 {
            return Ok(false);
        }
        let closure = if ghost.is_some() {
            Closure::FlatHull
        } else {
            Closure::Ring
        };
        let Some(new) = self.triangulate_ring(p, q, &ring[..chain], closure, target) else {
            return Ok(false);
        };
        self.apply(&tets, &new)?;
        match n {
            3 => self.counts.three_two += 1,
            4 => self.counts.four_four += 1,
            _ => self.counts.larger += 1,
        }
        Ok(true)
    }

    /// Flips inner edge `(p, q)` of a flat patch to the other diagonal `(b, d)` of the quadrilateral
    /// of its two in-plane faces, keeping every ring vertex. The ring splits at `b` and `d` into
    /// its two sides, one each side of the plane, and each side is triangulated with `(b, d)` as
    /// the closing chord; on the hull, where one side is the point at infinity, the ordinary
    /// hull-facet flip closes along the same chord. Returns whether the edge was flipped; `false`
    /// leaves the mesh exactly as it was: the edge is locked or a new tetrahedron failed the
    /// exact test even after the off-plane edges around the ring were flipped to reshape it.
    pub(crate) fn flip_patch_edge(
        &mut self,
        p: u32,
        q: u32,
        b: u32,
        d: u32,
        target: Target,
    ) -> Result<bool, TetraError> {
        if self.flip_patch_edge_once(p, q, b, d, target)? {
            return Ok(true);
        }
        let Some((ring, _)) = self.ring(p, q) else {
            return Ok(false);
        };
        // Every edge here has an end off the plane, so flipping it cannot disturb the patch's
        // in-plane faces, which are what the caller is keeping track of.
        let in_plane = [p, q, b, d];
        let mut relief: Vec<(u32, u32)> = Vec::with_capacity(3 * ring.len());
        for (i, &w) in ring.iter().enumerate() {
            let next = ring[(i + 1) % ring.len()];
            if w != GHOST && next != GHOST && !(in_plane.contains(&w) && in_plane.contains(&next)) {
                relief.push((w, next));
            }
            if w != GHOST && !in_plane.contains(&w) {
                relief.push((p, w));
                relief.push((q, w));
            }
        }
        for (x, y) in relief {
            if self.remove_edge_once(x, y, target)?
                && self.flip_patch_edge_once(p, q, b, d, target)?
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn flip_patch_edge_once(
        &mut self,
        p: u32,
        q: u32,
        b: u32,
        d: u32,
        target: Target,
    ) -> Result<bool, TetraError> {
        if self.locked_edges.contains(&edge_key(p, q)) {
            return Ok(false);
        }
        let Some((mut ring, mut tets)) = self.ring(p, q) else {
            return Ok(false);
        };
        let n = ring.len();
        if n > MAX_RING {
            return Ok(false);
        }
        if let Some(position) = ring.iter().position(|&w| w == GHOST) {
            ring.rotate_left(position + 1);
            let ends = [ring[0], ring[n - 2]];
            return if ends == [b, d] || ends == [d, b] {
                self.remove_edge_once(p, q, target)
            } else {
                Ok(false)
            };
        }
        let Some(start) = ring.iter().position(|&w| w == b) else {
            return Ok(false);
        };
        ring.rotate_left(start);
        tets.rotate_left(start);
        let Some(split) = ring.iter().position(|&w| w == d) else {
            return Ok(false);
        };
        // Two ring vertices side by side share a tetrahedron with the edge, which cannot be when
        // all four points lie in one plane; refusing here is only a guard.
        if split < 2 || n - split < 2 {
            return Ok(false);
        }
        let mut far_side = ring[split..].to_vec();
        far_side.push(b);
        let Some(mut new) = self.triangulate_ring(p, q, &ring[..=split], Closure::Ring, target)
        else {
            return Ok(false);
        };
        let Some(rest) = self.triangulate_ring(p, q, &far_side, Closure::Ring, target) else {
            return Ok(false);
        };
        new.extend_from_slice(&rest);
        self.apply(&tets, &new)?;
        match n {
            3 => self.counts.three_two += 1,
            4 => self.counts.four_four += 1,
            _ => self.counts.larger += 1,
        }
        Ok(true)
    }

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
    fn ghost_ring_is_flat(&self, p: u32, q: u32, ring: &[u32]) -> bool {
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

    /// The best triangulation of the ring chain `ring[0..=m]`, as new tetrahedra, or `None`.
    fn triangulate_ring(
        &self,
        p: u32,
        q: u32,
        chain: &[u32],
        closure: Closure,
        target: Target,
    ) -> Option<Vec<[u32; 4]>> {
        let m = chain.len() - 1;
        let width = m + 1;
        let mut best = vec![NO_TRIANGULATION; width * width];
        let mut split = vec![0usize; width * width];
        for i in 0..m {
            best[i * width + i + 1] = 0;
        }
        for length in 2..=m {
            for i in 0..=m - length {
                let j = i + length;
                let chord = self.chord_cost(chain[i], chain[j], target);
                for k in i + 1..j {
                    let left = best[i * width + k];
                    let right = best[k * width + j];
                    if left >= NO_TRIANGULATION || right >= NO_TRIANGULATION {
                        continue;
                    }
                    let pair = self.ring_pair(p, q, chain[i], chain[k], chain[j]);
                    if !self.tet_positive(&pair[0]) || !self.tet_positive(&pair[1]) {
                        continue;
                    }
                    // The chord closing the interval is new, unless it is a side of the ring.
                    let closing = if i == 0 && j == m && matches!(closure, Closure::Ring) {
                        0
                    } else {
                        chord
                    };
                    let cost = left + right + closing;
                    if cost < best[i * width + j] {
                        best[i * width + j] = cost;
                        split[i * width + j] = k;
                    }
                }
            }
        }
        let total = best[m];
        if total >= NO_TRIANGULATION {
            return None;
        }
        if matches!(target, Target::Face(_) | Target::Patch(_)) && total > 0 {
            return None;
        }
        let mut tets = Vec::with_capacity(2 * (m - 1) + 2);
        let mut stack = vec![(0usize, m)];
        while let Some((i, j)) = stack.pop() {
            if j - i < 2 {
                continue;
            }
            let k = split[i * width + j];
            tets.extend_from_slice(&self.ring_pair(p, q, chain[i], chain[k], chain[j]));
            stack.push((i, k));
            stack.push((k, j));
        }
        match closure {
            Closure::Ring => {}
            Closure::FlatHull => {
                // The hull keeps its two flat faces, now split along the chord closing the chain.
                tets.push([chain[0], GHOST, chain[m], p]);
                tets.push([chain[0], chain[m], GHOST, q]);
            }
            // The hull's fold is kept by the sliver between it and the closing chord.
            Closure::Sliver => tets.push([p, q, chain[0], chain[m]]),
        }
        Some(tets)
    }

    /// The two tetrahedra of ring triangle `(i, k, j)`, one to each end of the edge, wound so
    /// positive means valid: `(i, k, j)` runs the same way as the ring.
    fn ring_pair(&self, p: u32, q: u32, i: u32, k: u32, j: u32) -> [[u32; 4]; 2] {
        [[i, j, k, p], [i, k, j, q]]
    }

    fn chord_cost(&self, x: u32, y: u32, target: Target) -> i32 {
        match target {
            Target::Edge(u, v) => {
                if (x == u && y == v) || (x == v && y == u) {
                    -1
                } else {
                    0
                }
            }
            Target::Face(face) => i32::from(self.crosses_face(x, y, face)),
            Target::Patch(id) => i32::from(self.crosses_patch(x, y, id as usize)),
        }
    }

    /// Whether mesh edge `(x, y)` pierces patch `id`: its ends strictly on opposite sides of the
    /// plane and the piercing point within a triangle of the patch, boundary included (a point
    /// on an internal edge of the input triangulation is still inside the flat region).
    pub(crate) fn crosses_patch(&self, x: u32, y: u32, id: usize) -> bool {
        let point = |v: u32| &self.core.points[v as usize];
        let patch = &self.patches[id];
        let [a, b, c] = patch.triangles[0].map(point);
        let (sx, sy) = (orient(a, b, c, point(x)), orient(a, b, c, point(y)));
        if !((sx > 0.0 && sy < 0.0) || (sx < 0.0 && sy > 0.0)) {
            return false;
        }
        let Some(index) = &patch.index else {
            return self.pierces_triangle(x, y, patch.triangles[0]);
        };
        let (px, py) = (
            project(point(x), index.axes()),
            project(point(y), index.axes()),
        );
        let low = [px[0].min(py[0]), px[1].min(py[1])];
        let high = [px[0].max(py[0]), px[1].max(py[1])];
        index
            .candidates(low, high)
            .into_iter()
            .any(|t| self.pierces_triangle(x, y, patch.triangles[t as usize]))
    }

    /// Whether line `(x, y)` meets the closed triangle: the three edge sides do not disagree.
    fn pierces_triangle(&self, x: u32, y: u32, triangle: [u32; 3]) -> bool {
        let point = |v: u32| &self.core.points[v as usize];
        let [a, b, c] = triangle.map(point);
        let sides = [
            orient(point(x), point(y), a, b),
            orient(point(x), point(y), b, c),
            orient(point(x), point(y), c, a),
        ];
        sides.iter().all(|&s| s >= 0.0) || sides.iter().all(|&s| s <= 0.0)
    }

    /// Adds `point` as a new vertex by replacing the tetrahedra `cavity`, which must be
    /// star-shaped around it by exact test, with one tetrahedron per boundary face.
    pub(crate) fn add_steiner(
        &mut self,
        cavity: &[u32],
        point: [f64; 3],
    ) -> Result<(), TetraError> {
        // The vertex count stays well under `u32::MAX`, as the input is capped far below it.
        let index = self.core.points.len() as u32;
        self.core.points.push(point);
        self.vertex_tet.push(0);
        let mut new = Vec::new();
        for &t in cavity {
            let tet = self.core.tets[t as usize];
            for (i, positions) in FACE.iter().enumerate() {
                let n = self.core.neighbors[t as usize][i];
                if cavity.contains(&n) {
                    continue;
                }
                let face = positions.map(|position| tet[position]);
                new.push([face[0], face[1], face[2], index]);
            }
        }
        let mut before: Vec<u32> = cavity
            .iter()
            .flat_map(|&t| self.core.tets[t as usize])
            .filter(|&vertex| vertex != GHOST)
            .collect();
        before.sort_unstable();
        before.dedup();
        self.apply(cavity, &new)?;
        self.steiner_points += 1;
        // A point whose whole star was in the cavity is in no tetrahedron now: nothing may reach
        // it through a slot that has since been freed or reused.
        let kept: HashSet<u32> = new.iter().flatten().copied().collect();
        for vertex in before {
            if !kept.contains(&vertex) {
                self.vertex_tet[vertex as usize] = NO_TET;
                self.swallowed_points += 1;
            }
        }
        Ok(())
    }

    /// How many added points are still in the mesh.
    pub(crate) fn kept_steiner_points(&self) -> usize {
        self.steiner_points - self.swallowed_points
    }

    /// Whether segment `(x, y)` passes through the interior of the triangle, by exact tests: its
    /// ends strictly on opposite sides of the plane, and the plane's crossing strictly inside.
    pub(crate) fn crosses_face(&self, x: u32, y: u32, face: [u32; 3]) -> bool {
        let point = |v: u32| &self.core.points[v as usize];
        let [a, b, c] = face.map(point);
        let (sx, sy) = (orient(a, b, c, point(x)), orient(a, b, c, point(y)));
        if !((sx > 0.0 && sy < 0.0) || (sx < 0.0 && sy > 0.0)) {
            return false;
        }
        let sides = [
            orient(point(x), point(y), a, b),
            orient(point(x), point(y), b, c),
            orient(point(x), point(y), c, a),
        ];
        sides.iter().all(|&s| s > 0.0) || sides.iter().all(|&s| s < 0.0)
    }
}

/// The two other vertices `(x, y)` of `tet` such that `(p, q, x, y)` is an even permutation of it.
fn frame(tet: &[u32; 4], p: u32, q: u32) -> (u32, u32) {
    let others: Vec<usize> = (0..4).filter(|&i| tet[i] != p && tet[i] != q).collect();
    let position = |v: u32| tet.iter().position(|&w| w == v).unwrap_or(0);
    let permutation = [position(p), position(q), others[0], others[1]];
    let mut inversions = 0;
    for a in 0..4 {
        for b in a + 1..4 {
            if permutation[a] > permutation[b] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 {
        (tet[others[0]], tet[others[1]])
    } else {
        (tet[others[1]], tet[others[0]])
    }
}
