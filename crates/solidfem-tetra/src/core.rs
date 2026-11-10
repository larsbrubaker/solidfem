//! The triangulation itself: tetrahedra with adjacency, the walk that locates a point, and
//! insertion by cavity. `triangulator.rs` drives it point by point; `predicates.rs` decides every
//! geometric question exactly.
//!
//! **The point at infinity.** The triangulation is closed with one extra vertex, [`GHOST`], a
//! point at infinity: every convex hull face `a b c` carries a ghost tetrahedron `a b c GHOST`.
//! Every tetrahedron then has four neighbours and insertion outside the hull is the same code as
//! insertion inside it. No coordinate of a bounding structure is ever invented, so no vertex has
//! to be removed and nothing round-off sized is left behind on the hull. Ghosts never reach a
//! caller: `triangulator.rs` drops them.
//!
//! **Orientation.** Every stored tetrahedron has positive [`orient`] (ghosts by construction:
//! insertion preserves the parity of the combinatorics). `neighbors[t][i]` is the tetrahedron
//! across the face opposite `tets[t][i]`, and [`FACE`] lists each face so its own vertex is on
//! the positive side.
//!
//! **A ghost's conflict test.** A point strictly outside the hull plane of `a b c` conflicts with
//! the ghost, strictly inside does not. A point exactly in that plane is the limit case: as the
//! vertex at infinity is brought in from a very large finite height the ghost's sphere tends to
//! the plane everywhere except inside the circumcircle of `a b c`, so such a point conflicts
//! exactly when it is inside that circle, under the same tie rule. The real neighbour across the
//! face has precisely that in-sphere answer for points in the plane: its circumsphere cuts the
//! plane in that circle, and the perturbation cofactor of its fourth vertex vanishes because the
//! other four are coplanar. So a ghost defers to it. That one rule is what keeps coplanar hull
//! points triangulated in the plane, and it means the horizon never holds a face coplanar with
//! the new point, so every new tetrahedron has non-zero exact volume.
//!
//! **Why insertion is correct under degeneracy.** With the tie rule, the input is effectively in
//! general position in the lifted space, so the conflict region of a new point is a topological
//! ball and the horizon is a closed surface. The cavity's boundary is checked as it is rebuilt,
//! and a broken invariant returns [`TetraError::CavityInconsistent`] rather than a wrong mesh.
//!
//! Storage is flat arrays with slot reuse, in an order fixed by the insertion sequence alone, so a
//! run's output is the same however it is sliced.

use crate::error::TetraError;
use crate::predicates::{in_sphere, orient};

/// The vertex at infinity, in place of a point index.
pub(crate) const GHOST: u32 = u32::MAX;
/// Marks a freed tetrahedron slot in `tets`.
pub(crate) const DEAD: u32 = u32::MAX - 1;

/// Vertex positions of the face opposite vertex `i`, ordered so vertex `i` is on the positive side
/// of the face under [`orient`] when the tetrahedron is positive.
pub(crate) const FACE: [[usize; 3]; 4] = [[1, 3, 2], [0, 2, 3], [1, 0, 3], [0, 1, 2]];

/// A horizon face of a cavity, kept with what is needed to rebuild after its old tetrahedra go.
#[derive(Clone, Copy)]
struct Horizon {
    /// Vertices of the face, with the new point on their positive side.
    face: [u32; 3],
    /// The surviving tetrahedron across the face.
    outer: u32,
    /// Which neighbour slot of `outer` points back into the cavity.
    outer_slot: u8,
}

/// The triangulation being built, with its working buffers. Every buffer is reused across
/// insertions so a run allocates only as the mesh grows.
pub(crate) struct Core {
    pub(crate) points: Vec<[f64; 3]>,
    pub(crate) tets: Vec<[u32; 4]>,
    pub(crate) neighbors: Vec<[u32; 4]>,
    /// Stamp per tetrahedron for the cavity search: `generation * 2`, plus one if in conflict.
    stamps: Vec<u32>,
    free: Vec<u32>,
    generation: u32,
    /// Where the next walk starts: a tetrahedron made by the previous insertion.
    hint: u32,
    /// Advances the face order of the walk; deterministic, so a run replays exactly.
    walk_state: u32,
    stack: Vec<u32>,
    cavity: Vec<u32>,
    horizon: Vec<Horizon>,
    created: Vec<u32>,
    edges: Vec<(u64, u32, u8)>,
}

/// The three vertices of `tet` other than slot `i`, sorted: a key naming the face from either
/// side.
fn sorted_face(tet: &[u32; 4], i: usize) -> [u32; 3] {
    let mut face = [0; 3];
    let mut count = 0;
    for (slot, &vertex) in tet.iter().enumerate() {
        if slot != i {
            face[count] = vertex;
            count += 1;
        }
    }
    face.sort_unstable();
    face
}

/// What locating a point found.
enum Located {
    /// A tetrahedron in conflict with the point.
    Tet(u32),
    /// The point has the same coordinates as this vertex.
    Duplicate(u32),
}

impl Core {
    /// A triangulation of the four non-coplanar points `simplex` of `points`, with its four ghosts.
    pub(crate) fn new(points: Vec<[f64; 3]>, simplex: [u32; 4]) -> Result<Self, TetraError> {
        let [a, b, c, d] = simplex.map(|v| &points[v as usize]);
        let [p0, p1, p2, p3] = simplex;
        let real = if orient(a, b, c, d) > 0.0 {
            [p0, p1, p2, p3]
        } else {
            [p1, p0, p2, p3]
        };
        let mut core = Self {
            points,
            tets: vec![real],
            neighbors: vec![[0; 4]],
            stamps: vec![0],
            free: Vec::new(),
            generation: 0,
            hint: 0,
            walk_state: 1,
            stack: Vec::new(),
            cavity: Vec::new(),
            horizon: Vec::new(),
            created: Vec::new(),
            edges: Vec::new(),
        };
        for (i, face) in FACE.iter().enumerate() {
            // The ghost lies on the far side of the face from the real tetrahedron's vertex `i`,
            // so its base is the same face wound the other way.
            let base = [real[face[1]], real[face[0]], real[face[2]]];
            let ghost = core.place([base[0], base[1], base[2], GHOST], [0; 4]);
            core.neighbors[0][i] = ghost;
            core.neighbors[ghost as usize][3] = 0;
            core.created.push(ghost);
        }
        core.link_created(GHOST)?;
        core.created.clear();
        Ok(core)
    }

    /// Rebuilds a triangulation from a finished solid's right-handed tetrahedra (`tets`,
    /// `neighbors`, the public output order), closing each boundary face with a ghost so a
    /// refinement stage can edit the mesh through the ordinary operations. Every boundary
    /// face has a ghost and two ghosts meet across each boundary edge, which is exactly what a
    /// closed manifold surface gives.
    pub(crate) fn from_solid(
        points: Vec<[f64; 3]>,
        tets: &[[u32; 4]],
        neighbors: &[[u32; 4]],
    ) -> Result<Self, TetraError> {
        let internal: Vec<[u32; 4]> = tets.iter().map(|t| [t[1], t[0], t[2], t[3]]).collect();
        let adjacency: Vec<[u32; 4]> = neighbors.iter().map(|n| [n[1], n[0], n[2], n[3]]).collect();
        let count = internal.len();
        let mut core = Self {
            points,
            tets: internal,
            neighbors: adjacency,
            stamps: vec![0; count],
            free: Vec::new(),
            generation: 0,
            hint: 0,
            walk_state: 1,
            stack: Vec::new(),
            cavity: Vec::new(),
            horizon: Vec::new(),
            created: Vec::new(),
            edges: Vec::new(),
        };
        for slot in 0..count {
            for (i, face) in FACE.iter().enumerate() {
                if core.neighbors[slot][i] != GHOST {
                    continue;
                }
                let real = core.tets[slot];
                let base = [real[face[1]], real[face[0]], real[face[2]]];
                let ghost = core.place([base[0], base[1], base[2], GHOST], [0; 4]);
                core.neighbors[slot][i] = ghost;
                // `slot` fits u32: the solid's tetrahedra are indexed by u32 already.
                core.neighbors[ghost as usize][3] = slot as u32;
                core.created.push(ghost);
            }
        }
        core.link_created(GHOST)?;
        core.created.clear();
        Ok(core)
    }

    /// A slot for a new tetrahedron, reusing a freed one first.
    fn place(&mut self, tet: [u32; 4], neighbors: [u32; 4]) -> u32 {
        if let Some(slot) = self.free.pop() {
            self.tets[slot as usize] = tet;
            self.neighbors[slot as usize] = neighbors;
            slot
        } else {
            // The count stays well under `u32::MAX`: at most the ~7n tets of a 3D Delaunay
            // triangulation in practice and the input is capped far below it.
            let slot = self.tets.len() as u32;
            self.tets.push(tet);
            self.neighbors.push(neighbors);
            self.stamps.push(0);
            slot
        }
    }

    /// Inserts existing point `p`, or reports what it duplicates and leaves the mesh unchanged.
    pub(crate) fn insert(&mut self, p: u32) -> Result<(), TetraError> {
        let walk_state = self.walk_state;
        let start = match self.locate(p)? {
            Located::Tet(t) => t,
            Located::Duplicate(existing) => {
                // A refused point must leave no trace, so a later insertion replays exactly.
                self.walk_state = walk_state;
                return Err(TetraError::DuplicatePoint { index: p, existing });
            }
        };
        self.search_cavity(start, p)?;
        self.rebuild_cavity(p)?;
        Ok(())
    }

    /// The vertex position of [`GHOST`] in `tet`, if it has one.
    fn ghost_position(tet: &[u32; 4]) -> Option<usize> {
        tet.iter().position(|&v| v == GHOST)
    }

    /// Walks from the hint to a tetrahedron in conflict with `p`, crossing any face `p` is
    /// strictly outside of. The visiting order of a tetrahedron's faces varies from step to step
    /// so a walk cannot circle; a walk through a Delaunay triangulation is acyclic in any case,
    /// and the step limit only turns a broken invariant into an error.
    fn locate(&mut self, p: u32) -> Result<Located, TetraError> {
        let target = self.points[p as usize];
        let mut t = self.hint;
        let limit = self.tets.len() + 16;
        for _ in 0..limit {
            let tet = self.tets[t as usize];
            if let Some(g) = Self::ghost_position(&tet) {
                if self.face_orient(&tet, g, &target) > 0.0 {
                    return Ok(Located::Tet(t));
                }
                t = self.neighbors[t as usize][g];
                continue;
            }
            self.walk_state = self
                .walk_state
                .wrapping_mul(0x9E37_79B1)
                .wrapping_add(0x7F4A);
            let first = (self.walk_state >> 30) as usize;
            let mut moved = false;
            for k in 0..4 {
                let i = (first + k) % 4;
                if self.face_orient(&tet, i, &target) < 0.0 {
                    t = self.neighbors[t as usize][i];
                    moved = true;
                    break;
                }
            }
            if !moved {
                // Inside or on the boundary of a real tetrahedron: strictly inside its
                // circumsphere unless `p` coincides with one of its vertices.
                for &v in &tet {
                    if self.points[v as usize] == target {
                        return Ok(Located::Duplicate(v));
                    }
                }
                return Ok(Located::Tet(t));
            }
        }
        Err(TetraError::LocationFailed { point: p })
    }

    /// [`orient`] of face `i` of `tet` (a real face) against point `target`.
    fn face_orient(&self, tet: &[u32; 4], i: usize, target: &[f64; 3]) -> f64 {
        let [a, b, c] = FACE[i].map(|position| &self.points[tet[position] as usize]);
        orient(a, b, c, target)
    }

    /// Whether point `p` conflicts with tetrahedron `t`; see the file header for ghosts.
    fn in_conflict(&self, t: u32, p: u32) -> bool {
        let tet = self.tets[t as usize];
        let Some(g) = Self::ghost_position(&tet) else {
            return in_sphere(&self.points, tet, p);
        };
        let side = self.face_orient(&tet, g, &self.points[p as usize]);
        if side != 0.0 {
            return side > 0.0;
        }
        // `p` is exactly in the hull plane: the real neighbour's sphere decides.
        let real = self.neighbors[t as usize][g];
        in_sphere(&self.points, self.tets[real as usize], p)
    }

    /// Collects in `self.horizon` the boundary of the connected conflict region containing `start`
    /// and frees its tetrahedra.
    fn search_cavity(&mut self, start: u32, p: u32) -> Result<(), TetraError> {
        // At most `n` insertions, each one generation, and the input is capped so `generation`
        // times two fits a u32.
        self.generation += 1;
        let conflict_stamp = self.generation * 2 + 1;
        let clear_stamp = self.generation * 2;
        if !self.in_conflict(start, p) {
            return Err(TetraError::LocationFailed { point: p });
        }
        self.horizon.clear();
        self.cavity.clear();
        self.stack.clear();
        self.stack.push(start);
        self.stamps[start as usize] = conflict_stamp;
        while let Some(t) = self.stack.pop() {
            self.cavity.push(t);
            for (i, face_positions) in FACE.iter().enumerate() {
                let n = self.neighbors[t as usize][i];
                let stamp = self.stamps[n as usize];
                let in_cavity = if stamp == conflict_stamp {
                    true
                } else if stamp == clear_stamp {
                    false
                } else if self.in_conflict(n, p) {
                    self.stamps[n as usize] = conflict_stamp;
                    self.stack.push(n);
                    true
                } else {
                    self.stamps[n as usize] = clear_stamp;
                    false
                };
                if in_cavity {
                    continue;
                }
                let tet = self.tets[t as usize];
                let face = face_positions.map(|position| tet[position]);
                let outer_slot = (0..4u8)
                    .find(|&slot| self.neighbors[n as usize][slot as usize] == t)
                    .ok_or(TetraError::CavityInconsistent { point: p })?;
                self.horizon.push(Horizon {
                    face,
                    outer: n,
                    outer_slot,
                });
            }
        }
        for index in 0..self.cavity.len() {
            let t = self.cavity[index];
            self.tets[t as usize] = [DEAD; 4];
            self.free.push(t);
        }
        Ok(())
    }

    /// Replaces the cavity with one tetrahedron per horizon face, joining each to `p`.
    fn rebuild_cavity(&mut self, p: u32) -> Result<(), TetraError> {
        self.created.clear();
        for index in 0..self.horizon.len() {
            let horizon = self.horizon[index];
            let slot = self.place(
                [horizon.face[0], horizon.face[1], horizon.face[2], p],
                [0, 0, 0, horizon.outer],
            );
            self.neighbors[horizon.outer as usize][horizon.outer_slot as usize] = slot;
            self.created.push(slot);
        }
        self.link_created(p)?;
        // A new tetrahedron always exists: the horizon of a non-empty cavity is not empty.
        if let Some(&last) = self.created.last() {
            self.hint = last;
        }
        Ok(())
    }

    /// Joins the new tetrahedra in `self.created` to each other across their faces containing
    /// `apex`. Each such face is identified by its other two vertices, and in a closed horizon
    /// every such pair occurs exactly twice.
    fn link_created(&mut self, apex: u32) -> Result<(), TetraError> {
        self.edges.clear();
        for &slot in &self.created {
            let tet = self.tets[slot as usize];
            for i in 0..3 {
                // The face opposite `tet[i]` holds the apex and the other two base vertices.
                let (u, v) = match i {
                    0 => (tet[1], tet[2]),
                    1 => (tet[0], tet[2]),
                    _ => (tet[0], tet[1]),
                };
                let key = (u64::from(u.min(v)) << 32) | u64::from(u.max(v));
                self.edges.push((key, slot, i as u8));
            }
        }
        // The key alone orders distinct edges; equal keys are the pair being joined, whose order
        // does not change the links.
        self.edges.sort_unstable_by_key(|entry| entry.0);
        let mut pair = 0;
        while pair < self.edges.len() {
            let closed =
                pair + 1 < self.edges.len() && self.edges[pair].0 == self.edges[pair + 1].0;
            let triple =
                pair + 2 < self.edges.len() && self.edges[pair].0 == self.edges[pair + 2].0;
            if !closed || triple {
                return Err(TetraError::CavityInconsistent { point: apex });
            }
            let (_, first, first_face) = self.edges[pair];
            let (_, second, second_face) = self.edges[pair + 1];
            self.neighbors[first as usize][first_face as usize] = second;
            self.neighbors[second as usize][second_face as usize] = first;
            pair += 2;
        }
        Ok(())
    }

    /// Replaces the tetrahedra `old` with `new`, which must fill exactly the same region, and
    /// rebuilds every adjacency across the region's boundary and between the new tetrahedra.
    /// Returns the new slots. Flips of every size go through here, so adjacency has one author.
    pub(crate) fn replace(
        &mut self,
        old: &[u32],
        new: &[[u32; 4]],
    ) -> Result<Vec<u32>, TetraError> {
        // (face vertices sorted, tetrahedron, its slot, whether the tetrahedron is new)
        let mut faces: Vec<([u32; 3], u32, u8, bool)> =
            Vec::with_capacity(old.len() * 4 + new.len() * 4);
        for &t in old {
            let tet = self.tets[t as usize];
            for i in 0..4 {
                let n = self.neighbors[t as usize][i];
                if old.contains(&n) {
                    continue;
                }
                let slot = (0..4u8)
                    .find(|&j| self.neighbors[n as usize][j as usize] == t)
                    .ok_or(TetraError::CavityInconsistent { point: tet[0] })?;
                faces.push((sorted_face(&tet, i), n, slot, false));
            }
        }
        for &t in old {
            self.tets[t as usize] = [DEAD; 4];
            self.free.push(t);
        }
        let mut slots = Vec::with_capacity(new.len());
        for tet in new {
            let slot = self.place(*tet, [0; 4]);
            slots.push(slot);
            for i in 0..4 {
                faces.push((sorted_face(tet, i), slot, i as u8, true));
            }
        }
        // A region that closes up names each face exactly twice.
        faces.sort_unstable_by_key(|entry| entry.0);
        for pair in faces.chunks(2) {
            let &[(key_a, t_a, slot_a, new_a), (key_b, t_b, slot_b, new_b)] = pair else {
                return Err(TetraError::CavityInconsistent {
                    point: pair[0].0[0],
                });
            };
            if key_a != key_b || !(new_a || new_b) {
                return Err(TetraError::CavityInconsistent { point: key_a[0] });
            }
            self.neighbors[t_a as usize][slot_a as usize] = t_b;
            self.neighbors[t_b as usize][slot_b as usize] = t_a;
        }
        Ok(slots)
    }

    /// Slots holding a tetrahedron, real or ghost: every slot but the freed ones.
    pub(crate) fn live_slots(&self) -> usize {
        self.tets.len() - self.free.len()
    }

    /// Adds a point to the point list, returning its index.
    pub(crate) fn push_point(&mut self, point: [f64; 3]) -> u32 {
        // The caller checked the count is below the index limit.
        let index = self.points.len() as u32;
        self.points.push(point);
        index
    }

    /// Removes the most recently added point: its insertion failed, so nothing references it.
    pub(crate) fn pop_point(&mut self) {
        self.points.pop();
    }
}
