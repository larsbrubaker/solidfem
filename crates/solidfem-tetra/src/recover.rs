//! Recovering one surface constraint at a time, edges before faces, by the flips in `flips.rs`,
//! and telling which side of the recovered surface each tetrahedron is on. `surface.rs` runs
//! these in order and slices them.
//!
//! **An edge.** Walk along the missing segment from one end: the first tetrahedron it enters is
//! found by exact cone tests at the end vertex, and the segment leaves it through a face, through
//! an edge of that face, or through a vertex. A face is removed, so the new edge joins the
//! start vertex to the far apex and the segment has one fewer face to cross; if a neighbouring
//! edge blocks that flip, the edge is removed first. An edge in the segment's path is removed
//! (the coplanar case, where a flat facet's triangulation differs from the surface's). A vertex
//! on the segment means the surface touches itself and is an error.
//!
//! **A face.** Once its three edges exist, a surface triangle is missing only if mesh edges pass
//! through its interior. Each is removed by an edge removal that creates no new edge through the
//! triangle, so the crossings strictly decrease.
//!
//! **Refusal.** The flips are local and can be blocked (the Schoenhardt polyhedron admits no
//! tetrahedralization without a new point). A constraint that makes no progress is retried in a
//! later round, when others may have opened its way; when a round finishes nothing, what remains
//! is reported rather than papered over.

use crate::core::GHOST;
use crate::error::TetraError;
use crate::flips::{face_key, Flipper, Target};
use crate::predicates::orient;

/// A flip budget per constraint: every flip is legal, so this only stops an oscillation.
const MAX_FLIPS_PER_CONSTRAINT: usize = 256;

/// How the segment from a start vertex enters a tetrahedron.
enum Entry {
    /// Through the interior of the face opposite the start vertex.
    Face(u32, usize),
    /// Through the interior of this edge.
    Edge(u32, u32),
}

/// Recovers edge `(u, v)`, returning whether it now exists.
pub(crate) fn recover_edge(flipper: &mut Flipper, u: u32, v: u32) -> Result<bool, TetraError> {
    for _ in 0..MAX_FLIPS_PER_CONSTRAINT {
        if flipper.find_edge(u, v).is_some() {
            return Ok(true);
        }
        let Some(entry) = entry_at(flipper, u, v)? else {
            return Ok(false);
        };
        match entry {
            Entry::Edge(a, b) => {
                if !flipper.remove_edge(a, b, Target::Edge(u, v))? {
                    return Ok(false);
                }
            }
            Entry::Face(t, slot) => match flipper.remove_face(t, slot)? {
                Ok(()) => {}
                Err(blocked) => {
                    let mut flipped = false;
                    for (a, b) in blocked.edges {
                        if flipper.remove_edge(a, b, Target::Edge(u, v))? {
                            flipped = true;
                            break;
                        }
                    }
                    if !flipped {
                        return Ok(false);
                    }
                }
            },
        }
    }
    Ok(flipper.find_edge(u, v).is_some())
}

/// Finds where the segment from `u` towards `v` leaves the star of `u`, by exact orientation.
fn entry_at(flipper: &mut Flipper, u: u32, v: u32) -> Result<Option<Entry>, TetraError> {
    for t in flipper.tets_around(u) {
        let tet = flipper.core.tets[t as usize];
        if tet.contains(&GHOST) {
            continue;
        }
        let Some(slot) = tet.iter().position(|&w| w == u) else {
            continue;
        };
        // Wound so `(u, a, b, c)` is positive: seen from `u`, the far face runs a to b to c.
        let face = crate::core::FACE[slot].map(|position| tet[position]);
        let (a, b, c) = (face[1], face[0], face[2]);
        let point = |w: u32| &flipper.core.points[w as usize];
        let sides = [
            orient(point(u), point(a), point(b), point(v)),
            orient(point(u), point(b), point(c), point(v)),
            orient(point(u), point(c), point(a), point(v)),
        ];
        if sides.iter().any(|&s| s < 0.0) {
            continue;
        }
        let zeros = sides.iter().filter(|&&s| s == 0.0).count();
        match zeros {
            0 => return Ok(Some(Entry::Face(t, slot))),
            1 => {
                let edge = if sides[0] == 0.0 {
                    (a, b)
                } else if sides[1] == 0.0 {
                    (b, c)
                } else {
                    (c, a)
                };
                return Ok(Some(Entry::Edge(edge.0, edge.1)));
            }
            _ => {
                // Two zeros: the segment runs along the edge from `u` to the shared vertex.
                let vertex = if sides[0] == 0.0 && sides[2] == 0.0 {
                    a
                } else if sides[0] == 0.0 && sides[1] == 0.0 {
                    b
                } else {
                    c
                };
                return Err(TetraError::VertexOnSurfaceEdge {
                    vertex,
                    edge: [u, v],
                });
            }
        }
    }
    Ok(None)
}

/// Recovers face `(a, b, c)`, whose three edges must exist, returning whether it now exists.
pub(crate) fn recover_face(
    flipper: &mut Flipper,
    a: u32,
    b: u32,
    c: u32,
) -> Result<bool, TetraError> {
    let face = [a, b, c];
    for _ in 0..MAX_FLIPS_PER_CONSTRAINT {
        if flipper.face_exists(a, b, c) {
            return Ok(true);
        }
        let crossing = crossing_edges(flipper, face);
        if crossing.is_empty() {
            return Ok(false);
        }
        let mut flipped = false;
        for (p, q) in crossing {
            if flipper.remove_edge(p, q, Target::Face(face))? {
                flipped = true;
                break;
            }
        }
        if !flipped {
            return Ok(false);
        }
    }
    Ok(flipper.face_exists(a, b, c))
}

/// The mesh edges passing through the interior of triangle `face`, in a fixed order: gathered from
/// the tetrahedra around its corners and then outward through tetrahedra sharing a crossing edge.
fn crossing_edges(flipper: &mut Flipper, face: [u32; 3]) -> Vec<(u32, u32)> {
    let mut frontier: Vec<u32> = Vec::new();
    for &corner in &face {
        frontier.extend(flipper.tets_around(corner));
    }
    frontier.sort_unstable();
    frontier.dedup();
    let mut seen = frontier.clone();
    let mut crossing: Vec<(u32, u32)> = Vec::new();
    while let Some(t) = frontier.pop() {
        let tet = flipper.core.tets[t as usize];
        if tet.contains(&GHOST) {
            continue;
        }
        let mut any = false;
        for i in 0..4 {
            for j in i + 1..4 {
                let (p, q) = (tet[i].min(tet[j]), tet[i].max(tet[j]));
                if flipper.crosses_face(p, q, face) {
                    any = true;
                    if !crossing.contains(&(p, q)) {
                        crossing.push((p, q));
                    }
                }
            }
        }
        if any {
            for i in 0..4 {
                let n = flipper.core.neighbors[t as usize][i];
                if !seen.contains(&n) {
                    seen.push(n);
                    frontier.push(n);
                }
            }
        }
    }
    crossing.sort_unstable();
    crossing
}

/// The two tetrahedra sharing `face` (sorted): one has its apex on the positive side.
pub(crate) fn tets_of_face(flipper: &mut Flipper, face: [u32; 3]) -> Vec<(u32, u32)> {
    let around = flipper.tets_around(face[0]);
    let mut found = Vec::new();
    for t in around {
        let tet = flipper.core.tets[t as usize];
        if face.iter().all(|v| tet.contains(v)) {
            if let Some(&apex) = tet.iter().find(|v| !face.contains(v)) {
                found.push((t, apex));
            }
        }
    }
    found
}

/// Labels every tetrahedron by how many surface layers separate it from outside: the ghosts are
/// layer 0 and each surface face crossed adds one. A flood across non-surface faces is what
/// fills a layer, so a stray tetrahedron cannot be mislabelled by any geometric test.
pub(crate) fn label_layers(flipper: &Flipper) -> Result<Vec<i32>, TetraError> {
    let core = &flipper.core;
    let mut label = vec![-1i32; core.tets.len()];
    let mut layer = 0;
    let mut seeds: Vec<u32> = core
        .tets
        .iter()
        .enumerate()
        .filter(|(_, tet)| tet.contains(&GHOST) && tet[0] != crate::core::DEAD)
        // `slot` fits u32 as elsewhere in the triangulation.
        .map(|(slot, _)| slot as u32)
        .collect();
    while !seeds.is_empty() {
        let mut stack = Vec::new();
        for &seed in &seeds {
            if label[seed as usize] == -1 {
                label[seed as usize] = layer;
                stack.push(seed);
            }
        }
        let mut next = Vec::new();
        while let Some(t) = stack.pop() {
            let tet = core.tets[t as usize];
            for i in 0..4 {
                let n = core.neighbors[t as usize][i];
                let verts: Vec<u32> = (0..4).filter(|&j| j != i).map(|j| tet[j]).collect();
                let constraint = flipper
                    .locked_faces
                    .contains(&face_key(verts[0], verts[1], verts[2]));
                if constraint {
                    if label[n as usize] == -1 {
                        next.push(n);
                    }
                } else if label[n as usize] == -1 {
                    label[n as usize] = layer;
                    stack.push(n);
                }
            }
        }
        layer += 1;
        seeds = next;
    }
    Ok(label)
}
