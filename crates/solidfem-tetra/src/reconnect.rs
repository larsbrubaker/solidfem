//! Making a stuck surface edge exist by replacing the whole region its segment passes through,
//! with no point added. Flips (`flips.rs`) change a handful of tetrahedra at a time and each must
//! be legal alone, so a segment can be crossed by tetrahedra whose every single flip is blocked
//! while the region as a whole has a perfectly good triangulation holding the edge. `surface.rs`
//! tries this on a stuck edge before any point is added for it.
//!
//! **Why a star from one end works.** Take every tetrahedron the open segment `u v` touches and
//! grow that region across any face that does not see `u`, exactly as a point's cavity grows
//! (`Flipper::steiner_cavity`). Then the tetrahedra joining `u` to each rim face not holding `u`
//! fill the region exactly: an oriented closed surface covers each point by its winding number,
//! so if every one of those tetrahedra is strictly positive they cover the region once and
//! nothing else. `v` is a corner of the region, so the rim faces at `v` give tetrahedra holding
//! the edge. Growth stops at a locked face: a piece of surface behind `u` cannot be seen from it,
//! which is when the region cannot be starred from that end and the other end is tried.
//!
//! **What is kept.** The same rules as a point's cavity: no surface vertex disappears, no locked
//! edge or face is destroyed (one held by the apex end survives through the star), and a point
//! added earlier may be swallowed.
//!
//! **Why the star needs no further checks.** The rim of a region is a closed surface, so every
//! rim edge not at the apex lies in two rim faces: a third would give two star tetrahedra on one
//! side of the face through the apex and that edge, covering points twice, which the winding
//! number forbids. A rim edge at the apex may lie in four faces, where the star simply has two
//! fans around it. `other` is a surface vertex in the region, kept on the rim by the vertex
//! check, and a vertex on a rim has rim faces not holding the apex, so the edge always appears.

use crate::core::{FACE, GHOST};
use crate::error::TetraError;
use crate::flips::Flipper;
use crate::predicates::orient;

/// The most tetrahedra one reconnection may start from: a segment passes through a few dozen at
/// most, so more means this is not the local problem it is built for.
const MAX_SEGMENT_TETS: usize = 512;

impl Flipper {
    /// Tries to make missing edge `(u, v)` exist by a star on either end, returning whether it
    /// did. A refused attempt leaves the mesh exactly as it was.
    pub(crate) fn reconnect_edge(&mut self, u: u32, v: u32) -> Result<bool, TetraError> {
        if self.star_from(u, v)? || self.star_from(v, u)? {
            self.counts.star_reconnections += 1;
            return Ok(true);
        }
        Ok(false)
    }

    /// The star on `apex` of the region around segment `(apex, other)`, if one exists.
    fn star_from(&mut self, apex: u32, other: u32) -> Result<bool, TetraError> {
        let seeds = self.tets_on_segment(apex, other);
        if seeds.is_empty() {
            return Ok(false);
        }
        let position = self.core.points[apex as usize];
        let Some(cavity) = self.steiner_cavity(&seeds, position, Some(apex)) else {
            return Ok(false);
        };
        let new = self.star_tets(&cavity, apex);
        self.replace_cavity(&cavity, &new)?;
        Ok(true)
    }

    /// The tetrahedra starring `cavity` on `apex`, one per rim face not holding it. The cavity is
    /// already known to see the apex strictly from every such face.
    fn star_tets(&self, cavity: &[u32], apex: u32) -> Vec<[u32; 4]> {
        let mut new = Vec::new();
        for &t in cavity {
            let tet = self.core.tets[t as usize];
            for (i, positions) in FACE.iter().enumerate() {
                if cavity.contains(&self.core.neighbors[t as usize][i]) {
                    continue;
                }
                let face = positions.map(|position| tet[position]);
                if !face.contains(&apex) {
                    new.push([face[0], face[1], face[2], apex]);
                }
            }
        }
        new
    }

    /// The tetrahedra the open segment `(u, v)` touches, found from those around its ends and
    /// outward through neighbours that also touch it, in a fixed order; none when too many.
    fn tets_on_segment(&mut self, u: u32, v: u32) -> Vec<u32> {
        let mut around = self.tets_around(u);
        around.extend(self.tets_around(v));
        around.sort_unstable();
        around.dedup();
        let mut found: Vec<u32> = around
            .into_iter()
            .filter(|&t| self.tet_touches_segment(t, u, v))
            .collect();
        let mut next = 0;
        while next < found.len() {
            if found.len() > MAX_SEGMENT_TETS {
                return Vec::new();
            }
            let t = found[next];
            next += 1;
            for i in 0..4 {
                let neighbor = self.core.neighbors[t as usize][i];
                if !found.contains(&neighbor) && self.tet_touches_segment(neighbor, u, v) {
                    found.push(neighbor);
                }
            }
        }
        found.sort_unstable();
        found
    }

    /// Whether a real tetrahedron has a face, holding neither end, that the segment reaches.
    fn tet_touches_segment(&self, t: u32, u: u32, v: u32) -> bool {
        let tet = self.core.tets[t as usize];
        if tet.contains(&GHOST) {
            return false;
        }
        FACE.iter().any(|positions| {
            let face = positions.map(|position| tet[position]);
            !face.contains(&u) && !face.contains(&v) && self.segment_reaches(u, v, face)
        })
    }

    /// Exactly whether segment `(x, y)` meets the closed triangle `face`: its ends not strictly
    /// on one side of the plane, and the three edge sides not disagreeing.
    fn segment_reaches(&self, x: u32, y: u32, face: [u32; 3]) -> bool {
        let point = |w: u32| &self.core.points[w as usize];
        let [a, b, c] = face.map(point);
        let (sx, sy) = (orient(a, b, c, point(x)), orient(a, b, c, point(y)));
        if (sx > 0.0 && sy > 0.0) || (sx < 0.0 && sy < 0.0) {
            return false;
        }
        let sides = [
            orient(point(x), point(y), a, b),
            orient(point(x), point(y), b, c),
            orient(point(x), point(y), c, a),
        ];
        sides.iter().all(|&s| s >= 0.0) || sides.iter().all(|&s| s <= 0.0)
    }
}

#[cfg(test)]
mod tests {
    //! The checks that make a star safe, driven directly on Delaunay triangulations of integer
    //! point sets. Integer coordinates make the total volume an exact integer, so "the star
    //! tiled the same region" has no tolerance. Locks are random and a locked face's edges are
    //! left unlocked on purpose: the surface never does that, and it is what reaches each check
    //! on its own.

    use std::collections::HashSet;

    use super::*;
    use crate::core::DEAD;
    use crate::flips::{edge_key, face_key};
    use crate::triangulator::Triangulator;

    /// A fixed xorshift stream: every run sees the same clouds, locks and attempts.
    struct Stream(u64);

    impl Stream {
        fn below(&mut self, limit: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % limit
        }
    }

    /// Six times a tetrahedron's volume, exactly.
    fn six_volume(points: &[[i64; 3]], tet: [u32; 4]) -> i128 {
        let [a, b, c, d] = tet.map(|v| points[v as usize]);
        let row = |p: [i64; 3]| [p[0] - d[0], p[1] - d[1], p[2] - d[2]].map(i128::from);
        let (x, y, z) = (row(a), row(b), row(c));
        x[0] * (y[1] * z[2] - y[2] * z[1]) - x[1] * (y[0] * z[2] - y[2] * z[0])
            + x[2] * (y[0] * z[1] - y[1] * z[0])
    }

    fn is_real(tet: &[u32; 4]) -> bool {
        tet[0] != DEAD && !tet.contains(&GHOST)
    }

    fn sorted_face(tet: &[u32; 4], slot: usize) -> [u32; 3] {
        let mut face = [0; 3];
        let mut count = 0;
        for (index, &vertex) in tet.iter().enumerate() {
            if index != slot {
                face[count] = vertex;
                count += 1;
            }
        }
        face.sort_unstable();
        face
    }

    /// Everything a star must keep, read from the mesh's arrays alone.
    fn assert_valid(flipper: &Flipper, points: &[[i64; 3]], start_volume: i128, what: &str) {
        let core = &flipper.core;
        let mut volume = 0i128;
        let mut vertices = HashSet::new();
        let mut edges = HashSet::new();
        let mut faces = HashSet::new();
        for (slot, tet) in core.tets.iter().enumerate() {
            if tet[0] == DEAD {
                continue;
            }
            for (i, &n) in core.neighbors[slot].iter().enumerate() {
                let across = core.tets[n as usize];
                assert_ne!(across[0], DEAD, "{what}: tet {slot} has a dead neighbour");
                let back: Vec<usize> = (0..4)
                    .filter(|&j| core.neighbors[n as usize][j] as usize == slot)
                    .collect();
                assert_eq!(back.len(), 1, "{what}: tet {slot} and {n} disagree");
                assert_eq!(
                    sorted_face(tet, i),
                    sorted_face(&across, back[0]),
                    "{what}: tet {slot} and {n} do not share the face they name"
                );
            }
            vertices.extend(tet.iter().copied().filter(|&v| v != GHOST));
            if !is_real(tet) {
                continue;
            }
            let [a, b, c, d] = tet.map(|v| &core.points[v as usize]);
            assert!(
                orient(a, b, c, d) > 0.0,
                "{what}: tet {tet:?} is not strictly positive"
            );
            volume += six_volume(points, *tet).abs();
            for i in 0..4 {
                for j in i + 1..4 {
                    edges.insert(edge_key(tet[i], tet[j]));
                }
                faces.insert(sorted_face(tet, i));
            }
        }
        assert_eq!(volume, start_volume, "{what}: the region's volume changed");
        assert_eq!(vertices.len(), points.len(), "{what}: a vertex was lost");
        for edge in &flipper.locked_edges {
            assert!(
                edges.contains(edge),
                "{what}: locked edge {edge:x} was lost"
            );
        }
        for face in &flipper.locked_faces {
            assert!(
                faces.contains(face),
                "{what}: locked face {face:?} was lost"
            );
        }
    }

    /// One triangulation of `cloud` with random locks, and attempts on random missing edges,
    /// everything checked after each. Returns how many stars happened and how many were refused.
    fn run(cloud: &[[i64; 3]], stream: &mut Stream, lock_percent: u64) -> (usize, usize) {
        let mut points: Vec<[i64; 3]> = cloud.to_vec();
        points.sort_unstable();
        points.dedup();
        // Every coordinate is far below 2^53, so the cast is exact.
        let floats: Vec<[f64; 3]> = points.iter().map(|p| p.map(|c| c as f64)).collect();
        let mut triangulator = Triangulator::new(floats).expect("a cloud with volume");
        triangulator
            .step(usize::MAX, &mut || false)
            .expect("triangulates");
        let mut flipper = Flipper::new(
            triangulator.into_core(),
            HashSet::new(),
            HashSet::new(),
            Vec::new(),
        );
        for tet in flipper.core.tets.clone().iter().filter(|t| is_real(t)) {
            for i in 0..4 {
                for j in i + 1..4 {
                    if stream.below(100) < lock_percent {
                        flipper.locked_edges.insert(edge_key(tet[i], tet[j]));
                    }
                }
                if stream.below(100) < lock_percent {
                    let [a, b, c] = sorted_face(tet, i);
                    flipper.locked_faces.insert(face_key(a, b, c));
                }
            }
        }
        let start_volume: i128 = flipper
            .core
            .tets
            .iter()
            .filter(|tet| is_real(tet))
            .map(|tet| six_volume(&points, *tet).abs())
            .sum();
        assert_valid(&flipper, &points, start_volume, "before any star");
        let (mut stars, mut refused) = (0, 0);
        // The vertex count fits u64 and u32: these clouds hold at most a few hundred points.
        let count = points.len() as u64;
        for attempt in 0..40 {
            let u = stream.below(count) as u32;
            let v = stream.below(count) as u32;
            if u == v || flipper.find_edge(u, v).is_some() {
                continue;
            }
            let before = (flipper.core.tets.clone(), flipper.core.neighbors.clone());
            let what = format!("attempt {attempt} on edge {u} {v}");
            if flipper.reconnect_edge(u, v).expect("no inconsistency") {
                stars += 1;
                assert!(flipper.find_edge(u, v).is_some(), "{what}: edge is missing");
            } else {
                refused += 1;
                assert!(
                    before.0 == flipper.core.tets && before.1 == flipper.core.neighbors,
                    "{what}: a refused star changed the mesh"
                );
            }
            assert_valid(&flipper, &points, start_volume, &what);
        }
        (stars, refused)
    }

    /// Runs `cloud` for many seeds at several lock densities, and holds the run to having both
    /// made stars and refused them, so the checks above were reached.
    fn fuzz(mut cloud: impl FnMut(&mut Stream) -> Vec<[i64; 3]>) {
        let (mut stars, mut refused) = (0, 0);
        for seed in 1..=60 {
            let mut stream = Stream(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(seed));
            let points = cloud(&mut stream);
            for lock_percent in [0, 3, 10, 30] {
                let (made, not) = run(&points, &mut stream, lock_percent);
                stars += made;
                refused += not;
            }
        }
        assert!(stars > 100, "only {stars} stars happened");
        assert!(refused > 100, "only {refused} stars were refused");
    }

    #[test]
    fn stars_on_random_clouds_keep_the_mesh_valid() {
        fuzz(|stream| {
            (0..70)
                .map(|_| [(); 3].map(|()| stream.below(65536) as i64))
                .collect()
        });
    }

    /// Clouds in a slab a few units thick: faces and edges are nearly coplanar everywhere.
    #[test]
    fn stars_on_nearly_flat_clouds_keep_the_mesh_valid() {
        fuzz(|stream| {
            (0..70)
                .map(|_| {
                    [
                        stream.below(4096) as i64,
                        stream.below(4096) as i64,
                        stream.below(5) as i64,
                    ]
                })
                .collect()
        });
    }

    /// Exactly degenerate: every point on a small lattice, most of them kept, so faces are
    /// exactly coplanar and points exactly cospherical all the time.
    #[test]
    fn stars_on_lattice_clouds_keep_the_mesh_valid() {
        fuzz(|stream| {
            let mut points = Vec::new();
            for x in 0..5 {
                for y in 0..4 {
                    for z in 0..4 {
                        if stream.below(100) < 80 {
                            points.push([x, y, z]);
                        }
                    }
                }
            }
            points
        });
    }
}
