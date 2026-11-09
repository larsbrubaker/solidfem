//! The quality measures and the moves that improve the worst tetrahedra, worst first: edge removal
//! choosing the best triangulation of the ring, face removal, and guarded smoothing of vertices
//! not on the surface. `refine.rs` drives these a tetrahedron at a time.
//!
//! **The measure.** Minimum dihedral angle in degrees: a sliver has angles near zero and near 180
//! degrees and no good 10-node element has either. The scaled Jacobian (the volume over the three
//! edge lengths at a corner) is the second measure, reported with it. Floating-point values are
//! used only to choose among moves; legality is always the exact orientation test.
//!
//! **Why edge removal chooses by dynamic programming.** The ring around an edge has many
//! triangulations, and one poor tetrahedron in the choice is the bad outcome. The interval search
//! maximises the minimum over the whole ring rather than taking the first legal choice.
//!
//! **Walls stay split.** A flip never adds an edge joining opposite skins of a thin wall
//! ([`OppositeSkins`]), so what `skins.rs` removed stays removed. Edges along a skin and edges to
//! interior points stay free, which is what the flips need to repair the elements a split leaves.
//!
//! **Smoothing.** A vertex moves towards the centroid of its neighbours, at the first of a few
//! step lengths that strictly raises the worst tetrahedron in its star while every tetrahedron
//! stays exactly positive, so a move never makes anything worse. A point dividing a wall moves only
//! in the plane of that wall, so the layers stay planar.

use std::collections::HashMap;

use crate::core::GHOST;
use crate::error::TetraError;
use crate::flips::{edge_key, Flipper};
use crate::skins::OppositeSkins;

/// Rings larger than this are skipped: the search is cubic and big rings give poor elements.
const MAX_QUALITY_RING: usize = 7;

/// Step lengths tried for one smoothing move, largest first.
const SMOOTH_STEPS: [f64; 3] = [1.0, 0.5, 0.25];

/// Below this ratio of volume to edge-length product a tetrahedron is treated as flat: a few
/// thousand units of rounding of the coordinate magnitudes the mesher accepts, far under any
/// dihedral angle that would matter to the quality pass.
const FLAT_RELATIVE_VOLUME: f64 = 1e-12;

pub(crate) fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

pub(crate) fn normalized(a: [f64; 3]) -> [f64; 3] {
    let l = length(a);
    if l > 0.0 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        [0.0; 3]
    }
}

/// Whether a tetrahedron is flat within what its coordinates can resolve: its volume determinant
/// is below [`FLAT_RELATIVE_VOLUME`] of its three edge lengths at a corner multiplied together.
/// An exactly positive tetrahedron can be flat by this measure, four points in one plane but for
/// a unit of rounding; its face normals then carry no reliable sign, and reading an angle off
/// them would rate the worst possible shape as a good one.
fn flat_to_rounding(p: &[[f64; 3]; 4]) -> bool {
    let (u, v, w) = (
        subtract(p[1], p[0]),
        subtract(p[2], p[0]),
        subtract(p[3], p[0]),
    );
    let volume = dot(u, cross(v, w)).abs();
    volume <= FLAT_RELATIVE_VOLUME * length(u) * length(v) * length(w)
}

/// The smallest of the six dihedral angles of a tetrahedron, in degrees, zero for one flat to
/// rounding (`flat_to_rounding`).
pub(crate) fn min_dihedral_degrees(points: &[[f64; 3]], tet: &[u32; 4]) -> f64 {
    let p = tet.map(|v| points[v as usize]);
    // The outward normal of the face opposite vertex `i`.
    let normals: [[f64; 3]; 4] = std::array::from_fn(|i| {
        let others: Vec<[f64; 3]> = (0..4).filter(|&k| k != i).map(|k| p[k]).collect();
        let n = cross(
            subtract(others[1], others[0]),
            subtract(others[2], others[0]),
        );
        let inner = dot(n, subtract(p[i], others[0]));
        let oriented = if inner > 0.0 {
            [-n[0], -n[1], -n[2]]
        } else {
            n
        };
        normalized(oriented)
    });
    if flat_to_rounding(&p) {
        return 0.0;
    }
    let mut smallest = 180.0_f64;
    for i in 0..4 {
        for j in i + 1..4 {
            let cosine = dot(normals[i], normals[j]).clamp(-1.0, 1.0);
            // Outward normals: the interior angle is the supplement of the angle between them.
            let dihedral = 180.0 - cosine.acos().to_degrees();
            smallest = smallest.min(dihedral);
        }
    }
    smallest
}

/// The worst scaled Jacobian of a tetrahedron over its four corners: 1 for a regular corner,
/// near zero for a sliver.
pub(crate) fn scaled_jacobian(points: &[[f64; 3]], tet: &[u32; 4]) -> f64 {
    let p = tet.map(|v| points[v as usize]);
    let mut worst = f64::INFINITY;
    for i in 0..4 {
        let others: Vec<[f64; 3]> = (0..4).filter(|&k| k != i).map(|k| p[k]).collect();
        let (a, b, c) = (
            subtract(others[0], p[i]),
            subtract(others[1], p[i]),
            subtract(others[2], p[i]),
        );
        let volume = dot(a, cross(b, c)).abs();
        let scale = length(a) * length(b) * length(c);
        if scale > 0.0 {
            worst = worst.min(volume / scale);
        }
    }
    worst
}

/// How a set of tetrahedra stands on minimum dihedral angle.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DihedralHistogram {
    /// Counts of tetrahedra by minimum dihedral, in 5-degree bins from 0 to 90.
    pub counts: [usize; 18],
    pub tets: usize,
    pub min_degrees: f64,
    pub mean_degrees: f64,
    pub worst_scaled_jacobian: f64,
}

/// The histogram of the given tetrahedra.
pub(crate) fn histogram<'a>(
    points: &[[f64; 3]],
    tets: impl Iterator<Item = &'a [u32; 4]>,
) -> DihedralHistogram {
    let mut result = DihedralHistogram {
        min_degrees: 180.0,
        worst_scaled_jacobian: f64::INFINITY,
        ..DihedralHistogram::default()
    };
    let mut total = 0.0;
    for tet in tets {
        let angle = min_dihedral_degrees(points, tet);
        result.tets += 1;
        total += angle;
        result.min_degrees = result.min_degrees.min(angle);
        result.worst_scaled_jacobian = result
            .worst_scaled_jacobian
            .min(scaled_jacobian(points, tet));
        // The minimum dihedral of a tetrahedron is at most about 70.53 degrees.
        let bin = ((angle / 5.0).floor() as usize).min(17);
        result.counts[bin] += 1;
    }
    if result.tets > 0 {
        result.mean_degrees = total / result.tets as f64;
    }
    result
}

impl Flipper {
    /// The worst minimum dihedral of the tetrahedra in `slots`.
    fn worst_of(&self, slots: &[u32]) -> f64 {
        slots
            .iter()
            .map(|&t| min_dihedral_degrees(&self.core.points, &self.core.tets[t as usize]))
            .fold(180.0, f64::min)
    }

    /// Tries to improve tetrahedron `slot`, by edge removal, face removal, then smoothing of its
    /// free vertices. Returns whether anything changed.
    pub(crate) fn improve_tet(
        &mut self,
        slot: u32,
        fixed: &[bool],
        skins: &OppositeSkins,
        axes: &HashMap<u32, [f64; 3]>,
        target: f64,
    ) -> Result<bool, TetraError> {
        let tet = self.core.tets[slot as usize];
        if tet.iter().any(|&v| v >= crate::core::DEAD)
            || min_dihedral_degrees(&self.core.points, &tet) >= target
        {
            return Ok(false);
        }
        for a in 0..4 {
            for b in a + 1..4 {
                if self.remove_edge_for_quality(tet[a], tet[b], skins)? {
                    return Ok(true);
                }
            }
        }
        for face in 0..4 {
            if self.remove_face_for_quality(slot, face, skins)? {
                return Ok(true);
            }
        }
        let mut moved = false;
        for &vertex in &tet {
            let is_fixed = fixed.get(vertex as usize).copied().unwrap_or(false);
            if !is_fixed && self.smooth_vertex(vertex, axes.get(&vertex).copied()) {
                moved = true;
            }
        }
        Ok(moved)
    }

    /// Replaces the ring around edge `(p, q)` by its best triangulation if that raises the worst
    /// tetrahedron of the ring.
    fn remove_edge_for_quality(
        &mut self,
        p: u32,
        q: u32,
        skins: &OppositeSkins,
    ) -> Result<bool, TetraError> {
        if self.locked_edges.contains(&edge_key(p, q)) {
            return Ok(false);
        }
        let Some((ring, tets)) = self.ring_public(p, q) else {
            return Ok(false);
        };
        if ring.len() < 3 || ring.len() > MAX_QUALITY_RING || ring.contains(&GHOST) {
            return Ok(false);
        }
        let old = self.worst_of(&tets);
        let Some((new_worst, new)) = self.best_ring(p, q, &ring, skins) else {
            return Ok(false);
        };
        if new_worst <= old {
            return Ok(false);
        }
        self.apply(&tets, &new)?;
        match ring.len() {
            3 => self.counts.three_two += 1,
            4 => self.counts.four_four += 1,
            _ => self.counts.larger += 1,
        }
        Ok(true)
    }

    /// Removes a through-wall edge by retriangulating its ring, for an edge whose ring a point on
    /// the edge cannot be starred from (a flat ring tetrahedron holding an interior vertex in its
    /// plane). No new edge may join opposite skins, so the removal is real. Returns whether the
    /// edge is gone; `skins.rs` decides what that means for the wall.
    pub(crate) fn remove_through_edge(
        &mut self,
        p: u32,
        q: u32,
        skins: &OppositeSkins,
    ) -> Result<bool, TetraError> {
        let Some((ring, tets)) = self.ring_public(p, q) else {
            return Ok(false);
        };
        if ring.len() < 3 || ring.len() > MAX_QUALITY_RING || ring.contains(&GHOST) {
            return Ok(false);
        }
        let Some((_, new)) = self.best_ring(p, q, &ring, skins) else {
            return Ok(false);
        };
        self.apply(&tets, &new)?;
        match ring.len() {
            3 => self.counts.three_two += 1,
            4 => self.counts.four_four += 1,
            _ => self.counts.larger += 1,
        }
        Ok(true)
    }

    /// The triangulation of the ring with the largest worst tetrahedron, or `None` when no legal
    /// one exists.
    fn best_ring(
        &self,
        p: u32,
        q: u32,
        ring: &[u32],
        skins: &OppositeSkins,
    ) -> Option<(f64, Vec<[u32; 4]>)> {
        let m = ring.len() - 1;
        let width = m + 1;
        let mut best = vec![f64::NEG_INFINITY; width * width];
        let mut split = vec![0usize; width * width];
        for i in 0..m {
            best[i * width + i + 1] = f64::INFINITY;
        }
        for span in 2..=m {
            for i in 0..=m - span {
                let j = i + span;
                for k in i + 1..j {
                    let (left, right) = (best[i * width + k], best[k * width + j]);
                    if left == f64::NEG_INFINITY || right == f64::NEG_INFINITY {
                        continue;
                    }
                    // A chord other than the ring's closing side is a new edge.
                    let is_new = !(i == 0 && j == m);
                    if is_new && skins.joins(&self.core.points, ring[i], ring[j]) {
                        continue;
                    }
                    let pair = ring_pair(p, q, ring[i], ring[k], ring[j]);
                    if !self.tet_positive(&pair[0]) || !self.tet_positive(&pair[1]) {
                        continue;
                    }
                    let worst = left
                        .min(right)
                        .min(min_dihedral_degrees(&self.core.points, &pair[0]))
                        .min(min_dihedral_degrees(&self.core.points, &pair[1]));
                    if worst > best[i * width + j] {
                        best[i * width + j] = worst;
                        split[i * width + j] = k;
                    }
                }
            }
        }
        if best[m] == f64::NEG_INFINITY {
            return None;
        }
        let mut new = Vec::with_capacity(2 * m);
        let mut stack = vec![(0usize, m)];
        while let Some((i, j)) = stack.pop() {
            if j - i < 2 {
                continue;
            }
            let k = split[i * width + j];
            new.extend_from_slice(&ring_pair(p, q, ring[i], ring[k], ring[j]));
            stack.push((i, k));
            stack.push((k, j));
        }
        Some((best[m], new))
    }

    /// Replaces two tetrahedra sharing face `face` of `slot` by three around the edge joining
    /// their apexes, if that raises their worst.
    fn remove_face_for_quality(
        &mut self,
        slot: u32,
        face: usize,
        skins: &OppositeSkins,
    ) -> Result<bool, TetraError> {
        let tet = self.core.tets[slot as usize];
        let u = tet[face];
        let n = self.core.neighbors[slot as usize][face];
        let other = self.core.tets[n as usize];
        if other.contains(&GHOST) {
            return Ok(false);
        }
        let Some(y) = other.iter().copied().find(|v| !tet.contains(v)) else {
            return Ok(false);
        };
        if skins.joins(&self.core.points, u, y) {
            return Ok(false);
        }
        let positions = crate::core::FACE[face].map(|p| tet[p]);
        let (a, b, c) = (positions[1], positions[0], positions[2]);
        let new = [[u, y, a, b], [u, y, b, c], [u, y, c, a]];
        if !new.iter().all(|t| self.tet_positive(t)) {
            return Ok(false);
        }
        let old = self.worst_of(&[slot, n]);
        let worst = new
            .iter()
            .map(|t| min_dihedral_degrees(&self.core.points, t))
            .fold(180.0, f64::min);
        if worst <= old {
            return Ok(false);
        }
        self.apply(&[slot, n], &new)?;
        self.counts.two_three += 1;
        Ok(true)
    }

    /// Moves `vertex` towards the centroid of its neighbours where that raises the worst
    /// tetrahedron around it, a point dividing a wall only in that wall's plane (`axis` is its
    /// normal). Returns whether it moved.
    fn smooth_vertex(&mut self, vertex: u32, axis: Option<[f64; 3]>) -> bool {
        let star = self.tets_around(vertex);
        if star
            .iter()
            .any(|&t| self.core.tets[t as usize].contains(&GHOST))
        {
            return false;
        }
        let mut neighbours: Vec<u32> = star
            .iter()
            .flat_map(|&t| self.core.tets[t as usize])
            .filter(|&v| v != vertex)
            .collect();
        neighbours.sort_unstable();
        neighbours.dedup();
        if neighbours.is_empty() {
            return false;
        }
        let mut centroid = [0.0; 3];
        for &v in &neighbours {
            let p = self.core.points[v as usize];
            for k in 0..3 {
                centroid[k] += p[k];
            }
        }
        for value in &mut centroid {
            *value /= neighbours.len() as f64;
        }
        let start = self.core.points[vertex as usize];
        let mut step = subtract(centroid, start);
        if let Some(axis) = axis {
            let along = dot(step, axis);
            step = [
                step[0] - axis[0] * along,
                step[1] - axis[1] * along,
                step[2] - axis[2] * along,
            ];
        }
        let old = self.worst_of(&star);
        for fraction in SMOOTH_STEPS {
            let candidate = [
                start[0] + step[0] * fraction,
                start[1] + step[1] * fraction,
                start[2] + step[2] * fraction,
            ];
            if candidate == start {
                continue;
            }
            self.core.points[vertex as usize] = candidate;
            if star
                .iter()
                .all(|&t| self.tet_positive(&self.core.tets[t as usize]))
                && self.worst_of(&star) > old
            {
                return true;
            }
        }
        self.core.points[vertex as usize] = start;
        false
    }
}

/// The two tetrahedra of ring triangle `(i, k, j)`, as `flips.rs` builds them.
fn ring_pair(p: u32, q: u32, i: u32, k: u32, j: u32) -> [[u32; 4]; 2] {
    [[i, j, k, p], [i, k, j, q]]
}
