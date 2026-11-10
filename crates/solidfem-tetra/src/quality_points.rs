//! The last move of the quality pass: a point strictly inside the solid for a sliver no flip can
//! remove. `quality.rs` holds the flips and the smoothing, `interior.rs` the cavities; `refine.rs`
//! calls this after those have given up.
//!
//! **Which slivers need it.** A tetrahedron whose four vertices are on the surface, two of its
//! faces on the surface, has five edges fixed and one free edge, the other diagonal of the fold.
//! Two surface triangles meeting at a gentle fold on a curved wall make such a tetrahedron nearly
//! flat, and a surface recovered onto a triangulation of the same points leaves one wherever its
//! diagonal differs from the Delaunay one. Its ring around the free edge cannot be triangulated
//! without it (no legal flip exists: the line of the free edge meets the plane of any replacement
//! triangle outside the edge), its vertices are surface vertices and do not move, and no
//! size-driven point is drawn to it because its edges are short. So it survives every other
//! stage, as a tetrahedron as thin as the surface's curvature, down to flat to rounding, and the
//! same ring argument holds for the interior slivers that flips and smoothing leave beside it.
//!
//! **The move.** A point above the sliver, inward from its surface faces, or inside it where it
//! has none, sees the surface faces strictly; the cavity (the sliver, grown across every face the
//! point cannot see strictly, by `Flipper::visible_cavity`) replaces the sliver and its free edge
//! with tetrahedra on those faces. The candidate kept is the one whose star's worst tetrahedron is
//! best, and it goes in only when that strictly raises the worst tetrahedron of the cavity, so the
//! move never makes anything worse. Nothing about the surface changes: a surface face is an outer
//! face of the cavity (a cavity never crosses one), and every new edge has the new point at one
//! end, so no edge joins opposite skins of a wall.
//!
//! **What it cannot do.** It removes the sliver only where a point can see all of it; the worst
//! that remains is measured and reported, a sliver flat to rounding still reaches the mesh upgrade
//! as its named error.

use crate::core::{DEAD, FACE, GHOST};
use crate::error::TetraError;
use crate::flips::Flipper;
use crate::quality::{cross, dot, length, min_dihedral_degrees, normalized, subtract};
use crate::refine_mesh::{centroid_of, circumcentre_of};

/// Distances inward from a sliver's surface faces, as fractions of its mean edge, largest first:
/// a point closer than a quarter of an edge from the fold makes its own slivers, one further than
/// the edge leaves the neighbouring surface faces unseen. A judgement.
const HEIGHT_FRACTIONS: [f64; 4] = [1.0, 0.75, 0.5, 0.25];

impl Flipper {
    /// Adds a point that removes tetrahedron `slot`, if some candidate strictly raises the worst
    /// tetrahedron it replaces. Returns whether a point went in.
    pub(crate) fn add_quality_point(&mut self, slot: u32) -> Result<bool, TetraError> {
        let tet = self.core.tets[slot as usize];
        if tet.iter().any(|&v| v >= DEAD) || tet.contains(&GHOST) {
            return Ok(false);
        }
        let mut best: Option<(f64, Vec<u32>, [f64; 3])> = None;
        for candidate in self.candidates(slot) {
            if !candidate.iter().all(|value| value.is_finite()) {
                continue;
            }
            let Some(cavity) = self.visible_cavity(&[slot], &candidate) else {
                continue;
            };
            let worst = self.star_worst(&cavity, candidate);
            if worst > self.worst_of(&cavity) && best.as_ref().is_none_or(|b| worst > b.0) {
                best = Some((worst, cavity, candidate));
            }
        }
        let Some((_, cavity, point)) = best else {
            return Ok(false);
        };
        if !self.fits(cavity.len(), self.star_size(&cavity)) {
            return Ok(false);
        }
        self.add_steiner(&cavity, point)?;
        Ok(true)
    }

    /// Where a point for this tetrahedron might go, in a fixed order so ties fall the same way
    /// every run: heights above its surface faces, then its circumcentre and its centroid.
    fn candidates(&self, slot: u32) -> Vec<[f64; 3]> {
        let tet = self.core.tets[slot as usize];
        let points = &self.core.points;
        let centroid = centroid_of(points, &tet);
        let mut inward = [0.0; 3];
        // The surface faces' vertices, from which the heights are measured: a far apex of a
        // sliver lying on one small surface face must not pull the base away from that face.
        let mut base = [0.0; 3];
        let (mut base_count, mut edge_total, mut edge_count) = (0.0, 0.0, 0.0);
        for (i, positions) in FACE.iter().enumerate() {
            let n = self.core.neighbors[slot as usize][i];
            if !self.core.tets[n as usize].contains(&GHOST) {
                continue;
            }
            let [a, b, c] = positions.map(|p| points[tet[p] as usize]);
            let normal = normalized(cross(subtract(b, a), subtract(c, a)));
            let toward = dot(normal, subtract(points[tet[i] as usize], a));
            let sign = if toward >= 0.0 { 1.0 } else { -1.0 };
            for corner in [a, b, c] {
                for k in 0..3 {
                    base[k] += corner[k];
                }
                base_count += 1.0;
            }
            for (from, to) in [(a, b), (b, c), (c, a)] {
                edge_total += length(subtract(from, to));
                edge_count += 1.0;
            }
            for k in 0..3 {
                inward[k] += sign * normal[k];
            }
        }
        let inward = normalized(inward);
        let mut result = Vec::with_capacity(HEIGHT_FRACTIONS.len() + 2);
        if inward != [0.0; 3] {
            base = base.map(|value| value / base_count);
            let mean = edge_total / edge_count;
            for fraction in HEIGHT_FRACTIONS {
                let height = mean * fraction;
                result.push([
                    base[0] + inward[0] * height,
                    base[1] + inward[1] * height,
                    base[2] + inward[2] * height,
                ]);
            }
        }
        if let Some(circumcentre) = circumcentre_of(points, &tet) {
            result.push(circumcentre);
        }
        result.push(centroid);
        result
    }

    /// The worst tetrahedron of the star of `point` over the outer faces of `cavity`.
    fn star_worst(&mut self, cavity: &[u32], point: [f64; 3]) -> f64 {
        // The tetrahedra of the star are named by index, so the point is kept in the list only for
        // this measure. The count stays below `u32::MAX`, as the input is capped far below it.
        let index = self.core.points.len() as u32;
        self.core.points.push(point);
        let mut worst = 180.0_f64;
        for &t in cavity {
            let tet = self.core.tets[t as usize];
            for (i, positions) in FACE.iter().enumerate() {
                if cavity.contains(&self.core.neighbors[t as usize][i]) {
                    continue;
                }
                let face = positions.map(|p| tet[p]);
                worst = worst.min(min_dihedral_degrees(
                    &self.core.points,
                    &[face[0], face[1], face[2], index],
                ));
            }
        }
        self.core.points.pop();
        worst
    }
}
