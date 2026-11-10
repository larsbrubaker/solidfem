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
//! **The second attempt.** When the quality passes have ended, a tetrahedron of four surface
//! vertices that the upgrade to 10-node elements would still refuse gets one more try
//! (`refine.rs` drives it). Waiting for the end leaves alone every part the passes finish on their
//! own, since flips and points often remove a flat tetrahedron a little later. It reaches two cases
//! the ordinary point cannot. Where the wall meets the sliver at a tight angle every height from a
//! quarter of an edge up is already outside the solid, so lower ones are tried
//! ([`LOW_HEIGHT_FRACTIONS`]), and only when nothing higher fits, because a point low over the
//! surface is itself a thin element. And a face of three vertices nearly in a line makes a flat
//! tetrahedron with any apex, so the cavity is grown across such interior faces
//! (`quality_cavity`) and the star is built on sound faces only; a needle face on the surface
//! cannot be taken in, a surface triangle being fixed. A star any of whose tetrahedra the upgrade
//! would refuse is not accepted.
//!
//! **The second attempt and the layers.** It needs no rule of its own for thin walls. Every new
//! tetrahedron joins the new point to a face that already existed, so no element or edge through
//! a wall appears that was not there before, the property `skins.rs` establishes. Its point is not
//! a layer point, and like the ordinary quality point it is not placed on a layer plane. A flat
//! tetrahedron with a vertex off the surface is left alone: in a layered wall those lie in a layer
//! plane, and points for them landed between the planes (`tests/refine/wall_skins.rs`).
//!
//! **What it cannot do.** It removes the sliver only where a point can see all of it; the worst
//! that remains is measured and reported, and a sliver flat to rounding still reaches the mesh
//! upgrade as its named error. A removed flat element is replaced by real but possibly very thin
//! ones: the point goes in when it strictly raises the worst, so a star may measure a small
//! fraction of a degree. A flat element no longer stops the mesh, but a poor one stays poor.

use crate::core::{DEAD, FACE, GHOST};
use crate::error::TetraError;
use crate::flips::Flipper;
use crate::interior::MAX_CAVITY;
use crate::predicates::orient;
use crate::quality::{
    cross, length, min_dihedral_degrees, normalized, refused_by_upgrade, subtract,
};
use crate::refine_mesh::{centroid_of, circumcentre_of};

/// Distances inward from a sliver's surface faces, as fractions of its mean edge, largest first:
/// a point closer than a quarter of an edge from the fold makes its own slivers, one further than
/// the edge leaves the neighbouring surface faces unseen. A judgement.
const HEIGHT_FRACTIONS: [f64; 4] = [1.0, 0.75, 0.5, 0.25];

/// The heights tried only when none of [`HEIGHT_FRACTIONS`] improves the tetrahedron, largest
/// first: where a wall meets the sliver at a tight angle every point a quarter of an edge up is
/// already outside the solid, yet a lower one still sees both surface faces. The star's own
/// quality decides among them, so the lowest is taken only when it is all that fits.
const LOW_HEIGHT_FRACTIONS: [f64; 5] = [0.125, 0.0625, 0.03125, 0.015625, 0.0078125];

impl Flipper {
    /// Adds a point that removes tetrahedron `slot`, if some candidate strictly raises the worst
    /// tetrahedron it replaces. Returns whether a point went in.
    pub(crate) fn add_quality_point(&mut self, slot: u32) -> Result<bool, TetraError> {
        let tet = self.core.tets[slot as usize];
        if tet.iter().any(|&v| v >= DEAD) || tet.contains(&GHOST) {
            return Ok(false);
        }
        let mut best: Option<(f64, Vec<u32>, [f64; 3])> = None;
        for candidate in self.candidates(slot, &HEIGHT_FRACTIONS) {
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

    /// Whether tetrahedron `slot` is what the second attempt is for: alive, refused by the upgrade
    /// to 10-node elements, with all four vertices on the surface.
    pub(crate) fn needs_rescue(&self, slot: u32, fixed: &[bool]) -> bool {
        let tet = self.core.tets[slot as usize];
        tet.iter().all(|&v| v < DEAD)
            && refused_by_upgrade(&self.core.points, &tet)
            && tet
                .iter()
                .all(|&v| fixed.get(v as usize).copied().unwrap_or(false))
    }

    /// Makes the second attempt on tetrahedron `slot`. Returns whether a point went in.
    pub(crate) fn rescue_tet(&mut self, slot: u32) -> Result<bool, TetraError> {
        let Some((_, cavity, point)) = self.rescue_point(slot) else {
            return Ok(false);
        };
        if !self.fits(cavity.len(), self.star_size(&cavity)) {
            return Ok(false);
        }
        self.add_steiner(&cavity, point)?;
        Ok(true)
    }

    /// The second attempt on a flat tetrahedron of four surface vertices: the ordinary heights and
    /// then lower ones, each with a cavity grown past needle faces, keeping the candidate whose star
    /// is best. A list that yields a point ends the search, so the low heights are used only where
    /// nothing higher fits.
    fn rescue_point(&mut self, slot: u32) -> Option<(f64, Vec<u32>, [f64; 3])> {
        for fractions in [HEIGHT_FRACTIONS.as_slice(), LOW_HEIGHT_FRACTIONS.as_slice()] {
            let mut best: Option<(f64, Vec<u32>, [f64; 3])> = None;
            for candidate in self.candidates(slot, fractions) {
                if !candidate.iter().all(|value| value.is_finite()) {
                    continue;
                }
                let Some(cavity) = self.quality_cavity(slot, candidate) else {
                    continue;
                };
                if self.star_refused(&cavity, candidate) {
                    continue;
                }
                let worst = self.star_worst(&cavity, candidate);
                if worst > self.worst_of(&cavity) && best.as_ref().is_none_or(|b| worst > b.0) {
                    best = Some((worst, cavity, candidate));
                }
            }
            if best.is_some() {
                return best;
            }
        }
        None
    }

    /// The cavity for a point at `point` replacing tetrahedron `seed`: what the point must see
    /// (`Flipper::visible_cavity`), grown across any interior face a star tetrahedron on it would
    /// be flat on. A face of three vertices in a line reads flat with any apex, so a sliver beside
    /// one could never be removed by a star built on that face; taking the tetrahedron across it
    /// into the cavity keeps that face out of the star. `None` under the same limits as
    /// `visible_cavity`. Faces on the surface are never grown across, so a star on a degenerate
    /// surface triangle still reads flat and the caller's quality test refuses it.
    fn quality_cavity(&mut self, seed: u32, point: [f64; 3]) -> Option<Vec<u32>> {
        let mut cavity = self.visible_cavity(&[seed], &point)?;
        loop {
            // The vertex count stays below `u32::MAX`, as the input is capped far below it.
            let index = self.core.points.len() as u32;
            self.core.points.push(point);
            let blocked = cavity.iter().copied().find_map(|t| {
                let tet = self.core.tets[t as usize];
                FACE.iter().enumerate().find_map(|(i, positions)| {
                    let n = self.core.neighbors[t as usize][i];
                    if cavity.contains(&n) || self.core.tets[n as usize].contains(&GHOST) {
                        return None;
                    }
                    let face = positions.map(|p| tet[p]);
                    let star_tet = [face[0], face[1], face[2], index];
                    refused_by_upgrade(&self.core.points, &star_tet).then_some(n)
                })
            });
            self.core.points.pop();
            match blocked {
                None => return Some(cavity),
                Some(_) if cavity.len() >= MAX_CAVITY => return None,
                Some(n) => {
                    cavity.push(n);
                    cavity = self.visible_cavity(&cavity, &point)?;
                }
            }
        }
    }

    /// Where a point for this tetrahedron might go, in a fixed order so ties fall the same way
    /// every run: `fractions` of a mean surface edge above its surface faces, then its circumcentre
    /// and its centroid.
    fn candidates(&self, slot: u32, fractions: &[f64]) -> Vec<[f64; 3]> {
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
            // Which side of this face the sliver lies on is the exact orientation of its fourth
            // vertex: for a sliver flat to rounding a dot product computed here is noise and
            // would point the candidates out of the solid.
            let sign = if orient(&a, &b, &c, &points[tet[i] as usize]) > 0.0 {
                -1.0
            } else {
                1.0
            };
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
        let mut result = Vec::with_capacity(fractions.len() + 2);
        if inward != [0.0; 3] {
            base = base.map(|value| value / base_count);
            let mean = edge_total / edge_count;
            for &fraction in fractions {
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

    /// Whether the star of `point` over the outer faces of `cavity` has a tetrahedron the upgrade
    /// would refuse. Such a star only moves the refusal.
    fn star_refused(&mut self, cavity: &[u32], point: [f64; 3]) -> bool {
        // The vertex count stays below `u32::MAX`, as the input is capped far below it.
        let index = self.core.points.len() as u32;
        self.core.points.push(point);
        let refused = cavity.iter().any(|&t| {
            let tet = self.core.tets[t as usize];
            FACE.iter().enumerate().any(|(i, positions)| {
                if cavity.contains(&self.core.neighbors[t as usize][i]) {
                    return false;
                }
                let face = positions.map(|p| tet[p]);
                refused_by_upgrade(&self.core.points, &[face[0], face[1], face[2], index])
            })
        });
        self.core.points.pop();
        refused
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
