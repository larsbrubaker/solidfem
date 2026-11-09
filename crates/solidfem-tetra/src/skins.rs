//! Which surface vertices stand on opposite skins of a thin wall, and the removal of every edge
//! joining two of them. `refine.rs` runs the removal after the layer points are in; `quality.rs`
//! asks the same question so no flip can put such an edge back.
//!
//! **Why edges.** An element with a vertex on each skin spans the whole wall, and it can only
//! exist if the mesh has an edge joining those two vertices. Layer points alone do not remove those
//! edges: on a surface whose two skins carry different triangulations, the star of a layer point
//! leaves faces joining the skins, and those faces keep their edges. So the rule is stated on the
//! edges: no interior edge joins opposite skins.
//!
//! **Opposite skins.** Two surface vertices are on opposite skins of a thin wall when all three of
//! these hold:
//! - their outward normals (angle weighted over their surface triangles) oppose, by at least
//!   [`OPPOSED_COSINE`];
//! - the segment between them leaves each vertex going inward, against its own normal;
//! - the wall measured along the normals, the component of that segment on `n_u - n_v`, is at most
//!   the thin-wall ratio times the larger of the two local mean surface edge lengths, the same
//!   measure the layer rays use.
//!
//! The first condition is what makes rims, corners and junctions come out right without a special
//! case. A vertex on a ridge averages the faces meeting there, so its normal sits between them:
//! the two ends of a plate's rim have normals about 90 degrees apart and are never opposite, and
//! neither are the faces meeting at any convex or reflex corner of a part. The third condition
//! measures across the wall and ignores how far apart the vertices are along it, because a long
//! edge leaning through a thin wall is still an edge through it, and it is what leaves a solid
//! block alone: its faces are further apart than the ratio allows.
//!
//! **The split.** Each such edge is replaced by a chain of `layers_through_walls` pieces, its
//! interior points placed on the edge by an edge-ring split (`Flipper::split_edge`): the tetrahedra
//! around the edge are the cavity, so nothing away from the edge changes. Every new edge has a new
//! interior vertex at one end, so a split removes one through-edge and creates none, and the
//! stage ends after exactly as many splits as there were through-edges. That is what guarantees no
//! tetrahedron has a vertex pair on opposite skins, hence at least two tetrahedra through any
//! column of the wall; more requested layers lengthen the chain, and the guarantee stays two.

use crate::core::{DEAD, GHOST};
use crate::error::TetraError;
use crate::flips::{edge_key, Flipper};
use crate::quality::{dot, normalized, subtract};

/// The cosine of the angle between two outward normals at or beyond which two vertices count as
/// facing away from each other: 120 degrees. A vertex on a right-angled rim has a normal 45 degrees
/// from its own skin, so its partner on the other skin is 90 degrees away and clearly not opposite.
const OPPOSED_COSINE: f64 = -0.5;

/// The normals and local edge lengths that decide which surface vertices face each other.
/// Empty when through-wall layers were not requested, so nothing is ever opposite.
#[derive(Default)]
pub(crate) struct OppositeSkins {
    normals: Vec<[f64; 3]>,
    edge_mean: Vec<f64>,
    ratio: f64,
}

impl OppositeSkins {
    pub(crate) fn new(normals: Vec<[f64; 3]>, edge_mean: Vec<f64>, ratio: f64) -> Self {
        Self {
            normals,
            edge_mean,
            ratio,
        }
    }

    /// Whether an edge from `u` to `v` would join opposite skins of a thin wall. Interior vertices
    /// have no normal and points added later have no entry, so only surface vertices can qualify.
    pub(crate) fn joins(&self, points: &[[f64; 3]], u: u32, v: u32) -> bool {
        let (Some(&nu), Some(&nv)) = (self.normals.get(u as usize), self.normals.get(v as usize))
        else {
            return false;
        };
        if dot(nu, nv) > OPPOSED_COSINE {
            return false;
        }
        let along = subtract(points[v as usize], points[u as usize]);
        if dot(along, nu) >= 0.0 || dot(along, nv) <= 0.0 {
            return false;
        }
        let across = dot(along, normalized(subtract(nu, nv))).abs();
        across <= self.ratio * self.edge_mean[u as usize].max(self.edge_mean[v as usize])
    }

    /// The through-wall direction between two opposite vertices, normal to the wall they bound,
    /// which keeps a point split from their edge at its level when it is smoothed.
    fn axis(&self, u: u32, v: u32) -> [f64; 3] {
        normalized(subtract(self.normals[u as usize], self.normals[v as usize]))
    }

    /// Every interior edge joining opposite skins, sorted so the work order is the same every run.
    pub(crate) fn through_edges(&self, flipper: &Flipper) -> Vec<(u32, u32)> {
        let core = &flipper.core;
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for tet in core
            .tets
            .iter()
            .filter(|t| !t.contains(&DEAD) && !t.contains(&GHOST))
        {
            for a in 0..4 {
                for b in a + 1..4 {
                    let (u, v) = (tet[a].min(tet[b]), tet[a].max(tet[b]));
                    if !flipper.locked_edges.contains(&edge_key(u, v))
                        && self.joins(&core.points, u, v)
                    {
                        edges.push((u, v));
                    }
                }
            }
        }
        edges.sort_unstable();
        edges.dedup();
        edges
    }

    /// Splits one through-edge into `layers` pieces. Returns the new points with the axis each
    /// must keep to, the count of pieces cut, and whether the edge itself could be cut at all.
    pub(crate) fn split_through_edge(
        &self,
        flipper: &mut Flipper,
        edge: (u32, u32),
        layers: u32,
    ) -> Result<SplitOutcome, TetraError> {
        let (p, q) = edge;
        let axis = self.axis(p, q);
        let mut added = Vec::new();
        let mut from = p;
        for placed in 0..layers.saturating_sub(1) {
            let remaining = f64::from(layers - placed);
            let Some(point) = flipper.split_edge(from, q, 1.0 / remaining)? else {
                return Ok(SplitOutcome {
                    added,
                    axis,
                    cut: placed > 0,
                });
            };
            added.push(point);
            from = point;
        }
        Ok(SplitOutcome {
            added,
            axis,
            cut: true,
        })
    }
}

/// What one through-edge split did.
pub(crate) struct SplitOutcome {
    pub(crate) added: Vec<u32>,
    pub(crate) axis: [f64; 3],
    /// Whether the edge was actually removed.
    pub(crate) cut: bool,
}
