//! Which surface vertices stand on opposite skins of a thin wall, and the removal of every edge
//! joining two of them. `refine.rs` runs the removal after the layer points are in; `quality.rs`
//! asks the same question so no flip can put such an edge back. A wall face wide enough to have no
//! vertices inside it gets its points from `wall_fill.rs`.
//!
//! **Why edges.** An element with a vertex on each skin spans the whole wall, and it can only
//! exist if the mesh has an edge joining those two vertices. Layer points alone do not remove those
//! edges: on a surface whose two skins carry different triangulations, the star of a layer point
//! leaves faces joining the skins, and those faces keep their edges. So the rule is stated on the
//! edges: no interior edge joins opposite skins.
//!
//! **Opposite skins, by the vertex normals.** Two surface vertices are on opposite skins of a thin
//! wall when all three of these hold:
//! - their outward normals (angle weighted over their surface triangles) oppose, by at least
//!   [`OPPOSED_COSINE`];
//! - the segment between them leaves each vertex going inward, against its own normal;
//! - the wall measured along the normals, the component of that segment on `n_u - n_v`, is at most
//!   the thin-wall ratio times the larger of the two local mean surface edge lengths, the same
//!   measure the layer rays use.
//!
//! A vertex on a ridge averages the faces meeting there, so its normal sits between them: the two
//! ends of a plate's rim have normals about 90 degrees apart and are never opposite by this test,
//! and neither are the faces meeting at any convex or reflex corner of a part. The third condition
//! measures across the wall and ignores how far apart the vertices are along it, because a long
//! edge leaning through a thin wall is still an edge through it, and it is what leaves a solid
//! block alone: its faces are further apart than the ratio allows.
//!
//! **Opposite skins, by the faces at a rim.** That test cannot see a wall whose skins are large
//! faces with vertices only at their rims, because every vertex then is a rim vertex and its
//! average normal is about 45 degrees off either skin. So the same three conditions are also tried
//! for each pair of surface faces incident to the two vertices, with the faces' own normals, and
//! with one more condition that keeps chunky parts out: the longer of the two faces' longest edges
//! must be at least [`WALL_WIDTH_RATIO`] times the wall. A solid block's opposite faces are about
//! as far apart as they are wide, so they never qualify, and an edge lying in a side face shared by
//! both vertices (a rim quad's other diagonal) is excluded outright: it runs along the rim, not
//! through the wall.
//!
//! **The split.** Each such edge is replaced by a chain of `layers_through_walls` pieces, its
//! interior points placed on the edge by an edge-ring split (`Flipper::split_edge`), the tetrahedra
//! around the edge the cavity, grown across any flat ring tetrahedron the new point can only be
//! coplanar with. Every new edge has a new interior vertex at one end, so a split removes
//! through-edges and creates none. The stage repeats over what is left while any split is made,
//! and ends in [`crate::TetraError::ThroughEdgeNotSplit`] if an edge remains, so a run that returns
//! success has no interior edge joining opposite skins. A ring no point on the edge can be starred
//! from (a flat tetrahedron holding an interior vertex in its plane) is retriangulated instead,
//! removing the edge with no point added (`Flipper::remove_through_edge`). An edge joining opposite
//! skins that remains is a surface edge, at a wall's rim, which the fixed surface does not allow to
//! be cut, so a tetrahedron bridging a wall has to touch the rim, and the points `wall_fill.rs`
//! adds are what shorten its reach from there (measured, not bounded, in `wall_fill.rs`).

use crate::core::{DEAD, GHOST};
use crate::error::TetraError;
use crate::flips::{edge_key, Flipper};
use crate::quality::{cross, dot, length, normalized, subtract};

/// The cosine of the angle between two outward normals at or beyond which two vertices count as
/// facing away from each other: 120 degrees. A vertex on a right-angled rim has a normal 45 degrees
/// from its own skin, so its partner on the other skin is 90 degrees away and clearly not opposite.
pub(crate) const OPPOSED_COSINE: f64 = -0.5;

/// A wall is only treated as one by its faces when the longest edge of one of the two faces is at
/// least this many wall thicknesses wide: thin against the skin, wider than a chunky part can be.
pub(crate) const WALL_WIDTH_RATIO: f64 = 8.0;

/// Two normals this nearly equal are the same skin facing, at a vertex.
const SAME_FACING_COSINE: f64 = 1.0 - 1e-6;

/// A surface triangle's facing and width, for the face test of a wall.
#[derive(Clone, Copy)]
pub(crate) struct SkinFace {
    pub(crate) normal: [f64; 3],
    pub(crate) longest_edge: f64,
}

/// The normals and local edge lengths that decide which surface vertices face each other.
/// Empty when through-wall layers were not requested, so nothing is ever opposite.
#[derive(Default)]
pub(crate) struct OppositeSkins {
    normals: Vec<[f64; 3]>,
    edge_mean: Vec<f64>,
    /// Each surface vertex's distinct incident faces, the widest of each facing.
    faces: Vec<Vec<SkinFace>>,
    ratio: f64,
}

/// Each vertex's distinct incident faces, for [`OppositeSkins`], in a stable order: normals are
/// first seen in the surface's triangle order.
pub(crate) fn incident_faces(
    points: &[[f64; 3]],
    boundary: &[[u32; 3]],
    count: usize,
) -> Vec<Vec<SkinFace>> {
    let mut faces: Vec<Vec<SkinFace>> = vec![Vec::new(); count];
    for face in boundary {
        let p = face.map(|v| points[v as usize]);
        let normal = normalized(cross(subtract(p[1], p[0]), subtract(p[2], p[0])));
        let longest_edge = [(0, 1), (1, 2), (2, 0)]
            .map(|(a, b)| length(subtract(p[a], p[b])))
            .iter()
            .fold(0.0_f64, |longest, &edge| longest.max(edge));
        for &v in face {
            let distinct = &mut faces[v as usize];
            match distinct
                .iter()
                .position(|f| dot(f.normal, normal) > SAME_FACING_COSINE)
            {
                Some(i) => distinct[i].longest_edge = distinct[i].longest_edge.max(longest_edge),
                None => distinct.push(SkinFace {
                    normal,
                    longest_edge,
                }),
            }
        }
    }
    faces
}

impl OppositeSkins {
    pub(crate) fn new(
        normals: Vec<[f64; 3]>,
        edge_mean: Vec<f64>,
        faces: Vec<Vec<SkinFace>>,
        ratio: f64,
    ) -> Self {
        Self {
            normals,
            edge_mean,
            faces,
            ratio,
        }
    }

    /// Whether an edge from `u` to `v` would join opposite skins of a thin wall. Interior vertices
    /// have no normal and points added later have no entry, so only surface vertices can qualify.
    pub(crate) fn joins(&self, points: &[[f64; 3]], u: u32, v: u32) -> bool {
        self.through_axis(points, u, v).is_some()
    }

    /// The through-wall direction of the edge from `u` to `v` if it joins opposite skins.
    fn through_axis(&self, points: &[[f64; 3]], u: u32, v: u32) -> Option<[f64; 3]> {
        let (nu, nv) = (
            *self.normals.get(u as usize)?,
            *self.normals.get(v as usize)?,
        );
        let along = subtract(points[v as usize], points[u as usize]);
        let thin = self.ratio * self.edge_mean[u as usize].max(self.edge_mean[v as usize]);
        if let Some(axis) = opposed_axis(nu, nv, along, thin) {
            return Some(axis);
        }
        let (at_u, at_v) = (&self.faces[u as usize], &self.faces[v as usize]);
        if shares_plane(at_u, at_v, along) {
            return None;
        }
        at_u.iter()
            .flat_map(|a| at_v.iter().map(move |b| (a, b)))
            .find_map(|(a, b)| {
                let axis = opposed_axis(a.normal, b.normal, along, thin)?;
                let wide_enough =
                    dot(along, axis).abs() * WALL_WIDTH_RATIO <= a.longest_edge.max(b.longest_edge);
                wide_enough.then_some(axis)
            })
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
        let axis = self
            .through_axis(&flipper.core.points, p, q)
            .unwrap_or([0.0; 3]);
        let mut added = Vec::new();
        let mut from = p;
        for placed in 0..layers.saturating_sub(1) {
            let remaining = f64::from(layers - placed);
            let Some(point) = flipper.split_edge(from, q, 1.0 / remaining)? else {
                // A ring a point on the edge cannot be starred from can still be retriangulated
                // without the edge, so the edge goes either way; chains stop where they are.
                let cut = placed > 0 || flipper.remove_through_edge(from, q, self)?;
                return Ok(SplitOutcome { added, axis, cut });
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

/// The through-wall direction if two outward normals oppose, the segment leaves each vertex going
/// inward, and the segment's component on the wall normal is at most `thin`.
fn opposed_axis(nu: [f64; 3], nv: [f64; 3], along: [f64; 3], thin: f64) -> Option<[f64; 3]> {
    if dot(nu, nv) > OPPOSED_COSINE {
        return None;
    }
    if dot(along, nu) >= 0.0 || dot(along, nv) <= 0.0 {
        return None;
    }
    let axis = normalized(subtract(nu, nv));
    (dot(along, axis).abs() <= thin).then_some(axis)
}

/// How far from a shared plane a segment may lean and still be called in it: rounding on a
/// segment's end coordinates of the magnitudes the mesher accepts is far below this, and an edge
/// meant to lean through a wall tilts by far more than this.
const IN_PLANE_RATIO: f64 = 1e-9;

/// Whether both vertices have a face in one plane and the segment between them lies in it: such
/// an edge runs along a side wall, not through a wall.
fn shares_plane(at_u: &[SkinFace], at_v: &[SkinFace], along: [f64; 3]) -> bool {
    let reach = length(along);
    at_u.iter().any(|a| {
        dot(along, a.normal).abs() <= IN_PLANE_RATIO * reach
            && at_v
                .iter()
                .any(|b| dot(a.normal, b.normal) > SAME_FACING_COSINE)
    })
}

/// What one through-edge split did.
pub(crate) struct SplitOutcome {
    pub(crate) added: Vec<u32>,
    pub(crate) axis: [f64; 3],
    /// Whether the edge was actually removed.
    pub(crate) cut: bool,
}
