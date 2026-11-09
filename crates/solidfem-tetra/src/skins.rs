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
//! **Opposite skins.** A wrong "opposite" costs a few interior points; a wrong "not opposite"
//! silently leaves an element through the wall, so the test leans towards calling an edge through.
//! An interior edge between two surface vertices is through a wall when both of these hold:
//! - an outward facing at each end opposes one at the other, by at least [`OPPOSED_COSINE`]: the
//!   two vertex normals (angle weighted over their triangles), or any pair of faces incident to
//!   the two ends, so a ridge or reflex corner cannot hide a skin behind its average normal;
//! - the component of the edge on the wall normal, `n_u - n_v`, is at most the thin-wall ratio
//!   times the larger of the two local mean surface edge lengths.
//!
//! Which way the edge leaves each vertex is not asked: at a reflex ridge and on a chord through a
//! curved wall that test fails on a real wall. The second condition is the size relation between
//! the chord and the part, and the only one: a block whose surface edges are shorter than half its
//! thickness never qualifies, and a part meshed so coarsely that one element spans it (an
//! 8-vertex cube) does and loses its few interior edges, by the option's own definition of thin.
//! Where "opposite" has no clear answer, this falls on the side of splitting. The one exclusion is
//! an edge lying in a surface plane both of its ends share (a rim quad's other diagonal): it runs
//! along that surface, and a point on it could not be inside the solid.
//!
//! **The split.** Each such edge is replaced by a chain of `layers_through_walls` pieces, its
//! interior points placed on the edge by an edge-ring split (`Flipper::split_edge`), the tetrahedra
//! around the edge the cavity, grown across any flat ring tetrahedron the new point can only be
//! coplanar with. Every new edge has a new interior vertex at one end, so a split removes
//! through-edges and creates none. The stage repeats over what is left while any split is made,
//! and ends in [`crate::TetraError::ThroughEdgeNotSplit`] if an edge remains, so a run that returns
//! success has no interior edge this test calls through. A ring no point on the edge can be
//! starred from (a flat tetrahedron holding an interior vertex in its plane) is retriangulated
//! instead, removing the edge with no point added (`Flipper::remove_through_edge`).
//!
//! **What that does not promise.** Surface edges are fixed, so a surface edge joining the skins at
//! a wall's rim stays, and a tetrahedron touching it can bridge the wall there; the points
//! `wall_fill.rs` adds only shorten that reach, measured and not bounded. And the rule above is
//! this library's own: `tests/refine/sweep.rs` checks it independently, over shapes whose skins
//! are known from how they were built, and pins how many false successes it finds.

use std::collections::HashMap;

use crate::core::{DEAD, GHOST};
use crate::error::TetraError;
use crate::flips::{edge_key, Flipper};
use crate::quality::{cross, dot, length, normalized, subtract};

/// The cosine of the angle between two outward normals at or beyond which two faces count as
/// facing away from each other: 120 degrees. The value is a judgement, between the 90 degrees of
/// adjacent faces at a right-angled corner, which must not count, and the 180 of parallel skins;
/// 120 tolerates a chord leaning 60 degrees through a curved wall.
pub(crate) const OPPOSED_COSINE: f64 = -0.5;

/// Two normals this nearly equal are the same skin facing, at a vertex: well below the angle
/// between facets of a curved wall, well above the rounding of a unit normal.
const SAME_FACING_COSINE: f64 = 1.0 - 1e-6;

/// The normals and local edge lengths that decide which surface vertices face each other.
/// Empty when through-wall layers were not requested, so nothing is ever opposite.
#[derive(Default)]
pub(crate) struct OppositeSkins {
    normals: Vec<[f64; 3]>,
    edge_mean: Vec<f64>,
    /// Each surface vertex's distinct incident facings.
    faces: Vec<Vec<Facing>>,
    ratio: f64,
}

/// One facing at a surface vertex: a unit normal and the flat patch it belongs to, patches being
/// surface triangles connected across coplanar edges. Two faces of one plane in different patches
/// (the base either side of a stem) stay different patches.
#[derive(Clone, Copy)]
pub(crate) struct Facing {
    normal: [f64; 3],
    patch: u32,
}

/// Each surface triangle's flat patch, numbered in triangle order so the numbers are the same on
/// every run; the map from edge to triangles is read only by key.
fn flat_patches(normals: &[[f64; 3]], boundary: &[[u32; 3]]) -> Vec<u32> {
    let mut at_edge: HashMap<(u32, u32), Vec<usize>> = HashMap::with_capacity(boundary.len() * 2);
    for (index, face) in boundary.iter().enumerate() {
        for (a, b) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            at_edge.entry((a.min(b), a.max(b))).or_default().push(index);
        }
    }
    let mut patches = vec![u32::MAX; boundary.len()];
    let mut next = 0_u32;
    for start in 0..boundary.len() {
        if patches[start] != u32::MAX {
            continue;
        }
        patches[start] = next;
        let mut stack = vec![start];
        while let Some(index) = stack.pop() {
            let face = boundary[index];
            for (a, b) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
                let Some(sharing) = at_edge.get(&(a.min(b), a.max(b))) else {
                    continue;
                };
                for &other in sharing {
                    if patches[other] == u32::MAX
                        && dot(normals[index], normals[other]) > SAME_FACING_COSINE
                    {
                        patches[other] = next;
                        stack.push(other);
                    }
                }
            }
        }
        next += 1;
    }
    patches
}

/// Each vertex's distinct incident facings, for [`OppositeSkins`], in a stable order: first seen
/// in the surface's triangle order, one entry per flat patch at the vertex.
pub(crate) fn incident_faces(
    points: &[[f64; 3]],
    boundary: &[[u32; 3]],
    count: usize,
) -> Vec<Vec<Facing>> {
    let normals: Vec<[f64; 3]> = boundary
        .iter()
        .map(|face| {
            let p = face.map(|v| points[v as usize]);
            normalized(cross(subtract(p[1], p[0]), subtract(p[2], p[0])))
        })
        .collect();
    let patches = flat_patches(&normals, boundary);
    let mut faces: Vec<Vec<Facing>> = vec![Vec::new(); count];
    for (index, face) in boundary.iter().enumerate() {
        for &v in face {
            let distinct = &mut faces[v as usize];
            if !distinct.iter().any(|f| f.patch == patches[index]) {
                distinct.push(Facing {
                    normal: normals[index],
                    patch: patches[index],
                });
            }
        }
    }
    faces
}

impl OppositeSkins {
    pub(crate) fn new(
        normals: Vec<[f64; 3]>,
        edge_mean: Vec<f64>,
        faces: Vec<Vec<Facing>>,
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

    /// The through-wall direction of the edge from `u` to `v` if it joins opposite skins: the
    /// vertex normals first (one averaged direction per vertex), then every pair of faces at the
    /// two ends, so a ridge or reflex corner cannot hide a skin behind its average. An edge lying
    /// in a surface plane both ends share is the one exception.
    fn through_axis(&self, points: &[[f64; 3]], u: u32, v: u32) -> Option<[f64; 3]> {
        let (nu, nv) = (
            *self.normals.get(u as usize)?,
            *self.normals.get(v as usize)?,
        );
        let along = subtract(points[v as usize], points[u as usize]);
        let thin = self.ratio * self.edge_mean[u as usize].max(self.edge_mean[v as usize]);
        let (at_u, at_v) = (&self.faces[u as usize], &self.faces[v as usize]);
        if shares_plane(at_u, at_v, along) {
            return None;
        }
        opposed_axis(nu, nv, along, thin).or_else(|| {
            at_u.iter()
                .flat_map(|&a| at_v.iter().map(move |&b| (a.normal, b.normal)))
                .find_map(|(a, b)| opposed_axis(a, b, along, thin))
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

/// The through-wall direction if two outward normals oppose and the segment's component on the
/// wall normal is at most `thin`. Which way the segment leaves each vertex is deliberately not
/// asked: at a reflex ridge and on a chord through a curved wall that test fails on a real wall.
fn opposed_axis(nu: [f64; 3], nv: [f64; 3], along: [f64; 3], thin: f64) -> Option<[f64; 3]> {
    if dot(nu, nv) > OPPOSED_COSINE {
        return None;
    }
    let axis = normalized(subtract(nu, nv));
    (dot(along, axis).abs() <= thin).then_some(axis)
}

/// How far from a shared plane a segment may lean and still be called in it: rounding on a
/// segment's end coordinates of the magnitudes the mesher accepts is far below this, and an edge
/// meant to lean through a wall tilts by far more than this.
const IN_PLANE_RATIO: f64 = 1e-9;

/// Whether both vertices lie on one flat patch and the segment between them lies in its plane: such
/// an edge runs along that surface, not through a wall. Two ends on different patches of one plane
/// (the base either side of a stem) do not count: their chord passes through the solid.
fn shares_plane(at_u: &[Facing], at_v: &[Facing], along: [f64; 3]) -> bool {
    let reach = length(along);
    at_u.iter().any(|a| {
        dot(along, a.normal).abs() <= IN_PLANE_RATIO * reach
            && at_v.iter().any(|b| b.patch == a.patch)
    })
}

/// What one through-edge split did.
pub(crate) struct SplitOutcome {
    pub(crate) added: Vec<u32>,
    pub(crate) axis: [f64; 3],
    /// Whether the edge was actually removed.
    pub(crate) cut: bool,
}
