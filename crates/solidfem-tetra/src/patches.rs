//! Flat regions of the input surface. Triangles connected across edges that lie exactly in one
//! plane (and do not fold back on themselves) form a *patch*: a piece of plane whose internal
//! triangulation is not a real constraint, because any triangulation of the same polygon covers
//! the same surface. Only the patch's boundary loops are constraints; `recover.rs` recovers
//! whatever triangulation of the plane the mesh wants inside. `surface.rs` builds these before
//! recovery; every test here is exact.

use std::collections::HashMap;

use robust::{orient2d, Coord};

use crate::predicates::orient;

/// One flat region of the surface.
pub(crate) struct Patch {
    /// The input triangles, wound outward. The first is the plane's reference triangle.
    pub(crate) triangles: Vec<[u32; 3]>,
    /// Every surface vertex in the patch, sorted.
    pub(crate) vertices: Vec<u32>,
    /// Edges on the patch's boundary loops, as `(low, high)` sorted pairs.
    pub(crate) boundary: Vec<(u32, u32)>,
    /// Finds which input triangles a 2D box touches; only built when the patch has several.
    pub(crate) index: Option<PatchIndex>,
}

/// A uniform grid over a patch's triangles in the projection it does not collapse under, so
/// looking for what lies under a mesh edge touches a few triangles instead of all of them.
pub(crate) struct PatchIndex {
    axes: (usize, usize),
    low: [f64; 2],
    scale: [f64; 2],
    cols: usize,
    rows: usize,
    bins: Vec<Vec<u32>>,
}

impl PatchIndex {
    fn build(points: &[[f64; 3]], triangles: &[[u32; 3]], axes: (usize, usize)) -> Self {
        let mut low = [f64::INFINITY; 2];
        let mut high = [f64::NEG_INFINITY; 2];
        for triangle in triangles {
            for &v in triangle {
                let p = project(&points[v as usize], axes);
                for k in 0..2 {
                    low[k] = low[k].min(p[k]);
                    high[k] = high[k].max(p[k]);
                }
            }
        }
        // About one triangle per bin; `sqrt` of a `usize` count fits a `usize` after rounding up.
        let cells = (triangles.len() as f64).sqrt().ceil() as usize;
        let (cols, rows) = (cells.max(1), cells.max(1));
        let extent = |k: usize, count: usize| {
            if high[k] > low[k] {
                count as f64 / (high[k] - low[k])
            } else {
                0.0
            }
        };
        let mut index = Self {
            axes,
            low,
            scale: [extent(0, cols), extent(1, rows)],
            cols,
            rows,
            bins: vec![Vec::new(); cols * rows],
        };
        for (slot, triangle) in triangles.iter().enumerate() {
            let [a, b, c] = triangle.map(|v| project(&points[v as usize], axes));
            let box_low = [a[0].min(b[0]).min(c[0]), a[1].min(b[1]).min(c[1])];
            let box_high = [a[0].max(b[0]).max(c[0]), a[1].max(b[1]).max(c[1])];
            // `slot` fits u32: a patch holds at most the surface's triangles.
            index.for_bins(box_low, box_high, |bin, bins| bins[bin].push(slot as u32));
        }
        index
    }

    /// The triangles whose 2D boxes may overlap the box `[low, high]`.
    pub(crate) fn candidates(&self, low: [f64; 2], high: [f64; 2]) -> Vec<u32> {
        let mut found = Vec::new();
        for row in self.bin(1, low[1], self.rows)..=self.bin(1, high[1], self.rows) {
            for col in self.bin(0, low[0], self.cols)..=self.bin(0, high[0], self.cols) {
                found.extend_from_slice(&self.bins[row * self.cols + col]);
            }
        }
        found.sort_unstable();
        found.dedup();
        found
    }

    pub(crate) fn axes(&self) -> (usize, usize) {
        self.axes
    }

    /// The bin of coordinate `value`, the same monotonic mapping for every query so two boxes
    /// that overlap always share a bin; `as` saturates and the clamp holds the top edge.
    fn bin(&self, axis: usize, value: f64, count: usize) -> usize {
        let cell = ((value - self.low[axis]) * self.scale[axis]) as usize;
        cell.min(count - 1)
    }

    fn for_bins(
        &mut self,
        low: [f64; 2],
        high: [f64; 2],
        mut add: impl FnMut(usize, &mut Vec<Vec<u32>>),
    ) {
        let (lc, hc) = (
            self.bin(0, low[0], self.cols),
            self.bin(0, high[0], self.cols),
        );
        let (lr, hr) = (
            self.bin(1, low[1], self.rows),
            self.bin(1, high[1], self.rows),
        );
        for row in lr..=hr {
            for col in lc..=hc {
                add(row * self.cols + col, &mut self.bins);
            }
        }
    }
}

pub(crate) fn project(point: &[f64; 3], axes: (usize, usize)) -> [f64; 2] {
    [point[axes.0], point[axes.1]]
}

/// The surface split into patches, and the edges that are real constraints.
pub(crate) struct Patches {
    pub(crate) patches: Vec<Patch>,
    /// Surface edges that lie on a patch boundary: every edge recovery must hold, sorted.
    pub(crate) feature_edges: Vec<(u32, u32)>,
    /// For each undirected surface edge, its two triangles.
    pub(crate) edge_triangles: HashMap<u64, [u32; 2]>,
}

/// An axis pair in which triangle `[a, b, c]` does not collapse, by exact test. A triangle with
/// area projects non-degenerately to at least one axis-aligned plane.
fn working_axes(points: &[[f64; 3]], triangle: [u32; 3]) -> (usize, usize) {
    let [a, b, c] = triangle.map(|v| &points[v as usize]);
    for axes in [(0, 1), (1, 2), (0, 2)] {
        if orient2d(coord(a, axes), coord(b, axes), coord(c, axes)) != 0.0 {
            return axes;
        }
    }
    (0, 1)
}

fn coord(p: &[f64; 3], axes: (usize, usize)) -> Coord<f64> {
    Coord {
        x: p[axes.0],
        y: p[axes.1],
    }
}

/// Splits a checked surface into patches.
pub(crate) fn build_patches(points: &[[f64; 3]], triangles: &[[u32; 3]]) -> Patches {
    let mut directed: Vec<(u32, u32, u32)> = Vec::with_capacity(triangles.len() * 3);
    for (index, t) in triangles.iter().enumerate() {
        // `index` fits u32: the surface checks refuse more vertices than u32 holds, and a
        // manifold surface has no more triangles than twice its vertices.
        let index = index as u32;
        for k in 0..3 {
            directed.push((t[k], t[(k + 1) % 3], index));
        }
    }
    directed.sort_unstable();
    let opposite = |a: u32, b: u32| -> u32 {
        // The surface checks found each directed edge's reverse.
        let found = directed.partition_point(|&(x, y, _)| (x, y) < (a, b));
        directed[found].2
    };
    let mut edge_triangles = HashMap::with_capacity(directed.len() / 2);
    for &(a, b, t) in &directed {
        if a < b {
            let u64key = (u64::from(a) << 32) | u64::from(b);
            edge_triangles.insert(u64key, [t, opposite(b, a)]);
        }
    }
    let flat = |t: u32, s: u32, a: u32, b: u32| -> bool {
        let tri = triangles[t as usize];
        let apex_t = *tri.iter().find(|&&v| v != a && v != b).unwrap_or(&a);
        let apex_s = *triangles[s as usize]
            .iter()
            .find(|&&v| v != a && v != b)
            .unwrap_or(&a);
        let [pa, pb, pc] = tri.map(|v| &points[v as usize]);
        if orient(pa, pb, pc, &points[apex_s as usize]) != 0.0 {
            return false;
        }
        // Exactly coplanar: flat only if the two triangles lie on opposite sides of their
        // shared edge, so a fold back over itself stays a real edge.
        let axes = working_axes(points, tri);
        let side = |x: u32| {
            orient2d(
                coord(&points[a as usize], axes),
                coord(&points[b as usize], axes),
                coord(&points[x as usize], axes),
            )
        };
        side(apex_t) * side(apex_s) < 0.0
    };
    let mut component = vec![usize::MAX; triangles.len()];
    let mut patches = Vec::new();
    let mut feature_set = Vec::new();
    for seed in 0..triangles.len() {
        if component[seed] != usize::MAX {
            continue;
        }
        let id = patches.len();
        component[seed] = id;
        let mut stack = vec![seed as u32];
        let mut members: Vec<u32> = Vec::new();
        while let Some(t) = stack.pop() {
            members.push(t);
            let tri = triangles[t as usize];
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                let s = opposite(b, a);
                if component[s as usize] == usize::MAX && flat(t, s, a, b) {
                    component[s as usize] = id;
                    stack.push(s);
                }
            }
        }
        members.sort_unstable();
        patches.push(members);
    }
    let mut built = Vec::with_capacity(patches.len());
    for members in patches {
        let list: Vec<[u32; 3]> = members.iter().map(|&t| triangles[t as usize]).collect();
        let mut vertices: Vec<u32> = list.iter().flatten().copied().collect();
        vertices.sort_unstable();
        vertices.dedup();
        let id = component[members[0] as usize];
        let mut boundary = Vec::new();
        for &t in &members {
            let tri = triangles[t as usize];
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                let s = opposite(b, a);
                if component[s as usize] != id || !flat(t, s, a, b) {
                    boundary.push((a.min(b), a.max(b)));
                }
            }
        }
        boundary.sort_unstable();
        boundary.dedup();
        feature_set.extend_from_slice(&boundary);
        let index = (list.len() > 1)
            .then(|| PatchIndex::build(points, &list, working_axes(points, list[0])));
        built.push(Patch {
            triangles: list,
            vertices,
            boundary,
            index,
        });
    }
    feature_set.sort_unstable();
    feature_set.dedup();
    Patches {
        patches: built,
        feature_edges: feature_set,
        edge_triangles,
    }
}
