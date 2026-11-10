//! Where the points through a thin wall go. Delaunay refinement driven by a size alone would need
//! edges as short as the wall to put a vertex inside it, a thousand times the elements a thin
//! wall needs. Instead, each surface vertex looks straight across the wall and names the points
//! that divide the wall into the requested number of element layers.
//!
//! **Finding the wall.** From a surface vertex, along the inward angle-weighted normal of its
//! triangles, the nearest surface triangle not touching the vertex is the far side of the wall,
//! and the distance to it is the local thickness. Ray tests are plain floating point: a position
//! is only proposed here. Whether it can be a mesh vertex is decided exactly by `interior.rs`.
//! A hit from outside the solid (the ray left through a nearby reflex corner) means the
//! vertex is not facing a wall, and it proposes nothing.
//!
//! **When it counts as a wall.** The ray decides only where the thickness is at most the
//! requested ratio of the local surface edge length, a wall that isotropic elements of the
//! surface's own size cannot resolve.
//!
//! **Corners and junctions.** Both faces of a plate would each propose the same points, and the
//! rays from a rim or wall junction fan out, so a proposal is dropped when it is close to an
//! accepted point in the sense of the layer: within half the surface edge across the wall's
//! plane and half a layer through it, measured along its own ray. A proposal within a quarter
//! layer of any surface vertex is dropped too, keeping points off the surface.
//!
//! **A wall too thin for the numbers.** Below [`min_layer_thickness`] the coordinates of the
//! points dividing the wall would be separated by only a few units in the last place of the part
//! size, so the wall gets no layers at all: it is counted in the report, no point is added and no
//! error is raised. The mesh stays exactly valid with the wall one element thick there.
//!
//! [`layer_run`] prepares the stage `refine.rs` runs: the ray tree, the separation grid and each
//! surface vertex's normal and mean edge length.

use std::collections::HashMap;

use crate::quality::{cross, dot, length, normalized, subtract};

/// A surface triangle, with its vertex indices so a ray can ignore the ones at its start.
#[derive(Clone, Copy)]
struct Triangle {
    corners: [[f64; 3]; 3],
    vertices: [u32; 3],
}

#[derive(Clone, Copy)]
struct Node {
    low: [f64; 3],
    high: [f64; 3],
    /// The left child, or the first triangle of a leaf.
    first: u32,
    /// The right child of an inner node.
    second: u32,
    /// Triangles in a leaf; zero for an inner node.
    count: u32,
}

/// The surface triangles indexed for straight-line queries, a bounding volume tree over them.
pub(crate) struct SurfaceRays {
    triangles: Vec<Triangle>,
    nodes: Vec<Node>,
}

/// The nearest surface crossing along a ray.
#[derive(Clone, Copy)]
pub(crate) struct Hit {
    pub(crate) distance: f64,
    /// Whether the ray reached this triangle from inside the solid.
    pub(crate) from_inside: bool,
    /// The triangle's unit normal, the through-wall direction a layer point stays in plane with.
    pub(crate) normal: [f64; 3],
}

const LEAF: usize = 4;

impl SurfaceRays {
    pub(crate) fn new(points: &[[f64; 3]], triangles: &[[u32; 3]]) -> Self {
        let mut items: Vec<Triangle> = triangles
            .iter()
            .map(|t| Triangle {
                corners: t.map(|v| points[v as usize]),
                vertices: *t,
            })
            .collect();
        let mut nodes = Vec::with_capacity(items.len());
        let len = items.len();
        build(&mut items, 0, len, &mut nodes);
        Self {
            triangles: items,
            nodes,
        }
    }

    /// The nearest triangle crossed from `origin` along unit `direction`, ignoring triangles at
    /// the vertex the ray starts from.
    pub(crate) fn nearest(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        vertex: u32,
    ) -> Option<Hit> {
        self.nearest_where(origin, direction, |t| t.vertices.contains(&vertex))
    }

    /// The nearest triangle crossed from `origin`, a point on triangle `face`, along unit
    /// `direction`, ignoring `face` itself.
    pub(crate) fn nearest_off_face(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        face: [u32; 3],
    ) -> Option<Hit> {
        self.nearest_where(origin, direction, |t| t.vertices == face)
    }

    fn nearest_where(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        skip: impl Fn(&Triangle) -> bool,
    ) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        let mut stack = vec![0usize];
        while let Some(node_index) = stack.pop() {
            let node = self.nodes[node_index];
            let reach = best.map_or(f64::INFINITY, |hit| hit.distance);
            if !box_hit(node.low, node.high, origin, direction, reach) {
                continue;
            }
            if node.count == 0 {
                stack.push(node.second as usize);
                stack.push(node.first as usize);
                continue;
            }
            for triangle in &self.triangles[node.first as usize..(node.first + node.count) as usize]
            {
                if skip(triangle) {
                    continue;
                }
                let Some(hit) = ray_triangle(triangle.corners, origin, direction) else {
                    continue;
                };
                if hit.distance > 0.0 && best.is_none_or(|b| hit.distance < b.distance) {
                    best = Some(hit);
                }
            }
        }
        best
    }
}

/// Builds the node for `items[from..to]`, splitting at the median of the widest axis. Sorting is
/// by centre with the vertex indices as the tie-break, so the tree is the same on every run.
fn build(items: &mut [Triangle], from: usize, to: usize, nodes: &mut Vec<Node>) -> usize {
    let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for triangle in &items[from..to] {
        for corner in triangle.corners {
            for k in 0..3 {
                low[k] = low[k].min(corner[k]);
                high[k] = high[k].max(corner[k]);
            }
        }
    }
    let index = nodes.len();
    nodes.push(Node {
        low,
        high,
        first: 0,
        second: 0,
        count: 0,
    });
    if to - from <= LEAF {
        // `from` and the leaf size fit u32: triangles are indexed by u32.
        nodes[index].first = from as u32;
        nodes[index].count = (to - from) as u32;
        return index;
    }
    let extent = [high[0] - low[0], high[1] - low[1], high[2] - low[2]];
    let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] {
        0
    } else if extent[1] >= extent[2] {
        1
    } else {
        2
    };
    items[from..to].sort_unstable_by(|a, b| {
        let key = |t: &Triangle| t.corners[0][axis] + t.corners[1][axis] + t.corners[2][axis];
        key(a)
            .total_cmp(&key(b))
            .then_with(|| a.vertices.cmp(&b.vertices))
    });
    let middle = from + (to - from) / 2;
    let left = build(items, from, middle, nodes);
    let right = build(items, middle, to, nodes);
    // Node indices fit u32: there are fewer than two nodes per triangle.
    nodes[index].first = left as u32;
    nodes[index].second = right as u32;
    index
}

/// Whether the ray from `origin` along `direction` meets the box before `reach`.
fn box_hit(
    low: [f64; 3],
    high: [f64; 3],
    origin: [f64; 3],
    direction: [f64; 3],
    reach: f64,
) -> bool {
    let (mut enter, mut leave) = (0.0_f64, reach);
    for k in 0..3 {
        let d = direction[k];
        if d == 0.0 {
            if origin[k] < low[k] || origin[k] > high[k] {
                return false;
            }
            continue;
        }
        let (mut a, mut b) = ((low[k] - origin[k]) / d, (high[k] - origin[k]) / d);
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        enter = enter.max(a);
        leave = leave.min(b);
        if enter > leave {
            return false;
        }
    }
    true
}

/// Plain floating point ray against triangle, with a tiny allowance so a ray landing on the
/// shared edge of two triangles is not missed by both.
fn ray_triangle(corners: [[f64; 3]; 3], origin: [f64; 3], direction: [f64; 3]) -> Option<Hit> {
    let [a, b, c] = corners;
    let (e1, e2) = (subtract(b, a), subtract(c, a));
    let p = cross(direction, e2);
    let det = dot(e1, p);
    if det == 0.0 {
        return None;
    }
    let inverse = 1.0 / det;
    let s = subtract(origin, a);
    let u = dot(s, p) * inverse;
    let q = cross(s, e1);
    let v = dot(direction, q) * inverse;
    let allowance = 1e-9;
    if u < -allowance || v < -allowance || u + v > 1.0 + allowance {
        return None;
    }
    // The distance from the triangle's unit plane rather than the barycentric form, so a wall
    // whose plane is exactly axis aligned gets its exact thickness and every layer point on it
    // the same coordinate: a one-unit wobble in the plane of a layer makes slivers.
    let unit = crate::quality::normalized(cross(e1, e2));
    let distance = dot(unit, subtract(a, origin)) / dot(unit, direction);
    if !distance.is_finite() {
        return None;
    }
    // A triangle wound counterclockwise from outside has its normal outward, so a ray going the
    // same way as the normal is leaving the solid.
    Some(Hit {
        distance,
        from_inside: dot(unit, direction) > 0.0,
        normal: unit,
    })
}

/// Below this layer thickness the point coordinates differ by too few units in the last place
/// of the part's largest coordinate for the positions to mean anything: 2^16 units, or about
/// 1.5e-11 of the magnitude.
pub(crate) fn min_layer_thickness(magnitude: f64) -> f64 {
    magnitude * f64::EPSILON * f64::from(1u32 << 16)
}

/// The accepted proposals and the surface vertices, in a coarse hash grid for the
/// minimum-separation rule. Only membership is read, so its order cannot reach the output.
pub(crate) struct Separation {
    cell: f64,
    buckets: HashMap<(i64, i64, i64), Vec<Accepted>>,
}

#[derive(Clone, Copy)]
struct Accepted {
    point: [f64; 3],
    /// The ray's direction for a layer point, zero for a surface vertex (no direction matters).
    axis: [f64; 3],
    across: f64,
    through: f64,
}

impl Separation {
    /// A grid of cells about as wide as a surface edge.
    pub(crate) fn new(mean_edge: f64) -> Self {
        Self {
            cell: mean_edge.max(f64::MIN_POSITIVE),
            buckets: HashMap::new(),
        }
    }

    fn key(&self, point: [f64; 3]) -> (i64, i64, i64) {
        // `as` saturates, and a coordinate over cell bounded far inside i64 by the input check.
        let at = |value: f64| (value / self.cell).floor() as i64;
        (at(point[0]), at(point[1]), at(point[2]))
    }

    /// Records a surface vertex, which every layer point must clear by a quarter of its layer.
    pub(crate) fn add_surface(&mut self, point: [f64; 3]) {
        let key = self.key(point);
        self.buckets.entry(key).or_default().push(Accepted {
            point,
            axis: [0.0; 3],
            across: 0.0,
            through: 0.0,
        });
    }

    /// Whether `point` on its ray clears everything already kept; if so it is kept too.
    pub(crate) fn accept(
        &mut self,
        point: [f64; 3],
        axis: [f64; 3],
        across: f64,
        through: f64,
    ) -> bool {
        let reach = across.max(through);
        // `as` saturates; the query radius is bounded by the wall thickness over the cell width.
        let span = (reach / self.cell).ceil() as i64 + 1;
        let key = self.key(point);
        for ix in key.0 - span..=key.0 + span {
            for iy in key.1 - span..=key.1 + span {
                for iz in key.2 - span..=key.2 + span {
                    let Some(bucket) = self.buckets.get(&(ix, iy, iz)) else {
                        continue;
                    };
                    if bucket
                        .iter()
                        .any(|kept| too_close(kept, point, axis, across, through))
                    {
                        return false;
                    }
                }
            }
        }
        self.buckets.entry(key).or_default().push(Accepted {
            point,
            axis,
            across,
            through,
        });
        true
    }
}

fn too_close(kept: &Accepted, point: [f64; 3], axis: [f64; 3], across: f64, through: f64) -> bool {
    let delta = subtract(point, kept.point);
    if kept.axis == [0.0; 3] {
        // `through` is half the proposal's layer, so this is a quarter layer.
        return dot(delta, delta) < (0.5 * through) * (0.5 * through);
    }
    // Which ray's frame the test is made in does not matter much; the smaller separations of the
    // two points make the test the less restrictive of the two possible ones.
    let along = dot(delta, axis);
    let rest = (dot(delta, delta) - along * along).max(0.0).sqrt();
    let along = along.abs();
    along < through.min(kept.through) && rest < across.min(kept.across)
}

/// The layer stage's state: the ray tree, the separation grid, each surface vertex's normal and
/// mean incident edge length, and how far through the vertices and wall faces it is.
pub(crate) struct LayerRun {
    pub(crate) rays: SurfaceRays,
    pub(crate) separation: Separation,
    pub(crate) normals: Vec<[f64; 3]>,
    pub(crate) edge_mean: Vec<f64>,
    pub(crate) vertices: Vec<u32>,
    pub(crate) cursor: usize,
    /// The surface triangles and the one the wall fill is on, after the vertices are done.
    pub(crate) boundary: Vec<[u32; 3]>,
    pub(crate) fill_cursor: usize,
    pub(crate) magnitude: f64,
}

/// Prepares the layer stage: the ray tree, the separation grid and each surface vertex's normal
/// and mean incident edge length.
pub(crate) fn layer_run(points: &[[f64; 3]], boundary: &[[u32; 3]], fixed: &[bool]) -> LayerRun {
    let mut normals = vec![[0.0; 3]; points.len()];
    let mut edge_total = vec![0.0; points.len()];
    let mut edge_count = vec![0u32; points.len()];
    let (mut total_edge, mut edges) = (0.0_f64, 0u32);
    for face in boundary {
        let p = face.map(|v| points[v as usize]);
        let normal = normalized(cross(subtract(p[1], p[0]), subtract(p[2], p[0])));
        for k in 0..3 {
            let (v, from, to) = (face[k], p[(k + 1) % 3], p[(k + 2) % 3]);
            let (e1, e2) = (subtract(from, p[k]), subtract(to, p[k]));
            let angle = dot(normalized(e1), normalized(e2)).clamp(-1.0, 1.0).acos();
            let n = &mut normals[v as usize];
            n[0] += normal[0] * angle;
            n[1] += normal[1] * angle;
            n[2] += normal[2] * angle;
            let l = length(e1);
            edge_total[v as usize] += l;
            edge_count[v as usize] += 1;
            total_edge += l;
            edges += 1;
        }
    }
    let mean_edge = total_edge / f64::from(edges.max(1));
    let edge_mean: Vec<f64> = edge_total
        .iter()
        .zip(&edge_count)
        .map(|(total, count)| total / f64::from((*count).max(1)))
        .collect();
    let normals: Vec<[f64; 3]> = normals.into_iter().map(normalized).collect();
    let magnitude = points
        .iter()
        .flatten()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    let vertices: Vec<u32> = (0..points.len() as u32)
        .filter(|&v| fixed[v as usize])
        .collect();
    let rays = SurfaceRays::new(points, boundary);
    let mut separation = Separation::new(mean_edge);
    for &v in &vertices {
        separation.add_surface(points[v as usize]);
    }
    LayerRun {
        rays,
        separation,
        normals,
        edge_mean,
        vertices,
        cursor: 0,
        boundary: boundary.to_vec(),
        fill_cursor: 0,
        magnitude,
    }
}
