//! A bounding-volume tree over a whole triangle soup, for inward rays, nearest-surface distances
//! and the self-intersection broad phase. solidmesh's `DMeshAABBTree3` is built over its `DMesh3`,
//! which refuses non-manifold triangles, so a tree over that would quietly lose the very faces
//! being measured; this one keeps every triangle.

use super::soup::{cross, dot, point_triangle_distance_squared, sub, TriSoup, P3};

/// The triangles a leaf holds: small enough that leaf tests stay cheap.
const LEAF_SIZE: usize = 4;

/// One tree node over the sorted triangle order.
#[derive(Clone, Debug)]
struct Node {
    lo: P3,
    hi: P3,
    /// Child node indices when `count` is 0, else `first..first + count` of `order`.
    left: u32,
    right: u32,
    first: u32,
    count: u32,
}

/// A bounding-volume tree over one soup.
pub struct Bvh<'a> {
    soup: &'a TriSoup,
    nodes: Vec<Node>,
    order: Vec<u32>,
}

impl<'a> Bvh<'a> {
    /// Builds the tree over every triangle of `soup`.
    pub fn new(soup: &'a TriSoup) -> Self {
        let mut order: Vec<u32> = (0..soup.triangles.len() as u32).collect();
        let mut nodes = Vec::new();
        if !order.is_empty() {
            build(soup, &mut order, 0, &mut nodes);
        }
        Self { soup, nodes, order }
    }

    /// The nearest triangle along a unit-direction ray that `accept` allows: `(t, triangle)`.
    pub fn nearest_hit(
        &self,
        origin: P3,
        direction: P3,
        accept: impl Fn(u32) -> bool,
    ) -> Option<(f64, u32)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<(f64, u32)> = None;
        let mut stack = vec![0_u32];
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id as usize];
            let limit = best.map_or(f64::INFINITY, |(t, _)| t);
            if !ray_hits_box(origin, direction, node, limit) {
                continue;
            }
            if node.count > 0 {
                for i in node.first..node.first + node.count {
                    let tri = self.order[i as usize];
                    if !accept(tri) {
                        continue;
                    }
                    let corners = self.soup.corners(tri as usize);
                    if let Some(t) = ray_triangle(origin, direction, &corners) {
                        if best.is_none_or(|(b, _)| t < b) {
                            best = Some((t, tri));
                        }
                    }
                }
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        best
    }

    /// The squared distance from `p` to the nearest point on any triangle.
    pub fn distance_squared(&self, p: P3) -> f64 {
        if self.nodes.is_empty() {
            return f64::INFINITY;
        }
        let mut best = f64::INFINITY;
        let mut stack = vec![0_u32];
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id as usize];
            if box_distance_squared(p, node) >= best {
                continue;
            }
            if node.count > 0 {
                for i in node.first..node.first + node.count {
                    let tri = self.order[i as usize] as usize;
                    best = best.min(point_triangle_distance_squared(p, &self.soup.corners(tri)));
                }
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        best
    }

    /// Walks every pair of leaves whose boxes overlap, handing each triangle pair to `visit`.
    /// `visit` returns false to stop the walk early.
    pub fn visit_overlapping_pairs(&self, visit: &mut impl FnMut(u32, u32) -> bool) {
        if !self.nodes.is_empty() {
            self.descend(0, 0, visit);
        }
    }

    fn descend(&self, a: u32, b: u32, visit: &mut impl FnMut(u32, u32) -> bool) -> bool {
        let (na, nb) = (&self.nodes[a as usize], &self.nodes[b as usize]);
        if !boxes_overlap(na, nb) {
            return true;
        }
        match (na.count > 0, nb.count > 0) {
            (true, true) => self.visit_leaf_pair(na, nb, a == b, visit),
            (false, _) if a == b => {
                self.descend(na.left, na.left, visit)
                    && self.descend(na.left, na.right, visit)
                    && self.descend(na.right, na.right, visit)
            }
            (true, false) => self.descend(a, nb.left, visit) && self.descend(a, nb.right, visit),
            (false, _) => self.descend(na.left, b, visit) && self.descend(na.right, b, visit),
        }
    }

    fn visit_leaf_pair(
        &self,
        na: &Node,
        nb: &Node,
        same: bool,
        visit: &mut impl FnMut(u32, u32) -> bool,
    ) -> bool {
        for i in na.first..na.first + na.count {
            let from = if same { i + 1 } else { nb.first };
            for j in from..nb.first + nb.count {
                if !visit(self.order[i as usize], self.order[j as usize]) {
                    return false;
                }
            }
        }
        true
    }
}

/// Builds the subtree over `order`, whose first entry sits at `start` of the tree's order, and
/// returns its node id. Splits at the median on the longest centroid axis; ties break by triangle
/// index so two runs give the same tree.
fn build(soup: &TriSoup, order: &mut [u32], start: u32, nodes: &mut Vec<Node>) -> u32 {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    let (mut c_lo, mut c_hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for &t in order.iter() {
        let corners = soup.corners(t as usize);
        let mut centroid = [0.0; 3];
        for corner in corners {
            for k in 0..3 {
                lo[k] = lo[k].min(corner[k]);
                hi[k] = hi[k].max(corner[k]);
                centroid[k] += corner[k] / 3.0;
            }
        }
        for k in 0..3 {
            c_lo[k] = c_lo[k].min(centroid[k]);
            c_hi[k] = c_hi[k].max(centroid[k]);
        }
    }
    let id = nodes.len() as u32;
    nodes.push(Node {
        lo,
        hi,
        left: 0,
        right: 0,
        first: start,
        count: order.len() as u32,
    });
    if order.len() <= LEAF_SIZE {
        return id;
    }
    let axis = (0..3)
        .max_by(|&a, &b| {
            (c_hi[a] - c_lo[a])
                .total_cmp(&(c_hi[b] - c_lo[b]))
                .then(a.cmp(&b))
        })
        .unwrap_or(0);
    let middle = order.len() / 2;
    let key = |t: u32| -> f64 {
        soup.corners(t as usize)
            .iter()
            .map(|c| c[axis])
            .sum::<f64>()
    };
    order.select_nth_unstable_by(middle, |&a, &b| key(a).total_cmp(&key(b)).then(a.cmp(&b)));
    let (head, tail) = order.split_at_mut(middle);
    let left = build(soup, head, start, nodes);
    let right = build(soup, tail, start + middle as u32, nodes);
    let node = &mut nodes[id as usize];
    node.left = left;
    node.right = right;
    node.count = 0;
    id
}

fn ray_hits_box(origin: P3, direction: P3, node: &Node, limit: f64) -> bool {
    let (mut t0, mut t1) = (0.0_f64, limit);
    for k in 0..3 {
        // A zero direction component gives infinities here, which compare correctly; an origin
        // exactly on a slab face gives NaN, and that axis then cannot rule the box out.
        let inv = 1.0 / direction[k];
        let (mut near, mut far) = (
            (node.lo[k] - origin[k]) * inv,
            (node.hi[k] - origin[k]) * inv,
        );
        if near > far {
            std::mem::swap(&mut near, &mut far);
        }
        if near.is_nan() || far.is_nan() {
            continue;
        }
        t0 = t0.max(near);
        t1 = t1.min(far);
        if t0 > t1 {
            return false;
        }
    }
    true
}

/// The ray parameter where the ray crosses the triangle, either winding, if it does.
fn ray_triangle(origin: P3, direction: P3, tri: &[P3; 3]) -> Option<f64> {
    let e1 = sub(tri[1], tri[0]);
    let e2 = sub(tri[2], tri[0]);
    let pv = cross(direction, e2);
    let det = dot(e1, pv);
    if det.abs() < 1e-18 {
        return None;
    }
    let inv = 1.0 / det;
    let tv = sub(origin, tri[0]);
    let u = dot(tv, pv) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qv = cross(tv, e1);
    let v = dot(direction, qv) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = dot(e2, qv) * inv;
    (t > 0.0).then_some(t)
}

fn box_distance_squared(p: P3, node: &Node) -> f64 {
    (0..3)
        .map(|k| {
            let d = (node.lo[k] - p[k]).max(p[k] - node.hi[k]).max(0.0);
            d * d
        })
        .sum()
}

fn boxes_overlap(a: &Node, b: &Node) -> bool {
    (0..3).all(|k| a.lo[k] <= b.hi[k] && b.lo[k] <= a.hi[k])
}
