//! What a tetrahedral mesher needs to know about a closed triangle mesh: its topology (closed,
//! manifold, consistently wound, how many components), triangle quality, edge lengths, how many
//! edges are sharp, and how thin its walls are. [`quality`] and [`topology`] read the remesher's
//! output too. A wall is measured by casting a ray inward from each face centroid to the first
//! surface that faces back along the ray; a reading at a concave corner or along a slope is
//! filtered out by that facing test, so what is left is a material thickness.
//! [`sample_exclusion`] decides from these readings alone whether a part is a solid the remesher
//! should be asked to prepare at all.

use super::bvh::Bvh;
use super::self_intersect::{count_folds, count_penetrating_pairs, PairCount};
use super::soup::{cross, dot, face_normal, length, scale, sub, TriSoup, P3};

/// The bend between face normals beyond which an edge is a sharp feature.
pub const SHARP_DEGREES: f64 = 30.0;

/// Above this many faces, walls are sampled at a fixed stride: enough to read the 5th percentile
/// of a part, and it keeps the whole corpus inside its time budget.
const WALL_SAMPLE_LIMIT: usize = 50_000;

/// A ray only counts when the wall it reaches faces it within 60 degrees: glancing hits at
/// concave corners or along shallow slopes are not a material thickness.
const WALL_FACING_COSINE: f64 = 0.5;

/// A sample's median, 95th, 99th percentile and maximum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Distribution {
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
    pub worst: f64,
}

/// `sorted`'s value at quantile `q` (nearest rank).
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    match sorted.len() {
        0 => f64::NAN,
        n => sorted[(((n - 1) as f64) * q).round_ties_even() as usize],
    }
}

fn distribution(mut values: Vec<f64>) -> Distribution {
    values.sort_unstable_by(f64::total_cmp);
    Distribution {
        median: percentile(&values, 0.5),
        p95: percentile(&values, 0.95),
        p99: percentile(&values, 0.99),
        worst: values.last().copied().unwrap_or(f64::NAN),
    }
}

/// Triangle quality: aspect ratio (equilateral is 1) and the smallest interior angle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quality {
    pub aspect: Distribution,
    pub min_angle_degrees: f64,
    /// Triangles with no measurable area: their aspect ratio is infinite, so they are counted
    /// here rather than poisoning the distribution's upper tail.
    pub degenerate: usize,
}

/// The quality of every triangle of `soup`.
pub fn quality(soup: &TriSoup) -> Quality {
    let mut aspects = Vec::with_capacity(soup.triangles.len());
    let mut min_angle: f64 = 180.0;
    let mut degenerate = 0;
    for t in 0..soup.triangles.len() {
        let corners = soup.corners(t);
        let area = length(cross(
            sub(corners[1], corners[0]),
            sub(corners[2], corners[0]),
        )) * 0.5;
        let edges = [0, 1, 2].map(|i| length(sub(corners[(i + 1) % 3], corners[i])));
        if area <= f64::EPSILON * edges.iter().fold(0.0_f64, |m, &e| m.max(e * e)) {
            degenerate += 1;
            continue;
        }
        aspects.push(aspect_ratio(edges, area));
        min_angle = min_angle.min(smallest_angle_degrees(&corners));
    }
    Quality {
        aspect: distribution(aspects),
        min_angle_degrees: min_angle,
        degenerate,
    }
}

/// The longest edge over twice the inradius, scaled so an equilateral triangle is exactly 1: the
/// conventional finite-element aspect ratio, unbounded as a triangle degenerates.
pub fn aspect_ratio(edges: [f64; 3], area: f64) -> f64 {
    let longest = edges[0].max(edges[1]).max(edges[2]);
    let semi_perimeter = (edges[0] + edges[1] + edges[2]) * 0.5;
    let inradius = area / semi_perimeter;
    longest / (2.0 * 3.0_f64.sqrt() * inradius)
}

fn smallest_angle_degrees(corners: &[P3; 3]) -> f64 {
    (0..3)
        .map(|i| {
            let a = sub(corners[(i + 1) % 3], corners[i]);
            let b = sub(corners[(i + 2) % 3], corners[i]);
            libm::atan2(length(cross(a, b)), dot(a, b)).to_degrees()
        })
        .fold(180.0, f64::min)
}

/// Edge counts by how many faces meet there and how they wind.
#[derive(Clone, Debug, PartialEq)]
pub struct Topology {
    /// Edges with a single face: holes in the surface.
    pub boundary_edges: usize,
    /// Edges with more than two faces.
    pub nonmanifold_edges: usize,
    /// Two-face edges whose faces walk the edge the same way (a flipped face).
    pub misoriented_edges: usize,
    pub components: usize,
    /// Two-face edges bending by more than [`SHARP_DEGREES`], as sorted vertex pairs.
    pub sharp_edges: Vec<(u32, u32)>,
    pub edge_lengths: Vec<f64>,
}

impl Topology {
    /// Every edge is shared by exactly two faces.
    pub fn is_closed(&self) -> bool {
        self.boundary_edges == 0 && self.nonmanifold_edges == 0
    }

    /// [`Self::is_closed`] and consistently wound: the surface a tetrahedral mesher can trust.
    pub fn is_manifold_and_oriented(&self) -> bool {
        self.is_closed() && self.misoriented_edges == 0
    }
}

/// Reads the edges of every triangle, sorted so nothing depends on hash order.
pub fn topology(soup: &TriSoup) -> Topology {
    let mut half_edges: Vec<(u32, u32, u32, bool)> = Vec::with_capacity(soup.triangles.len() * 3);
    for (face, tri) in soup.triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            half_edges.push((a.min(b), a.max(b), face as u32, a < b));
        }
    }
    half_edges.sort_unstable();
    let mut topology = Topology {
        boundary_edges: 0,
        nonmanifold_edges: 0,
        misoriented_edges: 0,
        components: 0,
        sharp_edges: Vec::new(),
        edge_lengths: Vec::with_capacity(soup.triangles.len() * 3 / 2),
    };
    let mut parent: Vec<u32> = (0..soup.triangles.len() as u32).collect();
    let normals: Vec<Option<P3>> = (0..soup.triangles.len())
        .map(|t| face_normal(&soup.corners(t)))
        .collect();
    let mut i = 0;
    while i < half_edges.len() {
        let (a, b) = (half_edges[i].0, half_edges[i].1);
        let mut j = i + 1;
        while j < half_edges.len() && half_edges[j].0 == a && half_edges[j].1 == b {
            j += 1;
        }
        let edge = length(sub(soup.vertices[a as usize], soup.vertices[b as usize]));
        topology.edge_lengths.push(edge);
        let run = &half_edges[i..j];
        match run.len() {
            1 => topology.boundary_edges += 1,
            2 => classify_shared_edge(run, &normals, &mut parent, &mut topology, (a, b)),
            _ => {
                topology.nonmanifold_edges += 1;
                for pair in run.windows(2) {
                    join(&mut parent, pair[0].2, pair[1].2);
                }
            }
        }
        i = j;
    }
    topology.components = (0..parent.len() as u32)
        .filter(|&i| parent[i as usize] == i)
        .count();
    topology
}

fn classify_shared_edge(
    run: &[(u32, u32, u32, bool)],
    normals: &[Option<P3>],
    parent: &mut [u32],
    topology: &mut Topology,
    edge: (u32, u32),
) {
    let (fa, fb) = (run[0].2, run[1].2);
    join(parent, fa, fb);
    if run[0].3 == run[1].3 {
        topology.misoriented_edges += 1;
    }
    if let (Some(na), Some(nb)) = (normals[fa as usize], normals[fb as usize]) {
        let angle = libm::atan2(length(cross(na, nb)), dot(na, nb)).to_degrees();
        if angle > SHARP_DEGREES {
            topology.sharp_edges.push(edge);
        }
    }
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

fn join(parent: &mut [u32], a: u32, b: u32) {
    let (a, b) = (find(parent, a), find(parent, b));
    if a != b {
        parent[a.max(b) as usize] = a.min(b);
    }
}

/// Wall thickness read from inward rays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WallReading {
    /// Rays that found a facing wall.
    pub samples: usize,
    pub p5: f64,
    pub minimum: f64,
}

/// Reads the part's wall thickness by casting from each face centroid along its inward normal.
pub fn wall_reading(soup: &TriSoup, bvh: &Bvh<'_>) -> WallReading {
    let stride = (soup.triangles.len() / WALL_SAMPLE_LIMIT).max(1);
    let nudge = soup.bbox_diagonal() * 1e-9;
    let normals: Vec<Option<P3>> = (0..soup.triangles.len())
        .map(|t| face_normal(&soup.corners(t)))
        .collect();
    let mut walls = Vec::new();
    for face in (0..soup.triangles.len()).step_by(stride) {
        let Some(normal) = normals[face] else {
            continue;
        };
        let direction = scale(normal, -1.0);
        let corners = soup.corners(face);
        let centroid = [0, 1, 2].map(|k| (corners[0][k] + corners[1][k] + corners[2][k]) / 3.0);
        let origin = [
            centroid[0] + direction[0] * nudge,
            centroid[1] + direction[1] * nudge,
            centroid[2] + direction[2] * nudge,
        ];
        let accept = |tri: u32| -> bool {
            tri as usize != face
                && normals[tri as usize].is_some_and(|n| dot(n, direction) > WALL_FACING_COSINE)
        };
        if let Some((t, _)) = bvh.nearest_hit(origin, direction, accept) {
            walls.push(t + nudge);
        }
    }
    walls.sort_unstable_by(f64::total_cmp);
    WallReading {
        samples: walls.len(),
        p5: percentile(&walls, 0.05),
        minimum: walls.first().copied().unwrap_or(f64::NAN),
    }
}

/// Everything read from one input surface.
#[derive(Clone, Debug)]
pub struct InputMetrics {
    pub triangles: usize,
    pub topology: Topology,
    pub self_intersection: PairCount,
    /// Edge-sharing triangle pairs folded flat onto each other (`self_intersect::count_folds`).
    pub folds: u64,
    pub quality: Quality,
    pub edge_min: f64,
    pub edge_median: f64,
    pub edge_max: f64,
    pub wall: WallReading,
    pub diagonal: f64,
    pub area: f64,
    pub volume: f64,
}

/// Why a part is left out of the remesh sample, or `None` when it is in: only a closed,
/// manifold, consistently wound surface enclosing positive volume is a solid to mesh. The reasons
/// are checked in this order and the first that applies is named (`open`, `non-manifold`,
/// `misoriented`, `non-positive-volume`), so each part has one.
pub fn sample_exclusion(metrics: &InputMetrics) -> Option<&'static str> {
    let t = &metrics.topology;
    if t.boundary_edges > 0 {
        Some("open")
    } else if t.nonmanifold_edges > 0 {
        Some("non-manifold")
    } else if t.misoriented_edges > 0 {
        Some("misoriented")
    } else if metrics.volume > 0.0 {
        None
    } else {
        // Zero, negative, or not a number at all.
        Some("non-positive-volume")
    }
}

/// Reads every metric of one input surface.
pub fn measure_input(soup: &TriSoup) -> InputMetrics {
    let bvh = Bvh::new(soup);
    let topology = topology(soup);
    let mut lengths = topology.edge_lengths.clone();
    lengths.sort_unstable_by(f64::total_cmp);
    InputMetrics {
        triangles: soup.triangles.len(),
        self_intersection: count_penetrating_pairs(soup, &bvh),
        folds: count_folds(soup),
        quality: quality(soup),
        edge_min: lengths.first().copied().unwrap_or(f64::NAN),
        edge_median: percentile(&lengths, 0.5),
        edge_max: lengths.last().copied().unwrap_or(f64::NAN),
        wall: wall_reading(soup, &bvh),
        diagonal: soup.bbox_diagonal(),
        area: soup.surface_area(),
        volume: soup.signed_volume(),
        topology,
    }
}
