//! One remesh of a measured part and what it did: whether the surface stayed closed and manifold,
//! how far it moved, whether sharp edges survived, and the triangle quality afterwards. The
//! target edge length is tied to the part's thinnest wall, clamped so a tiny wall on a big part
//! cannot ask for more triangles than the corpus can afford.

use std::time::Instant;

use super::bvh::Bvh;
use super::g3_mount::remesh::{remesh, RemeshRefusal};
use super::measure::{quality, topology, InputMetrics, Quality};
use super::soup::{dot, length, sub, TriSoup, P3};

/// The most triangles a target edge length is allowed to ask for: past this the remesher's run
/// time, not the part, decides what the corpus can measure.
const TRIANGLE_BUDGET: f64 = 300_000.0;

/// The largest cell the target may be relative to the part, so a part with no readable wall still
/// gets a usable density.
const COARSEST_FRACTION_OF_DIAGONAL: f64 = 1.0 / 40.0;

/// Output-sharp-edge segments compared against each input sharp edge are capped here; past that
/// the search would dominate the run and the cap is reported in the record.
const SHARP_MATCH_CAP: usize = 30_000;

/// How many input sharp edges are checked for survival.
const SHARP_SAMPLE_LIMIT: usize = 2_000;

/// Input vertices sampled for the input-to-output deviation.
const DEVIATION_SAMPLE_LIMIT: usize = 50_000;

/// Which limit chose the target edge length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetSource {
    /// Half the part's 5th-percentile wall.
    ThinnestWall,
    /// The part's wall would ask for more triangles than the budget.
    TriangleBudget,
    /// The wall was large; the target is the coarsest allowed.
    CoarsestAllowed,
    /// No wall could be read.
    NoWall,
}

/// What a remesh did to one part.
#[derive(Clone, Debug)]
pub struct RemeshTrial {
    pub target_edge_length: f64,
    pub target_source: TargetSource,
    pub seconds: f64,
    pub outcome: Result<RemeshedPart, RemeshRefusal>,
}

/// A remesh that produced a surface, measured against its input.
#[derive(Clone, Debug)]
pub struct RemeshedPart {
    pub triangles: usize,
    pub closed: bool,
    pub manifold_and_oriented: bool,
    pub volume_change_percent: f64,
    /// Largest distance from an output vertex to the input surface, over the part's diagonal.
    pub deviation_output_to_input: f64,
    /// Largest distance from a sampled input vertex to the output surface, over the diagonal.
    pub deviation_input_to_output: f64,
    pub sharp_edges_in: usize,
    pub sharp_edges_held: usize,
    pub sharp_edges_after: usize,
    /// Of the sampled input sharp edges, the percentage still lying along an output sharp edge.
    pub sharp_survival_percent: f64,
    pub quality: Quality,
    /// The remeshed surface itself, which the volume stage (`volume_trial.rs`) meshes next.
    pub surface: TriSoup,
}

/// Chooses the target edge length for a part.
pub fn choose_target(metrics: &InputMetrics) -> (f64, TargetSource) {
    let area_limit = (metrics.area / (0.43 * TRIANGLE_BUDGET)).sqrt();
    let coarsest = metrics.diagonal * COARSEST_FRACTION_OF_DIAGONAL;
    if !metrics.wall.p5.is_finite() || metrics.wall.p5 <= 0.0 {
        return (coarsest.max(area_limit), TargetSource::NoWall);
    }
    let from_wall = metrics.wall.p5 * 0.5;
    if from_wall < area_limit {
        (area_limit, TargetSource::TriangleBudget)
    } else if from_wall > coarsest {
        (coarsest, TargetSource::CoarsestAllowed)
    } else {
        (from_wall, TargetSource::ThinnestWall)
    }
}

/// Runs the remesher on `soup` and measures the result against it.
pub fn remesh_and_measure(soup: &TriSoup, metrics: &InputMetrics) -> RemeshTrial {
    let (target, source) = choose_target(metrics);
    let start = Instant::now();
    let result = remesh(soup, &metrics.topology.sharp_edges, target);
    let seconds = start.elapsed().as_secs_f64();
    let outcome = result.map(|output| {
        let after = topology(&output.soup);
        let input_bvh = Bvh::new(soup);
        let output_bvh = Bvh::new(&output.soup);
        let volume_after = output.soup.signed_volume();
        RemeshedPart {
            triangles: output.soup.triangles.len(),
            closed: after.is_closed(),
            manifold_and_oriented: after.is_manifold_and_oriented(),
            volume_change_percent: (volume_after - metrics.volume) / metrics.volume * 100.0,
            deviation_output_to_input: farthest(
                &surface_points(&output.soup),
                &input_bvh,
                usize::MAX,
            ) / metrics.diagonal,
            deviation_input_to_output: farthest(
                &surface_points(soup),
                &output_bvh,
                DEVIATION_SAMPLE_LIMIT,
            ) / metrics.diagonal,
            sharp_edges_in: metrics.topology.sharp_edges.len(),
            sharp_edges_held: output.constrained_sharp_edges,
            sharp_edges_after: after.sharp_edges.len(),
            sharp_survival_percent: sharp_survival(soup, metrics, &output.soup, &after, target),
            quality: quality(&output.soup),
            surface: output.soup,
        }
    });
    RemeshTrial {
        target_edge_length: target,
        target_source: source,
        seconds,
        outcome,
    }
}

/// The largest distance from `points` (at most `limit`, at a fixed stride) to the surface.
fn farthest(points: &[P3], surface: &Bvh<'_>, limit: usize) -> f64 {
    let stride = (points.len() / limit.max(1)).max(1);
    points
        .iter()
        .step_by(stride)
        .map(|&p| surface.distance_squared(p))
        .fold(0.0_f64, f64::max)
        .sqrt()
}

/// Of the sampled input sharp edges, the percentage whose midpoint lies within a tenth of the
/// target of an output sharp edge running the same way.
fn sharp_survival(
    input: &TriSoup,
    metrics: &InputMetrics,
    output: &TriSoup,
    after: &super::measure::Topology,
    target: f64,
) -> f64 {
    let sharp_in = &metrics.topology.sharp_edges;
    if sharp_in.is_empty() {
        return 100.0;
    }
    let out_segments: Vec<(P3, P3)> = after
        .sharp_edges
        .iter()
        .step_by((after.sharp_edges.len() / SHARP_MATCH_CAP).max(1))
        .map(|&(a, b)| (output.vertices[a as usize], output.vertices[b as usize]))
        .collect();
    let stride = (sharp_in.len() / SHARP_SAMPLE_LIMIT).max(1);
    let reach = target * 0.1;
    let reach_squared = reach * reach;
    let sampled: Vec<&(u32, u32)> = sharp_in.iter().step_by(stride).collect();
    let survivors = sampled
        .iter()
        .filter(|&&&(a, b)| {
            let (p, q) = (input.vertices[a as usize], input.vertices[b as usize]);
            segment_lies_on_any(p, q, &out_segments, reach_squared)
        })
        .count();
    survivors as f64 * 100.0 / sampled.len() as f64
}

/// Whether the midpoint of `p..q` is within `reach_squared` of one of `segments` running the same
/// way (within about 37 degrees).
fn segment_lies_on_any(p: P3, q: P3, segments: &[(P3, P3)], reach_squared: f64) -> bool {
    let midpoint = [
        (p[0] + q[0]) * 0.5,
        (p[1] + q[1]) * 0.5,
        (p[2] + q[2]) * 0.5,
    ];
    let along = sub(q, p);
    segments.iter().any(|&(a, b)| {
        let s = sub(b, a);
        let aligned = dot(along, s).abs() >= 0.8 * length(along) * length(s);
        aligned && point_segment_distance_squared(midpoint, a, b) <= reach_squared
    })
}

fn point_segment_distance_squared(p: P3, a: P3, b: P3) -> f64 {
    let ab = sub(b, a);
    let ap = sub(p, a);
    let span = dot(ab, ab);
    if span <= 0.0 {
        return dot(ap, ap);
    }
    let t = (dot(ap, ab) / span).clamp(0.0, 1.0);
    let off = sub(ap, [ab[0] * t, ab[1] * t, ab[2] * t]);
    dot(off, off)
}

/// Every vertex and every face centroid: a deviation read at vertices alone is blind to faces
/// that chord across a curve between vertices the projection put exactly on the surface.
fn surface_points(soup: &TriSoup) -> Vec<P3> {
    let centroids = (0..soup.triangles.len()).map(|t| {
        let [a, b, c] = soup.corners(t);
        [0, 1, 2].map(|k| (a[k] + b[k] + c[k]) / 3.0)
    });
    soup.vertices.iter().copied().chain(centroids).collect()
}
