//! One remesh of a measured part and what it did: whether the surface stayed closed and manifold,
//! how far it moved, whether sharp edges survived, whether it crosses or folds onto itself, and the triangle
//! quality afterwards. The remesh is solidmesh's `remesh_for_analysis`, with the options the
//! benchmark surfaces use (`surfaces::analysis_options`), so the corpus measures the remesh a
//! user's part goes through. Every reading of the result here is the corpus's own, made on the
//! returned coordinates, independent of the check the remesher runs on itself.
//!
//! The target edge length is the one edge the part is meshed at, surface and volume alike: what
//! the part's thinnest wall asks for, raised to what keeps the finished 10-node mesh near
//! [`UNKNOWNS_BUDGET`] by the prediction in `benchmarks/sizing.rs`. When the budget's edge is too
//! coarse for two layers through the 5th-percentile wall, the part is still meshed at it and the
//! trial says so ([`RemeshTrial::wall_ratio`]): a result on that mesh is one a later error
//! estimate must flag. The deviation bound follows the target ([`deviation_bound`]). The remesh is cancelled after [`REMESH_TIME_CAP_SECONDS`],
//! which the record names `remesh:Cancelled`: nothing else cancels it. A refusal is named by its
//! variant ([`refusal_name`]), so a corpus counts refusals by kind.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use solidmesh_mesh::{remesh_for_analysis, AnalysisRemesh, AnalysisRemeshError, CancelToken};

use super::bvh::Bvh;
use super::measure::{quality, topology, InputMetrics, Quality};
use super::self_intersect::{count_folds, count_penetrating_pairs, PairCount};
use super::soup::{dot, length, sub, TriSoup, P3};
use crate::benchmarks::sizing::{edge_for_unknowns, predicted_tets, wall_resolution, PartSize};
use crate::surfaces::analysis_options;

/// The unknowns a part's finished 10-node mesh is sized for: past this the solve's memory (about
/// 0.94 GB of multigrid at `sizing::MULTIGRID_BYTES_PER_UNKNOWN`) and time, not the part, would
/// decide what the corpus can measure. At `sizing::UNKNOWNS_PER_TET` it is about 224,000
/// elements.
pub const UNKNOWNS_BUDGET: f64 = 1_000_000.0;

/// The largest cell the target may be relative to the part, so a part with no readable wall still
/// gets a usable density.
const COARSEST_FRACTION_OF_DIAGONAL: f64 = 1.0 / 40.0;

/// The deviation bound as a share of the target edge length. The benchmark surfaces' bounds,
/// chosen part by part (`surfaces::all_parts`), run from about a 250th to a 20th of the edge
/// length each asks for near its feature; a 40th sits inside that range.
const DEVIATION_PER_TARGET: f64 = 1.0 / 40.0;

/// The deviation bound as a share of the part's 5th-percentile wall. Two walls closer than twice
/// the bound may be pulled together until they cross, so the bound stays well under half the
/// thin walls.
const DEVIATION_PER_WALL: f64 = 0.25;

/// The remesh is cancelled after this long, so one part costs its own row, not the batch's whole
/// timeout. The slowest benchmark surface takes about 22 seconds; this leaves room for real
/// parts several times slower before calling them out.
pub const REMESH_TIME_CAP_SECONDS: f64 = 120.0;

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
    /// The part's wall would ask for more unknowns than [`UNKNOWNS_BUDGET`].
    Budget,
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
    /// The finished 10-node mesh's element count predicted at the target (`benchmarks/sizing.rs`).
    pub predicted_tets: f64,
    /// The target over half the 5th-percentile wall: above 1, two layers do not fit through the
    /// thin walls and they are under-resolved. `None` when no wall could be read.
    pub wall_ratio: Option<f64>,
    /// The `max_deviation` the remesh was given.
    pub deviation_bound: f64,
    pub seconds: f64,
    pub outcome: Result<RemeshedPart, AnalysisRemeshError>,
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
    /// The input creases the remesher reports it kept on a chain of output edges.
    pub sharp_edges_held: usize,
    pub sharp_edges_after: usize,
    /// Of the sampled input sharp edges, the percentage still lying along an output sharp edge.
    pub sharp_survival_percent: f64,
    pub quality: Quality,
    /// Triangle pairs of the output that cross each other, counted by the corpus's own test.
    pub self_intersection: PairCount,
    /// Edge-sharing triangle pairs of the output folded flat onto each other.
    pub folds: u64,
    /// The remeshed surface itself, which the volume stage (`volume_trial.rs`) meshes next.
    pub surface: TriSoup,
}

/// What the size prediction reads from a part's input metrics.
pub fn part_size(metrics: &InputMetrics) -> PartSize {
    PartSize {
        volume: metrics.volume.abs(),
        area: metrics.area,
        wall_p5: metrics.wall.p5,
    }
}

/// Chooses the target edge length for a part, sized for [`UNKNOWNS_BUDGET`].
pub fn choose_target(metrics: &InputMetrics) -> (f64, TargetSource) {
    choose_target_within(metrics, UNKNOWNS_BUDGET)
}

/// Chooses the target edge length for a part whose finished mesh is to stay near `budget`
/// unknowns: the edge its features ask for (half the 5th-percentile wall, no coarser than a 40th
/// of the diagonal), raised to the budget's edge when that is larger.
pub fn choose_target_within(metrics: &InputMetrics, budget: f64) -> (f64, TargetSource) {
    // A part whose volume or area is not a finite number has no budget edge; the NaN target
    // reaches the remesher, which refuses it by name (`remesh:InvalidTargetSize`), so the row
    // records it rather than meshing at a made-up size.
    let budget_edge = edge_for_unknowns(&part_size(metrics), budget).unwrap_or(f64::NAN);
    let coarsest = metrics.diagonal * COARSEST_FRACTION_OF_DIAGONAL;
    let (feature, source) = if !metrics.wall.p5.is_finite() || metrics.wall.p5 <= 0.0 {
        (coarsest, TargetSource::NoWall)
    } else if metrics.wall.p5 * 0.5 > coarsest {
        (coarsest, TargetSource::CoarsestAllowed)
    } else {
        (metrics.wall.p5 * 0.5, TargetSource::ThinnestWall)
    };
    if budget_edge.is_nan() || budget_edge > feature {
        (budget_edge, TargetSource::Budget)
    } else {
        (feature, source)
    }
}

/// The deviation bound for a remesh at `target` of the part `metrics` describes.
pub fn deviation_bound(target: f64, metrics: &InputMetrics) -> f64 {
    let from_target = target * DEVIATION_PER_TARGET;
    if metrics.wall.p5.is_finite() && metrics.wall.p5 > 0.0 {
        from_target.min(metrics.wall.p5 * DEVIATION_PER_WALL)
    } else {
        from_target
    }
}

/// `soup` remeshed at a uniform `target` edge length with the deviation bound that goes with it,
/// cancelled after `cap`.
pub fn remesh_capped(
    soup: &TriSoup,
    metrics: &InputMetrics,
    target: f64,
    cap: Duration,
) -> Result<AnalysisRemesh, AnalysisRemeshError> {
    let mesh = soup.to_mesh();
    let cancel = CancelToken::new();
    let (finished, wait) = mpsc::channel::<()>();
    let watchdog = {
        let cancel = cancel.clone();
        // Woken early when `finished` is dropped; only a full wait cancels.
        thread::spawn(move || {
            if let Err(mpsc::RecvTimeoutError::Timeout) = wait.recv_timeout(cap) {
                cancel.cancel();
            }
        })
    };
    let size_at = move |_: [f64; 3]| target;
    let options = analysis_options(&size_at, deviation_bound(target, metrics), &cancel);
    let result = remesh_for_analysis(&mesh, &options);
    drop(finished);
    if let Err(panic) = watchdog.join() {
        std::panic::resume_unwind(panic);
    }
    result
}

/// The stable name of a refusal: its variant, which the record writes as `remesh:<name>`.
pub fn refusal_name(error: &AnalysisRemeshError) -> &'static str {
    match error {
        AnalysisRemeshError::EmptyInput => "EmptyInput",
        AnalysisRemeshError::OpenInput { .. } => "OpenInput",
        AnalysisRemeshError::NonManifoldInput { .. } => "NonManifoldInput",
        AnalysisRemeshError::InconsistentWinding { .. } => "InconsistentWinding",
        AnalysisRemeshError::InvalidOptions(_) => "InvalidOptions",
        AnalysisRemeshError::InvalidTargetSize { .. } => "InvalidTargetSize",
        AnalysisRemeshError::TriangleBudgetExceeded { .. } => "TriangleBudgetExceeded",
        AnalysisRemeshError::Cancelled => "Cancelled",
        AnalysisRemeshError::BrokenTopology(_) => "BrokenTopology",
        AnalysisRemeshError::NonFiniteVertex { .. } => "NonFiniteVertex",
        AnalysisRemeshError::SelfIntersecting { .. } => "SelfIntersecting",
        AnalysisRemeshError::DeviationNotMet { .. } => "DeviationNotMet",
        AnalysisRemeshError::AngleFloorNotMet { .. } => "AngleFloorNotMet",
        AnalysisRemeshError::FoldedSurface { .. } => "FoldedSurface",
        AnalysisRemeshError::CreaseLost { .. } => "CreaseLost",
    }
}

/// Runs the remesher on `soup` and measures the result against it.
pub fn remesh_and_measure(soup: &TriSoup, metrics: &InputMetrics) -> RemeshTrial {
    let (target, source) = choose_target(metrics);
    let size = part_size(metrics);
    let start = Instant::now();
    let result = remesh_capped(
        soup,
        metrics,
        target,
        Duration::from_secs_f64(REMESH_TIME_CAP_SECONDS),
    );
    let seconds = start.elapsed().as_secs_f64();
    let outcome = result.map(|remeshed| {
        let output = TriSoup::from_analysis_surface(&remeshed.surface);
        let after = topology(&output);
        let input_bvh = Bvh::new(soup);
        let output_bvh = Bvh::new(&output);
        let volume_after = output.signed_volume();
        RemeshedPart {
            triangles: output.triangles.len(),
            closed: after.is_closed(),
            manifold_and_oriented: after.is_manifold_and_oriented(),
            volume_change_percent: (volume_after - metrics.volume) / metrics.volume * 100.0,
            deviation_output_to_input: farthest(&surface_points(&output), &input_bvh, usize::MAX)
                / metrics.diagonal,
            deviation_input_to_output: farthest(
                &surface_points(soup),
                &output_bvh,
                DEVIATION_SAMPLE_LIMIT,
            ) / metrics.diagonal,
            sharp_edges_in: metrics.topology.sharp_edges.len(),
            sharp_edges_held: remeshed.report.sharp_edges_kept,
            sharp_edges_after: after.sharp_edges.len(),
            sharp_survival_percent: sharp_survival(soup, metrics, &output, &after, target),
            quality: quality(&output),
            self_intersection: count_penetrating_pairs(&output, &output_bvh),
            folds: count_folds(&output),
            surface: output,
        }
    });
    RemeshTrial {
        target_edge_length: target,
        target_source: source,
        predicted_tets: predicted_tets(&size, target),
        wall_ratio: wall_resolution(&size, target),
        deviation_bound: deviation_bound(target, metrics),
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
