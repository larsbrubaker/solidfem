//! Why surface recovery fails on a part, measured: the part remeshed at a multiple of the corpus's
//! own target edge length (`remesh_trial.rs` chooses it and runs the same `remesh_for_analysis`
//! the corpus does, its deviation bound scaled with the length), then only the recovery stage of
//! the tetrahedral mesher run on it under a time cap, and the outcome read in detail. For a refusal
//! that leaves constraints missing it reads the mesher's list of what is missing and why each was
//! given up, and measures each leftover where it sits on the surface: the bend across a missing
//! edge, the smallest angle of the triangles beside it, and the gap to the nearest surface
//! triangle not touching it, relative to its own length. The same readings over a fixed stride of
//! all surface edges give the baseline a leftover is compared with. For a part the mesher refuses
//! as wound wrong or as not separating, the signed volumes, the shells and the crossing and folded
//! triangle pairs say what the surface is.
//! `src/bin/corpus.rs diagnose` prints one key=value line per multiple.

use std::time::{Duration, Instant};

use solidfem::tetra::{StepStatus, SurfaceMesher, TetraError};

use super::bvh::Bvh;
use super::measure::{measure_input, percentile, topology};
use super::remesh_trial::{
    choose_target, deviation_bound, refusal_name, remesh_capped, REMESH_TIME_CAP_SECONDS,
};
use super::self_intersect::{count_folds, count_penetrating_pairs};
use super::soup::{
    cross, dot, face_normal, length, point_triangle_distance_squared, sub, TriSoup, P3,
};

/// Leftovers measured one by one at most; a part missing more is sampled at a fixed stride.
const LEFTOVER_SAMPLE: usize = 200;

/// Surface edges measured for the baseline.
const BASELINE_SAMPLE: usize = 400;

/// The diagnosis of `soup` at each of `scales` times the corpus target edge length, recovery
/// cancelled after `cap_seconds`, as key=value fields per scale.
pub fn diagnose(
    name: &str,
    soup: &TriSoup,
    scales: &[f64],
    cap_seconds: f64,
) -> Vec<Vec<(String, String)>> {
    let metrics = measure_input(soup);
    let (target, _) = choose_target(&metrics);
    scales
        .iter()
        .map(|&scale| {
            let mut fields = vec![
                ("name".to_string(), name.to_string()),
                ("scale".to_string(), scale.to_string()),
                ("vol_in".to_string(), soup.signed_volume().to_string()),
                (
                    "in_misoriented".to_string(),
                    metrics.topology.misoriented_edges.to_string(),
                ),
                (
                    "in_open".to_string(),
                    metrics.topology.boundary_edges.to_string(),
                ),
                (
                    "in_nonmanifold".to_string(),
                    metrics.topology.nonmanifold_edges.to_string(),
                ),
            ];
            let h = target * scale;
            fields.push(("h".to_string(), h.to_string()));
            fields.push(("dev".to_string(), deviation_bound(h, &metrics).to_string()));
            let cap = Duration::from_secs_f64(REMESH_TIME_CAP_SECONDS);
            match remesh_capped(soup, &metrics, h, cap) {
                Ok(output) => {
                    let surface = TriSoup::from_analysis_surface(&output.surface);
                    diagnose_surface(&surface, h, cap_seconds, &mut fields);
                }
                Err(refusal) => {
                    push(
                        &mut fields,
                        "status",
                        format!("remesh:{}", refusal_name(&refusal)),
                    );
                    push(&mut fields, "detail", format!("{refusal:?}"));
                }
            }
            fields
        })
        .map(|fields| {
            fields
                .into_iter()
                .map(|(k, v)| (k, v.replace(char::is_whitespace, "_")))
                .collect()
        })
        .collect()
}

fn push(fields: &mut Vec<(String, String)>, key: &str, value: impl ToString) {
    fields.push((key.to_string(), value.to_string()));
}

/// Recovery on one remeshed surface, and the readings its outcome calls for.
fn diagnose_surface(
    surface: &TriSoup,
    h: f64,
    cap_seconds: f64,
    fields: &mut Vec<(String, String)>,
) {
    let after = topology(surface);
    push(fields, "tris", surface.triangles.len());
    push(fields, "vol_out", surface.signed_volume());
    push(fields, "out_open", after.boundary_edges);
    push(fields, "out_nonmanifold", after.nonmanifold_edges);
    push(fields, "out_misoriented", after.misoriented_edges);
    let bvh = Bvh::new(surface);
    let crossings = count_penetrating_pairs(surface, &bvh);
    push(fields, "out_xpairs", crossings.pairs);
    push(fields, "out_folds", count_folds(surface));
    shells(surface, fields);
    let start = Instant::now();
    let mut mesher = match SurfaceMesher::new(surface.vertices.clone(), surface.triangles.clone()) {
        Ok(mesher) => mesher,
        Err(error) => {
            push(fields, "status", format!("surface:{}", variant(&error)));
            push(fields, "detail", format!("{error:?}"));
            return;
        }
    };
    let mut cancel = || start.elapsed().as_secs_f64() >= cap_seconds;
    let outcome = mesher.step(usize::MAX, &mut cancel);
    push(fields, "sec", start.elapsed().as_secs_f64());
    let report = mesher.report();
    push(fields, "edges_total", report.edges_total);
    push(fields, "faces_total", report.faces_total);
    push(
        fields,
        "edges_after_delaunay",
        report.edges_present_after_delaunay,
    );
    push(
        fields,
        "faces_after_delaunay",
        report.faces_present_after_delaunay,
    );
    push(fields, "flat_patches", report.flat_patches);
    match outcome {
        Ok(StepStatus::Done) => push(fields, "status", "recovered"),
        Ok(_) => {
            let progress = mesher.progress();
            push(fields, "status", format!("time-cap:{:?}", progress.phase));
            push(fields, "progress_done", progress.done);
            push(fields, "progress_total", progress.total);
        }
        Err(error) => {
            push(fields, "status", variant(&error));
            push(fields, "detail", format!("{error:?}"));
            if let TetraError::SurfaceNotRecovered { steiner_points, .. } = error {
                push(fields, "steiner_kept", steiner_points);
                push(fields, "steiner_limit", surface.triangles.len() / 4 + 64);
                push(fields, "no_room", report.constraints_without_room);
                let g = report.gave_up;
                push(fields, "gu_point_limit", g.point_limit);
                push(fields, "gu_no_entry", g.no_entry);
                push(fields, "gu_no_place", g.no_place);
                push(fields, "gu_points_spent", g.points_spent);
                push(fields, "gu_nothing_crossing", g.nothing_crossing);
                push(fields, "gu_boundary_missing", g.boundary_missing);
                let (edges, faces) = mesher.missing_constraints();
                push(fields, "missing_edges", edges.len());
                push(fields, "missing_faces", faces.len());
                leftovers(surface, h, &edges, &faces, fields);
            }
        }
    }
}

/// The variant name at the head of an error's derived `Debug` text.
fn variant(error: &TetraError) -> String {
    let text = format!("{error:?}");
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .next()
        .unwrap_or_default()
        .to_string()
}

/// Connected shells, how many enclose negative volume (a void, or a shell wound inside out), and
/// how many sit inside another shell's bounding box (nested).
fn shells(surface: &TriSoup, fields: &mut Vec<(String, String)>) {
    let n = surface.vertices.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for t in &surface.triangles {
        for k in 1..3 {
            let (a, b) = (
                root(&mut parent, t[0] as usize),
                root(&mut parent, t[k] as usize),
            );
            if a != b {
                parent[a.max(b)] = a.min(b);
            }
        }
    }
    let mut ids: Vec<usize> = Vec::new();
    let mut volume: Vec<f64> = Vec::new();
    let mut boxes: Vec<[P3; 2]> = Vec::new();
    for (index, t) in surface.triangles.iter().enumerate() {
        let r = root(&mut parent, t[0] as usize);
        let shell = match ids.iter().position(|&id| id == r) {
            Some(shell) => shell,
            None => {
                ids.push(r);
                volume.push(0.0);
                boxes.push([[f64::INFINITY; 3], [f64::NEG_INFINITY; 3]]);
                ids.len() - 1
            }
        };
        let [a, b, c] = surface.corners(index);
        volume[shell] += dot(a, cross(b, c)) / 6.0;
        for p in [a, b, c] {
            for k in 0..3 {
                boxes[shell][0][k] = boxes[shell][0][k].min(p[k]);
                boxes[shell][1][k] = boxes[shell][1][k].max(p[k]);
            }
        }
    }
    let inside = |i: usize, j: usize| {
        (0..3).all(|k| boxes[i][0][k] >= boxes[j][0][k] && boxes[i][1][k] <= boxes[j][1][k])
    };
    let nested = (0..ids.len())
        .filter(|&i| (0..ids.len()).any(|j| j != i && inside(i, j)))
        .count();
    push(fields, "shells", ids.len());
    push(
        fields,
        "shells_negative",
        volume.iter().filter(|v| **v < 0.0).count(),
    );
    push(fields, "shells_nested", nested);
}

/// Where one surface edge sits: bend across it (degrees, 0 flat), the smaller of the smallest
/// angles of its two triangles (degrees), and the gap from its midpoint to the nearest triangle
/// touching neither end, over its length.
struct EdgeReading {
    bend: f64,
    min_angle: f64,
    gap: f64,
    length: f64,
}

fn read_edge(surface: &TriSoup, a: u32, b: u32) -> Option<EdgeReading> {
    let sides: Vec<usize> = (0..surface.triangles.len())
        .filter(|&t| surface.triangles[t].contains(&a) && surface.triangles[t].contains(&b))
        .collect();
    let [s, t] = sides[..] else { return None };
    let ns = face_normal(&surface.corners(s))?;
    let nt = face_normal(&surface.corners(t))?;
    let bend = dot(ns, nt).clamp(-1.0, 1.0).acos().to_degrees();
    let min_angle = smallest_angle(&surface.corners(s)).min(smallest_angle(&surface.corners(t)));
    let (pa, pb) = (surface.vertices[a as usize], surface.vertices[b as usize]);
    let mid = [0, 1, 2].map(|k| (pa[k] + pb[k]) * 0.5);
    let edge_length = length(sub(pb, pa));
    let gap = nearest_away(surface, mid, &[a, b]) / edge_length;
    Some(EdgeReading {
        bend,
        min_angle,
        gap,
        length: edge_length,
    })
}

/// The distance from `p` to the nearest triangle that uses none of `touching`.
fn nearest_away(surface: &TriSoup, p: P3, touching: &[u32]) -> f64 {
    (0..surface.triangles.len())
        .filter(|&t| !surface.triangles[t].iter().any(|v| touching.contains(v)))
        .map(|t| point_triangle_distance_squared(p, &surface.corners(t)))
        .fold(f64::INFINITY, f64::min)
        .sqrt()
}

fn smallest_angle(c: &[P3; 3]) -> f64 {
    (0..3)
        .map(|i| {
            let (u, v) = (sub(c[(i + 1) % 3], c[i]), sub(c[(i + 2) % 3], c[i]));
            (dot(u, v) / (length(u) * length(v)))
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees()
        })
        .fold(f64::INFINITY, f64::min)
}

/// Readings over a stride of `edges`, summarised as medians and extremes under `prefix`.
fn summarise(
    surface: &TriSoup,
    h: f64,
    edges: &[[u32; 2]],
    prefix: &str,
    fields: &mut Vec<(String, String)>,
) {
    let stride = (edges.len() / LEFTOVER_SAMPLE.max(1)).max(1);
    let readings: Vec<EdgeReading> = edges
        .iter()
        .step_by(stride)
        .filter_map(|&[a, b]| read_edge(surface, a, b))
        .collect();
    if readings.is_empty() {
        return;
    }
    let sorted = |f: &dyn Fn(&EdgeReading) -> f64| {
        let mut v: Vec<f64> = readings.iter().map(f).collect();
        v.sort_by(f64::total_cmp);
        v
    };
    let count = readings.len() as f64;
    let share =
        |f: &dyn Fn(&EdgeReading) -> bool| readings.iter().filter(|r| f(r)).count() as f64 / count;
    let bend = sorted(&|r| r.bend);
    let angle = sorted(&|r| r.min_angle);
    let gap = sorted(&|r| r.gap);
    let len = sorted(&|r| r.length / h);
    push(fields, &format!("{prefix}_n"), readings.len());
    push(
        fields,
        &format!("{prefix}_bend_med"),
        percentile(&bend, 0.5),
    );
    push(
        fields,
        &format!("{prefix}_flat_share"),
        share(&|r| r.bend < 1.0),
    );
    push(
        fields,
        &format!("{prefix}_sharp_share"),
        share(&|r| r.bend > 30.0),
    );
    push(
        fields,
        &format!("{prefix}_minang_med"),
        percentile(&angle, 0.5),
    );
    push(fields, &format!("{prefix}_minang_min"), angle[0]);
    push(
        fields,
        &format!("{prefix}_skinny_share"),
        share(&|r| r.min_angle < 10.0),
    );
    push(fields, &format!("{prefix}_gap_med"), percentile(&gap, 0.5));
    push(fields, &format!("{prefix}_gap_min"), gap[0]);
    push(
        fields,
        &format!("{prefix}_near_share"),
        share(&|r| r.gap < 1.0),
    );
    push(fields, &format!("{prefix}_len_med"), percentile(&len, 0.5));
}

/// The missing edges, and the edges of the missing triangles, against a stride of all edges.
fn leftovers(
    surface: &TriSoup,
    h: f64,
    edges: &[[u32; 2]],
    faces: &[[u32; 3]],
    fields: &mut Vec<(String, String)>,
) {
    summarise(surface, h, edges, "me", fields);
    let face_edges: Vec<[u32; 2]> = faces
        .iter()
        .flat_map(|t| [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]])
        .collect();
    summarise(surface, h, &face_edges, "mf", fields);
    let all: Vec<[u32; 2]> = surface
        .triangles
        .iter()
        .flat_map(|t| [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]])
        .filter(|[a, b]| a < b)
        .collect();
    let stride = (all.len() / BASELINE_SAMPLE).max(1);
    let baseline: Vec<[u32; 2]> = all.into_iter().step_by(stride).collect();
    summarise(surface, h, &baseline, "base", fields);
}
