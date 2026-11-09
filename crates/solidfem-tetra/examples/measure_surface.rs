//! Measures surface recovery in release: what the Delaunay step already holds, which flips
//! restored the rest, failures and timing, on boxes and hollow boxes of about 10k and 100k
//! surface triangles, rotated so flat faces are nearly but not exactly coplanar. Run with
//! `cargo run --release -p solidfem-tetra --example measure_surface`.

use std::collections::HashMap;
use std::time::Instant;

use solidfem_tetra::{StepStatus, SurfaceMesher};

/// A rotation applied to every point so flat faces stop being axis aligned.
fn rotate(p: [f64; 3]) -> [f64; 3] {
    let (sa, ca) = (0.71_f64.sin(), 0.71_f64.cos());
    let (sb, cb) = (1.13_f64.sin(), 1.13_f64.cos());
    [
        ca * cb * p[0] - sa * p[1] + ca * sb * p[2],
        sa * cb * p[0] + ca * p[1] + sa * sb * p[2],
        -sb * p[0] + cb * p[2],
    ]
}

struct Surface {
    points: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    index: HashMap<[u64; 3], u32>,
}

impl Surface {
    fn new() -> Self {
        Self {
            points: Vec::new(),
            triangles: Vec::new(),
            index: HashMap::new(),
        }
    }

    fn vertex(&mut self, p: [f64; 3]) -> u32 {
        let points = &mut self.points;
        *self.index.entry(p.map(f64::to_bits)).or_insert_with(|| {
            points.push(p);
            (points.len() - 1) as u32
        })
    }

    fn quad(&mut self, a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) {
        let [a, b, c, d] = [a, b, c, d].map(|p| self.vertex(p));
        self.triangles.push([a, b, c]);
        self.triangles.push([a, c, d]);
    }
}

/// Six faces of `n x n` quads, wound outward, or inward for a void.
fn add_box(s: &mut Surface, low: [f64; 3], high: [f64; 3], n: usize, inward: bool) {
    let at = |axis: usize, i: usize| {
        if i == 0 {
            low[axis]
        } else if i == n {
            high[axis]
        } else {
            low[axis] + (high[axis] - low[axis]) * i as f64 / n as f64
        }
    };
    for i in 0..n {
        for j in 0..n {
            let quads = [
                (
                    [at(0, i), at(1, j), low[2]],
                    [at(0, i), at(1, j + 1), low[2]],
                    [at(0, i + 1), at(1, j + 1), low[2]],
                    [at(0, i + 1), at(1, j), low[2]],
                ),
                (
                    [at(0, i), at(1, j), high[2]],
                    [at(0, i + 1), at(1, j), high[2]],
                    [at(0, i + 1), at(1, j + 1), high[2]],
                    [at(0, i), at(1, j + 1), high[2]],
                ),
                (
                    [at(0, i), low[1], at(2, j)],
                    [at(0, i + 1), low[1], at(2, j)],
                    [at(0, i + 1), low[1], at(2, j + 1)],
                    [at(0, i), low[1], at(2, j + 1)],
                ),
                (
                    [at(0, i), high[1], at(2, j)],
                    [at(0, i), high[1], at(2, j + 1)],
                    [at(0, i + 1), high[1], at(2, j + 1)],
                    [at(0, i + 1), high[1], at(2, j)],
                ),
                (
                    [low[0], at(1, i), at(2, j)],
                    [low[0], at(1, i), at(2, j + 1)],
                    [low[0], at(1, i + 1), at(2, j + 1)],
                    [low[0], at(1, i + 1), at(2, j)],
                ),
                (
                    [high[0], at(1, i), at(2, j)],
                    [high[0], at(1, i + 1), at(2, j)],
                    [high[0], at(1, i + 1), at(2, j + 1)],
                    [high[0], at(1, i), at(2, j + 1)],
                ),
            ];
            for (a, b, c, d) in quads {
                if inward {
                    s.quad(a, d, c, b);
                } else {
                    s.quad(a, b, c, d);
                }
            }
        }
    }
}

fn measure(name: &str, surface: Surface) {
    let triangles = surface.triangles.len();
    let start = Instant::now();
    let mut mesher = SurfaceMesher::new(surface.points, surface.triangles).expect("valid surface");
    let status = mesher.step(usize::MAX, &mut || false);
    let seconds = start.elapsed().as_secs_f64();
    let report = mesher.report();
    let tets = mesher.result().map(|solid| solid.tets.len()).unwrap_or(0);
    let outcome = match status {
        Ok(StepStatus::Done) => "ok".to_string(),
        Ok(other) => format!("{other:?}"),
        Err(error) => format!("refused: {error}"),
    };
    println!(
        "{name}: triangles={triangles} time {seconds:.3}s tets={tets} edges {}/{} present, faces \
         {}/{} present; flips 2-3={} 3-2={} 4-4={} larger={} steiner={} failed edges={} faces={} \
         => {outcome}",
        report.edges_present_after_delaunay,
        report.edges_total,
        report.faces_present_after_delaunay,
        report.faces_total,
        report.flips_two_to_three,
        report.flips_three_to_two,
        report.flips_four_to_four,
        report.flips_larger_ring,
        report.steiner_points,
        report.edges_failed,
        report.faces_failed,
    );
}

fn main() {
    for (n, label) in [(29usize, "~10k"), (91, "~100k")] {
        let mut solid = Surface::new();
        add_box(&mut solid, [0.0; 3], [100.0; 3], n, false);
        let rotated = Surface {
            points: solid.points.iter().map(|&p| rotate(p)).collect(),
            triangles: solid.triangles.clone(),
            index: HashMap::new(),
        };
        measure(&format!("box {label}, axis aligned"), solid);
        measure(&format!("box {label}, rotated"), rotated);
    }
    for (n, label) in [(20usize, "~10k"), (64, "~100k")] {
        let side = 100.0_f64;
        let wall = (3.0_f64 * side * side).sqrt() / 125.0;
        let mut hollow = Surface::new();
        add_box(&mut hollow, [0.0; 3], [side; 3], n, false);
        add_box(&mut hollow, [wall; 3], [side - wall; 3], n, true);
        let rotated = Surface {
            points: hollow.points.iter().map(|&p| rotate(p)).collect(),
            triangles: hollow.triangles.clone(),
            index: HashMap::new(),
        };
        measure(&format!("thin hollow box {label}, axis aligned"), hollow);
        measure(&format!("thin hollow box {label}, rotated"), rotated);
    }
}
