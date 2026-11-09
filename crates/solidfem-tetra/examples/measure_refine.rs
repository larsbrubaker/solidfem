//! Measures refinement in release: the through-wall layers, size-driven insertion and quality pass
//! on 100 x 100 x 0.8 mm plates at about 2.0 and 1.2 mm surface edges with two layers, and a block
//! at a moderate size. Prints tets, points, time per stage, the longest single step, and the
//! minimum dihedral histograms before and after the quality pass. Run with
//! `cargo run --release -p solidfem-tetra --example measure_refine`.

use std::time::Instant;

use solidfem_tetra::{
    DihedralHistogram, RefineOptions, RefinePhase, Refiner, SizeFn, SolidMesh, StepStatus,
    SurfaceMesher,
};

/// A `length x width x thick` plate whose top and bottom are one grid of squares of side
/// `length / cells`, with a single quad up each side of the rim.
fn plate(length: f64, width: f64, thick: f64, cells: usize) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let n = cells + 1;
    let at = |i: usize, j: usize, layer: usize| ((layer * n + i) * n + j) as u32;
    let mut points = Vec::with_capacity(2 * n * n);
    for layer in 0..2 {
        for i in 0..n {
            for j in 0..n {
                points.push([
                    length * i as f64 / cells as f64,
                    width * j as f64 / cells as f64,
                    thick * layer as f64,
                ]);
            }
        }
    }
    let mut triangles = Vec::with_capacity(4 * cells * cells + 8 * cells);
    let mut quad = |a: u32, b: u32, c: u32, d: u32| {
        triangles.push([a, b, c]);
        triangles.push([a, c, d]);
    };
    for i in 0..cells {
        for j in 0..cells {
            quad(
                at(i, j, 1),
                at(i + 1, j, 1),
                at(i + 1, j + 1, 1),
                at(i, j + 1, 1),
            );
            quad(
                at(i, j, 0),
                at(i, j + 1, 0),
                at(i + 1, j + 1, 0),
                at(i + 1, j, 0),
            );
        }
        quad(at(i, 0, 0), at(i + 1, 0, 0), at(i + 1, 0, 1), at(i, 0, 1));
        quad(
            at(i + 1, cells, 0),
            at(i, cells, 0),
            at(i, cells, 1),
            at(i + 1, cells, 1),
        );
        quad(at(0, i + 1, 0), at(0, i, 0), at(0, i, 1), at(0, i + 1, 1));
        quad(
            at(cells, i, 0),
            at(cells, i + 1, 0),
            at(cells, i + 1, 1),
            at(cells, i, 1),
        );
    }
    (points, triangles)
}

/// A cube of side `side` with every face split into `n x n` squares.
fn block(side: f64, n: usize) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let m = n + 1;
    let mut index = std::collections::HashMap::new();
    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut vertex = |x: usize, y: usize, z: usize| -> u32 {
        *index.entry((x, y, z)).or_insert_with(|| {
            points.push([
                side * x as f64 / n as f64,
                side * y as f64 / n as f64,
                side * z as f64 / n as f64,
            ]);
            (points.len() - 1) as u32
        })
    };
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut quad = |a: u32, b: u32, c: u32, d: u32| {
        triangles.push([a, b, c]);
        triangles.push([a, c, d]);
    };
    for i in 0..m - 1 {
        for j in 0..m - 1 {
            let (a, b, c, d) = (
                vertex(i, j, 0),
                vertex(i, j + 1, 0),
                vertex(i + 1, j + 1, 0),
                vertex(i + 1, j, 0),
            );
            quad(a, b, c, d);
            let (a, b, c, d) = (
                vertex(i, j, n),
                vertex(i + 1, j, n),
                vertex(i + 1, j + 1, n),
                vertex(i, j + 1, n),
            );
            quad(a, b, c, d);
            let (a, b, c, d) = (
                vertex(i, 0, j),
                vertex(i + 1, 0, j),
                vertex(i + 1, 0, j + 1),
                vertex(i, 0, j + 1),
            );
            quad(a, b, c, d);
            let (a, b, c, d) = (
                vertex(i, n, j),
                vertex(i, n, j + 1),
                vertex(i + 1, n, j + 1),
                vertex(i + 1, n, j),
            );
            quad(a, b, c, d);
            let (a, b, c, d) = (
                vertex(0, i, j),
                vertex(0, i, j + 1),
                vertex(0, i + 1, j + 1),
                vertex(0, i + 1, j),
            );
            quad(a, b, c, d);
            let (a, b, c, d) = (
                vertex(n, i, j),
                vertex(n, i + 1, j),
                vertex(n, i + 1, j + 1),
                vertex(n, i, j + 1),
            );
            quad(a, b, c, d);
        }
    }
    (points, triangles)
}

fn recover(points: Vec<[f64; 3]>, triangles: Vec<[u32; 3]>) -> SolidMesh {
    let start = Instant::now();
    let mut mesher = SurfaceMesher::new(points, triangles).expect("valid surface");
    assert_eq!(
        mesher.step(usize::MAX, &mut || false).expect("recovers"),
        StepStatus::Done
    );
    let solid = mesher.result().expect("finished");
    println!(
        "  surface recovery: {} tets, {} points, {:.3} s",
        solid.tets.len(),
        solid.points.len(),
        start.elapsed().as_secs_f64()
    );
    solid
}

fn print_histogram(name: &str, histogram: &DihedralHistogram) {
    println!(
        "  {name}: {} tets, min dihedral {:.2}, mean {:.2}, worst scaled Jacobian {:.3}",
        histogram.tets,
        histogram.min_degrees,
        histogram.mean_degrees,
        histogram.worst_scaled_jacobian
    );
    println!("    bins of 5 degrees from 0: {:?}", histogram.counts);
}

fn run(name: &str, points: Vec<[f64; 3]>, triangles: Vec<[u32; 3]>, size: f64, layers: u32) {
    println!("{name}:");
    let solid = recover(points, triangles);
    let size_function: SizeFn = Box::new(move |_| size);
    let options = RefineOptions {
        layers_through_walls: layers,
        ..RefineOptions::default()
    };
    let mut refiner = Refiner::new(solid, size_function, options).expect("valid");
    let (mut times, mut slowest, mut stage_start) = ([0.0_f64; 3], 0.0_f64, Instant::now());
    let mut phase = refiner.progress().phase;
    loop {
        let before = Instant::now();
        let status = refiner.step(1, &mut || false).expect("refines");
        slowest = slowest.max(before.elapsed().as_secs_f64());
        let current = refiner.progress().phase;
        if current != phase || status == StepStatus::Done {
            let slot = match phase {
                RefinePhase::Layers => 0,
                RefinePhase::Sizing => 1,
                RefinePhase::Quality | RefinePhase::Done => 2,
            };
            times[slot] += stage_start.elapsed().as_secs_f64();
            stage_start = Instant::now();
            phase = current;
        }
        if status == StepStatus::Done {
            break;
        }
    }
    let result = refiner.result().expect("finished");
    let report = refiner.report();
    println!(
        "  refined: {} tets, {} points; layers {:.3} s, sizing {:.3} s, quality {:.3} s; longest single step {:.1} ms",
        result.tets.len(),
        result.points.len(),
        times[0],
        times[1],
        times[2],
        slowest * 1e3,
    );
    println!(
        "  layer points {} (dropped {}, thin-wall vertices {}, too thin for rounding {}), size points {} (refused {}), flips 2-3 {}, 3-2 {}, 4-4 {}, larger {}, improvements {}",
        report.layer_points,
        report.layer_points_dropped,
        report.thin_wall_vertices,
        report.walls_too_thin_for_rounding,
        report.size_points,
        report.size_points_refused,
        report.flips_two_to_three,
        report.flips_three_to_two,
        report.flips_four_to_four,
        report.flips_larger_ring,
        report.quality_improvements,
    );
    print_histogram("before quality", &report.before);
    print_histogram("after quality", &report.after);
    let surface_count = result
        .neighbors
        .iter()
        .flatten()
        .filter(|&&n| n == solidfem_tetra::HULL)
        .count();
    let worst = result
        .tets
        .iter()
        .min_by(|a, b| min_dihedral(&result, a).total_cmp(&min_dihedral(&result, b)));
    if let Some(tet) = worst {
        println!(
            "  worst tetrahedron ({} boundary faces in the mesh): {:?}",
            surface_count,
            tet.map(|v| result.points[v as usize])
        );
    }
}

/// The smallest dihedral of one tetrahedron, for finding the worst.
fn min_dihedral(mesh: &SolidMesh, tet: &[u32; 4]) -> f64 {
    let p = tet.map(|v| mesh.points[v as usize]);
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let unit = |a: [f64; 3]| {
        let l = dot(a, a).sqrt();
        [a[0] / l, a[1] / l, a[2] / l]
    };
    let mut smallest = 180.0_f64;
    for (i, j, k, l) in [
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 3, 1, 2),
        (1, 2, 0, 3),
        (1, 3, 0, 2),
        (2, 3, 0, 1),
    ] {
        let e = unit(sub(p[j], p[i]));
        let across = |v: [f64; 3]| {
            let d = sub(v, p[i]);
            let along = dot(d, e);
            unit([
                d[0] - e[0] * along,
                d[1] - e[1] * along,
                d[2] - e[2] * along,
            ])
        };
        smallest = smallest.min(
            dot(across(p[k]), across(p[l]))
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees(),
        );
    }
    smallest
}

fn main() {
    let (points, triangles) = plate(100.0, 100.0, 0.8, 50);
    run(
        "plate 100 x 100 x 0.8, 2.0 mm edges, 2 layers",
        points,
        triangles,
        3.0,
        2,
    );
    let (points, triangles) = plate(100.0, 100.0, 0.8, 83);
    run(
        "plate 100 x 100 x 0.8, 1.2 mm edges, 2 layers",
        points,
        triangles,
        2.0,
        2,
    );
    let (points, triangles) = block(100.0, 16);
    run(
        "block 100 mm, 6.25 mm surface edges, size 10",
        points,
        triangles,
        10.0,
        0,
    );
}
