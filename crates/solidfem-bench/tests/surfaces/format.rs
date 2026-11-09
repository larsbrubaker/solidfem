//! The text format holds the remesher's exact coordinates and is what the benchmark runner reads.
//! A generator and reader that disagree on the format, or text that rounds a coordinate, would
//! change every benchmark result without any error.

use std::path::PathBuf;

use solidfem_bench::benchmarks::pipeline::read_surface;
use solidfem_bench::surfaces::{surface_text, write_surface, SurfaceError};
use solidmesh_mesh::AnalysisSurface;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "solidfem-surfaces-format-{}-{name}",
        std::process::id()
    ))
}

fn surface() -> AnalysisSurface {
    AnalysisSurface {
        // Values whose shortest text is long, the worst cases for a format that rounds.
        vertices: vec![
            [0.1 + 0.2, 1.0 / 3.0, -1.5e-7],
            [1234567890123.125, 123456.75, 2.0],
            [0.0, 7.0, -0.5],
        ],
        faces: vec![[0, 1, 2], [2, 1, 0]],
    }
}

#[test]
fn the_text_is_the_counts_then_shortest_round_trip_lines() {
    let text = surface_text(&surface());
    assert_eq!(
        text,
        "3 2\n\
         0.30000000000000004 0.3333333333333333 -0.00000015\n\
         1234567890123.125 123456.75 2\n\
         0 7 -0.5\n\
         0 1 2\n\
         2 1 0\n"
    );
}

#[test]
fn the_benchmark_reader_gets_back_the_exact_bits() {
    let path = temp_path("round-trip.txt");
    let surface = surface();
    write_surface(&path, &surface).expect("a surface is written");
    let (points, triangles) =
        read_surface(path.to_str().expect("utf-8 path")).expect("the runner reads the file");
    assert_eq!(points.len(), surface.vertices.len());
    for (read, written) in points.iter().zip(&surface.vertices) {
        for i in 0..3 {
            assert_eq!(
                read[i].to_bits(),
                written[i].to_bits(),
                "{read:?} {written:?}"
            );
        }
    }
    assert_eq!(triangles, vec![[0, 1, 2], [2, 1, 0]]);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_write_failure_names_the_path() {
    let missing_dir = temp_path("no-such-dir").join("a.txt");
    let Err(error) = write_surface(&missing_dir, &surface()) else {
        panic!("writing into a missing directory must fail");
    };
    assert!(matches!(error, SurfaceError::Write { .. }));
    let text = error.to_string();
    assert!(text.contains("a.txt"), "{text}");
}
