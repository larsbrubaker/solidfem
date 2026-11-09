// A surface whose edge is blocked by a mesh edge lying across it. The file `le10_n85.txt` beside
// this test is a remeshed thick elliptic plate (1369 vertices): on its curved outer wall a surface
// edge runs just below a chord joining two mid-height vertices, flips cannot remove that chord
// because its ring reaches across the plate, and points placed around the walk's face leave it
// standing. The whole surface is used because no smaller surface that fails has been found. The
// result is checked by the independent exact checks.

use crate::check::{assert_solid, mesh};

pub fn read_surface(text: &str) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let mut tokens = text.split_ascii_whitespace();
    let mut next = || tokens.next().expect("a complete surface file");
    let vertices: usize = next().parse().expect("a vertex count");
    let faces: usize = next().parse().expect("a face count");
    let points = (0..vertices)
        .map(|_| [(); 3].map(|()| next().parse::<f64>().expect("a coordinate")))
        .collect();
    let triangles = (0..faces)
        .map(|_| [(); 3].map(|()| next().parse::<u32>().expect("a vertex index")))
        .collect();
    (points, triangles)
}

#[test]
fn a_surface_edge_under_a_long_chord_of_a_thick_part_is_recovered() {
    let (points, triangles) = read_surface(include_str!("le10_n85.txt"));
    let (solid, report) = mesh(&points, &triangles).unwrap_or_else(|error| panic!("{error}"));
    assert_solid(&solid, &points, &triangles);
    assert_eq!(report.edges_failed, 0, "every surface edge is recovered");
    assert_eq!(
        report.faces_failed, 0,
        "every surface triangle is recovered"
    );
    assert_eq!(
        report.steiner_points,
        solid.points.len() - points.len(),
        "the report counts exactly the added points in the mesh"
    );
    assert!(
        report.steiner_points > 0,
        "the stuck edge is recovered with a point added strictly inside the solid"
    );
}
