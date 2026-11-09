// Surfaces the mesher must refuse before it works on them, each with the named error a caller can
// act on.

use crate::shapes::{subdivided_box, Surface};
use solidfem_tetra::{SurfaceMesher, TetraError};

fn new_error(points: &[[f64; 3]], triangles: &[[u32; 3]]) -> TetraError {
    SurfaceMesher::new(points.to_vec(), triangles.to_vec())
        .err()
        .expect("the surface must be refused")
}

#[test]
fn an_open_surface_is_refused_naming_an_edge() {
    let mut surface = subdivided_box([0.0; 3], [1.0; 3], 1);
    surface.triangles.pop();
    assert!(matches!(
        new_error(&surface.points, &surface.triangles),
        TetraError::SurfaceNotClosed { .. }
    ));
}

#[test]
fn a_flipped_triangle_breaks_the_winding_and_is_refused() {
    let mut surface = subdivided_box([0.0; 3], [1.0; 3], 1);
    surface.triangles[0].swap(1, 2);
    assert!(matches!(
        new_error(&surface.points, &surface.triangles),
        TetraError::SurfaceNotClosed { .. }
    ));
}

#[test]
fn a_surface_wound_inside_out_is_refused_after_recovery() {
    let mut surface = subdivided_box([0.0; 3], [1.0; 3], 2);
    for triangle in &mut surface.triangles {
        triangle.swap(1, 2);
    }
    let mut mesher =
        SurfaceMesher::new(surface.points.clone(), surface.triangles.clone()).expect("closed");
    let error = mesher
        .step(usize::MAX, &mut || false)
        .expect_err("wrong winding");
    assert!(
        matches!(error, TetraError::SurfaceWindingWrong { .. }),
        "{error}"
    );
}

#[test]
fn a_flat_triangle_is_refused() {
    let surface = Surface {
        points: vec![[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        triangles: vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
        ..Surface::default()
    };
    assert!(matches!(
        new_error(&surface.points, &surface.triangles),
        TetraError::DegenerateTriangle { triangle: 0 }
    ));
}

#[test]
fn an_unused_point_and_a_bad_index_are_refused() {
    let surface = subdivided_box([0.0; 3], [1.0; 3], 1);
    let mut points = surface.points.clone();
    points.push([9.0, 9.0, 9.0]);
    assert!(matches!(
        new_error(&points, &surface.triangles),
        TetraError::UnusedVertex { .. }
    ));
    let mut triangles = surface.triangles.clone();
    triangles[1][0] = 99;
    assert!(matches!(
        new_error(&surface.points, &triangles),
        TetraError::SurfaceIndexOutOfRange {
            triangle: 1,
            index: 99
        }
    ));
}

/// A 2-unit cube whose top is the pentagon A, e, B, C, D with `e` beyond the front edge A-B, while
/// the front wall still ends at A-B. The fin `[A, B, e]` (the last triangle) closes the surface by
/// covering the sliver between A-B and `e` a second time, wound the other way: a closed,
/// consistently wound surface of the right volume that folds back onto itself across `e`'s two
/// edges, with empty space on both sides of the fin. A remesher that collapses the vertices of a
/// curved rim onto its chords leaves exactly this along a fillet's end caps.
fn cube_with_folded_fin() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let points = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
        [2.0, 0.0, 2.0],
        [2.0, 2.0, 2.0],
        [0.0, 2.0, 2.0],
        [1.0, -0.5, 2.0],
    ];
    let triangles = vec![
        [0, 2, 1],
        [0, 3, 2],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
        [3, 7, 6],
        [3, 6, 2],
        [0, 1, 5],
        [0, 5, 4],
        [8, 5, 6],
        [8, 6, 7],
        [8, 7, 4],
        [4, 5, 8],
    ];
    (points, triangles)
}

#[test]
fn a_surface_folded_back_onto_itself_is_refused_naming_the_fin() {
    let (points, triangles) = cube_with_folded_fin();
    let mut mesher = SurfaceMesher::new(points, triangles).expect("closed and consistently wound");
    let error = mesher
        .step(usize::MAX, &mut || false)
        .expect_err("the fin has empty space on both sides");
    // The fin is the input triangle with nothing solid on either side; the flat top's other
    // triangles are numbered 10 to 12 and the walls below them, so any other number is wrong.
    assert!(
        matches!(error, TetraError::SurfaceDoesNotSeparate { triangle: 13 }),
        "{error:?}"
    );
}
