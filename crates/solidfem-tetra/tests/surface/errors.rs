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
