// The shapes surface recovery must get right, each in its exact axis-aligned form and rotated so
// flat faces are only nearly coplanar: nothing here may pass by coincidence of alignment.

use crate::check::{assert_solid, mesh};
use crate::shapes::{
    hollow_box, needle_box, plate_with_holes, prism, rotated, schoenhardt, schoenhardt_of_height,
    subdivided_box, Surface,
};
use solidfem_tetra::TetraError;

/// The surface is fully recovered and its solid mesh passes every independent check, axis
/// aligned and rotated.
fn assert_both_forms(name: &str, surface: &Surface) {
    let (exact, _) = mesh(&surface.points, &surface.triangles)
        .unwrap_or_else(|error| panic!("{name}, axis aligned: {error}"));
    assert_solid(&exact, &surface.points, &surface.triangles);
    let (points, triangles) = rotated(surface);
    let (spun, _) =
        mesh(&points, &triangles).unwrap_or_else(|error| panic!("{name}, rotated: {error}"));
    assert_solid(&spun, &points, &triangles);
}

#[test]
fn a_cube() {
    assert_both_forms("cube", &subdivided_box([0.0; 3], [1.0; 3], 1));
    assert_both_forms("subdivided cube", &subdivided_box([0.0; 3], [2.0; 3], 4));
}

#[test]
fn a_thin_slab() {
    assert_both_forms(
        "slab",
        &subdivided_box([0.0, 0.0, 0.0], [100.0, 80.0, 0.5], 6),
    );
}

#[test]
fn a_box_with_a_box_void() {
    assert_both_forms("box with void", &hollow_box(10.0, 2.0, 3, 2));
}

#[test]
fn a_thin_walled_hollow_box_with_no_vertex_through_the_wall() {
    // A wall of 1/125 of the diagonal of the part: the surface has no vertex through the wall.
    let side = 100.0;
    let diagonal = (3.0_f64 * side * side).sqrt();
    assert_both_forms("thin hollow box", &hollow_box(side, diagonal / 125.0, 6, 6));
}

#[test]
fn a_plate_with_square_and_round_holes() {
    let holes = [(1, 1, true), (3, 1, false), (2, 2, true)];
    assert_both_forms(
        "plate with holes",
        &plate_with_holes(10.0, 6.0, 0.6, 5, 3, 4, &holes),
    );
}

#[test]
fn separate_bodies_and_a_body_inside_a_void() {
    let mut two = subdivided_box([0.0; 3], [1.0; 3], 2);
    two.add(&subdivided_box([3.0; 3], [4.0; 3], 2));
    assert_both_forms("two separate bodies", &two);

    let mut nested = hollow_box(10.0, 1.0, 2, 2);
    nested.add(&subdivided_box([4.0; 3], [6.0; 3], 1));
    assert_both_forms("a body inside a void", &nested);
}

#[test]
fn flat_faces_triangulated_as_needles_and_fans() {
    assert_both_forms("short needle box", &needle_box(40.0, 8));
    assert_both_forms("long needle box", &needle_box(40.0, 200));
    assert_both_forms("fan prism", &prism(40, 5.0, 2.0, false));
    assert_both_forms("zigzag prism", &prism(40, 5.0, 2.0, true));
}

/// Six vertices have no tetrahedralization of this polyhedron, so at least one point must be
/// added, and it must be inside the solid.
#[test]
fn the_schoenhardt_polyhedron_recovers_with_an_interior_point() {
    let surface = schoenhardt();
    let (result, report) = mesh(&surface.points, &surface.triangles)
        .unwrap_or_else(|error| panic!("schoenhardt: {error}"));
    assert_solid(&result, &surface.points, &surface.triangles);
    assert_eq!(
        report.steiner_points, 1,
        "the polyhedron needs exactly one added point"
    );
}

/// With a void around its centre the polyhedron leaves no legal place for enough interior points:
/// the mesher stops with the named refusal, counting what is missing and saying there was no
/// room, not a mesh with a hole in the constraints.
#[test]
fn a_schoenhardt_polyhedron_with_a_central_void_is_refused_with_counts_saying_no_room() {
    let mut surface = schoenhardt();
    surface.add_reversed(&subdivided_box([-0.2, -0.2, 0.3], [0.2, 0.2, 0.7], 1));
    let error = mesh(&surface.points, &surface.triangles)
        .expect_err("this surface cannot be finished today");
    match error {
        TetraError::SurfaceNotRecovered {
            missing_edges,
            missing_faces,
            constraints_without_room,
            ..
        } => {
            assert!(
                missing_edges > 0 || missing_faces > 0,
                "the refusal counts what is missing"
            );
            assert!(
                constraints_without_room > 0,
                "the refusal says there was no room"
            );
        }
        other => panic!("expected the named refusal, got {other}"),
    }
}

/// A much flatter copy still has room inside for its point, so it recovers the same way.
#[test]
fn a_very_flat_schoenhardt_polyhedron_recovers_with_an_interior_point() {
    let surface = schoenhardt_of_height(1e-9);
    let (result, report) = mesh(&surface.points, &surface.triangles)
        .unwrap_or_else(|error| panic!("flat schoenhardt: {error}"));
    assert_solid(&result, &surface.points, &surface.triangles);
    assert!(
        report.steiner_points >= 1,
        "the polyhedron needs an added point"
    );
}
