// The shapes surface recovery must get right, each in its exact axis-aligned form and rotated so
// flat faces are only nearly coplanar: nothing here may pass by coincidence of alignment. Shapes
// that flips alone cannot finish today are pinned to the named refusal in the last section, so a
// missing triangle can never come back as a mesh, and an improvement to recovery shows up here.

use crate::check::{assert_solid, mesh};
use crate::shapes::{
    hollow_box, needle_box, plate_with_holes, prism, rotated, schoenhardt, subdivided_box, Surface,
};
use solidfem_tetra::TetraError;

/// The surface is fully recovered and its solid mesh passes every independent check, axis
/// aligned and rotated.
fn assert_both_forms(name: &str, surface: &Surface) {
    assert_axis_form(name, surface);
    assert_rotated_form(name, surface);
}

fn assert_axis_form(name: &str, surface: &Surface) {
    let exact = mesh(&surface.points, &surface.triangles)
        .unwrap_or_else(|error| panic!("{name}, axis aligned: {error}"));
    assert_solid(&exact, &surface.points, &surface.triangles);
}

fn assert_rotated_form(name: &str, surface: &Surface) {
    let (points, triangles) = rotated(surface);
    let spun = mesh(&points, &triangles).unwrap_or_else(|error| panic!("{name}, rotated: {error}"));
    assert_solid(&spun, &points, &triangles);
}

/// The mesher stops with the named refusal counting what is missing, never a partial mesh.
fn assert_refused(name: &str, points: &[[f64; 3]], triangles: &[[u32; 3]]) {
    match mesh(points, triangles) {
        Err(TetraError::SurfaceNotRecovered {
            missing_edges,
            missing_faces,
            ..
        }) => assert!(
            missing_edges > 0 || missing_faces > 0,
            "{name}: the refusal counts what is missing"
        ),
        Err(other) => panic!("{name}: expected the named refusal, got {other}"),
        Ok(_) => panic!("{name}: expected the named refusal, got a mesh"),
    }
}

#[test]
fn a_cube() {
    assert_both_forms("cube", &subdivided_box([0.0; 3], [1.0; 3], 1));
    assert_both_forms("subdivided cube", &subdivided_box([0.0; 3], [2.0; 3], 4));
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
fn separate_bodies_and_a_body_inside_a_void() {
    let mut two = subdivided_box([0.0; 3], [1.0; 3], 2);
    two.add(&subdivided_box([3.0; 3], [4.0; 3], 2));
    assert_both_forms("two separate bodies", &two);

    let mut nested = hollow_box(10.0, 1.0, 2, 2);
    nested.add(&subdivided_box([4.0; 3], [6.0; 3], 1));
    assert_both_forms("a body inside a void", &nested);
}

#[test]
fn rotated_plate_and_prisms_recover_flat_needles_and_fans() {
    let holes = [(1, 1, true), (3, 1, false), (2, 2, true)];
    assert_rotated_form(
        "plate with holes",
        &plate_with_holes(10.0, 6.0, 0.6, 5, 3, 4, &holes),
    );
    assert_rotated_form("fan prism", &prism(40, 5.0, 2.0, false));
    assert_rotated_form("zigzag prism", &prism(40, 5.0, 2.0, true));
    assert_rotated_form("short needle box", &needle_box(40.0, 8));
}

/// This stage only flips, so a polyhedron with no tetrahedralization without a new point is
/// refused by name, with the count of what could not be recovered.
#[test]
fn the_schoenhardt_polyhedron_is_refused_by_name() {
    let surface = schoenhardt();
    assert_refused("schoenhardt", &surface.points, &surface.triangles);
}

// Known gaps of flip-only recovery, pinned so a wrong result cannot pass quietly: exact coplanar
// sets (lattice points, ruled boxes) can leave a flat-face edge with no legal flip from any end.
// Each entry is the exact form that is refused today; the rotated forms of these shapes that do
// recover are in the test above.

#[test]
fn flip_only_recovery_refuses_these_axis_aligned_flat_shapes_by_name() {
    let plate = plate_with_holes(
        10.0,
        6.0,
        0.6,
        5,
        3,
        4,
        &[(1, 1, true), (3, 1, false), (2, 2, true)],
    );
    let slab = subdivided_box([0.0, 0.0, 0.0], [100.0, 80.0, 0.5], 6);
    for (name, surface) in [
        ("slab", slab),
        ("plate with holes", plate),
        ("fan prism", prism(40, 5.0, 2.0, false)),
        ("zigzag prism", prism(40, 5.0, 2.0, true)),
        ("needle box", needle_box(40.0, 200)),
    ] {
        assert_refused(name, &surface.points, &surface.triangles);
    }
}

#[test]
fn flip_only_recovery_refuses_these_rotated_flat_shapes_by_name() {
    for (name, surface) in [
        (
            "slab",
            subdivided_box([0.0, 0.0, 0.0], [100.0, 80.0, 0.5], 6),
        ),
        ("needle box", needle_box(40.0, 200)),
    ] {
        let (points, triangles) = rotated(&surface);
        assert_refused(name, &points, &triangles);
    }
}
