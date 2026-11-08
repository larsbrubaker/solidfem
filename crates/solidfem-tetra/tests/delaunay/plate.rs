// A real surface's vertices: a rectangular plate with round holes, the shape the mesher is built
// for, assembled here from simple geometry. Its convex hull is the plain box of the plate.

use crate::support::{
    assert_adjacency, assert_empty_spheres_sampled, assert_euler, assert_positive,
    assert_valid_small, total_volume, triangulate, volume_tolerance,
};

/// The vertices of the closed surface of a `length x width x thick` plate with `holes` round
/// holes: the outer rectangle, a lattice over the top and bottom faces, and an `n`-gon rim per
/// hole on each face. Hull corners and lattice points are exact; the rims are rounded `f64`.
fn plate_with_holes(
    length: f64,
    width: f64,
    thick: f64,
    holes: &[([f64; 2], f64)],
    rim: usize,
    lattice: usize,
) -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    for z in [0.0, thick] {
        for i in 0..=lattice {
            for j in 0..=lattice {
                let (x, y) = (
                    length * i as f64 / lattice as f64,
                    width * j as f64 / lattice as f64,
                );
                if holes.iter().all(|(centre, radius)| {
                    let (dx, dy) = (x - centre[0], y - centre[1]);
                    dx * dx + dy * dy > (radius * 1.05) * (radius * 1.05)
                }) {
                    points.push([x, y, z]);
                }
            }
        }
        for (centre, radius) in holes {
            for k in 0..rim {
                let angle = core::f64::consts::TAU * k as f64 / rim as f64;
                points.push([
                    centre[0] + radius * angle.cos(),
                    centre[1] + radius * angle.sin(),
                    z,
                ]);
            }
        }
    }
    points
}

#[test]
fn a_plate_with_holes_is_a_valid_delaunay_tetrahedralization() {
    let (length, width, thick) = (8.0, 5.0, 1.0);
    let holes = [([2.0, 2.5], 0.8), ([5.0, 2.5], 1.2), ([6.8, 1.0], 0.4)];
    let points = plate_with_holes(length, width, thick, &holes, 24, 12);
    let result = triangulate(points.clone());
    assert_eq!(
        result.points, points,
        "the surface's coordinates are untouched"
    );
    assert_valid_small(&result);
    let volume = length * width * thick;
    let tolerance = volume_tolerance(&result);
    assert!(
        (total_volume(&result) - volume).abs() <= tolerance,
        "volume {} expected {volume} tolerance {tolerance}",
        total_volume(&result)
    );
}

#[test]
fn a_finely_sampled_plate_with_many_holes_stays_valid() {
    let (length, width, thick) = (40.0, 25.0, 0.5);
    let mut holes = Vec::new();
    for i in 0..6 {
        for j in 0..4 {
            holes.push(([f64::from(i) * 6.0 + 4.0, f64::from(j) * 6.0 + 3.5], 1.5));
        }
    }
    let points = plate_with_holes(length, width, thick, &holes, 40, 80);
    let result = triangulate(points);
    assert_positive(&result);
    assert_adjacency(&result);
    assert_euler(&result);
    assert_empty_spheres_sampled(&result, 150);
    let volume = length * width * thick;
    assert!((total_volume(&result) - volume).abs() <= volume_tolerance(&result));
}
