// The inputs CAD produces: exact grids, points all on one sphere, two parallel planes, slabs of
// 1e-6 relative thickness, and many points on a line. Each must give a valid Delaunay
// tetrahedralization of the convex hull with exact coordinates and no flat tetrahedra.

use crate::support::{
    assert_adjacency, assert_empty_spheres_sampled, assert_euler, assert_positive,
    assert_valid_small, total_volume, triangulate, volume_tolerance, Rng,
};

fn grid(n: usize, spacing: f64) -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                points.push([i as f64 * spacing, j as f64 * spacing, k as f64 * spacing]);
            }
        }
    }
    points
}

#[test]
fn a_regular_grid_triangulates_its_cube() {
    let n = 7;
    let result = triangulate(grid(n, 1.0));
    assert_valid_small(&result);
    let cube = ((n - 1) as f64).powi(3);
    let tolerance = volume_tolerance(&result);
    assert!((total_volume(&result) - cube).abs() <= tolerance);
}

#[test]
fn a_larger_grid_stays_valid() {
    let n = 18;
    let result = triangulate(grid(n, 0.25));
    assert_positive(&result);
    assert_adjacency(&result);
    assert_euler(&result);
    assert_empty_spheres_sampled(&result, 300);
    let side = (n - 1) as f64 * 0.25;
    assert!((total_volume(&result) - side.powi(3)).abs() <= volume_tolerance(&result));
}

/// Every integer point on the sphere `x^2 + y^2 + z^2 = r2`: exactly cospherical in `f64`.
fn integer_sphere(r2: i64) -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    let r = (r2 as f64).sqrt().ceil() as i64;
    for x in -r..=r {
        for y in -r..=r {
            let rest = r2 - x * x - y * y;
            if rest < 0 {
                continue;
            }
            let z = (rest as f64).sqrt().round() as i64;
            if z * z == rest {
                points.push([x as f64, y as f64, z as f64]);
                if z != 0 {
                    points.push([x as f64, y as f64, -z as f64]);
                }
            }
        }
    }
    points
}

#[test]
fn points_exactly_on_a_sphere_triangulate() {
    let points = integer_sphere(49 * 49);
    assert!(points.len() > 100, "{} points on the sphere", points.len());
    let result = triangulate(points);
    assert_valid_small(&result);
}

#[test]
fn points_on_a_sphere_with_a_centre_point_triangulate() {
    let mut points = integer_sphere(49 * 49);
    points.push([0.0, 0.0, 0.0]);
    let result = triangulate(points);
    assert_valid_small(&result);
    assert_euler(&result);
}

/// `n x n` lattice points on the planes `z = 0` and `z = thickness`: the hull is a box, and every
/// point lies on its boundary.
fn two_planes(n: usize, spacing: f64, thickness: f64) -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let (x, y) = (i as f64 * spacing, j as f64 * spacing);
            points.push([x, y, 0.0]);
            points.push([x, y, thickness]);
        }
    }
    points
}

#[test]
fn a_plates_top_and_bottom_planes_triangulate_the_plate() {
    let points = two_planes(12, 1.0, 1.0);
    let result = triangulate(points);
    assert_valid_small(&result);
    let volume = 11.0 * 11.0 * 1.0;
    assert!((total_volume(&result) - volume).abs() <= volume_tolerance(&result));
}

#[test]
fn a_slab_of_one_millionth_relative_thickness_triangulates() {
    let side = 1.0;
    let thickness = 1e-6;
    // Random points in the planes, so nothing is a lattice but all are exactly coplanar.
    let mut rng = Rng(3);
    let mut points = Vec::new();
    for _ in 0..300 {
        let (x, y) = (rng.unit(), rng.unit());
        points.push([x, y, 0.0]);
        points.push([x, y, thickness]);
    }
    for corner in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] {
        points.push([corner[0], corner[1], 0.0]);
        points.push([corner[0], corner[1], thickness]);
    }
    let result = triangulate(points);
    assert_valid_small(&result);
    let volume = side * side * thickness;
    let tolerance = volume_tolerance(&result);
    assert!(
        (total_volume(&result) - volume).abs() <= tolerance,
        "volume {} tolerance {tolerance}",
        total_volume(&result)
    );
}

#[test]
fn a_thin_lattice_slab_triangulates() {
    let points = two_planes(10, 0.1, 1e-6);
    let result = triangulate(points);
    assert_valid_small(&result);
}

#[test]
fn many_points_on_a_line_plus_a_few_off_it_triangulate() {
    let mut points: Vec<[f64; 3]> = (0..200).map(|i| [i as f64 * 0.5, 0.0, 0.0]).collect();
    points.extend_from_slice(&[
        [10.0, 3.0, 0.0],
        [40.0, -2.0, 1.0],
        [70.0, 0.0, 5.0],
        [33.0, 1.0, -4.0],
    ]);
    let result = triangulate(points);
    assert_valid_small(&result);
}

#[test]
fn many_points_in_a_plane_plus_one_apex_triangulate() {
    let mut points = Vec::new();
    for i in 0..15 {
        for j in 0..15 {
            points.push([f64::from(i), f64::from(j), 0.0]);
        }
    }
    points.push([7.0, 7.0, 9.0]);
    let result = triangulate(points);
    assert_valid_small(&result);
    // A pyramid: the square base times a third of the height.
    let volume = 14.0 * 14.0 * 9.0 / 3.0;
    assert!((total_volume(&result) - volume).abs() <= volume_tolerance(&result));
}

#[test]
fn nearly_cospherical_tiny_slivers_stay_positive() {
    // Points on a sphere of radius one, rounded to f64, are only nearly cospherical: the exact
    // predicates must order them correctly and the checks must find no flat tetrahedra.
    let mut rng = Rng(11);
    let mut points = Vec::new();
    for _ in 0..400 {
        let z = rng.unit() * 2.0 - 1.0;
        let angle = rng.unit() * core::f64::consts::TAU;
        let radius = (1.0 - z * z).sqrt();
        points.push([radius * angle.cos(), radius * angle.sin(), z]);
    }
    let result = triangulate(points);
    assert_valid_small(&result);
}
