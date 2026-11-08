// General-position inputs: the exact properties that define a Delaunay tetrahedralization of the
// convex hull, on random points small enough to check against every point and on a large set
// checked by sampling tetrahedra throughout the list.

use crate::support::{
    assert_adjacency, assert_empty_spheres_sampled, assert_euler, assert_positive,
    assert_valid_small, brute_hull_volume, total_volume, triangulate, volume_tolerance, Rng,
};

fn random_points(count: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut rng = Rng(seed);
    (0..count).map(|_| rng.point()).collect()
}

/// The predicate conventions the whole crate relies on, checked on a shape with a known answer:
/// a positively oriented tetrahedron and a point at its centre.
#[test]
fn predicate_conventions_match_their_documentation() {
    use robust::{insphere, orient3d, Coord3D};
    let p = |x, y, z| Coord3D { x, y, z };
    let (a, b, c, d) = (
        p(0.0_f64, 0.0, 0.0),
        p(1.0, 0.0, 0.0),
        p(0.0, 1.0, 0.0),
        p(0.0, 0.0, -1.0),
    );
    assert!(
        orient3d(a, b, c, d) > 0.0,
        "`d` below counterclockwise `a b c` is positive"
    );
    assert!(
        insphere(a, b, c, d, p(0.25, 0.25, -0.25)) > 0.0,
        "a point inside is positive"
    );
    assert!(
        insphere(a, b, c, d, p(9.0, 9.0, 9.0)) < 0.0,
        "a point outside is negative"
    );
    assert_eq!(
        insphere(a, b, c, d, p(1.0, 1.0, 0.0)),
        0.0,
        "a cospherical point is zero"
    );
}

#[test]
fn a_small_random_set_is_delaunay_and_covers_its_hull() {
    let points = random_points(60, 7);
    let hull_volume = brute_hull_volume(&points);
    let result = triangulate(points.clone());
    assert_eq!(
        result.points, points,
        "input coordinates reach the output unchanged"
    );
    assert_valid_small(&result);
    let tolerance = volume_tolerance(&result);
    let volume = total_volume(&result);
    assert!(
        (volume - hull_volume).abs() <= tolerance,
        "tetrahedra sum to {volume}, the hull is {hull_volume}, tolerance {tolerance}"
    );
}

#[test]
fn several_random_sets_cover_their_hulls() {
    for seed in 1..=4 {
        let points = random_points(40, seed * 91);
        let hull_volume = brute_hull_volume(&points);
        let result = triangulate(points);
        assert_valid_small(&result);
        let tolerance = volume_tolerance(&result);
        assert!((total_volume(&result) - hull_volume).abs() <= tolerance);
    }
}

#[test]
fn a_large_random_set_passes_every_check_and_a_spatial_sample_of_empty_spheres() {
    let points = random_points(30_000, 99);
    let result = triangulate(points);
    assert_positive(&result);
    assert_adjacency(&result);
    assert_euler(&result);
    assert_empty_spheres_sampled(&result, 200);
    // A uniform random set in the unit cube has a known number of tetrahedra per point near 6.77;
    // this also shows the result is not a thin subset of the input.
    let per_point = result.tets.len() as f64 / result.points.len() as f64;
    assert!(
        (6.0..=7.5).contains(&per_point),
        "{per_point} tetrahedra per point"
    );
}

#[test]
fn a_set_with_points_inside_a_tetrahedron_keeps_every_point() {
    let mut points = vec![
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0],
    ];
    points.extend_from_slice(&[[1.0, 1.0, 1.0], [0.5, 0.5, 1.0], [1.0, 0.5, 0.5]]);
    let result = triangulate(points);
    assert_valid_small(&result);
    let tolerance = volume_tolerance(&result);
    assert!((total_volume(&result) - 64.0 / 6.0).abs() <= tolerance);
}
