// Adding points to a finished triangulation: refinement inserts points this way, so each must
// land exactly and leave a valid Delaunay tetrahedralization, inside the hull, outside it, or in
// the awkward positions on a face or a hull plane.

use crate::support::{assert_valid_small, total_volume, volume_tolerance, Rng};
use solidfem_tetra::{StepStatus, TetraError, Triangulator};

fn finished(points: Vec<[f64; 3]>) -> Triangulator {
    let mut triangulator = Triangulator::new(points).expect("the input is valid");
    assert_eq!(
        triangulator.step(usize::MAX, &mut || false),
        Ok(StepStatus::Done)
    );
    triangulator
}

fn unit_cube_corners() -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                points.push([f64::from(i), f64::from(j), f64::from(k)]);
            }
        }
    }
    points
}

#[test]
fn points_added_one_at_a_time_keep_the_delaunay_property() {
    let mut rng = Rng(17);
    let mut triangulator = finished((0..30).map(|_| rng.point()).collect());
    for _ in 0..40 {
        let point = rng.point();
        let index = triangulator
            .insert_point(point)
            .expect("a new point inserts");
        assert_eq!(
            triangulator.result().expect("done").points[index as usize],
            point
        );
    }
    assert_valid_small(&triangulator.result().expect("done"));
}

#[test]
fn a_point_on_a_cube_face_a_hull_edge_and_a_hull_vertex_plane_stays_valid() {
    let mut triangulator = finished(unit_cube_corners());
    for point in [
        [0.5, 0.5, 0.0],  // in a hull face
        [0.5, 0.0, 0.0],  // on a hull edge
        [0.5, 0.5, 0.5],  // the centre, cospherical corners around it
        [0.25, 0.5, 1.0], // on another hull face
    ] {
        triangulator.insert_point(point).expect("the point is new");
        assert_valid_small(&triangulator.result().expect("done"));
    }
    let result = triangulator.result().expect("done");
    assert!((total_volume(&result) - 1.0).abs() <= volume_tolerance(&result));
}

#[test]
fn a_point_outside_the_hull_grows_it() {
    let mut triangulator = finished(unit_cube_corners());
    // Directly above the cube's top face, so the new hull adds a pyramid of height 1 on a unit
    // square.
    triangulator
        .insert_point([0.5, 0.5, 2.0])
        .expect("the point is new");
    let result = triangulator.result().expect("done");
    assert_valid_small(&result);
    let expected = 1.0 + 1.0 / 3.0;
    assert!((total_volume(&result) - expected).abs() <= volume_tolerance(&result));
}

#[test]
fn a_point_in_the_plane_of_a_hull_face_but_outside_it_extends_the_face() {
    let mut triangulator = finished(unit_cube_corners());
    triangulator
        .insert_point([2.0, 0.5, 0.0])
        .expect("the point is new");
    let result = triangulator.result().expect("done");
    assert_valid_small(&result);
    // The hull is the cube joined to the point, a pyramid over the x = 1 face of height 1.
    let expected = 1.0 + 1.0 / 3.0;
    assert!((total_volume(&result) - expected).abs() <= volume_tolerance(&result));
}

#[test]
fn a_duplicate_point_is_refused_and_changes_nothing() {
    let mut triangulator = finished(unit_cube_corners());
    let before = triangulator.result().expect("done");
    assert!(matches!(
        triangulator.insert_point([1.0, 1.0, 0.0]),
        Err(TetraError::DuplicatePoint { .. })
    ));
    assert_eq!(triangulator.result().expect("done"), before);
    // Still usable afterwards.
    triangulator
        .insert_point([0.5, 0.25, 0.25])
        .expect("the point is new");
    assert_valid_small(&triangulator.result().expect("done"));
}

#[test]
fn a_point_cannot_be_added_before_the_set_is_finished() {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 2.0, 2.0],
    ];
    let mut triangulator = Triangulator::new(points).expect("the set spans a volume");
    assert!(matches!(
        triangulator.insert_point([0.1, 0.1, 0.1]),
        Err(TetraError::StillInserting { .. })
    ));
}

#[test]
fn an_invalid_point_is_refused() {
    let mut triangulator = finished(unit_cube_corners());
    assert!(matches!(
        triangulator.insert_point([f64::INFINITY, 0.0, 0.0]),
        Err(TetraError::NonFiniteCoordinate { .. })
    ));
}

#[test]
fn the_tetrahedra_of_cospherical_points_do_not_depend_on_how_the_points_arrive() {
    // A lattice is full of exact cospherical ties. The tie rule ranks points by index, so the
    // same points with the same indices must give the same tetrahedra whether the last few
    // arrive in the initial set or one at a time afterwards, with no other point order.
    let mut points = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                points.push([f64::from(i), f64::from(j), f64::from(k)]);
            }
        }
    }
    let sorted_tets = |result: solidfem_tetra::Tetrahedralization| {
        let mut tets: Vec<[u32; 4]> = result
            .tets
            .iter()
            .map(|tet| {
                let mut sorted = *tet;
                sorted.sort_unstable();
                sorted
            })
            .collect();
        tets.sort_unstable();
        tets
    };
    let all_at_once = sorted_tets(finished(points.clone()).result().expect("done"));
    for held_back in [1, 5, 13] {
        let split = points.len() - held_back;
        let mut triangulator = finished(points[..split].to_vec());
        for &point in &points[split..] {
            triangulator.insert_point(point).expect("the point is new");
        }
        assert_eq!(
            sorted_tets(triangulator.result().expect("done")),
            all_at_once,
            "holding back the last {held_back} points changed the tetrahedra"
        );
    }
}
