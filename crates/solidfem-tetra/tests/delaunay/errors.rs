// The input errors a caller can act on: each named, at the right point in the run, and never a
// partial result.

use solidfem_tetra::{StepStatus, TetraError, Triangulator};

#[test]
fn too_few_points_is_a_named_error() {
    let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let error = Triangulator::new(points).err();
    assert_eq!(error, Some(TetraError::TooFewPoints { count: 3 }));
}

#[test]
fn purely_coplanar_input_is_a_named_error() {
    let mut points = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            points.push([f64::from(i), f64::from(j), 2.5]);
        }
    }
    assert_eq!(
        Triangulator::new(points).err(),
        Some(TetraError::AllPointsCoplanar)
    );
}

#[test]
fn an_oblique_plane_with_irrational_looking_coordinates_is_still_coplanar() {
    // z = x + 2y on exact binary fractions: coplanar exactly though no axis is constant.
    let mut points = Vec::new();
    for i in 0..30 {
        for j in 0..30 {
            let (x, y) = (f64::from(i) * 0.125, f64::from(j) * 0.0625);
            points.push([x, y, x + 2.0 * y]);
        }
    }
    assert_eq!(
        Triangulator::new(points).err(),
        Some(TetraError::AllPointsCoplanar)
    );
}

#[test]
fn all_collinear_input_is_a_named_error() {
    let points: Vec<[f64; 3]> = (0..20)
        .map(|i| [f64::from(i), 2.0 * f64::from(i), 0.0])
        .collect();
    assert_eq!(
        Triangulator::new(points).err(),
        Some(TetraError::AllPointsCollinear)
    );
}

#[test]
fn all_coincident_input_is_a_named_error() {
    let points = vec![[1.0, 2.0, 3.0]; 8];
    assert_eq!(
        Triangulator::new(points).err(),
        Some(TetraError::AllPointsCoincident)
    );
}

#[test]
fn a_nan_coordinate_is_a_named_error() {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, f64::NAN, 0.0],
        [0.0, 0.0, 1.0],
    ];
    assert!(matches!(
        Triangulator::new(points).err(),
        Some(TetraError::NonFiniteCoordinate { index: 2, .. })
    ));
}

#[test]
fn an_extreme_coordinate_is_refused_before_the_predicates_lose_exactness() {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1e80, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    assert!(matches!(
        Triangulator::new(points).err(),
        Some(TetraError::CoordinateOutOfRange { index: 1, .. })
    ));
}

#[test]
fn a_duplicate_point_is_a_named_error_naming_both_points() {
    let mut points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    points.push([0.0, 1.0, 0.0]);
    let mut triangulator = Triangulator::new(points).expect("the set spans a volume");
    let error = triangulator
        .step(usize::MAX, &mut || false)
        .expect_err("a duplicate point is refused");
    match error {
        TetraError::DuplicatePoint { index, existing } => {
            // The two points are 2 and 5, in either order: which is met first depends on the
            // insertion order.
            assert_eq!([index.min(existing), index.max(existing)], [2, 5]);
        }
        other => panic!("expected a duplicate point error, got {other:?}"),
    }
}

#[test]
fn negative_zero_and_zero_are_the_same_point() {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-0.0, 0.0, 0.0],
    ];
    let mut triangulator = Triangulator::new(points).expect("the set spans a volume");
    let status = triangulator.step(usize::MAX, &mut || false);
    assert!(
        matches!(status, Err(TetraError::DuplicatePoint { .. })),
        "{status:?}"
    );
}

#[test]
fn the_result_is_refused_until_every_point_is_inserted() {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 1.0, 2.0],
    ];
    let mut triangulator = Triangulator::new(points).expect("the set spans a volume");
    assert!(matches!(
        triangulator.result(),
        Err(TetraError::NotFinished { .. })
    ));
    // Two points remain after the first tetrahedron, so one step of one leaves the run unfinished.
    assert_eq!(triangulator.step(1, &mut || false), Ok(StepStatus::Running));
    assert!(matches!(
        triangulator.result(),
        Err(TetraError::NotFinished { .. })
    ));
    assert_eq!(triangulator.step(1, &mut || false), Ok(StepStatus::Done));
    assert!(triangulator.result().is_ok());
}
