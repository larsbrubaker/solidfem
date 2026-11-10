// A surface edge flips leave stuck, recovered by replacing the region its segment passes through
// with a star from one end (`reconnect.rs` in the crate, where the star's safety checks are
// fuzzed directly). The surface is built in code: a thin-walled tube, slightly twisted so its
// skins are long nearly flat triangles close to each other, the shape in which corpus parts left
// edges stuck behind blocked flips. Those corpus parts stay outside this repository; this case
// is the end-to-end test that stars happen and leave a valid solid. A star is not the only way
// through here: without it the same surface also recovers, by adding more points, so this test
// does not show the star is needed.

use crate::check::{assert_solid, mesh};
use crate::shapes::twisted_tube;
use std::f64::consts::TAU;

#[test]
fn a_surface_edge_stuck_in_a_twisted_thin_wall_is_recovered_by_a_star() {
    let surface = twisted_tube(12, 4, 10.0, 0.5, 9.0, TAU * 0.05 / 12.0);
    let (points, triangles) = (surface.points.clone(), surface.triangles.clone());
    let (solid, report) = mesh(&points, &triangles).unwrap_or_else(|error| panic!("{error}"));
    assert_solid(&solid, &points, &triangles);
    assert_eq!(report.edges_failed, 0, "every surface edge is recovered");
    assert_eq!(
        report.faces_failed, 0,
        "every surface triangle is recovered"
    );
    assert!(
        report.star_reconnections > 0,
        "a stuck edge is recovered by a star, which is what this case exercises"
    );
}
