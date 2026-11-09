// Through-thickness layers measured on the output: a walk along columns across the wall counts
// the tetrahedra traversed, and a plate's added points are measured against the midplane.

use crate::shapes::{hollow_box, plate_with_holes};
use crate::support::{column_walk, constant_size, options_with_layers, recovered, run_refine};

const THICK: f64 = 0.8;

/// Columns placed where no mesh edge passes: offsets from a low-discrepancy sequence.
fn sample_columns(low: f64, high: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| low + (high - low) * (0.3117 + i as f64 * 0.61803).fract())
        .collect()
}

#[test]
fn a_plate_has_two_tetrahedra_through_every_column_and_a_layer_of_points_at_the_midplane() {
    let plate = plate_with_holes(20.0, 20.0, THICK, 10, 10, 1, &[]);
    let (refined, report) = run_refine(
        recovered(&plate),
        constant_size(5.0),
        options_with_layers(2),
    )
    .expect("refines");
    assert!(report.layer_points > 0, "the plate gets layer points");
    let mut largest_span = 0.0_f64;
    let (xs, ys) = (
        sample_columns(0.31, 19.77, 23),
        sample_columns(0.17, 19.53, 23),
    );
    for &x in &xs {
        for &y in &ys {
            let seen = column_walk(&refined, [x, y, -1.0], [x, y, THICK + 1.0]);
            assert!(
                seen.len() >= 2,
                "column ({x}, {y}) crossed only {} tetrahedra",
                seen.len()
            );
            for &t in &seen {
                let zs = refined.tets[t].map(|v| refined.points[v as usize][2]);
                let span = zs.iter().fold(f64::MIN, |a, &z| a.max(z))
                    - zs.iter().fold(f64::MAX, |a, &z| a.min(z));
                largest_span = largest_span.max(span);
            }
        }
    }
    eprintln!("plate: largest tetrahedron span along a sampled column {largest_span}");
    // A rim is a single surface quad with no point through it, so only columns clear of it can
    // measure the layers: two edge lengths away from the rim no tetrahedron crosses both skins.
    let (edge, margin) = (2.0, 4.0);
    let clear = |value: f64| value > margin && value < 20.0 - margin;
    for &x in xs.iter().filter(|&&x| clear(x)) {
        for &y in ys.iter().filter(|&&y| clear(y)) {
            for t in column_walk(&refined, [x, y, -1.0], [x, y, THICK + 1.0]) {
                let zs = refined.tets[t].map(|v| refined.points[v as usize][2]);
                let span = zs.iter().fold(f64::MIN, |a, &z| a.max(z))
                    - zs.iter().fold(f64::MAX, |a, &z| a.min(z));
                assert!(
                    span <= 0.5 * THICK * (1.0 + 1e-9),
                    "column ({x}, {y}): tetrahedron {t} spans {span} of the {THICK} wall,                      crossing both skins ({edge} mm edges)"
                );
            }
        }
    }
    let deviation = refined
        .points
        .iter()
        .skip(plate.points.len())
        .map(|p| (p[2] - THICK / 2.0).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("plate: largest added point deviation from the midplane {deviation}");
    assert!(
        deviation <= 0.1 * THICK / 2.0,
        "layer points left the midplane by {deviation}"
    );
}

#[test]
fn a_thin_hollow_box_has_two_tetrahedra_through_every_column_of_its_floor() {
    let wall = 1.0;
    let surface = hollow_box(30.0, wall, 6, 5);
    let (refined, _) = run_refine(
        recovered(&surface),
        constant_size(8.0),
        options_with_layers(2),
    )
    .expect("refines");
    let (xs, ys) = (sample_columns(1.2, 28.8, 17), sample_columns(1.3, 28.7, 17));
    for &x in &xs {
        for &y in &ys {
            let seen = column_walk(&refined, [x, y, -0.5], [x, y, 15.0]);
            assert!(
                seen.len() >= 2,
                "column ({x}, {y}) crossed only {} tetrahedra",
                seen.len()
            );
        }
    }
}
