// Surface recovery across many rotations, with inputs identical on every system: the rotation
// angles come from a fixed integer sequence and every trigonometric value from `shapes::sin_cos`
// (plain `+ - * /`). The sweep covers the solids whose flat faces meet at real edges; faces cut
// into long needles and fans are refused at some rotations, and those shapes are in `cases.rs` at
// one fixed rotation only.

use crate::check::{assert_solid, mesh};
use crate::shapes::{hollow_box, rotated_by, subdivided_box, Surface};

const ROTATIONS: u64 = 24;

/// The rotation numbered `index`, three angles from an integer scramble of it, so a failure names
/// its exact input.
fn rotation(index: u64) -> [f64; 3] {
    let mut x = index
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678);
    let mut next = || {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        // A `u64` shifted to 53 bits fits the mantissa exactly.
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
            * std::f64::consts::TAU
    };
    [next(), next(), next()]
}

fn assert_every_rotation_recovers(name: &str, surface: &Surface) {
    for index in 0..ROTATIONS {
        let (points, triangles) = rotated_by(surface, rotation(index));
        let (solid, _) = mesh(&points, &triangles)
            .unwrap_or_else(|error| panic!("{name}, rotation {index}: {error}"));
        assert_solid(&solid, &points, &triangles);
    }
}

#[test]
fn boxes_recover_at_every_sampled_rotation() {
    assert_every_rotation_recovers("cube", &subdivided_box([0.0; 3], [1.0; 3], 1));
    assert_every_rotation_recovers("subdivided cube", &subdivided_box([0.0; 3], [2.0; 3], 4));
}

#[test]
fn boxes_with_voids_recover_at_every_sampled_rotation() {
    assert_every_rotation_recovers("box with void", &hollow_box(10.0, 2.0, 3, 2));
    let side = 100.0;
    let diagonal = (3.0_f64 * side * side).sqrt();
    assert_every_rotation_recovers("thin hollow box", &hollow_box(side, diagonal / 125.0, 6, 6));
}

#[test]
fn separate_bodies_recover_at_every_sampled_rotation() {
    let mut two = subdivided_box([0.0; 3], [1.0; 3], 2);
    two.add(&subdivided_box([3.0; 3], [4.0; 3], 2));
    assert_every_rotation_recovers("two bodies", &two);
    let mut nested = hollow_box(10.0, 1.0, 2, 2);
    nested.add(&subdivided_box([4.0; 3], [6.0; 3], 1));
    assert_every_rotation_recovers("body in a void", &nested);
}
