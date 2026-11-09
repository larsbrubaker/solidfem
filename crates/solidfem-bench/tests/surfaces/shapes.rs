//! Each generated part is the part it says it is: a closed mesh whose bounding box is the one its
//! definition gives. A benchmark on the wrong dimensions would compare a correct solver against
//! the wrong reference, so each box is computed by hand from the part's dimensions and not read
//! from a run. Extrudes snap each outline point to a 0.001 grid and clean up the integer outline,
//! and vertices are `f32`, so the tolerance is those three together.

use solidfem_bench::surfaces::all_parts;
use solidmesh_mesh::{CancelToken, Mesh};

/// A box, within 0.0005 for the extrude's snap of each outline point (0.001 grid, nearest half to
/// even), 0.0015 for the clean-up of the snapped outline, which drops a vertex about 1.4 grid
/// cells off the chord of its neighbours (the gear's apex at y = 44 is such a vertex, so the
/// built tooth tops out at 43.999), and 2e-6 of the size for the `f32` rounding of a vertex. A
/// real change of a dimension is far larger.
fn assert_box(name: &str, mesh: &Mesh, lo: [f64; 3], hi: [f64; 3]) {
    let mut actual_lo = [f64::MAX; 3];
    let mut actual_hi = [f64::MIN; 3];
    for p in &mesh.vertices {
        let p = [f64::from(p.x), f64::from(p.y), f64::from(p.z)];
        for i in 0..3 {
            actual_lo[i] = actual_lo[i].min(p[i]);
            actual_hi[i] = actual_hi[i].max(p[i]);
        }
    }
    let scale = lo.iter().chain(&hi).fold(1.0_f64, |m, v| m.max(v.abs()));
    let tolerance = 0.0005 + 0.0015 + 2e-6 * scale;
    for i in 0..3 {
        assert!(
            (actual_lo[i] - lo[i]).abs() <= tolerance && (actual_hi[i] - hi[i]).abs() <= tolerance,
            "{name}: axis {i} spans {}..{}, expected {}..{}",
            actual_lo[i],
            actual_hi[i],
            lo[i],
            hi[i]
        );
    }
}

fn built(name: &str) -> Mesh {
    all_parts()
        .into_iter()
        .find(|part| part.name == name)
        .unwrap_or_else(|| panic!("no part {name}"))
        .mesh(&CancelToken::new())
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn the_extruded_parts_have_the_dimensions_they_state() {
    assert_box(
        "strip",
        &built("strip"),
        [0.0, 0.0, 0.0],
        [100.0, 10.0, 1.0],
    );
    assert_box(
        "plate",
        &built("plate"),
        [0.0, 0.0, 0.0],
        [300.0, 100.0, 2.0],
    );
    assert_box(
        "lsharp",
        &built("lsharp"),
        [0.0, 0.0, 0.0],
        [100.0, 100.0, 5.0],
    );
    assert_box(
        "lfillet",
        &built("lfillet"),
        [0.0, 0.0, 0.0],
        [100.0, 100.0, 5.0],
    );
    assert_box("snap", &built("snap"), [0.0, 0.0, 0.0], [20.0, 5.0, 2.0]);
    assert_box(
        "hollow",
        &built("hollow"),
        [0.0, 0.0, 0.0],
        [40.0, 40.0, 20.0],
    );
}

/// The membrane's outer ellipse has semi-axes 3250 and 2750, in the first quadrant.
#[test]
fn the_nafems_membranes_are_quarter_ellipses_of_the_stated_thickness() {
    assert_box(
        "le1",
        &built("le1"),
        [0.0, 0.0, 0.0],
        [3250.0, 2750.0, 100.0],
    );
    // LE10 is centred on the mid-plane, half its 600 either side.
    assert_box(
        "le10",
        &built("le10"),
        [0.0, 0.0, -300.0],
        [3250.0, 2750.0, 300.0],
    );
}

/// The hook's ring sector reaches radius 40 on the left and bottom (-x at 180 degrees, -y at 270,
/// both inside the 90 to 300 degree sweep), and its right side is the 300 degree end point at
/// x = 40 cos 300 = 20 or the shank at x = 10, so 20; the shank tops out at y = 110. The outer
/// arc is 180 chords over 210 degrees and neither 180 nor 270 degrees is a vertex, so the box is
/// short of 40 by at most a chord's sagitta, 40 (1 - cos(0.5 x 210 / 180 degrees)), under 0.0021.
#[test]
fn the_hook_reaches_its_ring_radius_and_shank_top() {
    let mesh = built("hook");
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in &mesh.vertices {
        let p = [f64::from(p.x), f64::from(p.y), f64::from(p.z)];
        for i in 0..3 {
            lo[i] = lo[i].min(p[i]);
            hi[i] = hi[i].max(p[i]);
        }
    }
    let sagitta = 0.0021;
    assert!(
        lo[0] <= -40.0 + sagitta && lo[0] >= -40.0 - 1e-3,
        "min x {}",
        lo[0]
    );
    assert!(
        lo[1] <= -40.0 + sagitta && lo[1] >= -40.0 - 1e-3,
        "min y {}",
        lo[1]
    );
    for (axis, (actual, expected)) in [(lo[2], 0.0), (hi[0], 20.0), (hi[1], 110.0), (hi[2], 10.0)]
        .iter()
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < 7e-4,
            "bound {axis}: {actual} expected {expected}"
        );
    }
}

/// The tip of a 20-tooth, module 4 gear is at the pitch radius 40 plus one module, 44, reached by
/// the tip vertex on the y axis. The root radius is 35 and the sector half-angle 1.25 pi / 20: the
/// widest points are the root corners, 35 sin of it either side. The rim is 11 inside the root,
/// radius 24, so the lowest vertices are its end points at 24 cos of the half-angle.
#[test]
fn the_gear_tooth_has_the_stated_addendum_and_root_sector() {
    let theta = 1.25 * std::f64::consts::PI / 20.0;
    let half_width = 35.0 * theta.sin();
    let bottom = 24.0 * theta.cos();
    assert_box(
        "gear",
        &built("gear"),
        [-half_width, bottom, 0.0],
        [half_width, 44.0, 20.0],
    );
}

/// The ring has outer radius 10 and height 8. Which axis it revolves about is the operation's
/// business, so this reads the axial span from whichever coordinate has it.
#[test]
fn the_ring_has_its_stated_radius_and_height() {
    let mesh = built("ring");
    let span = |pick: fn(&Vec<f64>) -> f64| {
        let values: Vec<f64> = mesh.vertices.iter().map(|p| f64::from(p.x)).collect();
        pick(&values)
    };
    // Outer radius 10: x reaches both -10 and +10, whatever the axis, since the revolve is whole.
    let max_x = span(|v| v.iter().copied().fold(f64::MIN, f64::max));
    let min_x = span(|v| v.iter().copied().fold(f64::MAX, f64::min));
    assert!((max_x - 10.0).abs() < 1e-3, "max x {max_x}");
    assert!((min_x + 10.0).abs() < 1e-3, "min x {min_x}");
    // The 8 tall axial extent is the only one of y, z that is 8 across (the other is 20).
    let extent = |values: Vec<f64>| {
        values.iter().copied().fold(f64::MIN, f64::max)
            - values.iter().copied().fold(f64::MAX, f64::min)
    };
    let y = extent(mesh.vertices.iter().map(|p| f64::from(p.y)).collect());
    let z = extent(mesh.vertices.iter().map(|p| f64::from(p.z)).collect());
    assert!(
        (y - 8.0).abs() < 1e-3 || (z - 8.0).abs() < 1e-3,
        "axial extent: y {y}, z {z}"
    );
}
