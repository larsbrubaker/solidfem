// Where a part's walls are too thin for the size asked for there (`Refiner::thin_walls`), on shapes
// whose every expected thickness, vertex count and box follows from the geometry. Built from
// axis-aligned cells: a plate far thinner than the size, one thin but not that thin, a thick block
// with one and with two thin fins, a web between two blocks seen only by wall fill, and a coarse
// plate with vertices only at its corners. Built otherwise: a plate whose sides are divided
// through it (`subdivided_box`), joined only across the wall; plates with a thick boss across them,
// short and long, cut as coarsely as they can be; and a plate fine on top and coarse below,
// joined only along its surface edges. Also: the warning ratio both ways, the mesh unchanged by
// it, a size defined only inside the solid, and nothing measured (`None`) against measured with
// nothing found (empty).

use solidfem_tetra::{
    RefineOptions, RefineReport, Refiner, SizeFn, SolidMesh, StepStatus, ThinWallSpot,
};

use crate::shapes::{subdivided_box, Surface};
use crate::support::{constant_size, fingerprint, options_with_layers, recovered};

/// The closed surface of the filled cells of the grid `coords` (cell `[i, j, k]` spans
/// `coords[0][i]..coords[0][i + 1]` and so on), a quad wherever a filled cell meets an empty one or
/// the outside. Every face is on the one grid, so the surface welds without a T-junction.
fn cells(coords: [&[f64]; 3], filled: impl Fn([usize; 3]) -> bool) -> Surface {
    let counts = coords.map(|c| c.len() - 1);
    let is_filled = |c: [i64; 3]| {
        (0..3).all(|a| c[a] >= 0 && (c[a] as usize) < counts[a])
            && filled(c.map(|value| value as usize))
    };
    let mut surface = Surface::default();
    for i in 0..counts[0] {
        for j in 0..counts[1] {
            for k in 0..counts[2] {
                let cell = [i, j, k];
                if !filled(cell) {
                    continue;
                }
                for axis in 0..3 {
                    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                    for outward in [false, true] {
                        let mut next = cell.map(|value| value as i64);
                        next[axis] += if outward { 1 } else { -1 };
                        if is_filled(next) {
                            continue;
                        }
                        let plane = coords[axis][cell[axis] + usize::from(outward)];
                        let corner = |du: usize, dv: usize| {
                            let mut p = [0.0; 3];
                            p[axis] = plane;
                            p[u] = coords[u][cell[u] + du];
                            p[v] = coords[v][cell[v] + dv];
                            p
                        };
                        // Counterclockwise from outside: u then v turns about +axis.
                        if outward {
                            surface.quad(corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1));
                        } else {
                            surface.quad(corner(0, 0), corner(0, 1), corner(1, 1), corner(1, 0));
                        }
                    }
                }
            }
        }
    }
    surface
}

/// `count` equal steps from `low` to `high`, both ends exact.
fn steps(low: f64, high: f64, count: usize) -> Vec<f64> {
    (0..=count)
        .map(|i| match i {
            0 => low,
            i if i == count => high,
            i => low + (high - low) * i as f64 / count as f64,
        })
        .collect()
}

/// A 50 by 50 plate `thickness` thick, its faces in 10 by 10 squares and each side square one
/// quad through the whole thickness, so every vertex is on the top or the bottom face.
fn plate(thickness: f64) -> Surface {
    let across = steps(0.0, 50.0, 10);
    cells([&across, &across, &[0.0, thickness]], |_| true)
}

/// A 40 by 40 block 10 thick with fins standing on it, each `[x low, x high]` across, 10 to 30 in
/// y and 10 to 30 in z, in two cells of 10 each way.
fn block_with_fins(xs: &[f64], fins: &[usize]) -> Surface {
    let ys = [0.0, 10.0, 20.0, 30.0, 40.0];
    let zs = [0.0, 5.0, 10.0, 20.0, 30.0];
    cells([xs, &ys, &zs], |[i, j, k]| {
        k < 2 || (fins.contains(&i) && (1..3).contains(&j) && (2..4).contains(&k))
    })
}

/// What a run gives: the mesh, the report and the spots, `None` when nothing was measured.
type Run = (SolidMesh, RefineReport, Option<Vec<ThinWallSpot>>);

/// Refines `surface` with `size` and `options` to the end.
fn run(surface: &Surface, size: SizeFn, options: RefineOptions) -> Run {
    let mut refiner = Refiner::new(recovered(surface), size, options).expect("refiner starts");
    assert_eq!(
        refiner.step(usize::MAX, &mut || false).expect("refines"),
        StepStatus::Done
    );
    let spots = refiner.thin_walls().map(<[ThinWallSpot]>::to_vec);
    (refiner.result().expect("result"), refiner.report(), spots)
}

fn with_ratio(ratio: f64) -> RefineOptions {
    RefineOptions {
        thin_wall_warning_ratio: ratio,
        ..options_with_layers(2)
    }
}

/// Refines `surface` at constant `size` with two layers and the warning `ratio`.
fn refine(
    surface: &Surface,
    size: f64,
    ratio: f64,
) -> (SolidMesh, RefineReport, Vec<ThinWallSpot>) {
    let (mesh, report, spots) = run(surface, constant_size(size), with_ratio(ratio));
    (mesh, report, spots.expect("the layer stage measured"))
}

#[test]
fn a_plate_far_thinner_than_the_size_is_one_spot_over_both_faces() {
    // 1 thick at size 200, a 200th of the size, well under the default 60th (3.33).
    let surface = plate(1.0);
    let (_, report, spots) = refine(&surface, 200.0, 60.0);
    assert_eq!(spots.len(), 1, "{spots:?}");
    let spot = spots[0];
    // Every vertex: a vertex inside a face looks straight across, 1; one on a rim edge looks
    // across at 45 degrees, its normal the mean of the face's and the side's, sqrt 2; a corner
    // at sqrt 3. All are under 3.33, and the two faces join through the rim and across the wall.
    assert_eq!(spot.vertices, surface.points.len());
    assert_eq!(spot.vertices, 2 * 11 * 11);
    // An interior vertex's normal is exactly along z and the distance is taken from the far
    // triangle's unit plane, so the thickness is the plate's to the last bit.
    assert_eq!(spot.thickness, 1.0);
    assert_eq!(spot.size, 200.0);
    assert!(spot.position[2] == 0.0 || spot.position[2] == 1.0);
    assert!((0..2).all(|k| (0.0..=50.0).contains(&spot.position[k])));
    let vertex = spot.vertex.expect("a vertex is thinnest") as usize;
    assert_eq!(surface.points[vertex], spot.position);
    assert_eq!(spot.low, [0.0, 0.0, 0.0]);
    assert_eq!(spot.high, [50.0, 50.0, 1.0]);
    assert_eq!(report.walls_too_thin_for_rounding, 0);
}

#[test]
fn the_faces_of_a_plate_whose_sides_are_divided_through_it_are_one_spot_across_the_wall() {
    // Each side square is cut into ten slices through the thickness, and a vertex inside a side
    // looks 50 across the plate, so it is not recorded and no recorded surface edge joins the top
    // to the bottom: the faces are one spot because each face's rays land on the other.
    let surface = subdivided_box([0.0; 3], [50.0, 50.0, 1.0], 10);
    let (_, _, spots) = refine(&surface, 200.0, 60.0);
    assert_eq!(spots.len(), 1, "{spots:?}");
    assert_eq!(spots[0].vertices, 2 * 11 * 11);
    assert_eq!(surface.points.len(), 2 * 11 * 11 + 9 * 40);
    assert_eq!((spots[0].low, spots[0].high), ([0.0; 3], [50.0, 50.0, 1.0]));
}

#[test]
fn a_plate_a_twentieth_of_the_size_is_a_thin_wall_but_no_spot_until_the_ratio_is_lowered() {
    let surface = plate(1.0);
    // At size 20 the default reports walls under 0.33; the plate is 1, layered as a thin wall.
    let (_, report, spots) = refine(&surface, 20.0, 60.0);
    assert!(spots.is_empty(), "{spots:?}");
    assert!(report.thin_wall_vertices > 0);
    // At a ratio of 10 the bound is 2, over the plate's every measure (1, sqrt 2, sqrt 3).
    let (_, _, spots) = refine(&surface, 20.0, 10.0);
    assert_eq!(spots.len(), 1, "{spots:?}");
    assert_eq!(spots[0].vertices, surface.points.len());
    assert_eq!(spots[0].thickness, 1.0);
}

#[test]
fn raising_the_ratio_past_the_plate_removes_its_spot() {
    // At 400 the bound is 0.5, under the plate's thinnest measure of 1.
    let (_, _, spots) = refine(&plate(1.0), 200.0, 400.0);
    assert!(spots.is_empty(), "{spots:?}");
}

#[test]
fn the_ratio_does_not_change_the_mesh() {
    let surface = plate(1.0);
    let (listed, listed_report, spots) = refine(&surface, 200.0, 60.0);
    let (unlisted, unlisted_report, none) = refine(&surface, 200.0, 400.0);
    assert_eq!((spots.len(), none.len()), (1, 0));
    assert_eq!(listed.points.len(), unlisted.points.len());
    assert_eq!(listed.tets, unlisted.tets);
    assert_eq!(fingerprint(&listed), fingerprint(&unlisted));
    assert_eq!(listed_report, unlisted_report);
}

/// Whether a column's measure `value` is `expected` to its rounding, on a part whose coordinates
/// are at most `scale`: a column starts at a barycentric sum of three corners (three products and
/// two sums, the weights summing to one within a unit in the last place), and its distance takes
/// two more roundings, so under 8 units in the last place of `scale`.
fn column_measure(value: f64, expected: f64, scale: f64) -> bool {
    (value - expected).abs() <= 8.0 * f64::EPSILON * scale
}

/// The fin of `block_with_fins` from `x0` to `x1` is `spot`, measured by its twelve vertices at
/// heights 20 and 30 (both faces, y 10, 20 and 30) and by the columns through its faces.
fn assert_fin(spot: &ThinWallSpot, x0: f64, x1: f64, scale: f64) {
    // Each of the twelve vertices looks across the fin (its thickness in a face, sqrt 2 times it on
    // a rim, sqrt 3 times at a top corner). A vertex where the fin meets the block, at height 10,
    // has a normal leaning out of the block's top and the fin's face, so its ray runs down into
    // the 10 thick block and is not recorded. The fin's faces are 10 by 10 squares, more than 8
    // walls wide, so wall fill lays columns through them, strictly inside each face: they reach
    // down towards the junction but never to it, so the spot holds nothing of the block.
    assert_eq!(spot.vertices, 12);
    let thickness = x1 - x0;
    assert!(column_measure(spot.thickness, thickness, scale), "{spot:?}");
    assert!(column_measure(spot.low[0], x0, scale), "{spot:?}");
    assert!(column_measure(spot.high[0], x1, scale), "{spot:?}");
    assert_eq!(
        [spot.low[1], spot.high[1], spot.high[2]],
        [10.0, 30.0, 30.0]
    );
    assert!(spot.low[2] > 10.0 && spot.low[2] < 20.0, "{spot:?}");
    let on_face = |x: f64| column_measure(x, x0, scale) || column_measure(x, x1, scale);
    assert!(on_face(spot.position[0]), "{spot:?}");
}

#[test]
fn a_thin_fin_on_a_thick_block_is_one_spot_on_the_fin() {
    let xs = [0.0, 10.0, 19.5, 20.5, 30.0, 40.0];
    let surface = block_with_fins(&xs, &[2]);
    let (_, _, spots) = refine(&surface, 200.0, 60.0);
    assert_eq!(spots.len(), 1, "{spots:?}");
    assert_fin(&spots[0], 19.5, 20.5, 40.0);
}

#[test]
fn two_thin_fins_far_apart_are_two_spots_thinnest_first() {
    let xs = [0.0, 10.0, 14.5, 15.5, 25.0, 34.75, 35.25, 45.0, 50.0];
    let surface = block_with_fins(&xs, &[2, 5]);
    let (_, _, spots) = refine(&surface, 200.0, 60.0);
    assert_eq!(spots.len(), 2, "{spots:?}");
    assert_fin(&spots[0], 34.75, 35.25, 50.0);
    assert_fin(&spots[1], 14.5, 15.5, 50.0);
}

#[test]
fn nothing_measured_is_none_and_measured_with_none_found_is_empty() {
    let surface = plate(1.0);
    let options = RefineOptions::default();
    assert_eq!(options.layers_through_walls, 0);
    let (_, _, spots) = run(&surface, constant_size(200.0), options);
    assert_eq!(spots, None, "no layer stage, so nothing was measured");
    let refiner = Refiner::new(recovered(&surface), constant_size(200.0), with_ratio(60.0))
        .expect("refiner starts");
    assert_eq!(refiner.thin_walls(), None, "the layer stage has not run");
    let (_, _, spots) = run(&surface, constant_size(20.0), with_ratio(60.0));
    assert_eq!(spots, Some(Vec::new()), "measured, and 1 is over 20 / 60");
}

#[test]
fn a_size_with_no_value_on_the_surface_refines_as_before_and_lists_the_wall() {
    // The plate's faces are at z 0 and 1; the size function has no value there, only inside. The
    // listing reads the size in the middle of each measured chord, at z 0.5, and refinement reads
    // it only at points strictly inside, so the run is the run of a constant 200.
    let surface = plate(1.0);
    let inside_only: SizeFn = Box::new(|p: [f64; 3]| {
        if p[2] == 0.0 || p[2] == 1.0 {
            f64::NAN
        } else {
            200.0
        }
    });
    let (mesh, report, spots) = run(&surface, inside_only, with_ratio(60.0));
    let (constant, constant_report, constant_spots) = refine(&surface, 200.0, 60.0);
    assert_eq!(fingerprint(&mesh), fingerprint(&constant));
    assert_eq!(report, constant_report);
    assert_eq!(spots, Some(constant_spots));
}

#[test]
fn a_web_whose_vertices_are_all_on_its_junctions_is_listed_by_its_columns() {
    // A 0.5 thick web between two 10 thick blocks: each web vertex stands where the web meets a
    // block, so its ray runs into the block. The web's faces are single 10 by 10 squares, more
    // than 8 walls wide, so wall fill measures the web square across.
    let (xs, ys, zs) = (
        [0.0, 10.0, 20.0, 30.0],
        [0.0, 10.0],
        [0.0, 4.75, 5.25, 10.0],
    );
    let surface = cells([&xs, &ys, &zs], |[i, _, k]| i != 1 || k == 1);
    let (_, report, spots) = refine(&surface, 200.0, 60.0);
    assert!(report.wall_fill_points > 0);
    assert_eq!(spots.len(), 1, "{spots:?}");
    let spot = spots[0];
    assert_eq!((spot.vertex, spot.vertices), (None, 0));
    assert!(column_measure(spot.thickness, 0.5, 30.0), "{spot:?}");
    let (low, high) = ([10.0, 0.0, 4.75], [20.0, 10.0, 5.25]);
    let inside = |k: usize, p: f64| p >= low[k] - 1e-12 && p <= high[k] + 1e-12;
    assert!((0..3).all(|k| inside(k, spot.low[k]) && inside(k, spot.high[k])));
}

#[test]
fn a_coarse_plate_with_vertices_only_at_its_corners_reports_its_true_thickness() {
    // 2.5 thick at size 200: the bound is 3.33. A corner looks across at sqrt 3 times the
    // thickness, 4.33, and is not recorded; the columns through the faces read 2.5.
    let surface = cells([&[0.0, 50.0], &[0.0, 50.0], &[0.0, 2.5]], |_| true);
    assert_eq!(surface.points.len(), 8);
    let (_, _, spots) = refine(&surface, 200.0, 60.0);
    assert_eq!(spots.len(), 1, "{spots:?}");
    let spot = spots[0];
    assert_eq!((spot.vertex, spot.vertices), (None, 0));
    assert!(column_measure(spot.thickness, 2.5, 50.0), "{spot:?}");
}

/// A `length` by 50 plate 0.5 thick with a boss 11 high across it from x `low` to `high`, cut as
/// coarsely as it can be: the bottom two triangles, each end profile six, every other face a
/// single quad. The bottom's long diagonal and the end profiles' triangles reach from one thin
/// side under the boss to the other.
fn bossed_plate(length: f64, low: f64, high: f64) -> Surface {
    let profile = [
        [0.0, 0.0],
        [length, 0.0],
        [length, 0.5],
        [high, 0.5],
        [high, 11.0],
        [low, 11.0],
        [low, 0.5],
        [0.0, 0.5],
    ];
    let at = |k: usize, y: f64| [profile[k][0], y, profile[k][1]];
    let mut surface = Surface::default();
    // The profile is counterclockwise in x and z, so each edge extruded this way faces outward.
    for k in 0..8 {
        let next = (k + 1) % 8;
        surface.quad(at(k, 0.0), at(k, 50.0), at(next, 50.0), at(next, 0.0));
    }
    let ends = [
        [0, 1, 3],
        [1, 2, 3],
        [0, 3, 6],
        [0, 6, 7],
        [3, 4, 5],
        [3, 5, 6],
    ];
    for [a, b, c] in ends {
        surface.triangle(at(a, 0.0), at(b, 0.0), at(c, 0.0));
        surface.triangle(at(a, 50.0), at(c, 50.0), at(b, 50.0));
    }
    surface
}

/// The bossed plate is two spots at size 60 (a bound of 1: the plate is 0.5, the boss 11), one on
/// either side, neither box reaching across the boss.
fn assert_split_at_the_boss(length: f64, low: f64, high: f64) {
    let surface = bossed_plate(length, low, high);
    assert_eq!(surface.triangles.len(), 2 * 8 + 2 * 6);
    let (_, _, spots) = refine(&surface, 60.0, 60.0);
    assert_eq!(
        spots.len(),
        2,
        "boss {low} to {high} on {length}: {spots:?}"
    );
    let (left, right) = if spots[0].low[0] < spots[1].low[0] {
        (spots[0], spots[1])
    } else {
        (spots[1], spots[0])
    };
    assert!(left.high[0] <= low, "{left:?}");
    assert!(right.low[0] >= high, "{right:?}");
}

#[test]
fn a_thin_plate_either_side_of_a_thick_boss_is_two_spots() {
    assert_split_at_the_boss(300.0, 100.0, 200.0);
}

#[test]
fn a_long_coarse_plate_either_side_of_a_thick_boss_is_two_spots() {
    // On a 3000 long face a column's grid step is about 188, more than a narrow boss: one step
    // links neighbouring columns, two would reach over the unrecorded columns under the boss.
    for (low, high) in [
        (1350.0, 1650.0),
        (1400.0, 1600.0),
        (1450.0, 1550.0),
        (600.0, 900.0),
    ] {
        assert_split_at_the_boss(3000.0, low, high);
    }
}

/// One side of `plate_fine_on_top` seen from outside: its bottom edge from `b0` to `b1`, left to
/// right, and its top rim vertices above it in the same order, a fan from the bottom corners.
fn fan_side(surface: &mut Surface, b0: [f64; 3], b1: [f64; 3], tops: &[[f64; 3]]) {
    let last = tops.len() - 1;
    surface.triangle(b0, b1, tops[last]);
    for i in 0..last {
        surface.triangle(b0, tops[i + 1], tops[i]);
    }
}

/// A 50 by 50 plate 1 thick whose top is cut into 5 wide squares and whose bottom is two
/// triangles, each side a fan from the bottom corners to the top's rim vertices.
fn plate_fine_on_top() -> Surface {
    let across = steps(0.0, 50.0, 10);
    let mut surface = Surface::default();
    for i in 0..10 {
        for j in 0..10 {
            let (x0, x1, y0, y1) = (across[i], across[i + 1], across[j], across[j + 1]);
            surface.quad([x0, y0, 1.0], [x1, y0, 1.0], [x1, y1, 1.0], [x0, y1, 1.0]);
        }
    }
    let (c00, c10, c11, c01) = (
        [0.0, 0.0, 0.0],
        [50.0, 0.0, 0.0],
        [50.0, 50.0, 0.0],
        [0.0, 50.0, 0.0],
    );
    surface.quad(c00, c01, c11, c10);
    let back: Vec<f64> = across.iter().rev().copied().collect();
    let side = |f: &dyn Fn(f64) -> [f64; 3], values: &[f64]| -> Vec<[f64; 3]> {
        values.iter().map(|&t| f(t)).collect()
    };
    fan_side(&mut surface, c00, c10, &side(&|t| [t, 0.0, 1.0], &across));
    fan_side(&mut surface, c10, c11, &side(&|t| [50.0, t, 1.0], &across));
    fan_side(&mut surface, c11, c01, &side(&|t| [t, 50.0, 1.0], &back));
    fan_side(&mut surface, c01, c00, &side(&|t| [0.0, t, 1.0], &back));
    surface
}

#[test]
fn a_plate_joined_only_along_its_surface_edges_is_one_spot() {
    // The top's vertices look down onto the two bottom triangles, whose corners are the plate's
    // corners, mostly far more than 8 walls away: only the top's 5 and 7.07 long edges, under 8
    // walls, join them. The thin-wall ratio is set so low that no face is a wall for wall fill,
    // so no column joins them either.
    let surface = plate_fine_on_top();
    let options = RefineOptions {
        thin_wall_ratio: 0.01,
        ..with_ratio(60.0)
    };
    let (_, report, spots) = run(&surface, constant_size(200.0), options);
    let spots = spots.expect("measured");
    assert_eq!(report.wall_fill_points, 0);
    assert_eq!(spots.len(), 1, "{spots:?}");
    // Every vertex: inside the top 1, on its rim sqrt 2, at a corner sqrt 3, all under 3.33.
    assert_eq!(spots[0].vertices, surface.points.len());
    assert_eq!(surface.points.len(), 11 * 11 + 4);
}
