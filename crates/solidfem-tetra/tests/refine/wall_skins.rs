// The through-wall guarantee measured from the outside on the shapes that used to escape it: a
// plate rotated so its faces are only nearly coplanar, a box whose faces have vertices only on
// their rims, and a tube with only end rings. Which skin a vertex is on, and whether a sample is
// inside a tetrahedron, are decided here from the input surface and exact orientation tests.

use robust::{orient3d, Coord3D};
use solidfem_tetra::{RefineOptions, SolidMesh, HULL};

use crate::shapes::{plate_with_holes, rotated_by, sin_cos, subdivided_box, Surface};
use crate::support::{
    assert_contracts, column_walk, constant_size, options_with_layers, recovered, run_refine,
};
use crate::through::unaligned_plate;

fn coord(p: [f64; 3]) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

/// Whether `sample` is inside or on the tetrahedron, by exact orientation against each face.
fn tet_holds(mesh: &SolidMesh, tet: &[u32; 4], sample: [f64; 3]) -> bool {
    let p = tet.map(|v| mesh.points[v as usize]);
    (0..4).all(|i| {
        let others: Vec<[f64; 3]> = (0..4).filter(|&k| k != i).map(|k| p[k]).collect();
        let (a, b, c) = (coord(others[0]), coord(others[1]), coord(others[2]));
        let (toward_vertex, toward_sample) = (
            orient3d(a, b, c, coord(p[i])),
            orient3d(a, b, c, coord(sample)),
        );
        toward_sample == 0.0 || (toward_sample > 0.0) == (toward_vertex > 0.0)
    })
}

/// The samples that fall in a tetrahedron with a surface vertex on each skin, by a skin rule
/// taking a vertex of the input surface.
fn samples_in_tets_joining_skins(
    mesh: &SolidMesh,
    surface_points: usize,
    skin: &dyn Fn(usize) -> Option<u8>,
    samples: &[[f64; 3]],
) -> Vec<[f64; 3]> {
    let joining: Vec<&[u32; 4]> = mesh
        .tets
        .iter()
        .filter(|tet| {
            let seen: Vec<u8> = tet
                .iter()
                .filter(|&&v| (v as usize) < surface_points)
                .filter_map(|&v| skin(v as usize))
                .collect();
            seen.contains(&0) && seen.contains(&1)
        })
        .collect();
    samples
        .iter()
        .copied()
        .filter(|&s| joining.iter().any(|tet| tet_holds(mesh, tet, s)))
        .collect()
}

#[test]
fn rotated_plates_have_every_through_edge_split_and_no_element_joining_their_skins() {
    // Rotations computed through `sin_cos`, so the near-flat faces are the same on every system.
    let rotations = [
        [0.37, 0.61, 0.0],
        [1.1, 0.23, 0.47],
        [0.71, 1.13, 0.37],
        [2.3, 0.9, 1.7],
    ];
    let (thick, length) = (0.8, 20.0);
    for rotation in rotations {
        for cells in [10usize, 14, 20] {
            let edge = length / cells as f64;
            let flat = unaligned_plate(length, length, thick, cells);
            let (points, triangles) = rotated_by(&flat, rotation);
            let surface = Surface {
                points,
                triangles,
                ..Surface::default()
            };
            let unrefined = recovered(&surface);
            let (refined, report) = run_refine(
                unrefined.clone(),
                constant_size(1.5 * edge),
                options_with_layers(2),
            )
            .unwrap_or_else(|error| {
                panic!("rotation {rotation:?} at {cells} cells: refinement failed: {error}")
            });
            assert_contracts(&refined, &unrefined, &surface);
            assert_eq!(
                report.through_edges_refused, 0,
                "rotation {rotation:?} at {cells} cells: a through edge was left"
            );
            // Skins read from the plate before it was rotated, by the same vertex order.
            let skin = |v: usize| {
                if flat.points[v][2] == 0.0 {
                    Some(0u8)
                } else if flat.points[v][2] == thick {
                    Some(1u8)
                } else {
                    None
                }
            };
            // Two cells from the rim, the margin the plate's own rim bridging tets stay within.
            let margin = 2.0 * edge;
            let clear = |v: usize| {
                let p = flat.points[v];
                p[0] > margin && p[0] < length - margin && p[1] > margin && p[1] < length - margin
            };
            let surface_points = flat.points.len();
            let spanning = refined
                .tets
                .iter()
                .filter(|tet| {
                    let vertices = || {
                        tet.iter()
                            .map(|&v| v as usize)
                            .filter(|&v| v < surface_points)
                    };
                    vertices().any(|v| skin(v) == Some(0) && clear(v))
                        && vertices().any(|v| skin(v) == Some(1) && clear(v))
                })
                .count();
            assert_eq!(
                spanning, 0,
                "rotation {rotation:?} at {cells} cells: tetrahedra with a vertex on each skin \
                 away from the rim"
            );
        }
    }
}

/// `count` evenly spread values strictly between `low` and `high`.
fn grid(low: f64, high: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| low + (high - low) * (i as f64 + 0.5) / count as f64)
        .collect()
}

#[test]
fn a_plain_eight_vertex_box_has_no_mid_plane_sample_in_a_tetrahedron_joining_its_skins() {
    let thick = 1.0;
    let surface = subdivided_box([0.0; 3], [50.0, 50.0, thick], 1);
    let unrefined = recovered(&surface);
    let (refined, report) = run_refine(
        unrefined.clone(),
        constant_size(200.0),
        options_with_layers(2),
    )
    .expect("refines");
    assert_contracts(&refined, &unrefined, &surface);
    assert_eq!(report.through_edges_refused, 0, "a through edge was left");
    let skin = |v: usize| {
        let z = surface.points[v][2];
        if z == 0.0 {
            Some(0u8)
        } else if z == thick {
            Some(1u8)
        } else {
            None
        }
    };
    // The rim band `wall_fill.rs` leaves is within two fill steps of the rim: the longest edge, the
    // seventy unit face diagonal, over the sixteen steps a face is laid at, twice over.
    let margin = 9.0;
    let cells = grid(margin, 50.0 - margin, 15);
    let samples: Vec<[f64; 3]> = cells
        .iter()
        .flat_map(|&x| cells.iter().map(move |&y| [x, y, 0.5 * thick]))
        .collect();
    let bad = samples_in_tets_joining_skins(&refined, surface.points.len(), &skin, &samples);
    assert_eq!(
        bad.len(),
        0,
        "{} of {} mid-plane samples are inside a tetrahedron joining both skins: {bad:?}",
        bad.len(),
        samples.len()
    );
    assert!(report.wall_fill_points > 0, "the wall was filled");
    // Columns through the middle of the face each cross at least two tetrahedra.
    for &x in &cells {
        for &y in &cells {
            let seen = column_walk(
                &refined,
                [x + 0.0173, y - 0.0137, -1.0],
                [x + 0.0173, y - 0.0137, thick + 1.0],
            );
            assert!(seen.len() >= 2, "column ({x}, {y}) crossed {}", seen.len());
        }
    }
}

/// A circular tube about the z axis with vertices only on its two end rings.
fn end_ring_tube(segments: usize, inner: f64, outer: f64, length: f64) -> Surface {
    let ring = |radius: f64, i: usize, z: f64| {
        let (s, c) = sin_cos(std::f64::consts::TAU * (i % segments) as f64 / segments as f64);
        [radius * c, radius * s, z]
    };
    let mut surface = Surface::default();
    for i in 0..segments {
        let next = i + 1;
        surface.quad(
            ring(outer, i, 0.0),
            ring(outer, next, 0.0),
            ring(outer, next, length),
            ring(outer, i, length),
        );
        surface.quad(
            ring(inner, next, 0.0),
            ring(inner, i, 0.0),
            ring(inner, i, length),
            ring(inner, next, length),
        );
        surface.quad(
            ring(outer, i, 0.0),
            ring(inner, i, 0.0),
            ring(inner, next, 0.0),
            ring(outer, next, 0.0),
        );
        surface.quad(
            ring(outer, i, length),
            ring(outer, next, length),
            ring(inner, next, length),
            ring(inner, i, length),
        );
    }
    surface
}

#[test]
fn a_tube_with_only_end_rings_has_no_mid_wall_sample_in_a_tetrahedron_joining_its_skins() {
    let (inner, outer, length, segments) = (9.0, 10.0, 40.0, 24);
    let surface = end_ring_tube(segments, inner, outer, length);
    let unrefined = recovered(&surface);
    let (refined, report) = run_refine(
        unrefined.clone(),
        constant_size(200.0),
        options_with_layers(2),
    )
    .expect("refines");
    assert_contracts(&refined, &unrefined, &surface);
    assert_eq!(report.through_edges_refused, 0, "a through edge was left");
    // A surface vertex is on the skin of the radius it was built at; both radii meet the ends.
    let skin = |v: usize| {
        let p = surface.points[v];
        let r2 = p[0] * p[0] + p[1] * p[1];
        if (r2 - inner * inner).abs() < 1e-9 {
            Some(0u8)
        } else if (r2 - outer * outer).abs() < 1e-9 {
            Some(1u8)
        } else {
            None
        }
    };
    // The same two fill steps from each end ring: the forty unit axial edge over sixteen steps,
    // twice over.
    let margin = 5.0;
    let radius = 0.5 * (inner + outer);
    let zs = grid(margin, length - margin, 8);
    let samples: Vec<[f64; 3]> = grid(0.0, 1.0, 40)
        .iter()
        .flat_map(|&fraction| {
            let (s, c) = sin_cos(std::f64::consts::TAU * fraction);
            zs.iter().map(move |&z| [radius * c, radius * s, z])
        })
        .collect();
    let bad = samples_in_tets_joining_skins(&refined, surface.points.len(), &skin, &samples);
    assert_eq!(
        bad.len(),
        0,
        "{} of {} mid-wall samples are inside a tetrahedron joining both skins: {bad:?}",
        bad.len(),
        samples.len()
    );
    assert!(report.wall_fill_points > 0, "the wall was filled");
}

#[test]
fn a_chunky_eight_vertex_cube_and_slab_are_not_treated_as_walls() {
    for (name, surface, size) in [
        ("cube", subdivided_box([0.0; 3], [30.0; 3], 1), 100.0),
        (
            "slab",
            subdivided_box([0.0, 0.0, 0.0], [60.0, 60.0, 20.0], 1),
            100.0,
        ),
    ] {
        let (_, report) = run_refine(
            recovered(&surface),
            constant_size(size),
            options_with_layers(2),
        )
        .expect("refines");
        // A miss is worse than a false alarm, so a chunky part may lose a few interior edges. Eight
        // vertices have 28 pairs, of which 12 box edges and 6 face diagonals are surface edges, so
        // at most 10 pairs can be interior edges whatever the classifier decides.
        eprintln!("{name}: {} edges split", report.through_edges_split);
        assert!(
            report.through_edges_split <= 10,
            "{name}: {} edges split",
            report.through_edges_split
        );
        assert_eq!(report.through_edges_refused, 0, "{name} has a through edge");
    }
}

#[test]
fn a_tetrahedron_flat_to_rounding_is_rated_zero_degrees_and_a_regular_one_is_not() {
    // Exactly right-handed, four points a unit apart in one plane but for 1e-18, far below what
    // the face normals of a tetrahedron this flat can resolve (`quality.rs`).
    let flat = SolidMesh {
        points: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 1e-18],
        ],
        tets: vec![[0, 1, 2, 3]],
        neighbors: vec![[HULL; 4]],
    };
    let options = RefineOptions {
        quality_passes: 0,
        ..RefineOptions::default()
    };
    let (_, report) =
        run_refine(flat, constant_size(10.0), options).expect("a single tetrahedron refines");
    assert_eq!(
        report.after.min_degrees, 0.0,
        "a tetrahedron flat to rounding is the worst shape there is"
    );
    // Vertices of a cube alternate into a regular tetrahedron, wound right-handed.
    let regular = SolidMesh {
        points: vec![
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
        ],
        tets: vec![[0, 1, 2, 3]],
        neighbors: vec![[HULL; 4]],
    };
    let (_, report) =
        run_refine(regular, constant_size(100.0), options).expect("a regular tetrahedron refines");
    // The regular tetrahedron's dihedral is acos(1/3); rounding reaches only the last digits.
    assert!(
        (report.after.min_degrees - 70.528_779_365_509_31).abs() < 1e-9,
        "{}",
        report.after.min_degrees
    );
}

#[test]
fn three_layers_through_a_plate_put_every_added_point_on_the_two_third_planes() {
    let thick = 0.8;
    let surface = plate_with_holes(20.0, 20.0, thick, 10, 10, 1, &[]);
    let (refined, report) = run_refine(
        recovered(&surface),
        constant_size(1000.0),
        options_with_layers(3),
    )
    .expect("refines");
    assert_eq!(report.through_edges_refused, 0, "a through edge was left");
    assert!(report.layer_points > 0, "the plate gets layer points");
    // Layer points are placed at a third and two thirds of the wall from each skin, and through
    // edges are cut at the same fractions, so z is a third of 0.8 to rounding; smoothing moves a
    // point only in the wall's plane. 1e-9 of a millimetre is far above the rounding of a few
    // arithmetic steps on coordinates of size 20.
    let (third, two_thirds) = (thick / 3.0, 2.0 * thick / 3.0);
    let added = &refined.points[surface.points.len()..];
    let lower = added.iter().filter(|p| (p[2] - third).abs() < 1e-9).count();
    let upper = added
        .iter()
        .filter(|p| (p[2] - two_thirds).abs() < 1e-9)
        .count();
    assert_eq!(lower + upper, added.len(), "a point left its plane");
    assert!(lower > 0 && upper > 0, "both planes get points");
    let columns = grid(2.3, 17.7, 11);
    for &x in &columns {
        for &y in &columns {
            let seen = column_walk(
                &refined,
                [x + 0.0173, y - 0.0137, -1.0],
                [x + 0.0173, y - 0.0137, thick + 1.0],
            );
            assert!(seen.len() >= 2, "column ({x}, {y}) crossed {}", seen.len());
        }
    }
}
