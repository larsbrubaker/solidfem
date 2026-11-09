// No element joins opposite skins of a thin wall, on surfaces whose two skins carry different
// triangulations, measured on the output alone: an independent skin test from the input surface
// (the z of a plate's flat faces), tets with a vertex on each skin, and a ray walk through the
// neighbour links.

use crate::shapes::{hollow_box, subdivided_box, Surface};
use crate::support::{
    assert_contracts, column_walk, constant_size, min_dihedral, options_with_layers, recovered,
    run_refine,
};

/// The four cell corners moved within their cell so the top and bottom triangulations differ
/// everywhere but the rim, which must match to close.
fn unaligned_plate(length: f64, width: f64, thick: f64, cells: usize) -> Surface {
    let mut s = Surface::default();
    let at = |i: usize, j: usize, top: bool| {
        let h = length / cells as f64;
        let (x, y) = (i as f64 * h, j as f64 * (width / cells as f64));
        // Rim vertices stay on the lattice; interior ones move by a sign-alternating fraction of
        // the cell that differs between the skins.
        let on_rim = i == 0 || j == 0 || i == cells || j == cells;
        let (dx, dy) = if on_rim {
            (0.0, 0.0)
        } else {
            let sign = if top { 1.0 } else { -1.0 };
            (
                0.31 * h * sign * if (i + j).is_multiple_of(2) { 1.0 } else { -1.0 },
                0.23 * h * sign * if i.is_multiple_of(2) { 1.0 } else { -1.0 },
            )
        };
        [x + dx, y + dy, if top { thick } else { 0.0 }]
    };
    for i in 0..cells {
        for j in 0..cells {
            s.quad(
                at(i, j, true),
                at(i + 1, j, true),
                at(i + 1, j + 1, true),
                at(i, j + 1, true),
            );
            s.quad(
                at(i, j, false),
                at(i, j + 1, false),
                at(i + 1, j + 1, false),
                at(i + 1, j, false),
            );
        }
        s.quad(
            at(i, 0, false),
            at(i + 1, 0, false),
            at(i + 1, 0, true),
            at(i, 0, true),
        );
        s.quad(
            at(i + 1, cells, false),
            at(i, cells, false),
            at(i, cells, true),
            at(i + 1, cells, true),
        );
    }
    for j in 0..cells {
        s.quad(
            at(0, j + 1, false),
            at(0, j, false),
            at(0, j, true),
            at(0, j + 1, true),
        );
        s.quad(
            at(cells, j, false),
            at(cells, j + 1, false),
            at(cells, j + 1, true),
            at(cells, j, true),
        );
    }
    s
}

/// The vertices on a plate's two flat skins, from the input surface's coordinates alone.
fn plate_skin(p: [f64; 3], thick: f64) -> Option<u8> {
    if p[2] == 0.0 {
        Some(0)
    } else if p[2] == thick {
        Some(1)
    } else {
        None
    }
}

/// Columns placed away from every mesh edge, by a low-discrepancy sequence.
fn sample_columns(low: f64, high: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| low + (high - low) * (0.3117 + i as f64 * 0.61803).fract())
        .collect()
}

#[test]
fn an_unaligned_two_skin_plate_has_no_element_joining_its_skins_away_from_the_rims() {
    let thick = 0.8;
    let (length, cells) = (20.0, 10);
    let surface = unaligned_plate(length, length, thick, cells);
    let unrefined = recovered(&surface);
    let edge = length / cells as f64;
    let (refined, report) = run_refine(
        unrefined.clone(),
        constant_size(1.5 * edge),
        options_with_layers(2),
    )
    .expect("refines");
    assert_contracts(&refined, &unrefined, &surface);
    assert_eq!(
        report.through_edges_refused, 0,
        "every through edge was cut"
    );
    let surface_points = surface.points.len();
    // The rim is two surface edges wide on purpose: at it, a tetrahedron bridging the rim's quad
    // is not a through-wall element in the sense of this contract.
    let margin = 2.0 * edge;
    let clear = |value: f64| value > margin && value < length - margin;
    let mut through_tets = 0usize;
    for tet in &refined.tets {
        let skins: Vec<u8> = tet
            .iter()
            .filter(|&&v| (v as usize) < surface_points)
            .filter_map(|&v| plate_skin(refined.points[v as usize], thick))
            .collect();
        let joins_both = skins.contains(&0) && skins.contains(&1);
        if !joins_both {
            continue;
        }
        let centre = |k: usize| {
            tet.iter()
                .map(|&v| refined.points[v as usize][k])
                .sum::<f64>()
                / 4.0
        };
        if clear(centre(0)) && clear(centre(1)) {
            through_tets += 1;
        }
    }
    assert_eq!(
        through_tets, 0,
        "tetrahedra with a vertex on both skins away from the rim"
    );
    let (xs, ys) = (
        sample_columns(margin + 0.1, length - margin - 0.1, 31),
        sample_columns(margin + 0.3, length - margin - 0.3, 31),
    );
    for &x in &xs {
        for &y in &ys {
            let seen = column_walk(&refined, [x, y, -1.0], [x, y, thick + 1.0]);
            assert!(
                seen.len() >= 2,
                "column ({x}, {y}) crossed only {} tetrahedra",
                seen.len()
            );
            for t in seen {
                let zs = refined.tets[t].map(|v| refined.points[v as usize][2]);
                let span = zs.iter().fold(f64::MIN, |a, &z| a.max(z))
                    - zs.iter().fold(f64::MAX, |a, &z| a.min(z));
                assert!(
                    span < thick,
                    "column ({x}, {y}): tetrahedron {t} spans {span}, the whole {thick} wall"
                );
            }
        }
    }
}

#[test]
fn a_thin_hollow_box_with_unaligned_skins_has_no_element_joining_its_floor_skins() {
    let surface = hollow_box(30.0, 1.0, 6, 5);
    let unrefined = recovered(&surface);
    let (refined, report) = run_refine(
        unrefined.clone(),
        constant_size(8.0),
        options_with_layers(2),
    )
    .expect("refines");
    assert_contracts(&refined, &unrefined, &surface);
    eprintln!(
        "unaligned hollow box: split {} refused {}, layer points {}, before {:.2} {:?}, after {:.2} {:?}",
        report.through_edges_split,
        report.through_edges_refused,
        report.layer_points,
        report.before.min_degrees,
        report.before.counts,
        report.after.min_degrees,
        report.after.counts,
    );
    assert_eq!(
        report.through_edges_refused, 0,
        "every through edge was cut"
    );
    assert!(report.through_edges_split > 0, "the skins were unaligned");
    // The floor's outer and inner skins: z = 0 and z = 1. Every tetrahedron must not join them.
    let first_interior = surface.points.len();
    let floor_skin = |p: [f64; 3]| {
        if p[2] == 0.0 {
            Some(0u8)
        } else if p[2] == 1.0 {
            Some(1u8)
        } else {
            None
        }
    };
    for tet in &refined.tets {
        let skins: Vec<u8> = tet
            .iter()
            .filter(|&&v| (v as usize) < first_interior)
            .filter_map(|&v| floor_skin(refined.points[v as usize]))
            .collect();
        let (x, y) = (
            tet.iter()
                .map(|&v| refined.points[v as usize][0])
                .sum::<f64>()
                / 4.0,
            tet.iter()
                .map(|&v| refined.points[v as usize][1])
                .sum::<f64>()
                / 4.0,
        );
        // Away from the floor's edges (the rim of the floor plate is one wall thickness plus one
        // surface edge from the walls, which meet it at right angles).
        let clear = |v: f64| v > 7.0 && v < 23.0;
        if clear(x) && clear(y) {
            assert!(
                !(skins.contains(&0) && skins.contains(&1)),
                "a floor tetrahedron joins the skins"
            );
        }
    }
    // A column through the middle of the floor: at least two tets and no spanning one.
    let (xs, ys) = (sample_columns(8.1, 21.9, 17), sample_columns(8.3, 21.7, 17));
    for &x in &xs {
        for &y in &ys {
            let seen = column_walk(&refined, [x, y, -0.5], [x, y, 15.0]);
            assert!(
                seen.len() >= 2,
                "floor column ({x}, {y}) crossed only {} tetrahedra",
                seen.len()
            );
        }
    }
    // Report the worst angle rather than asserting a rise: this surface's own recovery leaves
    // three boundary triangles of three collinear grid points, and a tetrahedron standing on one
    // of those faces is flat however its interior is arranged, so refinement alone cannot raise
    // this minimum above zero without changing the surface.
    let worst = refined
        .tets
        .iter()
        .map(|tet| min_dihedral(&refined, tet))
        .fold(180.0_f64, f64::min);
    eprintln!("unaligned hollow box: worst minimum dihedral {worst:.3}");
}

#[test]
fn a_chunky_block_and_a_thick_slab_are_not_split_at_all() {
    // Solid enough that no face is within the thin-wall ratio of the opposite one at the local
    // surface edge length: a cube 30 thick with 7.5 edges, and a 60 x 60 x 20 slab with 8 edges.
    for (name, surface, size) in [
        ("cube", subdivided_box([0.0; 3], [30.0; 3], 4), 6.0),
        (
            "slab",
            subdivided_box([0.0, 0.0, 0.0], [60.0, 60.0, 20.0], 8),
            9.0,
        ),
    ] {
        let (_, report) = run_refine(
            recovered(&surface),
            constant_size(size),
            options_with_layers(2),
        )
        .expect("refines");
        eprintln!(
            "{name}: split {}, layer points {}",
            report.through_edges_split, report.layer_points
        );
        assert_eq!(report.through_edges_split, 0, "{name} was over-split");
        assert_eq!(
            report.through_edges_refused, 0,
            "{name} found a through edge"
        );
    }
}
