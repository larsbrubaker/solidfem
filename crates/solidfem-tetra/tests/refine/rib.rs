// A thin rib standing in the middle of a plate, its foot surrounded by the plate's top face: that
// face is one flat patch holding the root vertices of both rib skins, so a chord from one foot to
// the other lies in the top plane yet passes through the rib's root, inside the solid. Which
// vertex is on which rib skin comes from how the shape is built (the sweep's tags), not from the
// library's own rule.

use crate::check::mesh;
use crate::support::{constant_size, options_with_layers, run_refine};
use crate::sweep::{false_edges, Probe, Tag};

/// The rib's two skins, its ends and the heights of its foot and top.
const RIB_LOW: f64 = 14.6;
const RIB_HIGH: f64 = 15.4;
const RIB_Y: [f64; 2] = [9.0, 21.0];
const PLATE_TOP: f64 = 2.0;
const RIB_TOP: f64 = 10.0;

/// The rib-skin tag of a vertex: wall 0, side 0 on x = 14.6 and 1 on x = 15.4, its top and two
/// ends being free rims.
fn rib_tags(x: f64, y: f64, z: f64) -> Vec<Tag> {
    let side = if x == RIB_LOW {
        0
    } else if x == RIB_HIGH {
        1
    } else {
        return Vec::new();
    };
    if z < PLATE_TOP || y < RIB_Y[0] || y > RIB_Y[1] {
        return Vec::new();
    }
    let rim =
        u8::from(z == RIB_TOP) | (u8::from(y == RIB_Y[0]) << 1) | (u8::from(y == RIB_Y[1]) << 2);
    vec![Tag { wall: 0, side, rim }]
}

fn at(p: &mut Probe, x: f64, y: f64, z: f64) -> u32 {
    p.vertex([x, y, z], || rib_tags(x, y, z))
}

/// Quads on the plane `x = fixed` (`along_x` false) or `y = fixed` (true), over `run` by `zs`,
/// facing the positive axis when `plus`.
fn wall_face(p: &mut Probe, along_x: bool, fixed: f64, run: &[f64], zs: &[f64], plus: bool) {
    for r in run.windows(2) {
        for z in zs.windows(2) {
            // Counterclockwise from outside: +x and -y faces run the same way along the lattice.
            let forward = plus != along_x;
            let (ra, rb) = if forward { (r[0], r[1]) } else { (r[1], r[0]) };
            let point = |run: f64, z: f64| {
                if along_x {
                    (run, fixed, z)
                } else {
                    (fixed, run, z)
                }
            };
            let corners = [
                point(ra, z[0]),
                point(rb, z[0]),
                point(rb, z[1]),
                point(ra, z[1]),
            ];
            let q = corners.map(|(x, y, z)| at(p, x, y, z));
            p.quad(q[0], q[1], q[2], q[3]);
        }
    }
}

/// A 30 x 30 x 2 plate on a 3 mm lattice with a rib 0.8 thick from x 14.6 to 15.4, y 9 to 21, z
/// 2 to 10, wound outward.
fn rib_on_plate() -> Probe {
    let xs = [
        0.0, 3.0, 6.0, 9.0, 12.0, RIB_LOW, RIB_HIGH, 18.0, 21.0, 24.0, 27.0, 30.0,
    ];
    let ys: Vec<f64> = (0..=10).map(|j| 3.0 * f64::from(j)).collect();
    let rib_ys: Vec<f64> = ys
        .iter()
        .copied()
        .filter(|y| (RIB_Y[0]..=RIB_Y[1]).contains(y))
        .collect();
    let rib_zs = [PLATE_TOP, 5.0, 8.0, RIB_TOP];
    let mut p = Probe::default();
    for x in xs.windows(2) {
        for y in ys.windows(2) {
            let under_rib = x[0] == RIB_LOW && y[0] >= RIB_Y[0] && y[1] <= RIB_Y[1];
            if !under_rib {
                let q = [(x[0], y[0]), (x[1], y[0]), (x[1], y[1]), (x[0], y[1])]
                    .map(|(a, b)| at(&mut p, a, b, PLATE_TOP));
                p.quad(q[0], q[1], q[2], q[3]);
            }
            let q = [(x[0], y[0]), (x[0], y[1]), (x[1], y[1]), (x[1], y[0])]
                .map(|(a, b)| at(&mut p, a, b, 0.0));
            p.quad(q[0], q[1], q[2], q[3]);
        }
    }
    wall_face(&mut p, false, 0.0, &ys, &[0.0, PLATE_TOP], false);
    wall_face(&mut p, false, 30.0, &ys, &[0.0, PLATE_TOP], true);
    wall_face(&mut p, true, 0.0, &xs, &[0.0, PLATE_TOP], false);
    wall_face(&mut p, true, 30.0, &xs, &[0.0, PLATE_TOP], true);
    wall_face(&mut p, false, RIB_LOW, &rib_ys, &rib_zs, false);
    wall_face(&mut p, false, RIB_HIGH, &rib_ys, &rib_zs, true);
    wall_face(&mut p, true, RIB_Y[0], &[RIB_LOW, RIB_HIGH], &rib_zs, false);
    wall_face(&mut p, true, RIB_Y[1], &[RIB_LOW, RIB_HIGH], &rib_zs, true);
    for y in rib_ys.windows(2) {
        let q = [
            (RIB_LOW, y[0]),
            (RIB_HIGH, y[0]),
            (RIB_HIGH, y[1]),
            (RIB_LOW, y[1]),
        ]
        .map(|(a, b)| at(&mut p, a, b, RIB_TOP));
        p.quad(q[0], q[1], q[2], q[3]);
    }
    p
}

#[test]
fn a_rib_standing_in_a_plate_has_no_edge_joining_its_skins_exact_or_rounded() {
    let base = rib_on_plate();
    let rotations = [[0.37, 0.61, 0.0], [1.1, 0.23, 0.47], [0.71, 1.13, 0.37]];
    let mut variants = vec![("exact".to_string(), base.clone())];
    for rotation in rotations {
        let turned = base.rotated(rotation);
        let mut rounded = turned.clone();
        // Single precision, as mesh files store positions.
        for point in &mut rounded.surface.points {
            *point = point.map(|c| f64::from(c as f32));
        }
        variants.push((format!("turned {rotation:?}"), turned));
        variants.push((format!("turned {rotation:?} rounded to f32"), rounded));
    }
    let mut found = Vec::new();
    for (name, probe) in &variants {
        let (solid, _) = mesh(&probe.surface.points, &probe.surface.triangles)
            .unwrap_or_else(|e| panic!("{name}: surface recovers: {e}"));
        let (refined, _) = run_refine(solid, constant_size(3.0), options_with_layers(2))
            .unwrap_or_else(|e| panic!("{name}: refines: {e}"));
        let edges = false_edges(probe, &refined.tets);
        if !edges.is_empty() {
            found.push(format!(
                "{name}: {} edges, first {:?}",
                edges.len(),
                edges[0]
            ));
        }
    }
    assert!(
        found.is_empty(),
        "edges joining the rib's skins: {found:#?}"
    );
}
