// Wall fill at a sharp edge and on a faceted curved wall. Two faces meeting at a sharp edge are a
// wedge, not a wall, where a fill column's patch reaches the edge; a blade's faces away from its
// edge, and a tube whose facets tilt against each other, are walls and must keep their columns.
// The through-edge sweep counts edges only and cannot see fill that was lost, so the blade and
// the tube here are what guard against a wedge rule that also drops real walls.

use std::f64::consts::{PI, TAU};

use crate::shapes::{sin_cos, Surface};
use crate::support::{
    assert_contracts, constant_size, fingerprint, options_with_layers, recovered, run_refine,
};
use crate::wall_skins::{grid, samples_in_tets_joining_skins};

/// A long prism whose cross-section is an isosceles triangle with angle `apex` at the edge along
/// y through the origin, `depth` from the edge to its flat back, each long face a single quad.
/// The back meets both faces at `(PI - apex) / 2`, so only the two faces at the edge face each
/// other.
fn wedge_prism(apex: f64, depth: f64, length: f64) -> Surface {
    cut_wedge_prism(apex, depth, length, 1, false)
}

/// The same prism with each long face cut into `strips` quads running along the edge, and the
/// bottom face's quads split on their other diagonal when `other_diagonal` is set, so a top
/// triangle and the bottom triangle behind it need not share a vertex though their faces meet.
fn cut_wedge_prism(
    apex: f64,
    depth: f64,
    length: f64,
    strips: usize,
    other_diagonal: bool,
) -> Surface {
    let (sine, cosine) = sin_cos(0.5 * apex);
    let half = depth * sine / cosine;
    let across = |k: usize, z: f64| {
        if k == strips {
            [depth, 0.0, z]
        } else {
            let f = k as f64 / strips as f64;
            [depth * f, 0.0, z * f]
        }
    };
    let top = |k: usize| across(k, half);
    let bottom = |k: usize| across(k, -half);
    let at = |p: [f64; 3], y: f64| [p[0], y, p[2]];
    let mut surface = Surface::default();
    // Outward: the top face looks up and away from the edge, the bottom down, the back along +x.
    for k in 0..strips {
        let (a, b) = (top(k), top(k + 1));
        surface.quad(a, b, at(b, length), at(a, length));
    }
    for k in 0..strips {
        let (a, b) = (bottom(k), bottom(k + 1));
        if other_diagonal {
            surface.quad(at(a, length), at(b, length), b, a);
        } else {
            surface.quad(a, at(a, length), at(b, length), b);
        }
    }
    let (top_back, bottom_back) = (top(strips), bottom(strips));
    surface.quad(
        bottom_back,
        at(bottom_back, length),
        at(top_back, length),
        top_back,
    );
    surface.triangle(bottom(0), bottom(1), top(1));
    surface.triangle(
        at(bottom(0), length),
        at(top(1), length),
        at(bottom(1), length),
    );
    for k in 1..strips {
        surface.quad(bottom(k), bottom(k + 1), top(k + 1), top(k));
        let (b0, b1) = (at(bottom(k), length), at(bottom(k + 1), length));
        surface.quad(b0, at(top(k), length), at(top(k + 1), length), b1);
    }
    surface
}

/// A tube about the z axis with vertices only on its end rings, the inner ring turned half a
/// segment: each inner facet faces the meeting of two outer ones, so the facet a column reaches
/// is tilted against the column's own by half the segment angle while the wall carries on.
fn out_of_step_tube(segments: usize, inner: f64, outer: f64, length: f64) -> Surface {
    let ring = |radius: f64, turn: f64, i: usize, z: f64| {
        let (s, c) = sin_cos(TAU * ((i % segments) as f64 + turn) / segments as f64);
        [radius * c, radius * s, z]
    };
    let o = |i: usize, z: f64| ring(outer, 0.0, i, z);
    let n = |i: usize, z: f64| ring(inner, 0.5, i, z);
    let mut surface = Surface::default();
    for i in 0..segments {
        let next = i + 1;
        surface.quad(o(i, 0.0), o(next, 0.0), o(next, length), o(i, length));
        surface.quad(n(next, 0.0), n(i, 0.0), n(i, length), n(next, length));
        surface.quad(o(i, 0.0), n(i, 0.0), n(next, 0.0), o(next, 0.0));
        surface.quad(o(i, length), o(next, length), n(next, length), n(i, length));
    }
    surface
}

#[test]
fn a_thirty_degree_edge_gets_no_wall_fill_columns_beside_it() {
    // Faces about two units wide and forty long: the fill grid's step is a sixteenth of the
    // forty-unit diagonal, wider than the face, so every column's patch reaches the edge where
    // the far face meets its own. The wedge is at most about one unit thick, a tenth of the
    // element size asked for.
    let surface = wedge_prism(PI / 6.0, 2.0, 40.0);
    let unrefined = recovered(&surface);
    let (refined, report) = run_refine(
        unrefined.clone(),
        constant_size(10.0),
        options_with_layers(2),
    )
    .expect("refines");
    assert_contracts(&refined, &unrefined, &surface);
    assert_eq!(
        report.wall_fill_points, 0,
        "fill columns were laid into the wedge at the sharp edge: {report:?}"
    );
}

#[test]
fn the_wedge_gets_no_wall_fill_however_its_faces_are_triangulated() {
    // The same thirty-degree prism with the bottom quads split the other way, so a top triangle
    // and the bottom one its columns reach can share no vertex, and with the faces cut into
    // strips along the edge, so a column reaches a strip away from it. Every column still stands
    // in the wedge: the faces are under two and a half units wide and the step is a sixteenth of
    // the forty-unit length, so no column is a grid step from the edge.
    let mut laid = Vec::new();
    for (strips, other_diagonal) in [(1, true), (2, true), (4, false), (8, false), (8, true)] {
        let surface = cut_wedge_prism(PI / 6.0, 2.0, 40.0, strips, other_diagonal);
        let unrefined = recovered(&surface);
        let (refined, report) = run_refine(
            unrefined.clone(),
            constant_size(10.0),
            options_with_layers(2),
        )
        .expect("refines");
        assert_contracts(&refined, &unrefined, &surface);
        if report.wall_fill_points != 0 {
            laid.push(format!(
                "{strips} strips, other diagonal {other_diagonal}: {} fill points",
                report.wall_fill_points
            ));
        }
    }
    assert!(
        laid.is_empty(),
        "fill columns were laid into the wedge: {laid:#?}"
    );
}

#[test]
fn a_blade_keeps_its_fill_columns_from_one_grid_step_off_its_edge() {
    // A two-degree blade sixty units deep: its thickness changes by about a thirtieth of a unit
    // per unit across, so only columns within one grid step of the edge see their patch reach it.
    let (depth, length) = (60.0, 40.0);
    let surface = wedge_prism(PI / 90.0, depth, length);
    let unrefined = recovered(&surface);
    let (refined, report) = run_refine(
        unrefined.clone(),
        constant_size(10.0),
        options_with_layers(2),
    )
    .expect("refines");
    assert_contracts(&refined, &unrefined, &surface);
    assert_eq!(report.through_edges_refused, 0, "a through edge was left");
    let skin = |v: usize| {
        let z = surface.points[v][2];
        if z > 0.0 {
            Some(1u8)
        } else if z < 0.0 {
            Some(0u8)
        } else {
            None
        }
    };
    // The grid step is a sixteenth of the face's diagonal. Away from the back and the ends the
    // samples keep the two steps of the plain box's rim band; from the edge, one step more, the
    // band in which columns see their patch reach the edge and are left out.
    let step = (depth * depth + length * length).sqrt() / 16.0;
    let xs = grid(3.0 * step, depth - 2.0 * step, 12);
    let ys = grid(2.0 * step, length - 2.0 * step, 8);
    let samples: Vec<[f64; 3]> = xs
        .iter()
        .flat_map(|&x| ys.iter().map(move |&y| [x, y, 0.0]))
        .collect();
    let bad = samples_in_tets_joining_skins(&refined, surface.points.len(), &skin, &samples);
    assert_eq!(
        bad.len(),
        0,
        "{} of {} mid-plane samples are inside a tetrahedron joining both skins: {bad:?}",
        bad.len(),
        samples.len()
    );
    // Every column the faces propose, placed or refused by the separation rule: each vertex ray
    // proposes the one middle point of two layers, and the rest are fill. Each long face is two
    // triangles laid at sixteen steps (the wedge is under a sixteenth of the diagonal thick), so
    // 105 interior grid points each. The triangle with two corners on the edge has a row one
    // sixteenth of the sixty-unit depth from it, 3.75 units, under the 4.5-unit step a column
    // needs at two layers: its 14 points go. The other triangle's nearest points are two
    // sixteenths away and stay.
    let proposed = report.layer_points + report.layer_points_dropped - report.thin_wall_vertices;
    assert_eq!(
        proposed,
        2 * (105 + 105 - 14),
        "the blade's faces did not keep exactly their columns a grid step from the edge: \
         {report:?}"
    );
}

#[test]
fn a_tube_with_out_of_step_facets_keeps_its_wall_fill() {
    // Half a segment of tilt is 11.25 degrees at sixteen segments: over a grid step of a sixteenth
    // of a hundred units, the far facet's plane would meet the near one inside the patch, though
    // the next facet carries the one-unit wall on unchanged.
    let (inner, outer) = (9.0, 10.0);
    let mut problems = Vec::new();
    for (segments, length) in [(16, 100.0), (16, 200.0), (24, 200.0)] {
        let surface = out_of_step_tube(segments, inner, outer, length);
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
        // Two fill steps from each end ring, as for the tube in step: the axial edge over sixteen
        // steps, twice over.
        let margin = length / 8.0;
        let radius = 0.5 * (inner + outer);
        let zs = grid(margin, length - margin, 8);
        let samples: Vec<[f64; 3]> = grid(0.0, 1.0, 40)
            .iter()
            .flat_map(|&fraction| {
                let (s, c) = sin_cos(TAU * fraction);
                zs.iter().map(move |&z| [radius * c, radius * s, z])
            })
            .collect();
        let bad = samples_in_tets_joining_skins(&refined, surface.points.len(), &skin, &samples);
        // Not derived: the fill as it was before any wedge rule leaves one sample in each of these
        // tubes inside an element joining the out-of-step skins, two to three steps from an end,
        // and this test does not set out to remove it. Losing the fill shows as about a hundred.
        const BRIDGED_BEFORE_WEDGE_RULE: usize = 1;
        if report.wall_fill_points == 0 || bad.len() > BRIDGED_BEFORE_WEDGE_RULE {
            problems.push(format!(
                "{segments} segments, {length} long: {} fill points, {} of {} mid-wall samples \
                 inside a tetrahedron joining both skins: {bad:?}",
                report.wall_fill_points,
                bad.len(),
                samples.len()
            ));
        }
    }
    assert!(problems.is_empty(), "{problems:#?}");
}

#[test]
fn the_out_of_step_tubes_mesh_to_the_same_bits_on_every_system() {
    // The tubes' input is `sin_cos` and plain arithmetic, the same bits everywhere, and the
    // mesher uses only exact predicates, `+ - * /`, `sqrt` and `libm`. So each value is the same
    // on every system, or the mesher depends on the platform: a math-library call whose last bit
    // differs moved a layer point or changed a choice. These are the tubes the wall-fill test
    // above measures; a one-ulp change in the layer normals' angle moves the 200-long one, and in
    // the quality pass's angle the 100-long one, so both measures are covered. The constants were
    // recorded once; a change to the mesher that changes these meshes on purpose records them
    // again and says why.
    const RECORDED: [(usize, f64, u64); 3] = [
        (16, 100.0, 0x7039_754e_d0cc_9f89),
        (16, 200.0, 0x4884_f5ee_80d6_7a9d),
        (24, 200.0, 0xbe0d_4859_4fc5_4fd0),
    ];
    let mut problems = Vec::new();
    for (segments, length, recorded) in RECORDED {
        let surface = out_of_step_tube(segments, 9.0, 10.0, length);
        let (refined, _) = run_refine(
            recovered(&surface),
            constant_size(200.0),
            options_with_layers(2),
        )
        .expect("refines");
        let found = fingerprint(&refined);
        if found != recorded {
            problems.push(format!(
                "{segments} segments, {length} long: {} points, {} tetrahedra, fingerprint \
                 {found:#018x}, recorded {recorded:#018x}",
                refined.points.len(),
                refined.tets.len()
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "a refined tube is not the recorded mesh: {problems:#?}"
    );
}
