// The flat tetrahedra a curved wall leaves behind: four surface vertices either side of a gentle
// fold, two surface faces and one free edge (the other diagonal). Flips cannot remove them and
// nothing else in refinement touches them, so they used to survive to the solver as elements
// thin as the fold, down to flat to rounding. `prism(128, 50.0, 5.0, _)` is the smallest shape
// here whose wall has a fold of 2.8 degrees yet whose worst element is below one degree. A flat
// element left in place makes the 10-node upgrade refuse the whole mesh, so one that no flip can
// remove has to be replaced by a point, however low over the surface that point has to go.

use crate::check::assert_solid;
use crate::shapes::{prism, sin_cos, Surface};
use crate::support::{
    assert_contracts, boundary_faces, constant_size, dot, min_dihedral, options_with_layers,
    recovered, run_refine, sub,
};
use robust::{orient3d, Coord3D};
use solidfem_tetra::{RefineOptions, RefineReport, Refiner, SolidMesh, StepStatus, HULL};
use std::collections::HashSet;
use std::f64::consts::TAU;

/// A wall with 128 facets and a size so large no size point goes in: every element comes from
/// recovery and the quality pass.
fn curved_wall() -> Surface {
    prism(128, 50.0, 5.0, false)
}

fn refine(surface: &Surface) -> (SolidMesh, RefineReport) {
    run_refine(
        recovered(surface),
        constant_size(1e6),
        RefineOptions::default(),
    )
    .expect("refines")
}

#[test]
fn a_gentle_fold_of_a_curved_wall_leaves_no_element_thinner_than_one_degree() {
    let surface = curved_wall();
    let unrefined = recovered(&surface);
    let (refined, report) = refine(&surface);
    assert_contracts(&refined, &unrefined, &surface);
    // One degree is the line this test draws for a sliver: under the 5 degrees the quality pass
    // aims for, and over the 0.69 degrees the fold sliver measures when nothing removes it.
    let worst = refined
        .tets
        .iter()
        .map(|tet| min_dihedral(&refined, tet))
        .fold(180.0, f64::min);
    assert!(worst >= 1.0, "the worst element is {worst} degrees");
    assert!(
        report.quality_points > 0,
        "the slivers need points: {report:?}"
    );
    assert_eq!(
        refined.points.len(),
        surface.points.len() + report.quality_points,
        "only the quality pass added points, and it reports them all"
    );
}

#[test]
fn removing_a_sliver_gives_the_same_bits_however_the_run_is_sliced() {
    let surface = curved_wall();
    let whole = refine(&surface).0;
    for slice in [1, 7] {
        let mut refiner = Refiner::new(
            recovered(&surface),
            constant_size(1e6),
            RefineOptions::default(),
        )
        .expect("valid");
        loop {
            match refiner.step(slice, &mut || false).expect("refines") {
                StepStatus::Done => break,
                StepStatus::Running => {}
                StepStatus::Cancelled => panic!("nothing asked to cancel"),
            }
        }
        assert_eq!(refiner.result().expect("finished"), whole, "slice {slice}");
    }
}

/// A tube of `facets` flat panels around, `wall` thick and one ring tall, panels wound outward.
fn tube(facets: usize, radius: f64, wall: f64, height: f64) -> Surface {
    let mut s = Surface::default();
    let at = |i: usize, r: f64, z: f64| {
        let (sine, cosine) = sin_cos(TAU * (i % facets) as f64 / facets as f64);
        [r * cosine, r * sine, z]
    };
    let inner = radius - wall;
    for i in 0..facets {
        s.quad(
            at(i, radius, 0.0),
            at(i + 1, radius, 0.0),
            at(i + 1, radius, height),
            at(i, radius, height),
        );
        s.quad(
            at(i + 1, inner, 0.0),
            at(i, inner, 0.0),
            at(i, inner, height),
            at(i + 1, inner, height),
        );
        s.quad(
            at(i, inner, height),
            at(i, radius, height),
            at(i + 1, radius, height),
            at(i + 1, inner, height),
        );
        s.quad(
            at(i, radius, 0.0),
            at(i, inner, 0.0),
            at(i + 1, inner, 0.0),
            at(i + 1, radius, 0.0),
        );
    }
    s
}

#[test]
fn a_quality_point_in_a_layered_tube_leaves_no_interior_edge_across_the_wall() {
    let surface = tube(128, 50.0, 1.0, 5.0);
    let unrefined = recovered(&surface);
    let (refined, report) = run_refine(
        unrefined.clone(),
        constant_size(1e6),
        options_with_layers(2),
    )
    .expect("the tube refines");
    assert_contracts(&refined, &unrefined, &surface);
    assert!(
        report.quality_points > 0,
        "a sliver needs a point here: {report:?}"
    );
    // Independent of the library's own rule: a surface vertex is on the outer or the inner skin
    // by its radius. An interior edge joining the two skins spans the wall, unless both ends lie
    // in one end cap, where an annulus quad's other diagonal runs along the surface.
    let on_skin = |v: u32| {
        ((v as usize) < surface.points.len()).then(|| {
            let p = surface.points[v as usize];
            (p[0] * p[0] + p[1] * p[1]).sqrt() > 49.5
        })
    };
    let boundary: HashSet<(u32, u32)> = boundary_faces(&refined)
        .iter()
        .flat_map(|f| [(f[0], f[1]), (f[1], f[2]), (f[0], f[2])])
        .collect();
    for tet in &refined.tets {
        let mut sorted = *tet;
        sorted.sort_unstable();
        for i in 0..4 {
            for j in i + 1..4 {
                let (u, v) = (sorted[i], sorted[j]);
                let across = matches!((on_skin(u), on_skin(v)), (Some(a), Some(b)) if a != b);
                if across && !boundary.contains(&(u, v)) {
                    assert_eq!(
                        surface.points[u as usize][2], surface.points[v as usize][2],
                        "interior edge {u}-{v} joins the two skins"
                    );
                }
            }
        }
    }
}

/// The 10-node upgrade's limit on a tetrahedron's determinant over its longest edge cubed: below
/// it the upgrade refuses the whole mesh (`FLAT_RELATIVE_VOLUME` in `solidfem-mesh/src/tet.rs`, a
/// crate this one does not depend on).
const UPGRADE_FLAT_LIMIT: f64 = 1e-12;

/// The determinant of `tet` over its longest edge cubed, as the upgrade measures it.
fn relative_volume(solid: &SolidMesh, tet: &[u32; 4]) -> f64 {
    let p = tet.map(|v| solid.points[v as usize]);
    let (u, v, w) = (sub(p[1], p[0]), sub(p[2], p[0]), sub(p[3], p[0]));
    let cross = [
        v[1] * w[2] - v[2] * w[1],
        v[2] * w[0] - v[0] * w[2],
        v[0] * w[1] - v[1] * w[0],
    ];
    let mut longest_squared = 0.0_f64;
    for i in 0..4 {
        for j in i + 1..4 {
            let e = sub(p[i], p[j]);
            longest_squared = longest_squared.max(dot(e, e));
        }
    }
    dot(u, cross) / (longest_squared * longest_squared.sqrt())
}

/// A solid given by right-handed tetrahedra, its neighbour links read off their shared faces.
fn solid_from(points: Vec<[f64; 3]>, tets: Vec<[u32; 4]>) -> SolidMesh {
    let face = |tet: &[u32; 4], opposite: usize| {
        let mut key: Vec<u32> = (0..4).filter(|&k| k != opposite).map(|k| tet[k]).collect();
        key.sort_unstable();
        key
    };
    let neighbors = tets
        .iter()
        .enumerate()
        .map(|(t, tet)| {
            std::array::from_fn(|i| {
                (0..tets.len())
                    .filter(|&other| other != t)
                    .flat_map(|other| (0..4).map(move |j| (other, j)))
                    .find(|&(other, j)| face(&tets[other], j) == face(tet, i))
                    .map_or(HULL, |(other, _)| other as u32)
            })
        })
        .collect();
    SolidMesh {
        points,
        tets,
        neighbors,
    }
}

/// The boundary faces of `solid` as surface triangles, each wound so the solid is on the side the
/// surface checks call inside: its opposite vertex makes a positive exact orientation.
fn outward_triangles(solid: &SolidMesh) -> Vec<[u32; 3]> {
    let coord = |v: u32| {
        let p = solid.points[v as usize];
        Coord3D {
            x: p[0],
            y: p[1],
            z: p[2],
        }
    };
    let mut triangles = Vec::new();
    for (t, tet) in solid.tets.iter().enumerate() {
        for i in 0..4 {
            if solid.neighbors[t][i] != HULL {
                continue;
            }
            let face: Vec<u32> = (0..4).filter(|&j| j != i).map(|j| tet[j]).collect();
            let (a, b, c) = (face[0], face[1], face[2]);
            if orient3d(coord(a), coord(b), coord(c), coord(tet[i])) > 0.0 {
                triangles.push([a, b, c]);
            } else {
                triangles.push([a, c, b]);
            }
        }
    }
    triangles
}

/// Four surface vertices `a b c d` a rounding error out of one plane, surface triangles `a b c`
/// and `a c d` under a flat tetrahedron whose free edge is the other diagonal `b d`, and a low
/// roof of two apexes so the ring around `b d` is `a c e1 e2`. Neither triangulation of that ring
/// is legal, each has a triangle with `b` and `d` on one side. The base's mean edge is 2.28, and
/// the only points above its centre that see all eight surface faces are near the base: every
/// height from a quarter of an edge up is outside, and an eighth of an edge is inside. The roof's
/// tetrahedra stay above the 5 degrees where a neighbour would get a point of its own.
fn flat_under_low_roof() -> SolidMesh {
    let (a, b, c, d, e1, e2) = (0u32, 1u32, 2u32, 3u32, 4u32, 5u32);
    solid_from(
        vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 1e-20],
            [0.0, 0.75, 1.0],
            [0.75, 0.0, 0.75],
        ],
        vec![[c, a, b, d], [e1, c, b, d], [e2, e1, b, d], [a, e2, b, d]],
    )
}

#[test]
fn a_flat_element_no_flip_can_remove_is_replaced_by_a_point_low_over_the_surface() {
    let solid = flat_under_low_roof();
    assert!(
        relative_volume(&solid, &solid.tets[0]) < UPGRADE_FLAT_LIMIT,
        "the shape must start with an element the upgrade refuses"
    );
    let surface_before = boundary_faces(&solid);
    let surface_points = solid.points.len();
    let surface_coordinates = solid.points.clone();
    let surface_triangles = outward_triangles(&solid);
    let (refined, report) =
        run_refine(solid, constant_size(1e6), RefineOptions::default()).expect("refines");
    assert_solid(&refined, &surface_coordinates, &surface_triangles);
    assert_eq!(
        boundary_faces(&refined),
        surface_before,
        "the surface changed"
    );
    assert!(report.quality_points >= 1, "it needs a point: {report:?}");
    assert_eq!(refined.points.len(), surface_points + report.quality_points);
    for tet in &refined.tets {
        let relative = relative_volume(&refined, tet);
        assert!(
            relative >= UPGRADE_FLAT_LIMIT,
            "an element the upgrade would refuse remains: {tet:?} measures {relative:e}"
        );
    }
}

/// How far single-precision storage may have put a vertex of `solid` from where it was meant:
/// each coordinate is rounded by at most `2^-24` of the largest magnitude `M`, so a point by at
/// most `sqrt(3) 2^-24 M`.
fn input_rounding(solid: &SolidMesh) -> f64 {
    let magnitude = solid
        .points
        .iter()
        .flatten()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    3.0_f64.sqrt() * magnitude / 16_777_216.0
}

/// Whether moving each vertex of `tet` by at most `rounding` could have made it from four
/// coplanar points: moving one vertex by `m` changes the determinant of the edge vectors by `m`
/// dotted with the opposite face's cross product, so four coplanar points moved that far have a
/// determinant of at most `rounding` times the sum of the four faces' cross-product lengths (terms
/// in two or more moves are smaller again by `rounding` over an edge).
fn flat_to_input_rounding(solid: &SolidMesh, tet: &[u32; 4], rounding: f64) -> bool {
    let p = tet.map(|v| solid.points[v as usize]);
    let cross = |u: [f64; 3], w: [f64; 3]| {
        [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ]
    };
    let determinant = dot(sub(p[1], p[0]), cross(sub(p[2], p[0]), sub(p[3], p[0])));
    let mut areas = 0.0;
    for [i, j, k] in [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]] {
        let n = cross(sub(p[j], p[i]), sub(p[k], p[i]));
        areas += dot(n, n).sqrt();
    }
    determinant.abs() <= rounding * areas
}

#[test]
fn a_flat_element_within_the_input_rounding_is_replaced_though_the_upgrade_would_take_it() {
    // The shape of `flat_under_low_roof` a thousand units from the origin, where storage rounding
    // reaches `e = sqrt(3) 2^-24 1002`, about 1.03e-4, with `d` 1e-5 off the plane of `a b c`: a
    // tenth of that rounding, as a flat face of a part read from a file comes, yet a determinant
    // over the longest edge cubed of about 1.8e-6, far above what the upgrade refuses.
    let mut solid = flat_under_low_roof();
    for point in &mut solid.points {
        *point = point.map(|value| value + 1000.0);
    }
    solid.points[3][2] = 1000.0 + 1e-5;
    let rounding = input_rounding(&solid);
    assert!(
        flat_to_input_rounding(&solid, &solid.tets[0], rounding),
        "the shape must start with an element flat to the input's rounding"
    );
    assert!(
        relative_volume(&solid, &solid.tets[0]) >= UPGRADE_FLAT_LIMIT,
        "the shape must start with an element the upgrade accepts"
    );
    let surface_before = boundary_faces(&solid);
    let surface_points = solid.points.len();
    let surface_coordinates = solid.points.clone();
    let surface_triangles = outward_triangles(&solid);
    // The worst angle a user is shown before: the flat element's, the roof's being far above it.
    let worst_before = solid
        .tets
        .iter()
        .map(|tet| min_dihedral(&solid, tet))
        .fold(180.0_f64, f64::min);
    let (refined, report) =
        run_refine(solid, constant_size(1e6), RefineOptions::default()).expect("refines");
    assert_solid(&refined, &surface_coordinates, &surface_triangles);
    assert_eq!(
        boundary_faces(&refined),
        surface_before,
        "the surface changed"
    );
    assert_eq!(refined.points.len(), surface_points + report.quality_points);
    // Every move of the quality pass, a flip, a smoothing step or a point, goes in only when it
    // strictly raises the worst tetrahedron of what it replaces, so the mesh's worst angle must
    // rise: the flat element was the worst, and it is gone.
    let worst_after = refined
        .tets
        .iter()
        .map(|tet| min_dihedral(&refined, tet))
        .fold(180.0_f64, f64::min);
    assert!(
        worst_after > worst_before,
        "the worst angle did not rise: {worst_before} degrees before, {worst_after} after"
    );
    for tet in &refined.tets {
        assert!(
            !flat_to_input_rounding(&refined, tet, rounding),
            "an element flat to the input's rounding remains: {tet:?}"
        );
    }
}
