// The flat tetrahedra a curved wall leaves behind: four surface vertices either side of a gentle
// fold, two surface faces and one free edge (the other diagonal). Flips cannot remove them and
// nothing else in refinement touches them, so they used to survive to the solver as elements
// thin as the fold, down to flat to rounding. `prism(128, 50.0, 5.0, _)` is the smallest shape
// here whose wall has a fold of 2.8 degrees yet whose worst element is below one degree.

use crate::shapes::{prism, sin_cos, Surface};
use crate::support::{
    assert_contracts, boundary_faces, constant_size, min_dihedral, options_with_layers, recovered,
    run_refine,
};
use solidfem_tetra::{RefineOptions, RefineReport, Refiner, SolidMesh, StepStatus};
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
