// Size control and the quality pass, measured on the output: interior edge lengths against a
// graded size function, the worst dihedral reported before and after, and the guarantee the
// quality pass never lowers the mesh's worst tetrahedron.

use crate::shapes::{hollow_box, plate_with_holes, subdivided_box};
use crate::support::{
    constant_size, edge_length, min_dihedral, options_with_layers, recovered, run_refine,
};
use solidfem_tetra::{RefineOptions, SIZE_RATIO};

#[test]
fn interior_edges_follow_a_graded_size_function() {
    // Sizes from 5 to 13 across a 40 mm cube whose surface edges are about 13.3: every interior
    // edge should be within the split rule of the wanted size.
    let surface = subdivided_box([0.0; 3], [40.0; 3], 3);
    let size = |p: [f64; 3]| 5.0 + 0.2 * p[2];
    let (refined, report) = run_refine(
        recovered(&surface),
        Box::new(size),
        RefineOptions::default(),
    )
    .expect("refines");
    assert!(report.size_points > 0, "the graded function adds points");
    let first_added = surface.points.len() as u32;
    let mut total = 0.0;
    let mut count = 0usize;
    let (mut low, mut high) = (f64::INFINITY, 0.0_f64);
    let mut within = 0usize;
    for tet in &refined.tets {
        for a in 0..4 {
            for b in a + 1..4 {
                let (u, v) = (tet[a], tet[b]);
                if u < first_added || v < first_added {
                    continue;
                }
                let (pu, pv) = (refined.points[u as usize], refined.points[v as usize]);
                let middle = [
                    (pu[0] + pv[0]) / 2.0,
                    (pu[1] + pv[1]) / 2.0,
                    (pu[2] + pv[2]) / 2.0,
                ];
                let ratio = edge_length(&refined, u, v) / size(middle);
                total += ratio;
                count += 1;
                low = low.min(ratio);
                high = high.max(ratio);
                if (0.4..=2.0).contains(&ratio) {
                    within += 1;
                }
            }
        }
    }
    let mean = total / count as f64;
    eprintln!(
        "graded block: {count} interior edges, mean edge/size {mean:.3}, range {low:.3}..{high:.3}, within [0.4, 2.0]: {within}"
    );
    // The derived bound: a tetrahedron is only left alone once its longest interior edge is at
    // most SIZE_RATIO times the size at its centroid, and a point is only added at least half
    // the local size from its neighbours, so the mean ratio lies within [0.5, SIZE_RATIO].
    assert!(
        (0.5..=SIZE_RATIO).contains(&mean),
        "mean interior edge {mean} is outside the requested size"
    );
}

#[test]
fn the_quality_pass_reports_its_histogram_and_never_lowers_the_worst_tetrahedron() {
    for (name, solid, layers) in [
        (
            "plate",
            recovered(&plate_with_holes(20.0, 20.0, 0.8, 10, 10, 1, &[])),
            2,
        ),
        ("thin box", recovered(&hollow_box(30.0, 1.0, 6, 5)), 2),
        (
            "block",
            recovered(&subdivided_box([0.0; 3], [30.0; 3], 4)),
            0,
        ),
    ] {
        let (refined, report) =
            run_refine(solid, constant_size(6.0), options_with_layers(layers)).expect("refines");
        let measured = refined
            .tets
            .iter()
            .map(|tet| min_dihedral(&refined, tet))
            .fold(180.0_f64, f64::min);
        eprintln!(
            "{name}: tets {}, min dihedral before {:.2}, after {:.2}; before {:?}, after {:?}",
            report.after.tets,
            report.before.min_degrees,
            report.after.min_degrees,
            report.before.counts,
            report.after.counts,
        );
        // The derived guarantee: every accepted move raised the worst tetrahedron of the region
        // it replaced, so the mesh's worst cannot fall. Rounding of the measure is 1e-9 degrees.
        assert!(
            report.after.min_degrees >= report.before.min_degrees - 1e-9,
            "{name}: quality fell from {} to {}",
            report.before.min_degrees,
            report.after.min_degrees
        );
        assert!(
            (report.after.min_degrees - measured).abs() < 1e-6,
            "{name}: the report's worst {} disagrees with the mesh's {measured}",
            report.after.min_degrees
        );
    }
}
