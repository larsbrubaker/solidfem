// Size control and the quality pass, measured on the output: interior edge lengths against a
// graded size function, the worst dihedral reported before and after, and the guarantee the
// quality pass never lowers the mesh's worst tetrahedron, and size points no longer flattening the
// tetrahedron on a needle surface triangle of a thin slab.

use crate::shapes::{hollow_box, plate_with_holes, subdivided_box, Surface};
use crate::support::{
    constant_size, edge_length, min_dihedral, options_with_layers, recovered, run_refine,
};
use solidfem_tetra::{RefineOptions, SolidMesh, SIZE_RATIO};

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

/// A slab `length` x 4 x `thickness` whose top face holds a needle triangle from the edge `x = 0`,
/// where it is `2 * half_width` wide, to a point on the edge `x = length`, the rest of each face
/// fanned so the sides conform. Returns the surface and the needle's three vertices.
fn needle_slab(length: f64, half_width: f64, thickness: f64) -> (Surface, [u32; 3]) {
    let mut s = Surface::default();
    let t = thickness;
    let (a, b, c) = (
        [0.0, 2.0 - half_width, t],
        [0.0, 2.0 + half_width, t],
        [length, 2.0, t],
    );
    let (top00, top10, top14, top04) = (
        [0.0, 0.0, t],
        [length, 0.0, t],
        [length, 4.0, t],
        [0.0, 4.0, t],
    );
    let (bottom00, bottom10, bottom14, bottom04) = (
        [0.0, 0.0, 0.0],
        [length, 0.0, 0.0],
        [length, 4.0, 0.0],
        [0.0, 4.0, 0.0],
    );
    s.triangle(a, c, b);
    s.triangle(top00, top10, c);
    s.triangle(top00, c, a);
    s.triangle(b, c, top14);
    s.triangle(b, top14, top04);
    s.quad(bottom00, bottom04, bottom14, bottom10);
    s.quad(bottom00, bottom10, top10, top00);
    s.quad(top04, top14, bottom14, bottom04);
    for (from, to) in [(bottom00, top00), (top00, a), (a, b), (b, top04)] {
        s.triangle(bottom04, from, to);
    }
    for (from, to) in [(bottom14, top14), (top14, c), (c, top10)] {
        s.triangle(bottom10, from, to);
    }
    let needle = [a, b, c].map(|p| s.vertex(p));
    (s, needle)
}

/// The minimum dihedral of the one tetrahedron on surface triangle `face`.
fn angle_on_face(mesh: &SolidMesh, face: [u32; 3]) -> f64 {
    let tets: Vec<&[u32; 4]> = mesh
        .tets
        .iter()
        .filter(|tet| face.iter().all(|v| tet.contains(v)))
        .collect();
    assert_eq!(tets.len(), 1, "one tetrahedron lies on a surface triangle");
    min_dihedral(mesh, tets[0])
}

#[test]
fn size_points_no_longer_flatten_the_tetrahedron_on_a_needle_surface_triangle() {
    // A tetrahedron on a needle triangle of a thin slab stays too long for the size for ever, as
    // its surface edges cannot be split, and with its circumcentre outside the slab its point is
    // its centroid, a quarter of its height over the needle. Point after point the tetrahedron on
    // the needle was flattened until its four corners were coplanar to the last bit: 0.0 degrees
    // here without the rule. The size stage now refuses a point inside the tetrahedron that asked
    // for it when the point would put a poor tetrahedron worse than that one on its surface face,
    // and leaves it oversized. Measured with the rule: 1.48e-2 degrees. The claim is only that it
    // is not flat, because a neighbour's point whose cavity reaches the needle is not checked
    // (`interior.rs`) and can still lower it.
    let (surface, needle) = needle_slab(15.0, 0.13, 0.2);
    let (refined, report) = run_refine(
        recovered(&surface),
        constant_size(0.3),
        RefineOptions::default(),
    )
    .expect("refines");
    assert!(
        report.size_points_left_oversized > 0,
        "the rule leaves tetrahedra on the needle slab oversized"
    );
    let angle = angle_on_face(&refined, needle);
    assert!(
        angle > 0.0,
        "the tetrahedron on the needle is flat ({angle:e} degrees)"
    );
    // Without the rule the mesh's worst element is exactly flat here as well (0.0 degrees; 6.9e-4
    // with it). `> 0` is categorical, flat or not, not a bound fitted to the output.
    let worst = refined
        .tets
        .iter()
        .map(|tet| min_dihedral(&refined, tet))
        .fold(180.0_f64, f64::min);
    assert!(
        worst > 0.0,
        "the mesh's worst element is flat ({worst:e} degrees)"
    );
}
