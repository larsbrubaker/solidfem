//! The corpus remesh stage against its production code: a synthetic part comes back from
//! `remesh_for_analysis` as a closed, consistently wound surface the corpus's own tests find
//! neither crossed nor folded; a fold is counted exactly; a refusal is recorded by its variant's
//! name; and an input that is not a solid is left out of the sample by name rather than handed
//! to the remesher.

use solidfem_bench::corpus::measure::{measure_input, sample_exclusion};
use solidfem_bench::corpus::parts::all_parts;
use solidfem_bench::corpus::record::{input_fields, parse_line, remesh_fields, to_line};
use solidfem_bench::corpus::remesh_trial::remesh_and_measure;
use solidfem_bench::corpus::self_intersect::count_folds;
use solidfem_bench::corpus::soup::TriSoup;
use solidmesh_mesh::AnalysisRemeshError;

use crate::metrics::cuboid;

#[test]
fn a_synthetic_part_remeshes_closed_and_uncrossed() {
    let parts = all_parts().expect("the synthetic parts build");
    let part = parts
        .iter()
        .find(|p| p.name == "extruded-curves")
        .expect("extruded-curves is a synthetic part");
    let soup = TriSoup::from_mesh(&part.mesh);
    let metrics = measure_input(&soup);
    assert_eq!(sample_exclusion(&metrics), None, "the part is a solid");
    let trial = remesh_and_measure(&soup, &metrics);
    let remeshed = trial.outcome.as_ref().expect("extruded-curves remeshes");
    assert!(remeshed.closed && remeshed.manifold_and_oriented);
    assert!(!remeshed.self_intersection.capped, "every pair was tested");
    assert_eq!(remeshed.self_intersection.pairs, 0);
    assert_eq!(remeshed.folds, 0);
    let record = parse_line(&to_line(&remesh_fields(part.name, &trial)));
    assert_eq!(record.get("status").map(String::as_str), Some("ok"));
    assert_eq!(record.get("xpairs_after").map(String::as_str), Some("0"));
    assert_eq!(record.get("folds_after").map(String::as_str), Some("0"));
}

/// Two triangles on the edge from the origin to (1, 0, 0), the second's far corner at `d`.
fn hinge(d: [f64; 3]) -> TriSoup {
    TriSoup {
        vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], d],
        triangles: vec![[0, 1, 2], [1, 0, 3]],
    }
}

#[test]
fn a_fold_is_counted_exactly_and_a_flat_or_bent_hinge_is_not() {
    // Folded flat: both far corners on the +y side of the edge, in one plane.
    assert_eq!(count_folds(&hinge([0.5, 0.5, 0.0])), 1);
    // The same plane with the corners on opposite sides: an ordinary flat pair.
    assert_eq!(count_folds(&hinge([0.5, -0.5, 0.0])), 0);
    // Off the plane by one step of a coordinate near 1: a crease, not a fold.
    let up = f64::EPSILON;
    assert_eq!(count_folds(&hinge([0.5, 0.5, up])), 0);
    // A cube's faces are pairs of coplanar triangles on opposite sides of their diagonal.
    assert_eq!(count_folds(&cuboid([4.0, 4.0, 4.0], [0.0, 0.0, 0.0])), 0);
}

/// Two cubes meeting at one corner: every edge has two faces wound opposite ways and the volume
/// is positive, so the corpus's reading lets the part in, but the shared corner joins two
/// separate fans of faces, which the remesher refuses by name.
fn cubes_touching_at_a_corner() -> TriSoup {
    let mut soup = cuboid([10.0, 10.0, 10.0], [0.0, 0.0, 0.0]);
    let other = cuboid([10.0, 10.0, 10.0], [10.0, 10.0, 10.0]);
    // The second cube's corner 0 is the first cube's corner 6, (10, 10, 10).
    let shared = 6_u32;
    let base = soup.vertices.len() as u32 - 1;
    soup.vertices.extend(other.vertices.iter().skip(1).copied());
    soup.triangles.extend(
        other
            .triangles
            .iter()
            .map(|t| t.map(|v| if v == 0 { shared } else { base + v })),
    );
    soup
}

#[test]
fn a_refusal_is_recorded_by_its_name() {
    let soup = cubes_touching_at_a_corner();
    let metrics = measure_input(&soup);
    assert_eq!(
        sample_exclusion(&metrics),
        None,
        "the corpus reads it closed"
    );
    let trial = remesh_and_measure(&soup, &metrics);
    assert!(
        matches!(
            trial.outcome,
            Err(AnalysisRemeshError::NonManifoldInput {
                non_manifold_vertices: 1,
                ..
            })
        ),
        "{:?}",
        trial.outcome.as_ref().err()
    );
    let record = parse_line(&to_line(&remesh_fields("corner", &trial)));
    assert_eq!(
        record.get("status").map(String::as_str),
        Some("remesh:NonManifoldInput")
    );
    assert!(record.contains_key("detail"), "the measured values travel");
    assert!(!record.contains_key("tris_after"), "no surface fields");
}

#[test]
fn an_open_or_inside_out_input_is_left_out_of_the_sample_by_name() {
    let mut open = cuboid([4.0, 4.0, 4.0], [0.0, 0.0, 0.0]);
    open.triangles.pop();
    let record = parse_line(&to_line(&input_fields("open", &measure_input(&open))));
    assert_eq!(record.get("sample").map(String::as_str), Some("input:open"));

    let mut inside_out = cuboid([4.0, 4.0, 4.0], [0.0, 0.0, 0.0]);
    for t in &mut inside_out.triangles {
        t.swap(1, 2);
    }
    let metrics = measure_input(&inside_out);
    assert_eq!(sample_exclusion(&metrics), Some("non-positive-volume"));

    let solid = cuboid([4.0, 4.0, 4.0], [0.0, 0.0, 0.0]);
    let record = parse_line(&to_line(&input_fields("solid", &measure_input(&solid))));
    assert_eq!(record.get("sample").map(String::as_str), Some("in"));
}
