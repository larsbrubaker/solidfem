//! The corpus volume record against its production code: a remeshed plate volume-meshes to
//! 10-node elements and its record reads back with the same values, a surface the tetrahedral
//! mesher refuses is named by its error's variant, the 10-node upgrade's and the time cap's
//! failures are named apart from the mesher's, and the summary counts outcomes most frequent
//! first. The corpus's failure table is read by these names, so a name that drifted or merged two
//! classes would misdirect the work on the mesher.

use solidfem::mesh::MeshError;
use solidfem::meshing::MeshingError;
use solidfem::tetra::StepStatus;
use solidfem_bench::benchmarks::volume::{VolumeFailure, VolumeStage};
use solidfem_bench::corpus::measure::measure_input;
use solidfem_bench::corpus::record::{
    counts_by, field, field_f64, parse_line, to_line, volume_fields, volume_lost_fields,
};
use solidfem_bench::corpus::remesh_trial::remesh_and_measure;
use solidfem_bench::corpus::volume_trial::{
    choose_volume_edge, outcome_name, volume_and_measure, VolumeEdgeSource, VolumePart, TET_BUDGET,
};

use crate::metrics::cuboid;

/// Every field survives the line format unchanged.
fn assert_round_trip(fields: &[(String, String)]) {
    let record = parse_line(&to_line(fields));
    assert_eq!(record.len(), fields.len(), "no field lost or merged");
    for (key, value) in fields {
        assert_eq!(record.get(key), Some(value), "field {key}");
    }
}

#[test]
fn a_remeshed_plate_volume_meshes_and_its_record_reads_back() {
    let soup = cuboid([40.0, 20.0, 4.0], [0.0, 0.0, 0.0]);
    let metrics = measure_input(&soup);
    let remesh = remesh_and_measure(&soup, &metrics);
    let part = remesh.outcome.expect("a plate remeshes");
    let trial = volume_and_measure(&part.surface, remesh.target_edge_length);
    assert_eq!(outcome_name(&trial.outcome), "ok", "{:?}", trial.outcome);
    let measured = trial.outcome.as_ref().expect("checked above");
    assert!(measured.tets > 0);
    // A 10-node mesh has every corner plus one node per edge, so more nodes than corners and,
    // for any connected mesh of more than one element, more nodes than elements.
    assert!(measured.nodes > measured.tets, "{measured:?}");
    // A tetrahedron's smallest dihedral angle is at most that of the regular one, acos(1/3).
    let regular = (1.0_f64 / 3.0).acos().to_degrees();
    assert!(
        measured.min_dihedral_degrees > 0.0 && measured.min_dihedral_degrees <= regular,
        "{measured:?}"
    );
    assert!(measured.under_one_degree <= measured.under_five_degrees);
    assert!(measured.under_five_degrees <= measured.tets);

    let fields = volume_fields("plate", &trial);
    assert_round_trip(&fields);
    let record = parse_line(&to_line(&fields));
    assert_eq!(field(&record, "kind"), "volume");
    assert_eq!(field(&record, "status"), "ok");
    // Shortest round-trip text: the number read back is the number measured, bit for bit.
    assert_eq!(
        field_f64(&record, "mindih"),
        Some(measured.min_dihedral_degrees)
    );
    assert_eq!(field(&record, "tets"), measured.tets.to_string());
    assert_eq!(field(&record, "nodes"), measured.nodes.to_string());
}

#[test]
fn an_open_surface_is_named_by_the_mesher_error_and_carries_no_mesh_fields() {
    let mut soup = cuboid([10.0, 10.0, 10.0], [0.0, 0.0, 0.0]);
    soup.triangles.truncate(10);
    let trial = volume_and_measure(&soup, 2.0);
    assert!(
        matches!(trial.outcome, Err(VolumeFailure::Surface(_))),
        "{:?}",
        trial.outcome
    );
    assert_eq!(outcome_name(&trial.outcome), "tetra:SurfaceNotClosed");
    let fields = volume_fields("open box", &trial);
    assert_round_trip(&fields);
    let record = parse_line(&to_line(&fields));
    assert_eq!(
        field(&record, "name"),
        "open_box",
        "spaces cannot split a record"
    );
    assert_eq!(field(&record, "status"), "tetra:SurfaceNotClosed");
    assert_eq!(
        field(&record, "tets"),
        "-",
        "a refusal has no mesh to count"
    );
}

#[test]
fn upgrade_and_time_cap_failures_are_named_apart_from_the_mesher() {
    let upgrade = |error| outcome_name(&Err::<VolumePart, _>(VolumeFailure::Upgrade(error)));
    assert_eq!(
        upgrade(MeshingError::NonPositiveJacobian {
            tet: 3,
            point: 1,
            determinant: -1e-20
        }),
        "quadratic:NonPositiveJacobian"
    );
    assert_eq!(
        upgrade(MeshingError::Mesh(MeshError::Empty)),
        "quadratic:Mesh.Empty"
    );
    let stopped = |stage| {
        outcome_name(&Err::<VolumePart, _>(VolumeFailure::Stopped {
            stage,
            status: StepStatus::Cancelled,
        }))
    };
    assert_eq!(stopped(VolumeStage::Recovery), "time-cap:recovery");
    assert_eq!(stopped(VolumeStage::Refinement), "time-cap:refinement");
}

#[test]
fn a_child_lost_in_the_volume_stage_has_its_own_record() {
    let fields = volume_lost_fields("part 7", "timeout");
    assert_round_trip(&fields);
    let record = parse_line(&to_line(&fields));
    assert_eq!(field(&record, "kind"), "volume");
    assert_eq!(field(&record, "status"), "child:timeout");
}

#[test]
fn outcomes_count_most_frequent_first_with_ties_by_name() {
    let records: Vec<_> = [
        "tetra:B",
        "ok",
        "tetra:B",
        "quadratic:A",
        "ok",
        "tetra:B",
        "tetra:A",
    ]
    .iter()
    .map(|status| parse_line(&format!("kind=volume status={status}")))
    .collect();
    let counts = counts_by(&records, "status");
    let expected = [
        ("tetra:B", 3),
        ("ok", 2),
        ("quadratic:A", 1),
        ("tetra:A", 1),
    ];
    assert_eq!(
        counts,
        expected.map(|(s, n)| (s.to_string(), n)).to_vec(),
        "count descending, then name ascending"
    );
}

#[test]
fn a_large_part_at_a_fine_remesh_edge_is_held_to_the_tet_budget() {
    let (edge, source) = choose_volume_edge(1.0, 1.0e9);
    assert_eq!(source, VolumeEdgeSource::TetBudget);
    // A fill of regular tetrahedra of this edge is the budget, to the cube root's rounding.
    let tets = 6.0 * 2.0_f64.sqrt() * 1.0e9 / (edge * edge * edge);
    assert!((tets / TET_BUDGET - 1.0).abs() < 1e-12, "{tets}");
    assert_eq!(
        choose_volume_edge(1.0, 1000.0),
        (1.0, VolumeEdgeSource::Remesh)
    );
}
