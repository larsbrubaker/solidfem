//! The corpus solve record against its production code: a remeshed, volume-meshed plate solves
//! under the corpus's bending load case with its reactions balancing the load and a deflection
//! inside the beam-theory bracket, a part with too few nodes to hold or no face to load is a named
//! outcome rather than a guess, the study's errors and the time cap are named apart, and solve
//! records sort into the worst-dihedral buckets the summary reads iterations by. A solve that
//! "converged" to a wrong answer would read as a mesh that works, so the checks are physical.

use solidfem::elastic::ModelError;
use solidfem::mesh::{QuadraticTetMesh, TetMesh};
use solidfem::solve::PcgError;
use solidfem_bench::benchmarks::verdict::LOAD_BALANCE_LIMIT;
use solidfem_bench::corpus::measure::measure_input;
use solidfem_bench::corpus::record::{
    by_dihedral_bucket, field, field_f64, parse_line, solve_fields, to_line,
};
use solidfem_bench::corpus::remesh_trial::remesh_and_measure;
use solidfem_bench::corpus::solve_trial::{
    bending_case, solve_and_check, solve_outcome_name, LoadCaseError, SolveFailure, SolvedPart,
    LOAD_STRAIN, STEEL,
};
use solidfem_bench::corpus::volume_trial::volume_and_measure;

use crate::metrics::cuboid;

/// One 10-node tetrahedron on the corners `corners`, positively wound.
fn one_tet(corners: [[f64; 3]; 4]) -> QuadraticTetMesh {
    let linear = TetMesh::new(corners.to_vec(), vec![[0, 1, 2, 3]]).expect("a valid tetrahedron");
    QuadraticTetMesh::from_linear(&linear)
}

#[test]
fn a_meshed_plate_solves_with_balanced_reactions_and_a_beam_deflection() {
    let (length, width, thickness) = (40.0, 20.0, 4.0);
    let soup = cuboid([length, width, thickness], [0.0, 0.0, 0.0]);
    let metrics = measure_input(&soup);
    let remesh = remesh_and_measure(&soup, &metrics);
    let surface = remesh.outcome.expect("a plate remeshes").surface;
    let volume = volume_and_measure(&surface, remesh.target_edge_length);
    let meshed = volume.outcome.expect("a plate volume-meshes");

    let case = bending_case(&meshed.mesh).expect("a plate has both ends");
    assert_eq!(case.length_axis, 0, "held and loaded along its length");
    assert_eq!(case.force_axis, 2, "pushed through its thickness");
    let trial = solve_and_check(&meshed.mesh);
    assert_eq!(
        solve_outcome_name(&trial.outcome),
        "ok",
        "{:?}",
        trial.outcome
    );
    let solved = trial.outcome.as_ref().expect("checked above");
    // `ok` already holds the reactions to LOAD_BALANCE_LIMIT (1e-6, derived in `verdict.rs`):
    // the imbalance is the solve residual summed over the free values, about sqrt(n) times the
    // attained relative residual of 1e-9 against the load, so for this plate's few thousand
    // unknowns it is near 1e-7 at worst. Asserted here so a change to `ok` cannot drop it.
    assert!(solved.imbalance <= LOAD_BALANCE_LIMIT, "{solved:?}");
    let report = trial.report.expect("a solved study has a report");
    assert!(
        report.iterations > 0 && report.relative_residual <= 1e-9,
        "{report:?}"
    );
    assert_eq!(trial.unknowns, report.free_dofs);

    // The corpus load scales with the part: E * D^2 * LOAD_STRAIN.
    let diagonal = (length * length + width * width + thickness * thickness).sqrt();
    let force = STEEL.youngs_modulus * diagonal * diagonal * LOAD_STRAIN;
    assert!((case.force / force - 1.0).abs() < 1e-12, "{}", case.force);

    // Cantilever bending, F a^3 / (3 E I). The surface nodes of the first 5% of the length are
    // held and the last 5% is loaded, so the arm from the held slab's edge to the nearest loaded
    // point is 36 and the tip is at most 40 from the fixed end. The lower bound is the shorter
    // arm with the plate's anticlastic stiffening, at most 1/(1 - nu^2), taken in full; the
    // upper bound is the longer arm from a rigid clamp with shear deflection F L / (k G A),
    // k = 5/6, added. A displacement element is never softer than the solid, and the
    // surface-only hold is no stiffer than a rigid clamp at the slab's edge.
    let inertia = width * thickness * thickness * thickness / 12.0;
    let e = STEEL.youngs_modulus;
    let nu = STEEL.poisson_ratio;
    let shear_modulus = e / (2.0 * (1.0 + nu));
    let low = force * 36.0_f64.powi(3) * (1.0 - nu * nu) / (3.0 * e * inertia);
    let high = force * 40.0_f64.powi(3) / (3.0 * e * inertia)
        + force * 40.0 / (5.0 / 6.0 * shear_modulus * width * thickness);
    let deflection = solved.max_displacement_over_diagonal * diagonal;
    assert!(
        deflection >= low && deflection <= high,
        "deflection {deflection} outside [{low}, {high}]"
    );

    let fields = solve_fields("plate", &trial, meshed.min_dihedral_degrees);
    let record = parse_line(&to_line(&fields));
    assert_eq!(record.len(), fields.len(), "no field lost or merged");
    assert_eq!(field(&record, "kind"), "solve");
    assert_eq!(field(&record, "status"), "ok");
    assert_eq!(field(&record, "iters"), report.iterations.to_string());
    assert_eq!(field_f64(&record, "imbalance"), Some(solved.imbalance));
    assert_eq!(
        field_f64(&record, "mindih"),
        Some(meshed.min_dihedral_degrees)
    );
    assert!(field_f64(&record, "mg_bytes").is_some_and(|b| b > 0.0));
}

#[test]
fn a_part_with_one_node_at_its_low_end_has_nothing_to_hold() {
    // Long along x with one corner alone at the low end: only that corner is in the held slab.
    let mesh = one_tet([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 1.0, 0.0],
        [10.0, 0.0, 1.0],
    ]);
    assert_eq!(
        bending_case(&mesh).map(|_| ()),
        Err(LoadCaseError::TooFewHeldNodes { found: 1 })
    );
    let trial = solve_and_check(&mesh);
    assert_eq!(
        solve_outcome_name(&trial.outcome),
        "load-case:TooFewHeldNodes"
    );
    assert!(trial.report.is_none(), "nothing was solved");
    let record = parse_line(&to_line(&solve_fields("needle", &trial, 3.0)));
    assert_eq!(field(&record, "status"), "load-case:TooFewHeldNodes");
    assert_eq!(field(&record, "iters"), "-", "no study, no iterations");
}

#[test]
fn a_part_with_one_node_at_its_high_end_has_nothing_to_load() {
    let mesh = one_tet([
        [10.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    assert_eq!(
        bending_case(&mesh).map(|_| ()),
        Err(LoadCaseError::NoLoadedFace)
    );
    assert_eq!(
        solve_outcome_name(&solve_and_check(&mesh).outcome),
        "load-case:NoLoadedFace"
    );
}

#[test]
fn study_errors_the_time_cap_and_failed_checks_are_named_apart() {
    let name = |failure| solve_outcome_name(&Err(failure));
    assert_eq!(
        name(SolveFailure::Model(ModelError::Solver(
            PcgError::NotConverged {
                iterations: 10,
                relative_residual: 1e-3
            }
        ))),
        "solve:not-converged"
    );
    assert_eq!(
        name(SolveFailure::Model(ModelError::Solver(
            PcgError::NotPositiveDefinite {
                iteration: 3,
                curvature: -1.0
            }
        ))),
        "solve:Solver.NotPositiveDefinite"
    );
    assert_eq!(
        name(SolveFailure::Model(ModelError::NoFixedNodes)),
        "solve:NoFixedNodes"
    );
    assert_eq!(name(SolveFailure::TimeCap), "time-cap:solve");
    let measured = SolvedPart {
        imbalance: 1e-3,
        max_displacement_over_diagonal: 1e3,
    };
    assert_eq!(name(SolveFailure::Unphysical(measured)), "solve:unphysical");
    assert_eq!(name(SolveFailure::Imbalanced(measured)), "solve:imbalanced");
}

#[test]
fn solve_records_sort_into_dihedral_buckets_at_their_lower_edges() {
    let records: Vec<_> = ["0.05", "0.1", "0.99", "1", "4.9", "5", "30", "nan"]
        .iter()
        .map(|d| parse_line(&format!("kind=solve mindih={d} iters=7")))
        .collect();
    let counts: Vec<(&str, usize)> = by_dihedral_bucket(&records)
        .into_iter()
        .map(|(label, inside)| (label, inside.len()))
        .collect();
    assert_eq!(
        counts,
        vec![
            ("< 0.1 deg", 1),
            ("0.1 to 1 deg", 2),
            ("1 to 5 deg", 2),
            (">= 5 deg", 2),
        ],
        "each bucket holds its lower edge; an unreadable angle is in none"
    );
}
