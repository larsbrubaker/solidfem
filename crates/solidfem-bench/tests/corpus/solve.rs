//! The corpus solve record against its production code: a remeshed, volume-meshed plate solves
//! under the corpus's bending load case with its reactions balancing the load and a deflection
//! inside the beam-theory bracket, a part with too few nodes to hold or no face to load is a named
//! outcome rather than a guess, the study's errors and the time cap are named apart, and solve
//! records sort into the worst-dihedral buckets the summary reads iterations by. A plate solves in
//! each multigrid precision, the stage's one-iteration stepping giving the same solve as coarser
//! slices, and a child lost partway through a three-precision solve keeps the records it wrote and
//! gets one lost record per precision it did not reach. A solve that
//! "converged" to a wrong answer would read as a mesh that works, so the checks are physical.

use solidfem::elastic::{
    Model, ModelError, PcgOptions, Precision, PreconditionerChoice, Quadrature, StaticStudy,
    StepOutcome, StudyOptions,
};
use solidfem::mesh::{QuadraticTetMesh, TetMesh};
use solidfem::solve::PcgError;
use solidfem_bench::benchmarks::verdict::LOAD_BALANCE_LIMIT;
use solidfem_bench::corpus::measure::measure_input;
use solidfem_bench::corpus::record::{
    by_dihedral_bucket, field, field_f64, lost_child_records, parse_line, solve_fields, to_line,
    Record,
};
use solidfem_bench::corpus::remesh_trial::remesh_and_measure;
use solidfem_bench::corpus::solve_trial::{
    bending_case, precision_name, solve_and_check, solve_and_check_in, solve_outcome_name,
    LoadCase, LoadCaseError, SolveFailure, SolvedPart, LOAD_STRAIN, STEEL,
};
use solidfem_bench::corpus::volume_trial::volume_and_measure;

use solidfem_bench::gate_diagnostics::multigrid_options;

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

/// The 40 x 20 x 4 plate of the first test, remeshed and volume-meshed as the corpus does.
fn meshed_plate() -> QuadraticTetMesh {
    let soup = cuboid([40.0, 20.0, 4.0], [0.0, 0.0, 0.0]);
    let metrics = measure_input(&soup);
    let remesh = remesh_and_measure(&soup, &metrics);
    let surface = remesh.outcome.expect("a plate remeshes").surface;
    volume_and_measure(&surface, remesh.target_edge_length)
        .outcome
        .expect("a plate volume-meshes")
        .mesh
}

/// The stage's study of `mesh` built here, to `tolerance`, stepped `slice` iterations at a time:
/// (iterations, final relative residual).
fn reference_solve(
    mesh: &QuadraticTetMesh,
    case: &LoadCase,
    precision: Precision,
    tolerance: f64,
    slice: usize,
) -> (u64, f64) {
    let mut model = Model::new(mesh.clone(), STEEL);
    model.set_quadrature(Quadrature::FourPoint);
    for node in &case.supports.fixed {
        model.fix_node(*node);
    }
    for (face, traction) in &case.supports.tractions {
        model.add_traction(*face, *traction);
    }
    let options = StudyOptions {
        pcg: PcgOptions {
            tolerance,
            ..PcgOptions::default()
        },
        preconditioner: PreconditionerChoice::Multigrid(solidfem::elastic::MultigridOptions {
            precision,
            ..multigrid_options()
        }),
        ..StudyOptions::default()
    };
    let mut study = StaticStudy::new(model, options, &mut || false).expect("valid");
    loop {
        match study.step(slice, &mut || false).expect("solves") {
            StepOutcome::Converged(_) => break,
            StepOutcome::Running(_) => {}
            StepOutcome::Cancelled(_) => panic!("nothing cancelled"),
        }
    }
    let report = study.report();
    (report.iterations, report.relative_residual)
}

/// Solves the plate in `precision` as the stage does and checks the record; returns the trial's
/// iterations, final residual and 1e-6 crossing.
fn stage_solve(mesh: &QuadraticTetMesh, precision: Precision) -> (u64, f64, u64) {
    let trial = solve_and_check_in(mesh, precision);
    assert_eq!(trial.precision, precision);
    assert_eq!(
        solve_outcome_name(&trial.outcome),
        "ok",
        "{precision:?}: {:?}",
        trial.outcome
    );
    let report = trial.report.as_ref().expect("a solved study has a report");
    assert!(
        report.relative_residual <= 1e-9 || report.at_rounding_floor,
        "{precision:?}: stopped at {} short of its floor",
        report.relative_residual
    );
    // `below_1e6` reads the running residual estimate, so it comes no later than the end.
    let crossing = trial.below_mark.expect("the residual passed 1e-6");
    assert!(crossing <= report.iterations, "{precision:?}: {crossing}");
    let record = parse_line(&to_line(&solve_fields("plate", &trial, 10.0)));
    assert_eq!(field(&record, "precision"), precision_name(precision));
    assert_eq!(field(&record, "below_1e6"), crossing.to_string());
    assert_eq!(
        field(&record, "floor"),
        report.at_rounding_floor.to_string()
    );
    (report.iterations, report.relative_residual, crossing)
}

#[test]
fn a_plate_solves_in_the_default_precision_as_a_coarser_sliced_solve_does() {
    // The stage runs the library's default, an f32-stored hierarchy with f64 arithmetic.
    let precision = multigrid_options().precision;
    assert_eq!(
        precision,
        Precision::SingleStorage,
        "the stage's default precision"
    );
    sliced_solve_matches(precision);
}

#[test]
fn a_plate_solves_in_double_as_a_coarser_sliced_solve_does() {
    sliced_solve_matches(Precision::Double);
}

/// The stage steps one iteration at a time to see the 1e-6 crossing; a solve's bits do not depend
/// on slicing, so seven at a time stops at the same iteration and residual.
fn sliced_solve_matches(precision: Precision) {
    let mesh = meshed_plate();
    let case = bending_case(&mesh).expect("a plate has both ends");
    let (iterations, residual, _) = stage_solve(&mesh, precision);
    let (sliced_iterations, sliced_residual) = reference_solve(&mesh, &case, precision, 1e-9, 7);
    assert_eq!(iterations, sliced_iterations, "{precision:?}");
    assert_eq!(
        residual.to_bits(),
        sliced_residual.to_bits(),
        "{precision:?}"
    );
}

#[test]
fn the_plate_crosses_1e6_where_a_solve_to_that_tolerance_stops() {
    let mesh = meshed_plate();
    let case = bending_case(&mesh).expect("a plate has both ends");
    let (_, _, crossing) = stage_solve(&mesh, Precision::Double);
    // A solve to a 1e-6 tolerance stops where the running estimate first crossed 1e-6 when the
    // true residual there agrees, as it does on this plate.
    let (to_mark, _) = reference_solve(&mesh, &case, Precision::Double, 1e-6, 7);
    assert_eq!(crossing, to_mark);
}

#[test]
fn a_plate_solves_with_an_f32_stored_hierarchy() {
    stage_solve(&meshed_plate(), Precision::SingleStorage);
}

#[test]
fn a_plate_solves_with_an_f32_cycle() {
    stage_solve(&meshed_plate(), Precision::Single);
}

#[test]
fn a_child_lost_partway_through_three_precisions_keeps_its_records_and_names_the_rest() {
    let lines = [
        "name=p.stl kind=remesh status=ok",
        "name=p.stl kind=volume status=ok",
        "name=p.stl kind=solve status=ok precision=double iters=40",
    ];
    let partial: Vec<Record> = lines.iter().map(|line| parse_line(line)).collect();
    let all = ["double", "single-storage", "single"];
    let lost = lost_child_records(&partial, &all, "timeout").expect("two solves were lost");
    let lost: Vec<Record> = lost
        .iter()
        .map(|fields| parse_line(&to_line(fields)))
        .collect();
    let named: Vec<(&str, &str, &str)> = lost
        .iter()
        .map(|r| (field(r, "kind"), field(r, "status"), field(r, "precision")))
        .collect();
    assert_eq!(
        named,
        vec![
            ("solve", "child:timeout", "single-storage"),
            ("solve", "child:timeout", "single"),
        ]
    );
    assert!(lost.iter().all(|r| field(r, "name") == "p.stl"));
    // With every expected solve present nothing is lost in a stage, and a child lost after a
    // remesh with no volume record loses the volume stage.
    assert_eq!(lost_child_records(&partial, &["double"], "timeout"), None);
    let before_volume: Vec<Record> = partial[..1].to_vec();
    let lost = lost_child_records(&before_volume, &all, "crashed").expect("the volume was lost");
    let lost = parse_line(&to_line(&lost[0]));
    assert_eq!(
        (field(&lost, "kind"), field(&lost, "status")),
        ("volume", "child:crashed")
    );
}
