//! The size prediction (`benchmarks/sizing.rs`) against meshes the production pipeline makes: a
//! thin plate with its walls resolved and a stocky block, each remeshed and
//! volume-meshed at one edge length, land within the band the prediction was fitted to; the
//! corpus's chosen edge for a thin shell lands its finished mesh near the budget and records its
//! walls as under-resolved; the inverse finds the edge whose prediction is the budget. A caller
//! that trusted a prediction several times too low would run out of memory mid-solve.
//!
//! Every band is the fit's own spread on 73 real parts: measured over predicted from its minimum
//! (0.885) to its 90th percentile (1.128) for boxes whose walls get their two layers. The upper
//! tail beyond that came from surfaces whose curved detail or creases forced the remesher to
//! several times the ideal triangle count; flat boxes have none, so they belong inside the bulk
//! of the fit. A shell whose walls the budget leaves short is held to its own measured bound
//! (`UNDER_RESOLVED_SHELL_UPPER`).

use std::time::Duration;

use solidmesh_mesh::AnalysisRemeshError;

use solidfem_bench::benchmarks::sizing::{
    edge_for_unknowns, predicted_tets, predicted_unknowns, wall_resolution, PartSize, SizingError,
    PREDICTION_BAND, UNKNOWNS_PER_TET,
};
use solidfem_bench::corpus::measure::measure_input;
use solidfem_bench::corpus::record::{field, parse_line, remesh_fields, to_line};
use solidfem_bench::corpus::remesh_trial::{
    choose_target, choose_target_within, part_size, remesh_and_measure, remesh_capped, RemeshTrial,
    TargetSource, REMESH_TIME_CAP_SECONDS,
};
use solidfem_bench::corpus::soup::TriSoup;
use solidfem_bench::corpus::volume_trial::volume_and_measure;

use crate::metrics::cuboid;

/// `soup` remeshed and volume-meshed at `edge`: the finished element count.
fn meshed_tets(soup: &TriSoup, edge: f64) -> usize {
    let metrics = measure_input(soup);
    let remeshed = remesh_capped(
        soup,
        &metrics,
        edge,
        Duration::from_secs_f64(REMESH_TIME_CAP_SECONDS),
    )
    .expect("a box remeshes");
    let surface = TriSoup::from_analysis_surface(&remeshed.surface);
    let volume = volume_and_measure(&surface, edge);
    volume.outcome.expect("a box volume-meshes").tets
}

fn assert_in_band(name: &str, measured: f64, predicted: f64) {
    assert_within(name, measured, predicted, PREDICTION_BAND.p90);
}

fn assert_within(name: &str, measured: f64, predicted: f64, upper: f64) {
    let ratio = measured / predicted;
    assert!(
        (PREDICTION_BAND.min..=upper).contains(&ratio),
        "{name}: measured {measured} over predicted {predicted} is {ratio}, outside {:?}",
        PREDICTION_BAND
    );
}

#[test]
fn a_thin_plate_meshes_to_the_predicted_size_with_its_walls_resolved() {
    let soup = cuboid([40.0, 20.0, 2.0], [0.0, 0.0, 0.0]);
    let size = part_size(&measure_input(&soup));
    assert_eq!(size.wall_p5, 2.0, "the plate's thickness is its thin wall");
    // At 1.0 two layers fit through the 2.0 wall exactly.
    assert_eq!(wall_resolution(&size, 1.0), Some(1.0));
    let predicted = predicted_tets(&size, 1.0);
    assert_in_band("plate at 1.0", meshed_tets(&soup, 1.0) as f64, predicted);
}

#[test]
fn a_stocky_block_meshes_to_the_predicted_size() {
    let soup = cuboid([20.0, 20.0, 20.0], [0.0, 0.0, 0.0]);
    let size = part_size(&measure_input(&soup));
    let predicted = predicted_tets(&size, 2.0);
    assert_in_band("block at 2.0", meshed_tets(&soup, 2.0) as f64, predicted);
}

/// A closed box shell: `outer` with a cavity `wall` in from every face, the cavity wound inward.
fn box_shell(outer: [f64; 3], wall: f64) -> TriSoup {
    let mut shell = cuboid(outer, [0.0, 0.0, 0.0]);
    let cavity = cuboid(outer.map(|side| side - 2.0 * wall), [wall, wall, wall]);
    let base = shell.vertices.len() as u32;
    shell.vertices.extend(cavity.vertices);
    shell.triangles.extend(
        cavity
            .triangles
            .iter()
            .map(|t| [t[0] + base, t[2] + base, t[1] + base]),
    );
    shell
}

/// The most a shell whose walls the budget leaves short may come in over the budget's element
/// count. Box shells were measured at 1.158 (20 x 15 x 8 with 0.6 walls) and 0.98 (the synthetic
/// 60 x 40 x 20 shell at the corpus budget); 1.25 holds both with a little room for the
/// mesher's own variation.
/// The fit's whole-spread maximum (2.251) comes from curved detail, a different cause, and would
/// let a prediction 2.2 times low pass.
const UNDER_RESOLVED_SHELL_UPPER: f64 = 1.25;

#[test]
fn the_corpus_edge_for_a_thin_shell_lands_its_mesh_within_a_quarter_over_the_budget() {
    let soup = box_shell([20.0, 15.0, 8.0], 0.6);
    let metrics = measure_input(&soup);
    // A budget the debug build meshes in seconds that leaves this shell's walls as far short of
    // two layers as the corpus's own budget leaves the synthetic thin shell's (wall ratio 2.4).
    let budget = 165_000.0;
    let (edge, source) = choose_target_within(&metrics, budget);
    assert_eq!(source, TargetSource::Budget, "the walls ask for more");
    let ratio = wall_resolution(&part_size(&metrics), edge).expect("the shell has a wall");
    assert!(
        (1.5..=2.5).contains(&ratio),
        "the budget's edge leaves the walls short as the corpus's does: {ratio}"
    );
    // Walls this far short are past the layer shortfalls the fit saw (up to 0.95 of a layer), so
    // the bound is the shells' own, not the fit's p90.
    let tets = meshed_tets(&soup, edge) as f64;
    assert_within(
        "thin shell at the budget",
        tets,
        budget / UNKNOWNS_PER_TET,
        UNDER_RESOLVED_SHELL_UPPER,
    );
}

#[test]
fn the_remesh_record_says_when_the_budget_left_walls_under_resolved() {
    // A plate whose wall asks for a far finer edge than the corpus budget allows.
    let metrics = measure_input(&cuboid([4000.0, 4000.0, 2.0], [0.0, 0.0, 0.0]));
    let (edge, source) = choose_target(&metrics);
    assert_eq!(source, TargetSource::Budget);
    let size = part_size(&metrics);
    let predicted = predicted_tets(&size, edge);
    let trial = RemeshTrial {
        target_edge_length: edge,
        target_source: source,
        predicted_tets: predicted,
        wall_ratio: wall_resolution(&size, edge),
        deviation_bound: 0.5,
        seconds: 0.0,
        outcome: Err(AnalysisRemeshError::Cancelled),
    };
    let record = parse_line(&to_line(&remesh_fields("big plate", &trial)));
    assert_eq!(field(&record, "walls"), "under-resolved");
    let ratio: f64 = field(&record, "wall_ratio").parse().expect("a number");
    assert_eq!(Some(ratio), trial.wall_ratio);
    assert!(ratio > 1.0, "{ratio}");
    let recorded: f64 = field(&record, "pred_tets").parse().expect("a number");
    assert_eq!(recorded, predicted.round_ties_even());

    // A small plate keeps the edge its wall asks for, which is two layers exactly.
    let small = cuboid([40.0, 20.0, 2.0], [0.0, 0.0, 0.0]);
    let small_trial = remesh_and_measure(&small, &measure_input(&small));
    assert_eq!(small_trial.target_source, TargetSource::ThinnestWall);
    let record = parse_line(&to_line(&remesh_fields("plate", &small_trial)));
    assert_eq!(field(&record, "walls"), "resolved");
}

#[test]
fn the_budget_edge_predicts_the_budget_and_bad_inputs_are_refused() {
    let part = PartSize {
        volume: 1.0e6,
        area: 6.0e4,
        wall_p5: 3.0,
    };
    let edge = edge_for_unknowns(&part, 1.0e6).expect("a valid part and budget");
    let at = predicted_unknowns(&part, edge);
    // The bisection pins the edge to the last bits, so the prediction meets the budget to the
    // relative rounding of a few dozen operations.
    assert!(at <= 1.0e6 && at > 1.0e6 * (1.0 - 1e-9), "{at}");
    assert_eq!(
        edge_for_unknowns(&part, 0.0),
        Err(SizingError::InvalidBudget(0.0))
    );
    assert!(matches!(
        edge_for_unknowns(
            &PartSize {
                volume: f64::NAN,
                ..part
            },
            1.0e6
        ),
        Err(SizingError::InvalidPart { .. })
    ));
    let empty = PartSize {
        volume: 0.0,
        area: 0.0,
        wall_p5: f64::NAN,
    };
    assert_eq!(edge_for_unknowns(&empty, 1.0e6), Ok(0.0));
}

#[test]
fn a_budget_no_finite_edge_can_meet_is_refused_rather_than_searched_forever() {
    // The largest finite area over the smallest positive budget: the starting edge itself is
    // past what an f64 holds.
    let huge = PartSize {
        volume: f64::MAX,
        area: f64::MAX,
        wall_p5: 1.0,
    };
    assert_eq!(
        edge_for_unknowns(&huge, f64::MIN_POSITIVE),
        Err(SizingError::NoFiniteEdge {
            budget: f64::MIN_POSITIVE
        })
    );
    // A tiny budget on an ordinary part still has a finite edge, and it meets the budget.
    let part = PartSize {
        volume: 1.0e3,
        area: 6.0e2,
        wall_p5: 1.0,
    };
    let edge = edge_for_unknowns(&part, 1.0e-6).expect("a finite edge exists");
    assert!(edge.is_finite() && predicted_unknowns(&part, edge) <= 1.0e-6);
}

#[test]
fn a_ratio_that_is_not_a_number_reads_as_walls_unread() {
    // A part whose volume is not a number gets a NaN edge, so its wall ratio is NaN too: that
    // says nothing about its walls and must not read as under-resolved.
    let trial = RemeshTrial {
        target_edge_length: f64::NAN,
        target_source: TargetSource::Budget,
        predicted_tets: f64::NAN,
        wall_ratio: Some(f64::NAN),
        deviation_bound: f64::NAN,
        seconds: 0.0,
        outcome: Err(AnalysisRemeshError::Cancelled),
    };
    let record = parse_line(&to_line(&remesh_fields("nan part", &trial)));
    assert_eq!(field(&record, "walls"), "unread");
}
