//! One linear static solve of a volume-meshed corpus part and what came of it: whether the meshes
//! the corpus produces give an answer, not only whether they exist. The solve goes through the
//! production study (`StaticStudy` with the multigrid preconditioner the benchmarks use), and the
//! answer is checked two ways that do not trust the solver's own report.
//!
//! **The load case needs no knowledge of the part.** Along the longest axis of the part's
//! bounding box (ties to the lower axis index), every surface node within [`SLAB_FRACTION`] of
//! that length of the lowest extent is held; every boundary face whose three corners lie within
//! the same fraction of the highest extent carries a uniform traction across the length, along
//! the shortest of the other two axes (ties to the lower index), so the part bends. The traction
//! is spread by area, not as equal nodal forces, so the loaded end is not a set of point loads
//! whose local displacement grows without bound as the mesh is refined. A part with fewer than
//! [`MIN_HELD_NODES`] held nodes or no loaded face is its own named outcome, never a guess.
//!
//! **The load scales with the part.** Corpus files come in unknown units (millimetres, inches,
//! metres), so the total force is `E * D^2 * LOAD_STRAIN` for the part's bounding-box diagonal
//! `D` and the fixed material [`STEEL`]: under linear elasticity a part and its scaled copy then
//! deflect by the same fraction of their size, and the displacement check reads the same in any
//! unit. For a 100 mm part in millimetres and megapascals that is 2.1 N.
//!
//! **The two physical checks.** The reactions are integrated from the recovered stresses at the
//! held nodes and summed against the applied load from each loaded face's own area
//! (`benchmarks::pipeline::load_balance`, the benchmarks' own check), so a wrong answer that the
//! solver reports as converged still shows as an imbalance over [`LOAD_BALANCE_LIMIT`]. The
//! largest displacement is read against the diagonal: a non-finite one, or one over
//! [`UNPHYSICAL_DIAGONALS`] diagonals, is `solve:unphysical`. With [`LOAD_STRAIN`] at 1e-9, a
//! cantilever of square section deflects `4e-9 (L/h)^4` of its length, so ten diagonals needs a
//! needle over 200 times longer than thick; a mesh that lets a near-flat element move freely
//! does not need one.
//!
//! The stage is cancelled at [`SOLVE_TIME_CAP_SECONDS`] (assembly, multigrid setup, iterations and
//! stress recovery together), which is its own named outcome.
//!
//! **Precision.** The multigrid hierarchy is stored and applied in the [`Precision`] chosen by
//! [`PRECISION_VARIABLE`] (unset is the library default, `single-storage`; `double`; `single`; or
//! `all`, which solves each meshed part once in each, back to back, so the remesh and volume
//! stages run once).
//! Each trial records the iteration at which the relative residual first fell to
//! [`RESIDUAL_MARK`], so one run also says what a looser tolerance would cost. The solve runs one
//! iteration per `step` to see that crossing; slicing never changes a solve's bits.
//!
//! **Diagnostics.** [`HISTORY_VARIABLE`] prints the study's report after setup and the residual
//! every that many iterations, to stderr, and [`SOLVE_CAP_VARIABLE`] replaces the time cap, to see
//! how a part that the cap stops would have gone on ([`SolveDiagnostics`]).
//!
//! **Threads.** [`THREADS_VARIABLE`] sets the threads of a solve, and each record says how many it
//! ran on. Parts run side by side under `batch` share the machine: left unset, each child gets
//! [`batch_threads`], so jobs times threads stays within the logical processors and the wall-clock
//! cap does not come to depend on how many jobs were asked for.

use std::collections::BTreeSet;
use std::time::Instant;

use solidfem::elastic::{
    Clock, Material, Model, ModelError, Precision, PreconditionerChoice, Quadrature, Report,
    StaticStudy, StepOutcome, StudyOptions,
};
use solidfem::elastic::{MultigridOptions, Progress};
use solidfem::mesh::QuadraticTetMesh;
use solidfem::solve::PcgError;

use super::volume_trial::variant_path;
use crate::benchmarks::pipeline::{face_area, load_balance, Supports};
use crate::benchmarks::verdict::{within_limit, LOAD_BALANCE_LIMIT};
use crate::gate_diagnostics::multigrid_options;

/// The solve stage is cancelled after this long, so a part costs its own row.
pub const SOLVE_TIME_CAP_SECONDS: f64 = 120.0;

/// The thickness of each end slab as a fraction of the part's length: thin enough that the two
/// slabs are far apart on any part, thick enough to reach the surface nodes of a coarse mesh near
/// a rounded end.
pub const SLAB_FRACTION: f64 = 0.05;

/// Held nodes needed: three not in a line are the fewest that stop a body's rigid motions.
pub const MIN_HELD_NODES: usize = 3;

/// The total force as a fraction of `E * D^2`; see the file header.
pub const LOAD_STRAIN: f64 = 1e-9;

/// A largest displacement over this many diagonals is no answer a solid can give to this load.
pub const UNPHYSICAL_DIAGONALS: f64 = 10.0;

/// The one material every corpus part is solved in: structural steel in millimetres and
/// megapascals (newtons per square millimetre).
pub const STEEL: Material = Material {
    youngs_modulus: 210_000.0,
    poisson_ratio: 0.3,
};

/// The environment variable choosing the multigrid precision of the solve stage.
pub const PRECISION_VARIABLE: &str = "CORPUS_PRECISION";

/// The environment variable printing the residual every that many iterations, to stderr.
pub const HISTORY_VARIABLE: &str = "CORPUS_HISTORY";

/// The environment variable replacing [`SOLVE_TIME_CAP_SECONDS`], in seconds.
pub const SOLVE_CAP_VARIABLE: &str = "CORPUS_SOLVE_CAP";

/// The environment variable setting the threads of a solve; see the file header.
pub const THREADS_VARIABLE: &str = "CORPUS_THREADS";

/// The threads each of `jobs` side-by-side children takes when [`THREADS_VARIABLE`] is unset: an
/// equal share of `logical` processors, at least one, and at most `library_default`, past which
/// more threads buy little.
pub fn batch_threads(jobs: usize, logical: usize, library_default: usize) -> usize {
    (logical / jobs.max(1)).clamp(1, library_default.max(1))
}

/// What the solve stage's diagnostic variables ask for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveDiagnostics {
    /// Print the residual every this many iterations; `None` prints nothing.
    pub history_every: Option<u64>,
    /// The solve stage's time cap.
    pub cap_seconds: f64,
    /// The threads asked for; `None` leaves the multigrid options' count.
    pub threads: Option<usize>,
}

impl Default for SolveDiagnostics {
    fn default() -> Self {
        SolveDiagnostics {
            history_every: None,
            cap_seconds: SOLVE_TIME_CAP_SECONDS,
            threads: None,
        }
    }
}

impl SolveDiagnostics {
    /// Reads the three variables' values, `None` meaning unset.
    ///
    /// # Errors
    /// A message naming the variable and what it accepts: a whole number of iterations from one,
    /// a positive number of seconds, or a whole number of threads from one.
    pub fn parse(
        history: Option<&str>,
        cap: Option<&str>,
        threads: Option<&str>,
    ) -> Result<Self, String> {
        let history_every = match history {
            None => None,
            Some(text) => match text.parse::<u64>() {
                Ok(every) if every >= 1 => Some(every),
                _ => {
                    return Err(format!(
                        "{HISTORY_VARIABLE}={text} is not a whole number of iterations from 1"
                    ))
                }
            },
        };
        let cap_seconds = match cap {
            None => SOLVE_TIME_CAP_SECONDS,
            Some(text) => match text.parse::<f64>() {
                Ok(seconds) if seconds.is_finite() && seconds > 0.0 => seconds,
                _ => {
                    return Err(format!(
                        "{SOLVE_CAP_VARIABLE}={text} is not a positive number of seconds"
                    ))
                }
            },
        };
        let threads = match threads {
            None => None,
            Some(text) => match text.parse::<usize>() {
                Ok(count) if count >= 1 => Some(count),
                _ => {
                    return Err(format!(
                        "{THREADS_VARIABLE}={text} is not a whole number of threads from 1"
                    ))
                }
            },
        };
        Ok(SolveDiagnostics {
            history_every,
            cap_seconds,
            threads,
        })
    }

    /// [`SolveDiagnostics::parse`] of the environment.
    ///
    /// # Errors
    /// As [`SolveDiagnostics::parse`].
    pub fn from_env() -> Result<Self, String> {
        let history = std::env::var(HISTORY_VARIABLE).ok();
        let cap = std::env::var(SOLVE_CAP_VARIABLE).ok();
        let threads = std::env::var(THREADS_VARIABLE).ok();
        Self::parse(history.as_deref(), cap.as_deref(), threads.as_deref())
    }
}

/// The relative residual whose first crossing each trial records: what a looser tolerance than
/// the solver's 1e-9 would cost.
pub const RESIDUAL_MARK: f64 = 1e-6;

/// The precisions [`PRECISION_VARIABLE`] names: unset is the library default alone
/// (`MultigridOptions::default().precision`), `double`, `single-storage` or `single` that one,
/// `all` the three in the order double, single storage, single.
///
/// # Errors
/// A message naming the accepted values when the variable holds anything else.
pub fn precisions_from_env() -> Result<Vec<Precision>, String> {
    match std::env::var(PRECISION_VARIABLE) {
        Err(_) => Ok(vec![MultigridOptions::default().precision]),
        Ok(text) => match text.as_str() {
            "double" => Ok(vec![Precision::Double]),
            "single-storage" => Ok(vec![Precision::SingleStorage]),
            "single" => Ok(vec![Precision::Single]),
            "all" => Ok(vec![
                Precision::Double,
                Precision::SingleStorage,
                Precision::Single,
            ]),
            other => Err(format!(
                "{PRECISION_VARIABLE}={other} is not a precision: use double, single-storage, \
                 single or all"
            )),
        },
    }
}

/// The record's name for a precision.
pub fn precision_name(precision: Precision) -> &'static str {
    match precision {
        Precision::Double => "double",
        Precision::SingleStorage => "single-storage",
        Precision::Single => "single",
    }
}

/// Why a part has no load case.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadCaseError {
    /// The mesh has no extent along any axis.
    NoExtent,
    /// Fewer than [`MIN_HELD_NODES`] surface nodes lie in the held slab.
    TooFewHeldNodes { found: usize },
    /// No boundary face lies wholly in the loaded slab.
    NoLoadedFace,
}

/// The bending load case of one mesh.
#[derive(Clone, Debug)]
pub struct LoadCase {
    /// Held nodes and loaded faces, as the benchmarks describe them.
    pub supports: Supports,
    /// The axis the part is held and loaded along.
    pub length_axis: usize,
    /// The axis the force points along.
    pub force_axis: usize,
    /// The total applied force.
    pub force: f64,
    /// The bounding-box diagonal.
    pub diagonal: f64,
}

/// What a solved part's answer measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolvedPart {
    /// `|applied + reactions| / applied`.
    pub imbalance: f64,
    /// The largest nodal displacement over the bounding-box diagonal.
    pub max_displacement_over_diagonal: f64,
}

/// Why a part's solve gave no checked answer.
#[derive(Clone, Debug, PartialEq)]
pub enum SolveFailure {
    /// The part has no load case.
    LoadCase(LoadCaseError),
    /// The study refused the model or the solve failed.
    Model(ModelError),
    /// The stage ran past [`SOLVE_TIME_CAP_SECONDS`].
    TimeCap,
    /// The reactions could not be integrated from the recovered stresses: an element's geometry
    /// could not be evaluated at an integration point, named.
    Reactions(String),
    /// The solve converged to a displacement no solid gives under this load.
    Unphysical(SolvedPart),
    /// The reactions do not balance the applied load within [`LOAD_BALANCE_LIMIT`].
    Imbalanced(SolvedPart),
}

/// What the solve stage did to one meshed part.
#[derive(Clone, Debug)]
pub struct SolveTrial {
    /// Displacement components solved for: three per node not held.
    pub unknowns: usize,
    /// The study's report when assembly finished; its iterations and residual are where the
    /// solve stopped.
    pub report: Option<Report>,
    /// Wall seconds in the whole stage.
    pub seconds: f64,
    /// The precision the multigrid hierarchy was stored and applied in.
    pub precision: Precision,
    /// The threads the solve ran on; what it asked for when no hierarchy was built.
    pub threads: usize,
    /// The iteration at which the relative residual first fell to [`RESIDUAL_MARK`], if it did.
    /// It reads the solver's running residual estimate, so it is a lower bound on the iterations
    /// a real 1e-6 tolerance would take (that also checks the true residual).
    pub below_mark: Option<u64>,
    pub outcome: Result<SolvedPart, SolveFailure>,
}

/// The bending load case of `mesh`; see the file header.
///
/// # Errors
/// The [`LoadCaseError`] naming what the part lacks.
pub fn bending_case(mesh: &QuadraticTetMesh) -> Result<LoadCase, LoadCaseError> {
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for p in mesh.nodes() {
        for axis in 0..3 {
            low[axis] = low[axis].min(p[axis]);
            high[axis] = high[axis].max(p[axis]);
        }
    }
    let extent = [0, 1, 2].map(|axis| high[axis] - low[axis]);
    // Longest first, ties to the lower axis; the force goes along the shortest of the rest.
    let mut order = [0usize, 1, 2];
    order.sort_by(|a, b| extent[*b].total_cmp(&extent[*a]).then(a.cmp(b)));
    let length_axis = order[0];
    // Sorted descending with ties ascending by index, so on a tie `order[1]` is the lower axis.
    let force_axis = if extent[order[2]] < extent[order[1]] {
        order[2]
    } else {
        order[1]
    };
    let length = extent[length_axis];
    if !(length.is_finite() && length > 0.0) {
        return Err(LoadCaseError::NoExtent);
    }
    let diagonal = (extent[0] * extent[0] + extent[1] * extent[1] + extent[2] * extent[2]).sqrt();
    let held_below = low[length_axis] + SLAB_FRACTION * length;
    let loaded_above = high[length_axis] - SLAB_FRACTION * length;

    let faces = mesh.boundary_faces();
    let mut held = BTreeSet::new();
    for face in &faces {
        for node in face.nodes {
            if mesh.nodes()[node as usize][length_axis] <= held_below {
                held.insert(node);
            }
        }
    }
    if held.len() < MIN_HELD_NODES {
        return Err(LoadCaseError::TooFewHeldNodes { found: held.len() });
    }
    let loaded: Vec<_> = faces
        .iter()
        .filter(|face| {
            face.nodes[..3]
                .iter()
                .all(|n| mesh.nodes()[*n as usize][length_axis] >= loaded_above)
        })
        .copied()
        .collect();
    let area: f64 = loaded.iter().map(|face| face_area(mesh, face)).sum();
    if loaded.is_empty() || area.is_nan() || area <= 0.0 {
        return Err(LoadCaseError::NoLoadedFace);
    }
    let force = STEEL.youngs_modulus * diagonal * diagonal * LOAD_STRAIN;
    let mut traction = [0.0; 3];
    traction[force_axis] = force / area;
    Ok(LoadCase {
        supports: Supports {
            fixed: held.into_iter().collect(),
            tractions: loaded.into_iter().map(|face| (face, traction)).collect(),
            prescribed: Vec::new(),
        },
        length_axis,
        force_axis,
        force,
        diagonal,
    })
}

/// Solves `mesh` under its bending load case with the default multigrid and checks the answer.
pub fn solve_and_check(mesh: &QuadraticTetMesh) -> SolveTrial {
    solve_and_check_in(mesh, multigrid_options().precision)
}

/// Solves `mesh` under its bending load case with the multigrid in `precision` and checks the
/// answer.
pub fn solve_and_check_in(mesh: &QuadraticTetMesh, precision: Precision) -> SolveTrial {
    let start = Instant::now();
    let case = match bending_case(mesh) {
        Ok(case) => case,
        Err(error) => {
            return SolveTrial {
                unknowns: 3 * mesh.nodes().len(),
                report: None,
                seconds: start.elapsed().as_secs_f64(),
                precision,
                threads: requested_threads(),
                below_mark: None,
                outcome: Err(SolveFailure::LoadCase(error)),
            }
        }
    };
    let unknowns = 3 * (mesh.nodes().len() - case.supports.fixed.len());
    let solved = solve(mesh, &case, precision, start);
    SolveTrial {
        unknowns,
        report: solved.report,
        seconds: start.elapsed().as_secs_f64(),
        precision,
        threads: solved.threads,
        below_mark: solved.below_mark,
        outcome: solved.outcome,
    }
}

/// The threads a solve asks for: [`THREADS_VARIABLE`], or else the multigrid options' count.
fn requested_threads() -> usize {
    SolveDiagnostics::from_env()
        .ok()
        .and_then(|diagnostics| diagnostics.threads)
        .unwrap_or_else(|| multigrid_options().threads)
}

/// What [`solve`] found.
struct Solved {
    report: Option<Report>,
    threads: usize,
    below_mark: Option<u64>,
    outcome: Result<SolvedPart, SolveFailure>,
}

/// The study under the time cap, then the two checks.
fn solve(mesh: &QuadraticTetMesh, case: &LoadCase, precision: Precision, start: Instant) -> Solved {
    let mut below_mark = None;
    let mut threads = requested_threads();
    let (report, outcome) =
        study_and_checks(mesh, case, precision, start, &mut below_mark, &mut threads);
    Solved {
        report,
        threads,
        below_mark,
        outcome,
    }
}

fn study_and_checks(
    mesh: &QuadraticTetMesh,
    case: &LoadCase,
    precision: Precision,
    start: Instant,
    below_mark: &mut Option<u64>,
    threads: &mut usize,
) -> (Option<Report>, Result<SolvedPart, SolveFailure>) {
    let quadrature = Quadrature::FourPoint;
    let mut model = Model::new(mesh.clone(), STEEL);
    model.set_quadrature(quadrature);
    for node in &case.supports.fixed {
        model.fix_node(*node);
    }
    for (face, traction) in &case.supports.tractions {
        model.add_traction(*face, *traction);
    }
    let clock: Clock = Box::new(move || start.elapsed().as_secs_f64());
    // The tool refuses a bad value before anything runs, so only a caller that skipped that check
    // reaches the plain stage here.
    let diagnostics = SolveDiagnostics::from_env().unwrap_or_default();
    let cap = diagnostics.cap_seconds;
    let history = diagnostics.history_every;
    let mut past_cap = || start.elapsed().as_secs_f64() >= cap;
    let options = StudyOptions {
        clock: Some(clock),
        preconditioner: PreconditionerChoice::Multigrid(MultigridOptions {
            precision,
            threads: *threads,
            ..multigrid_options()
        }),
        ..StudyOptions::default()
    };
    let mut study = match StaticStudy::new(model, options, &mut past_cap) {
        Ok(study) => study,
        Err(error) => return (None, Err(model_failure(error))),
    };
    if let Some(multigrid) = study.multigrid() {
        *threads = multigrid.threads();
    }
    if history.is_some() {
        eprintln!("history setup {:?}", study.report());
    }
    let mut mark = |progress: Progress| {
        if let Some(every) = history {
            if progress.iteration.is_multiple_of(every) {
                eprintln!(
                    "history iteration={} residual={:.3e} seconds={:.1}",
                    progress.iteration,
                    progress.relative_residual,
                    start.elapsed().as_secs_f64()
                );
            }
        }
        if below_mark.is_none() && progress.relative_residual <= RESIDUAL_MARK {
            *below_mark = Some(progress.iteration);
        }
    };
    loop {
        match study.step(1, &mut past_cap) {
            Ok(StepOutcome::Converged(progress)) => {
                mark(progress);
                break;
            }
            Ok(StepOutcome::Running(progress)) => mark(progress),
            Ok(StepOutcome::Cancelled(_)) => {
                return (Some(study.report()), Err(SolveFailure::TimeCap))
            }
            Err(error) => return (Some(study.report()), Err(model_failure(error))),
        }
    }
    let solution = match study.solution(&mut past_cap) {
        Ok(solution) => solution,
        Err(error) => return (Some(study.report()), Err(model_failure(error))),
    };
    let report = Some(solution.report);
    let largest = solution
        .displacements
        .iter()
        .map(|u| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt())
        .fold(0.0_f64, |worst, value| {
            // A NaN must win, so it is carried rather than dropped by `max`.
            if value.is_nan() || worst.is_nan() {
                f64::NAN
            } else {
                worst.max(value)
            }
        });
    let balance = match load_balance(mesh, &case.supports, &solution.point_stresses, quadrature) {
        Ok(balance) => balance,
        Err(fault) => return (report, Err(SolveFailure::Reactions(fault))),
    };
    let measured = SolvedPart {
        imbalance: balance.error,
        max_displacement_over_diagonal: largest / case.diagonal,
    };
    let outcome = if !within_limit(
        measured.max_displacement_over_diagonal,
        UNPHYSICAL_DIAGONALS,
    ) {
        Err(SolveFailure::Unphysical(measured))
    } else if !within_limit(measured.imbalance, LOAD_BALANCE_LIMIT) {
        Err(SolveFailure::Imbalanced(measured))
    } else {
        Ok(measured)
    };
    (report, outcome)
}

/// A study error, with the time cap's cancellation told apart from a refusal.
fn model_failure(error: ModelError) -> SolveFailure {
    match error {
        ModelError::Cancelled => SolveFailure::TimeCap,
        other => SolveFailure::Model(other),
    }
}

/// The stable short name of a solve outcome: `ok`, `load-case:<variant>` for a part with no load
/// case, `solve:not-converged` for a solver that ran out of iterations, `solve:<variant>` for
/// the study's other errors, `time-cap:solve`, `solve:reactions` when the reactions cannot be
/// integrated, and `solve:unphysical` or `solve:imbalanced` for
/// an answer that failed a physical check.
pub fn solve_outcome_name(outcome: &Result<SolvedPart, SolveFailure>) -> String {
    match outcome {
        Ok(_) => "ok".to_string(),
        Err(SolveFailure::LoadCase(error)) => {
            format!("load-case:{}", variant_path(&format!("{error:?}")))
        }
        Err(SolveFailure::Model(ModelError::Solver(PcgError::NotConverged { .. }))) => {
            "solve:not-converged".to_string()
        }
        Err(SolveFailure::Model(error)) => format!("solve:{}", variant_path(&format!("{error:?}"))),
        Err(SolveFailure::TimeCap) => "time-cap:solve".to_string(),
        Err(SolveFailure::Reactions(_)) => "solve:reactions".to_string(),
        Err(SolveFailure::Unphysical(_)) => "solve:unphysical".to_string(),
        Err(SolveFailure::Imbalanced(_)) => "solve:imbalanced".to_string(),
    }
}
