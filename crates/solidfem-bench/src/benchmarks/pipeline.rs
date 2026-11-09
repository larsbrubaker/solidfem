//! The pipeline each real-part benchmark runs at each mesh density: a closed surface, this
//! workspace's volume mesher with through-wall layers and a size function and the upgrade to
//! 10-node elements (`volume.rs`, shared with the corpus), the multigrid solve under a time cap,
//! and a load-balance check from the recovered stresses.
//! `cases/` says what each part is; `verdict.rs` decides what its numbers mean.
//!
//! **The load balance is measured, not assumed.** The applied load is summed here from each loaded
//! face's own area and traction, independently of the solver's consistent load vector; the
//! reaction is the sum, over the fixed nodes, of the element internal forces integrated from the
//! recovered integration-point stresses with the same rule the assembly used. A wrong support
//! selection, a load integrated badly or an unconverged solve all show up as an imbalance.

use std::time::Instant;

use solidfem::elastic::{
    point_geometry, Clock, Material, Model, PointStress, PreconditionerChoice, Quadrature, Report,
    Solution, StaticStudy, StepOutcome, StudyOptions,
};
use solidfem::mesh::{BoundaryFace, QuadraticTetMesh};
use solidfem::tetra::{RefineOptions, SizeFn};

use super::cases::{Case, Density};
use super::verdict::{within_limit, LoadBalance, Verdict, LOAD_BALANCE_LIMIT};
use super::volume::{volume_mesh, VolumeFailure, VolumeStage};
use crate::gate_diagnostics::multigrid_options;

/// A closed triangle surface: vertices and outward-wound triangles.
pub type Surface = (Vec<[f64; 3]>, Vec<[u32; 3]>);

/// Solver iterations between looks at the clock.
const SOLVE_SLICE: usize = 5;

/// What a case applies to a mesh: which nodes are held and which boundary faces are loaded.
#[derive(Debug, Default, Clone)]
pub struct Supports {
    /// Nodes fixed in all three directions.
    pub fixed: Vec<u32>,
    /// Boundary faces with a uniform traction (force per area).
    pub tractions: Vec<(BoundaryFace, [f64; 3])>,
    /// Individually prescribed displacement components: (node, axis, value). A roller or symmetry
    /// plane is a prescribed zero; an enforced displacement is its value.
    pub prescribed: Vec<(u32, usize, f64)>,
}

/// One density's solved result, what the tables print and the judges read.
#[derive(Debug, Clone)]
pub struct DensityResult {
    /// The density's label.
    pub label: String,
    /// The local mesh size that series of this density is measured against.
    pub size: f64,
    /// Named measured quantities, in a fixed order.
    pub values: Vec<(String, f64)>,
    /// Applied load against reactions.
    pub balance: LoadBalance,
    /// Nodes of the 10-node mesh.
    pub nodes: usize,
    /// Elements.
    pub elements: usize,
    /// Solved degrees of freedom.
    pub free_dofs: usize,
    /// Conjugate gradient iterations.
    pub iterations: u64,
    /// Final relative residual.
    pub residual: f64,
    /// Smallest dihedral angle of the mesh, degrees.
    pub min_dihedral: f64,
    /// Seconds in the whole pipeline.
    pub seconds: f64,
}

impl DensityResult {
    /// The quantity named `name`.
    ///
    /// # Panics
    /// When the case's measure did not produce that name, a programming error in the case.
    pub fn value(&self, name: &str) -> f64 {
        match self.values.iter().find(|(key, _)| key == name) {
            Some((_, value)) => *value,
            None => panic!("the case did not measure {name}"),
        }
    }
}

/// What one density ended as.
enum DensityOutcome {
    Solved(Box<DensityResult>),
    /// A stage refused the part, with the reason.
    Refused(String),
    /// This workspace's surface recovery could not recover a surface it had accepted.
    MesherRefused(String),
    /// The solve hit the cap.
    Capped {
        iterations: u64,
        residual: f64,
    },
}

/// A whole case: its density rows and its verdict.
#[derive(Debug, Clone)]
pub struct CaseResult {
    /// The rows that were solved, coarse to fine.
    pub rows: Vec<DensityResult>,
    /// The verdict.
    pub verdict: Verdict,
}

/// Reads a surface in the text format `solidfem-gate file` reads: `vertex-count face-count`, then
/// `x y z` lines, then `a b c` lines.
///
/// # Errors
/// A message naming the file when it cannot be read or parsed.
pub fn read_surface(path: &str) -> Result<Surface, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    parse_surface(&text, path)
}

/// Parses a surface in that text format; `path` names it in any error. The demo page reads the
/// same files through this, from a fetch rather than a disk.
///
/// # Errors
/// A message naming `path` when the text cannot be parsed.
pub fn parse_surface(text: &str, path: &str) -> Result<Surface, String> {
    let mut tokens = text.split_ascii_whitespace();
    let mut next = |what: &str| {
        tokens
            .next()
            .ok_or_else(|| format!("{path}: ran out of {what}"))
    };
    let parse_count = |value: &str| {
        value
            .parse::<usize>()
            .map_err(|e| format!("{path}: bad count {value}: {e}"))
    };
    let vertex_count = parse_count(next("header")?)?;
    let face_count = parse_count(next("header")?)?;
    let mut points = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        let mut p = [0.0; 3];
        for value in &mut p {
            let token = next("vertices")?;
            *value = token
                .parse()
                .map_err(|e| format!("{path}: bad coordinate {token}: {e}"))?;
        }
        points.push(p);
    }
    let mut triangles = Vec::with_capacity(face_count);
    for _ in 0..face_count {
        let mut t = [0u32; 3];
        for value in &mut t {
            let token = next("faces")?;
            *value = token
                .parse()
                .map_err(|e| format!("{path}: bad index {token}: {e}"))?;
        }
        triangles.push(t);
    }
    Ok((points, triangles))
}

/// The failure reason when this workspace's surface recovery cannot recover a surface the mesher
/// accepted: names the density and says the refusal is the mesher's, because such a failure shows
/// nothing wrong with the surface itself.
pub fn mesher_refused(density: &str, reason: &str) -> String {
    format!(
        "{density}: mesher refused: this workspace's surface recovery could not recover the \
         surface, with no surface defect shown ({reason})"
    )
}

/// Runs a case at every density from `surface_of`, coarse to fine, with each solve capped at
/// `cap_seconds`, and judges the result. A stage refusing any density fails the case with that
/// stage's reason; a capped solve makes it not assessed.
pub fn run_case(
    case: &Case,
    surface_of: &mut dyn FnMut(&Density) -> Result<Surface, String>,
    cap_seconds: f64,
    progress: &mut dyn FnMut(&str),
) -> CaseResult {
    let mut rows = Vec::with_capacity(case.densities.len());
    for density in &case.densities {
        progress(&format!("{}: {} starting", case.id, density.label));
        let outcome = match surface_of(density) {
            Ok(surface) => solve_density(case, density, surface, cap_seconds),
            Err(reason) => DensityOutcome::Refused(format!("surface: {reason}")),
        };
        match outcome {
            DensityOutcome::Solved(row) => {
                progress(&format!(
                    "{}: {} solved, {} nodes, {} iterations, {:.1} s, load balance error {:.2e}",
                    case.id, row.label, row.nodes, row.iterations, row.seconds, row.balance.error
                ));
                rows.push(*row);
            }
            DensityOutcome::Refused(reason) => {
                return CaseResult {
                    rows,
                    verdict: Verdict::Fail(format!("{} refused: {reason}", density.label)),
                };
            }
            DensityOutcome::MesherRefused(reason) => {
                return CaseResult {
                    rows,
                    verdict: Verdict::Fail(mesher_refused(&density.label, &reason)),
                };
            }
            DensityOutcome::Capped {
                iterations,
                residual,
            } => {
                return CaseResult {
                    rows,
                    verdict: Verdict::NotAssessed(format!(
                        "{}: the solve hit its {cap_seconds} s cap at iteration {iterations} \
                         (residual {residual:.2e}); no error is quoted",
                        density.label
                    )),
                };
            }
        }
    }
    let imbalanced = rows
        .iter()
        .find(|row| !within_limit(row.balance.error, LOAD_BALANCE_LIMIT));
    let verdict = match imbalanced {
        Some(row) => Verdict::Fail(format!(
            "{}: the reactions do not balance the applied load: error {:.3e} of {} gross, over the limit {LOAD_BALANCE_LIMIT:e}",
            row.label, row.balance.error, row.balance.gross
        )),
        None => (case.judge)(&rows),
    };
    CaseResult { rows, verdict }
}

fn solve_density(
    case: &Case,
    density: &Density,
    (points, triangles): Surface,
    cap_seconds: f64,
) -> DensityOutcome {
    let start = Instant::now();
    let sizing = density.size;
    let size_function: SizeFn = Box::new(move |p| sizing.at(p));
    let options = RefineOptions {
        layers_through_walls: case.layers,
        ..RefineOptions::default()
    };
    let quadrature = Quadrature::FourPoint;
    let volume = match volume_mesh(
        (points, triangles),
        size_function,
        options,
        quadrature,
        &mut || false,
    ) {
        Ok(volume) => volume,
        Err(failure) => return refused_by_volume(failure),
    };
    let min_dihedral = volume.report.after.min_degrees;
    let mesh = volume.mesh;
    let supports = match (case.apply)(&mesh) {
        Ok(supports) => supports,
        Err(error) => return DensityOutcome::Refused(format!("supports and loads: {error}")),
    };
    let is_fixed = fixed_mask(&mesh, &supports);
    if supports
        .tractions
        .iter()
        .any(|(face, _)| face.nodes.iter().any(|n| is_fixed[*n as usize]))
    {
        return DensityOutcome::Refused(
            "a loaded face touches a fixed node, so its load would go straight to the support"
                .to_string(),
        );
    }
    let (solution, report) = match solve(
        mesh.clone(),
        case.material,
        &supports,
        cap_seconds,
        case.iteration_budget,
        quadrature,
    ) {
        Ok(solved) => solved,
        Err(SolveEnd::Refused(reason)) => return DensityOutcome::Refused(reason),
        Err(SolveEnd::Capped(report)) => {
            return DensityOutcome::Capped {
                iterations: report.iterations,
                residual: report.relative_residual,
            }
        }
    };
    let balance = match load_balance(&mesh, &supports, &solution.point_stresses, quadrature) {
        Ok(balance) => balance,
        Err(error) => return DensityOutcome::Refused(format!("reaction recovery: {error}")),
    };
    let values = match (case.measure)(&mesh, &solution) {
        Ok(values) => values,
        Err(error) => return DensityOutcome::Refused(format!("measurement: {error}")),
    };
    DensityOutcome::Solved(Box::new(DensityResult {
        label: density.label.to_string(),
        size: density.series_size,
        values,
        balance,
        nodes: mesh.nodes().len(),
        elements: mesh.tets().len(),
        free_dofs: report.free_dofs,
        iterations: report.iterations,
        residual: report.relative_residual,
        min_dihedral,
        seconds: start.elapsed().as_secs_f64(),
    }))
}

/// The wording of a volume-stage refusal: surface recovery's own failures are the mesher's,
/// everything else is the stage's refusal of this part.
fn refused_by_volume(failure: VolumeFailure) -> DensityOutcome {
    match failure {
        VolumeFailure::Surface(error) => {
            DensityOutcome::Refused(format!("surface mesher: {error}"))
        }
        VolumeFailure::Recovery(error) => DensityOutcome::MesherRefused(error.to_string()),
        VolumeFailure::Stopped {
            stage: VolumeStage::Recovery,
            status,
        } => DensityOutcome::MesherRefused(format!("stopped: {status:?}")),
        VolumeFailure::Refiner(error) => DensityOutcome::Refused(format!("refiner: {error}")),
        VolumeFailure::Refinement(error) => DensityOutcome::Refused(format!("refinement: {error}")),
        VolumeFailure::Stopped {
            stage: VolumeStage::Refinement,
            status,
        } => DensityOutcome::Refused(format!("refinement stopped: {status:?}")),
        VolumeFailure::Upgrade(error) => {
            DensityOutcome::Refused(format!("10-node upgrade: {error}"))
        }
    }
}

pub(crate) enum SolveEnd {
    Refused(String),
    Capped(Report),
}

pub(crate) fn solve(
    mesh: QuadraticTetMesh,
    material: Material,
    supports: &Supports,
    cap_seconds: f64,
    iteration_budget: Option<u64>,
    quadrature: Quadrature,
) -> Result<(Solution, Report), SolveEnd> {
    let mut model = Model::new(mesh, material);
    model.set_quadrature(quadrature);
    for node in &supports.fixed {
        model.fix_node(*node);
    }
    for (face, traction) in &supports.tractions {
        model.add_traction(*face, *traction);
    }
    for (node, axis, value) in &supports.prescribed {
        model.prescribe_displacement(*node, *axis, *value);
    }
    let start = Instant::now();
    let clock: Clock = Box::new(move || start.elapsed().as_secs_f64());
    let mut study = StaticStudy::new(
        model,
        StudyOptions {
            clock: Some(clock),
            preconditioner: PreconditionerChoice::Multigrid(multigrid_options()),
            ..StudyOptions::default()
        },
        &mut || false,
    )
    .map_err(|error| SolveEnd::Refused(format!("study: {error}")))?;
    let solve_start = Instant::now();
    loop {
        match study.step(SOLVE_SLICE, &mut || false) {
            Ok(StepOutcome::Converged(_)) => break,
            Ok(StepOutcome::Running(_)) => {}
            Ok(StepOutcome::Cancelled(_)) => {
                return Err(SolveEnd::Refused("the solve was cancelled".to_string()))
            }
            Err(error) => return Err(SolveEnd::Refused(format!("solve: {error}"))),
        }
        if solve_start.elapsed().as_secs_f64() >= cap_seconds {
            return Err(SolveEnd::Capped(study.report()));
        }
        if let Some(budget) = iteration_budget {
            let spent = study.report().iterations;
            if spent >= budget {
                return Err(SolveEnd::Refused(format!(
                    "over the iteration budget: {spent} iterations reached the budget of {budget}"
                )));
            }
        }
    }
    let solution = study
        .solution(&mut || false)
        .map_err(|error| SolveEnd::Refused(format!("recovery: {error}")))?;
    let report = solution.report;
    Ok((solution, report))
}

/// The resultant of the applied tractions and the sum of their magnitudes, from each face's own
/// corner area, independent of the solver's consistent load vector.
fn applied_load(mesh: &QuadraticTetMesh, supports: &Supports) -> ([f64; 3], f64) {
    let mut resultant = [0.0; 3];
    let mut gross = 0.0;
    for (face, traction) in &supports.tractions {
        let area = face_area(mesh, face);
        for axis in 0..3 {
            resultant[axis] += traction[axis] * area;
        }
        gross += area
            * (traction[0] * traction[0] + traction[1] * traction[1] + traction[2] * traction[2])
                .sqrt();
    }
    (resultant, gross)
}

/// A boundary face's area, from its corners.
pub fn face_area(mesh: &QuadraticTetMesh, face: &BoundaryFace) -> f64 {
    let normal = face_normal(mesh, face);
    0.5 * (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt()
}

/// A boundary face's right-hand corner normal, twice the area long: outward from its element.
pub fn face_normal(mesh: &QuadraticTetMesh, face: &BoundaryFace) -> [f64; 3] {
    let [a, b, c] = [face.nodes[0], face.nodes[1], face.nodes[2]].map(|n| mesh.nodes()[n as usize]);
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

/// Which nodes `supports` fixes in all three directions.
fn fixed_mask(mesh: &QuadraticTetMesh, supports: &Supports) -> Vec<bool> {
    let mut is_fixed = vec![false; mesh.nodes().len()];
    for node in &supports.fixed {
        is_fixed[*node as usize] = true;
    }
    is_fixed
}

/// The applied load against the support reactions recovered from `point_stresses`. An enforced
/// displacement is its own loading: with no applied traction the reactions alone set the scale.
///
/// # Errors
/// When an integration point's element geometry cannot be evaluated, naming the fault.
pub fn load_balance(
    mesh: &QuadraticTetMesh,
    supports: &Supports,
    point_stresses: &[PointStress],
    quadrature: Quadrature,
) -> Result<LoadBalance, String> {
    let is_fixed = fixed_mask(mesh, supports);
    let (applied, gross) = applied_load(mesh, supports);
    let (reaction, reaction_magnitude) =
        support_reaction(mesh, point_stresses, &is_fixed, supports, quadrature)?;
    let gross = if supports.tractions.is_empty() {
        reaction_magnitude
    } else {
        gross
    };
    Ok(LoadBalance::new(applied, reaction, gross))
}

/// The force the supports exert on the part and the sum of the magnitudes of its components at
/// each held value. At a fully fixed node the reaction is the internal nodal force integrated
/// from the point stresses, since no loaded face touches a fixed node (the pipeline refuses one
/// that does). At an individually prescribed component the reaction is that internal force
/// less the consistent nodal share of any traction on a face through the node, because a
/// roller can sit under a loaded face.
fn support_reaction(
    mesh: &QuadraticTetMesh,
    point_stresses: &[PointStress],
    is_fixed: &[bool],
    supports: &Supports,
    quadrature: Quadrature,
) -> Result<([f64; 3], f64), String> {
    let mut internal = vec![[0.0; 3]; mesh.nodes().len()];
    let rule = quadrature.points();
    for stress in point_stresses {
        let tet = &mesh.tets()[stress.tet as usize];
        let nodes = tet.map(|n| mesh.nodes()[n as usize]);
        let q = &rule[stress.point as usize];
        let geometry = point_geometry(&nodes, q.natural, stress.point as usize)
            .map_err(|fault| format!("{fault:?}"))?;
        let measure = q.weight * geometry.det_j / 6.0;
        let s = stress.stress;
        for (local, node) in tet.iter().enumerate() {
            let g = geometry.gradients[local];
            let force = &mut internal[*node as usize];
            force[0] += measure * (s[0] * g[0] + s[3] * g[1] + s[5] * g[2]);
            force[1] += measure * (s[3] * g[0] + s[1] * g[1] + s[4] * g[2]);
            force[2] += measure * (s[5] * g[0] + s[4] * g[1] + s[2] * g[2]);
        }
    }
    let mut external = vec![[0.0; 3]; mesh.nodes().len()];
    for (face, traction) in &supports.tractions {
        // The consistent load of a uniform traction on a straight 6-node face is zero at its
        // corners and a third of the face resultant at each midside.
        let area = face_area(mesh, face);
        for node in &face.nodes[3..] {
            for axis in 0..3 {
                external[*node as usize][axis] += traction[axis] * area / 3.0;
            }
        }
    }
    let (mut reaction, mut magnitude) = ([0.0; 3], 0.0);
    for (node, force) in internal.iter().enumerate() {
        if is_fixed[node] {
            for axis in 0..3 {
                reaction[axis] += force[axis];
                magnitude += force[axis].abs();
            }
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    for (node, axis, _) in &supports.prescribed {
        if seen.insert((*node, *axis)) {
            let share = internal[*node as usize][*axis] - external[*node as usize][*axis];
            reaction[*axis] += share;
            magnitude += share.abs();
        }
    }
    Ok((reaction, magnitude))
}
