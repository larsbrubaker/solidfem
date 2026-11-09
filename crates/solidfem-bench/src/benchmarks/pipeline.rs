//! The pipeline each real-part benchmark runs at each mesh density: a closed surface, this
//! workspace's volume mesher with through-wall layers and a size function, 10-node elements, the
//! multigrid solve under a time cap, and a load-balance check from the recovered stresses.
//! `cases/` says what each part is; `verdict.rs` decides what its numbers mean.
//!
//! **The load balance is measured, not assumed.** The applied load is summed here from each loaded
//! face's own area and traction, independently of the solver's consistent load vector; the
//! reaction is the sum, over the fixed nodes, of the element internal forces integrated from the
//! recovered integration-point stresses with the same rule the assembly used. A wrong support
//! selection, a load integrated badly or an unconverged solve all show up as an imbalance.

use std::time::Instant;

use solidfem::elastic::{
    point_geometry, Clock, Material, Model, PreconditionerChoice, Quadrature, Report, Solution,
    StaticStudy, StepOutcome, StudyOptions,
};
use solidfem::mesh::{BoundaryFace, QuadraticTetMesh};
use solidfem::meshing::quadratic_from_solid;
use solidfem::tetra::{RefineOptions, Refiner, SizeFn, StepStatus, SurfaceMesher};

use super::cases::{Case, Density};
use super::verdict::{LoadBalance, Verdict, LOAD_BALANCE_LIMIT};
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
        .find(|row| row.balance.error.is_nan() || row.balance.error > LOAD_BALANCE_LIMIT);
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
    let mut mesher = match SurfaceMesher::new(points, triangles) {
        Ok(mesher) => mesher,
        Err(error) => return DensityOutcome::Refused(format!("surface mesher: {error}")),
    };
    match mesher.step(usize::MAX, &mut || false) {
        Ok(StepStatus::Done) => {}
        Ok(other) => {
            return DensityOutcome::Refused(format!("surface recovery stopped: {other:?}"))
        }
        Err(error) => return DensityOutcome::Refused(format!("surface recovery: {error}")),
    }
    let solid = match mesher.result() {
        Ok(solid) => solid,
        Err(error) => return DensityOutcome::Refused(format!("surface recovery: {error}")),
    };
    let sizing = density.size;
    let size_function: SizeFn = Box::new(move |p| sizing.at(p));
    let options = RefineOptions {
        layers_through_walls: case.layers,
        ..RefineOptions::default()
    };
    let mut refiner = match Refiner::new(solid, size_function, options) {
        Ok(refiner) => refiner,
        Err(error) => return DensityOutcome::Refused(format!("refiner: {error}")),
    };
    match refiner.step(usize::MAX, &mut || false) {
        Ok(StepStatus::Done) => {}
        Ok(other) => return DensityOutcome::Refused(format!("refinement stopped: {other:?}")),
        Err(error) => return DensityOutcome::Refused(format!("refinement: {error}")),
    }
    let refined = match refiner.result() {
        Ok(refined) => refined,
        Err(error) => return DensityOutcome::Refused(format!("refinement: {error}")),
    };
    let min_dihedral = refiner.report().after.min_degrees;
    let quadrature = Quadrature::FourPoint;
    let mesh = match quadratic_from_solid(&refined, quadrature) {
        Ok(mesh) => mesh,
        Err(error) => return DensityOutcome::Refused(format!("10-node upgrade: {error}")),
    };
    let supports = match (case.apply)(&mesh) {
        Ok(supports) => supports,
        Err(error) => return DensityOutcome::Refused(format!("supports and loads: {error}")),
    };
    let mut is_fixed = vec![false; mesh.nodes().len()];
    for node in &supports.fixed {
        is_fixed[*node as usize] = true;
    }
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
    let (applied, gross) = applied_load(&mesh, &supports);
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
    let (reaction, reaction_magnitude) =
        match support_reaction(&mesh, &solution, &is_fixed, &supports, quadrature) {
            Ok(reaction) => reaction,
            Err(error) => return DensityOutcome::Refused(format!("reaction recovery: {error}")),
        };
    // An enforced displacement is its own loading: with no applied traction the reactions alone
    // set the scale of the balance check.
    let gross = if supports.tractions.is_empty() {
        reaction_magnitude
    } else {
        gross
    };
    let balance = LoadBalance::new(applied, reaction, gross);
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

enum SolveEnd {
    Refused(String),
    Capped(Report),
}

fn solve(
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

/// The force the supports exert on the part and the sum of the magnitudes of its components at
/// each held value. At a fully fixed node the reaction is the internal nodal force integrated
/// from the point stresses, since no loaded face touches a fixed node (the pipeline refuses one
/// that does). At an individually prescribed component the reaction is that internal force
/// less the consistent nodal share of any traction on a face through the node, because a
/// roller can sit under a loaded face.
fn support_reaction(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
    is_fixed: &[bool],
    supports: &Supports,
    quadrature: Quadrature,
) -> Result<([f64; 3], f64), String> {
    let mut internal = vec![[0.0; 3]; mesh.nodes().len()];
    let rule = quadrature.points();
    for stress in &solution.point_stresses {
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
