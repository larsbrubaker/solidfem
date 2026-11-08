//! The demo's engine: one study from a request through assembly, solve and recovery in small
//! slices, with the comparison against the textbook and the render arrays kept current as a live
//! estimate during the solve. A slice's budget is one number from the page: iterations while
//! solving, and [`ELEMENTS_PER_BUDGET_UNIT`] elements per unit while assembling or recovering,
//! because an element costs about as much as a sliver of one solver iteration.
//!
//! A new study replaces the old one, which cancels a running solve at once: there is no thread to
//! stop, so whatever was mid-slice is simply dropped between calls.
//!
//! The rendered surface is the 10-node faces split into four straight triangles each: the
//! quadratic elements' edge midpoints carry real stress and displacement, and showing them keeps
//! the colour field and the bent shape as detailed as what was solved, at a triangle count the
//! page handles easily. The von Mises array follows the latest finished recovery of the current
//! estimate (recoveries are themselves sliced); displacement follows every step.

use solidfem::elastic::{
    strain_at, stress_at, Material, ModelError, PcgOptions, RecoveredField, Recovery, StaticStudy,
    StepOutcome, StudyBuilder, StudyOptions,
};
use solidfem::mesh::QuadraticTetMesh;

use crate::catalog::StudyRequest;
use crate::plan::{plan_study, Comparison, PlanError, Probe, Textbook};
use crate::validity::{assess, Validity};
use crate::view::{Surface, ViewData};

/// Elements assembled or recovered per budget unit.
pub const ELEMENTS_PER_BUDGET_UNIT: usize = 48;

/// Where a study is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudyState {
    Assembling,
    Solving,
    Recovering,
    Done,
    Cancelled,
    Failed,
}

/// One row the page shows: the current FEA estimate, the textbook value and how far apart they
/// are, or just the FEA value when there is no textbook answer.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonRow {
    pub quantity: &'static str,
    pub unit: &'static str,
    pub place: String,
    pub fea: f64,
    pub textbook: f64,
    /// Percent difference of FEA from the textbook.
    pub percent_difference: f64,
}

enum Phase {
    Assembling(Box<StudyBuilder<Material>>),
    Solving(Box<StaticStudy<Material>>),
    Recovering(Box<StaticStudy<Material>>, Box<Recovery>),
    Done,
    Failed,
}

/// A study in progress, and its latest state.
pub struct DemoStudy {
    request: StudyRequest,
    material: Material,
    yield_strength: f64,
    mesh: QuadraticTetMesh,
    textbook: Textbook,
    clamps: Vec<f64>,
    clamp_zone: f64,
    surface: Surface,
    phase: Phase,
    cancelled: bool,
    displacements: Vec<[f64; 3]>,
    /// The latest finished stress recovery of the estimate, for the colours and the peak.
    nodal_von_mises: Vec<f64>,
    preview: Option<Recovery>,
    rows: Vec<ComparisonRow>,
    iterations: u64,
    residual: f64,
    failure: Option<String>,
}

impl DemoStudy {
    /// Plans a request and starts its study, refusing a bad or oversized one.
    pub fn start(request: StudyRequest) -> Result<Self, PlanError> {
        let planned = plan_study(&request)?;
        let surface = Surface::of(&planned.mesh);
        let node_count = planned.mesh.nodes().len();
        let builder = StudyBuilder::new(
            planned.model,
            StudyOptions {
                pcg: PcgOptions::default(),
                ..StudyOptions::default()
            },
        )
        .map_err(PlanError::Model)?;
        let mut study = DemoStudy {
            request,
            material: planned.material,
            yield_strength: planned.yield_strength,
            mesh: planned.mesh,
            textbook: planned.textbook,
            clamps: planned.clamps,
            clamp_zone: planned.clamp_zone,
            surface,
            phase: Phase::Assembling(Box::new(builder)),
            cancelled: false,
            displacements: vec![[0.0; 3]; node_count],
            nodal_von_mises: vec![0.0; node_count],
            preview: None,
            rows: Vec::new(),
            iterations: 0,
            residual: 1.0,
            failure: None,
        };
        study.update_rows();
        Ok(study)
    }

    /// The request this study was started with.
    pub fn request(&self) -> &StudyRequest {
        &self.request
    }

    /// Where the study is.
    pub fn state(&self) -> StudyState {
        if self.cancelled {
            return StudyState::Cancelled;
        }
        match &self.phase {
            Phase::Assembling(_) => StudyState::Assembling,
            Phase::Solving(_) => StudyState::Solving,
            Phase::Recovering(..) => StudyState::Recovering,
            Phase::Done => StudyState::Done,
            Phase::Failed => StudyState::Failed,
        }
    }

    /// Why the study failed, when it did.
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    /// Solver iterations so far.
    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// The current relative residual.
    pub fn residual(&self) -> f64 {
        self.residual
    }

    /// Cancels the study at once; nothing further happens.
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// The comparison rows at the current estimate: empty when there is no textbook answer.
    pub fn rows(&self) -> &[ComparisonRow] {
        &self.rows
    }

    /// The textbook column, or the reason there is none.
    pub fn textbook(&self) -> &Textbook {
        &self.textbook
    }

    /// The mesh size of this study: elements and nodes.
    pub fn mesh_size(&self) -> (usize, usize) {
        (self.mesh.tets().len(), self.mesh.nodes().len())
    }

    /// Runs one slice of the study with a budget of `budget` units, returning the new state.
    /// Elapses nothing else: a slice's cost is bounded by its budget.
    pub fn step(&mut self, budget: u32) -> StudyState {
        if self.cancelled {
            return StudyState::Cancelled;
        }
        let budget = budget.max(1) as usize;
        let elements = budget.saturating_mul(ELEMENTS_PER_BUDGET_UNIT);
        let mut no_cancel = || false;
        match std::mem::replace(&mut self.phase, Phase::Failed) {
            Phase::Assembling(mut builder) => match builder.step(elements, &mut no_cancel) {
                Ok(progress) if progress.elements_done == progress.elements_total => {
                    match builder.finish() {
                        Ok(study) => self.phase = Phase::Solving(Box::new(study)),
                        Err(e) => self.fail(&e),
                    }
                }
                Ok(_) => self.phase = Phase::Assembling(builder),
                Err(e) => self.fail(&e),
            },
            Phase::Solving(mut study) => match study.step(budget, &mut no_cancel) {
                Ok(StepOutcome::Converged(progress)) => {
                    self.iterations = progress.iteration;
                    self.residual = progress.relative_residual;
                    self.displacements = study.displacements();
                    self.update_rows();
                    let recovery = study.begin_recovery();
                    self.phase = Phase::Recovering(study, Box::new(recovery));
                }
                Ok(StepOutcome::Running(progress)) => {
                    self.iterations = progress.iteration;
                    self.residual = progress.relative_residual;
                    self.displacements = study.displacements();
                    self.update_rows();
                    // One element per iteration of this slice: the live colours refresh without
                    // the preview ever costing more than the solve it illustrates.
                    self.advance_preview(&mut study, budget);
                    self.phase = Phase::Solving(study);
                }
                Ok(StepOutcome::Cancelled(_)) => self.cancelled = true,
                Err(e) => self.fail(&e),
            },
            Phase::Recovering(mut study, mut recovery) => {
                match study.continue_recovery(&mut recovery, elements, &mut no_cancel) {
                    Ok(true) => match study.finish_recovery(*recovery) {
                        Ok(field) => self.finish(field),
                        Err(e) => self.fail(&e),
                    },
                    Ok(false) => self.phase = Phase::Recovering(study, recovery),
                    Err(e) => self.fail(&e),
                }
            }
            other => self.phase = other,
        }
        self.state()
    }

    fn fail(&mut self, error: &ModelError) {
        self.failure = Some(error.to_string());
        self.phase = Phase::Failed;
    }

    /// Advances a preview recovery of the estimate, replacing the stress field when it finishes.
    fn advance_preview(&mut self, study: &mut StaticStudy<Material>, elements: usize) {
        let mut recovery = match self.preview.take() {
            Some(recovery) => recovery,
            None => study.begin_recovery(),
        };
        let done = study.continue_recovery(&mut recovery, elements, &mut || false);
        match done {
            Ok(true) => {
                if let Ok(field) = study.finish_recovery(recovery) {
                    self.nodal_von_mises = field.nodal_von_mises;
                }
            }
            Ok(false) => self.preview = Some(recovery),
            Err(e) => self.fail(&e),
        }
    }

    fn finish(&mut self, field: RecoveredField) {
        self.displacements = field.displacements;
        self.nodal_von_mises = field.nodal_von_mises;
        self.preview = None;
        self.update_rows();
        self.phase = Phase::Done;
    }

    /// Rebuilds the comparison rows from the current displacement estimate.
    fn update_rows(&mut self) {
        let Textbook::Answers(comparisons) = &self.textbook else {
            return;
        };
        self.rows = comparisons
            .iter()
            .map(|comparison| {
                let fea = measure(comparison, &self.mesh, &self.displacements, &self.material);
                ComparisonRow {
                    quantity: quantity_name(comparison),
                    unit: comparison.unit,
                    place: comparison.place.clone(),
                    fea,
                    textbook: comparison.textbook,
                    percent_difference: 100.0 * (fea - comparison.textbook) / comparison.textbook,
                }
            })
            .collect();
    }

    /// Whether the estimate in `view` is inside what the linear model can describe.
    pub fn validity(&self, view: &ViewData) -> Validity {
        assess(&self.request, self.yield_strength, view)
    }

    /// The arrays and summary for the current estimate.
    pub fn view(&self) -> ViewData {
        self.surface.view(
            &self.mesh,
            &self.displacements,
            &self.nodal_von_mises,
            &self.clamps,
            self.clamp_zone,
            self.yield_strength,
        )
    }
}

fn quantity_name(comparison: &Comparison) -> &'static str {
    match comparison.quantity {
        crate::plan::Quantity::Deflection => "Deflection",
        crate::plan::Quantity::BendingStress => "Bending stress",
        crate::plan::Quantity::TwistRate => "Twist rate",
    }
}

fn measure(
    comparison: &Comparison,
    mesh: &QuadraticTetMesh,
    displacements: &[[f64; 3]],
    material: &Material,
) -> f64 {
    match &comparison.probe {
        Probe::Deflection { nodes } => {
            let sum: f64 = nodes.iter().map(|n| displacements[*n][2]).sum();
            -sum / nodes.len() as f64
        }
        Probe::Fibre {
            points,
            centre_z,
            station_x,
            half_height,
        } => {
            // sigma_xx = a + b (z - centre) + c (x - station): least squares over the slab points
            // (the design is well conditioned and the normal equations small).
            let mut ata = [[0.0; 3]; 3];
            let mut atb = [0.0; 3];
            for point in points {
                let tet = mesh.tets()[point.element as usize];
                let u = tet.map(|n| displacements[n as usize]);
                let stress = stress_at(material, &strain_at(&point.geometry, &u));
                let row = [
                    1.0,
                    point.geometry.position[2] - centre_z,
                    point.geometry.position[0] - station_x,
                ];
                for i in 0..3 {
                    atb[i] += row[i] * stress[0];
                    for j in 0..3 {
                        ata[i][j] += row[i] * row[j];
                    }
                }
            }
            solve3(ata, atb)[1].abs() * half_height
        }
        Probe::TwistRate {
            first,
            second,
            first_x,
            second_x,
            centre,
        } => {
            let twist = |nodes: &[usize]| {
                let (mut numerator, mut denominator) = (0.0, 0.0);
                for n in nodes {
                    let p = mesh.nodes()[*n];
                    let (y, z) = (p[1] - centre[0], p[2] - centre[1]);
                    let u = displacements[*n];
                    numerator += y * u[2] - z * u[1];
                    denominator += y * y + z * z;
                }
                numerator / denominator
            };
            (twist(second) - twist(first)) / (second_x - first_x)
        }
    }
}

/// Solves a 3x3 symmetric positive definite system by Gaussian elimination with partial pivoting;
/// a singular system (no slab points, impossible on a valid grid) yields zeros.
#[allow(clippy::needless_range_loop)] // Index pairs across rows are the elimination itself.
fn solve3(mut m: [[f64; 3]; 3], mut b: [f64; 3]) -> [f64; 3] {
    for col in 0..3 {
        let mut pivot = col;
        for row in col + 1..3 {
            if m[row][col].abs() > m[pivot][col].abs() {
                pivot = row;
            }
        }
        m.swap(col, pivot);
        b.swap(col, pivot);
        if m[col][col] == 0.0 {
            return [0.0; 3];
        }
        for row in col + 1..3 {
            let factor = m[row][col] / m[col][col];
            for k in col..3 {
                m[row][k] -= factor * m[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; 3];
    for col in (0..3).rev() {
        let mut sum = b[col];
        for k in col + 1..3 {
            sum -= m[col][k] * x[k];
        }
        x[col] = sum / m[col][col];
    }
    x
}

impl std::fmt::Debug for DemoStudy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DemoStudy({:?})", self.state())
    }
}
