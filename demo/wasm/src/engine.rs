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
//! estimate (recoveries are themselves sliced); displacement follows every step, and the local
//! rotation is computed from it when the page next asks, once per estimate, because a view is
//! rebuilt for every array the page reads.
//!
//! A study is of one of two subjects: a bar from the catalog, with its clamps and its textbook, or
//! an example part (`example_study.rs` meshes it, then starts one of these), where stress is left
//! out of the peak only at the elements touching a node the case fixes outright, and the results
//! scale linearly with the visitor's load size without any re-solve.

use std::cell::RefCell;

use solidfem::elastic::{
    nodal_rotations, peak_von_mises, strain_at, stress_at, Material, Model, ModelError, PcgOptions,
    PointStress, PreconditionerChoice, RecoveredField, Recovery, StaticStudy, StepOutcome,
    StudyBuilder, StudyOptions,
};
use solidfem::mesh::QuadraticTetMesh;

use crate::catalog::StudyRequest;
use crate::plan::{plan_study, Comparison, NoTextbookAnswer, PlanError, Probe, Textbook};
use crate::validity::{assess, one_end_deflection_limit, Validity};
use crate::view::{StressPeaks, Surface, ViewData};

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

/// What is being studied; it decides what counts towards the peak and what the validity checks.
enum Subject {
    Bar(StudyRequest),
    Part(PartSubject),
}

/// What an example part's study needs beyond the solve itself.
struct PartSubject {
    /// The part's largest extent in mm, the length the deflection limit is read from.
    size: f64,
    /// Nodes left out of the peak stress: those fixed outright and the elements touching them.
    excluded: Vec<bool>,
    /// The elements touching a node fixed outright, whose integration points the peak skips.
    excluded_elements: Vec<bool>,
    /// The factor on the benchmark's load the results are shown at.
    load_scale: f64,
}

/// A meshed example part ready to solve: `example_study.rs` builds one.
pub struct PartStart {
    pub mesh: QuadraticTetMesh,
    pub model: Model<Material>,
    pub material: Material,
    /// The material's yield stress, if the chosen material has one on this page.
    pub yield_strength: Option<f64>,
    pub size: f64,
    pub excluded: Vec<bool>,
    pub excluded_elements: Vec<bool>,
    pub load_scale: f64,
    pub preconditioner: PreconditionerChoice,
}

/// A part whose surface and geometry check are done, waiting for its matrix pattern.
pub struct ReadyPart {
    part: PartStart,
    surface: Surface,
    rotations: Vec<[f64; 3]>,
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
    subject: Subject,
    material: Material,
    yield_strength: Option<f64>,
    mesh: QuadraticTetMesh,
    textbook: Textbook,
    clamps: Vec<f64>,
    clamp_zone: f64,
    surface: Surface,
    phase: Phase,
    cancelled: bool,
    displacements: Vec<[f64; 3]>,
    /// The local rotation of the current estimate, once something has asked for it.
    rotations: RefCell<Option<Vec<[f64; 3]>>>,
    /// The latest finished stress recovery of the estimate, for the colours.
    nodal_von_mises: Vec<f64>,
    /// The peaks of that recovery, which also hold its integration points.
    peaks: StressPeaks,
    /// Every integration point's stress once the study is done, for checking the load balance.
    point_stresses: Vec<PointStress>,
    /// The recovered stress tensor at every mesh node once the study is done, for those checks.
    nodal_stress: Vec<[f64; 6]>,
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
        // Refused here, by name, so the rotation of a later estimate has no way left to fail.
        let rotations = nodal_rotations(&planned.mesh, &vec![[0.0; 3]; node_count])
            .map_err(PlanError::Model)?;
        let builder = StudyBuilder::new(
            planned.model,
            StudyOptions {
                pcg: PcgOptions::default(),
                ..StudyOptions::default()
            },
        )
        .map_err(PlanError::Model)?;
        let mut study = DemoStudy {
            subject: Subject::Bar(request),
            material: planned.material,
            yield_strength: Some(planned.yield_strength),
            mesh: planned.mesh,
            textbook: planned.textbook,
            clamps: planned.clamps,
            clamp_zone: planned.clamp_zone,
            surface,
            phase: Phase::Assembling(Box::new(builder)),
            cancelled: false,
            displacements: vec![[0.0; 3]; node_count],
            rotations: RefCell::new(Some(rotations)),
            nodal_von_mises: vec![0.0; node_count],
            peaks: StressPeaks::default(),
            point_stresses: Vec::new(),
            nodal_stress: Vec::new(),
            preview: None,
            rows: Vec::new(),
            iterations: 0,
            residual: 1.0,
            failure: None,
        };
        study.update_rows();
        Ok(study)
    }

    /// Extracts a part's surface and checks its element geometry: the first half of starting a
    /// part, a frame's work of its own on the page.
    pub fn ready_part(part: PartStart) -> Result<ReadyPart, ModelError> {
        let surface = Surface::of(&part.mesh);
        let node_count = part.mesh.nodes().len();
        let rotations = nodal_rotations(&part.mesh, &vec![[0.0; 3]; node_count])?;
        Ok(ReadyPart {
            part,
            surface,
            rotations,
        })
    }

    /// Starts the study of a ready part, building its matrix pattern and load; a study the model
    /// refuses is a named error.
    pub fn start_part(ready: ReadyPart) -> Result<Self, ModelError> {
        let ReadyPart {
            part,
            surface,
            rotations,
        } = ready;
        let node_count = part.mesh.nodes().len();
        let builder = StudyBuilder::new(
            part.model,
            StudyOptions {
                pcg: PcgOptions::default(),
                preconditioner: part.preconditioner,
                ..StudyOptions::default()
            },
        )?;
        Ok(DemoStudy {
            subject: Subject::Part(PartSubject {
                size: part.size,
                excluded: part.excluded,
                excluded_elements: part.excluded_elements,
                load_scale: part.load_scale,
            }),
            material: part.material,
            yield_strength: part.yield_strength,
            mesh: part.mesh,
            textbook: Textbook::NoTextbook(NoTextbookAnswer::ExamplePart),
            clamps: Vec::new(),
            clamp_zone: 0.0,
            surface,
            phase: Phase::Assembling(Box::new(builder)),
            cancelled: false,
            displacements: vec![[0.0; 3]; node_count],
            rotations: RefCell::new(Some(rotations)),
            nodal_von_mises: vec![0.0; node_count],
            peaks: StressPeaks::default(),
            point_stresses: Vec::new(),
            nodal_stress: Vec::new(),
            preview: None,
            rows: Vec::new(),
            iterations: 0,
            residual: 1.0,
            failure: None,
        })
    }

    /// Sets the factor on the benchmark's load an example part's results are shown at; a linear
    /// solve's answer scales with its load, so nothing is solved again. Ignored for a bar.
    pub fn set_load_scale(&mut self, factor: f64) {
        if let Subject::Part(part) = &mut self.subject {
            part.load_scale = factor;
        }
    }

    /// The request this study was started with, when it is a bar.
    pub fn request(&self) -> Option<&StudyRequest> {
        match &self.subject {
            Subject::Bar(request) => Some(request),
            Subject::Part(_) => None,
        }
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

    /// The displacement of every mesh node at the current estimate, at the benchmark's load.
    pub fn displacements(&self) -> &[[f64; 3]] {
        &self.displacements
    }

    /// The volume-weighted von Mises stress of every mesh node at the benchmark's load.
    pub fn nodal_von_mises(&self) -> &[f64] {
        &self.nodal_von_mises
    }

    /// The recovered stress tensor at every node once the study is done; empty before that.
    pub fn nodal_stress(&self) -> &[[f64; 6]] {
        &self.nodal_stress
    }

    /// Every integration point's stress once the study is done; empty before that.
    pub fn point_stresses(&self) -> &[PointStress] {
        &self.point_stresses
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
                Ok(progress) if progress.setup_finished => match builder.finish() {
                    Ok(study) => self.phase = Phase::Solving(Box::new(study)),
                    Err(e) => self.fail(&e),
                },
                Ok(_) => self.phase = Phase::Assembling(builder),
                Err(e) => self.fail(&e),
            },
            Phase::Solving(mut study) => match study.step(budget, &mut no_cancel) {
                Ok(StepOutcome::Converged(progress)) => {
                    self.iterations = progress.iteration;
                    self.residual = progress.relative_residual;
                    self.set_displacements(study.displacements());
                    self.update_rows();
                    let recovery = study.begin_recovery();
                    self.phase = Phase::Recovering(study, Box::new(recovery));
                }
                Ok(StepOutcome::Running(progress)) => {
                    self.iterations = progress.iteration;
                    self.residual = progress.relative_residual;
                    self.set_displacements(study.displacements());
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

    /// Replaces the estimate; its rotation is computed again when the page next asks.
    fn set_displacements(&mut self, displacements: Vec<[f64; 3]>) {
        self.displacements = displacements;
        *self.rotations.get_mut() = None;
    }

    /// The local rotation at each node of the current estimate. The mesh's element geometry was
    /// accepted by `nodal_rotations` in `start` and the geometry is all that can fail, so the
    /// zeros (which would draw the plain straight-line motion) are never reached.
    fn rotations(&self) -> Vec<[f64; 3]> {
        self.rotations
            .borrow_mut()
            .get_or_insert_with(|| {
                nodal_rotations(&self.mesh, &self.displacements)
                    .unwrap_or_else(|_| vec![[0.0; 3]; self.displacements.len()])
            })
            .clone()
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
                    self.peaks = self.peaks_of(&field);
                    self.nodal_von_mises = field.nodal_von_mises;
                }
            }
            Ok(false) => self.preview = Some(recovery),
            Err(e) => self.fail(&e),
        }
    }

    /// The peaks of a finished recovery, from the library's peak measure, which holds the
    /// integration points as well as the recovered nodes. A bar's peak excludes what lies within
    /// a clamp's zone by physical position, because a clamp's corners are a singularity whose
    /// value does not converge. An example part's excludes the region its full fixtures reach
    /// (`PartSubject`): those elements' integration points and their nodes.
    fn peaks_of(&self, field: &RecoveredField) -> StressPeaks {
        let overall = field.peak_von_mises(&self.mesh, |_| true).unwrap_or(0.0);
        let outside_clamps = match &self.subject {
            Subject::Bar(_) => {
                let (clamps, zone) = (&self.clamps, self.clamp_zone);
                field
                    .peak_von_mises(&self.mesh, |p| {
                        clamps.iter().all(|x| (p[0] - x).abs() > zone)
                    })
                    .unwrap_or(0.0)
            }
            Subject::Part(part) => {
                // Nodes of the held region read as no stress, which the peak can never choose.
                let nodal: Vec<f64> = field
                    .nodal_von_mises
                    .iter()
                    .zip(&part.excluded)
                    .map(|(value, held)| if *held { 0.0 } else { *value })
                    .collect();
                let points: Vec<PointStress> = field
                    .point_stresses
                    .iter()
                    .filter(|point| !part.excluded_elements[point.tet as usize])
                    .copied()
                    .collect();
                peak_von_mises(&self.mesh, &nodal, &points, |_| true).unwrap_or(0.0)
            }
        };
        StressPeaks {
            outside_clamps,
            overall,
        }
    }

    fn finish(&mut self, field: RecoveredField) {
        self.peaks = self.peaks_of(&field);
        self.set_displacements(field.displacements);
        self.nodal_von_mises = field.nodal_von_mises;
        self.point_stresses = field.point_stresses;
        self.nodal_stress = field.nodal_stress;
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
        match &self.subject {
            Subject::Bar(request) => {
                assess(request, self.yield_strength.unwrap_or(f64::INFINITY), view)
            }
            Subject::Part(part) => {
                let limit = one_end_deflection_limit(part.size);
                Validity {
                    exceeds_yield: self
                        .yield_strength
                        .is_some_and(|y| view.peak_stress_outside_clamps > y),
                    excessive_deflection: view.max_displacement > limit,
                    excessive_twist: false,
                    deflection_limit: Some(limit),
                    twist_radians: None,
                    twist_limit_radians: None,
                }
            }
        }
    }

    /// The arrays and summary for the current estimate.
    pub fn view(&self) -> ViewData {
        let rotations = self.rotations();
        let yield_strength = self.yield_strength.unwrap_or(f64::INFINITY);
        match &self.subject {
            Subject::Bar(_) => self.surface.view(
                &self.mesh,
                &self.displacements,
                &rotations,
                &self.nodal_von_mises,
                self.peaks,
                yield_strength,
            ),
            Subject::Part(part) => self.part_view(part, &rotations, yield_strength),
        }
    }

    /// An example part's view: every field and peak is the benchmark-load solve times the load
    /// size, and a material with no stated yield has no safety factor.
    fn part_view(
        &self,
        part: &PartSubject,
        rotations: &[[f64; 3]],
        yield_strength: f64,
    ) -> ViewData {
        let f = part.load_scale;
        let times = |v: &[f64; 3]| [v[0] * f, v[1] * f, v[2] * f];
        let u: Vec<[f64; 3]> = self.displacements.iter().map(times).collect();
        let w: Vec<[f64; 3]> = rotations.iter().map(times).collect();
        let vm: Vec<f64> = self.nodal_von_mises.iter().map(|s| s * f).collect();
        let peaks = StressPeaks {
            outside_clamps: self.peaks.outside_clamps * f,
            overall: self.peaks.overall * f,
        };
        let mut view = self
            .surface
            .view(&self.mesh, &u, &w, &vm, peaks, yield_strength);
        if self.yield_strength.is_none() {
            view.safety_factor = None;
        }
        view
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
