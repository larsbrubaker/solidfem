//! A static study: the model a caller describes and the resumable solve over it. `new` validates
//! and assembles, `step` runs bounded slices of the conjugate gradient solve and `solution`
//! recovers results. A host with no threads uses the resumable forms of the other two phases:
//! [`StudyBuilder`] assembles in bounded slices of elements, and
//! [`StaticStudy::begin_recovery`] with [`StaticStudy::continue_recovery`] recovers in bounded
//! slices, on the converged solution or on a live estimate. Each gives exactly the bits of the
//! uninterrupted call, which runs the same code. The caller's clock (no `std::time` here, it
//! panics on wasm32) times each phase so a preconditioner's or a slicing decision's effect can be
//! measured.
//!
//! [`StudyOptions::preconditioner`] chooses the preconditioner. Jacobi needs no setup; multigrid
//! is built after assembly from the assembled matrix and rigid-body modes taken from the node
//! coordinates, restricted to the free nodes, and [`StudyBuilder::step`] advances that setup in
//! the same slices as assembly.

use std::collections::BTreeSet;

use solidfem_mesh::{BoundaryFace, QuadraticTetMesh};
use solidfem_solve::{
    CsrMatrix, Jacobi, Multigrid, MultigridOptions, MultigridSetup, MultigridStats, Pcg,
    PcgOptions, Preconditioner, SetupStatus, StepOutcome,
};

use crate::assembly::{add_element_stiffness, free_stiffness_pattern, FIXED};
use crate::constraints::check_constraints;
use crate::error::ModelError;
use crate::loads::add_traction;
use crate::material::MaterialField;
use crate::quadrature::Quadrature;
use crate::results::{RecoveredField, Recovery, Solution};
use crate::rigid_modes::free_rigid_body_modes;

/// A caller-supplied monotonic clock, in seconds.
pub type Clock = Box<dyn FnMut() -> f64>;

/// A structure, its material, its supports and its loads.
pub struct Model<M: MaterialField> {
    mesh: QuadraticTetMesh,
    material: M,
    quadrature: Quadrature,
    fixed: BTreeSet<u32>,
    nodal_forces: Vec<(u32, [f64; 3])>,
    tractions: Vec<(BoundaryFace, [f64; 3])>,
}

impl<M: MaterialField> Model<M> {
    /// A model with no supports or loads yet, using the 4-point rule.
    pub fn new(mesh: QuadraticTetMesh, material: M) -> Self {
        Model {
            mesh,
            material,
            quadrature: Quadrature::default(),
            fixed: BTreeSet::new(),
            nodal_forces: Vec::new(),
            tractions: Vec::new(),
        }
    }

    /// The mesh.
    pub fn mesh(&self) -> &QuadraticTetMesh {
        &self.mesh
    }

    /// Chooses the integration rule.
    pub fn set_quadrature(&mut self, quadrature: Quadrature) {
        self.quadrature = quadrature;
    }

    /// Fixes all three displacement components of `node`.
    pub fn fix_node(&mut self, node: u32) {
        self.fixed.insert(node);
    }

    /// Adds a force at `node`.
    pub fn add_nodal_force(&mut self, node: u32, force: [f64; 3]) {
        self.nodal_forces.push((node, force));
    }

    /// Adds a uniform traction (force per area) over a boundary face.
    pub fn add_traction(&mut self, face: BoundaryFace, traction: [f64; 3]) {
        self.tractions.push((face, traction));
    }
}

/// Which preconditioner a study solves with.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PreconditionerChoice {
    /// Diagonal scaling: no setup, and enough for compact parts.
    #[default]
    Jacobi,
    /// Block smoothed aggregation multigrid over the rigid-body modes, for large or slender models
    /// diagonal scaling cannot bring to convergence.
    Multigrid(MultigridOptions),
}

/// The preconditioner behind a study's solve.
enum StudyPreconditioner {
    Jacobi(Jacobi),
    Multigrid(Box<Multigrid>),
}

impl Preconditioner for StudyPreconditioner {
    fn apply(&self, residual: &[f64], output: &mut [f64]) {
        match self {
            StudyPreconditioner::Jacobi(jacobi) => jacobi.apply(residual, output),
            StudyPreconditioner::Multigrid(multigrid) => multigrid.apply(residual, output),
        }
    }
}

/// Settings for a study.
#[derive(Default)]
pub struct StudyOptions {
    /// Solver settings.
    pub pcg: PcgOptions,
    /// The preconditioner to solve with; Jacobi by default.
    pub preconditioner: PreconditionerChoice,
    /// The clock timing each phase; without one the reported times are zero.
    pub clock: Option<Clock>,
}

/// What a study did: sizes, iterations and the time each phase took.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Report {
    /// Degrees of freedom before fixed nodes are removed.
    pub total_dofs: usize,
    /// Degrees of freedom solved for.
    pub free_dofs: usize,
    /// Stored entries in the system matrix.
    pub nonzeros: usize,
    /// True when the solve stopped at the rounding floor because the tolerance was below what f64
    /// can resolve for this model; `relative_residual` is what was reached.
    pub at_rounding_floor: bool,
    /// Conjugate gradient iterations.
    pub iterations: u64,
    /// Final relative residual.
    pub relative_residual: f64,
    /// Seconds validating, assembling and building the load.
    pub assembly_seconds: f64,
    /// Seconds building the preconditioner, zero for Jacobi.
    pub preconditioner_seconds: f64,
    /// The multigrid hierarchy's shape and cost, when multigrid was chosen.
    pub multigrid: Option<MultigridStats>,
    /// Seconds in `step`, over all slices.
    pub solve_seconds: f64,
    /// Seconds recovering stresses.
    pub recovery_seconds: f64,
}

/// A study in progress.
pub struct StaticStudy<M: MaterialField> {
    model: Model<M>,
    free: Vec<u32>,
    pcg: Pcg<StudyPreconditioner>,
    clock: Option<Clock>,
    report: Report,
}

impl<M: MaterialField> StaticStudy<M> {
    /// Validates the model and assembles its system. Refuses a model that is not held or has no
    /// effective load before any solving; `cancel` is read per element during assembly.
    pub fn new(
        model: Model<M>,
        options: StudyOptions,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Self, ModelError> {
        let mut builder = StudyBuilder::new(model, options)?;
        builder.step(usize::MAX, cancel)?;
        builder.finish()
    }

    /// Runs up to `max_iterations` of the solve; identical bits however the work is sliced.
    pub fn step(
        &mut self,
        max_iterations: usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<StepOutcome, ModelError> {
        let start = now(&mut self.clock);
        let outcome = self.pcg.step(max_iterations, cancel);
        self.report.solve_seconds += elapsed(&mut self.clock, start);
        let progress = self.pcg.progress();
        self.report.iterations = progress.iteration;
        self.report.relative_residual = progress.relative_residual;
        self.report.at_rounding_floor = progress.at_rounding_floor;
        Ok(outcome?)
    }

    /// The current displacement estimate of every mesh node; fixed nodes are zero. Partial until
    /// the solve converges, so it is a preview, not a result.
    pub fn displacements(&self) -> Vec<[f64; 3]> {
        let x = self.pcg.estimate();
        self.free
            .iter()
            .map(|index| {
                if *index == FIXED {
                    [0.0; 3]
                } else {
                    let at = 3 * *index as usize;
                    [x[at], x[at + 1], x[at + 2]]
                }
            })
            .collect()
    }

    /// Sizes, iterations and times so far.
    pub fn report(&self) -> Report {
        self.report
    }

    /// Recovers stresses; refuses until the solve has converged. `cancel` is read per element.
    pub fn solution(&mut self, cancel: &mut dyn FnMut() -> bool) -> Result<Solution, ModelError> {
        if !self.pcg.is_converged() {
            return Err(ModelError::NotFinished);
        }
        let mut recovery = self.begin_recovery();
        self.continue_recovery(&mut recovery, usize::MAX, cancel)?;
        let field = self.finish_recovery(recovery)?;
        Ok(Solution {
            displacements: field.displacements,
            point_stresses: field.point_stresses,
            nodal_von_mises: field.nodal_von_mises,
            report: self.report,
        })
    }

    /// Starts a resumable recovery of the current displacements: the converged answer, or a
    /// partial estimate for a preview while the solve is still running.
    pub fn begin_recovery(&self) -> Recovery {
        Recovery::new(
            self.displacements(),
            self.model.mesh.nodes().len(),
            self.model.mesh.tets().len(),
            self.model.quadrature.points().len(),
        )
    }

    /// Recovers up to `max_elements` more elements; `Ok(true)` once all are done. `cancel` is read
    /// per element.
    pub fn continue_recovery(
        &mut self,
        recovery: &mut Recovery,
        max_elements: usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<bool, ModelError> {
        let start = now(&mut self.clock);
        let result = recovery.advance(
            &self.model.mesh,
            &self.model.material,
            self.model.quadrature,
            max_elements,
            cancel,
        );
        self.report.recovery_seconds += elapsed(&mut self.clock, start);
        result?;
        Ok(recovery.elements_done() == self.model.mesh.tets().len())
    }

    /// The field of a finished recovery; refuses one with elements left to do.
    pub fn finish_recovery(&self, recovery: Recovery) -> Result<RecoveredField, ModelError> {
        if recovery.elements_done() != self.model.mesh.tets().len() {
            return Err(ModelError::NotFinished);
        }
        Ok(recovery.into_field())
    }

    /// The mesh being solved.
    pub fn mesh(&self) -> &QuadraticTetMesh {
        self.model.mesh()
    }

    /// The multigrid hierarchy, when multigrid was chosen, for inspecting what it built.
    pub fn multigrid(&self) -> Option<&Multigrid> {
        match self.pcg.preconditioner() {
            StudyPreconditioner::Multigrid(multigrid) => Some(multigrid),
            StudyPreconditioner::Jacobi(_) => None,
        }
    }

    /// The material being solved.
    pub fn material(&self) -> &M {
        &self.model.material
    }
}

/// How far a [`StudyBuilder`] has got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuildProgress {
    /// Elements whose stiffness has been added.
    pub elements_done: usize,
    /// Elements in the mesh.
    pub elements_total: usize,
    /// True once any preconditioner setup is finished too, so [`StudyBuilder::finish`] will
    /// succeed; always true for Jacobi once the elements are done.
    pub setup_finished: bool,
}

/// A resumable [`StaticStudy::new`]: [`StudyBuilder::new`] validates and builds the load and the
/// matrix pattern, and [`StudyBuilder::step`] adds element stiffness and then builds the
/// preconditioner in bounded slices. Stepped in any slice sizes it builds exactly the bits of
/// `new`, which runs the same code in one slice.
pub struct StudyBuilder<M: MaterialField> {
    model: Model<M>,
    free: Vec<u32>,
    free_count: usize,
    rhs: Vec<f64>,
    matrix: CsrMatrix,
    ke: [f64; 900],
    next: usize,
    pcg_options: PcgOptions,
    preconditioner: PreconditionerChoice,
    setup: Option<MultigridSetup>,
    setup_status: SetupStatus,
    clock: Option<Clock>,
    assembly_seconds: f64,
    preconditioner_seconds: f64,
}

impl<M: MaterialField> StudyBuilder<M> {
    /// Validates the model and prepares its assembly. Refuses a model that is not held or has no
    /// effective load before any element is integrated.
    pub fn new(model: Model<M>, options: StudyOptions) -> Result<Self, ModelError> {
        let mut clock = options.clock;
        let start = now(&mut clock);
        let node_count = model.mesh.nodes().len();
        let mut is_fixed = vec![false; node_count];
        for node in &model.fixed {
            *is_fixed
                .get_mut(*node as usize)
                .ok_or(ModelError::NodeOutOfRange { node: *node })? = true;
        }
        check_constraints(&model.mesh, &is_fixed)?;
        let load = build_load(&model)?;

        let mut free = vec![FIXED; node_count];
        let mut free_count = 0usize;
        for (node, fixed) in is_fixed.iter().enumerate() {
            if !fixed {
                free[node] = free_count as u32;
                free_count += 1;
            }
        }
        let mut rhs = Vec::with_capacity(3 * free_count);
        for (node, index) in free.iter().enumerate() {
            if *index != FIXED {
                rhs.extend_from_slice(&load[3 * node..3 * node + 3]);
            }
        }
        if rhs.iter().all(|v| *v == 0.0) {
            return Err(ModelError::NoEffectiveLoad);
        }
        let matrix = free_stiffness_pattern(&model.mesh, &free, free_count);
        let assembly_seconds = elapsed(&mut clock, start);
        Ok(StudyBuilder {
            model,
            free,
            free_count,
            rhs,
            matrix,
            ke: [0.0; 900],
            next: 0,
            pcg_options: options.pcg,
            preconditioner: options.preconditioner,
            setup: None,
            setup_status: SetupStatus::Running,
            clock,
            assembly_seconds,
            preconditioner_seconds: 0.0,
        })
    }

    /// Adds the stiffness of up to `max_elements` more elements, then advances any multigrid
    /// setup by up to `max_elements` block rows; `cancel` is read per element and per row.
    pub fn step(
        &mut self,
        max_elements: usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<BuildProgress, ModelError> {
        let start = now(&mut self.clock);
        let total = self.model.mesh.tets().len();
        let end = self.next.saturating_add(max_elements).min(total);
        let result = add_element_stiffness(
            &self.model.mesh,
            &self.model.material,
            self.model.quadrature,
            &self.free,
            &mut self.matrix,
            &mut self.ke,
            self.next..end,
            cancel,
        );
        self.assembly_seconds += elapsed(&mut self.clock, start);
        result?;
        self.next = end;
        if self.next == total {
            if let PreconditionerChoice::Multigrid(options) = self.preconditioner {
                self.advance_setup(options, max_elements, cancel)?;
            }
        }
        Ok(BuildProgress {
            elements_done: self.next,
            elements_total: total,
            setup_finished: self.next == total && self.setup_is_finished(),
        })
    }

    /// True when no preconditioner setup is outstanding.
    fn setup_is_finished(&self) -> bool {
        match self.preconditioner {
            PreconditionerChoice::Jacobi => true,
            PreconditionerChoice::Multigrid(_) => self.setup_status == SetupStatus::Finished,
        }
    }

    /// Starts the multigrid setup on first call and runs up to `budget` rows of it.
    fn advance_setup(
        &mut self,
        options: MultigridOptions,
        budget: usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<(), ModelError> {
        if self.setup_status == SetupStatus::Finished {
            return Ok(());
        }
        let start = now(&mut self.clock);
        if self.setup.is_none() {
            let modes = free_rigid_body_modes(self.model.mesh.nodes(), &self.free, self.free_count);
            let positions: Vec<[f64; 3]> = self
                .model
                .mesh
                .nodes()
                .iter()
                .zip(&self.free)
                .filter(|(_, index)| **index != FIXED)
                .map(|(point, _)| *point)
                .collect();
            self.setup = Some(MultigridSetup::new(
                &self.matrix,
                &modes,
                &positions,
                options,
            )?);
        }
        let status = match &mut self.setup {
            Some(setup) => setup.step(budget, cancel),
            None => Ok(SetupStatus::Running),
        };
        self.preconditioner_seconds += elapsed(&mut self.clock, start);
        self.setup_status = status?;
        Ok(())
    }

    /// The mesh being assembled.
    pub fn mesh(&self) -> &QuadraticTetMesh {
        self.model.mesh()
    }

    /// The assembled study, ready to solve; refuses while elements remain.
    pub fn finish(self) -> Result<StaticStudy<M>, ModelError> {
        if self.next != self.model.mesh.tets().len() || !self.setup_is_finished() {
            return Err(ModelError::NotFinished);
        }
        let StudyBuilder {
            model,
            free,
            free_count,
            rhs,
            matrix,
            pcg_options,
            setup,
            mut clock,
            assembly_seconds,
            preconditioner_seconds,
            ..
        } = self;
        let start = now(&mut clock);
        let mut multigrid_stats = None;
        let preconditioner = match setup {
            Some(setup) => {
                let multigrid = Box::new(setup.finish()?);
                multigrid_stats = Some(multigrid.stats());
                StudyPreconditioner::Multigrid(multigrid)
            }
            None => StudyPreconditioner::Jacobi(Jacobi::new(&matrix)?),
        };
        let nonzeros = matrix.nonzeros();
        let pcg = Pcg::new(matrix, preconditioner, rhs, pcg_options);
        let report = Report {
            total_dofs: 3 * model.mesh.nodes().len(),
            free_dofs: 3 * free_count,
            nonzeros,
            at_rounding_floor: false,
            iterations: 0,
            relative_residual: 1.0,
            assembly_seconds: assembly_seconds + elapsed(&mut clock, start),
            preconditioner_seconds,
            multigrid: multigrid_stats,
            solve_seconds: 0.0,
            recovery_seconds: 0.0,
        };
        Ok(StaticStudy {
            model,
            free,
            pcg,
            clock,
            report,
        })
    }
}

/// Reads the clock, or zero with none.
fn now(clock: &mut Option<Clock>) -> f64 {
    clock.as_mut().map_or(0.0, |read| read())
}

/// Seconds since `start`, zero with no clock.
fn elapsed(clock: &mut Option<Clock>, start: f64) -> f64 {
    clock.as_mut().map_or(0.0, |read| read() - start)
}

/// The full load vector, refusing bad nodes, non-finite loads and interior faces.
fn build_load<M: MaterialField>(model: &Model<M>) -> Result<Vec<f64>, ModelError> {
    let node_count = model.mesh.nodes().len();
    let mut load = vec![0.0; 3 * node_count];
    for (node, force) in &model.nodal_forces {
        if *node as usize >= node_count {
            return Err(ModelError::NodeOutOfRange { node: *node });
        }
        if force.iter().any(|c| !c.is_finite()) {
            return Err(ModelError::NonFiniteLoad);
        }
        for axis in 0..3 {
            load[3 * *node as usize + axis] += force[axis];
        }
    }
    if !model.tractions.is_empty() {
        let mut boundary: Vec<(u32, u8, [u32; 6])> = model
            .mesh
            .boundary_faces()
            .iter()
            .map(|f| (f.tet, f.face, f.nodes))
            .collect();
        boundary.sort_unstable();
        for (face, traction) in &model.tractions {
            let on_boundary = boundary
                .binary_search(&(face.tet, face.face, face.nodes))
                .is_ok();
            if !on_boundary {
                return Err(ModelError::FaceNotOnBoundary {
                    tet: face.tet,
                    face: face.face,
                });
            }
            if traction.iter().any(|c| !c.is_finite()) {
                return Err(ModelError::NonFiniteLoad);
            }
            add_traction(&model.mesh, face, *traction, &mut load);
        }
    }
    Ok(load)
}
