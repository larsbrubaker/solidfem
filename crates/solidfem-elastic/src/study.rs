//! A static study: the model a caller describes and the resumable solve over it. `new` validates
//! and assembles, `step` runs bounded slices of the conjugate gradient solve and `solution`
//! recovers results. The caller's clock (no `std::time` here, it panics on wasm32) times each
//! phase so a preconditioner's or a slicing decision's effect can be measured.

use std::collections::BTreeSet;

use solidfem_mesh::{BoundaryFace, QuadraticTetMesh};
use solidfem_solve::{Jacobi, Pcg, PcgOptions, StepOutcome};

use crate::assembly::{assemble_free, FIXED};
use crate::constraints::check_constraints;
use crate::error::ModelError;
use crate::loads::add_traction;
use crate::material::MaterialField;
use crate::quadrature::Quadrature;
use crate::results::{recover, Solution};

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

/// Settings for a study.
#[derive(Default)]
pub struct StudyOptions {
    /// Solver settings.
    pub pcg: PcgOptions,
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
    /// Seconds in `step`, over all slices.
    pub solve_seconds: f64,
    /// Seconds recovering stresses.
    pub recovery_seconds: f64,
}

/// A study in progress.
pub struct StaticStudy<M: MaterialField> {
    model: Model<M>,
    free: Vec<u32>,
    pcg: Pcg<Jacobi>,
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

        let matrix = assemble_free(
            &model.mesh,
            &model.material,
            model.quadrature,
            &free,
            free_count,
            cancel,
        )?;
        let report = Report {
            total_dofs: 3 * node_count,
            free_dofs: 3 * free_count,
            nonzeros: matrix.nonzeros(),
            at_rounding_floor: false,
            iterations: 0,
            relative_residual: 1.0,
            assembly_seconds: elapsed(&mut clock, start),
            solve_seconds: 0.0,
            recovery_seconds: 0.0,
        };
        let jacobi = Jacobi::new(&matrix)?;
        let pcg = Pcg::new(matrix, jacobi, rhs, options.pcg);
        Ok(StaticStudy {
            model,
            free,
            pcg,
            clock,
            report,
        })
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
        let start = now(&mut self.clock);
        let displacements = self.displacements();
        let (point_stresses, nodal_von_mises) = recover(
            &self.model.mesh,
            &self.model.material,
            self.model.quadrature,
            &displacements,
            cancel,
        )?;
        self.report.recovery_seconds += elapsed(&mut self.clock, start);
        Ok(Solution {
            displacements,
            point_stresses,
            nodal_von_mises,
            report: self.report,
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
