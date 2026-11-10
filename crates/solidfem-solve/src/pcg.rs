//! The resumable preconditioned conjugate gradient solver. All iteration state lives in [`Pcg`],
//! so [`Pcg::step`] can run a bounded number of iterations, return, and be called again later
//! (a browser host has no threads and solves between frames) with exactly the same sequence of
//! floating-point operations as an uninterrupted run. The stopping test is on the true residual:
//! when the recursively updated residual reports convergence, the residual is recomputed
//! directly, so rounding drift cannot end the solve with a wrong answer, and the solve also stops
//! when that residual reaches the rounding floor of the product, reporting what it attained.

use std::fmt;

use crate::csr::CsrMatrix;
use crate::preconditioner::Preconditioner;

/// Settings for a solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgOptions {
    /// Converged when `||b - A x|| <= tolerance * ||b||`. The default is 1e-9: the residual of a
    /// stiffness solve shrinks the error in strain energy roughly with its square, and tighter
    /// tolerances spend iterations on digits a real load case cannot justify.
    pub tolerance: f64,
    /// Iteration cap; `None` uses `10 * size + 1000`, far past the `size` iterations exact
    /// arithmetic needs, so hitting it means the solve is failing.
    pub max_iterations: Option<u64>,
}

impl Default for PcgOptions {
    fn default() -> Self {
        PcgOptions {
            tolerance: 1e-9,
            max_iterations: None,
        }
    }
}

/// Where a solve stands after a call to [`Pcg::step`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Progress {
    /// Iterations completed so far.
    pub iteration: u64,
    /// `||r|| / ||b||`, the relative residual of the current estimate.
    pub relative_residual: f64,
    /// True when the solve stopped at the rounding floor of `A x`: the requested tolerance is
    /// below what f64 can resolve for this system, and `relative_residual` is what was reached.
    pub at_rounding_floor: bool,
}

/// What a call to [`Pcg::step`] ended with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepOutcome {
    /// Budget spent, more iterations are needed; call `step` again.
    Running(Progress),
    /// Converged. Calling `step` again changes nothing.
    Converged(Progress),
    /// The cancel check returned true. The solve is over; the current estimate is partial and is
    /// not a valid answer. Calling `step` again changes nothing.
    Cancelled(Progress),
}

/// Why a solve failed. No estimate can be trusted as a result after one of these.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PcgError {
    /// The matrix or preconditioner is not positive definite along the search direction, which
    /// for a stiffness matrix means an under-constrained model that validation did not catch.
    NotPositiveDefinite {
        /// The iteration that detected it.
        iteration: u64,
        /// The curvature along the search direction: not positive.
        curvature: f64,
    },
    /// The iteration cap ran out before reaching the tolerance.
    NotConverged {
        /// Iterations spent.
        iterations: u64,
        /// The relative residual when it stopped.
        relative_residual: f64,
    },
}

impl fmt::Display for PcgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PcgError::NotPositiveDefinite {
                iteration,
                curvature,
            } => write!(
                f,
                "the system is not positive definite (curvature {curvature} at iteration \
                 {iteration}): something is free to move that should be held"
            ),
            PcgError::NotConverged {
                iterations,
                relative_residual,
            } => write!(
                f,
                "the solver did not converge after {iterations} iterations (relative residual \
                 {relative_residual}); no result is reported"
            ),
        }
    }
}

impl std::error::Error for PcgError {}

/// Terminal state a [`Pcg`] remembers across `step` calls.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Status {
    Active,
    Converged,
    Cancelled,
    Failed(PcgError),
}

/// A preconditioned conjugate gradient solve in progress.
pub struct Pcg<P: Preconditioner> {
    matrix: CsrMatrix,
    preconditioner: P,
    rhs: Vec<f64>,
    x: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
    rz: f64,
    rhs_norm: f64,
    relative_residual: f64,
    iterations: u64,
    max_iterations: u64,
    tolerance: f64,
    floor_reached: bool,
    status: Status,
}

impl<P: Preconditioner> Pcg<P> {
    /// Starts a solve of `matrix * x = rhs` from a zero estimate.
    pub fn new(matrix: CsrMatrix, preconditioner: P, rhs: Vec<f64>, options: PcgOptions) -> Self {
        let n = matrix.size();
        let rhs_norm = norm(&rhs);
        let max_iterations = options.max_iterations.unwrap_or(10 * n as u64 + 1000);
        let mut solver = Pcg {
            matrix,
            preconditioner,
            r: rhs.clone(),
            rhs,
            x: vec![0.0; n],
            z: vec![0.0; n],
            p: vec![0.0; n],
            ap: vec![0.0; n],
            rz: 0.0,
            rhs_norm,
            relative_residual: 0.0,
            iterations: 0,
            max_iterations,
            tolerance: options.tolerance,
            floor_reached: false,
            status: Status::Active,
        };
        if rhs_norm == 0.0 {
            solver.status = Status::Converged;
        } else {
            solver.reset_direction();
            solver.relative_residual = 1.0;
        }
        solver
    }

    /// Runs up to `max_iterations` iterations, stopping sooner on convergence or when `cancel`
    /// returns true; `cancel` is called before every iteration.
    pub fn step(
        &mut self,
        max_iterations: usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<StepOutcome, PcgError> {
        match self.status {
            Status::Active => {}
            Status::Converged => return Ok(StepOutcome::Converged(self.progress())),
            Status::Cancelled => return Ok(StepOutcome::Cancelled(self.progress())),
            Status::Failed(error) => return Err(error),
        }
        for _ in 0..max_iterations {
            if cancel() {
                self.status = Status::Cancelled;
                return Ok(StepOutcome::Cancelled(self.progress()));
            }
            if let Some(outcome) = self.iterate()? {
                return Ok(outcome);
            }
        }
        Ok(StepOutcome::Running(self.progress()))
    }

    /// The current estimate of the solution.
    pub fn estimate(&self) -> &[f64] {
        &self.x
    }

    /// The preconditioner this solve runs with.
    pub fn preconditioner(&self) -> &P {
        &self.preconditioner
    }

    /// The progress reached so far.
    pub fn progress(&self) -> Progress {
        Progress {
            iteration: self.iterations,
            relative_residual: self.relative_residual,
            at_rounding_floor: self.floor_reached,
        }
    }

    /// True once the solve has converged.
    pub fn is_converged(&self) -> bool {
        self.status == Status::Converged
    }

    /// One iteration. Returns the outcome when the solve ended, `None` to keep going.
    fn iterate(&mut self) -> Result<Option<StepOutcome>, PcgError> {
        self.preconditioner
            .system_multiply(&self.matrix, &self.p, &mut self.ap);
        let curvature = dot(&self.p, &self.ap);
        if curvature.is_nan() || curvature <= 0.0 || !self.rz.is_finite() {
            let error = PcgError::NotPositiveDefinite {
                iteration: self.iterations,
                curvature,
            };
            self.status = Status::Failed(error);
            return Err(error);
        }
        let alpha = self.rz / curvature;
        add_scaled(&mut self.x, alpha, &self.p);
        add_scaled(&mut self.r, -alpha, &self.ap);
        self.iterations += 1;
        self.relative_residual = norm(&self.r) / self.rhs_norm;

        if self.relative_residual <= self.tolerance {
            let mut residual = vec![0.0; self.rhs.len()];
            self.preconditioner
                .system_multiply(&self.matrix, &self.x, &mut residual);
            for (r, b) in residual.iter_mut().zip(&self.rhs) {
                *r = b - *r;
            }
            self.relative_residual = norm(&residual) / self.rhs_norm;
            // A residual at the rounding floor of the product `A x` cannot be improved by iterating:
            // on a slender part `||A|| ||x||` is far larger than `||b||`, so a tolerance relative to
            // `||b||` alone can sit below what f64 can resolve. Stopping there reports the attained
            // residual honestly and keeps the solve from running to its cap on digits that do not exist.
            let floor = self.matrix.multiply_rounding_bound(&self.x) / self.rhs_norm;
            self.floor_reached =
                self.relative_residual > self.tolerance && self.relative_residual <= floor;
            if self.relative_residual <= self.tolerance || self.floor_reached {
                self.status = Status::Converged;
                return Ok(Some(StepOutcome::Converged(self.progress())));
            }
            // Rounding made the recursive residual over-optimistic: restart from the true one.
            self.r = residual;
            self.reset_direction();
        } else {
            self.preconditioner.apply(&self.r, &mut self.z);
            let rz_next = dot(&self.r, &self.z);
            let beta = rz_next / self.rz;
            self.rz = rz_next;
            for (p, z) in self.p.iter_mut().zip(&self.z) {
                *p = z + beta * *p;
            }
        }

        if self.iterations >= self.max_iterations {
            let error = PcgError::NotConverged {
                iterations: self.iterations,
                relative_residual: self.relative_residual,
            };
            self.status = Status::Failed(error);
            return Err(error);
        }
        Ok(None)
    }

    /// Sets the search direction from the current residual.
    fn reset_direction(&mut self) {
        self.preconditioner.apply(&self.r, &mut self.z);
        self.p.copy_from_slice(&self.z);
        self.rz = dot(&self.r, &self.z);
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b) {
        sum += x * y;
    }
    sum
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn add_scaled(target: &mut [f64], scale: f64, direction: &[f64]) {
    for (t, d) in target.iter_mut().zip(direction) {
        *t += scale * d;
    }
}
