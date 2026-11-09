//! The resumable construction of the multigrid hierarchy. [`MultigridSetup`] walks a fixed
//! sequence of phases per level (diagonal, strength, three aggregation passes, grouping, tentative
//! prolongator, spectral estimate, smoothing, the two halves of the Galerkin product) keeping a
//! row cursor, so `step(budget, cancel)` can stop after any row and continue later with exactly
//! the same operations as an uninterrupted run. The coarsest level ends in a dense factorization
//! taken one column at a time. The aggregation phases are in `setup_aggregation.rs`.

use super::aggregate::Aggregation;
use super::block::{
    apply_block_diagonal, invert_spd_block, norm2, normalize, start_value, Accumulator, BlockCsr,
    DenseCholesky,
};
use super::cycle::{Level, Multigrid};
use super::transfer::{
    galerkin_row, orthonormalize_aggregate, product_row, prolongator_columns, smooth_row,
    ColumnView,
};
use super::{MultigridError, MultigridOptions, NEAR_NULL_MODES};
use crate::csr::CsrMatrix;

#[path = "setup_aggregation.rs"]
mod aggregation;

/// Above this many values a level is too large to factor densely (a factor's work grows with the
/// cube of its size and a browser must not stall), so the coarsest level falls back to its
/// smoother.
const DENSE_LIMIT: usize = 3000;

/// A level whose coarse values come to more than this fraction of its own has not coarsened: its
/// operator would be larger, not smaller, once six-value blocks and the wider stencil count.
const MAX_COARSE_FRACTION: f64 = 0.7;

/// A stalled level widens its neighbour reach by this factor and aggregates again, at most
/// [`MAX_RETRIES`] times, before the hierarchy stops there.
const RETRY_WIDENING: f64 = 1.5;
const MAX_RETRIES: usize = 3;

/// Power iterations for the spectral estimate. Each costs one matrix product; twenty brings the
/// estimate within the margin below for matrices of this kind.
const POWER_ITERATIONS: usize = 20;

/// Multiplies the power estimate, which approaches the largest eigenvalue from below. A value too
/// small would stop the smoother damping the top of the spectrum and could make it amplify.
const SPECTRAL_MARGIN: f64 = 1.15;

/// Where a [`MultigridSetup`] stands after a call to [`MultigridSetup::step`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupStatus {
    /// Budget spent; call `step` again.
    Running,
    /// The hierarchy is complete; call `finish`.
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Diagonal,
    Strength,
    SeedPass,
    JoinPass,
    SweepPass,
    Group,
    Tentative,
    Spectral,
    Smooth,
    Product,
    Galerkin,
    Factor,
    Done,
}

enum Flow {
    Advanced,
    Paused,
    Cancelled,
}

/// A multigrid hierarchy under construction.
pub struct MultigridSetup {
    options: MultigridOptions,
    reach: f64,
    retries: usize,
    positions: Vec<[f64; 3]>,
    coarse_positions: Vec<[f64; 3]>,
    levels: Vec<Level>,
    current: BlockCsr,
    near_null: Vec<f64>,
    inverse_diagonal: Vec<f64>,
    strong_starts: Vec<usize>,
    strong_columns: Vec<u32>,
    aggregation: Option<Aggregation>,
    member_starts: Vec<usize>,
    members: Vec<u32>,
    tentative_values: Vec<f64>,
    tentative_positions: Vec<usize>,
    coarse_null: Vec<f64>,
    tentative: BlockCsr,
    power_vector: Vec<f64>,
    power_scratch: Vec<f64>,
    power_scratch_two: Vec<f64>,
    power_iterations_done: usize,
    lambda_max: f64,
    prolongator: BlockCsr,
    a_times_p: BlockCsr,
    coarse: BlockCsr,
    column_starts: Vec<usize>,
    column_entries: Vec<(u32, usize)>,
    accumulator: Option<Accumulator>,
    factor: Option<DenseCholesky>,
    cursor: usize,
    is_coarsest: bool,
    phase: Phase,
    cancelled: bool,
    failure: Option<MultigridError>,
}

impl MultigridSetup {
    /// Starts building a hierarchy over `matrix`. `near_null` holds six values for every scalar
    /// row, `(row, mode)` at `6 * row + mode`, `positions` one point per block row (node), and a
    /// wrong length of either is refused here by name.
    pub fn new(
        matrix: &CsrMatrix,
        near_null: &[f64],
        positions: &[[f64; 3]],
        options: MultigridOptions,
    ) -> Result<Self, MultigridError> {
        let expected = NEAR_NULL_MODES * matrix.size();
        if near_null.len() != expected {
            return Err(MultigridError::NearNullLength {
                expected,
                found: near_null.len(),
            });
        }
        let current = BlockCsr::from_nodal(matrix)?;
        if positions.len() != current.rows {
            return Err(MultigridError::PositionLength {
                expected: current.rows,
                found: positions.len(),
            });
        }
        Ok(MultigridSetup {
            options,
            reach: options.neighbour_reach,
            retries: 0,
            positions: positions.to_vec(),
            coarse_positions: Vec::new(),
            levels: Vec::new(),
            near_null: near_null.to_vec(),
            current,
            inverse_diagonal: Vec::new(),
            strong_starts: Vec::new(),
            strong_columns: Vec::new(),
            aggregation: None,
            member_starts: Vec::new(),
            members: Vec::new(),
            tentative_values: Vec::new(),
            tentative_positions: Vec::new(),
            coarse_null: Vec::new(),
            tentative: BlockCsr::default(),
            power_vector: Vec::new(),
            power_scratch: Vec::new(),
            power_scratch_two: Vec::new(),
            power_iterations_done: 0,
            lambda_max: 0.0,
            prolongator: BlockCsr::default(),
            a_times_p: BlockCsr::default(),
            coarse: BlockCsr::default(),
            column_starts: Vec::new(),
            column_entries: Vec::new(),
            accumulator: None,
            factor: None,
            cursor: 0,
            is_coarsest: false,
            phase: Phase::Diagonal,
            cancelled: false,
            failure: None,
        })
    }

    /// Does up to `budget` block rows of work; `cancel` is read before every row. A hierarchy
    /// built in any slice sizes is bit-identical to one built in a single call.
    pub fn step(
        &mut self,
        budget: usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<SetupStatus, MultigridError> {
        if let Some(failure) = self.failure {
            return Err(failure);
        }
        if self.cancelled {
            return Err(MultigridError::Cancelled);
        }
        let mut budget = budget;
        loop {
            if self.phase == Phase::Done {
                return Ok(SetupStatus::Finished);
            }
            match self.run_phase(&mut budget, cancel) {
                Ok(Flow::Advanced) => {}
                Ok(Flow::Paused) => return Ok(SetupStatus::Running),
                Ok(Flow::Cancelled) => {
                    self.cancelled = true;
                    return Err(MultigridError::Cancelled);
                }
                Err(error) => {
                    self.failure = Some(error);
                    return Err(error);
                }
            }
        }
    }

    /// True once `step` has reported [`SetupStatus::Finished`].
    pub fn is_finished(&self) -> bool {
        self.phase == Phase::Done
    }

    /// The finished preconditioner; refuses before `step` reports it is done.
    pub fn finish(self) -> Result<Multigrid, MultigridError> {
        if self.phase != Phase::Done {
            return Err(MultigridError::NotFinished);
        }
        Ok(Multigrid::from_levels(
            self.levels,
            self.options.smoother_degree,
        ))
    }

    fn run_phase(
        &mut self,
        budget: &mut usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Flow, MultigridError> {
        match self.phase {
            Phase::Diagonal => self.run_diagonal(budget, cancel),
            Phase::Strength => self.run_strength(budget, cancel),
            Phase::SeedPass => self.run_seed(budget, cancel),
            Phase::JoinPass => self.run_join(budget, cancel),
            Phase::SweepPass => self.run_sweep(budget, cancel),
            Phase::Group => Ok(self.run_group()),
            Phase::Tentative => self.run_tentative(budget, cancel),
            Phase::Spectral => self.run_spectral(budget, cancel),
            Phase::Smooth => self.run_smooth(budget, cancel),
            Phase::Product => self.run_product(budget, cancel),
            Phase::Galerkin => self.run_galerkin(budget, cancel),
            Phase::Factor => self.run_factor(budget, cancel),
            Phase::Done => Ok(Flow::Advanced),
        }
    }

    fn level_index(&self) -> usize {
        self.levels.len()
    }

    fn run_diagonal(
        &mut self,
        budget: &mut usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Flow, MultigridError> {
        let rb = self.current.row_block;
        if self.cursor == 0 {
            self.inverse_diagonal = vec![0.0; self.current.rows * rb * rb];
        }
        while self.cursor < self.current.rows {
            if *budget == 0 {
                return Ok(Flow::Paused);
            }
            if cancel() {
                return Ok(Flow::Cancelled);
            }
            let row = self.cursor;
            let range = self.current.row_range(row);
            let at = match self.current.columns[range.clone()].binary_search(&(row as u32)) {
                Ok(found) => range.start + found,
                Err(_) => {
                    return Err(MultigridError::NotPositiveDefinite {
                        level: self.level_index(),
                        block: row,
                        pivot: 0.0,
                    })
                }
            };
            let block = self.current.block(at);
            let output = &mut self.inverse_diagonal[row * rb * rb..(row + 1) * rb * rb];
            if let Err(pivot) = invert_spd_block(block, rb, output) {
                return Err(MultigridError::NotPositiveDefinite {
                    level: self.level_index(),
                    block: row,
                    pivot,
                });
            }
            self.cursor += 1;
            *budget -= 1;
        }
        self.cursor = 0;
        let level = self.level_index();
        if self.current.dofs() <= self.options.coarse_dofs || level + 1 >= self.options.max_levels {
            self.is_coarsest = true;
            self.phase = Phase::Spectral;
        } else {
            self.strong_starts.clear();
            self.strong_starts.push(0);
            self.strong_columns.clear();
            self.phase = Phase::Strength;
        }
        Ok(Flow::Advanced)
    }

    fn aggregate_count(&self) -> usize {
        self.aggregation.as_ref().map_or(0, |a| a.count as usize)
    }

    fn run_tentative(
        &mut self,
        budget: &mut usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Flow, MultigridError> {
        let rb = self.current.row_block;
        let aggregates = self.aggregate_count();
        while self.cursor < aggregates {
            if *budget == 0 {
                return Ok(Flow::Paused);
            }
            if cancel() {
                return Ok(Flow::Cancelled);
            }
            let range = self.member_starts[self.cursor]..self.member_starts[self.cursor + 1];
            let members = &self.members[range.clone()];
            let start = self.tentative_values.len();
            let mut coarse = [0.0; 36];
            orthonormalize_aggregate(
                members,
                rb,
                &self.near_null,
                &mut self.tentative_values,
                &mut coarse,
            );
            let block = rb * NEAR_NULL_MODES;
            for (index, member) in members.iter().enumerate() {
                self.tentative_positions[*member as usize] = start + index * block;
            }
            self.coarse_null.extend_from_slice(&coarse);
            self.cursor += 1;
            *budget = budget.saturating_sub(range.end - range.start);
        }
        self.cursor = 0;
        self.build_tentative_matrix();
        self.phase = Phase::Spectral;
        Ok(Flow::Advanced)
    }

    fn build_tentative_matrix(&mut self) {
        let rb = self.current.row_block;
        let block = rb * NEAR_NULL_MODES;
        let rows = self.current.rows;
        let mut tentative = BlockCsr::with_shape(rows, rb, NEAR_NULL_MODES);
        tentative.columns.reserve(rows);
        tentative.values.reserve(rows * block);
        if let Some(aggregation) = &self.aggregation {
            for row in 0..rows {
                tentative.columns.push(aggregation.aggregate[row]);
                let at = self.tentative_positions[row];
                tentative
                    .values
                    .extend_from_slice(&self.tentative_values[at..at + block]);
                tentative.starts.push(tentative.columns.len());
            }
        }
        self.tentative = tentative;
        self.tentative_values = Vec::new();
        self.tentative_positions = Vec::new();
    }

    /// A largest eigenvalue estimate for `D^-1 A` by power iteration; each iteration is one
    /// matrix product charged at the level's row count.
    fn run_spectral(
        &mut self,
        budget: &mut usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Flow, MultigridError> {
        let dofs = self.current.dofs();
        if self.power_iterations_done == 0 && self.power_vector.len() != dofs {
            self.power_vector = (0..dofs).map(start_value).collect();
            normalize(&mut self.power_vector);
            self.power_scratch = vec![0.0; dofs];
            self.power_scratch_two = vec![0.0; dofs];
        }
        while self.power_iterations_done < POWER_ITERATIONS {
            if *budget == 0 {
                return Ok(Flow::Paused);
            }
            if cancel() {
                return Ok(Flow::Cancelled);
            }
            self.current
                .multiply(&self.power_vector, &mut self.power_scratch);
            apply_block_diagonal(
                &self.inverse_diagonal,
                self.current.row_block,
                &self.power_scratch,
                &mut self.power_scratch_two,
            );
            let norm = norm2(&self.power_scratch_two);
            self.lambda_max = norm;
            for (v, next) in self.power_vector.iter_mut().zip(&self.power_scratch_two) {
                *v = next / norm;
            }
            self.power_iterations_done += 1;
            *budget = budget.saturating_sub(self.current.rows);
        }
        self.power_iterations_done = 0;
        self.power_vector = Vec::new();
        self.power_scratch = Vec::new();
        self.power_scratch_two = Vec::new();
        self.lambda_max *= SPECTRAL_MARGIN;
        if self.is_coarsest {
            if self.current.dofs() <= DENSE_LIMIT {
                self.begin_factor();
                self.phase = Phase::Factor;
            } else {
                self.push_final_level(None);
                self.phase = Phase::Done;
            }
            return Ok(Flow::Advanced);
        }
        let aggregates = self.aggregate_count();
        self.accumulator = Some(Accumulator::new(
            aggregates,
            self.current.row_block * NEAR_NULL_MODES,
        ));
        self.prolongator =
            BlockCsr::with_shape(self.current.rows, self.current.row_block, NEAR_NULL_MODES);
        self.phase = Phase::Smooth;
        Ok(Flow::Advanced)
    }

    fn begin_factor(&mut self) {
        let n = self.current.dofs();
        let (rb, cb) = (self.current.row_block, self.current.col_block);
        let mut dense = vec![0.0; n * n];
        for row in 0..self.current.rows {
            for at in self.current.row_range(row) {
                let column = self.current.columns[at] as usize;
                let block = self.current.block(at);
                for p in 0..rb {
                    for q in 0..cb {
                        dense[(row * rb + p) * n + column * cb + q] = block[p * cb + q];
                    }
                }
            }
        }
        self.factor = Some(DenseCholesky::new(dense, n));
        self.cursor = 0;
    }

    fn run_factor(
        &mut self,
        budget: &mut usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Flow, MultigridError> {
        let level = self.level_index();
        let n = self.current.dofs();
        loop {
            if *budget == 0 {
                return Ok(Flow::Paused);
            }
            if cancel() {
                return Ok(Flow::Cancelled);
            }
            let Some(factor) = &mut self.factor else {
                break;
            };
            match factor.factor_column() {
                Ok(true) => break,
                Ok(false) => {}
                Err((block, pivot)) => {
                    return Err(MultigridError::NotPositiveDefinite {
                        level,
                        block,
                        pivot,
                    })
                }
            }
            *budget = budget.saturating_sub(n);
        }
        let factor = self.factor.take();
        self.push_final_level(factor);
        self.phase = Phase::Done;
        Ok(Flow::Advanced)
    }

    fn push_final_level(&mut self, factor: Option<DenseCholesky>) {
        let (lambda_min, lambda_max) = self.smoother_range();
        let level = Level {
            a: std::mem::take(&mut self.current),
            inverse_diagonal: std::mem::take(&mut self.inverse_diagonal),
            lambda_min,
            lambda_max,
            prolongator: None,
            coarse: factor,
            aggregates: Vec::new(),
        };
        self.levels.push(level);
    }

    fn smoother_range(&self) -> (f64, f64) {
        (
            self.lambda_max / self.options.smoother_range,
            self.lambda_max,
        )
    }

    fn run_smooth(
        &mut self,
        budget: &mut usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Flow, MultigridError> {
        // The standard weight for one prolongator smoothing step: it minimises the energy of the
        // smoothed column when the eigenvalues of the scaled operator fill [0, lambda_max].
        let omega = 4.0 / (3.0 * self.lambda_max);
        let Some(accumulator) = &mut self.accumulator else {
            return Ok(Flow::Advanced);
        };
        while self.cursor < self.current.rows {
            if *budget == 0 {
                return Ok(Flow::Paused);
            }
            if cancel() {
                return Ok(Flow::Cancelled);
            }
            let neighbours = &self.strong_columns
                [self.strong_starts[self.cursor]..self.strong_starts[self.cursor + 1]];
            smooth_row(
                &self.current,
                &self.tentative,
                neighbours,
                &self.inverse_diagonal,
                omega,
                self.cursor,
                accumulator,
                &mut self.prolongator,
            );
            self.cursor += 1;
            *budget -= 1;
        }
        self.cursor = 0;
        self.tentative = BlockCsr::default();
        self.a_times_p =
            BlockCsr::with_shape(self.current.rows, self.current.row_block, NEAR_NULL_MODES);
        self.phase = Phase::Product;
        Ok(Flow::Advanced)
    }

    fn run_product(
        &mut self,
        budget: &mut usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Flow, MultigridError> {
        let Some(accumulator) = &mut self.accumulator else {
            return Ok(Flow::Advanced);
        };
        while self.cursor < self.current.rows {
            if *budget == 0 {
                return Ok(Flow::Paused);
            }
            if cancel() {
                return Ok(Flow::Cancelled);
            }
            product_row(
                &self.current,
                &self.prolongator,
                self.cursor,
                accumulator,
                &mut self.a_times_p,
            );
            self.cursor += 1;
            *budget -= 1;
        }
        let aggregates = self.aggregate_count();
        let (starts, entries) = prolongator_columns(&self.prolongator, aggregates);
        self.column_starts = starts;
        self.column_entries = entries;
        self.coarse = BlockCsr::with_shape(aggregates, NEAR_NULL_MODES, NEAR_NULL_MODES);
        self.accumulator = Some(Accumulator::new(
            aggregates,
            NEAR_NULL_MODES * NEAR_NULL_MODES,
        ));
        self.cursor = 0;
        self.phase = Phase::Galerkin;
        Ok(Flow::Advanced)
    }

    fn run_galerkin(
        &mut self,
        budget: &mut usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Flow, MultigridError> {
        let aggregates = self.aggregate_count();
        let Some(accumulator) = &mut self.accumulator else {
            return Ok(Flow::Advanced);
        };
        let view = ColumnView {
            prolongator: &self.prolongator,
            starts: &self.column_starts,
            entries: &self.column_entries,
        };
        while self.cursor < aggregates {
            if *budget == 0 {
                return Ok(Flow::Paused);
            }
            if cancel() {
                return Ok(Flow::Cancelled);
            }
            galerkin_row(
                &view,
                &self.a_times_p,
                self.cursor,
                accumulator,
                &mut self.coarse,
            );
            self.cursor += 1;
            *budget -= 1;
        }
        self.finish_level();
        Ok(Flow::Advanced)
    }

    /// Stores the level just finished and starts the next one from its coarse matrix.
    fn finish_level(&mut self) {
        let (lambda_min, lambda_max) = self.smoother_range();
        let mut coarse = std::mem::take(&mut self.coarse);
        // A coarse value with no direction on its aggregate has an all-zero row and column. A
        // unit diagonal there keeps the level positive definite and leaves the value decoupled.
        let dimension = coarse.row_block;
        for row in 0..coarse.rows {
            for at in coarse.row_range(row) {
                if coarse.columns[at] as usize != row {
                    continue;
                }
                let block_at = at * dimension * dimension;
                for d in 0..dimension {
                    let index = block_at + d * dimension + d;
                    if coarse.values[index] == 0.0 {
                        coarse.values[index] = 1.0;
                    }
                }
            }
        }
        let level = Level {
            a: std::mem::replace(&mut self.current, coarse),
            inverse_diagonal: std::mem::take(&mut self.inverse_diagonal),
            lambda_min,
            lambda_max,
            prolongator: Some(std::mem::take(&mut self.prolongator)),
            coarse: None,
            aggregates: self
                .aggregation
                .take()
                .map_or(Vec::new(), |aggregation| aggregation.aggregate),
        };
        self.levels.push(level);
        self.a_times_p = BlockCsr::default();
        self.column_starts = Vec::new();
        self.column_entries = Vec::new();
        self.accumulator = None;
        self.aggregation = None;
        self.near_null = std::mem::take(&mut self.coarse_null);
        self.positions = std::mem::take(&mut self.coarse_positions);
        self.reach = self.options.neighbour_reach;
        self.retries = 0;
        self.is_coarsest = false;
        self.cursor = 0;
        self.phase = Phase::Diagonal;
    }
}
