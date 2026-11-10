//! The finished multigrid hierarchy and its symmetric V-cycle. The cycle smooths with a Chebyshev
//! polynomial, corrects from the next coarser level and smooths with the same polynomial again,
//! which keeps the operator symmetric; the coarsest level is solved by a dense factorization, or
//! by its smoother when coarsening stalled above the factorization size limit.
//!
//! The levels are stored and the cycle runs in the [`Precision`] the options chose, through one
//! implementation generic over a storage [`Scalar`] `S` and a vector [`Scalar`] `V`; every stored
//! value is widened to `V` as it is read. `SingleStorage` stores `f32` and keeps vectors and sums
//! in `f64`. `Single` rounds the `f64` residual to `f32`, runs the whole V-cycle in `f32` and
//! widens the correction back. The coarsest level's dense factor stays `f64` in every case (at
//! most a few thousand values square, solved once per cycle), its right-hand side widened and its
//! solution rounded to `V`. The Chebyshev coefficients are computed in
//! `f64` and rounded where they are used, so all three precisions apply the same polynomial before and
//! after the coarse correction and the cycle stays symmetric up to rounding.
//!
//! A host that runs the cycle elsewhere (the GPU kernels in `crate::gpu`) reads each level as
//! stored through [`Multigrid::level_matrix`], [`Multigrid::level_prolongator`] and
//! [`Multigrid::level_inverse_diagonal`] ([`LevelMatrix`] arrays, values widened exactly to `f64`)
//! and hands the coarsest level back to [`Multigrid::solve_coarsest`].
//!
//! A level holding enough stored values runs its row-by-row work on the [`WorkerPool`], one chunk
//! of rows per thread, in two kinds of region: a product reads other rows of a vector no other
//! region is writing, and the updates that follow it touch only each chunk's own rows of several
//! vectors at once. Each row sees the operations of a serial cycle in the same order, so the thread
//! count changes no bit.

use std::cell::RefCell;

use super::block::{BlockCsr, DenseCholesky};
use super::scalar::Scalar;
use super::{MultigridError, MultigridStats, Precision};
use crate::csr::CsrMatrix;
use crate::parallel::{balanced_bounds, for_rows, WorkerPool, MIN_PARALLEL_VALUES};
use crate::preconditioner::Preconditioner;

/// One level of the hierarchy.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Level<T: Scalar = f64> {
    pub a: BlockCsr<T>,
    pub inverse_diagonal: Vec<T>,
    pub lambda_min: f64,
    pub lambda_max: f64,
    /// The prolongator to the next coarser level, absent on the coarsest.
    pub prolongator: Option<BlockCsr<T>>,
    /// The direct solve on the coarsest level; absent means smoothing alone.
    pub coarse: Option<DenseCholesky>,
    /// The aggregate of each block row, empty on the coarsest level.
    pub aggregates: Vec<u32>,
}

impl Level<f64> {
    /// The same level with its matrices and block-diagonal inverse rounded to `U`.
    fn into_precision<U: Scalar>(self) -> Level<U> {
        Level {
            a: self.a.into_precision(),
            inverse_diagonal: self.inverse_diagonal.into_iter().map(U::from_f64).collect(),
            lambda_min: self.lambda_min,
            lambda_max: self.lambda_max,
            prolongator: self.prolongator.map(BlockCsr::into_precision),
            coarse: self.coarse,
            aggregates: self.aggregates,
        }
    }
}

/// Reused vectors of one level, so applying the preconditioner allocates nothing.
#[derive(Debug, Clone, PartialEq, Default)]
struct LevelWork<V: Scalar> {
    residual: Vec<V>,
    scaled: Vec<V>,
    direction: Vec<V>,
    product: Vec<V>,
    post_rhs: Vec<V>,
    post_x: Vec<V>,
    coarse_rhs: Vec<V>,
    coarse_x: Vec<V>,
    /// The dense solve's `f64` right-hand side and solution, on a factored coarsest level.
    dense: Vec<f64>,
    /// The chunk boundaries of this level's rows, a single chunk when it runs on one thread.
    bounds: Vec<usize>,
}

impl<V: Scalar> LevelWork<V> {
    fn for_level<S: Scalar>(level: &Level<S>, coarse_dofs: usize, threads: usize) -> Self {
        let dofs = level.a.dofs();
        let chunks = if level.a.values.len() >= MIN_PARALLEL_VALUES {
            threads
        } else {
            1
        };
        LevelWork {
            residual: vec![V::ZERO; dofs],
            scaled: vec![V::ZERO; dofs],
            direction: vec![V::ZERO; dofs],
            product: vec![V::ZERO; dofs],
            post_rhs: vec![V::ZERO; dofs],
            post_x: vec![V::ZERO; dofs],
            coarse_rhs: vec![V::ZERO; coarse_dofs],
            coarse_x: vec![V::ZERO; coarse_dofs],
            dense: if level.coarse.is_some() {
                vec![0.0; dofs]
            } else {
                Vec::new()
            },
            bounds: balanced_bounds(&level.a.starts, chunks),
        }
    }
}

/// The levels stored in `S` with their work vectors in `V`.
struct Hierarchy<S: Scalar, V: Scalar> {
    levels: Vec<Level<S>>,
    work: RefCell<HierarchyWork<V>>,
}

/// `rhs` and `x` hold the rounded residual and the correction when the cycle's vectors are `f32`,
/// and stay empty when they are `f64` and the cycle runs on the caller's vectors.
struct HierarchyWork<V: Scalar> {
    levels: Vec<LevelWork<V>>,
    rhs: Vec<V>,
    x: Vec<V>,
}

impl<S: Scalar> Hierarchy<S, f64> {
    /// Runs the cycle on the caller's `f64` vectors with no copies.
    fn apply_direct(
        &self,
        degree: usize,
        pool: Option<&WorkerPool>,
        residual: &[f64],
        output: &mut [f64],
    ) {
        let mut work = self.work.borrow_mut();
        cycle(
            &self.levels,
            degree,
            pool,
            &mut work.levels,
            residual,
            output,
        );
    }
}

impl<S: Scalar, V: Scalar> Hierarchy<S, V> {
    fn new(levels: Vec<Level<S>>, converts: bool, threads: usize) -> Self {
        let mut work = Vec::with_capacity(levels.len());
        for (index, level) in levels.iter().enumerate() {
            let coarse_dofs = levels.get(index + 1).map_or(0, |next| next.a.dofs());
            work.push(LevelWork::for_level(level, coarse_dofs, threads));
        }
        let dofs = match (converts, levels.first()) {
            (true, Some(level)) => level.a.dofs(),
            _ => 0,
        };
        Hierarchy {
            levels,
            work: RefCell::new(HierarchyWork {
                levels: work,
                rhs: vec![V::ZERO; dofs],
                x: vec![V::ZERO; dofs],
            }),
        }
    }

    fn level_sizes(&self) -> Vec<(usize, usize, usize)> {
        self.levels
            .iter()
            .map(|level| (level.a.rows, level.a.row_block, level.a.values.len()))
            .collect()
    }

    fn aggregates(&self, level: usize) -> &[u32] {
        self.levels
            .get(level)
            .map_or(&[], |level| level.aggregates.as_slice())
    }

    fn smoother_spectrum(&self, level: usize) -> Option<(f64, f64)> {
        self.levels
            .get(level)
            .map(|level| (level.lambda_min, level.lambda_max))
    }

    fn level_matrix(&self, level: usize, prolongator: bool) -> Option<LevelMatrix> {
        let stored = self.levels.get(level)?;
        let (matrix, block_columns) = if prolongator {
            (
                stored.prolongator.as_ref()?,
                self.levels.get(level + 1)?.a.rows,
            )
        } else {
            (&stored.a, stored.a.rows)
        };
        Some(LevelMatrix {
            block_columns,
            row_block: matrix.row_block,
            col_block: matrix.col_block,
            starts: matrix.starts.clone(),
            columns: matrix.columns.clone(),
            values: matrix.values.iter().map(|v| v.to_f64()).collect(),
        })
    }

    fn level_inverse_diagonal(&self, level: usize) -> Option<Vec<f64>> {
        let stored = self.levels.get(level)?;
        Some(stored.inverse_diagonal.iter().map(|v| v.to_f64()).collect())
    }

    fn solve_coarsest(&self, values: &mut [f64]) -> Result<(), MultigridError> {
        let dofs = self.levels.last().map_or(0, |level| level.a.dofs());
        let factor = self.levels.last().and_then(|level| level.coarse.as_ref());
        let Some(factor) = factor else {
            return Err(MultigridError::CoarsestNotFactored { values: dofs });
        };
        if values.len() != dofs {
            return Err(MultigridError::CoarseLength {
                expected: dofs,
                found: values.len(),
            });
        }
        factor.solve(values);
        Ok(())
    }

    /// Rounds `residual` into `V`, runs the cycle and widens the correction into `output`.
    fn apply_converted(
        &self,
        degree: usize,
        pool: Option<&WorkerPool>,
        residual: &[f64],
        output: &mut [f64],
    ) {
        let mut work = self.work.borrow_mut();
        let HierarchyWork { levels, rhs, x } = &mut *work;
        for (to, from) in rhs.iter_mut().zip(residual) {
            *to = V::from_f64(*from);
        }
        cycle(&self.levels, degree, pool, levels, rhs, x);
        for (to, from) in output.iter_mut().zip(x.iter()) {
            *to = from.to_f64();
        }
    }
}

/// The hierarchy in the precision the options chose.
enum Stored {
    Double(Hierarchy<f64, f64>),
    SingleStorage(Hierarchy<f32, f64>),
    Single(Hierarchy<f32, f32>),
}

/// A multigrid V-cycle preconditioner over a hierarchy built by [`super::MultigridSetup`].
pub struct Multigrid {
    stored: Stored,
    degree: usize,
    stats: MultigridStats,
    /// Present when `threads` asked for more than one and some level is large enough to use them.
    pool: Option<WorkerPool>,
}

impl Multigrid {
    /// Takes the `f64` levels setup built and stores them in `precision`. Converting consumes one
    /// level at a time, so the peak is the `f64` hierarchy plus one converted level.
    pub(super) fn from_levels(
        levels: Vec<Level>,
        degree: usize,
        precision: Precision,
        threads: usize,
    ) -> Self {
        let large = levels
            .iter()
            .any(|level| level.a.values.len() >= MIN_PARALLEL_VALUES);
        let pool = if large {
            WorkerPool::new(threads)
        } else {
            None
        };
        let threads = pool.as_ref().map_or(1, WorkerPool::threads);
        let (stored, stats) = match precision {
            Precision::Double => {
                let stats = MultigridStats::of_levels(&levels);
                (
                    Stored::Double(Hierarchy::new(levels, false, threads)),
                    stats,
                )
            }
            Precision::SingleStorage => {
                let levels: Vec<Level<f32>> =
                    levels.into_iter().map(Level::into_precision).collect();
                let stats = MultigridStats::of_levels(&levels);
                let hierarchy = Hierarchy::new(levels, false, threads);
                (Stored::SingleStorage(hierarchy), stats)
            }
            Precision::Single => {
                let levels: Vec<Level<f32>> =
                    levels.into_iter().map(Level::into_precision).collect();
                let stats = MultigridStats::of_levels(&levels);
                (Stored::Single(Hierarchy::new(levels, true, threads)), stats)
            }
        };
        Multigrid {
            stored,
            degree,
            stats,
            pool,
        }
    }

    /// The threads the large products run on, the caller included: one when the options asked
    /// for one, no level is large enough to use more, or the target has no threads. Otherwise
    /// what the options asked for, at most 64, or fewer when the system refused to start a worker;
    /// the answer's bits are the same in every case.
    pub fn threads(&self) -> usize {
        self.pool.as_ref().map_or(1, WorkerPool::threads)
    }

    /// The shape and cost of the hierarchy.
    pub fn stats(&self) -> MultigridStats {
        self.stats
    }

    /// The precision the hierarchy is stored and applied in.
    pub fn precision(&self) -> Precision {
        match &self.stored {
            Stored::Double(_) => Precision::Double,
            Stored::SingleStorage(_) => Precision::SingleStorage,
            Stored::Single(_) => Precision::Single,
        }
    }

    /// Block rows, values per block row and stored matrix values of every level, finest first.
    pub fn level_sizes(&self) -> Vec<(usize, usize, usize)> {
        match &self.stored {
            Stored::Double(hierarchy) => hierarchy.level_sizes(),
            Stored::SingleStorage(hierarchy) => hierarchy.level_sizes(),
            Stored::Single(hierarchy) => hierarchy.level_sizes(),
        }
    }

    /// The interval `(lambda_min, lambda_max)` of the block-Jacobi operator the Chebyshev
    /// smoother of `level` targets, so the same smoother can be rebuilt elsewhere (on a GPU);
    /// `None` past the coarsest level.
    /// The bounds are the setup's `f64` values in every precision.
    pub fn smoother_spectrum(&self, level: usize) -> Option<(f64, f64)> {
        match &self.stored {
            Stored::Double(hierarchy) => hierarchy.smoother_spectrum(level),
            Stored::SingleStorage(hierarchy) => hierarchy.smoother_spectrum(level),
            Stored::Single(hierarchy) => hierarchy.smoother_spectrum(level),
        }
    }

    /// The aggregate of each block row of `level`, so a caller can see what was grouped; empty on
    /// the coarsest level.
    pub fn aggregates(&self, level: usize) -> &[u32] {
        match &self.stored {
            Stored::Double(hierarchy) => hierarchy.aggregates(level),
            Stored::SingleStorage(hierarchy) => hierarchy.aggregates(level),
            Stored::Single(hierarchy) => hierarchy.aggregates(level),
        }
    }

    /// The matrix of `level` as stored, so its cycle can run elsewhere (on a GPU); `None` past
    /// the coarsest level. Values are widened to `f64`, exactly, from whatever precision the
    /// hierarchy stores, and each call makes a full `f64` copy of the level's matrix: read it once
    /// to upload it, never per cycle.
    pub fn level_matrix(&self, level: usize) -> Option<LevelMatrix> {
        match &self.stored {
            Stored::Double(hierarchy) => hierarchy.level_matrix(level, false),
            Stored::SingleStorage(hierarchy) => hierarchy.level_matrix(level, false),
            Stored::Single(hierarchy) => hierarchy.level_matrix(level, false),
        }
    }

    /// The prolongator from the coarser level `level + 1` up to `level`: block rows of `level`,
    /// block columns of `level + 1` (3x6 blocks from the finest level, 6x6 below it). `None` on
    /// the coarsest level and past it. Values are widened to `f64` as in
    /// [`Multigrid::level_matrix`], and each call makes a full `f64` copy of the prolongator.
    pub fn level_prolongator(&self, level: usize) -> Option<LevelMatrix> {
        match &self.stored {
            Stored::Double(hierarchy) => hierarchy.level_matrix(level, true),
            Stored::SingleStorage(hierarchy) => hierarchy.level_matrix(level, true),
            Stored::Single(hierarchy) => hierarchy.level_matrix(level, true),
        }
    }

    /// The inverse of each diagonal block of `level`, row-major, as the smoother reads it, widened
    /// to `f64`, a full copy on each call; `None` past the coarsest level.
    pub fn level_inverse_diagonal(&self, level: usize) -> Option<Vec<f64>> {
        match &self.stored {
            Stored::Double(hierarchy) => hierarchy.level_inverse_diagonal(level),
            Stored::SingleStorage(hierarchy) => hierarchy.level_inverse_diagonal(level),
            Stored::Single(hierarchy) => hierarchy.level_inverse_diagonal(level),
        }
    }

    /// Solves the coarsest level's system in place with its `f64` factorization, the cycle's own
    /// coarse solve. On an error `values` is untouched and is not a correction:
    /// [`MultigridError::CoarsestNotFactored`] when the coarsest level is above the factorization
    /// size and is smoothed instead, which a host handles by running that level's smoother, and
    /// [`MultigridError::CoarseLength`] when `values` is not that level's length.
    pub fn solve_coarsest(&self, values: &mut [f64]) -> Result<(), MultigridError> {
        match &self.stored {
            Stored::Double(hierarchy) => hierarchy.solve_coarsest(values),
            Stored::SingleStorage(hierarchy) => hierarchy.solve_coarsest(values),
            Stored::Single(hierarchy) => hierarchy.solve_coarsest(values),
        }
    }
}

/// A block sparse matrix of the hierarchy as plain arrays: `starts` holds `rows + 1` block
/// offsets, `columns` one block column per block (sorted within a row, below `block_columns`),
/// `values` the row-major `row_block` by `col_block` blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelMatrix {
    /// Block columns.
    pub block_columns: usize,
    /// Rows of one block.
    pub row_block: usize,
    /// Columns of one block.
    pub col_block: usize,
    /// Block offsets of each block row.
    pub starts: Vec<usize>,
    /// The block column of each block.
    pub columns: Vec<u32>,
    /// Row-major block values.
    pub values: Vec<f64>,
}

impl Preconditioner for Multigrid {
    fn apply(&self, residual: &[f64], output: &mut [f64]) {
        let pool = self.pool.as_ref();
        match &self.stored {
            // The vectors are already f64: the cycle reads and writes the caller's directly.
            Stored::Double(hierarchy) => {
                hierarchy.apply_direct(self.degree, pool, residual, output)
            }
            Stored::SingleStorage(hierarchy) => {
                hierarchy.apply_direct(self.degree, pool, residual, output)
            }
            Stored::Single(hierarchy) => {
                hierarchy.apply_converted(self.degree, pool, residual, output)
            }
        }
    }

    fn system_multiply(&self, matrix: &CsrMatrix, x: &[f64], y: &mut [f64]) {
        match self.pool.as_ref() {
            Some(pool) if matrix.nonzeros() >= MIN_PARALLEL_VALUES => {
                let bounds = balanced_bounds(matrix.row_starts(), pool.threads());
                for_rows(Some(pool), &bounds, 1, [y], &|rows, [y]| {
                    matrix.multiply_rows(rows, x, y);
                });
            }
            _ => matrix.multiply(x, y),
        }
    }
}

/// One V-cycle from `levels[0]`, writing the correction for right-hand side `rhs` to `x`.
fn cycle<S: Scalar, V: Scalar>(
    levels: &[Level<S>],
    degree: usize,
    pool: Option<&WorkerPool>,
    work: &mut [LevelWork<V>],
    rhs: &[V],
    x: &mut [V],
) {
    let [level, coarser @ ..] = levels else {
        return;
    };
    let Some((level_work, coarser_work)) = work.split_first_mut() else {
        return;
    };
    let Some(prolongator) = &level.prolongator else {
        match &level.coarse {
            Some(factor) => {
                let dense = &mut level_work.dense;
                for (to, from) in dense.iter_mut().zip(rhs) {
                    *to = from.to_f64();
                }
                factor.solve(dense);
                for (to, from) in x.iter_mut().zip(dense.iter()) {
                    *to = V::from_f64(*from);
                }
            }
            None => smooth(level, degree, pool, level_work, rhs, x),
        }
        return;
    };
    smooth(level, degree, pool, level_work, rhs, x);
    let block = level.a.row_block;
    residual(
        level,
        pool,
        &level_work.bounds,
        rhs,
        x,
        &mut level_work.residual,
    );
    prolongator.multiply_transpose(&level_work.residual, &mut level_work.coarse_rhs);
    level_work.coarse_x.fill(V::ZERO);
    cycle(
        coarser,
        degree,
        pool,
        coarser_work,
        &level_work.coarse_rhs,
        &mut level_work.coarse_x,
    );
    let coarse_x = &level_work.coarse_x;
    for_rows(
        pool,
        &level_work.bounds,
        block,
        [level_work.product.as_mut_slice(), &mut *x],
        &|rows, [product, x]| {
            prolongator.multiply_rows(rows, coarse_x, product);
            add(x, product);
        },
    );
    residual(
        level,
        pool,
        &level_work.bounds,
        rhs,
        x,
        &mut level_work.post_rhs,
    );
    // Taken out for the call: `smooth` borrows this level's work vectors while reading these two.
    let post_rhs = std::mem::take(&mut level_work.post_rhs);
    let mut post_x = std::mem::take(&mut level_work.post_x);
    smooth(level, degree, pool, level_work, &post_rhs, &mut post_x);
    for_rows(pool, &level_work.bounds, block, [&mut *x], &|rows, [x]| {
        add(x, &post_x[rows.start * block..]);
    });
    level_work.post_rhs = post_rhs;
    level_work.post_x = post_x;
}

fn add<V: Scalar>(x: &mut [V], y: &[V]) {
    for (target, value) in x.iter_mut().zip(y) {
        *target += *value;
    }
}

/// `r = b - A x`.
fn residual<S: Scalar, V: Scalar>(
    level: &Level<S>,
    pool: Option<&WorkerPool>,
    bounds: &[usize],
    b: &[V],
    x: &[V],
    r: &mut [V],
) {
    let block = level.a.row_block;
    for_rows(pool, bounds, block, [r], &|rows, [r]| {
        let first = rows.start * block;
        level.a.multiply_rows(rows, x, r);
        for (out, value) in r.iter_mut().zip(&b[first..]) {
            *out = *value - *out;
        }
    });
}

/// Applies the Chebyshev smoother from a zero guess: `x` becomes the polynomial in the
/// block-Jacobi operator applied to `b`. The recurrence's coefficients are computed in `f64` and
/// rounded to `V` where they are used: in `f64` these are the same products in the same order as
/// the recurrence written out in full.
fn smooth<S: Scalar, V: Scalar>(
    level: &Level<S>,
    degree: usize,
    pool: Option<&WorkerPool>,
    work: &mut LevelWork<V>,
    b: &[V],
    x: &mut [V],
) {
    let theta = 0.5 * (level.lambda_max + level.lambda_min);
    let delta = 0.5 * (level.lambda_max - level.lambda_min);
    let sigma = theta / delta;
    let mut rho = 1.0 / sigma;
    let block = level.a.row_block;
    let LevelWork {
        residual,
        scaled,
        direction,
        product,
        bounds,
        ..
    } = work;
    let first = V::from_f64(theta);
    for_rows(
        pool,
        bounds,
        block,
        [
            residual.as_mut_slice(),
            scaled.as_mut_slice(),
            direction.as_mut_slice(),
            &mut *x,
        ],
        &|rows, [r, z, d, x]| {
            r.copy_from_slice(&b[rows.start * block..rows.end * block]);
            apply_inverse_diagonal(level, rows, r, z);
            for (d, z) in d.iter_mut().zip(z.iter()) {
                *d = *z / first;
            }
            x.copy_from_slice(d);
        },
    );
    for _ in 1..degree {
        // The product reads every row of `direction`, so it is its own region: the updates below
        // change `direction` chunk by chunk.
        let reads: &[V] = direction;
        for_rows(
            pool,
            bounds,
            block,
            [product.as_mut_slice()],
            &|rows, [ad]| {
                level.a.multiply_rows(rows, reads, ad);
            },
        );
        let next_rho = 1.0 / (2.0 * sigma - rho);
        let keep = V::from_f64(next_rho * rho);
        let step = V::from_f64(2.0 * next_rho / delta);
        let products: &[V] = product;
        for_rows(
            pool,
            bounds,
            block,
            [
                residual.as_mut_slice(),
                scaled.as_mut_slice(),
                direction.as_mut_slice(),
                &mut *x,
            ],
            &|rows, [r, z, d, x]| {
                for (r, ad) in r.iter_mut().zip(&products[rows.start * block..]) {
                    *r -= *ad;
                }
                apply_inverse_diagonal(level, rows, r, z);
                for (d, z) in d.iter_mut().zip(z.iter()) {
                    *d = keep * *d + step * *z;
                }
                add(x, d);
            },
        );
        rho = next_rho;
    }
}

/// Block rows `rows` of the block-diagonal inverse applied to `x` into `y`, both holding exactly
/// those rows.
fn apply_inverse_diagonal<S: Scalar, V: Scalar>(
    level: &Level<S>,
    rows: std::ops::Range<usize>,
    x: &[V],
    y: &mut [V],
) {
    let block = level.a.row_block;
    let start = rows.start;
    for row in rows {
        let d = &level.inverse_diagonal[row * block * block..(row + 1) * block * block];
        let local = row - start;
        for p in 0..block {
            let mut sum = V::ZERO;
            for q in 0..block {
                sum += d[p * block + q].widen::<V>() * x[local * block + q];
            }
            y[local * block + p] = sum;
        }
    }
}

impl MultigridStats {
    fn of_levels<T: Scalar>(levels: &[Level<T>]) -> Self {
        let finest = levels.first().map_or(1, |level| level.a.values.len()) as f64;
        let mut stored = 0usize;
        let mut memory = 0usize;
        for level in levels {
            stored += level.a.values.len();
            memory += level.a.memory_bytes();
            memory += level.inverse_diagonal.len() * std::mem::size_of::<T>();
            if let Some(p) = &level.prolongator {
                memory += p.memory_bytes();
            }
            if let Some(factor) = &level.coarse {
                memory += factor.memory_bytes();
            }
        }
        MultigridStats {
            levels: levels.len(),
            coarse_dofs: levels.last().map_or(0, |level| level.a.dofs()),
            operator_complexity: stored as f64 / finest,
            memory_bytes: memory,
        }
    }
}
