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

use std::cell::RefCell;

use super::block::{BlockCsr, DenseCholesky};
use super::scalar::Scalar;
use super::{MultigridStats, Precision};
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
}

impl<V: Scalar> LevelWork<V> {
    fn for_level<S: Scalar>(level: &Level<S>, coarse_dofs: usize) -> Self {
        let dofs = level.a.dofs();
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
    fn apply_direct(&self, degree: usize, residual: &[f64], output: &mut [f64]) {
        let mut work = self.work.borrow_mut();
        cycle(&self.levels, degree, &mut work.levels, residual, output);
    }
}

impl<S: Scalar, V: Scalar> Hierarchy<S, V> {
    fn new(levels: Vec<Level<S>>, converts: bool) -> Self {
        let mut work = Vec::with_capacity(levels.len());
        for (index, level) in levels.iter().enumerate() {
            let coarse_dofs = levels.get(index + 1).map_or(0, |next| next.a.dofs());
            work.push(LevelWork::for_level(level, coarse_dofs));
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

    /// Rounds `residual` into `V`, runs the cycle and widens the correction into `output`.
    fn apply_converted(&self, degree: usize, residual: &[f64], output: &mut [f64]) {
        let mut work = self.work.borrow_mut();
        let HierarchyWork { levels, rhs, x } = &mut *work;
        for (to, from) in rhs.iter_mut().zip(residual) {
            *to = V::from_f64(*from);
        }
        cycle(&self.levels, degree, levels, rhs, x);
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
}

impl Multigrid {
    /// Takes the `f64` levels setup built and stores them in `precision`. Converting consumes one
    /// level at a time, so the peak is the `f64` hierarchy plus one converted level.
    pub(super) fn from_levels(levels: Vec<Level>, degree: usize, precision: Precision) -> Self {
        let (stored, stats) = match precision {
            Precision::Double => {
                let stats = MultigridStats::of_levels(&levels);
                (Stored::Double(Hierarchy::new(levels, false)), stats)
            }
            Precision::SingleStorage => {
                let levels: Vec<Level<f32>> =
                    levels.into_iter().map(Level::into_precision).collect();
                let stats = MultigridStats::of_levels(&levels);
                (Stored::SingleStorage(Hierarchy::new(levels, false)), stats)
            }
            Precision::Single => {
                let levels: Vec<Level<f32>> =
                    levels.into_iter().map(Level::into_precision).collect();
                let stats = MultigridStats::of_levels(&levels);
                (Stored::Single(Hierarchy::new(levels, true)), stats)
            }
        };
        Multigrid {
            stored,
            degree,
            stats,
        }
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
}

impl Preconditioner for Multigrid {
    fn apply(&self, residual: &[f64], output: &mut [f64]) {
        match &self.stored {
            // The vectors are already f64: the cycle reads and writes the caller's directly.
            Stored::Double(hierarchy) => hierarchy.apply_direct(self.degree, residual, output),
            Stored::SingleStorage(hierarchy) => {
                hierarchy.apply_direct(self.degree, residual, output)
            }
            Stored::Single(hierarchy) => hierarchy.apply_converted(self.degree, residual, output),
        }
    }
}

/// One V-cycle from `levels[0]`, writing the correction for right-hand side `rhs` to `x`.
fn cycle<S: Scalar, V: Scalar>(
    levels: &[Level<S>],
    degree: usize,
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
            None => smooth(level, degree, level_work, rhs, x),
        }
        return;
    };
    smooth(level, degree, level_work, rhs, x);
    residual(level, rhs, x, &mut level_work.residual);
    prolongator.multiply_transpose(&level_work.residual, &mut level_work.coarse_rhs);
    level_work.coarse_x.fill(V::ZERO);
    cycle(
        coarser,
        degree,
        coarser_work,
        &level_work.coarse_rhs,
        &mut level_work.coarse_x,
    );
    prolongator.multiply(&level_work.coarse_x, &mut level_work.product);
    add(x, &level_work.product);
    residual(level, rhs, x, &mut level_work.post_rhs);
    // Taken out for the call: `smooth` borrows this level's work vectors while reading these two.
    let post_rhs = std::mem::take(&mut level_work.post_rhs);
    let mut post_x = std::mem::take(&mut level_work.post_x);
    smooth(level, degree, level_work, &post_rhs, &mut post_x);
    add(x, &post_x);
    level_work.post_rhs = post_rhs;
    level_work.post_x = post_x;
}

fn add<V: Scalar>(x: &mut [V], y: &[V]) {
    for (target, value) in x.iter_mut().zip(y) {
        *target += *value;
    }
}

/// `r = b - A x`.
fn residual<S: Scalar, V: Scalar>(level: &Level<S>, b: &[V], x: &[V], r: &mut [V]) {
    level.a.multiply(x, r);
    for (out, value) in r.iter_mut().zip(b) {
        *out = *value - *out;
    }
}

/// Applies the Chebyshev smoother from a zero guess: `x` becomes the polynomial in the
/// block-Jacobi operator applied to `b`. The recurrence's coefficients are computed in `f64` and
/// rounded to `V` where they are used: in `f64` these are the same products in the same order as
/// the recurrence written out in full.
fn smooth<S: Scalar, V: Scalar>(
    level: &Level<S>,
    degree: usize,
    work: &mut LevelWork<V>,
    b: &[V],
    x: &mut [V],
) {
    let theta = 0.5 * (level.lambda_max + level.lambda_min);
    let delta = 0.5 * (level.lambda_max - level.lambda_min);
    let sigma = theta / delta;
    let mut rho = 1.0 / sigma;
    work.residual.copy_from_slice(b);
    apply_inverse_diagonal(level, &work.residual, &mut work.scaled);
    let first = V::from_f64(theta);
    for (d, z) in work.direction.iter_mut().zip(&work.scaled) {
        *d = *z / first;
    }
    x.copy_from_slice(&work.direction);
    for _ in 1..degree {
        level.a.multiply(&work.direction, &mut work.product);
        for (r, ad) in work.residual.iter_mut().zip(&work.product) {
            *r -= *ad;
        }
        apply_inverse_diagonal(level, &work.residual, &mut work.scaled);
        let next_rho = 1.0 / (2.0 * sigma - rho);
        let keep = V::from_f64(next_rho * rho);
        let step = V::from_f64(2.0 * next_rho / delta);
        for (d, z) in work.direction.iter_mut().zip(&work.scaled) {
            *d = keep * *d + step * *z;
        }
        add(x, &work.direction);
        rho = next_rho;
    }
}

fn apply_inverse_diagonal<S: Scalar, V: Scalar>(level: &Level<S>, x: &[V], y: &mut [V]) {
    let block = level.a.row_block;
    for row in 0..level.a.rows {
        let d = &level.inverse_diagonal[row * block * block..(row + 1) * block * block];
        for p in 0..block {
            let mut sum = V::ZERO;
            for q in 0..block {
                sum += d[p * block + q].widen::<V>() * x[row * block + q];
            }
            y[row * block + p] = sum;
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
