//! The finished multigrid hierarchy and its symmetric V-cycle. The cycle smooths with a Chebyshev
//! polynomial, corrects from the next coarser level and smooths with the same polynomial again,
//! which keeps the operator symmetric; the coarsest level is solved by a dense factorization, or
//! by its smoother when coarsening stalled above the factorization size limit.

use std::cell::RefCell;

use super::block::{BlockCsr, DenseCholesky};
use super::MultigridStats;
use crate::preconditioner::Preconditioner;

/// One level of the hierarchy.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Level {
    pub a: BlockCsr,
    pub inverse_diagonal: Vec<f64>,
    pub lambda_min: f64,
    pub lambda_max: f64,
    /// The prolongator to the next coarser level, absent on the coarsest.
    pub prolongator: Option<BlockCsr>,
    /// The direct solve on the coarsest level; absent means smoothing alone.
    pub coarse: Option<DenseCholesky>,
}

/// Reused vectors of one level, so applying the preconditioner allocates nothing.
#[derive(Debug, Clone, PartialEq, Default)]
struct LevelWork {
    residual: Vec<f64>,
    scaled: Vec<f64>,
    direction: Vec<f64>,
    product: Vec<f64>,
    post_rhs: Vec<f64>,
    post_x: Vec<f64>,
    coarse_rhs: Vec<f64>,
    coarse_x: Vec<f64>,
}

impl LevelWork {
    fn for_level(level: &Level, coarse_dofs: usize) -> Self {
        let dofs = level.a.dofs();
        LevelWork {
            residual: vec![0.0; dofs],
            scaled: vec![0.0; dofs],
            direction: vec![0.0; dofs],
            product: vec![0.0; dofs],
            post_rhs: vec![0.0; dofs],
            post_x: vec![0.0; dofs],
            coarse_rhs: vec![0.0; coarse_dofs],
            coarse_x: vec![0.0; coarse_dofs],
        }
    }
}

/// A multigrid V-cycle preconditioner over a hierarchy built by [`super::MultigridSetup`].
pub struct Multigrid {
    levels: Vec<Level>,
    degree: usize,
    stats: MultigridStats,
    workspace: RefCell<Vec<LevelWork>>,
}

impl Multigrid {
    pub(super) fn from_levels(levels: Vec<Level>, degree: usize) -> Self {
        let mut work = Vec::with_capacity(levels.len());
        for (index, level) in levels.iter().enumerate() {
            let coarse_dofs = levels.get(index + 1).map_or(0, |next| next.a.dofs());
            work.push(LevelWork::for_level(level, coarse_dofs));
        }
        let stats = MultigridStats::of_levels(&levels);
        Multigrid {
            levels,
            degree,
            stats,
            workspace: RefCell::new(work),
        }
    }

    /// The shape and cost of the hierarchy.
    pub fn stats(&self) -> MultigridStats {
        self.stats
    }
}

impl Preconditioner for Multigrid {
    fn apply(&self, residual: &[f64], output: &mut [f64]) {
        let mut workspace = self.workspace.borrow_mut();
        cycle(&self.levels, self.degree, &mut workspace, residual, output);
    }
}

/// One V-cycle from `levels[0]`, writing the correction for right-hand side `rhs` to `x`.
fn cycle(levels: &[Level], degree: usize, work: &mut [LevelWork], rhs: &[f64], x: &mut [f64]) {
    let [level, coarser @ ..] = levels else {
        return;
    };
    let Some((level_work, coarser_work)) = work.split_first_mut() else {
        return;
    };
    let Some(prolongator) = &level.prolongator else {
        match &level.coarse {
            Some(factor) => {
                x.copy_from_slice(rhs);
                factor.solve(x);
            }
            None => smooth(level, degree, level_work, rhs, x),
        }
        return;
    };
    smooth(level, degree, level_work, rhs, x);
    residual(level, rhs, x, &mut level_work.residual);
    prolongator.multiply_transpose(&level_work.residual, &mut level_work.coarse_rhs);
    level_work.coarse_x.fill(0.0);
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

fn add(x: &mut [f64], y: &[f64]) {
    for (target, value) in x.iter_mut().zip(y) {
        *target += value;
    }
}

/// `r = b - A x`.
fn residual(level: &Level, b: &[f64], x: &[f64], r: &mut [f64]) {
    level.a.multiply(x, r);
    for (out, value) in r.iter_mut().zip(b) {
        *out = value - *out;
    }
}

/// Applies the Chebyshev smoother from a zero guess: `x` becomes the polynomial in the
/// block-Jacobi operator applied to `b`.
fn smooth(level: &Level, degree: usize, work: &mut LevelWork, b: &[f64], x: &mut [f64]) {
    let theta = 0.5 * (level.lambda_max + level.lambda_min);
    let delta = 0.5 * (level.lambda_max - level.lambda_min);
    let sigma = theta / delta;
    let mut rho = 1.0 / sigma;
    work.residual.copy_from_slice(b);
    apply_inverse_diagonal(level, &work.residual, &mut work.scaled);
    for (d, z) in work.direction.iter_mut().zip(&work.scaled) {
        *d = z / theta;
    }
    x.copy_from_slice(&work.direction);
    for _ in 1..degree {
        level.a.multiply(&work.direction, &mut work.product);
        for (r, ad) in work.residual.iter_mut().zip(&work.product) {
            *r -= ad;
        }
        apply_inverse_diagonal(level, &work.residual, &mut work.scaled);
        let next_rho = 1.0 / (2.0 * sigma - rho);
        for (d, z) in work.direction.iter_mut().zip(&work.scaled) {
            *d = next_rho * rho * *d + 2.0 * next_rho / delta * z;
        }
        add(x, &work.direction);
        rho = next_rho;
    }
}

fn apply_inverse_diagonal(level: &Level, x: &[f64], y: &mut [f64]) {
    let block = level.a.row_block;
    for row in 0..level.a.rows {
        let d = &level.inverse_diagonal[row * block * block..(row + 1) * block * block];
        for p in 0..block {
            let mut sum = 0.0;
            for q in 0..block {
                sum += d[p * block + q] * x[row * block + q];
            }
            y[row * block + p] = sum;
        }
    }
}

impl MultigridStats {
    pub(super) fn of_levels(levels: &[Level]) -> Self {
        let finest = levels.first().map_or(1, |level| level.a.values.len()) as f64;
        let mut stored = 0usize;
        let mut memory = 0usize;
        for level in levels {
            stored += level.a.values.len();
            memory += level.a.memory_bytes();
            memory += level.inverse_diagonal.len() * std::mem::size_of::<f64>();
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
