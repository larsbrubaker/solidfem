//! The CPU reference of the kernels in [`super::SMOOTHER_WGSL`]: the same operations in the same
//! order, in either precision. In `f32` it is what a device computes, up to the fused
//! multiply-adds and the 2.5 ULP division WGSL allows; in `f64` it is the multigrid's own
//! smoother, bit for bit (the solver tests compare it with a one-level hierarchy). [`multiply_rounding_bound`] and
//! [`smoother_rounding_bound`] bound how far an `f32` evaluation may sit from the exact result on
//! the same `f32` inputs, so two `f32` evaluations (device and reference) differ by at most twice
//! the bound.

use super::layout::{BlockMatrix, ChebyshevCoefficients, Real};
use super::GpuError;
use crate::multigrid::invert_spd_block;

/// `y = A x`, one sum per scalar row over its blocks in stored order and each block's columns in
/// order, from zero: `multiply_main` with `subtract` 0.
pub fn multiply<T: Real>(matrix: &BlockMatrix<T>, x: &[T], y: &mut [T]) {
    for (row, out) in y
        .iter_mut()
        .enumerate()
        .take(matrix.rows() * matrix.row_block())
    {
        *out = row_product(matrix, row, x);
    }
}

/// `y = y - A x`: `multiply_main` with `subtract` 1.
pub fn multiply_subtract<T: Real>(matrix: &BlockMatrix<T>, x: &[T], y: &mut [T]) {
    for (row, out) in y
        .iter_mut()
        .enumerate()
        .take(matrix.rows() * matrix.row_block())
    {
        *out = *out - row_product(matrix, row, x);
    }
}

fn row_product<T: Real>(matrix: &BlockMatrix<T>, row: usize, x: &[T]) -> T {
    let (rb, cb) = (matrix.row_block(), matrix.col_block());
    let block_row = row / rb;
    let p = row - block_row * rb;
    let starts = matrix.starts();
    let mut sum = T::ZERO;
    for at in starts[block_row] as usize..starts[block_row + 1] as usize {
        let base = at * rb * cb + p * cb;
        let column = matrix.columns()[at] as usize * cb;
        for q in 0..cb {
            sum = sum + matrix.values()[base + q] * x[column + q];
        }
    }
    sum
}

/// The inverse of each diagonal block, row-major, `rows * n * n` values for square `n` by `n`
/// blocks: the block-Jacobi scaling, computed in `f64` by the multigrid's own block inverse, so
/// it is the hierarchy's inverse diagonal to the bit. Round it to `f32` for a device.
pub fn inverse_diagonal(matrix: &BlockMatrix<f64>) -> Result<Vec<f64>, GpuError> {
    let n = matrix.row_block();
    if n != matrix.col_block() {
        return Err(GpuError::BlockSize {
            row_block: n,
            col_block: matrix.col_block(),
        });
    }
    let mut output = vec![0.0; matrix.rows() * n * n];
    let starts = matrix.starts();
    for row in 0..matrix.rows() {
        let range = starts[row] as usize..starts[row + 1] as usize;
        let found = matrix.columns()[range.clone()].binary_search(&(row as u32));
        let at = match found {
            Ok(offset) => range.start + offset,
            Err(_) => return Err(GpuError::NotPositiveDefinite { row, pivot: 0.0 }),
        };
        let block = &matrix.values()[at * n * n..(at + 1) * n * n];
        invert_spd_block(block, n, &mut output[row * n * n..(row + 1) * n * n])
            .map_err(|pivot| GpuError::NotPositiveDefinite { row, pivot })?;
    }
    Ok(output)
}

/// The vectors one smoother application works in, each `rows * row_block` long; `solution`
/// holds the result.
#[derive(Debug, Clone, PartialEq)]
pub struct SmootherVectors<T> {
    /// The running residual.
    pub residual: Vec<T>,
    /// The current Chebyshev direction.
    pub direction: Vec<T>,
    /// The smoothed correction.
    pub solution: Vec<T>,
}

impl<T: Real> SmootherVectors<T> {
    /// Zero vectors of `length`.
    pub fn zeros(length: usize) -> Self {
        SmootherVectors {
            residual: vec![T::ZERO; length],
            direction: vec![T::ZERO; length],
            solution: vec![T::ZERO; length],
        }
    }
}

/// One smoother application from a zero guess, the device's dispatch sequence: the start step,
/// then for each later step the subtracting product and the update. `vectors.solution` becomes
/// the Chebyshev polynomial in the block-Jacobi operator applied to `rhs`.
pub fn chebyshev_smooth<T: Real>(
    matrix: &BlockMatrix<T>,
    inverse_diagonal: &[T],
    coefficients: &ChebyshevCoefficients,
    rhs: &[T],
    vectors: &mut SmootherVectors<T>,
) {
    let n = matrix.row_block();
    let theta = T::from_f64(coefficients.theta);
    for block_row in 0..matrix.rows() {
        let first = block_row * n;
        for p in 0..n {
            let sum = scaled(inverse_diagonal, n, block_row, p, rhs);
            let d = sum / theta;
            vectors.residual[first + p] = rhs[first + p];
            vectors.direction[first + p] = d;
            vectors.solution[first + p] = d;
        }
    }
    for &(direction_scale, scaled_scale) in &coefficients.steps {
        multiply_subtract(matrix, &vectors.direction, &mut vectors.residual);
        let (a, b) = (T::from_f64(direction_scale), T::from_f64(scaled_scale));
        for block_row in 0..matrix.rows() {
            let first = block_row * n;
            for p in 0..n {
                let sum = scaled(inverse_diagonal, n, block_row, p, &vectors.residual);
                let d = a * vectors.direction[first + p] + b * sum;
                vectors.direction[first + p] = d;
                vectors.solution[first + p] = vectors.solution[first + p] + d;
            }
        }
    }
}

/// Row `p` of the diagonal block of `block_row` applied to `x`, summed in column order from zero.
fn scaled<T: Real>(inverse_diagonal: &[T], n: usize, block_row: usize, p: usize, x: &[T]) -> T {
    let first = block_row * n;
    let block = &inverse_diagonal[first * n..(first + n) * n];
    let mut sum = T::ZERO;
    for q in 0..n {
        sum = sum + block[p * n + q] * x[first + q];
    }
    sum
}

/// The `f32` unit of rounding, as `f64`.
const UNIT: f64 = f32::EPSILON as f64;

/// Per scalar row, a bound on `|computed - exact|` for an `f32` evaluation of `A x` on the
/// matrix's `f32` values: a row sum of `k` products rounds at most `k` times, each by at most
/// `UNIT` of a partial sum, so the error is at most `k * UNIT * sum |a_ij x_j|`. A fused
/// multiply-add rounds fewer times, so it stays inside.
pub fn multiply_rounding_bound(matrix: &BlockMatrix<f32>, x: &[f32]) -> Vec<f64> {
    let wide = matrix.convert::<f64>();
    let magnitude: Vec<f64> = x.iter().map(|v| f64::from(v.abs())).collect();
    let abs_matrix = absolute(&wide);
    let mut row_sums = vec![0.0; wide.rows() * wide.row_block()];
    multiply(&abs_matrix, &magnitude, &mut row_sums);
    let starts = wide.starts();
    (0..row_sums.len())
        .map(|row| {
            let block_row = row / wide.row_block();
            let terms = (starts[block_row + 1] - starts[block_row]) as usize * wide.col_block();
            terms as f64 * UNIT * row_sums[row]
        })
        .collect()
}

/// Per entry, a first-order bound on `|computed - exact|` for an `f32` smoother application on
/// the `f32` matrix, inverse diagonal and right-hand side, with the scalars rounded to `f32`:
/// the rounding of every product, sum and scaling is bounded as in [`multiply_rounding_bound`],
/// the division by `theta` by WGSL's 2.5 ULP,
/// and carried through the recurrence with the absolute values of the operators. Products of two
/// rounding errors are below `f32` resolution and are dropped.
pub fn smoother_rounding_bound(
    matrix: &BlockMatrix<f32>,
    inverse_diagonal: &[f32],
    coefficients: &ChebyshevCoefficients,
    rhs: &[f32],
) -> Vec<f64> {
    let wide = matrix.convert::<f64>();
    let abs_a = absolute(&wide);
    let n = wide.row_block();
    let length = wide.rows() * n;
    let diag: Vec<f64> = inverse_diagonal.iter().map(|v| f64::from(*v)).collect();
    let abs_diag: Vec<f64> = diag.iter().map(|v| v.abs()).collect();
    let b: Vec<f64> = rhs.iter().map(|v| f64::from(*v)).collect();
    let theta = f64::from(f32::from_f64(coefficients.theta));
    let mut exact = SmootherVectors::<f64>::zeros(length);
    // The exact values, near enough: f64 on the same f32 inputs.
    let mut z = vec![0.0; length];
    let mut work = vec![0.0; length];
    let mut abs_b = vec![0.0; length];
    block_apply(
        &abs_diag,
        n,
        &b.iter().map(|v| v.abs()).collect::<Vec<_>>(),
        &mut abs_b,
    );
    block_apply(&diag, n, &b, &mut z);
    let mut error_d: Vec<f64> = (0..length)
        // WGSL's f32 division is accurate to 2.5 ULP, not correctly rounded, and one ULP of a
        // value is at most UNIT times its magnitude.
        .map(|i| (n as f64 * UNIT * abs_b[i]) / theta.abs() + 2.5 * UNIT * (z[i] / theta).abs())
        .collect();
    let mut error_r = vec![0.0; length];
    for i in 0..length {
        exact.residual[i] = b[i];
        exact.direction[i] = z[i] / theta;
        exact.solution[i] = exact.direction[i];
    }
    let mut error_x = error_d.clone();
    let terms: Vec<f64> = (0..length)
        .map(|row| {
            let block_row = row / n;
            let starts = wide.starts();
            (starts[block_row + 1] - starts[block_row]) as f64 * wide.col_block() as f64
        })
        .collect();
    let mut abs_work = vec![0.0; length];
    let mut propagated = vec![0.0; length];
    for &(c1, c2) in &coefficients.steps {
        let (c1, c2) = (f64::from(f32::from_f64(c1)), f64::from(f32::from_f64(c2)));
        let abs_direction: Vec<f64> = exact.direction.iter().map(|v| v.abs()).collect();
        multiply(&abs_a, &abs_direction, &mut abs_work);
        multiply(&abs_a, &error_d, &mut propagated);
        multiply_subtract(&wide, &exact.direction, &mut exact.residual);
        for i in 0..length {
            error_r[i] +=
                terms[i] * UNIT * abs_work[i] + propagated[i] + UNIT * exact.residual[i].abs();
        }
        block_apply(&diag, n, &exact.residual, &mut z);
        let abs_residual: Vec<f64> = exact.residual.iter().map(|v| v.abs()).collect();
        block_apply(&abs_diag, n, &abs_residual, &mut abs_work);
        block_apply(&abs_diag, n, &error_r, &mut work);
        for i in 0..length {
            let error_z = n as f64 * UNIT * abs_work[i] + work[i];
            let d = c1 * exact.direction[i] + c2 * z[i];
            error_d[i] = c1.abs() * error_d[i]
                + c2.abs() * error_z
                + 3.0 * UNIT * ((c1 * exact.direction[i]).abs() + (c2 * z[i]).abs());
            exact.direction[i] = d;
            exact.solution[i] += d;
            error_x[i] += error_d[i] + UNIT * exact.solution[i].abs();
        }
    }
    error_x
}

/// `y = blockdiag(d) x` for row-major `n` by `n` diagonal blocks.
fn block_apply(d: &[f64], n: usize, x: &[f64], y: &mut [f64]) {
    for block_row in 0..x.len() / n {
        for p in 0..n {
            y[block_row * n + p] = scaled(d, n, block_row, p, x);
        }
    }
}

/// The matrix of absolute values.
fn absolute(matrix: &BlockMatrix<f64>) -> BlockMatrix<f64> {
    matrix.map_values(f64::abs)
}
