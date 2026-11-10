//! Block sparse storage for the multigrid and the small dense kernels it is built from. A row of
//! a [`BlockCsr`] holds sorted block columns; values are row-major dense blocks. Everything here
//! runs in row and column order, so rounding depends only on the data. Setup builds every matrix
//! in `f64`; a finished level's matrices may be converted to `f32` ([`BlockCsr::into_precision`]).
//! The products widen each stored value to the vector type as they read it and run in that type,
//! so an `f32` matrix can be applied to `f64` vectors or to `f32` ones.

use crate::csr::CsrMatrix;

use super::scalar::Scalar;
use super::MultigridError;

/// A sparse matrix of dense `row_block` by `col_block` blocks.
#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct BlockCsr<T: Scalar = f64> {
    pub rows: usize,
    pub row_block: usize,
    pub col_block: usize,
    pub starts: Vec<usize>,
    pub columns: Vec<u32>,
    pub values: Vec<T>,
}

impl BlockCsr<f64> {
    /// The same matrix with its values rounded to `U`; the structure moves over unchanged.
    pub fn into_precision<U: Scalar>(self) -> BlockCsr<U> {
        BlockCsr {
            rows: self.rows,
            row_block: self.row_block,
            col_block: self.col_block,
            starts: self.starts,
            columns: self.columns,
            values: self.values.into_iter().map(U::from_f64).collect(),
        }
    }

    /// An empty matrix to append rows to.
    pub fn with_shape(rows: usize, row_block: usize, col_block: usize) -> Self {
        BlockCsr {
            rows,
            row_block,
            col_block,
            starts: vec![0],
            columns: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Reads a scalar matrix whose rows are in groups of three sharing one pattern of whole
    /// three-column blocks, which is what element assembly over nodes produces.
    pub fn from_nodal(matrix: &CsrMatrix) -> Result<Self, MultigridError> {
        let size = matrix.size();
        if !size.is_multiple_of(3) {
            return Err(MultigridError::BlockStructure { row: size });
        }
        let rows = size / 3;
        let mut block = BlockCsr::with_shape(rows, 3, 3);
        block.columns.reserve(matrix.nonzeros() / 9);
        block.values.reserve(matrix.nonzeros());
        for node in 0..rows {
            let (one, one_values) = matrix.row(3 * node);
            let (two, two_values) = matrix.row(3 * node + 1);
            let (three, three_values) = matrix.row(3 * node + 2);
            if one != two {
                return Err(MultigridError::BlockStructure { row: 3 * node + 1 });
            }
            if one != three {
                return Err(MultigridError::BlockStructure { row: 3 * node + 2 });
            }
            if !one.len().is_multiple_of(3) {
                return Err(MultigridError::BlockStructure { row: 3 * node });
            }
            for at in (0..one.len()).step_by(3) {
                let triple = &one[at..at + 3];
                let column = triple[0] as usize;
                if !column.is_multiple_of(3)
                    || triple[1] as usize != column + 1
                    || triple[2] as usize != column + 2
                {
                    return Err(MultigridError::BlockStructure { row: 3 * node });
                }
                // Fits in u32: it is a third of a column index that is already a u32.
                block.columns.push((column / 3) as u32);
            }
            for at in 0..one.len() / 3 {
                block
                    .values
                    .extend_from_slice(&one_values[3 * at..3 * at + 3]);
                block
                    .values
                    .extend_from_slice(&two_values[3 * at..3 * at + 3]);
                block
                    .values
                    .extend_from_slice(&three_values[3 * at..3 * at + 3]);
            }
            block.starts.push(block.columns.len());
        }
        Ok(block)
    }
}

impl<T: Scalar> BlockCsr<T> {
    /// Values in one block.
    #[inline]
    pub fn block_len(&self) -> usize {
        self.row_block * self.col_block
    }

    /// Scalar rows.
    #[inline]
    pub fn dofs(&self) -> usize {
        self.rows * self.row_block
    }

    /// The block range of a block row.
    #[inline]
    pub fn row_range(&self, row: usize) -> std::ops::Range<usize> {
        self.starts[row]..self.starts[row + 1]
    }

    /// The values of block `at`.
    #[inline]
    pub fn block(&self, at: usize) -> &[T] {
        let len = self.block_len();
        &self.values[at * len..(at + 1) * len]
    }

    /// `y = self * x`, each stored value widened to the vector type `V` as it is read; rows sum
    /// in column order.
    pub fn multiply<V: Scalar>(&self, x: &[V], y: &mut [V]) {
        let (rb, cb) = (self.row_block, self.col_block);
        for row in 0..self.rows {
            let out = &mut y[row * rb..(row + 1) * rb];
            out.fill(V::ZERO);
            for at in self.row_range(row) {
                let block = self.block(at);
                let col = self.columns[at] as usize * cb;
                for (p, out) in out.iter_mut().enumerate() {
                    let mut sum = *out;
                    for q in 0..cb {
                        sum += block[p * cb + q].widen::<V>() * x[col + q];
                    }
                    *out = sum;
                }
            }
        }
    }

    /// `y = transpose(self) * x`, in row order.
    pub fn multiply_transpose<V: Scalar>(&self, x: &[V], y: &mut [V]) {
        let (rb, cb) = (self.row_block, self.col_block);
        y.fill(V::ZERO);
        for row in 0..self.rows {
            for at in self.row_range(row) {
                let block = self.block(at);
                let col = self.columns[at] as usize * cb;
                for p in 0..rb {
                    let value = x[row * rb + p];
                    for q in 0..cb {
                        y[col + q] += block[p * cb + q].widen::<V>() * value;
                    }
                }
            }
        }
    }

    /// Bytes held.
    pub fn memory_bytes(&self) -> usize {
        self.starts.len() * std::mem::size_of::<usize>()
            + self.columns.len() * std::mem::size_of::<u32>()
            + self.values.len() * std::mem::size_of::<T>()
    }
}

/// Inverts the symmetric positive definite `n` by `n` block `input` (`n` at most six) into
/// `output`, both row-major. The inverse is the product of the inverted Cholesky factor with its
/// own transpose, so the two sides of the diagonal come from the same products in the same order
/// and agree to the bit. Returns the failing pivot otherwise.
pub(super) fn invert_spd_block(input: &[f64], n: usize, output: &mut [f64]) -> Result<(), f64> {
    let mut lower = [0.0f64; 36];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = input[i * n + j];
            for k in 0..j {
                sum -= lower[i * n + k] * lower[j * n + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(sum);
                }
                lower[i * n + j] = sum.sqrt();
            } else {
                lower[i * n + j] = sum / lower[j * n + j];
            }
        }
    }
    let mut inverse = [0.0f64; 36];
    for column in 0..n {
        for i in column..n {
            let mut sum = if i == column { 1.0 } else { 0.0 };
            for k in column..i {
                sum -= lower[i * n + k] * inverse[k * n + column];
            }
            inverse[i * n + column] = sum / lower[i * n + i];
        }
    }
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in i.max(j)..n {
                sum += inverse[k * n + i] * inverse[k * n + j];
            }
            output[i * n + j] = sum;
        }
    }
    Ok(())
}

/// `C += A B` for row-major dense blocks with `a_rows` rows, `middle` columns of `A` and
/// `b_cols` columns of `B`.
#[inline]
pub(super) fn add_product(
    out: &mut [f64],
    a: &[f64],
    a_rows: usize,
    middle: usize,
    b: &[f64],
    b_cols: usize,
) {
    for row in 0..a_rows {
        for mid in 0..middle {
            let scale = a[row * middle + mid];
            for col in 0..b_cols {
                out[row * b_cols + col] += scale * b[mid * b_cols + col];
            }
        }
    }
}

/// Gathers a sparse row whose columns arrive in any order; blocks are summed into per-column
/// slots and read back sorted, so the output depends only on the data.
pub(super) struct Accumulator {
    slot: Vec<u32>,
    touched: Vec<u32>,
    data: Vec<f64>,
    block_len: usize,
}

const NO_SLOT: u32 = u32::MAX;

impl Accumulator {
    /// An accumulator for columns below `columns`, summing blocks of `block_len` values.
    pub fn new(columns: usize, block_len: usize) -> Self {
        Accumulator {
            slot: vec![NO_SLOT; columns],
            touched: Vec::new(),
            data: Vec::new(),
            block_len,
        }
    }

    /// Clears the row.
    pub fn begin(&mut self) {
        for column in self.touched.drain(..) {
            self.slot[column as usize] = NO_SLOT;
        }
        self.data.clear();
    }

    /// The block for `column`, adding a zero one the first time it is touched.
    #[inline]
    pub fn entry(&mut self, column: u32) -> &mut [f64] {
        let at = self.slot[column as usize];
        let at = if at == NO_SLOT {
            let at = self.touched.len() as u32;
            self.slot[column as usize] = at;
            self.touched.push(column);
            self.data.resize(self.data.len() + self.block_len, 0.0);
            at
        } else {
            at
        };
        let start = at as usize * self.block_len;
        &mut self.data[start..start + self.block_len]
    }

    /// Calls `emit(column, block)` for each touched column in ascending order.
    pub fn drain_sorted(&mut self, mut emit: impl FnMut(u32, &[f64])) {
        self.touched.sort_unstable();
        for &column in &self.touched {
            let at = self.slot[column as usize] as usize * self.block_len;
            emit(column, &self.data[at..at + self.block_len]);
        }
    }
}

/// A dense symmetric positive definite matrix factored one column per call, so the direct solve
/// on the coarsest level can be spread over slices.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DenseCholesky {
    n: usize,
    /// Row-major; the lower triangle becomes the factor column by column.
    factor: Vec<f64>,
    next: usize,
}

impl DenseCholesky {
    /// Starts factoring the row-major `n` by `n` matrix.
    pub fn new(values: Vec<f64>, n: usize) -> Self {
        debug_assert_eq!(values.len(), n * n);
        DenseCholesky {
            n,
            factor: values,
            next: 0,
        }
    }

    /// Factors one more column; returns `Ok(true)` once all are done and `Err((column, pivot))`
    /// when a pivot is not positive.
    pub fn factor_column(&mut self) -> Result<bool, (usize, f64)> {
        let (n, j) = (self.n, self.next);
        if j == n {
            return Ok(true);
        }
        for i in j..n {
            let mut sum = self.factor[i * n + j];
            for k in 0..j {
                sum -= self.factor[i * n + k] * self.factor[j * n + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return Err((j, sum));
                }
                self.factor[j * n + j] = sum.sqrt();
            } else {
                self.factor[i * n + j] = sum / self.factor[j * n + j];
            }
        }
        self.next += 1;
        Ok(self.next == n)
    }

    /// Bytes held.
    pub fn memory_bytes(&self) -> usize {
        self.factor.len() * std::mem::size_of::<f64>()
    }

    /// Solves in place with the finished factor: forward then back substitution.
    #[allow(clippy::needless_range_loop)] // `x` and the factor share the index.
    pub fn solve(&self, x: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            let mut sum = x[i];
            for k in 0..i {
                sum -= self.factor[i * n + k] * x[k];
            }
            x[i] = sum / self.factor[i * n + i];
        }
        for i in (0..n).rev() {
            let mut sum = x[i];
            for k in i + 1..n {
                sum -= self.factor[k * n + i] * x[k];
            }
            x[i] = sum / self.factor[i * n + i];
        }
    }
}

/// A deterministic start value for the power iteration: positive and uncorrelated with node
/// numbering, so it has a component along any eigenvector the matrix can have. Integer mixing
/// keeps it bit-identical on every platform.
pub(super) fn start_value(index: usize) -> f64 {
    let mixed = (index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    0.5 + (mixed >> 11) as f64 / ((1u64 << 53) as f64)
}

pub(super) fn apply_block_diagonal(inverse: &[f64], block: usize, x: &[f64], y: &mut [f64]) {
    let rows = x.len() / block;
    for row in 0..rows {
        let d = &inverse[row * block * block..(row + 1) * block * block];
        for p in 0..block {
            let mut sum = 0.0;
            for q in 0..block {
                sum += d[p * block + q] * x[row * block + q];
            }
            y[row * block + p] = sum;
        }
    }
}

pub(super) fn norm2(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for v in x {
        sum += v * v;
    }
    sum.sqrt()
}

pub(super) fn normalize(x: &mut [f64]) {
    let norm = norm2(x);
    for v in x {
        *v /= norm;
    }
}
