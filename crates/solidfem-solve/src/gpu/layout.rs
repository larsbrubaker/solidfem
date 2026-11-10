//! The records a GPU reads, as the `gpu` module header specifies: [`BlockMatrix`], block sparse
//! rows of dense blocks in either precision (`f32` is what a device holds, `f64` runs the same
//! reference arithmetic at full precision), [`GpuParams`], the uniform of one dispatch, and
//! [`ChebyshevCoefficients`], the smoother's per-step scalars. Bytes are written field by field
//! in little-endian order, never by reinterpreting memory, so the packing is the same on every
//! host.

use crate::csr::CsrMatrix;
use crate::LevelMatrix;

use super::shader::MAX_BLOCK;
use super::GpuError;

/// Threads per workgroup in every kernel of [`super::SMOOTHER_WGSL`] and
/// [`super::TRANSFER_WGSL`].
pub const WORKGROUP_SIZE: u32 = 64;

/// Bytes of one [`GpuParams`] record.
pub const PARAMS_STRIDE: usize = 32;

/// The most workgroups WebGPU guarantees in one dispatch dimension.
const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65_535;

/// The workgroup counts `[x, y]` that cover `threads` threads: a kernel's thread index is
/// `x + y * groups_x * WORKGROUP_SIZE`, and threads past the end return at once.
pub fn dispatch_size(threads: u32) -> [u32; 2] {
    let groups = threads.div_ceil(WORKGROUP_SIZE).max(1);
    if groups <= MAX_WORKGROUPS_PER_DIMENSION {
        return [groups, 1];
    }
    let y = groups.div_ceil(MAX_WORKGROUPS_PER_DIMENSION);
    [groups.div_ceil(y), y]
}

mod sealed {
    /// Only `f32` and `f64` implement [`super::Real`].
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// A floating-point type the reference arithmetic runs in: `f32` or `f64`, sealed.
pub trait Real:
    sealed::Sealed
    + Copy
    + PartialEq
    + std::fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    /// Zero.
    const ZERO: Self;
    /// The nearest value to `value`.
    fn from_f64(value: f64) -> Self;
    /// The value widened to `f64`.
    fn to_f64(self) -> f64;
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    fn from_f64(value: f64) -> Self {
        // Rounds to nearest; magnitudes beyond f32 become infinite, which the comparison tests
        // would show.
        value as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    fn from_f64(value: f64) -> Self {
        value
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// A sparse matrix of dense `row_block` by `col_block` blocks, in the layout a device reads.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockMatrix<T> {
    rows: usize,
    columns_in_blocks: usize,
    row_block: usize,
    col_block: usize,
    starts: Vec<u32>,
    columns: Vec<u32>,
    values: Vec<T>,
}

impl<T: Real> BlockMatrix<T> {
    /// Builds from plain block sparse arrays: `starts` holds `rows + 1` offsets into the blocks,
    /// `columns` one block column (below `block_columns`) per block, sorted within a row, and
    /// `values` the row-major blocks, rounded to `T`.
    pub fn from_parts(
        block_columns: usize,
        row_block: usize,
        col_block: usize,
        starts: &[usize],
        columns: &[u32],
        values: &[f64],
    ) -> Result<Self, GpuError> {
        if row_block == 0 || col_block == 0 || row_block > MAX_BLOCK || col_block > MAX_BLOCK {
            return Err(GpuError::BlockSize {
                row_block,
                col_block,
            });
        }
        let rows = starts
            .len()
            .checked_sub(1)
            .ok_or(GpuError::Structure { row: 0 })?;
        let blocks = columns.len();
        if starts[0] != 0 || starts[rows] != blocks {
            return Err(GpuError::Structure { row: rows });
        }
        if values.len() != blocks * row_block * col_block {
            return Err(GpuError::Structure { row: rows });
        }
        let limit = u32::MAX as usize;
        for count in [values.len(), rows * row_block, block_columns * col_block] {
            if count >= limit {
                return Err(GpuError::TooLarge { count });
            }
        }
        for row in 0..rows {
            let (start, end) = (starts[row], starts[row + 1]);
            if end < start || end > blocks {
                return Err(GpuError::Structure { row });
            }
            let row_columns = &columns[start..end];
            let sorted = row_columns.windows(2).all(|pair| pair[0] < pair[1]);
            let inside = row_columns.iter().all(|&c| (c as usize) < block_columns);
            if !sorted || !inside {
                return Err(GpuError::Structure { row });
            }
        }
        Ok(BlockMatrix {
            rows,
            columns_in_blocks: block_columns,
            row_block,
            col_block,
            // Fit in u32: every offset is at most `blocks`, below `values.len()`, checked above.
            starts: starts.iter().map(|&s| s as u32).collect(),
            columns: columns.to_vec(),
            values: values.iter().map(|&v| T::from_f64(v)).collect(),
        })
    }

    /// Reads a stiffness matrix of 3x3 nodal blocks: rows in groups of three sharing one pattern
    /// of whole three-column blocks, which is what element assembly over nodes produces.
    pub fn from_nodal(matrix: &CsrMatrix) -> Result<Self, GpuError> {
        let size = matrix.size();
        if !size.is_multiple_of(3) {
            return Err(GpuError::Structure { row: size / 3 });
        }
        let rows = size / 3;
        let mut starts = Vec::with_capacity(rows + 1);
        let mut columns = Vec::with_capacity(matrix.nonzeros() / 9);
        let mut values = Vec::with_capacity(matrix.nonzeros());
        starts.push(0);
        for node in 0..rows {
            let lines = [
                matrix.row(3 * node),
                matrix.row(3 * node + 1),
                matrix.row(3 * node + 2),
            ];
            let pattern = lines[0].0;
            let whole = pattern.len().is_multiple_of(3)
                && lines.iter().all(|(c, _)| *c == pattern)
                && pattern
                    .chunks(3)
                    .all(|t| t[0].is_multiple_of(3) && t[1] == t[0] + 1 && t[2] == t[0] + 2);
            if !whole {
                return Err(GpuError::Structure { row: node });
            }
            for (block, triple) in pattern.chunks(3).enumerate() {
                columns.push(triple[0] / 3);
                for (_, line_values) in &lines {
                    values.extend_from_slice(&line_values[3 * block..3 * block + 3]);
                }
            }
            starts.push(columns.len());
        }
        Self::from_parts(rows, 3, 3, &starts, &columns, &values)
    }

    /// Reads a matrix of a multigrid hierarchy ([`crate::Multigrid::level_matrix`] or
    /// [`crate::Multigrid::level_prolongator`]): rounding to `f32` gives back exactly the values
    /// a [`crate::Precision::Single`] hierarchy stores.
    pub fn from_level(matrix: &LevelMatrix) -> Result<Self, GpuError> {
        Self::from_parts(
            matrix.block_columns,
            matrix.row_block,
            matrix.col_block,
            &matrix.starts,
            &matrix.columns,
            &matrix.values,
        )
    }

    /// Block rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Block columns.
    pub fn block_columns(&self) -> usize {
        self.columns_in_blocks
    }

    /// Rows of one block.
    pub fn row_block(&self) -> usize {
        self.row_block
    }

    /// Columns of one block.
    pub fn col_block(&self) -> usize {
        self.col_block
    }

    /// Block offsets of each block row, `rows + 1` of them.
    pub fn starts(&self) -> &[u32] {
        &self.starts
    }

    /// The block column of each block.
    pub fn columns(&self) -> &[u32] {
        &self.columns
    }

    /// Row-major block values.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// The same pattern with every value passed through `f`.
    pub(super) fn map_values(&self, f: impl Fn(T) -> T) -> Self {
        BlockMatrix {
            values: self.values.iter().map(|&v| f(v)).collect(),
            ..self.clone()
        }
    }

    /// The transpose, stored explicitly so that applying it is a gather: block row `c` of the
    /// result holds every block of block column `c` of this matrix, in ascending order of this
    /// matrix's block row, each block transposed. A thread per scalar row of the transpose then
    /// sums its terms in that order with no scatter and no atomics, which is exactly the order a
    /// row-by-row scatter of `transpose(self) * x` adds them in, so the two give the same bits.
    pub fn transpose(&self) -> Self {
        let (rb, cb) = (self.row_block, self.col_block);
        let mut starts = vec![0u32; self.columns_in_blocks + 1];
        for &column in &self.columns {
            starts[column as usize + 1] += 1;
        }
        for i in 0..self.columns_in_blocks {
            starts[i + 1] += starts[i];
        }
        let mut filled: Vec<usize> = starts.iter().map(|&s| s as usize).collect();
        let mut columns = vec![0u32; self.columns.len()];
        let mut values = vec![T::ZERO; self.values.len()];
        for row in 0..self.rows {
            for at in self.starts[row] as usize..self.starts[row + 1] as usize {
                let column = self.columns[at] as usize;
                let to = filled[column];
                filled[column] += 1;
                // Fits in u32: `from_parts` refuses 2^32 scalar rows.
                columns[to] = row as u32;
                let source = &self.values[at * rb * cb..(at + 1) * rb * cb];
                let target = &mut values[to * rb * cb..(to + 1) * rb * cb];
                for p in 0..rb {
                    for q in 0..cb {
                        target[q * rb + p] = source[p * cb + q];
                    }
                }
            }
        }
        BlockMatrix {
            rows: self.columns_in_blocks,
            columns_in_blocks: self.rows,
            row_block: cb,
            col_block: rb,
            starts,
            columns,
            values,
        }
    }

    /// The same matrix in another precision, each value rounded to nearest.
    pub fn convert<U: Real>(&self) -> BlockMatrix<U> {
        BlockMatrix {
            rows: self.rows,
            columns_in_blocks: self.columns_in_blocks,
            row_block: self.row_block,
            col_block: self.col_block,
            starts: self.starts.clone(),
            columns: self.columns.clone(),
            values: self
                .values
                .iter()
                .map(|&v| U::from_f64(v.to_f64()))
                .collect(),
        }
    }
}

impl BlockMatrix<f32> {
    /// The `starts` buffer's bytes.
    pub fn starts_le_bytes(&self) -> Vec<u8> {
        u32_le_bytes(&self.starts)
    }

    /// The `columns` buffer's bytes.
    pub fn columns_le_bytes(&self) -> Vec<u8> {
        u32_le_bytes(&self.columns)
    }

    /// The `values` buffer's bytes, the largest of the three: compare it with the device's
    /// storage binding limit.
    pub fn values_le_bytes(&self) -> Vec<u8> {
        f32_le_bytes(&self.values)
    }

    /// Checks that each of the matrix's buffers fits one storage binding of `limit` bytes
    /// ([`GUARANTEED_BINDING_BYTES`] unless the host requested more). The kernels index each
    /// buffer from zero, so a matrix that does not fit cannot run; the error says how many block
    /// rows (nodes, on the finest level) would.
    pub fn check_binding_size(&self, limit: u64) -> Result<(), GpuError> {
        let buffers = [self.starts.len(), self.columns.len(), self.values.len()];
        // Four bytes per u32 or f32; a usize length times four fits u64 on every target.
        let bytes = buffers.iter().map(|&n| 4 * n as u64).max().unwrap_or(0);
        if bytes <= limit {
            return Ok(());
        }
        // The rows that fit at this matrix's mean bytes per row, rounded down; u128 cannot
        // overflow for a u64 limit times a usize row count.
        let fitting = u128::from(limit) * self.rows as u128 / u128::from(bytes);
        Err(GpuError::BindingTooLarge {
            bytes,
            limit,
            rows: self.rows,
            // Fits usize: below `self.rows`, since `limit < bytes`.
            rows_that_fit: fitting as usize,
        })
    }
}

/// The storage binding size every WebGPU device supports (`maxStorageBufferBindingSize`), bytes.
pub const GUARANTEED_BINDING_BYTES: u64 = 134_217_728;

/// Packs `u32`s little-endian.
pub fn u32_le_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Packs `f32`s little-endian, the layout of every vector buffer.
pub fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Reads little-endian `f32`s, the inverse of [`f32_le_bytes`]; a trailing partial value is
/// ignored.
pub fn f32_from_le_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// The uniform of one dispatch, [`PARAMS_STRIDE`] bytes, the WGSL struct `Params`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuParams {
    /// Block rows of the matrix.
    pub rows: u32,
    /// Rows of one block.
    pub row_block: u32,
    /// Columns of one block.
    pub col_block: u32,
    /// `multiply_main` writes `A x` at 0 and subtracts it from the output at 1.
    pub subtract: u32,
    /// The Chebyshev centre `theta`, read by `chebyshev_start_main`.
    pub theta: f32,
    /// The weight of the previous direction in `chebyshev_update_main`.
    pub direction_scale: f32,
    /// The weight of the scaled residual in `chebyshev_update_main`.
    pub scaled_scale: f32,
    /// Padding to a 16-byte multiple, zero.
    pub pad: u32,
}

impl GpuParams {
    /// A record for `matrix` with the smoother fields zero.
    pub fn for_matrix(matrix: &BlockMatrix<f32>, subtract: bool) -> Self {
        GpuParams {
            // Fit in u32: `from_parts` refuses a matrix with 2^32 scalar rows.
            rows: matrix.rows() as u32,
            row_block: matrix.row_block() as u32,
            col_block: matrix.col_block() as u32,
            subtract: u32::from(subtract),
            theta: 0.0,
            direction_scale: 0.0,
            scaled_scale: 0.0,
            pad: 0,
        }
    }

    /// The record's bytes, in field order.
    pub fn to_le_bytes(&self) -> [u8; PARAMS_STRIDE] {
        let mut bytes = [0u8; PARAMS_STRIDE];
        let words = [
            self.rows.to_le_bytes(),
            self.row_block.to_le_bytes(),
            self.col_block.to_le_bytes(),
            self.subtract.to_le_bytes(),
            self.theta.to_le_bytes(),
            self.direction_scale.to_le_bytes(),
            self.scaled_scale.to_le_bytes(),
            self.pad.to_le_bytes(),
        ];
        for (at, word) in words.iter().enumerate() {
            bytes[4 * at..4 * at + 4].copy_from_slice(word);
        }
        bytes
    }
}

/// The scalars of a Chebyshev smoother of `degree` targeting `[lambda_min, lambda_max]` of the
/// block-Jacobi preconditioned operator, computed in `f64` in the order the CPU cycle uses.
#[derive(Debug, Clone, PartialEq)]
pub struct ChebyshevCoefficients {
    /// The centre of the interval; the first direction is the scaled right-hand side over it.
    pub theta: f64,
    /// For each later step, `(direction_scale, scaled_scale)`.
    pub steps: Vec<(f64, f64)>,
}

impl ChebyshevCoefficients {
    /// The coefficients, refusing bounds that do not define the polynomial.
    pub fn new(lambda_min: f64, lambda_max: f64, degree: usize) -> Result<Self, GpuError> {
        let valid = lambda_min.is_finite()
            && lambda_max.is_finite()
            && lambda_min > 0.0
            && lambda_min < lambda_max
            && degree >= 1;
        if !valid {
            return Err(GpuError::Spectrum {
                lambda_min,
                lambda_max,
            });
        }
        let theta = 0.5 * (lambda_max + lambda_min);
        let delta = 0.5 * (lambda_max - lambda_min);
        let sigma = theta / delta;
        let mut rho = 1.0 / sigma;
        let mut steps = Vec::with_capacity(degree - 1);
        for _ in 1..degree {
            let next_rho = 1.0 / (2.0 * sigma - rho);
            steps.push((next_rho * rho, 2.0 * next_rho / delta));
            rho = next_rho;
        }
        Ok(ChebyshevCoefficients { theta, steps })
    }

    /// The uniform of the start dispatch for `matrix`.
    pub fn start_params(&self, matrix: &BlockMatrix<f32>) -> GpuParams {
        GpuParams {
            theta: f32::from_f64(self.theta),
            ..GpuParams::for_matrix(matrix, true)
        }
    }

    /// The uniform of step `step` (from zero) for `matrix`: its multiply subtracts and its update
    /// reads the two scales. `None` past the last step.
    pub fn step_params(&self, matrix: &BlockMatrix<f32>, step: usize) -> Option<GpuParams> {
        let &(direction_scale, scaled_scale) = self.steps.get(step)?;
        Some(GpuParams {
            theta: f32::from_f64(self.theta),
            direction_scale: f32::from_f64(direction_scale),
            scaled_scale: f32::from_f64(scaled_scale),
            ..GpuParams::for_matrix(matrix, true)
        })
    }
}
