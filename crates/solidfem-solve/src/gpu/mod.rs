//! The kernels of one multigrid V-cycle level for a GPU: the packed single-precision block sparse
//! layout a device reads, the WGSL compute shaders that run on it, and a CPU reference of exactly
//! the same arithmetic. The library holds no GPU dependency: a host uploads the packed bytes with
//! whatever WebGPU binding it has (`wgpu` natively, the browser's WebGPU from WebAssembly) and runs
//! the shader constants. The intended split is the V-cycle in `f32` on the device while the
//! residual and the conjugate gradient stay `f64` on the CPU; this module holds the kernels of a
//! level, with the coarsest level's solve left to the CPU.
//!
//! **Layout** (`layout.rs`). A [`BlockMatrix`] is block compressed sparse rows of dense
//! `row_block` by `col_block` blocks (3 on the finest level, 6 on coarse ones, rectangular for a
//! prolongator): `starts` (`rows + 1` block offsets, `u32`), `columns` (one block column per
//! block, `u32`) and `values` (row-major blocks, `f32`), each its own storage buffer packed
//! little-endian with no padding. [`GpuParams`] is the one uniform record of a dispatch,
//! [`PARAMS_STRIDE`] bytes.
//!
//! **Kernels** (`shader.rs`, [`SMOOTHER_WGSL`]). `multiply_main` writes `A x` (or subtracts it
//! from the output), one thread per scalar row; `chebyshev_start_main` and
//! `chebyshev_update_main` are the block-Jacobi preconditioned Chebyshev recurrence, one thread
//! per block row. A smoother application of degree `k` is the fixed sequence start, then `k - 1`
//! times multiply-subtract and update, with no readback between dispatches: the per-step
//! coefficients are computed on the CPU in `f64` once ([`ChebyshevCoefficients`]) and live in one
//! uniform buffer per step.
//!
//! **Transfers** (`transfer_shader.rs`, [`TRANSFER_WGSL`]). `residual_main` (`b - A x`),
//! `restrict_main` (`P^T r`), `prolong_correct_main` (`x + P e`) and `add_main` complete one
//! V-cycle level: pre-smooth, residual, restrict, coarse solve, prolong and correct, residual,
//! post-smooth, add. Restriction reads the prolongator's explicitly stored transpose
//! ([`BlockMatrix::transpose`]), so every coarse value is a gather by one thread, never a scatter.
//! A host reads a hierarchy's levels with [`crate::Multigrid::level_matrix`],
//! [`crate::Multigrid::level_prolongator`] and [`crate::Multigrid::level_inverse_diagonal`]
//! ([`BlockMatrix::from_level`]); prolongators are 3x6 blocks from the finest level and 6x6 below
//! it, coarse matrices 6x6. The coarse solve stays on the CPU
//! ([`crate::Multigrid::solve_coarsest`]). [`reference::two_level_cycle`] is the dispatch sequence
//! on the CPU, and in `f32` it is the [`crate::Precision::Single`] cycle to the bit.
//!
//! **Order.** Every sum runs in a fixed order: a scalar row sums its blocks in stored order and
//! each block's columns in order, starting from zero, exactly as the CPU block product does. One
//! thread owns each sum, so results do not depend on scheduling and repeat to the bit on a given
//! device. A device may still differ from the CPU reference ([`reference`]) in the last bits:
//! WGSL lets an implementation contract `a * b + c` into a fused multiply-add, which rounds once
//! where the reference rounds twice. Both stay within the rounding bound the reference computes
//! ([`reference::smoother_rounding_bound`]), which is how tests compare them.
//!
//! **Limits.** WebGPU guarantees a storage binding of 128 MiB (`maxStorageBufferBindingSize`,
//! 134,217,728 bytes) and a buffer of 256 MiB (`maxBufferSize`, 268,435,456 bytes); adapters
//! often allow more, up to the 2^32-byte buffer ceiling, when a host requests the adapter's
//! limits, and browsers often do not. The largest buffer is `values`: 36 bytes per 3x3 block, so
//! the guaranteed binding holds 3,728,270 fine blocks, about 138,000 nodes (414,000 unknowns) at
//! the 27 blocks per row a 10-node tetrahedral mesh has. All indices are `u32`, so a matrix must
//! have fewer than 2^32 stored values and scalar rows ([`GpuError::TooLarge`]).
//! [`BlockMatrix::check_binding_size`] refuses a matrix whose buffers exceed a binding
//! ([`GpuError::BindingTooLarge`], with how many block rows would fit). Splitting a larger
//! matrix is not provided: the kernels index rows and blocks from zero. It needs either a first
//! row and a first block in [`GpuParams`] (each slice its own `starts`/`columns`/`values`
//! buffers, writing its rows of the shared output), or slices bound at offsets of one buffer,
//! which WebGPU requires to be multiples of 256 bytes (`minStorageBufferOffsetAlignment`).
//! Dispatches are two-dimensional ([`dispatch_size`]) because one dimension holds at most 65,535
//! workgroups.
//!
//! **Shown on.** The device comparison and the repeat-to-the-bit check have run on one device so
//! far: an NVIDIA GeForce RTX 5070 through Vulkan. DX12, Metal and browser WebGPU are untested.

mod layout;
pub mod reference;
mod shader;
mod transfer_shader;

use std::fmt;

pub use layout::{
    dispatch_size, f32_from_le_bytes, f32_le_bytes, u32_le_bytes, BlockMatrix,
    ChebyshevCoefficients, GpuParams, Real, GUARANTEED_BINDING_BYTES, PARAMS_STRIDE,
    WORKGROUP_SIZE,
};
pub use shader::{MAX_BLOCK, SMOOTHER_WGSL};
pub use transfer_shader::TRANSFER_WGSL;

/// Why a matrix or smoother could not be prepared for a GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuError {
    /// A block is wider than [`MAX_BLOCK`] or empty.
    BlockSize {
        /// Rows of a block.
        row_block: usize,
        /// Columns of a block.
        col_block: usize,
    },
    /// The arrays do not describe a block sparse matrix: wrong lengths, starts that decrease, or
    /// a block column outside the matrix.
    Structure {
        /// The block row where the description breaks.
        row: usize,
    },
    /// The matrix needs an index of 2^32 or more, which the `u32` device indices cannot hold.
    TooLarge {
        /// The count that does not fit.
        count: usize,
    },
    /// A buffer of the matrix is larger than one storage binding may be, and the kernels do not
    /// split a matrix across bindings.
    BindingTooLarge {
        /// Bytes of the largest buffer.
        bytes: u64,
        /// Bytes one binding may hold.
        limit: u64,
        /// Block rows of the matrix.
        rows: usize,
        /// Block rows (nodes, on the finest level) that fit at this matrix's bytes per row.
        rows_that_fit: usize,
    },
    /// A diagonal block is missing or not symmetric positive definite, so block-Jacobi cannot
    /// scale that row.
    NotPositiveDefinite {
        /// The block row.
        row: usize,
        /// The failing pivot, zero when the block is missing.
        pivot: f64,
    },
    /// The spectrum bounds cannot define a Chebyshev smoother: they must satisfy
    /// `0 < lambda_min < lambda_max`, both finite, and the degree must be at least one.
    Spectrum {
        /// The lower bound given.
        lambda_min: f64,
        /// The upper bound given.
        lambda_max: f64,
    },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::BlockSize {
                row_block,
                col_block,
            } => write!(
                f,
                "a {row_block}x{col_block} block is outside the 1 to {MAX_BLOCK} the GPU kernels \
                 handle"
            ),
            GpuError::Structure { row } => {
                write!(f, "block row {row} does not form a block sparse matrix")
            }
            GpuError::TooLarge { count } => write!(
                f,
                "{count} entries need indices beyond the GPU's 32-bit range, which the kernels \
                 cannot address: solve this matrix on the CPU"
            ),
            GpuError::BindingTooLarge {
                bytes,
                limit,
                rows,
                rows_that_fit,
            } => write!(
                f,
                "the matrix needs a {bytes}-byte GPU buffer where one binding holds {limit} \
                 bytes: about {rows_that_fit} of its {rows} block rows fit, so solve it on the \
                 CPU or request a larger binding from the device"
            ),
            GpuError::NotPositiveDefinite { row, pivot } => write!(
                f,
                "the diagonal block of block row {row} is not positive definite (pivot {pivot})"
            ),
            GpuError::Spectrum {
                lambda_min,
                lambda_max,
            } => write!(
                f,
                "the spectrum bounds {lambda_min} and {lambda_max} (and degree) cannot define a \
                 Chebyshev smoother"
            ),
        }
    }
}

impl std::error::Error for GpuError {}
