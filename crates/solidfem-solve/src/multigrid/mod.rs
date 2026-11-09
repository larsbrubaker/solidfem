//! Block smoothed aggregation algebraic multigrid, the preconditioner for large stiffness systems
//! where diagonal scaling cannot converge. The matrix is treated as 3x3 nodal blocks. Each level
//! groups nodes into aggregates, spans the six rigid-body modes on each aggregate with an
//! orthonormal tentative prolongator, smooths it once with the block-diagonal scaled matrix, and
//! forms the next level as the Galerkin product. A coarse block is 6x6, one value per mode.
//!
//! **Near-null space.** The six rigid-body modes are the near-null space the coarse levels must
//! represent; `MultigridSetup::new` refuses data of the wrong length with a named error, because
//! a wrong-length near-null space silently produces garbage. An aggregate whose nodes cannot
//! span all six modes (a line of nodes cannot see rotation about that line) leaves those coarse
//! values with a zero prolongator column: the coarse matrix then has an empty row and column
//! there, which gets a unit diagonal so it stays positive definite and those values never couple
//! to anything.
//!
//! **Strength of connection on thin plates.** A block connection `i-j` counts as strong when
//! `||A_ij||_F >= theta * sqrt(||A_ii||_F ||A_jj||_F)`, with the threshold halved each level, scaled
//! by the diagonal blocks so the measure does not change with units or element sizes. On a plate
//! whose elements are much wider than the wall is thick, through-thickness couplings dominate this
//! ratio, so a threshold high enough to reject the in-plane couplings makes level one aggregate
//! stacks through the wall, and the coarse levels then see the bending problem, where the
//! rotations of the near-null space are the bending modes. The default of 0.25 is chosen on the
//! plate: a low threshold aggregates across the plate's width and the bending modes are lost
//! (on a 20,000-unknown 100 x 100 x 0.8 mm plate, 0.08 took 1,577 iterations); 0.25 took 109, at
//! the cost of an operator complexity of 18, because semi-coarsening through the wall shrinks a
//! level far less than aggregating in-plane does.
//!
//! **Smoother.** Chebyshev polynomials in the block-Jacobi preconditioned operator, the same
//! polynomial before and after the coarse correction. It needs only an upper bound on the spectrum
//! (estimated by power iteration with a safety margin) where damped Jacobi needs a tuned damping,
//! and it damps the upper part of the spectrum more per matrix product, which matters because a
//! matrix product is most of the cost of a cycle. A polynomial in the block-Jacobi operator is
//! symmetric, so the V-cycle is a symmetric positive definite preconditioner.
//!
//! **Slices.** Setup is resumable: `MultigridSetup::step(budget, cancel)` does a bounded amount
//! of work and returns; slicing never changes the bits. The budget counts block rows of work.
//!
//! The files: `block.rs` holds the block sparse storage and the small dense kernels,
//! `aggregate.rs` the strength graph and the aggregation passes, `transfer.rs` the tentative
//! prolongator, its smoothing and the Galerkin product, `setup.rs` the resumable state machine
//! over those phases, and `cycle.rs` the finished preconditioner and its V-cycle.

mod aggregate;
mod block;
mod cycle;
mod setup;
mod transfer;

use std::fmt;

pub use cycle::Multigrid;
pub use setup::{MultigridSetup, SetupStatus};

/// The number of near-null-space vectors the multigrid carries: three translations and three
/// rotations.
pub const NEAR_NULL_MODES: usize = 6;

/// Settings for the multigrid preconditioner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultigridOptions {
    /// Strength threshold at the finest level, halved each level below it. Blocks weaker than
    /// this fraction of the diagonal norms are not aggregated together. High thresholds make thin
    /// plates semi-coarsen through the wall, trading memory and setup time for iterations.
    pub strength_threshold: f64,
    /// Coarsening stops once a level has at most this many values; it is then factored directly.
    pub coarse_dofs: usize,
    /// The most levels to build.
    pub max_levels: usize,
    /// The degree of the Chebyshev smoother before and after each coarse correction.
    pub smoother_degree: usize,
    /// The smoother targets `[lambda_max / smoother_range, lambda_max]`. The coarse levels must
    /// handle everything below that.
    pub smoother_range: f64,
}

impl Default for MultigridOptions {
    fn default() -> Self {
        MultigridOptions {
            strength_threshold: 0.25,
            coarse_dofs: 1500,
            max_levels: 12,
            smoother_degree: 3,
            smoother_range: 30.0,
        }
    }
}

/// What the finished hierarchy looks like.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultigridStats {
    /// Levels including the finest and the directly factored coarsest.
    pub levels: usize,
    /// Values at the coarsest level.
    pub coarse_dofs: usize,
    /// Stored matrix entries over all levels divided by those of the finest level.
    pub operator_complexity: f64,
    /// Bytes held by the hierarchy's matrices and transfer operators.
    pub memory_bytes: usize,
}

/// Why a multigrid could not be built.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MultigridError {
    /// The matrix is not made of 3x3 nodal blocks: its size is not a multiple of three, or a row
    /// does not group its columns into whole blocks.
    BlockStructure {
        /// The first scalar row that breaks the block layout.
        row: usize,
    },
    /// The near-null-space array does not hold six values for every matrix row.
    NearNullLength {
        /// `6 * matrix size`.
        expected: usize,
        /// The length supplied.
        found: usize,
    },
    /// A block that must be positive definite is not, so the matrix is not.
    NotPositiveDefinite {
        /// The level it was found on, zero being the finest.
        level: usize,
        /// The block row, or the coarse value for the direct factorization.
        block: usize,
        /// The pivot that failed.
        pivot: f64,
    },
    /// The cancel check stopped setup.
    Cancelled,
    /// A result was asked for before setup finished.
    NotFinished,
}

impl fmt::Display for MultigridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultigridError::BlockStructure { row } => write!(
                f,
                "matrix row {row} does not group into 3x3 nodal blocks, which the multigrid needs"
            ),
            MultigridError::NearNullLength { expected, found } => write!(
                f,
                "the rigid-body modes have {found} values where the matrix needs {expected} (six \
                 per unknown)"
            ),
            MultigridError::NotPositiveDefinite {
                level,
                block,
                pivot,
            } => write!(
                f,
                "multigrid level {level} block {block} is not positive definite (pivot {pivot}): \
                 something is free to move that should be held"
            ),
            MultigridError::Cancelled => write!(f, "the multigrid setup was cancelled"),
            MultigridError::NotFinished => write!(f, "the multigrid setup has not finished"),
        }
    }
}

impl std::error::Error for MultigridError {}
