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
//! **Strength of connection.** Whether two block rows belong together is decided geometrically: a
//! stencil neighbour is strong when it lies within `neighbour_reach` times the row's closest
//! stencil neighbour, on the level's block positions (node coordinates, then aggregate
//! centroids). Thin walls are why: in a flat 10-node element nearly every pair of nodes couples
//! through the through-wall strain, so the matrix entries cannot separate through-wall neighbours
//! from in-plane ones, and any threshold on block norms either merges across the wall's width or
//! stalls on corner-midside pairs (measured on a 20,000-unknown plate: mean aggregate 2.6 nodes
//! and operator complexity 18 at one threshold, 18.5 nodes spanning the width and 1,577
//! iterations at another; on a cantilever, 2.5 nodes per aggregate and 1,620 singletons).
//! Distance needs no threshold on values: where the through-wall spacing is the closest, a row
//! keeps only through-wall neighbours (aggregates become stacks through the wall, inspected at
//! zero in-plane extent), and on a compact part the same test takes the surrounding patch. A
//! level that still stalls widens its reach and aggregates again, at most three times, before the
//! hierarchy stops there.
//!
//! The default reach is three times the closest neighbour. A real wall has unaligned skins and
//! midside nodes close to some neighbours and far from others, so a tighter reach follows that
//! irregularity row by row instead of the wall: at 2.0 a remeshed two-layer strip left 314 of
//! 2,926 aggregates under three nodes, too few to carry all six rigid-body modes, and needed
//! 1,127 iterations; at 3.0 the same strip needs 662 and operator complexity falls from 1.66 to
//! 1.27, while a 158,760-unknown stocky cantilever goes from 36 to 42 iterations in the same
//! solve time. A wider reach still costs compact parts more than it gains on plates (4.0: 530
//! iterations on the strip, 57 on the cantilever).
//!
//! **Prolongator smoothing is filtered.** Weak couplings are lumped into the diagonal instead of
//! widening the prolongator, which is what bounds the coarse stencil: without it the 20,000-unknown
//! plate's first coarse level stored 8.6 times the finest level's values, with it 0.9 times. Only
//! the prolongator sees the filtered matrix; the coarse operators are Galerkin products of the true
//! matrix and the smoother uses the true matrix, so the preconditioner stays symmetric positive
//! definite.
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
//! **Precision.** Setup computes in `f64`. The finished hierarchy is stored and its V-cycle applied
//! in the [`Precision`] the options choose: `Double` is `f64` throughout; `SingleStorage` stores
//! the operators in `f32` and does the cycle's arithmetic in `f64`; `Single` does both in `f32`.
//! Storing in `f32` halves the stored values' memory and the bandwidth a cycle reads. The
//! conjugate gradient iteration and the stiffness product its convergence is measured with stay
//! `f64` in every case, so a solve still reaches an `f64` tolerance and only the preconditioner's
//! quality depends on the rounding. Rounding the stored operators costs no iterations on the
//! cases measured; doing the cycle's vector arithmetic in `f32` costs some (66 to 86 on a stocky
//! cantilever, 84 to 90 on a plate).
//!
//! The files: `scalar.rs` holds the value type the cycle is generic over, `block.rs` the block
//! sparse storage and the small dense kernels,
//! `aggregate.rs` the strength graph and the aggregation passes, `transfer.rs` the tentative
//! prolongator, its smoothing and the Galerkin product, `setup.rs` the resumable state machine
//! over those phases, and `cycle.rs` the finished preconditioner and its V-cycle.

mod aggregate;
mod block;
mod cycle;
mod scalar;
mod setup;
mod transfer;

use std::fmt;

pub(crate) use block::invert_spd_block;
pub use cycle::Multigrid;
pub use setup::{MultigridSetup, SetupStatus};

/// The number of near-null-space vectors the multigrid carries: three translations and three
/// rotations.
pub const NEAR_NULL_MODES: usize = 6;

/// The floating-point type the finished hierarchy is stored and applied in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Stored and applied in `f64` throughout.
    #[default]
    Double,
    /// Matrices, prolongators and block-diagonal inverses stored in `f32`, each value widened to
    /// `f64` as it is read; vectors and sums stay `f64`. Half the stored values' memory and the
    /// bandwidth a cycle reads them with, and the rounding of the stored operators alone.
    SingleStorage,
    /// Stored in `f32` and the V-cycle's vectors and sums in `f32` too: the residual is rounded
    /// on the way in and the correction widened on the way out. The form a device without `f64`
    /// arithmetic runs.
    Single,
}

/// Settings for the multigrid preconditioner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultigridOptions {
    /// How far, as a multiple of a block row's closest stencil neighbour, a neighbour may be and
    /// still count as strongly connected. A row whose nearest neighbours are through a thin wall
    /// keeps only neighbours through the wall.
    pub neighbour_reach: f64,
    /// Coarsening stops once a level has at most this many values; it is then factored directly.
    pub coarse_dofs: usize,
    /// The most levels to build.
    pub max_levels: usize,
    /// The degree of the Chebyshev smoother before and after each coarse correction.
    pub smoother_degree: usize,
    /// The smoother targets `[lambda_max / smoother_range, lambda_max]`. The coarse levels must
    /// handle everything below that.
    pub smoother_range: f64,
    /// The precision of the finished hierarchy and its cycle; setup always computes in `f64`.
    pub precision: Precision,
}

impl Default for MultigridOptions {
    fn default() -> Self {
        MultigridOptions {
            neighbour_reach: 3.0,
            coarse_dofs: 1500,
            max_levels: 12,
            smoother_degree: 3,
            smoother_range: 30.0,
            precision: Precision::Double,
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
    /// The positions array does not hold one entry for every block row.
    PositionLength {
        /// Block rows of the matrix.
        expected: usize,
        /// Positions supplied.
        found: usize,
    },
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
            MultigridError::PositionLength { expected, found } => write!(
                f,
                "the multigrid has {found} block positions where the matrix needs {expected}"
            ),
            MultigridError::Cancelled => write!(f, "the multigrid setup was cancelled"),
            MultigridError::NotFinished => write!(f, "the multigrid setup has not finished"),
        }
    }
}

impl std::error::Error for MultigridError {}
