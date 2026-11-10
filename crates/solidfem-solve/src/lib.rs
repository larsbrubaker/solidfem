//! solidfem-solve: the linear algebra under the structural solver.
//!
//! [`CsrMatrix`] is a compressed sparse row matrix built from element connectivity, assembled
//! symmetric. [`Pcg`] is a preconditioned conjugate gradient solver whose state lives on the
//! heap between [`Pcg::step`] calls, so a host with no threads can run a large solve in bounded
//! slices between frames, inspect the current estimate, and cancel. [`Preconditioner`] is the
//! seam the preconditioners sit behind: [`Jacobi`] is diagonal scaling, and [`Multigrid`] is block
//! smoothed aggregation algebraic multigrid over the six rigid-body modes, built in resumable
//! slices by [`MultigridSetup`] because a large setup must not freeze a browser.
//!
//! [`gpu`] holds the kernels of one V-cycle level for a GPU (smoother, residual, restriction,
//! prolongation): the packed single-precision block layout, the WGSL compute shaders, and a CPU
//! reference of the same arithmetic, with no GPU dependency.
//!
//! Slicing never changes results: a solve's floating-point operations depend only on its input
//! and iteration count, so one long run and many short `step` calls give identical bits. Neither
//! does the thread count: on native targets the multigrid runs its large row-by-row products on
//! `MultigridOptions::threads` threads (`parallel.rs` says why the bits cannot differ), and
//! `wasm32` always runs on the calling thread.

mod csr;
pub mod gpu;
mod multigrid;
mod parallel;
mod pcg;
mod preconditioner;

pub use csr::CsrMatrix;
pub use multigrid::{
    LevelMatrix, Multigrid, MultigridError, MultigridOptions, MultigridSetup, MultigridStats,
    Precision, SetupStatus, NEAR_NULL_MODES,
};
pub use pcg::{Pcg, PcgError, PcgOptions, Progress, StepOutcome};
pub use preconditioner::{Jacobi, JacobiError, Preconditioner};
