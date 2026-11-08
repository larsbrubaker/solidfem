//! solidfem-solve: the linear algebra under the structural solver.
//!
//! [`CsrMatrix`] is a compressed sparse row matrix built from element connectivity, assembled
//! symmetric. [`Pcg`] is a preconditioned conjugate gradient solver whose state lives on the
//! heap between [`Pcg::step`] calls, so a host with no threads can run a large solve in bounded
//! slices between frames, inspect the current estimate, and cancel. [`Preconditioner`] is the
//! seam a multigrid preconditioner replaces [`Jacobi`] behind.
//!
//! Slicing never changes results: a solve's floating-point operations depend only on its input
//! and iteration count, so one long run and many short `step` calls give identical bits.

mod csr;
mod pcg;
mod preconditioner;

pub use csr::CsrMatrix;
pub use pcg::{Pcg, PcgError, PcgOptions, Progress, StepOutcome};
pub use preconditioner::{Jacobi, JacobiError, Preconditioner};
