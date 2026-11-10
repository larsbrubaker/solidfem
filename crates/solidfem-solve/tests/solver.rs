// The solver area's test binary: the resumable conjugate gradient solve, the multigrid, and the
// GPU smoother kernels.

#[path = "solver/gpu_execution.rs"]
mod gpu_execution;
#[path = "solver/gpu_kernels.rs"]
mod gpu_kernels;
#[path = "solver/multigrid.rs"]
mod multigrid;
#[path = "solver/pcg.rs"]
mod pcg;
