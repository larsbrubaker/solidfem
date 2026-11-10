// The solver area's test binary: the resumable conjugate gradient solve, the multigrid, the GPU
// kernels of a multigrid V-cycle level, whole V-cycles on a GPU and a `Pcg` solve preconditioned
// by one.

#[path = "solver/gpu_cycle_execution.rs"]
mod gpu_cycle_execution;
#[path = "solver/gpu_execution.rs"]
mod gpu_execution;
#[path = "solver/gpu_hierarchy_execution.rs"]
mod gpu_hierarchy_execution;
#[path = "solver/gpu_kernels.rs"]
mod gpu_kernels;
#[path = "solver/gpu_solve_execution.rs"]
mod gpu_solve_execution;
#[path = "solver/gpu_transfer.rs"]
mod gpu_transfer;
#[path = "solver/multigrid.rs"]
mod multigrid;
#[path = "solver/pcg.rs"]
mod pcg;
