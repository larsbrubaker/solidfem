//! The real-part benchmarks: does this library help with real work, or only pass friendly tests?
//! Each case in `cases/` (one file per part) is a part built by a real CAD operation and remeshed by a real surface
//! remesher, then meshed by this workspace's mesher with layers through thin walls, solved through
//! 10-node elements with multigrid at two or three densities, and compared with a closed form or a
//! published reference where one exists, with limits derived in the case before it first ran. A
//! case with no accuracy reference, or one that measures only a nominal quantity, says so on its
//! verdict: a PASS claims only what its limits tested. `pipeline.rs` does the work; `verdict.rs` holds the verdicts, the convergence and singularity reading and the
//! load-balance check, free of solving so they can be tested alone. `bin/benchmarks.rs` prints the
//! results. The surfaces come from outside this repository: a run names the directory holding them.

pub mod cases;
pub mod pipeline;
pub mod verdict;
