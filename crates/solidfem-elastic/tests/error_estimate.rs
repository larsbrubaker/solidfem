// The error estimate area's test binary: the recovery-based error estimate against fields whose
// exact stress is known.

#[path = "error_estimate/convergence.rs"]
mod convergence;
#[path = "error_estimate/exact.rs"]
mod exact;
#[path = "error_estimate/loaded.rs"]
mod loaded;
#[path = "error_estimate/support.rs"]
mod support;
