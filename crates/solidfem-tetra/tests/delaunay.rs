// The Delaunay triangulator's area test binary. Every property is checked by `support`, which uses
// exact predicates and plain geometry directly and shares nothing with the triangulator's code.

#[path = "delaunay/degenerate.rs"]
mod degenerate;
#[path = "delaunay/errors.rs"]
mod errors;
#[path = "delaunay/general.rs"]
mod general;
#[path = "delaunay/incremental.rs"]
mod incremental;
#[path = "delaunay/plate.rs"]
mod plate;
#[path = "delaunay/resumable.rs"]
mod resumable;
#[path = "delaunay/support.rs"]
mod support;
