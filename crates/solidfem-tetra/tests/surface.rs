// Surface recovery's area test binary. `check` reads only the public result and decides with the
// exact predicates or plain arithmetic; `shapes` builds the closed surfaces the tests mesh.

#[path = "surface/cases.rs"]
mod cases;
#[path = "surface/check.rs"]
mod check;
#[path = "surface/errors.rs"]
mod errors;
#[path = "surface/resumable.rs"]
mod resumable;
#[path = "surface/shapes.rs"]
mod shapes;
