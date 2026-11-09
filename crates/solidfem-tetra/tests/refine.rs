// Refinement's area test binary: every prior surface contract after refinement, the layers and
// size control measured on the output tetrahedra, determinism, cancel and the budget.
// `support` reads only public results; `shapes` and `check` are shared with the surface tests.

#[allow(dead_code)]
#[path = "surface/check.rs"]
mod check;
#[path = "refine/contracts.rs"]
mod contracts;
#[path = "refine/layers.rs"]
mod layers;
#[path = "refine/resumable.rs"]
mod resumable;
#[allow(dead_code)]
#[path = "surface/shapes.rs"]
mod shapes;
#[path = "refine/sizing.rs"]
mod sizing;
#[path = "refine/support.rs"]
mod support;
#[path = "refine/through.rs"]
mod through;
#[path = "refine/wall_skins.rs"]
mod wall_skins;
