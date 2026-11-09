//! The surface generator, against its production code: the table agrees with the benchmark cases
//! that read its files, each shape is the part it says it is, the text format holds exact
//! coordinates, and regenerating a surface gives the same bytes.

#![cfg(not(target_arch = "wasm32"))]

#[path = "surfaces/format.rs"]
mod format;
#[path = "surfaces/generation.rs"]
mod generation;
#[path = "surfaces/shapes.rs"]
mod shapes;
#[path = "surfaces/table.rs"]
mod table;
