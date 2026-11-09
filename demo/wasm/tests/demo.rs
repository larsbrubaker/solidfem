// The demo engine tests, one binary (`cargo nextest run --workspace --test demo`).

#[path = "demo/accuracy.rs"]
mod accuracy;
#[path = "demo/engine.rs"]
mod engine;
#[path = "demo/page_catalog.rs"]
mod page_catalog;
#[path = "demo/parts.rs"]
mod parts;
#[path = "demo/parts_timing.rs"]
mod parts_timing;
#[path = "demo/support.rs"]
mod support;
#[path = "demo/timing.rs"]
mod timing;
#[path = "demo/validity.rs"]
mod validity;
