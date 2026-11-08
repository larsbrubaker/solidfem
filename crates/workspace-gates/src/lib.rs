//! workspace-gates: a test-only crate. Its gates live in `tests/`: `file_compliance.rs` (the
//! 800-non-empty-line limit, header comments and merge-conflict markers across the whole
//! repository), `markdown_placement.rs` (Markdown only where it isn't documentation of code;
//! both walk the repository through `tests/support/repo_walk.rs`), `workspace_members.rs` (every
//! crate under `crates/` is a member and takes the workspace lints), `dependencies.rs` (every
//! dependency is permissively licensed and compiles no native code) and `notices.rs` (every
//! published crate ships the root LICENSE and its own section of the root
//! THIRD_PARTY_NOTICES.md, unchanged).
//! Each gate walks the workspace rather than naming crates, so a crate that moves in is covered
//! at once, and each gate's failure message teaches the rule it holds. There is deliberately no
//! library code here.
