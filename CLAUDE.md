# solidfem

Linear-static structural analysis on tetrahedral meshes in pure Rust. The results must be correct
against known physics: a wrong "this part is fine" is the worst failure this library can have.
`README.md` says what it does and the root `Cargo.toml` lists the crates; how any part works is in
the header of its crate's `lib.rs` and of each file.

## Never
- **Code that isn't Rust**: no C/C++, no `-sys` crates, no native build scripts. Every library
  crate builds for `wasm32-unknown-unknown` and depends on no GUI, wgpu or MatterCAD crate. The
  `dependencies` gate refuses the rest.
- **A license that isn't permissive** (MIT/BSD/Apache/zlib-compatible). Third-party code goes in
  `THIRD_PARTY_NOTICES.md`, under the crate that carries it, in the same change.
- **Stubs**: no `todo!()`, placeholders or partial implementations.
- **Weakening a test to make it pass.** Every failure is a real bug: find the root cause. A test
  tolerance is derived and stated in a comment, never fitted to the output.
- **A garbage answer.** A model that cannot be solved returns a specific error, before solving
  where that is possible.

## Exactness
Two runs on the same input give the same bits, however a solve is sliced.
- `f64`, in a stable operation order, never `mul_add` (`clippy.toml` refuses it).
- Where map order reaches an output, use `IndexMap`; where sort ties do, give an explicit
  tie-break.

## Working
- **Bug fixes are test-first:** a failing test, the root-cause fix, the test passing.
- **Tests exercise production code**, never copies. Tests live in `crates/<crate>/tests/<area>/`,
  one binary per area (`tests/<area>.rs`).
- **Errors are `Result`s**, never swallowed; `unwrap` stays out of library code.
- **A host with no threads solves in slices.** Nothing long runs without a way to pause between
  frames and cancel.

## Code and docs
- Every file starts with a header saying what it is and how it fits its neighbours. That header
  is the file's documentation: keep it true in the same commit.
- Comments explain why, in this code's own terms, never by citing another implementation.
- A file over 800 lines is split by responsibility, never trimmed.
- There are no design docs: Markdown lives only where `markdown_placement` allows.

## Done
`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo nextest run --workspace` and
`cargo check --target wasm32-unknown-unknown --workspace --exclude workspace-gates` are green.
Each change is one commit whose message says what a library user can now do.
