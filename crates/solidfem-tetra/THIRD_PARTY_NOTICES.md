# Third-party notices for this crate

This is the section of the repository-root notices file that covers this crate.

# solidfem-tetra (`crates/solidfem-tetra`)

No third-party code is copied into this crate. It depends on the `robust` crate (MIT OR Apache-2.0,
https://github.com/georust/robust) for exact orientation and in-sphere predicates, and on the
`libm` crate (MIT, https://github.com/rust-lang/compiler-builtins) for the arc cosine, the same on
every system, and takes both from crates.io as ordinary dependencies.
