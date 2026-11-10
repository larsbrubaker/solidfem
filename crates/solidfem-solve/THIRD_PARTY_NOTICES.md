# Third-party notices for this crate

This is the section of the repository-root notices file that covers this crate.

# solidfem-solve (`crates/solidfem-solve`)

No third-party code is copied into this crate. Its tests, never the library, use `naga` and `wgpu`
(MIT OR Apache-2.0, https://github.com/gfx-rs/wgpu) to validate and run the GPU kernels and
`pollster` (Apache-2.0 OR MIT) to wait on them, taken from crates.io as dev-dependencies.
