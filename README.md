# solidfem

Linear-static structural analysis on tetrahedral meshes, in pure Rust, for native and
`wasm32-unknown-unknown` hosts.

- `solidfem-mesh`: tetrahedral meshes from plain arrays, the upgrade to 10-node elements, boundary
  faces and a structured box builder.
- `solidfem-solve`: sparse matrices and a preconditioned conjugate gradient solver that runs in
  bounded slices, so a host with no threads can solve between frames.
- `solidfem`: the facade re-exporting them.

BSD-2-Clause licensed; see `LICENSE`.
