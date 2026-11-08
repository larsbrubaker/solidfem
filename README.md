# solidfem

Linear-static structural analysis on tetrahedral meshes, in pure Rust, for native and
`wasm32-unknown-unknown` hosts.

- `solidfem-mesh`: tetrahedral meshes from plain arrays, the upgrade to 10-node elements, boundary
  faces and a structured box builder.
- `solidfem-solve`: sparse matrices and a preconditioned conjugate gradient solver that runs in
  bounded slices, so a host with no threads can solve between frames.
- `solidfem-elastic`: the 10-node tetrahedron, materials evaluated per integration point, fixed
  nodes, forces and tractions, resumable static studies and von Mises results.
- `solidfem`: the facade re-exporting them.
- `solidfem-bench` (never published): measures the Quartet mesher into the solver.

BSD-2-Clause licensed; see `LICENSE`.
