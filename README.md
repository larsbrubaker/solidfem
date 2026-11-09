# solidfem

Linear-static structural analysis on tetrahedral meshes, in pure Rust, for native and
`wasm32-unknown-unknown` hosts.

- `solidfem-mesh`: tetrahedral meshes from plain arrays, the upgrade to 10-node elements, boundary
  faces and a structured box builder.
- `solidfem-solve`: sparse matrices and a preconditioned conjugate gradient solver that runs in
  bounded slices, so a host with no threads can solve between frames.
- `solidfem-elastic`: the 10-node tetrahedron, materials evaluated per integration point, fixed
  nodes, forces and tractions, resumable static studies and von Mises results.
- `solidfem-tetra`: exact Delaunay tetrahedralization of a point set, and meshing the solid inside
  a closed surface while keeping its shape exactly, the core of the tetrahedral mesher, resumable
  in slices and tolerant of grids, cospherical points and thin plates.
- `solidfem`: the facade re-exporting them.
- `solidfem-bench` (never published): measures the Quartet mesher into the solver.
- `demo/wasm` (never published): the WebAssembly engine of the public demo page, where a visitor
  changes a part and sees its stress and how close it is to the textbook.

## Demo page

`demo/` is a browser demo: pick a bar or a thick-walled tube, change the material and the load, and
watch the stress, the bending and the comparison with the textbook update as the solve runs on the
page's main thread, in small slices, with no worker. Bun and `wasm-pack` are needed.

```sh
cd demo
bun install
bun run build:wasm   # the WebAssembly engine, into public/pkg
bun run dev          # http://localhost:3040, rebuilding the wasm when Rust sources change
```

`bun run build` writes the static site to `demo/dist` and `bun run preview` serves it on the same
port. `bun test` runs the page's tests and `bun run typecheck` the TypeScript check. Three.js is
loaded from a CDN by `demo/index.html`, so the page needs network access.

BSD-2-Clause licensed; see `LICENSE`.
