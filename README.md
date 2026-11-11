# solidfem

[![License: BSD-2-Clause](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://larsbrubaker.github.io/solidfem/)

**Linear-static structural analysis on tetrahedral meshes, in pure Rust, for native and
`wasm32-unknown-unknown` hosts.**

[![Live WASM demo](README_HERO.jpg)](https://larsbrubaker.github.io/solidfem/)

<p align="center"><a href="https://larsbrubaker.github.io/solidfem/"><b>Try the live demo</b></a></p>

solidfem is developed alongside MatterCAD, in a repository of its own so that others can use it
and check it, and it depends on no CAD or GUI code. It takes a tetrahedral mesh, materials, supports
and loads, and solves for displacement and stress with 10-node elements and a conjugate gradient
solver that runs in bounded slices, so a host with no threads can solve between frames and stop
when asked. It contains no C or C++, and two runs on the same input give the same bits.

> Part of the [rust-apps](https://github.com/larsbrubaker/rust-apps) suite, a collection of Rust
> graphics and geometry libraries by Lars Brubaker.

## Status

A wrong "this part is fine" is the worst failure this library can have, so here is exactly where
it stands.

**Works today**

- Solving a tetrahedral mesh you give it: 10-node elements, materials evaluated per integration
  point, fixed nodes, forces and tractions, von Mises results. The validation tests compare it
  with closed-form answers for tension, torsion and cantilevers, a two-material bar, and a printed
  part modelled as a solid shell around a weaker infill core.
- Models that cannot be solved are refused with a specific error, before solving where that is
  possible.
- Solves are resumable and cancellable, and slicing a solve differently does not change its answer.
- An estimate of how converged a single-material result is: a per-element error indicator and the
  estimated relative error in the energy norm, from the difference between the smoothed and the
  element stress. It is an energy-norm measure of the whole part, not the accuracy of a peak
  stress: a hole's peak can read 14% low while the estimate says under 3%. Measured against the
  true energy error, it over-read by 1.5 to 1.75 times on a smooth field held at every boundary
  node, and under-read on clamped parts under load, to about 0.7 times, most of its error sitting
  in the clamp's elements. It falls as the mesh refines, except under a point load, whose exact
  energy is infinite. Nothing drives refinement from it yet, and a part of more than one material
  is refused.
- Comparing two solves of one part on a coarser and a finer mesh: how much the peak stress, its
  position and the safety factor moved between them, and whether the peak has settled at the
  host's tolerance, is still moving, or was read on meshes too alike (elements under 4 times
  smaller where the peak is) to tell. It shows whether the number is still moving, not how far it
  is from the true stress. At a clamp's edge or a sharp inside corner the model's stress is
  unbounded and the peak keeps rising with refinement (on a test cantilever by a measured 9.6% to
  19.7% per halving of the element size). Leaving such a spot out of the region read lets the rest
  settle, but its stress then goes unassessed, and at a sharp inside corner of a real part that
  stress is real: the model needs the fillet. Meshing twice and applying the supports and loads to
  each mesh is the host's work.
- Solid bars and thick-walled tubes, as the demo offers them, compared with the textbook on the
  page, and the real-part benchmark shapes offered as example parts, each with the benchmark's
  verdict shown on the page.
- Exact Delaunay tetrahedralization of a point set, and meshing the solid inside a closed surface
  with its shape kept exactly.

**Does not work yet**

- **Your own parts and arbitrary thin-walled parts are not ready.** The demo's controls change
  only bars and tubes, whose meshes follow the part, and its example parts are the fixed benchmark
  shapes, which may fail their benchmark; each says so.
- **The volume-mesher gate has passed on two surfaces, which is not a general claim.** That gate
  (`solidfem-gate`) runs this library's mesher into its solver on a 100 x 100 x 0.8 mm plate with
  two element layers through the wall, against beam theory, using the multigrid preconditioner.
  It passes within the 1% limit on a structured plate surface and on one real remeshed surface
  whose two skins have unaligned vertices. Requesting layers removes every interior edge the
  mesher's own rule calls through a thin wall, or stops with a named error; a sweep of plates,
  sections and tubes checks that rule against each shape's own construction. Surface edges are
  never changed, so at a rim an element can still bridge the wall, and the path from an arbitrary
  triangle mesh to a trustworthy stress result is still unproven beyond these cases.
- Multigrid is optional and off by default. Thin plates take several hundred iterations: 353 on
  a 97,200-unknown structured two-layer plate and 372 on the 606,483-unknown structured gate plate,
  where a stocky part takes tens.
- **Real-part benchmarks pass, on these shapes only.** The `solidfem-benchmarks` runner builds each
  part, meshes it and judges it against a limit set in advance. Passing: a thin strip, a plate with a
  hole, a thick ring, NAFEMS LE1 and LE10, a hook, a nominal snap-fit strain, a 1 mm hollow box
  pressed on its top (within 2% on refinement and against a structured solve), and a gear tooth,
  whose root fillet peak reads 0.2% to 1.3% from a structured solve across three densities. Both
  structured solves use this library's own elements and solver, so they check the mesher and the
  stress recovery, not the element formulation. An L-bracket's deflection is mesh-stable but has no
  accuracy reference, and its sharp corner is reported as a singularity. The ring and the gear
  tooth still contain elements flatter than half a degree. Buckling and large deflection are
  reported as out of scope.
- Only linear statics: no contact, no large deflection, no buckling, no fatigue, no plasticity. A
  stress past the material's yield says the part would yield, not how it would fail.

**Not for safety-critical use.** Material values in the demo are typical values, so its results
are estimates. Do not rely on any of this to decide whether something that can hurt someone is
safe.

## Crates

None of these is on crates.io yet.

| Crate | Folder | What it holds |
|---|---|---|
| `solidfem` | `crates/solidfem` | The facade re-exporting the others, the path from the mesher to a solvable mesh, and the physics validation tests |
| `solidfem-mesh` | `crates/solidfem-mesh` | Tetrahedral meshes from plain arrays, the upgrade to 10-node elements, boundary faces and a structured box builder |
| `solidfem-solve` | `crates/solidfem-solve` | Sparse matrices and a preconditioned conjugate gradient solver that runs in bounded slices |
| `solidfem-elastic` | `crates/solidfem-elastic` | The 10-node tetrahedron, materials, supports and loads, resumable static studies and von Mises results |
| `solidfem-tetra` | `crates/solidfem-tetra` | Exact Delaunay tetrahedralization of a point set, the solid mesher built on it and through-thickness refinement, resumable in slices |
| `solidfem-bench` (never published) | `crates/solidfem-bench` | The end-of-mesher gate, measurements of real triangle meshes, the surface remesher and sliver triangles, and the real-part benchmark runner (`solidfem-benchmarks`) with the surface generator it reads (`solidfem-surfaces`) |
| `solidfem-demo-wasm` (never published) | `demo/wasm` | The WebAssembly engine of the demo page |

## Demo

The [live demo](https://larsbrubaker.github.io/solidfem/) runs the solver in the browser: pick a
bar or a thick-walled tube, change the material and the load, and watch the stress, the bending and
the comparison with the textbook update as the solve runs on the page's main thread, in small
slices, with no worker. Under "Example parts" it also offers the real-part benchmark shapes (a thin
strip, a thick ring, a plate with a hole, L-brackets, NAFEMS LE1 and LE10, a hook, a snap-fit, a
hollow box and a gear tooth): each is meshed and solved in the browser at its benchmark's coarsest
mesh or, for the plate, the hollow box and the gear, a coarser demo-only one whose own error is
stated, with where it is held and loaded marked on
the part and a status line stating the benchmark's verdict, pass or fail.

It lives in `demo/`: `demo/wasm` is the engine, the page is TypeScript built with
[Bun](https://bun.sh), and `.github/workflows/deploy-demo.yml` publishes it to GitHub Pages on
every push to `main`. To run it locally you need Bun and `wasm-pack`; from `demo/`:

```sh
bun install
bun run build:wasm     # the WebAssembly engine, into public/pkg
bun run dev            # http://localhost:3040, rebuilding the wasm when Rust sources change
```

`bun run build` writes the static site to `demo/dist`. `bun run preview` serves it on the same
port, and `bun run preview:pages` serves it under `/solidfem/`, as GitHub Pages does.
`bun test` runs the page's tests and `bun run typecheck` the TypeScript check. Three.js is loaded
from a CDN by `demo/index.html`, so the page needs network access.

## Building and testing

The workspace reaches [solidmesh](https://github.com/larsbrubaker/solidmesh) by path, for the
never-published `solidfem-bench` measurements, and solidmesh reaches three more libraries the same
way, so check them out beside this repository (CI checks solidmesh out at the commit
`.github/sibling-pins` names; moving that pin means updating its SHA and `Cargo.lock` in the same
commit):

```
<any folder>/
  solidfem/        github.com/larsbrubaker/solidfem
  solidmesh/       github.com/larsbrubaker/solidmesh
  manifold-rust/   github.com/larsbrubaker/manifold-rust
  clipper2-rust/   github.com/larsbrubaker/clipper2-rust
  tess2-rust/      github.com/larsbrubaker/tess2-rust
```

`rust-toolchain.toml` pins the Rust version (rustup installs it on first use). From `solidfem/`:

```
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace     # or: cargo test --workspace
cargo check --target wasm32-unknown-unknown --workspace --exclude workspace-gates
```

The tests include the repository gates (file headers and size, Markdown placement, permissive
licenses and pure-Rust dependencies, third-party notices).

## License

BSD-2-Clause ([LICENSE](LICENSE)). Third-party code is listed, by the crate that carries it, in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
