# Third-party notices

solidfem is BSD-2-Clause licensed (see `LICENSE`). Third-party code it ports or adapts is listed
below, in the section of the crate that carries it. Each crate ships its own section in
`crates/<folder>/THIRD_PARTY_NOTICES.md`; this file is the one to edit (the workspace-gates
`notices` test refuses a crate file that differs from its section here).

# solidfem-mesh (`crates/solidfem-mesh`)

No third-party code.

# solidfem-solve (`crates/solidfem-solve`)

No third-party code.

# solidfem (`crates/solidfem`)

No third-party code.

# solidfem-elastic (`crates/solidfem-elastic`)

No third-party code.

# Demo page (`demo/`)

## three.js (MIT)

Used by: the demo page, loaded at run time from a CDN by `demo/index.html` (version 0.170.0, its
`build/three.module.js` and `examples/jsm/` modules). Not copied into this repository. The page's
development-only tools (TypeScript, Apache-2.0; `@types/bun` and `@types/three`, MIT) are installed
by `bun install`, used only to type-check and test, and never shipped.

Source: https://github.com/mrdoob/three.js

```
The MIT License

Copyright (c) 2010-2024 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
