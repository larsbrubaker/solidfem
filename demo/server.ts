// Local server for the demo page. With no arguments (the way dev.ts starts it, or `bun run
// server.ts`) it serves index.html, transpiles src/*.ts per request and serves styles/ and
// public/ (the wasm package); with --dist (`bun run preview`) it serves the bundle `bun run build`
// wrote to dist/; with --base=/solidfem/ as well (`bun run preview:pages`) it serves that bundle
// the way GitHub Pages does, under the subpath and with no fallback for a missing file. site.ts
// holds what each URL returns.
//
// Port: $PORT, else 3040 (manifold-rust's demo keeps 3000 and solidmesh's 3030, so all can run
// side by side). It listens on localhost only.

import { join } from "path";
import { answer, type SiteOptions } from "./site.ts";

const PORT = parseInt(process.env.PORT || "3040");
const SERVE_DIST = process.argv.includes("--dist");
const base = process.argv.find(arg => arg.startsWith("--base="))?.slice("--base=".length);

// A shell can rewrite the argument before it arrives (Git Bash turns /solidfem/ into a Windows
// path); serving nothing but 404s would hide that.
if (base !== undefined && !/^\/[^:]*\/$/.test(base)) {
  console.error(`--base must be a URL path with both slashes, such as /solidfem/ (got ${base})`);
  process.exit(1);
}

const options: SiteOptions = {
  root: SERVE_DIST ? join(import.meta.dir, "dist") : import.meta.dir,
  serveDist: SERVE_DIST,
  base,
};

const server = Bun.serve({
  port: PORT,
  hostname: "localhost",
  fetch: req => answer(options, req),
});

console.log(
  `demo running at http://localhost:${server.port}${base ?? ""} (${SERVE_DIST ? "dist/" : "source"})`,
);
