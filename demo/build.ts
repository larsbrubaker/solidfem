// Static build of the demo page: bundles src/main.ts into dist/, then copies index.html, the
// styles, the wasm package (public/pkg, built first by `bun run build:wasm`) and the favicon
// beside it. The bundle loads the wasm package by URL at run time (src/main.ts), so wasm-pack's
// output is copied rather than bundled.
//
// Usage: bun run build.ts

import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "fs";
import { join } from "path";

const ROOT = import.meta.dir;
const DIST = join(ROOT, "dist");

rmSync(DIST, { recursive: true, force: true });
mkdirSync(DIST, { recursive: true });

if (!existsSync(join(ROOT, "public/pkg/solidfem_wasm.js"))) {
  console.error("public/pkg is missing: run `bun run build:wasm` (wasm-pack) first.");
  process.exit(1);
}

console.log("Bundling TypeScript...");
const result = await Bun.build({
  entrypoints: [join(ROOT, "src/main.ts")],
  outdir: DIST,
  target: "browser",
  format: "esm",
  minify: true,
  naming: "[dir]/[name].[ext]",
  // Three.js comes from the CDN through index.html's import map.
  external: ["three", "three/addons/*"],
});
if (!result.success) {
  console.error("Bundling src/main.ts failed:");
  for (const msg of result.logs) console.error(msg);
  process.exit(1);
}

console.log("Copying index.html, styles, wasm, favicon...");
const html = readFileSync(join(ROOT, "index.html"), "utf-8").replace(
  '<script type="module" src="./src/main.ts"></script>',
  '<script type="module" src="./main.js"></script>',
);
writeFileSync(join(DIST, "index.html"), html);
cpSync(join(ROOT, "styles"), join(DIST, "styles"), { recursive: true });
cpSync(join(ROOT, "public/pkg"), join(DIST, "public/pkg"), { recursive: true });
cpSync(join(ROOT, "public/favicon.svg"), join(DIST, "public/favicon.svg"));
// GitHub Pages would otherwise run Jekyll over the site.
writeFileSync(join(DIST, ".nojekyll"), "");

console.log(`Build complete -> ${DIST}`);
