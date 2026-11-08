// Development entry (`bun run dev`): starts server.ts and rebuilds the wasm package with wasm-pack
// whenever a Rust source of the demo bindings (demo/wasm/src) or of the solver crates
// (crates/*/src) changes, so a reload shows the change. The TypeScript needs no rebuild: the
// server transpiles src/*.ts per request.

import { spawn } from "bun";
import { watch } from "fs";
import { join, resolve } from "path";

const ROOT = import.meta.dir;
const REPO_ROOT = resolve(ROOT, "..");

const WATCH_DIRS = [
  join(ROOT, "wasm", "src"),
  join(REPO_ROOT, "crates", "solidfem-mesh", "src"),
  join(REPO_ROOT, "crates", "solidfem-solve", "src"),
  join(REPO_ROOT, "crates", "solidfem-elastic", "src"),
  join(REPO_ROOT, "crates", "solidfem", "src"),
];

let building = false;
let pendingBuild = false;

async function buildWasm() {
  if (building) {
    pendingBuild = true;
    return;
  }
  building = true;
  console.log("\n[dev] Rust source changed: rebuilding the wasm package...");
  const start = performance.now();
  const proc = spawn({
    cmd: ["wasm-pack", "build", join(ROOT, "wasm"), "--target", "web", "--out-dir",
      join(ROOT, "public", "pkg"), "--no-typescript"],
    cwd: REPO_ROOT,
    stdout: "inherit",
    stderr: "inherit",
  });
  const exitCode = await proc.exited;
  const elapsed = ((performance.now() - start) / 1000).toFixed(1);
  if (exitCode === 0) console.log(`[dev] wasm rebuild succeeded (${elapsed}s)`);
  else console.error(`[dev] wasm rebuild failed (exit code ${exitCode})`);
  building = false;
  if (pendingBuild) {
    pendingBuild = false;
    buildWasm();
  }
}

let debounceTimer: ReturnType<typeof setTimeout> | null = null;

for (const dir of WATCH_DIRS) {
  try {
    watch(dir, { recursive: true }, (_event, filename) => {
      const name = filename as string | null;
      if (name && !name.endsWith(".rs")) return;
      if (debounceTimer) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => buildWasm(), 300);
    });
    console.log(`[dev] Watching ${dir}`);
  } catch (e: any) {
    console.warn(`[dev] Could not watch ${dir}: ${e.message}`);
  }
}

console.log("[dev] Starting server...");
await import("./server.ts");
