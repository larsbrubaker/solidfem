// Local server for the demo page. With no arguments (the way dev.ts starts it, or `bun run
// server.ts`) it serves index.html, transpiles src/*.ts per request and serves styles/ and
// public/ (the wasm package); with --dist (`bun run preview`) it serves the bundle `bun run build`
// wrote to dist/ exactly as a static host would. Unknown paths fall back to index.html.
//
// Port: $PORT, else 3040 (manifold-rust's demo keeps 3000 and solidmesh's 3030, so all can run
// side by side). It listens on localhost only.

import { existsSync, readFileSync, statSync } from "fs";
import { extname, join, resolve } from "path";

const PORT = parseInt(process.env.PORT || "3040");
const SERVE_DIST = process.argv.includes("--dist");
const ROOT = SERVE_DIST ? join(import.meta.dir, "dist") : import.meta.dir;

const MIME_TYPES: Record<string, string> = {
  ".html": "text/html",
  ".css": "text/css",
  ".js": "application/javascript",
  ".ts": "application/javascript",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
};
const BINARY_EXTS = new Set([".wasm", ".png", ".ico"]);

const transpiler = new Bun.Transpiler({ loader: "ts", target: "browser" });

function tryServe(pathname: string): { content: string | Uint8Array; mime: string } | null {
  if (pathname === "/" || pathname === "/index.html") {
    return { content: readFileSync(join(ROOT, "index.html"), "utf-8"), mime: "text/html" };
  }
  const filePath = join(ROOT, pathname);
  // Never serve outside the served tree.
  if (!resolve(filePath).startsWith(resolve(ROOT))) return null;
  if (!existsSync(filePath) || !statSync(filePath).isFile()) return null;
  const ext = extname(filePath);
  if (!SERVE_DIST && pathname.startsWith("/src/") && ext === ".ts") {
    const code = transpiler.transformSync(readFileSync(filePath, "utf-8"));
    return { content: code, mime: "application/javascript" };
  }
  const mime = MIME_TYPES[ext] || "application/octet-stream";
  if (BINARY_EXTS.has(ext)) return { content: new Uint8Array(readFileSync(filePath)), mime };
  return { content: readFileSync(filePath, "utf-8"), mime };
}

const server = Bun.serve({
  port: PORT,
  hostname: "localhost",
  fetch(req) {
    const url = new URL(req.url);
    const resolved = tryServe(url.pathname);
    if (resolved) {
      return new Response(resolved.content as BodyInit, {
        headers: { "Content-Type": resolved.mime, "Cache-Control": "no-cache" },
      });
    }
    return new Response(readFileSync(join(ROOT, "index.html"), "utf-8"), {
      headers: { "Content-Type": "text/html", "Cache-Control": "no-cache" },
    });
  },
});

console.log(`demo running at http://localhost:${server.port} (${SERVE_DIST ? "dist/" : "source"})`);
