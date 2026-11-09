// What the demo's local server answers for a URL, apart from the socket: server.ts binds this to a
// port, and src/subpath.test.ts asks it directly. It serves a source tree or a built dist/; with a
// `base` it behaves as a static host mounted under that subpath, the way GitHub Pages serves the
// site at /solidfem/: requests outside the base and files that do not exist are 404, and the base
// without its trailing slash redirects to it. Only without a base do unknown paths fall back to
// index.html, so a wrong URL under a base shows as a 404 rather than a page.

import { existsSync, readFileSync, statSync } from "fs";
import { extname, join, resolve } from "path";

export interface SiteOptions {
  /** The folder served: the demo folder, or its dist/. */
  root: string;
  /** Serving a built dist/, where src/*.ts is not transpiled. */
  serveDist: boolean;
  /** The subpath the site is mounted under, with both slashes, such as `/solidfem/`. */
  base?: string;
}

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

function tryServe(
  options: SiteOptions,
  pathname: string,
): { content: string | Uint8Array; mime: string } | null {
  const { root, serveDist } = options;
  if (pathname === "/" || pathname === "/index.html") {
    return { content: readFileSync(join(root, "index.html"), "utf-8"), mime: "text/html" };
  }
  const filePath = join(root, pathname);
  // Never serve outside the served tree.
  if (!resolve(filePath).startsWith(resolve(root))) return null;
  if (!existsSync(filePath) || !statSync(filePath).isFile()) return null;
  const ext = extname(filePath);
  if (!serveDist && pathname.startsWith("/src/") && ext === ".ts") {
    const code = transpiler.transformSync(readFileSync(filePath, "utf-8"));
    return { content: code, mime: "application/javascript" };
  }
  const mime = MIME_TYPES[ext] || "application/octet-stream";
  if (BINARY_EXTS.has(ext)) return { content: new Uint8Array(readFileSync(filePath)), mime };
  return { content: readFileSync(filePath, "utf-8"), mime };
}

function fileResponse(resolved: { content: string | Uint8Array; mime: string }): Response {
  return new Response(resolved.content as BodyInit, {
    headers: { "Content-Type": resolved.mime, "Cache-Control": "no-cache" },
  });
}

/** Answers one request as the site described by `options`. */
export function answer(options: SiteOptions, req: Request): Response {
  const url = new URL(req.url);
  const { base } = options;
  if (base === undefined) {
    const resolved = tryServe(options, url.pathname);
    if (resolved) return fileResponse(resolved);
    return new Response(readFileSync(join(options.root, "index.html"), "utf-8"), {
      headers: { "Content-Type": "text/html", "Cache-Control": "no-cache" },
    });
  }
  if (url.pathname === base.slice(0, -1)) {
    return new Response(null, { status: 301, headers: { Location: `${base}${url.search}` } });
  }
  if (!url.pathname.startsWith(base)) return new Response("Not found", { status: 404 });
  const resolved = tryServe(options, `/${url.pathname.slice(base.length)}`);
  return resolved ? fileResponse(resolved) : new Response("Not found", { status: 404 });
}
