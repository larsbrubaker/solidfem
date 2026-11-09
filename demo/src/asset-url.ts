// Where the page finds its wasm package, resolved against the page's own URL so the site works at
// any depth: at a host's root in development and at https://larsbrubaker.github.io/solidfem/ when
// deployed. src/main.ts loads the module from this URL; the module finds its .wasm beside itself.

/** The URL of wasm-pack's JavaScript module, for a page at `pageUrl`. */
export function wasmModuleUrl(pageUrl: string): string {
  return new URL('./public/pkg/solidfem_wasm.js', pageUrl).href;
}

/** The URL of an example part's surface file, for a page at `pageUrl`. */
export function exampleSurfaceUrl(pageUrl: string, asset: string): string {
  return new URL(`./public/parts/${asset}`, pageUrl).href;
}
