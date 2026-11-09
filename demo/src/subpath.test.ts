// The site must work when served from a subpath (https://larsbrubaker.github.io/solidfem/): every
// asset URL resolves below the page, whether written in index.html or computed for the wasm
// package. server.ts's subpath mode is held here too, since it is how the rule is checked by hand.
// The bundle checks run when `bun run build` has written dist/, as CI's demo job does first.

import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { answer, type SiteOptions } from '../site.ts';
import { exampleSurfaceUrl, wasmModuleUrl } from './asset-url.ts';

const PAGES = 'https://larsbrubaker.github.io/solidfem/';
const demo = join(import.meta.dir, '..');
const dist = join(demo, 'dist');
const built =
  existsSync(join(dist, 'index.html')) && existsSync(join(dist, 'public/pkg/solidfem_wasm.js'));

/** Every local URL an HTML page names: its `href` and `src` attributes, minus absolute ones. */
function localUrls(html: string): string[] {
  const noComments = html.replace(/<!--[\s\S]*?-->/g, '');
  return [...noComments.matchAll(/\b(?:href|src)="([^"]+)"/g)]
    .map(match => match[1])
    .filter(url => !/^[a-z]+:/i.test(url) && !url.startsWith('#'));
}

function expectStaysBelowTheSubpath(url: string) {
  expect(url.startsWith('/')).toBe(false);
  expect(new URL(url, PAGES).href.startsWith(PAGES)).toBe(true);
}

describe('asset URLs under a subpath', () => {
  test('the wasm package resolves below the subpath, wherever the page is entered', () => {
    const expected = `${PAGES}public/pkg/solidfem_wasm.js`;
    expect(wasmModuleUrl(PAGES)).toBe(expected);
    expect(wasmModuleUrl(`${PAGES}index.html`)).toBe(expected);
    expect(wasmModuleUrl(`${PAGES}?part=tube#view`)).toBe(expected);
  });

  test('the wasm package resolves at a host root too', () => {
    expect(wasmModuleUrl('http://localhost:3040/')).toBe(
      'http://localhost:3040/public/pkg/solidfem_wasm.js',
    );
  });

  test('example surfaces resolve below the subpath, wherever the page is entered', () => {
    const expected = `${PAGES}public/parts/snap_h1.0.txt`;
    expect(exampleSurfaceUrl(PAGES, 'snap_h1.0.txt')).toBe(expected);
    expect(exampleSurfaceUrl(`${PAGES}index.html`, 'snap_h1.0.txt')).toBe(expected);
    expect(exampleSurfaceUrl('http://localhost:3040/', 'snap_h1.0.txt')).toBe(
      'http://localhost:3040/public/parts/snap_h1.0.txt',
    );
  });

  test.skipIf(!built)('every example surface the catalog names is in the built site', () => {
    const catalog = JSON.parse(readFileSync(join(demo, 'src/testdata/catalog.json'), 'utf-8'));
    for (const part of catalog.examples) {
      expect(existsSync(join(dist, 'public/parts', part.asset))).toBe(true);
    }
  });

  test('the page names only URLs relative to itself', () => {
    const urls = localUrls(readFileSync(join(demo, 'index.html'), 'utf-8'));
    expect(urls.length).toBeGreaterThan(0);
    for (const url of urls) expectStaysBelowTheSubpath(url);
  });

  test('the stylesheet names no root-relative URL', () => {
    const css = readFileSync(join(demo, 'styles/main.css'), 'utf-8');
    expect(css).not.toMatch(/url\(\s*['"]?\//);
  });

  test.skipIf(!built)('the built page names only files the build wrote', () => {
    const urls = localUrls(readFileSync(join(dist, 'index.html'), 'utf-8'));
    expect(urls).toContain('./main.js');
    for (const url of urls) {
      expectStaysBelowTheSubpath(url);
      expect(existsSync(join(dist, url))).toBe(true);
    }
  });

  test.skipIf(!built)('the bundle looks for the wasm package beside the page', () => {
    const bundle = readFileSync(join(dist, 'main.js'), 'utf-8');
    expect(bundle).toContain('./public/pkg/solidfem_wasm.js');
    expect(bundle).not.toMatch(/["'`]\/public\//);
  });

  test.skipIf(!built)('the wasm module finds its binary beside itself', () => {
    const module = readFileSync(join(dist, 'public/pkg/solidfem_wasm.js'), 'utf-8');
    expect(module).toContain("new URL('solidfem_wasm_bg.wasm', import.meta.url)");
    expect(existsSync(join(dist, 'public/pkg/solidfem_wasm_bg.wasm'))).toBe(true);
  });
});

describe('the local server under a subpath', () => {
  let root: string;
  let options: SiteOptions;
  const get = (path: string) => answer(options, new Request(`http://localhost${path}`));

  beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), 'solidfem-site-'));
    writeFileSync(join(root, 'index.html'), '<title>page</title>');
    mkdirSync(join(root, 'public/pkg'), { recursive: true });
    writeFileSync(join(root, 'public/pkg/x_bg.wasm'), new Uint8Array([0, 97, 115, 109]));
    options = { root, serveDist: true, base: '/solidfem/' };
  });
  afterAll(() => rmSync(root, { recursive: true, force: true }));

  test('serves the page and its files below the subpath', async () => {
    const page = get('/solidfem/');
    expect(page.status).toBe(200);
    expect(await page.text()).toContain('<title>page</title>');
    const wasm = get('/solidfem/public/pkg/x_bg.wasm');
    expect(wasm.status).toBe(200);
    expect(wasm.headers.get('Content-Type')).toBe('application/wasm');
  });

  test('sends the subpath without its slash to the page, as GitHub Pages does', () => {
    const redirect = get('/solidfem?part=tube');
    expect(redirect.status).toBe(301);
    expect(redirect.headers.get('Location')).toBe('/solidfem/?part=tube');
  });

  test('refuses a root-relative request, so a wrong URL shows as a failure', () => {
    expect(get('/public/pkg/x_bg.wasm').status).toBe(404);
    expect(get('/').status).toBe(404);
  });

  test('has no fallback page for a missing file', () => {
    expect(get('/solidfem/public/pkg/missing.wasm').status).toBe(404);
  });
});
