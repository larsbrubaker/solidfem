// The Three.js viewer the page draws the study in: z up (the engine's convention), orbit controls,
// flat-shaded stress colours from src/colormap.ts, the bending animation from src/animation.ts,
// and the markings that say what is happening to the part: a hatched plate at each fixed end and
// an arrow (push) or curved arrow (twist) at the load. The part stays on screen across a restart;
// the page calls `showPart` when it has something new to draw.
//
// Vertex positions change every frame, so they are computed on the CPU and normals come from
// flatShading's screen-space derivatives, with no normal recomputation. The mesh edges share the
// part's position buffer, so they follow the animation at no cost.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { animationPhase, displacePositions } from './animation.ts';
import { fillColors } from './colormap.ts';
import type { Load, Restraint } from './catalog.ts';

/** What the viewer draws for one result. */
export interface PartView {
  positions: Float32Array;
  indices: Uint32Array;
  displacements: Float32Array;
  /** Empty until the first stress estimate exists; the part is then drawn neutral grey. */
  vonMises: Float32Array;
  rampLow: number;
  rampHigh: number;
  scale: number;
}

/** The markings that belong to the part being shown. */
export interface PartSetup {
  length: number;
  width: number;
  height: number;
  restraint: Restraint;
  load: Load;
  loadLabel: string;
}

const COLORS = { neutral: 0x9aa6b8, plate: 0x14213d, hatch: 0xf6f7f9, arrow: 0xd1495b, edge: 0x14213d };

/** A text label that always faces the camera. */
function labelSprite(text: string, color: string): THREE.Sprite {
  const canvas = document.createElement('canvas');
  const height = 64;
  const font = `600 ${height * 0.62}px -apple-system, "Segoe UI", Roboto, sans-serif`;
  const context = canvas.getContext('2d')!;
  context.font = font;
  canvas.width = Math.ceil(context.measureText(text).width) + 28;
  canvas.height = height;
  context.font = font;
  context.fillStyle = 'rgba(255, 255, 255, 0.92)';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = color;
  context.textBaseline = 'middle';
  context.fillText(text, 14, height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }));
  sprite.renderOrder = 20;
  sprite.userData.aspect = canvas.width / canvas.height;
  return sprite;
}

function disposeTree(root: THREE.Object3D) {
  root.traverse(object => {
    const mesh = object as THREE.Mesh;
    mesh.geometry?.dispose();
    const material = mesh.material as THREE.Material | undefined;
    if (material) {
      const map = (material as THREE.SpriteMaterial).map;
      map?.dispose();
      material.dispose();
    }
  });
}

/** The unique edges of a triangle list, as a line-segment index. */
function edgeIndices(indices: Uint32Array, vertexCount: number): Uint32Array {
  const seen = new Set<number>();
  const edges: number[] = [];
  const add = (a: number, b: number) => {
    const key = Math.min(a, b) * vertexCount + Math.max(a, b);
    if (!seen.has(key)) {
      seen.add(key);
      edges.push(a, b);
    }
  };
  for (let i = 0; i + 2 < indices.length; i += 3) {
    add(indices[i], indices[i + 1]);
    add(indices[i + 1], indices[i + 2]);
    add(indices[i + 2], indices[i]);
  }
  return new Uint32Array(edges);
}

export class Viewer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private markings = new THREE.Group();
  private loadMarker: THREE.Group | null = null;
  private loadVertex = -1;
  private mesh: THREE.Mesh | null = null;
  private lines: THREE.LineSegments | null = null;
  private current: PartView | null = null;
  private drawn = new Float32Array(0);
  private colors = new Float32Array(0);
  private framedFor = '';
  private markingsFor = '';
  private animId = 0;
  private resize: ResizeObserver;
  private start = performance.now();
  /** Fully bent while paused, so the shape can be studied. */
  paused = false;

  constructor(private container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0xf4f6f9);
    container.prepend(this.renderer.domElement);

    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    this.camera = new THREE.PerspectiveCamera(36, 1, 1, 5000);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 1.5));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(40, -60, 100);
    this.scene.add(key);
    this.scene.add(this.markings);

    this.resize = new ResizeObserver(() => this.fit());
    this.resize.observe(container);
    this.fit();
    this.animate();
  }

  private fit() {
    const w = Math.max(this.container.clientWidth, 1);
    const h = Math.max(this.container.clientHeight, 1);
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private animate = () => {
    this.animId = requestAnimationFrame(this.animate);
    this.update();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  private update() {
    const view = this.current;
    if (!view || !this.mesh) return;
    const phase = this.paused ? 1 : animationPhase(performance.now() - this.start);
    displacePositions(view.positions, view.displacements, view.scale, phase, this.drawn);
    this.mesh.geometry.getAttribute('position').needsUpdate = true;
    if (this.loadMarker && this.loadVertex >= 0) {
      const i = this.loadVertex * 3;
      this.loadMarker.position.set(
        view.displacements[i] * view.scale * phase,
        view.displacements[i + 1] * view.scale * phase,
        view.displacements[i + 2] * view.scale * phase,
      );
    }
  }

  /** Shows a result, replacing the shown part only when its mesh changed. */
  showPart(view: PartView, setup: PartSetup) {
    const sameMesh = this.current !== null && this.current.indices.length === view.indices.length
      && this.current.positions.length === view.positions.length && this.sameData(this.current.positions, view.positions)
      && this.sameData(this.current.indices, view.indices);
    if (!sameMesh) this.buildMesh(view);
    const colors = this.mesh!.geometry.getAttribute('color') as THREE.BufferAttribute;
    if (view.vonMises.length > 0) {
      fillColors(view.vonMises, view.rampLow, view.rampHigh, this.colors);
      this.markColored(true);
    } else {
      this.markColored(false);
    }
    colors.needsUpdate = true;
    this.current = view;
    this.placeMarkings(view, setup);
    this.frame(setup);
  }

  private sameData(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }

  private buildMesh(view: PartView) {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      disposeTree(this.mesh);
      this.mesh = null;
    }
    if (this.lines) {
      this.scene.remove(this.lines);
      disposeTree(this.lines);
      this.lines = null;
    }
    this.drawn = new Float32Array(view.positions);
    this.colors = new Float32Array(view.positions.length);
    const geometry = new THREE.BufferGeometry();
    const position = new THREE.BufferAttribute(this.drawn, 3);
    position.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('position', position);
    geometry.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));
    geometry.setIndex(new THREE.BufferAttribute(view.indices, 1));
    const material = new THREE.MeshStandardMaterial({
      vertexColors: false, color: COLORS.neutral, flatShading: true, roughness: 0.6, metalness: 0.05, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });
    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);

    const edges = new THREE.BufferGeometry();
    edges.setAttribute('position', position);
    edges.setIndex(new THREE.BufferAttribute(edgeIndices(view.indices, view.positions.length / 3), 1));
    this.lines = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({
      color: COLORS.edge, transparent: true, opacity: 0.18,
    }));
    this.lines.frustumCulled = false;
    this.scene.add(this.lines);
  }

  private markColored(colored: boolean) {
    const material = this.mesh!.material as THREE.MeshStandardMaterial;
    if (material.vertexColors === colored) return;
    material.vertexColors = colored;
    material.color.set(colored ? 0xffffff : COLORS.neutral);
    material.needsUpdate = true;
  }

  /** Draws the fixed-end plates and the load arrow for the current choice. */
  private placeMarkings(view: PartView, setup: PartSetup) {
    const { length, width, height, restraint, load, loadLabel } = setup;
    const key = `${length}|${width}|${height}|${restraint}|${load}|${loadLabel}|${view.positions.length}`;
    if (key === this.markingsFor) return;
    this.markingsFor = key;
    disposeTree(this.markings);
    this.markings.clear();
    const side = Math.max(width, height);
    this.addPlate(0, width, height, side);
    if (restraint === 'both_ends') this.addPlate(length, width, height, side);

    const loadX = restraint === 'both_ends' ? 0.5 * length : length;
    this.loadMarker = new THREE.Group();
    if (load === 'push') {
      this.addPushArrow(this.loadMarker, loadX, width, height, side, loadLabel);
    } else {
      this.addTwistArrow(this.loadMarker, loadX, width, height, side, loadLabel);
    }
    this.markings.add(this.loadMarker);

    // A push moves with the part; a twist turns about the axis, which stays put.
    this.loadVertex = load === 'push' ? this.nearestVertex(view.positions, loadX, 0.5 * width, height) : -1;
    if (load !== 'push') this.loadMarker.position.set(0, 0, 0);
  }

  private nearestVertex(positions: Float32Array, x: number, y: number, z: number): number {
    let best = -1;
    let bestDistance = Infinity;
    for (let i = 0; i + 2 < positions.length; i += 3) {
      const d = (positions[i] - x) ** 2 + (positions[i + 1] - y) ** 2 + (positions[i + 2] - z) ** 2;
      if (d < bestDistance) {
        bestDistance = d;
        best = i / 3;
      }
    }
    return best;
  }

  /** A dark plate behind the bar's end, with hatching and a "Fixed" label. */
  private addPlate(x: number, width: number, height: number, side: number) {
    const thickness = side * 0.18;
    const span = side * 1.9;
    const centreY = 0.5 * width;
    const centreZ = 0.5 * height;
    const outward = x === 0 ? -1 : 1;
    const plate = new THREE.Mesh(
      new THREE.BoxGeometry(thickness, span, span),
      new THREE.MeshStandardMaterial({ color: COLORS.plate, roughness: 0.85 }),
    );
    plate.position.set(x + outward * 0.5 * thickness, centreY, centreZ);
    this.markings.add(plate);

    // Hatch lines on both large faces, the drawing convention for a fixed support: from the
    // opening view the outer face of the near-end plate faces away, so the inner one must carry
    // it too. Each 45-degree line z = y - t is clipped to the plate's square.
    const lines: number[] = [];
    const half = 0.5 * span;
    const faces = [x + outward * (thickness + side * 0.02), x - outward * side * 0.02];
    for (const face of faces) {
      for (let t = -2 * half + span / 7; t < 2 * half; t += span / 7) {
        const low = Math.max(-half, t - half);
        const high = Math.min(half, t + half);
        if (low >= high) continue;
        lines.push(face, centreY + low, centreZ + low - t, face, centreY + high, centreZ + high - t);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(lines), 3));
    this.markings.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: COLORS.hatch })));

    const label = labelSprite('Fixed', '#14213d');
    const labelHeight = side * 0.62;
    label.scale.set(labelHeight * (label.userData.aspect as number), labelHeight, 1);
    label.position.set(x + outward * (thickness + side * 0.45), centreY, centreZ + half + side * 0.45);
    this.markings.add(label);
  }

  private addPushArrow(group: THREE.Group, x: number, width: number, height: number, side: number, text: string) {
    const material = new THREE.MeshStandardMaterial({ color: COLORS.arrow, roughness: 0.5 });
    const shaftLength = side * 2.4;
    const radius = side * 0.13;
    const headLength = side * 0.75;
    const top = height + side * 0.12;
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, shaftLength, 16), material);
    shaft.rotation.x = Math.PI / 2;
    shaft.position.set(x, 0.5 * width, top + headLength + 0.5 * shaftLength);
    group.add(shaft);
    const head = new THREE.Mesh(new THREE.ConeGeometry(radius * 2.7, headLength, 20), material);
    head.rotation.x = -Math.PI / 2;
    head.position.set(x, 0.5 * width, top + 0.5 * headLength);
    group.add(head);
    const label = labelSprite(text, '#d1495b');
    const labelHeight = side * 0.62;
    label.scale.set(labelHeight * (label.userData.aspect as number), labelHeight, 1);
    label.position.set(x, 0.5 * width, top + headLength + shaftLength + side * 0.5);
    group.add(label);
  }

  /** A curved arrow about the bar's axis, turning in the torque's positive sense (y towards z). */
  private addTwistArrow(group: THREE.Group, x: number, width: number, height: number, side: number, text: string) {
    const material = new THREE.MeshStandardMaterial({ color: COLORS.arrow, roughness: 0.5 });
    const centreY = 0.5 * width;
    const centreZ = 0.5 * height;
    const radius = side * 1.05;
    const arcStart = -0.2 * Math.PI;
    const arcEnd = 1.25 * Math.PI;
    const points: THREE.Vector3[] = [];
    for (let i = 0; i <= 40; i++) {
      const angle = arcStart + (arcEnd - arcStart) * (i / 40);
      points.push(new THREE.Vector3(x, centreY + radius * Math.cos(angle), centreZ + radius * Math.sin(angle)));
    }
    const curve = new THREE.CatmullRomCurve3(points);
    group.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 60, side * 0.09, 10, false), material));
    const tip = points[points.length - 1];
    const before = points[points.length - 2];
    const headLength = side * 0.55;
    const head = new THREE.Mesh(new THREE.ConeGeometry(side * 0.26, headLength, 18), material);
    const direction = tip.clone().sub(before).normalize();
    head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    head.position.copy(tip).addScaledVector(direction, 0.5 * headLength);
    group.add(head);
    const label = labelSprite(text, '#d1495b');
    const labelHeight = side * 0.62;
    label.scale.set(labelHeight * (label.userData.aspect as number), labelHeight, 1);
    label.position.set(x, centreY, centreZ + radius + side * 0.7);
    group.add(label);
  }

  /** Aims the camera at the part, once per part size, leaving later orbiting alone. */
  private frame(setup: PartSetup) {
    const key = `${setup.length}|${setup.width}|${setup.height}|${setup.restraint}`;
    if (key === this.framedFor) return;
    const firstFrame = this.framedFor === '';
    this.framedFor = key;
    const centre = new THREE.Vector3(0.5 * setup.length, 0.5 * setup.width, 0.5 * setup.height);
    const radius = 0.5 * Math.hypot(setup.length, setup.width, setup.height) + Math.max(setup.width, setup.height) * 2;
    const fov = (this.camera.fov * Math.PI) / 180;
    const fitHeight = radius / Math.sin(fov / 2);
    const fitWidth = fitHeight / Math.min(this.camera.aspect, 1);
    const dist = Math.max(fitHeight, fitWidth) * 1.02;
    this.controls.target.copy(centre);
    if (firstFrame) {
      this.camera.position.copy(centre).addScaledVector(new THREE.Vector3(0.3, -0.85, 0.45).normalize(), dist);
    } else {
      // Keep the visitor's viewing direction; only the distance follows the new size.
      const direction = this.camera.position.clone().sub(centre).normalize();
      this.camera.position.copy(centre).addScaledVector(direction, dist);
    }
    this.camera.near = dist / 100;
    this.camera.far = dist * 10;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  dispose() {
    cancelAnimationFrame(this.animId);
    this.resize.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
  }
}
