// The markers on an example part: small dots spread evenly over the held region and red arrows
// spread evenly over the load, at the positions and directions the engine picks from the case's
// own holds and loads, so the page shows what was actually applied. src/viewer.ts places the group; the arrow
// tails sit just outside the part when the load presses on it and on the surface when it pulls.

import * as THREE from 'three';

const HELD_COLOR = 0x14213d;
const LOAD_COLOR = 0xd1495b;
/** Pixel size of a held dot, constant on screen so a big and a small part read the same. */
const HELD_DOT_PIXELS = 4;

/** The markers of one part: three numbers per held node, seven per arrow (tail, direction, length). */
export interface ExampleMarkerData {
  heldPoints: Float32Array;
  loadArrows: Float32Array;
}

/** The held dots and load arrows as a group, in the part's coordinates. */
export function buildExampleMarkers(markers: ExampleMarkerData): THREE.Group {
  const group = new THREE.Group();
  if (markers.heldPoints.length > 0) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(markers.heldPoints), 3));
    group.add(new THREE.Points(geometry, new THREE.PointsMaterial({
      color: HELD_COLOR, size: HELD_DOT_PIXELS, sizeAttenuation: false,
    })));
  }
  for (let i = 0; i + 6 < markers.loadArrows.length; i += 7) {
    const direction = new THREE.Vector3(markers.loadArrows[i + 3], markers.loadArrows[i + 4], markers.loadArrows[i + 5]);
    const tail = new THREE.Vector3(markers.loadArrows[i], markers.loadArrows[i + 1], markers.loadArrows[i + 2]);
    const length = markers.loadArrows[i + 6];
    group.add(new THREE.ArrowHelper(direction, tail, length, LOAD_COLOR, length * 0.35, length * 0.2));
  }
  return group;
}
