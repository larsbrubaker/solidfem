// Independent checks of a tetrahedralization: everything here reads only the public result and
// decides with the exact predicates directly or plain arithmetic, so a wrong table or sign in the
// triangulator cannot also fool the checks. The tolerance on a summed volume is derived below.

use std::collections::BTreeMap;

use robust::{insphere, orient3d, Coord3D};
use solidfem_tetra::{StepStatus, Tetrahedralization, Triangulator, HULL};

fn coord(p: [f64; 3]) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

/// A small deterministic generator for test inputs (SplitMix64).
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn point(&mut self) -> [f64; 3] {
        [self.unit(), self.unit(), self.unit()]
    }
}

/// Runs the triangulator to completion and returns its result.
pub fn triangulate(points: Vec<[f64; 3]>) -> Tetrahedralization {
    let mut triangulator = Triangulator::new(points).expect("the input is valid");
    let status = triangulator
        .step(usize::MAX, &mut || false)
        .expect("every point inserts");
    assert_eq!(status, StepStatus::Done);
    triangulator.result().expect("the run finished")
}

/// Every tetrahedron has strictly positive volume by exact test, in the right-handed sense
/// `(b-a) x (c-a) * (d-a) > 0`: the predicate's sign is the opposite of that volume.
pub fn assert_positive(result: &Tetrahedralization) {
    for (index, tet) in result.tets.iter().enumerate() {
        let [a, b, c, d] = tet.map(|v| coord(result.points[v as usize]));
        assert!(
            orient3d(a, b, c, d) < 0.0,
            "tetrahedron {index} {tet:?} is not right-handed by exact test"
        );
    }
}

/// How far `point` is inside the circumsphere of tetrahedron `tet`: positive strictly inside,
/// whichever way the tetrahedron is ordered.
fn inside(result: &Tetrahedralization, tet: [u32; 4], point: [f64; 3]) -> f64 {
    let [a, b, c, d] = tet.map(|v| coord(result.points[v as usize]));
    let sign = orient3d(a, b, c, d).signum();
    sign * insphere(a, b, c, d, coord(point))
}

/// The empty-sphere property, exactly, for every tetrahedron against every point. A vertex on the
/// sphere is allowed: it is what degenerate inputs have.
pub fn assert_empty_spheres(result: &Tetrahedralization) {
    for (index, &tet) in result.tets.iter().enumerate() {
        for (point_index, &point) in result.points.iter().enumerate() {
            if tet.contains(&(point_index as u32)) {
                continue;
            }
            assert!(
                inside(result, tet, point) <= 0.0,
                "point {point_index} is strictly inside the sphere of tetrahedron {index} {tet:?}"
            );
        }
    }
}

/// The empty-sphere property for `sample` tetrahedra spread through the list, against every point.
pub fn assert_empty_spheres_sampled(result: &Tetrahedralization, sample: usize) {
    let step = (result.tets.len() / sample).max(1);
    for index in (0..result.tets.len()).step_by(step) {
        let tet = result.tets[index];
        for (point_index, &point) in result.points.iter().enumerate() {
            if tet.contains(&(point_index as u32)) {
                continue;
            }
            assert!(
                inside(result, tet, point) <= 0.0,
                "point {point_index} is strictly inside the sphere of tetrahedron {index} {tet:?}"
            );
        }
    }
}

/// The neighbour table is symmetric and complete: `neighbors[t][i]` shares exactly the face
/// opposite `tets[t][i]`, the face points back, and each hull face is marked hull once. Returns
/// the number of hull faces.
pub fn assert_adjacency(result: &Tetrahedralization) -> usize {
    let mut faces: BTreeMap<[u32; 3], Vec<(u32, usize)>> = BTreeMap::new();
    for (t, tet) in result.tets.iter().enumerate() {
        for i in 0..4 {
            let mut face: Vec<u32> = (0..4).filter(|&j| j != i).map(|j| tet[j]).collect();
            face.sort_unstable();
            faces
                .entry([face[0], face[1], face[2]])
                .or_default()
                .push((t as u32, i));
        }
    }
    let mut hull = 0;
    for (face, owners) in &faces {
        match owners.as_slice() {
            [(t, i)] => {
                assert_eq!(
                    result.neighbors[*t as usize][*i], HULL,
                    "face {face:?} has one tetrahedron but tetrahedron {t} claims a neighbour there"
                );
                hull += 1;
            }
            [(t, i), (u, j)] => {
                assert_eq!(
                    result.neighbors[*t as usize][*i], *u,
                    "face {face:?} neighbour"
                );
                assert_eq!(
                    result.neighbors[*u as usize][*j], *t,
                    "face {face:?} neighbour"
                );
            }
            more => panic!("face {face:?} is shared by {} tetrahedra", more.len()),
        }
    }
    hull
}

/// `V - E + F - T = 1` for a triangulated ball and `V - E + F = 2` for its boundary, counted from
/// the tetrahedra and hull faces alone, and every input point is used.
pub fn assert_euler(result: &Tetrahedralization) {
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();
    let mut used = vec![false; result.points.len()];
    for tet in &result.tets {
        for &v in tet {
            used[v as usize] = true;
        }
        for a in 0..4 {
            for b in a + 1..4 {
                edges.push((tet[a].min(tet[b]), tet[a].max(tet[b])));
                for c in b + 1..4 {
                    let mut face = [tet[a], tet[b], tet[c]];
                    face.sort_unstable();
                    faces.push(face);
                }
            }
        }
    }
    assert!(
        used.iter().all(|&u| u),
        "a point is missing from the tetrahedra"
    );
    edges.sort_unstable();
    edges.dedup();
    let total_faces = faces.len();
    faces.sort_unstable();
    faces.dedup();
    let v = result.points.len() as i64;
    let e = edges.len() as i64;
    let f = faces.len() as i64;
    let t = result.tets.len() as i64;
    assert_eq!(v - e + f - t, 1, "Euler characteristic of the ball");

    // Boundary: faces owned by one tetrahedron only.
    let hull_faces: Vec<[u32; 3]> = {
        let mut unique = Vec::new();
        for (t, tet) in result.tets.iter().enumerate() {
            for i in 0..4 {
                if result.neighbors[t][i] == HULL {
                    let mut face: Vec<u32> = (0..4).filter(|&j| j != i).map(|j| tet[j]).collect();
                    face.sort_unstable();
                    unique.push([face[0], face[1], face[2]]);
                }
            }
        }
        unique
    };
    assert_eq!(
        2 * (total_faces - faces.len()) + hull_faces.len(),
        total_faces
    );
    let mut boundary_edges: Vec<(u32, u32)> = Vec::new();
    let mut boundary_vertices: Vec<u32> = Vec::new();
    for face in &hull_faces {
        boundary_vertices.extend_from_slice(face);
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            boundary_edges.push((face[i], face[j]));
        }
    }
    boundary_edges.sort_unstable();
    boundary_edges.dedup();
    boundary_vertices.sort_unstable();
    boundary_vertices.dedup();
    assert_eq!(
        boundary_vertices.len() as i64 - boundary_edges.len() as i64 + hull_faces.len() as i64,
        2,
        "Euler characteristic of the hull surface"
    );
}

/// The sum of the tetrahedra's volumes, by plain signed arithmetic.
pub fn total_volume(result: &Tetrahedralization) -> f64 {
    let mut total = 0.0;
    for tet in &result.tets {
        let [a, b, c, d] = tet.map(|v| result.points[v as usize]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
        let det = u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0]);
        total += det / 6.0;
    }
    total
}

/// The tolerance for comparing [`total_volume`] with an exact value on `result`: each tetrahedron's
/// determinant is about 12 products and sums of edges no longer than the bounding diagonal, so
/// its error is at most `16 * f64::EPSILON * diagonal^3` divided by 6; the sum adds at most
/// `f64::EPSILON` of the running total per addition.
pub fn volume_tolerance(result: &Tetrahedralization) -> f64 {
    let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for point in &result.points {
        for axis in 0..3 {
            low[axis] = low[axis].min(point[axis]);
            high[axis] = high[axis].max(point[axis]);
        }
    }
    let diagonal = (0..3).map(|a| high[a] - low[a]).fold(0.0, f64::hypot);
    let total = total_volume(result).abs();
    let per_tet = 16.0 * f64::EPSILON * diagonal.powi(3) / 6.0;
    result.tets.len() as f64 * (per_tet + f64::EPSILON * total)
}

/// All the checks that hold for every input, with the empty-sphere check done against all points.
pub fn assert_valid_small(result: &Tetrahedralization) {
    assert_positive(result);
    assert_empty_spheres(result);
    assert_adjacency(result);
    assert_euler(result);
}

/// The convex hull volume of points in general position, by brute force: a triple is a hull face
/// when all other points are strictly on one side, and the face contributes its pyramid to a
/// point inside the hull. Returns the volume; panics on points not in general position.
pub fn brute_hull_volume(points: &[[f64; 3]]) -> f64 {
    let n = points.len();
    let inside = {
        let mut sum = [0.0; 3];
        for point in points {
            for axis in 0..3 {
                sum[axis] += point[axis] / n as f64;
            }
        }
        sum
    };
    let mut volume = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let (mut positive, mut negative) = (false, false);
                for (l, &point) in points.iter().enumerate() {
                    if l == i || l == j || l == k {
                        continue;
                    }
                    let side = orient3d(
                        coord(points[i]),
                        coord(points[j]),
                        coord(points[k]),
                        coord(point),
                    );
                    assert!(side != 0.0, "input is not in general position");
                    if side > 0.0 {
                        positive = true;
                    } else {
                        negative = true;
                    }
                }
                if positive && negative {
                    continue;
                }
                let d = inside;
                let (a, b, c) = (points[i], points[j], points[k]);
                let u = [a[0] - d[0], a[1] - d[1], a[2] - d[2]];
                let v = [b[0] - d[0], b[1] - d[1], b[2] - d[2]];
                let w = [c[0] - d[0], c[1] - d[1], c[2] - d[2]];
                let det = u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
                    + u[2] * (v[0] * w[1] - v[1] * w[0]);
                volume += det.abs() / 6.0;
            }
        }
    }
    volume
}
