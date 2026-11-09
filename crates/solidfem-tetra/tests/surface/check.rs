// The independent checks of a solid mesh against the surface it was given, reading only the public
// result. Exact predicates decide every yes-or-no question; areas and volumes are compared under
// tolerances derived from the size of the part and the number of terms summed.
//
// The surface contract these check: every surface vertex keeps its bits and its position, no point
// is added on the surface, every non-flat surface edge is present, and each flat patch (a region
// of input triangles exactly coplanar across their shared edges) keeps its boundary loops, its
// plane and its area. The triangulation inside a patch is the mesher's choice: an internal
// edge of a flat region is not part of the surface's shape.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use robust::{orient2d, orient3d, Coord, Coord3D};
use solidfem_tetra::{SolidMesh, StepStatus, SurfaceMesher};

fn coord(p: [f64; 3]) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

/// Runs the mesher to the end, returning the solid and the report, or the error it stopped with.
pub fn mesh(
    points: &[[f64; 3]],
    triangles: &[[u32; 3]],
) -> Result<(SolidMesh, solidfem_tetra::RecoveryReport), solidfem_tetra::TetraError> {
    let mut mesher = SurfaceMesher::new(points.to_vec(), triangles.to_vec())?;
    assert_eq!(
        mesher.step(usize::MAX, &mut || false)?,
        StepStatus::Done,
        "the run finishes"
    );
    let result = mesher.result()?;
    Ok((result, mesher.report()))
}

/// Exact: are the two triangles sharing an edge flat and folded out flat, their apexes on
/// opposite sides of the shared edge? Decided in an axis plane the triangle does not collapse in.
fn flat_across(points: &[[f64; 3]], t: [u32; 3], s: [u32; 3], a: u32, b: u32) -> bool {
    let apex_t = t
        .iter()
        .copied()
        .find(|&v| v != a && v != b)
        .expect("a third vertex");
    let apex_s = s
        .iter()
        .copied()
        .find(|&v| v != a && v != b)
        .expect("a third vertex");
    let [pa, pb, pc] = t.map(|v| coord(points[v as usize]));
    if orient3d(pa, pb, pc, coord(points[apex_s as usize])) != 0.0 {
        return false;
    }
    for axes in [(0, 1), (1, 2), (0, 2)] {
        let at = |v: u32| {
            let p = points[v as usize];
            Coord {
                x: p[axes.0],
                y: p[axes.1],
            }
        };
        if orient2d(at(t[0]), at(t[1]), at(t[2])) == 0.0 {
            continue;
        }
        return orient2d(at(a), at(b), at(apex_t)) * orient2d(at(a), at(b), at(apex_s)) < 0.0;
    }
    false
}

/// One flat region: its sorted boundary edges and its triangles.
type Patch = (Vec<(u32, u32)>, Vec<[u32; 3]>);

/// The patches of a triangle soup: connected sets of triangles flat across their shared edges,
/// each with its sorted boundary edges (used by one triangle of the patch, or shared but not
/// flat) and its triangles.
fn patches_of(points: &[[f64; 3]], triangles: &[[u32; 3]]) -> Vec<Patch> {
    let mut by_edge: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (index, t) in triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            by_edge.entry((a.min(b), a.max(b))).or_default().push(index);
        }
    }
    let mut component = vec![usize::MAX; triangles.len()];
    let mut results = Vec::new();
    for seed in 0..triangles.len() {
        if component[seed] != usize::MAX {
            continue;
        }
        let id = results.len();
        component[seed] = id;
        let mut stack = vec![seed];
        let mut members = vec![seed];
        while let Some(t) = stack.pop() {
            let tri = triangles[t];
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                for &s in &by_edge[&(a.min(b), a.max(b))] {
                    if s != t
                        && component[s] == usize::MAX
                        && flat_across(points, tri, triangles[s], a, b)
                    {
                        component[s] = id;
                        stack.push(s);
                        members.push(s);
                    }
                }
            }
        }
        let mut boundary = Vec::new();
        for &t in &members {
            let tri = triangles[t];
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                let sharing = &by_edge[&(a.min(b), a.max(b))];
                let interior =
                    sharing.len() == 2 && sharing.iter().all(|&s| component[s] == id) && {
                        let other = sharing
                            .iter()
                            .copied()
                            .find(|&s| s != t)
                            .expect("two sharers");
                        flat_across(points, tri, triangles[other], a, b)
                    };
                if !interior {
                    boundary.push((a.min(b), a.max(b)));
                }
            }
        }
        boundary.sort_unstable();
        boundary.dedup();
        results.push((boundary, members.iter().map(|&t| triangles[t]).collect()));
    }
    results
}

/// Every check of a successful result against its surface.
pub fn assert_solid(result: &SolidMesh, points: &[[f64; 3]], triangles: &[[u32; 3]]) {
    assert_eq!(
        &result.points[..points.len()],
        points,
        "surface coordinates must be untouched, bit for bit"
    );
    let surface_count = points.len() as u32;
    let mut faces: BTreeMap<[u32; 3], Vec<(usize, u32)>> = BTreeMap::new();
    for (t, tet) in result.tets.iter().enumerate() {
        let [a, b, c, d] = tet.map(|v| coord(result.points[v as usize]));
        assert!(
            orient3d(a, b, c, d) < 0.0,
            "tetrahedron {t} {tet:?} is not strictly right-handed"
        );
        for i in 0..4 {
            let mut face: Vec<u32> = (0..4).filter(|&j| j != i).map(|j| tet[j]).collect();
            face.sort_unstable();
            faces
                .entry([face[0], face[1], face[2]])
                .or_default()
                .push((t, tet[i]));
        }
    }
    let mut boundary: Vec<[u32; 3]> = Vec::new();
    for (key, owners) in &faces {
        match owners.len() {
            1 => {
                assert!(
                    key.iter().all(|&v| v < surface_count),
                    "boundary face {key:?} uses a point added inside the solid"
                );
                boundary.push(*key);
            }
            2 => {}
            many => panic!("face {key:?} is shared by {many} tetrahedra"),
        }
    }
    // Added points: none on the surface, by exact test against every surface triangle.
    for (index, &point) in result.points.iter().enumerate().skip(points.len()) {
        assert!(
            triangles
                .iter()
                .all(|t| !on_triangle_exact(points, *t, point)),
            "added point {index} lies on the surface"
        );
    }
    for added in surface_count..result.points.len() as u32 {
        assert!(
            result.tets.iter().any(|tet| tet.contains(&added)),
            "added point {added} is in no tetrahedron, so it was not placed inside the solid"
        );
    }
    let used: BTreeSet<u32> = boundary.iter().flatten().copied().collect();
    assert_eq!(
        used.len(),
        points.len(),
        "every surface vertex must be on the boundary"
    );

    // The flat patches of the output and the input must match in their boundaries exactly.
    let input_patches = patches_of(points, triangles);
    let output_patches = patches_of(&result.points, &boundary);
    assert_eq!(
        input_patches.len(),
        output_patches.len(),
        "the boundary has a different number of flat patches than the surface"
    );
    let mut matched_outputs: BTreeMap<Vec<(u32, u32)>, Vec<[u32; 3]>> = BTreeMap::new();
    for (bounds, tris) in output_patches {
        matched_outputs.insert(bounds, tris);
    }
    let mut owner_plane: HashMap<[u32; 3], [u32; 3]> = HashMap::new();
    for (bounds, tris) in &input_patches {
        let output = matched_outputs
            .get(bounds)
            .unwrap_or_else(|| panic!("input patch boundary {bounds:?} is not in the output"));
        let [a, b, c] = tris[0].map(|v| coord(points[v as usize]));
        let in_plane = |t: &[u32; 3]| {
            t.iter()
                .all(|&v| orient3d(a, b, c, coord(points[v as usize])) == 0.0)
        };
        assert!(
            output.iter().all(in_plane),
            "an output face left its patch plane"
        );
        let tolerance = area_tolerance(points, tris.len() + output.len());
        let (input_area, output_area) = (total_area(points, tris), total_area(points, output));
        assert!(
            (input_area - output_area).abs() <= tolerance,
            "patch area changed from {input_area} to {output_area}, tolerance {tolerance}"
        );
        for face in output {
            let mut key = *face;
            key.sort_unstable();
            owner_plane.insert(key, tris[0]);
        }
    }
    // Orientation: the solid apex of every boundary face is below its patch's outward plane.
    for key in &boundary {
        let owners = &faces[key];
        let plane = owner_plane[key];
        let apex = result.points[owners[0].1 as usize];
        let [a, b, c] = plane.map(|v| coord(points[v as usize]));
        assert!(
            orient3d(a, b, c, coord(apex)) > 0.0,
            "boundary face {key:?} has the solid on its outer side"
        );
    }
    let surface_volume = divergence_volume(points, triangles);
    let summed: f64 = result
        .tets
        .iter()
        .map(|tet| tet_volume(&result.points, tet))
        .sum();
    let tolerance = volume_tolerance(points, result.tets.len() + triangles.len());
    assert!(
        (summed - surface_volume).abs() <= tolerance,
        "tetrahedra sum to {summed}, the surface encloses {surface_volume}, tolerance {tolerance}"
    );
}

fn on_triangle_exact(points: &[[f64; 3]], t: [u32; 3], point: [f64; 3]) -> bool {
    let [a, b, c] = t.map(|v| points[v as usize]);
    if orient3d(coord(a), coord(b), coord(c), coord(point)) != 0.0 {
        return false;
    }
    for axes in [(0, 1), (1, 2), (0, 2)] {
        let at = |p: [f64; 3]| Coord {
            x: p[axes.0],
            y: p[axes.1],
        };
        if orient2d(at(a), at(b), at(c)) == 0.0 {
            continue;
        }
        let sides = [
            orient2d(at(a), at(b), at(point)),
            orient2d(at(b), at(c), at(point)),
            orient2d(at(c), at(a), at(point)),
        ];
        return sides.iter().all(|&s| s >= 0.0) || sides.iter().all(|&s| s <= 0.0);
    }
    false
}

fn triangle_area(points: &[[f64; 3]], t: &[u32; 3]) -> f64 {
    let [a, b, c] = t.map(|v| points[v as usize]);
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
}

fn total_area(points: &[[f64; 3]], triangles: &[[u32; 3]]) -> f64 {
    triangles.iter().map(|t| triangle_area(points, t)).sum()
}

fn part_size(points: &[[f64; 3]]) -> f64 {
    points
        .iter()
        .flatten()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()))
        * 2.0
}

/// Each area is built from about a dozen products of terms no larger than the part's size, so
/// each has error at most `16 * EPSILON * size^2`; the sum adds one EPSILON of its running total
/// per term.
fn area_tolerance(points: &[[f64; 3]], terms: usize) -> f64 {
    let size = part_size(points);
    terms as f64 * (17.0 * f64::EPSILON * size * size)
}

/// The volume enclosed by outward-wound triangles, by the divergence theorem (each triangle's
/// signed tetrahedron with the origin).
pub fn divergence_volume(points: &[[f64; 3]], triangles: &[[u32; 3]]) -> f64 {
    triangles
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|v| points[v as usize]);
            (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0
        })
        .sum()
}

fn tet_volume(points: &[[f64; 3]], tet: &[u32; 4]) -> f64 {
    let [a, b, c, d] = tet.map(|v| points[v as usize]);
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0]))
        / 6.0
}

/// Each determinant is about 12 products and sums of terms no larger than the part's largest
/// coordinate cubed, so a term's error is at most `16 * EPSILON * size^3`; the sums add one
/// EPSILON of the running total per term. The size includes the distance from the origin because
/// the divergence sum is taken about the origin.
fn volume_tolerance(points: &[[f64; 3]], terms: usize) -> f64 {
    let size = part_size(points);
    terms as f64 * (17.0 * f64::EPSILON * size.powi(3))
}
