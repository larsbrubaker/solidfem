// The independent checks of a solid mesh against the surface it was given, reading only the
// public result. Exact predicates decide every yes-or-no question; the volume comparison uses a
// tolerance derived from the size of the part and the number of terms summed.

use std::collections::HashMap;

use robust::{orient3d, Coord3D};
use solidfem_tetra::{SolidMesh, StepStatus, SurfaceMesher};

fn coord(p: [f64; 3]) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

/// Runs the mesher to the end, returning the solid or the error it stopped with.
pub fn mesh(
    points: &[[f64; 3]],
    triangles: &[[u32; 3]],
) -> Result<SolidMesh, solidfem_tetra::TetraError> {
    let mut mesher = SurfaceMesher::new(points.to_vec(), triangles.to_vec())?;
    assert_eq!(
        mesher.step(usize::MAX, &mut || false)?,
        StepStatus::Done,
        "the run finishes"
    );
    assert_eq!(mesher.report().steiner_points, 0);
    mesher.result()
}

/// Every check of a successful result against its surface.
pub fn assert_solid(result: &SolidMesh, points: &[[f64; 3]], triangles: &[[u32; 3]]) {
    assert_eq!(
        &result.points[..points.len()],
        points,
        "surface coordinates must be untouched, bit for bit"
    );
    assert_eq!(
        result.points.len(),
        points.len(),
        "no point may be added: this stage adds none"
    );
    let mut faces: HashMap<[u32; 3], Vec<(usize, u32)>> = HashMap::new();
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
    let surface: HashMap<[u32; 3], [u32; 3]> = triangles
        .iter()
        .map(|t| {
            let mut key = *t;
            key.sort_unstable();
            (key, *t)
        })
        .collect();
    assert_eq!(
        surface.len(),
        triangles.len(),
        "the input repeats a triangle"
    );
    let mut boundary = 0usize;
    for (key, owners) in &faces {
        match owners.len() {
            1 => {
                let winding = surface.get(key).unwrap_or_else(|| {
                    panic!("face {key:?} has one tetrahedron but is not a surface triangle")
                });
                boundary += 1;
                // Outward winding: the solid's apex is below the triangle seen from outside.
                let apex = result.points[owners[0].1 as usize];
                let [a, b, c] = winding.map(|v| coord(result.points[v as usize]));
                assert!(
                    orient3d(a, b, c, coord(apex)) > 0.0,
                    "surface triangle {winding:?} has the solid on its outer side"
                );
            }
            2 => assert!(
                !surface.contains_key(key),
                "surface triangle {key:?} is inside the solid (both sides are tetrahedra)"
            ),
            many => panic!("face {key:?} is shared by {many} tetrahedra"),
        }
    }
    assert_eq!(
        boundary,
        triangles.len(),
        "every surface triangle must be exactly one boundary face"
    );
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
/// EPSILON of the running total per term. `size` includes the distance from the origin because
/// the divergence sum is taken about the origin.
fn volume_tolerance(points: &[[f64; 3]], terms: usize) -> f64 {
    let size = points
        .iter()
        .flatten()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()))
        * 2.0;
    let per_term = 16.0 * f64::EPSILON * size.powi(3);
    terms as f64 * (per_term + f64::EPSILON * size.powi(3))
}
