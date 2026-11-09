// A flat patch's triangulation, measured on the public result: the inner patch edges not locally
// Delaunay are exactly the ones the recovery report counts as refused, and no boundary triangle
// is only a rounding away from a straight line, except the one known input pinned here by face.
// The cases include lattices rounded to `f64`, whose points are not exactly on their grid lines,
// and each shape's rotated form.

use crate::check::{assert_solid, mesh, non_delaunay_edges, patches_of};
use crate::shapes::{
    hollow_box, needle_box, plate_with_holes, prism, rotated, subdivided_box, Surface,
};
use solidfem_tetra::{SolidMesh, HULL};

/// The relative height (height over longest edge) under which a triangle is only rounding away
/// from a line. Rounded lattice coordinates carry one unit of rounding of at most the part's
/// largest coordinate, so a triple on a lattice line is off it by about `EPSILON * coordinate`.
/// With coordinates up to 100 and a triangle's longest edge at least the shortest lattice step a
/// few percent of that, the ratio stays near `2.2e-16 * 100 / 3`, under `1e-14`; the bound below
/// is a few times that, and every real triangle measured here is above `1e-3`.
const ROUNDING_RELATIVE_HEIGHT: f64 = 1e-13;

/// Boundary faces as sorted triples.
fn boundary_triangles(mesh: &SolidMesh) -> Vec<[u32; 3]> {
    let mut faces = Vec::new();
    for (t, tet) in mesh.tets.iter().enumerate() {
        for i in 0..4 {
            if mesh.neighbors[t][i] == HULL {
                let mut face: Vec<u32> = (0..4).filter(|&j| j != i).map(|j| tet[j]).collect();
                face.sort_unstable();
                faces.push([face[0], face[1], face[2]]);
            }
        }
    }
    faces.sort_unstable();
    faces
}

/// Height over longest edge, from the cross product and the three lengths.
fn relative_height(points: &[[f64; 3]], face: [u32; 3]) -> (f64, (u32, u32)) {
    let [a, b, c] = face.map(|v| points[v as usize]);
    let (u, v) = (sub(b, a), sub(c, a));
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let edges = [
        (length_squared(sub(a, b)), (face[0], face[1])),
        (length_squared(sub(a, c)), (face[0], face[2])),
        (length_squared(sub(b, c)), (face[1], face[2])),
    ];
    let (longest, edge) =
        edges.into_iter().fold(
            (0.0_f64, (0, 0)),
            |best, e| if e.0 > best.0 { e } else { best },
        );
    let cross_length = length_squared(n).sqrt();
    (cross_length / longest, edge)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn length_squared(a: [f64; 3]) -> f64 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}

/// One boundary triangle a rounding away from a line: its corners and its longest edge.
struct RoundingTriangle {
    face: [u32; 3],
    longest_edge: (u32, u32),
    /// The longest edge is on the patch's boundary, so the boundary chain forces the triangle.
    forced: bool,
}

/// The boundary triangles of `result` a rounding away from a line, sorted by face.
fn rounding_triangles(result: &SolidMesh) -> Vec<RoundingTriangle> {
    let boundary = boundary_triangles(result);
    let mut thin = Vec::new();
    for (bounds, tris) in &patches_of(&result.points, &boundary) {
        for &face in tris {
            let mut key = face;
            key.sort_unstable();
            let (height, longest_edge) = relative_height(&result.points, key);
            if height < ROUNDING_RELATIVE_HEIGHT {
                thin.push(RoundingTriangle {
                    face: key,
                    longest_edge,
                    forced: bounds.binary_search(&longest_edge).is_ok(),
                });
            }
        }
    }
    thin.sort_unstable_by_key(|triangle| triangle.face);
    thin
}

/// Meshes both forms. The axis-aligned form has exactly the rounding triangles `known_slivers`
/// and the rotated form has none; none is forced by a patch boundary, each one's longest edge is
/// an inner edge the report counts as refused, and the inner edges that are not Delaunay are
/// exactly as many as the report refuses.
fn assert_rounding_triangles(name: &str, surface: &Surface, known_slivers: &[[u32; 3]]) {
    let (exact, exact_report) = mesh(&surface.points, &surface.triangles)
        .unwrap_or_else(|error| panic!("{name}, axis aligned: {error}"));
    assert_solid(&exact, &surface.points, &surface.triangles);
    let (points, triangles) = rotated(surface);
    let (spun, spun_report) =
        mesh(&points, &triangles).unwrap_or_else(|error| panic!("{name}, rotated: {error}"));
    assert_solid(&spun, &points, &triangles);
    for (form, result, report, expected) in [
        ("axis aligned", &exact, exact_report, known_slivers),
        ("rotated", &spun, spun_report, &[][..]),
    ] {
        let thin = rounding_triangles(result);
        let non_delaunay = non_delaunay_edges(result);
        let faces: Vec<[u32; 3]> = thin.iter().map(|triangle| triangle.face).collect();
        eprintln!(
            "{name}, {form}: rounding triangles {faces:?}; in-plane flips {}, refused {}, inner \
             edges not Delaunay {}",
            report.in_plane_flips,
            report.in_plane_flips_refused,
            non_delaunay.len(),
        );
        assert_eq!(
            faces, expected,
            "{name}, {form}: boundary triangles a rounding away from a line"
        );
        for triangle in &thin {
            assert!(
                !triangle.forced,
                "{name}, {form}: {:?} is forced by the patch boundary",
                triangle.face
            );
            assert!(
                non_delaunay.contains(&triangle.longest_edge),
                "{name}, {form}: the long edge of {:?} is not among the refused inner edges",
                triangle.face
            );
        }
        assert_eq!(
            non_delaunay.len(),
            report.in_plane_flips_refused,
            "{name}, {form}: every inner patch edge is Delaunay but those the report refuses"
        );
    }
}

fn thin_hollow_box() -> Surface {
    let side = 100.0;
    let diagonal = (3.0_f64 * side * side).sqrt();
    hollow_box(side, diagonal / 125.0, 6, 6)
}

/// A known limitation, pinned by input. Axis aligned, this box keeps one triangle a rounding away
/// from a line, on the void's face at x = 29: the flip that would remove it has no legal
/// retriangulation of the tetrahedra around its edge by exact test, so it stays and the report
/// counts it among the refused flips. Every other surveyed shape has none.
#[test]
fn one_hollow_box_keeps_one_rounding_sliver_and_the_report_counts_its_refused_flip() {
    assert_rounding_triangles(
        "hollow 30/1/6/5",
        &hollow_box(30.0, 1.0, 6, 5),
        &[[295, 319, 342]],
    );
}

#[test]
fn hollow_boxes_with_rounded_lattices_have_no_triangle_a_rounding_away_from_a_line() {
    assert_rounding_triangles("hollow 30/1/7/5", &hollow_box(30.0, 1.0, 7, 5), &[]);
    assert_rounding_triangles("hollow 30/1/8/5", &hollow_box(30.0, 1.0, 8, 5), &[]);
    assert_rounding_triangles("hollow 100/thin/6/6", &thin_hollow_box(), &[]);
}

#[test]
fn every_surveyed_shape_has_no_triangle_a_rounding_away_from_a_line() {
    let holes = [(1, 1, true), (3, 1, false), (2, 2, true)];
    let cases: [(&str, Surface); 10] = [
        ("hollow 10/2/3/2", hollow_box(10.0, 2.0, 3, 2)),
        ("hollow 30/1/5/6", hollow_box(30.0, 1.0, 5, 6)),
        ("hollow 30/1/6/7", hollow_box(30.0, 1.0, 6, 7)),
        (
            "plate 10x6",
            plate_with_holes(10.0, 6.0, 0.6, 5, 3, 4, &holes),
        ),
        (
            "plate 20x12",
            plate_with_holes(20.0, 12.0, 0.8, 5, 3, 2, &[(1, 1, true), (3, 1, false)]),
        ),
        ("box 2/4", subdivided_box([0.0; 3], [2.0; 3], 4)),
        ("box 40/3", subdivided_box([0.0; 3], [40.0; 3], 3)),
        (
            "slab",
            subdivided_box([0.0, 0.0, 0.0], [100.0, 80.0, 0.5], 6),
        ),
        ("needle 200", needle_box(40.0, 200)),
        ("fan prism", prism(40, 5.0, 2.0, false)),
    ];
    for (name, surface) in cases {
        assert_rounding_triangles(name, &surface, &[]);
    }
}
