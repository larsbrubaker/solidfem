//! Known-answer checks for the corpus readings: a cube's topology, sharp edges and wall thickness
//! are exact, an equilateral triangle's aspect ratio is 1, two crossed cubes penetrate, a cube
//! with a face removed is open, and the remesher keeps a plate closed and on its surface. These
//! are what the corpus tables rest on: a wrong reading here would silently mislead the choice of
//! tetrahedral mesher.

use solidfem_bench::corpus::bvh::Bvh;
use solidfem_bench::corpus::measure::{aspect_ratio, measure_input, topology, wall_reading};
use solidfem_bench::corpus::remesh_trial::remesh_and_measure;
use solidfem_bench::corpus::self_intersect::count_penetrating_pairs;
use solidfem_bench::corpus::soup::TriSoup;

/// A closed axis-aligned box from the origin to `size`, wound outward.
pub fn cuboid(size: [f64; 3], offset: [f64; 3]) -> TriSoup {
    let corner = |x: usize, y: usize, z: usize| {
        [
            offset[0] + x as f64 * size[0],
            offset[1] + y as f64 * size[1],
            offset[2] + z as f64 * size[2],
        ]
    };
    let vertices = vec![
        corner(0, 0, 0),
        corner(1, 0, 0),
        corner(1, 1, 0),
        corner(0, 1, 0),
        corner(0, 0, 1),
        corner(1, 0, 1),
        corner(1, 1, 1),
        corner(0, 1, 1),
    ];
    let triangles = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ];
    TriSoup {
        vertices,
        triangles,
    }
}

#[test]
fn a_cube_is_closed_with_twelve_sharp_edges_and_its_full_thickness_as_wall() {
    let soup = cuboid([20.0, 20.0, 20.0], [0.0, 0.0, 0.0]);
    let m = measure_input(&soup);
    assert!(m.topology.is_manifold_and_oriented());
    assert_eq!(m.topology.components, 1);
    // The 12 cube edges; the 6 face diagonals are flat and do not count.
    assert_eq!(m.topology.sharp_edges.len(), 12);
    assert_eq!(m.self_intersection.pairs, 0);
    // Axis-aligned, so the rays are exact: every face sees the far wall at exactly 20.
    assert_eq!(m.wall.minimum, 20.0);
    assert_eq!(m.wall.p5, 20.0);
    assert_eq!(m.volume, 8000.0);
}

#[test]
fn a_flat_plate_reads_its_thickness_not_its_width() {
    let soup = cuboid([50.0, 30.0, 2.0], [0.0, 0.0, 0.0]);
    let bvh = Bvh::new(&soup);
    let wall = wall_reading(&soup, &bvh);
    assert_eq!(wall.minimum, 2.0, "the thin dimension is the wall");
}

#[test]
fn an_equilateral_triangle_has_aspect_ratio_one() {
    let area = 3.0_f64.sqrt() / 4.0;
    let ratio = aspect_ratio([1.0, 1.0, 1.0], area);
    // The formula is exact arithmetic on these values apart from the square root's rounding.
    assert!((ratio - 1.0).abs() < 1e-14, "ratio {ratio}");
}

#[test]
fn a_cube_missing_a_face_is_open() {
    let mut soup = cuboid([10.0, 10.0, 10.0], [0.0, 0.0, 0.0]);
    soup.triangles.truncate(10);
    let t = topology(&soup);
    assert!(!t.is_closed());
    assert_eq!(t.boundary_edges, 4, "the missing square's rim");
}

#[test]
fn two_interpenetrating_cubes_are_reported_and_a_single_cube_is_not() {
    let one = cuboid([10.0, 10.0, 10.0], [0.0, 0.0, 0.0]);
    let bvh = Bvh::new(&one);
    assert_eq!(count_penetrating_pairs(&one, &bvh).pairs, 0);

    let other = cuboid([10.0, 10.0, 10.0], [5.0, 5.0, 5.0]);
    let offset = one.vertices.len() as u32;
    let mut crossed = one.clone();
    crossed.vertices.extend_from_slice(&other.vertices);
    crossed
        .triangles
        .extend(other.triangles.iter().map(|t| t.map(|v| v + offset)));
    let bvh = Bvh::new(&crossed);
    assert!(count_penetrating_pairs(&crossed, &bvh).pairs > 0);
}

#[test]
fn the_remesher_keeps_a_plate_closed_on_its_surface_with_edges_held() {
    let soup = cuboid([40.0, 20.0, 4.0], [0.0, 0.0, 0.0]);
    let metrics = measure_input(&soup);
    let trial = remesh_and_measure(&soup, &metrics);
    let part = trial.outcome.expect("a plate remeshes");
    assert!(part.closed, "stayed closed");
    assert!(part.manifold_and_oriented, "stayed consistently wound");
    assert!(
        part.triangles > soup.triangles.len(),
        "refined toward the target"
    );
    // The faces are flat and the sharp edges held, so the volume is preserved to rounding. The
    // surface deviation is bounded by the remesher's projection tolerance: it leaves a vertex
    // unmoved when the projection is within 1e-6 per axis (`ZERO_TOLERANCEF`), so a point off
    // the surface by up to sqrt(3) * 1e-6 survives, over the diagonal.
    assert!(
        part.volume_change_percent.abs() < 1e-9,
        "{}",
        part.volume_change_percent
    );
    assert!(
        part.deviation_output_to_input < 3.0_f64.sqrt() * 1e-6 / metrics.diagonal,
        "{}",
        part.deviation_output_to_input
    );
    assert_eq!(part.sharp_survival_percent, 100.0);
}
