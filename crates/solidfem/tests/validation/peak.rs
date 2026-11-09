// The reported peak against the stresses the solver itself computed. A sharp re-entrant corner is
// where a smoothed field goes wrong: the vertex patch wraps round the corner and one linear tensor
// is fitted to both legs, so the fitted nodes read well below the integration points beside them.
// The reported peak must still hold those points, on every refinement.

use solidfem::elastic::{Material, Model, Solution};
use solidfem::mesh::{QuadraticTetMesh, TetMesh};

use crate::support::run;

const STEEL: Material = Material {
    youngs_modulus: 200_000.0,
    poisson_ratio: 0.3,
};

/// The six tetrahedra of a unit cell walk its diagonal, one axis per edge; the second of each
/// mirrored pair is flipped to keep its volume positive.
const WALKS: [([usize; 3], bool); 6] = [
    ([0, 1, 2], true),
    ([1, 2, 0], true),
    ([2, 0, 1], true),
    ([0, 2, 1], false),
    ([2, 1, 0], false),
    ([1, 0, 2], false),
];

/// A structured mesh of the cells of a `dims` grid of pitch `h` that `keep` accepts. The grid's
/// `box_tet_mesh` cannot leave cells out, and a re-entrant corner needs them gone.
fn hollowed_grid(
    h: f64,
    dims: [usize; 3],
    keep: impl Fn(usize, usize, usize) -> bool,
) -> QuadraticTetMesh {
    let index = |i: usize, j: usize, k: usize| i + (dims[0] + 1) * (j + (dims[1] + 1) * k);
    let mut node_of = vec![u32::MAX; (dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1)];
    let mut positions: Vec<[f64; 3]> = Vec::new();
    let mut tets: Vec<[u32; 4]> = Vec::new();
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                if !keep(i, j, k) {
                    continue;
                }
                for (walk, even) in WALKS {
                    let mut at = [i, j, k];
                    let mut tet = [0u32; 4];
                    for (step, slot) in tet.iter_mut().enumerate() {
                        let id = index(at[0], at[1], at[2]);
                        if node_of[id] == u32::MAX {
                            node_of[id] = positions.len() as u32;
                            positions.push([at[0] as f64 * h, at[1] as f64 * h, at[2] as f64 * h]);
                        }
                        *slot = node_of[id];
                        if step < 3 {
                            at[walk[step]] += 1;
                        }
                    }
                    tets.push(if even {
                        tet
                    } else {
                        [tet[0], tet[2], tet[1], tet[3]]
                    });
                }
            }
        }
    }
    QuadraticTetMesh::from_linear(&TetMesh::new(positions, tets).expect("a positive mesh"))
}

const LEG: f64 = 10.0;

/// An L angle of leg `LEG`, leg width and depth one tenth of that, `cells` to the leg: clamped at
/// one end, side-loaded at the other, with a sharp 270 degree re-entrant edge.
fn solve_angle(cells: usize) -> (QuadraticTetMesh, Solution) {
    let width = cells / 10;
    let h = LEG / cells as f64;
    let mesh = hollowed_grid(h, [cells, cells, width], |i, j, _| j < width || i < width);
    let nodes = mesh.nodes().to_vec();
    let mut model = Model::new(mesh.clone(), STEEL);
    for (node, p) in nodes.iter().enumerate() {
        if (p[0] - LEG).abs() < 1e-9 {
            model.fix_node(node as u32);
        }
    }
    for face in mesh.boundary_faces() {
        let on_end = face.nodes[..3]
            .iter()
            .all(|n| (nodes[*n as usize][1] - LEG).abs() < 1e-9);
        if on_end {
            model.add_traction(face, [10.0, 0.0, 0.0]);
        }
    }
    let solution = run(model);
    (mesh, solution)
}

/// The window away from the clamp and the load end: the re-entrant corner is inside it.
fn away_from_the_clamp(p: [f64; 3]) -> bool {
    p[0] < 0.75 * LEG
}

fn largest_point(solution: &Solution, inside: impl Fn([f64; 3]) -> bool) -> f64 {
    solution
        .point_stresses
        .iter()
        .filter(|point| inside(point.position))
        .map(|point| point.von_mises)
        .fold(f64::MIN, f64::max)
}

fn largest_node(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
    inside: impl Fn([f64; 3]) -> bool,
) -> f64 {
    mesh.nodes()
        .iter()
        .zip(&solution.nodal_von_mises)
        .filter(|(p, _)| inside(**p))
        .map(|(_, value)| *value)
        .fold(f64::MIN, f64::max)
}

#[test]
fn a_re_entrant_corner_never_reads_below_its_integration_points() {
    for cells in [20, 30, 40] {
        let (mesh, solution) = solve_angle(cells);
        let point = largest_point(&solution, away_from_the_clamp);
        let node = largest_node(&mesh, &solution, away_from_the_clamp);
        let peak = solution
            .peak_von_mises(&mesh, away_from_the_clamp)
            .expect("the window holds the corner");
        println!(
            "VALIDATION L angle cells={cells}: peak {peak:.3}, largest node {node:.3}, largest integration point {point:.3}"
        );
        assert!(peak.is_finite() && peak > 0.0, "peak {peak}");
        assert!(
            peak >= point,
            "cells {cells}: the peak {peak} is below the largest integration point {point}"
        );
        assert!(peak >= node, "cells {cells}: {peak} below node {node}");
    }
}

/// The region a peak reads is the one its caller gives: a point in the clamp zone counts for
/// neither a node nor an integration point. The comparison is exact because both sides read
/// the same values, classified by the same physical positions.
#[test]
fn the_peak_is_read_by_physical_position_in_the_region() {
    let (mesh, solution) = solve_angle(20);
    let whole = solution.peak_von_mises(&mesh, |_| true).expect("a mesh");
    let inside = solution
        .peak_von_mises(&mesh, away_from_the_clamp)
        .expect("the window");
    let expected = largest_point(&solution, away_from_the_clamp).max(largest_node(
        &mesh,
        &solution,
        away_from_the_clamp,
    ));
    assert_eq!(inside.to_bits(), expected.to_bits());
    assert!(whole >= inside, "{whole} against {inside}");
    assert!(
        solution.peak_von_mises(&mesh, |p| p[0] < -1.0).is_none(),
        "an empty region has no peak"
    );
}
