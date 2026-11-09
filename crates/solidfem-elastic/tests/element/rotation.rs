// Local rotation recovery against fields with known answers, on a structured mesh and on a mesh
// with jittered interior vertices: a rigid rotation is the case the page's arc drawing must keep
// rigid, and the torsion field is the one a twisted bar shows. Both are at most quadratic, which
// the 10-node element represents exactly, so every node must get the analytic value.

use solidfem_elastic::{nodal_rotations, ModelError};
use solidfem_mesh::{box_tet_mesh, tet_volume, QuadraticTetMesh, TetMesh};

const SIZE: f64 = 3.0;
const CELLS: usize = 3;

fn structured_mesh() -> QuadraticTetMesh {
    let linear = box_tet_mesh([0.0; 3], [SIZE; 3], [CELLS; 3]).expect("box");
    QuadraticTetMesh::from_linear(&linear)
}

fn distorted_mesh() -> QuadraticTetMesh {
    let h = SIZE / CELLS as f64;
    let linear = box_tet_mesh([0.0; 3], [SIZE; 3], [CELLS; 3])
        .expect("box")
        .with_mapped_positions(|p| {
            if p.iter().any(|c| *c == 0.0 || *c == SIZE) {
                return p;
            }
            let jitter = 0.15 * h;
            [
                p[0] + jitter * (13.0 * p[0] + 7.0 * p[1] + 3.0 * p[2]).sin(),
                p[1] + jitter * (5.0 * p[0] + 11.0 * p[1] + 2.0 * p[2]).sin(),
                p[2] + jitter * (3.0 * p[0] + 4.0 * p[1] + 17.0 * p[2]).sin(),
            ]
        })
        .expect("the jitter keeps every tetrahedron positive");
    QuadraticTetMesh::from_linear(&linear)
}

fn meshes() -> [(&'static str, QuadraticTetMesh); 2] {
    [
        ("structured", structured_mesh()),
        ("distorted", distorted_mesh()),
    ]
}

/// Asserts every node's rotation against `exact`, within `tolerance`.
fn assert_rotations(
    name: &str,
    mesh: &QuadraticTetMesh,
    rotations: &[[f64; 3]],
    exact: impl Fn([f64; 3]) -> [f64; 3],
    tolerance: f64,
) {
    assert_eq!(rotations.len(), mesh.nodes().len());
    for (node, (got, position)) in rotations.iter().zip(mesh.nodes()).enumerate() {
        let want = exact(*position);
        for axis in 0..3 {
            assert!(
                (got[axis] - want[axis]).abs() < tolerance,
                "{name} node {node} axis {axis}: {} vs {}",
                got[axis],
                want[axis]
            );
        }
    }
}

#[test]
fn a_rigid_rotation_gives_its_own_rotation_at_every_corner_and_midside_node() {
    let omega = [0.0007, -0.0011, 0.0013];
    for (name, mesh) in meshes() {
        let u: Vec<[f64; 3]> = mesh
            .nodes()
            .iter()
            .map(|p| {
                [
                    omega[1] * p[2] - omega[2] * p[1],
                    omega[2] * p[0] - omega[0] * p[2],
                    omega[0] * p[1] - omega[1] * p[0],
                ]
            })
            .collect();
        let rotations = nodal_rotations(&mesh, &u).expect("rotations");
        // Rounding: the gradient loses about the Jacobian's condition number (under 1e2 for this
        // jitter) times 2.2e-16 relative to the size of the nodal displacement over the element
        // size; both are of order one here, so 1e-11 relative to |omega| is that loss with a
        // margin of several hundred.
        let tolerance = 1e-11 * 0.0013;
        assert_rotations(name, &mesh, &rotations, |_| omega, tolerance);
    }
}

#[test]
fn a_torsion_field_gives_the_analytic_rotation_at_every_node() {
    // u = kappa z (-y, x, 0): half its curl is kappa (-x / 2, -y / 2, z).
    let kappa = 1.7e-3;
    for (name, mesh) in meshes() {
        let u: Vec<[f64; 3]> = mesh
            .nodes()
            .iter()
            .map(|p| [-kappa * p[2] * p[1], kappa * p[2] * p[0], 0.0])
            .collect();
        let rotations = nodal_rotations(&mesh, &u).expect("rotations");
        // The field is quadratic, which the 10-node element represents exactly on its straight
        // edges, so the error is the same rounding as above against a field of size kappa * SIZE
        // per unit length: 1e-11 relative to kappa * SIZE.
        let tolerance = 1e-11 * kappa * SIZE;
        assert_rotations(
            name,
            &mesh,
            &rotations,
            |p| [-0.5 * kappa * p[0], -0.5 * kappa * p[1], kappa * p[2]],
            tolerance,
        );
    }
}

#[test]
fn a_kinked_field_weights_each_element_by_its_volume() {
    // Two tetrahedra of different volume share the face c0-c1-c2. A displacement that is rigid
    // rotation `a` in the first and `a` plus a shear in the second is continuous across that face
    // and has a different rotation in each, so the nodes of the shared face see the average.
    let (a, b, c) = ([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]);
    let (top, bottom) = ([0.5, 0.5, 2.0], [0.5, 0.5, -5.0]);
    let linear = TetMesh::new(vec![a, b, c, top, bottom], vec![[0, 1, 2, 3], [1, 0, 2, 4]])
        .expect("two tets");
    let volume_top = tet_volume(a, b, c, top).abs();
    let volume_bottom = tet_volume(b, a, c, bottom).abs();
    let mesh = QuadraticTetMesh::from_linear(&linear);

    let gamma = 0.002;
    // Below the shared plane z = 0, u_x gains gamma * z: half its curl adds (0, gamma / 2, 0).
    let u: Vec<[f64; 3]> = mesh
        .nodes()
        .iter()
        .map(|p| [if p[2] < 0.0 { gamma * p[2] } else { 0.0 }, 0.0, 0.0])
        .collect();
    let rotations = nodal_rotations(&mesh, &u).expect("rotations");

    let on_shared_face = |p: [f64; 3]| p[2] == 0.0;
    // Weights are each element's volume; rotation is 0 in the top and gamma / 2 in the bottom.
    let want = (0.5 * gamma * volume_bottom) / (volume_top + volume_bottom);
    let mut checked = 0;
    for (p, rotation) in mesh.nodes().iter().zip(&rotations) {
        if on_shared_face(*p) {
            // Rounding of a sum of a few terms of size gamma: 1e-13 is about a thousand ulps of
            // the sum, far above that rounding and far below the difference a different weight
            // would make (which is of order gamma).
            assert!(
                (rotation[1] - want).abs() < 1e-13,
                "{} vs {want}",
                rotation[1]
            );
            checked += 1;
        }
    }
    assert!(checked >= 6, "the shared face has its nodes");
}

#[test]
fn a_field_of_the_wrong_length_is_refused_by_name() {
    let mesh = structured_mesh();
    let error = nodal_rotations(&mesh, &[[0.0; 3]; 3]).expect_err("wrong length");
    assert_eq!(
        error,
        ModelError::DisplacementFieldLength {
            nodes: mesh.nodes().len(),
            values: 3,
        }
    );
    assert!(error.to_string().contains("one vector per node"));
}
