// The patch test on a distorted mesh: a linear displacement field must give constant strain at
// every integration point, and the assembled stiffness must balance it to rounding at interior
// nodes. An element that fails this converges to the wrong answer however fine the mesh is.

use solidfem_elastic::{
    assemble_stiffness, point_geometry, strain_at, Material, Quadrature, NODE_NATURAL,
};
use solidfem_mesh::{box_tet_mesh, QuadraticTetMesh};

const GRADIENT: [[f64; 3]; 3] = [
    [0.01, 0.002, -0.003],
    [0.004, -0.005, 0.006],
    [0.007, 0.001, 0.008],
];
const OFFSET: [f64; 3] = [0.3, -0.2, 0.1];

fn distorted_mesh() -> QuadraticTetMesh {
    let size = 3.0;
    let cells = 3usize;
    let h = size / cells as f64;
    let linear = box_tet_mesh([0.0; 3], [size; 3], [cells; 3])
        .expect("box")
        .with_mapped_positions(|p| {
            if p.iter().any(|c| *c == 0.0 || *c == size) {
                return p;
            }
            // Interior vertices only, so the outer boundary stays planar and the check of
            // interior balance below has a clean definition of interior.
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

fn displacement(p: [f64; 3]) -> [f64; 3] {
    let mut u = OFFSET;
    for i in 0..3 {
        for j in 0..3 {
            u[i] += GRADIENT[i][j] * p[j];
        }
    }
    u
}

#[test]
fn a_linear_field_gives_constant_strain_on_a_distorted_mesh() {
    let mesh = distorted_mesh();
    let exact = [
        GRADIENT[0][0],
        GRADIENT[1][1],
        GRADIENT[2][2],
        GRADIENT[0][1] + GRADIENT[1][0],
        GRADIENT[1][2] + GRADIENT[2][1],
        GRADIENT[2][0] + GRADIENT[0][2],
    ];
    // Rounding: Jacobian inversion loses about its condition number (under 1e2 for this jitter)
    // times 2.2e-16 of the gradient, about 1e-14 relative, so 1e-11 relative to the largest strain
    // component is near machine precision with margin.
    let tolerance = 1e-11 * 0.01;
    for tet in mesh.tets() {
        let nodes = tet.map(|n| mesh.nodes()[n as usize]);
        let u = nodes.map(displacement);
        for natural in NODE_NATURAL {
            let geometry = point_geometry(&nodes, natural, 0).expect("element integrates");
            let strain = strain_at(&geometry, &u);
            for (got, want) in strain.iter().zip(&exact) {
                assert!((got - want).abs() < tolerance, "{got} vs {want}");
            }
        }
    }
}

#[test]
fn the_stiffness_balances_a_linear_field_at_interior_nodes() {
    let mesh = distorted_mesh();
    let material = Material {
        youngs_modulus: 2100.0,
        poisson_ratio: 0.3,
    };
    let matrix = assemble_stiffness(&mesh, &material, Quadrature::FourPoint, &mut || false)
        .expect("assembles");
    let mut u = vec![0.0; 3 * mesh.nodes().len()];
    for (node, p) in mesh.nodes().iter().enumerate() {
        u[3 * node..3 * node + 3].copy_from_slice(&displacement(*p));
    }
    let mut f = vec![0.0; u.len()];
    matrix.multiply(&u, &mut f);
    let scale = matrix.diagonal().iter().fold(0.0f64, |m, v| m.max(v.abs()))
        * u.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let interior = |p: [f64; 3]| p.iter().all(|c| *c > 1e-9 && *c < 3.0 - 1e-9);
    let mut checked = 0;
    for (node, p) in mesh.nodes().iter().enumerate() {
        if interior(*p) {
            for axis in 0..3 {
                // The internal forces are sums of about 150 terms of size `scale`, cancelling to
                // zero exactly in exact arithmetic: rounding leaves about 150 * 2.2e-16 * scale,
                // so 1e-12 * scale is near machine precision with margin.
                assert!(
                    f[3 * node + axis].abs() < 1e-12 * scale,
                    "residual {}",
                    f[3 * node + axis]
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "the mesh has interior nodes");
}
