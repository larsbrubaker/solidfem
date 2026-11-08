// A rigid-body motion carries no strain and a linear displacement field carries exactly its own
// strain, in any valid element however bad its shape: the shape functions reproduce linear fields
// by construction. So on every distorted mesh of the measurement, at every element node and
// integration point, the production geometry and strain routines must return those answers. A
// failure here means a case in the table measured a broken element rather than a badly shaped one.

use solidfem_bench::sliver::{
    build, triangle_min_corner, Distorted, Extent, Location, Shape, Spec,
};
use solidfem_elastic::{point_geometry, strain_at, Quadrature, NODE_NATURAL};
use solidfem_mesh::tet_volume;

const ANGLES: [f64; 9] = [30.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.1];

fn all_specs() -> Vec<Spec> {
    let mut specs = Vec::new();
    for shape in [Shape::Cap, Shape::Needle] {
        for location in [Location::NearRoot, Location::NeutralAxis] {
            for extent in [Extent::Single, Extent::Cluster] {
                for target_degrees in ANGLES {
                    specs.push(Spec {
                        shape,
                        location,
                        extent,
                        target_degrees,
                    });
                }
            }
        }
    }
    specs
}

fn built(spec: Spec) -> Distorted {
    build(spec).unwrap_or_else(|error| panic!("{spec:?} could not be built: {error}"))
}

/// Checks `strain_at` against `expected` for the displacement field `u` on every element.
///
/// Tolerance, per element: rounding reaches the strain through the Jacobian's determinant and
/// cofactors, relative error about machine epsilon times `L^3 / det J` (the worst edge length
/// cubed over the determinant, the determinant's condition number), and through the sum of ten
/// products `u * gradient`. The bound is 128 machine epsilons (a generous constant for the
/// handful of operations each entry goes through) times that condition times the largest
/// displacement component times the largest gradient entry in the element.
fn check_field(distorted: &Distorted, u: impl Fn([f64; 3]) -> [f64; 3], expected: [f64; 6]) {
    let integration = Quadrature::default().points();
    let mesh = &distorted.quadratic;
    for (index, tet) in mesh.tets().iter().enumerate() {
        let nodes = tet.map(|n| mesh.nodes()[n as usize]);
        let displacement = nodes.map(&u);
        let mut largest_edge: f64 = 0.0;
        for a in 0..4 {
            for b in a + 1..4 {
                let d = [
                    nodes[a][0] - nodes[b][0],
                    nodes[a][1] - nodes[b][1],
                    nodes[a][2] - nodes[b][2],
                ];
                largest_edge = largest_edge.max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
            }
        }
        let naturals = NODE_NATURAL
            .iter()
            .copied()
            .chain(integration.iter().map(|q| q.natural));
        for natural in naturals {
            let geometry = point_geometry(&nodes, natural, 0)
                .unwrap_or_else(|fault| panic!("element {index} is not valid: {fault:?}"));
            let strain = strain_at(&geometry, &displacement);
            let largest_gradient = geometry
                .gradients
                .iter()
                .flatten()
                .fold(0.0f64, |m, g| m.max(g.abs()));
            let largest_displacement = displacement
                .iter()
                .flatten()
                .fold(0.0f64, |m, d| m.max(d.abs()));
            let condition = largest_edge.powi(3) / geometry.det_j;
            let tolerance = 128.0
                * f64::EPSILON
                * condition
                * largest_gradient.max(1.0)
                * largest_displacement.max(1.0);
            for (got, want) in strain.iter().zip(&expected) {
                assert!(
                    (got - want).abs() <= tolerance,
                    "element {index} at {natural:?}: strain {got} against {want}, tolerance {tolerance}"
                );
            }
        }
    }
}

#[test]
fn every_distorted_element_carries_no_strain_under_a_rigid_motion() {
    for spec in all_specs() {
        let distorted = built(spec);
        let translation = [0.3, -0.7, 1.1];
        let rotation = [0.02, -0.05, 0.03];
        check_field(
            &distorted,
            |p| {
                [
                    translation[0] + rotation[1] * p[2] - rotation[2] * p[1],
                    translation[1] + rotation[2] * p[0] - rotation[0] * p[2],
                    translation[2] + rotation[0] * p[1] - rotation[1] * p[0],
                ]
            },
            [0.0; 6],
        );
    }
}

#[test]
fn every_distorted_element_reproduces_a_linear_displacement_exactly() {
    let g = [[0.01, 0.02, -0.03], [0.04, -0.05, 0.06], [0.07, 0.08, 0.09]];
    let expected = [
        g[0][0],
        g[1][1],
        g[2][2],
        g[0][1] + g[1][0],
        g[1][2] + g[2][1],
        g[2][0] + g[0][2],
    ];
    for spec in all_specs() {
        let distorted = built(spec);
        check_field(
            &distorted,
            |p| {
                [
                    g[0][0] * p[0] + g[0][1] * p[1] + g[0][2] * p[2],
                    g[1][0] * p[0] + g[1][1] * p[1] + g[1][2] * p[2],
                    g[2][0] * p[0] + g[2][1] * p[1] + g[2][2] * p[2],
                ]
            },
            expected,
        );
    }
}

#[test]
fn every_built_triangle_has_the_asked_corner_and_lies_in_the_original_part() {
    // The cap's two small corners and the needle's far corner equal the target by construction,
    // within the rounding of a few trigonometric operations on these coordinates: 1e-9 degrees.
    const CORNER_TOLERANCE: f64 = 1e-9;
    for spec in all_specs() {
        let distorted = built(spec);
        let positions = distorted.linear.positions();
        let mut smallest = f64::INFINITY;
        for corners in &distorted.bad_triangles {
            let [a, b, c] = corners.map(|n| positions[n as usize]);
            smallest = smallest.min(triangle_min_corner(a, b, c));
        }
        assert!(
            (smallest - spec.target_degrees).abs() <= CORNER_TOLERANCE,
            "{spec:?}: smallest built corner {smallest}"
        );
        // Moving boundary nodes within their face keeps the bar's volume: 1 x 1 x 10.
        let volume: f64 = distorted
            .linear
            .tets()
            .iter()
            .map(|t| {
                let [a, b, c, d] = t.map(|n| positions[n as usize]);
                tet_volume(a, b, c, d)
            })
            .sum();
        assert!(
            (volume - 10.0).abs() < 1e-9,
            "{spec:?}: volume {volume} changed: a node moved off its face"
        );
    }
}
