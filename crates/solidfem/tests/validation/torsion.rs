// A square bar in torsion against the Saint-Venant solution. The twist rate is measured away from
// both ends: the clamped root restrains warping and the tip torque is not the Saint-Venant shear
// distribution, and both effects decay exponentially along the bar.

use solidfem::elastic::{Material, Model};

use crate::support::{bar, faces_at, fix_root, nodes_at, report_case, run};

/// The torsion constant J = beta * a^4 of a square section from the Saint-Venant series.
fn saint_venant_beta() -> f64 {
    let pi = std::f64::consts::PI;
    let mut sum = 0.0;
    for n in 0..200 {
        let k = (2 * n + 1) as f64;
        sum += (k * pi / 2.0).tanh() / k.powi(5);
    }
    (1.0 - 192.0 / pi.powi(5) * sum) / 3.0
}

#[test]
fn the_twist_rate_matches_saint_venant() {
    // Limit 1%: end effects at 3 side lengths from each end decay below exp(-3 pi), about 1e-4,
    // and quadratic elements at four cells across capture the warping field to well under 1%.
    let material = Material {
        youngs_modulus: 2100.0,
        poisson_ratio: 0.3,
    };
    let g = material.youngs_modulus / (2.0 * (1.0 + material.poisson_ratio));
    let (length, side) = (10.0, 1.0);
    let center = 0.5 * side;
    let mesh = bar([length, side, side], [20, 4, 4]);
    let mut model = Model::new(mesh.clone(), material);
    fix_root(&mut model);

    // A traction circulating about the bar axis, per face; the delivered torque is summed from
    // what each face actually applies, so the reference torque is exact whatever the mesh.
    let strength = 1.0;
    let mut torque = 0.0;
    for face in faces_at(&mesh, length) {
        let [a, b, c] =
            [face.nodes[0], face.nodes[1], face.nodes[2]].map(|n| mesh.nodes()[n as usize]);
        let (cy, cz) = (
            (a[1] + b[1] + c[1]) / 3.0 - center,
            (a[2] + b[2] + c[2]) / 3.0 - center,
        );
        let traction = [0.0, -strength * cz, strength * cy];
        let (u, v) = (
            [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
            [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
        );
        let normal = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let area =
            0.5 * (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        torque += area * (cy * traction[2] - cz * traction[1]);
        model.add_traction(face, traction);
    }
    assert!(torque > 0.0);

    let solution = run(model);
    let twist_at = |x: f64| {
        let (mut numerator, mut denominator) = (0.0, 0.0);
        for node in nodes_at(&mesh, x) {
            let p = mesh.nodes()[node];
            let (y, z) = (p[1] - center, p[2] - center);
            let u = solution.displacements[node];
            numerator += y * u[2] - z * u[1];
            denominator += y * y + z * z;
        }
        numerator / denominator
    };
    let (x1, x2) = (3.0, 7.0);
    let rate = (twist_at(x2) - twist_at(x1)) / (x2 - x1);
    let expected = torque / (g * saint_venant_beta() * side.powi(4));
    let error = ((rate - expected) / expected).abs();
    report_case("torsion_20x4x4", error * 100.0, &solution);
    assert!(
        error < 0.01,
        "twist rate {rate}, expected {expected}, error {error}"
    );
}
