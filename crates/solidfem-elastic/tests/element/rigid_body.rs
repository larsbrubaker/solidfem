// The element stiffness has exactly six zero-energy modes: the rigid motions. More would be a
// mechanism that lets a part deform for free and under-report stress; fewer would mean a rigid
// motion costs energy and the element cannot represent a part simply being moved.

use solidfem_elastic::{element_stiffness, Material, Quadrature};

const MATERIAL: Material = Material {
    youngs_modulus: 2100.0,
    poisson_ratio: 0.3,
};

/// A well-shaped element with straight edges, not aligned with the axes.
fn straight_nodes() -> [[f64; 3]; 10] {
    with_midpoints([
        [0.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [0.2, 1.0, 0.1],
        [0.1, 0.3, 1.2],
    ])
}

fn with_midpoints(c: [[f64; 3]; 4]) -> [[f64; 3]; 10] {
    let mid = |a: usize, b: usize| {
        [
            0.5 * (c[a][0] + c[b][0]),
            0.5 * (c[a][1] + c[b][1]),
            0.5 * (c[a][2] + c[b][2]),
        ]
    };
    [
        c[0],
        c[1],
        c[2],
        c[3],
        mid(0, 1),
        mid(1, 2),
        mid(2, 0),
        mid(0, 3),
        mid(1, 3),
        mid(2, 3),
    ]
}

/// Eigenvalues of a symmetric matrix by cyclic Jacobi rotations, ascending.
fn eigenvalues(matrix: &[f64; 900]) -> Vec<f64> {
    let n = 30;
    let mut a = matrix.to_vec();
    for _sweep in 0..60 {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off == 0.0 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut values: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    values.sort_by(f64::total_cmp);
    values
}

/// Rounding in a 30 x 30 eigenproblem is about 30 * 2.2e-16 of the largest eigenvalue, so 1e-11
/// of it separates a true zero from rounding with four digits of margin. The element's softest
/// true deformation mode is a shape-and-Poisson-ratio constant of order 1e-2 to 1e-3 of the
/// stiffest for a well-shaped element, so 1e-5 cannot mistake a soft mode for a zero mode.
fn assert_six_zero_modes(values: &[f64]) {
    let largest = values[29];
    assert!(largest > 0.0);
    for (index, value) in values.iter().enumerate().take(6) {
        assert!(
            value.abs() < 1e-11 * largest,
            "mode {index}: {value}, largest {largest}: {values:?}"
        );
    }
    assert!(
        values[6] > 1e-5 * largest,
        "a seventh zero-energy mode: {values:?}"
    );
}

fn stiffness(nodes: &[[f64; 3]; 10], rule: Quadrature) -> Box<[f64; 900]> {
    let mut ke = Box::new([0.0; 900]);
    element_stiffness(nodes, &MATERIAL, rule, &mut ke).expect("element integrates");
    ke
}

#[test]
fn a_straight_element_has_exactly_six_zero_energy_modes_with_the_4_point_rule() {
    assert_six_zero_modes(&eigenvalues(&stiffness(
        &straight_nodes(),
        Quadrature::FourPoint,
    )));
}

#[test]
fn a_curved_element_keeps_six_zero_energy_modes_with_the_14_point_rule() {
    let mut nodes = straight_nodes();
    // Bow the edge midpoints: the rigid motions must stay zero-energy on curved geometry too.
    nodes[4][2] += 0.04;
    nodes[8][0] -= 0.05;
    assert_six_zero_modes(&eigenvalues(&stiffness(&nodes, Quadrature::FourteenPoint)));
}

#[test]
fn the_stiffness_is_exactly_symmetric() {
    let ke = stiffness(&straight_nodes(), Quadrature::FourPoint);
    for i in 0..30 {
        for j in 0..30 {
            assert_eq!(ke[i * 30 + j].to_bits(), ke[j * 30 + i].to_bits());
        }
    }
}

#[test]
fn both_rules_agree_on_a_straight_uniform_element() {
    // The integrand is quadratic, exact in both rules, so they differ only by rounding: about
    // 900 * 2.2e-16 relative to the largest entry, well under 1e-12.
    let four = stiffness(&straight_nodes(), Quadrature::FourPoint);
    let fourteen = stiffness(&straight_nodes(), Quadrature::FourteenPoint);
    let largest = four.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for (a, b) in four.iter().zip(fourteen.iter()) {
        assert!((a - b).abs() < 1e-12 * largest, "{a} vs {b}");
    }
}
