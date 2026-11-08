// Integration rules checked against exact integrals of monomials, so a wrong table constant cannot
// pass: the reference tetrahedron integral of r^a s^b t^c is a! b! c! / (a+b+c+3)!, and the
// reference triangle integral of r^a s^b is a! b! / (a+b+2)!. Tolerances are 1e-13 of the
// integral's scale: the tables hold 15 to 17 digits and the sums are short.

use solidfem_elastic::{shape_derivatives, shape_values, von_mises, Quadrature, TRIANGLE_POINTS};

fn factorial(n: u32) -> f64 {
    (1..=n).map(f64::from).product()
}

fn assert_tet_rule_exact(rule: Quadrature, degree: u32) {
    for a in 0..=degree {
        for b in 0..=degree - a {
            for c in 0..=degree - a - b {
                let exact = factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3);
                let mut sum = 0.0;
                for q in rule.points() {
                    let [r, s, t] = q.natural;
                    sum += q.weight * r.powi(a as i32) * s.powi(b as i32) * t.powi(c as i32);
                }
                let integral = sum / 6.0;
                assert!(
                    (integral - exact).abs() < 1e-13,
                    "{rule:?} r^{a} s^{b} t^{c}: {integral} vs {exact}"
                );
            }
        }
    }
}

#[test]
fn the_4_point_rule_is_exact_to_degree_2() {
    assert_tet_rule_exact(Quadrature::FourPoint, 2);
}

#[test]
fn the_14_point_rule_is_exact_to_degree_5() {
    assert_tet_rule_exact(Quadrature::FourteenPoint, 5);
}

#[test]
fn the_triangle_rule_is_exact_to_degree_4() {
    for a in 0..=4u32 {
        for b in 0..=4 - a {
            let exact = factorial(a) * factorial(b) / factorial(a + b + 2);
            let mut sum = 0.0;
            for ([r, s], weight) in TRIANGLE_POINTS {
                sum += weight * r.powi(a as i32) * s.powi(b as i32);
            }
            let integral = sum / 2.0;
            assert!(
                (integral - exact).abs() < 1e-13,
                "r^{a} s^{b}: {integral} vs {exact}"
            );
        }
    }
}

#[test]
fn shape_functions_form_a_partition_of_unity() {
    for natural in [
        [0.1, 0.2, 0.3],
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [0.7, 0.1, 0.05],
    ] {
        let values = shape_values(natural);
        let sum: f64 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-15, "{natural:?}: {sum}");
        let derivatives = shape_derivatives(natural);
        for axis in 0..3 {
            let total: f64 = derivatives.iter().map(|d| d[axis]).sum();
            assert!(total.abs() < 1e-14, "{natural:?} axis {axis}: {total}");
        }
    }
}

#[test]
fn von_mises_matches_its_closed_forms() {
    // Uniaxial: |sigma|. Pure shear: sqrt(3) * tau. Hydrostatic: zero. Exact arithmetic on short
    // decimals, so 1e-12 is a rounding limit.
    assert!((von_mises(&[7.0, 0.0, 0.0, 0.0, 0.0, 0.0]) - 7.0).abs() < 1e-12);
    let shear = von_mises(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
    assert!((shear - 2.0 * 3.0f64.sqrt()).abs() < 1e-12);
    assert!(von_mises(&[5.0, 5.0, 5.0, 0.0, 0.0, 0.0]).abs() < 1e-12);
}
