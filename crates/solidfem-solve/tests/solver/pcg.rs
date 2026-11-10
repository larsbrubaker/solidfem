// The conjugate gradient solve on a 1D Laplacian, whose exact solution is known in closed form:
// the tridiagonal [-1, 2, -1] system with a unit load on the last row has x[i] = (i + 1) / (n + 1).
// Pins convergence, slicing, cancel and refusal.

use solidfem_solve::{CsrMatrix, Jacobi, Pcg, PcgError, PcgOptions, StepOutcome};

/// The tridiagonal [-1, 2, -1] matrix of size `n`, built through element connectivity.
fn laplacian(n: usize) -> CsrMatrix {
    let elements: Vec<[u32; 2]> = (0..n as u32 - 1).map(|i| [i, i + 1]).collect();
    let mut matrix =
        CsrMatrix::from_block_connectivity(n, 1, elements.iter().map(|e| e.as_slice()));
    for i in 0..n {
        matrix.add(i, i as u32, 2.0);
        if i + 1 < n {
            matrix.add(i, i as u32 + 1, -1.0);
            matrix.add(i + 1, i as u32, -1.0);
        }
    }
    matrix
}

fn solver(n: usize) -> Pcg<Jacobi> {
    let matrix = laplacian(n);
    let jacobi = Jacobi::new(&matrix).expect("positive diagonal");
    let mut rhs = vec![0.0; n];
    rhs[n - 1] = 1.0;
    Pcg::new(matrix, jacobi, rhs, PcgOptions::default())
}

fn never() -> impl FnMut() -> bool {
    || false
}

/// A longer output would keep stale values past the product's end and read as a result.
#[test]
#[should_panic(expected = "the product has 4 rows and the output holds 5")]
fn a_matrix_product_refuses_an_output_longer_than_the_matrix() {
    let matrix = laplacian(4);
    let mut output = vec![0.0; 5];
    matrix.multiply(&[1.0; 4], &mut output);
}

#[test]
fn it_solves_a_system_with_a_known_solution() {
    let n = 50;
    let mut pcg = solver(n);
    let outcome = pcg.step(10_000, &mut never()).expect("solves");
    assert!(matches!(outcome, StepOutcome::Converged(_)));
    // Laplacian with unit load at the end: x[i] = (i + 1) / (n + 1).
    for (i, x) in pcg.estimate().iter().enumerate() {
        let exact = (i + 1) as f64 / (n + 1) as f64;
        assert!((x - exact).abs() < 1e-8, "x[{i}] = {x}, exact {exact}");
    }
}

#[test]
fn many_small_steps_give_identical_bits_to_one_long_run() {
    let n = 80;
    let mut whole = solver(n);
    whole.step(100_000, &mut never()).expect("solves");
    let mut sliced = solver(n);
    loop {
        match sliced.step(3, &mut never()).expect("solves") {
            StepOutcome::Running(progress) => assert!(progress.iteration > 0),
            StepOutcome::Converged(_) => break,
            StepOutcome::Cancelled(_) => panic!("nothing cancelled this solve"),
        }
    }
    for (a, b) in whole.estimate().iter().zip(sliced.estimate()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn cancel_stops_the_solve_and_it_stays_stopped() {
    let mut pcg = solver(50);
    pcg.step(3, &mut never()).expect("runs");
    let outcome = pcg.step(1000, &mut || true).expect("cancels");
    let StepOutcome::Cancelled(progress) = outcome else {
        panic!("expected cancelled, got {outcome:?}");
    };
    assert_eq!(progress.iteration, 3);
    assert!(matches!(
        pcg.step(1000, &mut never()).expect("still cancelled"),
        StepOutcome::Cancelled(_)
    ));
}

#[test]
fn an_indefinite_matrix_returns_a_specific_error() {
    let mut matrix = CsrMatrix::from_block_connectivity(2, 1, [[0u32, 1].as_slice()]);
    // Positive diagonal but indefinite: eigenvalues 1 +/- 3.
    matrix.add(0, 0, 1.0);
    matrix.add(1, 1, 1.0);
    matrix.add(0, 1, 3.0);
    matrix.add(1, 0, 3.0);
    let jacobi = Jacobi::new(&matrix).expect("positive diagonal");
    let mut pcg = Pcg::new(matrix, jacobi, vec![1.0, -1.0], PcgOptions::default());
    let error = pcg.step(100, &mut never()).expect_err("indefinite");
    assert!(matches!(error, PcgError::NotPositiveDefinite { .. }));
}

#[test]
fn a_tolerance_below_rounding_stops_at_the_floor_and_says_so() {
    // 1e-18 is below f64's epsilon relative to ||b||, so no iteration can reach it; the solve must
    // stop at the rounding floor of A x with an honest report, not run out its iteration cap.
    let n = 50;
    let matrix = laplacian(n);
    let jacobi = Jacobi::new(&matrix).expect("positive diagonal");
    let mut rhs = vec![0.0; n];
    rhs[n - 1] = 1.0;
    let options = PcgOptions {
        tolerance: 1e-18,
        ..PcgOptions::default()
    };
    let mut pcg = Pcg::new(matrix, jacobi, rhs, options);
    let StepOutcome::Converged(progress) = pcg.step(100_000, &mut never()).expect("stops") else {
        panic!("expected a converged solve");
    };
    assert!(progress.at_rounding_floor);
    assert!(progress.relative_residual > 1e-18);
    for (i, x) in pcg.estimate().iter().enumerate() {
        let exact = (i + 1) as f64 / (n + 1) as f64;
        assert!((x - exact).abs() < 1e-10, "x[{i}] = {x}, exact {exact}");
    }
}
