// The multigrid setup's refusals and cancellation on hand-built block matrices: a wrong-length
// near-null space and a matrix without 3x3 blocks must be named errors, and a cancelled setup must
// stay cancelled. The f32 cycle gives the same bits however setup is sliced and stays within
// rounding of the f64 cycle.

use solidfem_solve::{
    CsrMatrix, MultigridError, MultigridOptions, MultigridSetup, Precision, Preconditioner,
    SetupStatus, NEAR_NULL_MODES,
};

/// A chain of nodes with symmetric positive definite 3x3 couplings, `3 * nodes` unknowns.
fn block_chain(nodes: usize) -> CsrMatrix {
    let links: Vec<[u32; 2]> = (0..nodes as u32 - 1).map(|i| [i, i + 1]).collect();
    let mut matrix =
        CsrMatrix::from_block_connectivity(nodes, 3, links.iter().map(|link| link.as_slice()));
    for node in 0..nodes {
        for axis in 0..3 {
            let row = 3 * node + axis;
            matrix.add(row, row as u32, 4.0);
            if node > 0 {
                matrix.add(row, (row - 3) as u32, -1.0);
            }
            if node + 1 < nodes {
                matrix.add(row, (row + 3) as u32, -1.0);
            }
        }
    }
    matrix
}

fn positions_for(nodes: usize) -> Vec<[f64; 3]> {
    (0..nodes).map(|i| [i as f64, 0.0, 0.0]).collect()
}

fn modes_for(size: usize) -> Vec<f64> {
    (0..size * NEAR_NULL_MODES)
        .map(|i| i as f64 % 7.0 + 1.0)
        .collect()
}

#[test]
fn a_near_null_space_of_the_wrong_length_is_refused_by_name() {
    let matrix = block_chain(4);
    let wrong = modes_for(matrix.size() - 3);
    let result = MultigridSetup::new(
        &matrix,
        &wrong,
        &positions_for(4),
        MultigridOptions::default(),
    );
    match result {
        Err(MultigridError::NearNullLength { expected, found }) => {
            assert_eq!(expected, NEAR_NULL_MODES * matrix.size());
            assert_eq!(found, wrong.len());
        }
        other => panic!(
            "expected a NearNullLength error, got {:?}",
            other.map(|_| ())
        ),
    }
}

#[test]
fn positions_of_the_wrong_length_are_refused_by_name() {
    let matrix = block_chain(4);
    let modes = modes_for(matrix.size());
    let result = MultigridSetup::new(
        &matrix,
        &modes,
        &positions_for(3),
        MultigridOptions::default(),
    );
    match result {
        Err(MultigridError::PositionLength { expected, found }) => {
            assert_eq!((expected, found), (4, 3));
        }
        other => panic!(
            "expected a PositionLength error, got {:?}",
            other.map(|_| ())
        ),
    }
}

#[test]
fn a_matrix_without_three_by_three_blocks_is_refused_by_name() {
    // Block size 1: the rows are single scalars, so the first node's rows do not group into 3.
    let links: Vec<[u32; 2]> = vec![[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]];
    let matrix = CsrMatrix::from_block_connectivity(6, 1, links.iter().map(|link| link.as_slice()));
    let near_null = vec![1.0; matrix.size() * NEAR_NULL_MODES];
    match MultigridSetup::new(
        &matrix,
        &near_null,
        &positions_for(2),
        MultigridOptions::default(),
    ) {
        Err(MultigridError::BlockStructure { .. }) => {}
        other => panic!(
            "expected a BlockStructure error, got {:?}",
            other.map(|_| ())
        ),
    }
}

#[test]
fn a_cancelled_setup_reports_cancelled_and_stays_cancelled() {
    let matrix = block_chain(4000);
    let modes = modes_for(matrix.size());
    let mut setup = MultigridSetup::new(
        &matrix,
        &modes,
        &positions_for(4000),
        MultigridOptions::default(),
    )
    .expect("a valid setup");
    let mut polls = 0usize;
    let result = setup.step(usize::MAX, &mut || {
        polls += 1;
        polls > 10
    });
    assert_eq!(result, Err(MultigridError::Cancelled));
    assert!(polls >= 11, "the cancel check was read {polls} times");
    assert_eq!(
        setup.step(usize::MAX, &mut || false),
        Err(MultigridError::Cancelled)
    );
    assert!(matches!(setup.finish(), Err(MultigridError::NotFinished)));
}

/// The finished preconditioner applied to a fixed vector, after setup in slices of `slice` rows.
fn applied(precision: Precision, slice: usize) -> Vec<f64> {
    let nodes = 3000;
    let matrix = block_chain(nodes);
    let modes = modes_for(matrix.size());
    let options = MultigridOptions {
        coarse_dofs: 60,
        precision,
        ..MultigridOptions::default()
    };
    let mut setup =
        MultigridSetup::new(&matrix, &modes, &positions_for(nodes), options).expect("valid");
    while setup.step(slice, &mut || false).expect("builds") == SetupStatus::Running {}
    let multigrid = setup.finish().expect("finished");
    assert_eq!(multigrid.precision(), precision);
    let residual: Vec<f64> = (0..matrix.size())
        .map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0)
        .collect();
    let mut first = vec![0.0; matrix.size()];
    let mut second = vec![0.0; matrix.size()];
    multigrid.apply(&residual, &mut first);
    multigrid.apply(&residual, &mut second);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&first), bits(&second), "a second application differs");
    first
}

#[test]
fn an_f32_cycle_is_deterministic_and_independent_of_setup_slicing() {
    // An f32 cycle is a different operator from the f64 one only by rounding: each output is
    // within a relative 1e-3 of the f64 cycle's, against the largest. The chain's cycle runs about
    // fifty f32 operations per output, each 6e-8 relative, on a block-Jacobi scaled operator whose
    // Chebyshev polynomial amplifies a rounding error by at most its degree times the range of 30.
    let double = applied(Precision::Double, usize::MAX);
    let largest = double.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for precision in [Precision::SingleStorage, Precision::Single] {
        let whole = applied(precision, usize::MAX);
        assert!(whole.iter().any(|v| *v != 0.0));
        for slice in [1usize, 13, 977] {
            assert_eq!(
                bits(&applied(precision, slice)),
                bits(&whole),
                "{precision:?} slice {slice}"
            );
        }
        for (a, b) in double.iter().zip(&whole) {
            assert!(
                (a - b).abs() <= 1e-3 * largest,
                "{precision:?}: {a} against {b}"
            );
        }
    }
}
