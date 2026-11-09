// The multigrid setup's refusals and cancellation on hand-built block matrices: a wrong-length
// near-null space and a matrix without 3x3 blocks must be named errors, and a cancelled setup must
// stay cancelled.

use solidfem_solve::{
    CsrMatrix, MultigridError, MultigridOptions, MultigridSetup, NEAR_NULL_MODES,
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

fn modes_for(size: usize) -> Vec<f64> {
    (0..size * NEAR_NULL_MODES)
        .map(|i| i as f64 % 7.0 + 1.0)
        .collect()
}

#[test]
fn a_near_null_space_of_the_wrong_length_is_refused_by_name() {
    let matrix = block_chain(4);
    let wrong = modes_for(matrix.size() - 3);
    let result = MultigridSetup::new(&matrix, &wrong, MultigridOptions::default());
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
fn a_matrix_without_three_by_three_blocks_is_refused_by_name() {
    // Block size 1: the rows are single scalars, so the first node's rows do not group into 3.
    let links: Vec<[u32; 2]> = vec![[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]];
    let matrix = CsrMatrix::from_block_connectivity(6, 1, links.iter().map(|link| link.as_slice()));
    let near_null = vec![1.0; matrix.size() * NEAR_NULL_MODES];
    match MultigridSetup::new(&matrix, &near_null, MultigridOptions::default()) {
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
    let mut setup =
        MultigridSetup::new(&matrix, &modes, MultigridOptions::default()).expect("a valid setup");
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
