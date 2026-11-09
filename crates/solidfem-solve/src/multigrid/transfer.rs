//! The transfer operator between two levels, built row by row: the tentative prolongator
//! orthonormalizes the near-null space over each aggregate, smoothing widens it by one application
//! of the block-Jacobi scaled matrix so the coarse level's energy falls, and the Galerkin product
//! `P^T A P` forms the next level. Each function does one row so `setup.rs` can pause between rows.

use super::block::{add_product, Accumulator, BlockCsr};
use super::NEAR_NULL_MODES as MODES;

/// A remainder below this fraction of a near-null column's size is rounding, not a new direction
/// in the aggregate: the modes are exact to machine precision, so the remainder of a true
/// dependency sits many digits below the column size.
const RANK_TOLERANCE: f64 = 1e-10;

/// Orthonormalizes the near-null space over one aggregate, appending the tentative prolongator
/// blocks of its members to `blocks` (`row_block` by 6, row-major, members in order) and writing
/// the coarse near-null block to `coarse` (6x6, row-major). `members` lists the aggregate's rows,
/// `near_null` holds six values per scalar row. A column that adds no new direction becomes a
/// zero column with a zero coarse row, so that coarse value can never be excited.
pub(super) fn orthonormalize_aggregate(
    members: &[u32],
    row_block: usize,
    near_null: &[f64],
    blocks: &mut Vec<f64>,
    coarse: &mut [f64; 36],
) {
    let rows = members.len() * row_block;
    let mut local = vec![0.0; rows * MODES];
    for (m, member) in members.iter().enumerate() {
        let source = &near_null[*member as usize * row_block * MODES..];
        for p in 0..row_block {
            local[(m * row_block + p) * MODES..(m * row_block + p + 1) * MODES]
                .copy_from_slice(&source[p * MODES..(p + 1) * MODES]);
        }
    }
    coarse.fill(0.0);
    for column in 0..MODES {
        let original = column_norm(&local, rows, column);
        // Two passes of Gram-Schmidt: one pass leaves a remainder that is no longer orthogonal
        // once most of the column has cancelled, the normal case for a rotation column on a
        // small aggregate.
        for _ in 0..2 {
            for previous in 0..column {
                let mut projection = 0.0;
                for row in 0..rows {
                    projection += local[row * MODES + previous] * local[row * MODES + column];
                }
                coarse[previous * MODES + column] += projection;
                for row in 0..rows {
                    let value = local[row * MODES + previous];
                    local[row * MODES + column] -= projection * value;
                }
            }
        }
        let remainder = column_norm(&local, rows, column);
        if original == 0.0 || remainder <= RANK_TOLERANCE * original {
            for row in 0..rows {
                local[row * MODES + column] = 0.0;
            }
            coarse[column * MODES + column] = 0.0;
        } else {
            for row in 0..rows {
                local[row * MODES + column] /= remainder;
            }
            coarse[column * MODES + column] = remainder;
        }
    }
    blocks.extend_from_slice(&local);
}

fn column_norm(local: &[f64], rows: usize, column: usize) -> f64 {
    let mut sum = 0.0;
    for row in 0..rows {
        let value = local[row * MODES + column];
        sum += value * value;
    }
    sum.sqrt()
}

/// Adds row `row` of `matrix * right` to `accumulator`, whose blocks are `row_block` by
/// `right.col_block`. Smoothing and the operator product share this.
pub(super) fn accumulate_product_row(
    matrix: &BlockCsr,
    row: usize,
    right: &BlockCsr,
    accumulator: &mut Accumulator,
) {
    let (rb, mid, cb) = (matrix.row_block, matrix.col_block, right.col_block);
    for at in matrix.row_range(row) {
        let column = matrix.columns[at] as usize;
        for inner in right.row_range(column) {
            let out = accumulator.entry(right.columns[inner]);
            add_product(out, matrix.block(at), rb, mid, right.block(inner), cb);
        }
    }
}

/// Appends row `row` of the smoothed prolongator `P = T - omega D^-1 A_f T`, with `A_f` the
/// filtered matrix: couplings outside the strong neighbours `neighbours` (sorted) are added to the
/// diagonal instead of spreading the coarse function along them. The filtering keeps the
/// prolongator as narrow as the strong stencil, which is what keeps the coarse operator small
/// when weak couplings would otherwise fill it in. Lumping onto the diagonal keeps the row's
/// action on a translation. Only the prolongator sees this matrix: the coarse operator is the
/// Galerkin product with the true matrix and the smoother uses the true matrix, so the
/// preconditioner stays symmetric positive definite.
#[allow(clippy::too_many_arguments)] // One row of one product; a struct would only rename these.
pub(super) fn smooth_row(
    matrix: &BlockCsr,
    tentative: &BlockCsr,
    neighbours: &[u32],
    inverse_diagonal: &[f64],
    omega: f64,
    row: usize,
    accumulator: &mut Accumulator,
    output: &mut BlockCsr,
) {
    let rb = matrix.row_block;
    let cb = tentative.col_block;
    accumulator.begin();
    // The tentative block's column is always in the row, even if the product leaves it zero.
    let t_range = tentative.row_range(row);
    let t_at = t_range.start;
    let t_column = tentative.columns[t_at];
    accumulator.entry(t_column);
    let mid = matrix.col_block;
    for at in matrix.row_range(row) {
        let column = matrix.columns[at];
        let source = if column as usize == row || neighbours.binary_search(&column).is_ok() {
            column as usize
        } else {
            row
        };
        for inner in tentative.row_range(source) {
            let out = accumulator.entry(tentative.columns[inner]);
            add_product(out, matrix.block(at), rb, mid, tentative.block(inner), cb);
        }
    }
    let dinv = &inverse_diagonal[row * rb * rb..(row + 1) * rb * rb];
    let mut scaled = vec![0.0; rb * cb];
    let t_block = tentative.block(t_at);
    accumulator.drain_sorted(|column, block| {
        scaled.fill(0.0);
        add_product(&mut scaled, dinv, rb, rb, block, cb);
        output.columns.push(column);
        for at in 0..rb * cb {
            let t_value = if column == t_column { t_block[at] } else { 0.0 };
            output.values.push(t_value - omega * scaled[at]);
        }
    });
    output.starts.push(output.columns.len());
}

/// Appends row `row` of `A P`.
pub(super) fn product_row(
    matrix: &BlockCsr,
    prolongator: &BlockCsr,
    row: usize,
    accumulator: &mut Accumulator,
    output: &mut BlockCsr,
) {
    accumulator.begin();
    accumulate_product_row(matrix, row, prolongator, accumulator);
    accumulator.drain_sorted(|column, block| {
        output.columns.push(column);
        output.values.extend_from_slice(block);
    });
    output.starts.push(output.columns.len());
}

/// For each coarse column, where `prolongator` stores it: the fine row and block index, fine rows
/// ascending so the Galerkin sum's order is fixed by the data.
pub(super) fn prolongator_columns(
    prolongator: &BlockCsr,
    coarse_rows: usize,
) -> (Vec<usize>, Vec<(u32, usize)>) {
    let mut starts = vec![0usize; coarse_rows + 1];
    for column in &prolongator.columns {
        starts[*column as usize + 1] += 1;
    }
    for i in 0..coarse_rows {
        starts[i + 1] += starts[i];
    }
    let mut filled = starts.clone();
    let mut entries = vec![(0u32, 0usize); prolongator.columns.len()];
    for row in 0..prolongator.rows {
        for at in prolongator.row_range(row) {
            let column = prolongator.columns[at] as usize;
            entries[filled[column]] = (row as u32, at);
            filled[column] += 1;
        }
    }
    (starts, entries)
}

/// The prolongator as columns, the way the Galerkin product reads it.
pub(super) struct ColumnView<'a> {
    pub prolongator: &'a BlockCsr,
    pub starts: &'a [usize],
    pub entries: &'a [(u32, usize)],
}

/// Appends row `coarse_row` of `P^T (A P)`, using `view` for the columns of `P`.
pub(super) fn galerkin_row(
    view: &ColumnView<'_>,
    a_times_p: &BlockCsr,
    coarse_row: usize,
    accumulator: &mut Accumulator,
    output: &mut BlockCsr,
) {
    let (fine, coarse) = (view.prolongator.row_block, view.prolongator.col_block);
    accumulator.begin();
    let mut transposed = vec![0.0; fine * coarse];
    for (fine_row, at) in &view.entries[view.starts[coarse_row]..view.starts[coarse_row + 1]] {
        let block = view.prolongator.block(*at);
        for p in 0..fine {
            for q in 0..coarse {
                transposed[q * fine + p] = block[p * coarse + q];
            }
        }
        for inner in a_times_p.row_range(*fine_row as usize) {
            let out = accumulator.entry(a_times_p.columns[inner]);
            add_product(
                out,
                &transposed,
                coarse,
                fine,
                a_times_p.block(inner),
                coarse,
            );
        }
    }
    accumulator.drain_sorted(|column, block| {
        output.columns.push(column);
        output.values.extend_from_slice(block);
    });
    output.starts.push(output.columns.len());
}
