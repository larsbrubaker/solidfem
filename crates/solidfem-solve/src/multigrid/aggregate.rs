//! Strength of connection and the aggregation of block rows. Connections are strong when the
//! block norm is large against the geometric mean of the two diagonal block norms, a test that
//! does not change with units or element size. Aggregation makes three passes: a pass grouping
//! whole untouched neighbourhoods around seed rows, a pass joining leftovers to the group they are
//! most strongly connected to, and a pass grouping what is still left. Rows are visited in index
//! order, so the aggregates depend only on the matrix.

use super::block::BlockCsr;

/// Not yet in any aggregate.
pub(super) const NO_AGGREGATE: u32 = u32::MAX;

/// The Frobenius norm of a block.
pub(super) fn block_norm(block: &[f64]) -> f64 {
    let mut sum = 0.0;
    for v in block {
        sum += v * v;
    }
    sum.sqrt()
}

/// Appends the strong neighbours of one row of `matrix` to `columns`.
pub(super) fn strong_row(
    matrix: &BlockCsr,
    diagonal_norms: &[f64],
    threshold: f64,
    row: usize,
    columns: &mut Vec<u32>,
) {
    for at in matrix.row_range(row) {
        let column = matrix.columns[at] as usize;
        if column == row {
            continue;
        }
        let scale = (diagonal_norms[row] * diagonal_norms[column]).sqrt();
        if block_norm(matrix.block(at)) >= threshold * scale {
            columns.push(column as u32);
        }
    }
}

/// The aggregation state across passes.
pub(super) struct Aggregation {
    /// The aggregate of each row, or [`NO_AGGREGATE`].
    pub aggregate: Vec<u32>,
    /// True for rows placed in the first pass.
    pub seeded: Vec<bool>,
    /// Aggregates made so far.
    pub count: u32,
}

impl Aggregation {
    /// All rows free.
    pub fn new(rows: usize) -> Self {
        Aggregation {
            aggregate: vec![NO_AGGREGATE; rows],
            seeded: vec![false; rows],
            count: 0,
        }
    }

    /// First pass: `row` seeds an aggregate when it and all its strong neighbours are free.
    pub fn seed(&mut self, row: usize, neighbours: &[u32]) {
        if self.aggregate[row] != NO_AGGREGATE
            || neighbours.is_empty()
            || neighbours
                .iter()
                .any(|n| self.aggregate[*n as usize] != NO_AGGREGATE)
        {
            return;
        }
        let id = self.count;
        self.count += 1;
        self.aggregate[row] = id;
        self.seeded[row] = true;
        for n in neighbours {
            self.aggregate[*n as usize] = id;
            self.seeded[*n as usize] = true;
        }
    }

    /// Second pass: a free `row` joins the aggregate of its strongest seeded neighbour; the lowest
    /// column wins ties because columns are visited ascending and only a strictly larger value
    /// replaces the choice.
    pub fn join(&mut self, row: usize, matrix: &BlockCsr, neighbours: &[u32]) {
        if self.aggregate[row] != NO_AGGREGATE {
            return;
        }
        let mut best = 0.0;
        let mut chosen = NO_AGGREGATE;
        for at in matrix.row_range(row) {
            let column = matrix.columns[at];
            if neighbours.binary_search(&column).is_err() || !self.seeded[column as usize] {
                continue;
            }
            let norm = block_norm(matrix.block(at));
            if norm > best {
                best = norm;
                chosen = self.aggregate[column as usize];
            }
        }
        if chosen != NO_AGGREGATE {
            self.aggregate[row] = chosen;
        }
    }

    /// Third pass: a free `row` starts an aggregate with its free strong neighbours; alone if it
    /// has none, so every row ends up in an aggregate.
    pub fn sweep(&mut self, row: usize, neighbours: &[u32]) {
        if self.aggregate[row] != NO_AGGREGATE {
            return;
        }
        let id = self.count;
        self.count += 1;
        self.aggregate[row] = id;
        for n in neighbours {
            if self.aggregate[*n as usize] == NO_AGGREGATE {
                self.aggregate[*n as usize] = id;
            }
        }
    }
}
