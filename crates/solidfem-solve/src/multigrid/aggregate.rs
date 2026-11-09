//! Strength of connection and the aggregation of block rows. A connection is strong when the
//! neighbour is near the row's closest neighbour, measured geometrically on the level's block
//! positions, so strength follows the mesh's aspect ratio and changes with neither units nor
//! material values. Aggregation makes three passes: a pass grouping whole untouched neighbourhoods
//! around seed rows, a pass joining leftovers to the group they are most strongly connected to,
//! and a pass grouping what is still left. Rows are visited in index order, so the aggregates
//! depend only on the matrix pattern and the positions.

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

/// Appends the strong neighbours of one row of `matrix` to `columns`: the stencil neighbours at
/// most `reach` times as far away as the row's closest stencil neighbour.
pub(super) fn strong_row(
    matrix: &BlockCsr,
    positions: &[[f64; 3]],
    reach: f64,
    row: usize,
    columns: &mut Vec<u32>,
) {
    let here = positions[row];
    let squared = |column: usize| {
        let there = positions[column];
        let (dx, dy, dz) = (there[0] - here[0], there[1] - here[1], there[2] - here[2]);
        dx * dx + dy * dy + dz * dz
    };
    let mut closest = f64::MAX;
    for at in matrix.row_range(row) {
        let column = matrix.columns[at] as usize;
        if column != row {
            closest = closest.min(squared(column));
        }
    }
    let limit = reach * reach * closest;
    for at in matrix.row_range(row) {
        let column = matrix.columns[at] as usize;
        if column != row && squared(column) <= limit {
            // Fits in u32: it came from a u32 column index.
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
