//! Strength of connection and the aggregation of block rows. A connection is strong when the
//! neighbour is near the row's closest neighbour, measured geometrically on the level's block
//! positions, so strength follows the mesh's aspect ratio and changes with neither units nor
//! material values. The neighbours a coarse level's prolongator smoothing keeps are measured the
//! same way, from the widest spacing around the row ([`smoothing_row`]). Aggregation makes three
//! passes: a pass grouping whole untouched neighbourhoods
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

/// The squared distance between the positions of two block rows.
fn squared_distance(positions: &[[f64; 3]], row: usize, column: usize) -> f64 {
    let here = positions[row];
    let there = positions[column];
    let (dx, dy, dz) = (there[0] - here[0], there[1] - here[1], there[2] - here[2]);
    dx * dx + dy * dy + dz * dz
}

/// Appends to `columns` the stencil neighbours of `row` within squared distance `limit`.
fn push_within(
    matrix: &BlockCsr,
    positions: &[[f64; 3]],
    row: usize,
    limit: f64,
    columns: &mut Vec<u32>,
) {
    for at in matrix.row_range(row) {
        let column = matrix.columns[at] as usize;
        if column != row && squared_distance(positions, row, column) <= limit {
            // Fits in u32: it came from a u32 column index.
            columns.push(column as u32);
        }
    }
}

/// Appends the strong neighbours of one row of `matrix` to `columns`: the stencil neighbours at
/// most `reach` times as far away as the row's closest stencil neighbour. Returns that closest
/// squared distance, `f64::MAX` for a row with no stencil neighbour.
pub(super) fn strong_row(
    matrix: &BlockCsr,
    positions: &[[f64; 3]],
    reach: f64,
    row: usize,
    columns: &mut Vec<u32>,
) -> f64 {
    let mut closest = f64::MAX;
    for at in matrix.row_range(row) {
        let column = matrix.columns[at] as usize;
        if column != row {
            closest = closest.min(squared_distance(positions, row, column));
        }
    }
    push_within(matrix, positions, row, reach * reach * closest, columns);
    closest
}

/// Appends the neighbours a coarse level's prolongator smoothing keeps for `row`: the stencil
/// neighbours within `reach` times the widest closest-neighbour spacing among the row and its
/// stencil neighbours, `closest` holding each row's squared closest distance from
/// [`strong_row`]. A coarse level's block rows are stacks of nodes of unequal size, so a row's
/// closest neighbour can be the other half of its own wall, and a filter set by that distance
/// lumps away part of the row's in-plane couplings: it keeps 69% of the in-plane stencil entries
/// on the first coarse level of a 97,200-unknown two-layer plate, and 58% on the 100-cell
/// unaligned gate plate, where 10 of 10,841 rows lose all of them. `mod.rs` says why that
/// matters.
pub(super) fn smoothing_row(
    matrix: &BlockCsr,
    positions: &[[f64; 3]],
    reach: f64,
    closest: &[f64],
    row: usize,
    columns: &mut Vec<u32>,
) {
    let mut scale = closest[row];
    for at in matrix.row_range(row) {
        let column = matrix.columns[at] as usize;
        if column != row {
            scale = scale.max(closest[column]);
        }
    }
    push_within(matrix, positions, row, reach * reach * scale, columns);
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
