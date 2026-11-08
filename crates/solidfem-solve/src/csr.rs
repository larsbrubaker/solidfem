//! Compressed sparse row storage for symmetric block-structured matrices. The sparsity pattern
//! comes from element connectivity over blocks (mesh nodes), which keeps symbolic work to a sort
//! over block pairs rather than over every degree of freedom. Rows hold sorted columns so
//! lookups can binary search and output order is fixed by the data, not by the build.

/// A square sparse matrix in compressed sparse row form.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    size: usize,
    row_starts: Vec<usize>,
    columns: Vec<u32>,
    values: Vec<f64>,
}

impl CsrMatrix {
    /// Builds the zero matrix with the pattern of element connectivity over `block_count` blocks
    /// of `block_size` degrees of freedom. Every element couples all of its blocks with each
    /// other. Elements name blocks by index; a block index must be below `block_count`.
    pub fn from_block_connectivity<'a>(
        block_count: usize,
        block_size: usize,
        elements: impl IntoIterator<Item = &'a [u32]>,
    ) -> Self {
        // Sorting one packed key per coupled pair is cheap and deterministic: the keys are unique
        // once deduplicated, so sort stability cannot reach the output.
        let mut pairs: Vec<u64> = Vec::new();
        for element in elements {
            for &row in element {
                for &column in element {
                    pairs.push(((row as u64) << 32) | u64::from(column));
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();

        let size = block_count * block_size;
        let mut block_rows = vec![0usize; block_count + 1];
        for pair in &pairs {
            block_rows[(pair >> 32) as usize + 1] += 1;
        }
        for i in 0..block_count {
            block_rows[i + 1] += block_rows[i];
        }

        let total = pairs.len() * block_size * block_size;
        let mut row_starts = Vec::with_capacity(size + 1);
        let mut columns: Vec<u32> = Vec::with_capacity(total);
        row_starts.push(0);
        for block in 0..block_count {
            for _ in 0..block_size {
                for pair in &pairs[block_rows[block]..block_rows[block + 1]] {
                    let column_block = (*pair & 0xffff_ffff) as usize;
                    for offset in 0..block_size {
                        // Fits in u32: blocks are u32 and a node holds three components, so the
                        // degree of freedom index stays well under the block limit for any mesh
                        // that fits in memory.
                        columns.push((column_block * block_size + offset) as u32);
                    }
                }
                row_starts.push(columns.len());
            }
        }
        let values = vec![0.0; columns.len()];
        CsrMatrix {
            size,
            row_starts,
            columns,
            values,
        }
    }

    /// The matrix's row and column count.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The number of stored entries.
    pub fn nonzeros(&self) -> usize {
        self.columns.len()
    }

    /// Adds `value` at `(row, column)`, which the pattern must contain.
    ///
    /// # Panics
    /// When `(row, column)` is outside the pattern: assembly only writes where connectivity put
    /// an entry, so a miss is a programming error here, not a user input.
    #[inline]
    pub fn add(&mut self, row: usize, column: u32, value: f64) {
        let range = self.row_starts[row]..self.row_starts[row + 1];
        let slice = &self.columns[range.clone()];
        let at = slice
            .binary_search(&column)
            .expect("assembly writes only inside the element connectivity pattern");
        self.values[range.start + at] += value;
    }

    /// The value stored at `(row, column)`, zero outside the pattern.
    pub fn get(&self, row: usize, column: u32) -> f64 {
        let range = self.row_starts[row]..self.row_starts[row + 1];
        match self.columns[range.clone()].binary_search(&column) {
            Ok(at) => self.values[range.start + at],
            Err(_) => 0.0,
        }
    }

    /// Writes `self * x` into `y`. Rows sum in column order, so the rounding of the result
    /// depends only on the data.
    pub fn multiply(&self, x: &[f64], y: &mut [f64]) {
        for (row, out) in y.iter_mut().enumerate() {
            let mut sum = 0.0;
            for at in self.row_starts[row]..self.row_starts[row + 1] {
                sum += self.values[at] * x[self.columns[at] as usize];
            }
            *out = sum;
        }
    }

    /// The main diagonal.
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.size)
            .map(|row| self.get(row, row as u32))
            .collect()
    }
}
