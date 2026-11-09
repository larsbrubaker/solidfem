//! The polynomial a node's stress is read from: a linear least-squares fit of the stress tensor,
//! one component at a time, to integration points around a mesh vertex. `results.rs` builds one
//! per vertex after the element loop and says why a patch is the right shape for the nodal
//! stress. A linear fit is what a 10-node element's stress can support, since its stress is
//! linear inside each element; a fit of higher degree only amplifies the differences between
//! neighbouring elements, and it is determined by any patch that holds one non-flat element, so a
//! one-sided patch at a boundary needs no special case.

/// Monomials of the fit: the constant term and the three offsets from the patch centre.
const TERMS: usize = 4;

/// The largest offset of a patch's points is scaled to one, so the normal matrix is well formed
/// whatever the element size; a pivot this small then means the patch's points are coplanar.
const SINGULAR_PIVOT: f64 = 1e-12;

/// A fitted stress field around a patch's vertex.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PatchFit {
    centre: [f64; 3],
    scale: f64,
    /// Per term, the six stress components of that term's coefficient.
    coefficients: [[f64; 6]; TERMS],
}

fn offsets(point: [f64; 3], centre: [f64; 3], scale: f64) -> [f64; TERMS] {
    [
        1.0,
        (point[0] - centre[0]) / scale,
        (point[1] - centre[1]) / scale,
        (point[2] - centre[2]) / scale,
    ]
}

impl PatchFit {
    /// Fits `samples` (position, tensor), in the order given, about `centre`. `None` when the
    /// points do not determine a linear field: no points, or every point in one plane.
    pub(crate) fn fit(centre: [f64; 3], samples: &[([f64; 3], [f64; 6])]) -> Option<PatchFit> {
        let mut scale = 0.0f64;
        for (position, _) in samples {
            for axis in 0..3 {
                scale = scale.max((position[axis] - centre[axis]).abs());
            }
        }
        if samples.is_empty() || scale == 0.0 {
            return None;
        }
        let mut matrix = [[0.0; TERMS]; TERMS];
        let mut rhs = [[0.0; 6]; TERMS];
        for (position, stress) in samples {
            let terms = offsets(*position, centre, scale);
            for i in 0..TERMS {
                for j in 0..TERMS {
                    matrix[i][j] += terms[i] * terms[j];
                }
                for (component, value) in stress.iter().enumerate() {
                    rhs[i][component] += terms[i] * value;
                }
            }
        }
        let coefficients = solve(matrix, rhs)?;
        Some(PatchFit {
            centre,
            scale,
            coefficients,
        })
    }

    /// The fitted tensor at `point`.
    pub(crate) fn at(&self, point: [f64; 3]) -> [f64; 6] {
        let terms = offsets(point, self.centre, self.scale);
        let mut value = [0.0; 6];
        for (term, coefficient) in terms.iter().zip(&self.coefficients) {
            for (out, c) in value.iter_mut().zip(coefficient) {
                *out += term * c;
            }
        }
        value
    }
}

/// Gaussian elimination with partial pivoting, the same pivot order for every right-hand side, so
/// a fit is the same bits on every run.
fn solve(
    mut matrix: [[f64; TERMS]; TERMS],
    mut rhs: [[f64; 6]; TERMS],
) -> Option<[[f64; 6]; TERMS]> {
    for column in 0..TERMS {
        let mut pivot = column;
        for row in column + 1..TERMS {
            if matrix[row][column].abs() > matrix[pivot][column].abs() {
                pivot = row;
            }
        }
        if matrix[pivot][column].abs() <= SINGULAR_PIVOT {
            return None;
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let (pivot_row, pivot_rhs) = (matrix[column], rhs[column]);
        for row in column + 1..TERMS {
            let factor = matrix[row][column] / pivot_row[column];
            for (entry, value) in matrix[row].iter_mut().zip(&pivot_row).skip(column) {
                *entry -= factor * value;
            }
            for (entry, value) in rhs[row].iter_mut().zip(&pivot_rhs) {
                *entry -= factor * value;
            }
        }
    }
    for column in (0..TERMS).rev() {
        let mut value = rhs[column];
        for j in column + 1..TERMS {
            for (entry, known) in value.iter_mut().zip(&rhs[j]) {
                *entry -= matrix[column][j] * known;
            }
        }
        rhs[column] = value.map(|v| v / matrix[column][column]);
    }
    Some(rhs)
}
