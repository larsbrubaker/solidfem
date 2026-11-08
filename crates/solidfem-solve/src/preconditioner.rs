//! The preconditioner seam of the conjugate gradient solver and its diagonal implementation.
//! `pcg.rs` only sees the [`Preconditioner`] trait, so a multigrid preconditioner replaces
//! [`Jacobi`] without touching the loop.

use std::fmt;

use crate::csr::CsrMatrix;

/// A symmetric positive definite approximation of the inverse of the system matrix.
pub trait Preconditioner {
    /// Writes `M^-1 * residual` into `output`. Must be linear, symmetric and positive definite,
    /// or the conjugate gradient iteration loses its guarantees.
    fn apply(&self, residual: &[f64], output: &mut [f64]);
}

/// Diagonal (Jacobi) preconditioning.
#[derive(Debug, Clone, PartialEq)]
pub struct Jacobi {
    inverse_diagonal: Vec<f64>,
}

/// A diagonal entry that cannot be inverted.
#[derive(Debug, Clone, PartialEq)]
pub struct JacobiError {
    /// The degree of freedom whose diagonal is not positive.
    pub row: usize,
    /// Its diagonal value.
    pub diagonal: f64,
}

impl fmt::Display for JacobiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "matrix row {} has diagonal {}; a stiffness matrix needs a positive diagonal",
            self.row, self.diagonal
        )
    }
}

impl std::error::Error for JacobiError {}

impl Jacobi {
    /// Builds the preconditioner from the matrix diagonal. A diagonal that is not a positive
    /// finite number means the matrix is not positive definite, refused here rather than
    /// surfacing later as a silently wrong solve.
    pub fn new(matrix: &CsrMatrix) -> Result<Self, JacobiError> {
        let diagonal = matrix.diagonal();
        let mut inverse_diagonal = Vec::with_capacity(diagonal.len());
        for (row, value) in diagonal.into_iter().enumerate() {
            if value.is_nan() || value <= 0.0 || !value.is_finite() {
                return Err(JacobiError {
                    row,
                    diagonal: value,
                });
            }
            inverse_diagonal.push(1.0 / value);
        }
        Ok(Jacobi { inverse_diagonal })
    }
}

impl Preconditioner for Jacobi {
    fn apply(&self, residual: &[f64], output: &mut [f64]) {
        for ((out, r), inverse) in output
            .iter_mut()
            .zip(residual)
            .zip(self.inverse_diagonal.iter())
        {
            *out = r * inverse;
        }
    }
}
