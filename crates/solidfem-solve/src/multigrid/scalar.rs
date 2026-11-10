//! The value types a finished multigrid hierarchy is stored in and applied in. Setup always
//! computes in `f64`; [`super::cycle`] and the block products in `block.rs` are generic over a
//! storage [`Scalar`] and a vector [`Scalar`], each stored value widened to the vector type as it
//! is read, so the same V-cycle code runs in any of the three [`super::Precision`] choices. Every
//! operation keeps its order whatever the type, so `f64` gives the same bits as code written for
//! `f64` alone, and `f32` gives the same bits on every run.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

/// A floating-point type the hierarchy can be stored in.
pub(super) trait Scalar:
    Copy
    + Debug
    + Default
    + Send
    + Sync
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
{
    /// Zero.
    const ZERO: Self;
    /// The nearest value of this type.
    fn from_f64(value: f64) -> Self;
    /// The exact value as `f64`.
    fn to_f64(self) -> f64;
    /// This value as `V`: exact when `V` is at least as wide, the identity when it is the same
    /// type.
    #[inline]
    fn widen<V: Scalar>(self) -> V {
        V::from_f64(self.to_f64())
    }
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    #[inline]
    fn from_f64(value: f64) -> Self {
        value
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    #[inline]
    fn from_f64(value: f64) -> Self {
        // Rounds to nearest. A value beyond f32's range (about 3.4e38) becomes infinite rather
        // than saturating; stiffness entries scale with Young's modulus times element size, far
        // inside that range in any unit system.
        value as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
