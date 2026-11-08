//! Why a triangulation cannot be built: the named failures of [`crate::Triangulator`]. Input
//! problems (`TooFewPoints` .. `AllPointsCoplanar`, `DuplicatePoint`) say what the caller must
//! fix; `LocationFailed` and `CavityInconsistent` report a broken internal invariant and refuse
//! to return a mesh rather than hand back a wrong one.

use std::fmt;

/// Why a triangulation could not be built or continued.
#[derive(Debug, Clone, PartialEq)]
pub enum TetraError {
    /// Fewer than four points were given; a tetrahedron needs four.
    TooFewPoints { count: usize },
    /// More points than the 32-bit indices of the output can address.
    TooManyPoints { count: usize },
    /// A coordinate is NaN or infinite.
    NonFiniteCoordinate { index: usize, value: f64 },
    /// A coordinate is so large or so small (and not zero) that the exact predicates could
    /// overflow or underflow, where they stop being exact. Scale the model.
    CoordinateOutOfRange { index: usize, value: f64 },
    /// Every point is the same point.
    AllPointsCoincident,
    /// Every point lies on one straight line, so no tetrahedron exists.
    AllPointsCollinear,
    /// Every point lies in one plane, so no tetrahedron exists.
    AllPointsCoplanar,
    /// Point `index` has the same coordinates as point `existing`. Tetrahedra index the caller's
    /// point list one for one, so the caller decides how to weld.
    DuplicatePoint { index: u32, existing: u32 },
    /// The result was requested before every point was inserted.
    NotFinished { inserted: usize, total: usize },
    /// A point can only be added once the initial set has been fully inserted.
    StillInserting { inserted: usize, total: usize },
    /// Point location did not reach a tetrahedron: a broken invariant, not a property of the
    /// input.
    LocationFailed { point: u32 },
    /// Inserting a point left a cavity whose boundary is not closed: a broken invariant, not a
    /// property of the input.
    CavityInconsistent { point: u32 },
}

impl fmt::Display for TetraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewPoints { count } => {
                write!(
                    f,
                    "{count} points were given; a tetrahedron needs at least 4"
                )
            }
            Self::TooManyPoints { count } => {
                write!(f, "{count} points is more than this triangulator can index")
            }
            Self::NonFiniteCoordinate { index, value } => {
                write!(
                    f,
                    "point {index} has coordinate {value}; every coordinate must be finite"
                )
            }
            Self::CoordinateOutOfRange { index, value } => write!(
                f,
                "point {index} has coordinate {value}, outside the range where the exact \
                 predicates stay exact; rescale the model"
            ),
            Self::AllPointsCoincident => write!(f, "every point is the same point"),
            Self::AllPointsCollinear => write!(f, "every point lies on one straight line"),
            Self::AllPointsCoplanar => write!(f, "every point lies in one plane"),
            Self::DuplicatePoint { index, existing } => write!(
                f,
                "point {index} has the same coordinates as point {existing}; weld duplicate \
                 points before meshing"
            ),
            Self::NotFinished { inserted, total } => write!(
                f,
                "only {inserted} of {total} points are inserted; run step until it reports Done"
            ),
            Self::StillInserting { inserted, total } => write!(
                f,
                "{inserted} of {total} points are inserted; a point can be added once the \
                 initial set is done"
            ),
            Self::LocationFailed { point } => {
                write!(
                    f,
                    "point location failed for point {point}; this is a defect in the mesher"
                )
            }
            Self::CavityInconsistent { point } => write!(
                f,
                "inserting point {point} left an open cavity; this is a defect in the mesher"
            ),
        }
    }
}

impl std::error::Error for TetraError {}
