//! Why a triangulation cannot be built: the named failures of [`crate::Triangulator`] and
//! [`crate::SurfaceMesher`]. Input problems (`TooFewPoints` .. `AllPointsCoplanar`,
//! `DuplicatePoint`, the `Surface...` shape errors) say what the caller must fix;
//! `SurfaceNotRecovered` says how much of a surface flips could not recover;
//! `LocationFailed` and `CavityInconsistent` report a broken internal invariant and refuse to
//! return a mesh rather than hand back a wrong one.

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
    /// A surface triangle names a vertex past the end of the point list.
    SurfaceIndexOutOfRange { triangle: usize, index: u32 },
    /// A surface triangle has no area: its corners repeat or are exactly collinear.
    DegenerateTriangle { triangle: usize },
    /// A surface edge is not used by exactly one triangle in each direction: the surface is not
    /// closed, not manifold, or its triangles are not consistently wound.
    SurfaceNotClosed { edge: [u32; 2] },
    /// A point is on no triangle. Only surface vertices may be given.
    UnusedVertex { index: u32 },
    /// A surface vertex lies exactly on another surface edge, so that edge cannot exist in any
    /// tetrahedralization of these vertices; the surface touches itself.
    VertexOnSurfaceEdge { vertex: u32, edge: [u32; 2] },
    /// Flips could not recover the whole surface: what remains, and how much. The surface may
    /// intersect itself, or may need a point added inside the solid.
    SurfaceNotRecovered {
        missing_edges: usize,
        missing_faces: usize,
        first_missing_edge: Option<[u32; 2]>,
        first_missing_face: Option<[u32; 3]>,
        constraints_without_room: usize,
        steiner_points: usize,
    },
    /// The winding does not leave the solid on the inner side of this triangle.
    SurfaceWindingWrong { triangle: usize },
    /// This triangle does not separate solid from empty, so the shells are not consistent.
    SurfaceDoesNotSeparate { triangle: usize },
    /// The result was requested before the surface was recovered and classified.
    SurfaceNotFinished,
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
            Self::SurfaceIndexOutOfRange { triangle, index } => write!(
                f,
                "surface triangle {triangle} names vertex {index}, past the end of the points"
            ),
            Self::DegenerateTriangle { triangle } => {
                write!(f, "surface triangle {triangle} has no area")
            }
            Self::SurfaceNotClosed { edge } => write!(
                f,
                "surface edge {edge:?} is not used once in each direction; the surface must be \
                 closed, manifold and consistently wound"
            ),
            Self::UnusedVertex { index } => {
                write!(
                    f,
                    "point {index} is on no surface triangle; give only surface vertices"
                )
            }
            Self::VertexOnSurfaceEdge { vertex, edge } => write!(
                f,
                "surface vertex {vertex} lies exactly on surface edge {edge:?}; the surface \
                 touches itself"
            ),
            Self::SurfaceNotRecovered {
                missing_edges,
                missing_faces,
                first_missing_edge,
                first_missing_face,
                constraints_without_room,
                steiner_points,
            } => write!(
                f,
                "{missing_edges} surface edges and {missing_faces} surface triangles could not \
                 be recovered (first edge {first_missing_edge:?}, first triangle \
                 {first_missing_face:?}) after adding {steiner_points} interior points; \
                 {constraints_without_room} had no room inside the solid for a point. The \
                 surface may intersect itself, or a wall may be too thin to hold a point"
            ),
            Self::SurfaceWindingWrong { triangle } => write!(
                f,
                "surface triangle {triangle} is wound with the solid on its outer side; wind \
                 every triangle counterclockwise seen from outside the solid"
            ),
            Self::SurfaceDoesNotSeparate { triangle } => write!(
                f,
                "surface triangle {triangle} does not separate solid from empty space; check \
                 the shells are closed and do not cross"
            ),
            Self::SurfaceNotFinished => {
                write!(
                    f,
                    "the surface is not finished; run step until it reports Done"
                )
            }
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
