//! Why a triangulation cannot be built: the named failures of [`crate::Triangulator`] and
//! [`crate::SurfaceMesher`]. Input problems (`TooFewPoints` .. `AllPointsCoplanar`,
//! `DuplicatePoint`, the `Surface...` shape errors) say what the caller must fix;
//! `SurfaceNotRecovered` says how much of a surface flips could not recover;
//! `TetBudgetExceeded` says how many tetrahedra a solid already has past the caller's limit;
//! `LocationFailed` and `CavityInconsistent` report a broken internal invariant and refuse to
//! return a mesh rather than hand back a wrong one.

use std::fmt;

use crate::refine::RefinePhase;

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
    /// The winding does not leave the solid on the inner side of this input triangle (an index
    /// into the triangles given to the mesher): the surface is inside out, or folds back onto
    /// itself.
    SurfaceWindingWrong { triangle: usize },
    /// This input triangle (an index into the triangles given to the mesher) does not separate
    /// solid from empty: the shells cross, or the surface folds back onto itself so that two
    /// coplanar triangles cover the same area wound opposite ways.
    SurfaceDoesNotSeparate { triangle: usize },
    /// The result was requested before the surface was recovered and classified.
    SurfaceNotFinished,
    /// Refinement stopped because it had added the budgeted number of points and still had work
    /// to do. A larger budget, or coarser requested sizes, lets it finish.
    RefinementBudgetExhausted { added: usize, limit: usize },
    /// The solid already has `tets` tetrahedra, more than the caller's `budget`
    /// ([`crate::RefineOptions::max_tets`]), before `phase` could run: the surface itself, or the
    /// surface with the layers through its thin walls, needs more. `Done` is the final count
    /// checked before the result is handed out, which no gated move can reach. A coarser surface, fewer
    /// layers or a larger budget lets it finish.
    TetBudgetExceeded {
        tets: usize,
        budget: usize,
        phase: RefinePhase,
    },
    /// The size function returned something that is not a positive finite length.
    InvalidSizeFunction { position: [f64; 3], value: f64 },
    /// The refined mesh was requested before refinement finished.
    RefinementNotFinished,
    /// `remaining` interior edges still join opposite skins of a thin wall after every legal split
    /// was tried, `edge` the first of them: an element joining the skins could not be removed
    /// without changing the surface. Refine the surface there, or ask for no layers.
    ThroughEdgeNotSplit { edge: [u32; 2], remaining: usize },
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
                 {first_missing_face:?}) with {steiner_points} interior points added; \
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
                 the shells are closed, do not cross and do not fold back onto themselves"
            ),
            Self::SurfaceNotFinished => {
                write!(
                    f,
                    "the surface is not finished; run step until it reports Done"
                )
            }
            Self::RefinementBudgetExhausted { added, limit } => write!(
                f,
                "refinement added {added} points, its budget of {limit}, with work remaining; \
                 raise the budget or request coarser sizes"
            ),
            Self::TetBudgetExceeded {
                tets,
                budget,
                phase,
            } => write!(
                f,
                "the solid already has {tets} tetrahedra before the {phase:?} stage, {} over its \
                 budget of {budget}; mesh a coarser surface, ask for fewer layers or raise the \
                 budget",
                tets.saturating_sub(*budget)
            ),
            Self::InvalidSizeFunction { position, value } => write!(
                f,
                "the size function returned {value} at {position:?}; it must be a positive \
                 finite length everywhere"
            ),
            Self::ThroughEdgeNotSplit { edge, remaining } => write!(
                f,
                "{remaining} interior edges joining opposite skins of a thin wall could not be \
                 split (first edge {edge:?}), so an element would still join the skins; refine \
                 the surface there, or ask for no layers"
            ),
            Self::RefinementNotFinished => {
                write!(
                    f,
                    "refinement is not finished; run step until it reports Done"
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
