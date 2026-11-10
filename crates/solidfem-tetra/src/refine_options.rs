//! What a [`crate::Refiner`] is asked to do and what it says back: [`RefineOptions`], the
//! [`RefinePhase`] a run is in, its [`RefineProgress`] and the [`RefineReport`] of what it did.
//! `refine.rs` holds the run itself and its header the rules these values describe.

use crate::quality::DihedralHistogram;

/// What a [`crate::Refiner`] should do.
#[derive(Clone, Copy, Debug)]
pub struct RefineOptions {
    /// Elements requested through each thin wall; less than 2 adds no layers.
    pub layers_through_walls: u32,
    /// A wall is thin when its thickness is at most this many local surface edge lengths.
    pub thin_wall_ratio: f64,
    /// Full passes of the quality improvement. Zero runs none, but the last attempt still runs on
    /// a tetrahedron of four surface vertices that the 10-node upgrade would refuse, without which
    /// the mesh could not be used at all, or that is flat to the surface's coordinate rounding.
    pub quality_passes: usize,
    /// The most points refinement may add. Reaching it with layer or size work left is a named
    /// error; a quality point is an optional improvement, so the quality pass just stops adding
    /// them at the limit and still finishes.
    pub max_added_points: usize,
    /// The most tetrahedra the refined solid may have, a promise: a run that returns `Ok` has at
    /// most this many. A solid already over it, as given or once its layers are in, is
    /// [`crate::TetraError::TetBudgetExceeded`]; past that the size stage ends at the first point
    /// that would cross it ([`RefineReport::size_stopped_at_budget`]) and the quality pass makes
    /// no flip or point that would.
    pub max_tets: usize,
}

impl Default for RefineOptions {
    fn default() -> Self {
        Self {
            layers_through_walls: 0,
            thin_wall_ratio: 2.0,
            quality_passes: 8,
            max_added_points: 50_000_000,
            max_tets: usize::MAX,
        }
    }
}

/// Which stage a [`crate::Refiner`] is in, in the order they run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefinePhase {
    Layers,
    Sizing,
    Quality,
    Done,
}

/// Where a run has got to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefineProgress {
    pub phase: RefinePhase,
    /// Units finished in this phase: surface vertices, slots looked at, or tetrahedra improved.
    pub done: usize,
    /// Units this phase has in its current pass.
    pub total: usize,
}

/// What refinement did, with the quality before and after.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RefineReport {
    /// Points added inside thin walls, from surface vertices and from wall faces too wide for
    /// their vertices to reach the middle of ([`Self::wall_fill_points`]).
    pub layer_points: usize,
    /// The layer points that came from the interior of a wall's large faces (`wall_fill.rs`).
    pub wall_fill_points: usize,
    /// Points a thin wall proposed that the separation rule or exact placement dropped.
    pub layer_points_dropped: usize,
    /// Surface vertices that found a thin wall.
    pub thin_wall_vertices: usize,
    /// Interior edges joining opposite skins of a thin wall that were split.
    pub through_edges_split: usize,
    /// Such edges left when a whole pass could split no more of them. A success always has none,
    /// because the run then ends in [`crate::TetraError::ThroughEdgeNotSplit`]; read this from the
    /// report of the failed run.
    pub through_edges_refused: usize,
    /// Surface vertices across a wall too thin to divide above rounding, so no points were added.
    pub walls_too_thin_for_rounding: usize,
    /// Points added by the size function.
    pub size_points: usize,
    /// Size candidates that were refused: too close to a neighbour, with no legal cavity, or
    /// over the tetrahedron budget.
    pub size_points_refused: usize,
    /// The size stage ended because its next point would have crossed
    /// [`RefineOptions::max_tets`], so the interior may be coarser than the size function asks.
    pub size_stopped_at_budget: bool,
    /// Size points, quality flips and quality points refused because they would have crossed
    /// [`RefineOptions::max_tets`].
    pub moves_refused_for_budget: usize,
    /// Flips made by the quality pass, by kind.
    pub flips_two_to_three: usize,
    pub flips_three_to_two: usize,
    pub flips_four_to_four: usize,
    pub flips_larger_ring: usize,
    /// Tetrahedra the quality pass improved.
    pub quality_improvements: usize,
    /// Points the quality pass added for slivers no flip or smoothing could remove.
    pub quality_points: usize,
    /// Minimum dihedral angles before the quality pass.
    pub before: DihedralHistogram,
    /// Minimum dihedral angles after the quality pass.
    pub after: DihedralHistogram,
}
