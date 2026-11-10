//! What a [`crate::SurfaceMesher`] run reports about itself: the phase it is in, how far along,
//! and once recovery has ended, what was present, which flips and points restored the rest and,
//! when something is still missing, why each missing constraint was given up. `surface.rs` fills
//! these in as its phases complete; nothing here changes what the mesher does.

/// Which phase a [`crate::SurfaceMesher`] is in, in the order they run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfacePhase {
    Delaunay,
    RecoveringEdges,
    RecoveringFaces,
    Classifying,
    Done,
}

/// Where a run has got to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceProgress {
    pub phase: SurfacePhase,
    /// Units finished in this phase: points inserted, or surface constraints settled.
    pub done: usize,
    /// Units this phase has in total.
    pub total: usize,
}

/// How recovery went: counts of what was already present and which flips restored the rest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    pub edges_total: usize,
    pub edges_present_after_delaunay: usize,
    pub faces_total: usize,
    pub faces_present_after_delaunay: usize,
    /// Tetrahedra-to-tetrahedra flips made, by kind, across both recovery phases.
    pub flips_two_to_three: usize,
    pub flips_three_to_two: usize,
    pub flips_four_to_four: usize,
    pub flips_larger_ring: usize,
    /// Stuck edges recovered by replacing the region around them with a star from one end, adding
    /// no point.
    pub star_reconnections: usize,
    /// Points added strictly inside the solid and still in the mesh. A point swallowed by a later
    /// point's cavity is gone and is not counted.
    pub steiner_points: usize,
    pub edges_failed: usize,
    pub faces_failed: usize,
    /// Input triangles that are not faces of the result, because a flat patch took a different
    /// triangulation of the same region.
    pub surface_triangles_changed: usize,
    /// Flat regions in the input, including those of a single triangle.
    pub flat_patches: usize,
    /// Edges and patches stuck for lack of any legal place for a point inside the solid.
    pub constraints_without_room: usize,
    /// Why each constraint still missing at the end was given up on its last pass.
    pub gave_up: GaveUpCounts,
    /// Edges inside flat patches flipped to make each patch Delaunay in its own plane.
    pub in_plane_flips: usize,
    /// Edges inside a flat patch that fail the in-circle test but could not be flipped: the
    /// patch keeps a valid triangulation that is not fully Delaunay at this many edges.
    pub in_plane_flips_refused: usize,
}

/// Why the constraints still missing were given up, one count per constraint (a missing edge or a
/// missing flat patch), read from its last pass, the one that may add points. Every missing
/// constraint is in exactly one count, so they sum to `edges_failed` plus the missing patches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GaveUpCounts {
    /// The total budget of added points was already spent.
    pub point_limit: usize,
    /// A missing edge's walk entered no tetrahedron from either end, so nothing said where a
    /// point would help.
    pub no_entry: usize,
    /// No candidate place took a point: outside the solid, too close to the surface, or a cavity
    /// that would have broken a surface constraint.
    pub no_place: usize,
    /// Points went in, as many as one constraint may have, and it was still missing.
    pub points_spent: usize,
    /// A patch with no mesh edge left crossing it that still did not become faces.
    pub nothing_crossing: usize,
    /// A patch whose boundary edges were not all present, so it was never tried: it follows from
    /// a missing edge.
    pub boundary_missing: usize,
}

impl GaveUpCounts {
    /// Every constraint counted, whatever its reason.
    pub fn total(&self) -> usize {
        self.point_limit
            + self.no_entry
            + self.no_place
            + self.points_spent
            + self.nothing_crossing
            + self.boundary_missing
    }

    pub(crate) fn add(&mut self, reason: GaveUp) {
        let count = match reason {
            GaveUp::PointLimit => &mut self.point_limit,
            GaveUp::NoEntry => &mut self.no_entry,
            GaveUp::NoPlace => &mut self.no_place,
            GaveUp::PointsSpent => &mut self.points_spent,
            GaveUp::NothingCrossing => &mut self.nothing_crossing,
            GaveUp::BoundaryMissing => &mut self.boundary_missing,
        };
        *count += 1;
    }
}

/// Why one constraint was given up on a pass that may add points; tallied into [`GaveUpCounts`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GaveUp {
    PointLimit,
    NoEntry,
    NoPlace,
    PointsSpent,
    NothingCrossing,
    BoundaryMissing,
}
