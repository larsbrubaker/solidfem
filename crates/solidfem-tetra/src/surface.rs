//! The public face of surface recovery: [`SurfaceMesher`] takes a closed surface, and through
//! resumable steps produces a [`SolidMesh`] whose boundary is that surface and which holds only
//! the solid's tetrahedra. It owns the phases: Delaunay of the surface's vertices, recovering
//! edges, recovering flat patches, then classifying which side of the surface is solid; the
//! flips and per-constraint recovery are in `flips.rs` and `recover.rs`, and what a run reports
//! about itself (progress, counts, why a missing constraint was given up) in `surface_report.rs`.
//!
//! **What the surface pins down.** Triangles exactly coplanar across an edge form a flat patch
//! (`patches.rs`): its internal triangulation does not change the surface, so only the patch's
//! boundary loops are constraints. Every vertex stays where it is and no point is added on the
//! surface, but the triangles inside a flat patch may differ from the input's: they are the
//! patch's constrained Delaunay triangulation in its own plane (`plane_delaunay.rs`), except at
//! the inner edges [`RecoveryReport::in_plane_flips_refused`] counts, where no legal flip exists.
//!
//! **When flips run out.** Flips alone cannot recover every surface: a stuck edge or patch gets a
//! point added strictly inside the solid (`steiner.rs`), a bounded number per constraint and in
//! total, after the rounds of retrying without one. What remains after that is reported by
//! [`TetraError::SurfaceNotRecovered`], never returned as a mesh with a triangle missing.
//!
//! **Which tetrahedra are solid.** Layers are counted from outside through the surface faces
//! (`recover.rs`): an odd layer is solid, whether a body, a void or a body in a void. The
//! triangles' winding only confirms this answer; it never decides it. Each is checked exactly
//! against the solid tetrahedron across it.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::core::GHOST;
use crate::error::TetraError;
use crate::flips::{edge_key, face_key, Flipper, Target};
use crate::patches::{build_patches, Patch};
use crate::predicates::orient;
use crate::recover::{
    crossing_edges, crossing_edges_of_patch, entry_at, label_layers, recover_edge, recover_patch,
    tets_of_face,
};
use crate::steiner::{
    edge_obstructions, normalized, segment_plane_crossing, triangle_normal, Obstruction, Placed,
    SolidRegion, MAX_STEINER_PER_CONSTRAINT,
};
use crate::surface_check::check_surface;
use crate::surface_report::GaveUp;
pub use crate::surface_report::{GaveUpCounts, RecoveryReport, SurfacePhase, SurfaceProgress};
use crate::triangulator::{StepStatus, Triangulator, HULL};

/// The most rounds of retrying constraints a pass makes after earlier recoveries opened up room.
const MAX_ROUNDS: usize = 8;

/// A closed surface's tetrahedralization: only the solid's tetrahedra.
#[derive(Debug, Clone, PartialEq)]
pub struct SolidMesh {
    /// The surface's vertices, unchanged, followed by any points added inside the solid.
    pub points: Vec<[f64; 3]>,
    /// Four point indices per tetrahedron, right-handed (`(b-a) x (c-a) * (d-a) > 0` exactly).
    pub tets: Vec<[u32; 4]>,
    /// The tetrahedron across the face opposite each vertex, or [`HULL`] across the surface.
    pub neighbors: Vec<[u32; 4]>,
}

/// One constraint phase: a work list retried in rounds.
struct Rounds<T> {
    pending: Vec<T>,
    retry: Vec<T>,
    cursor: usize,
    round: usize,
    progressed: bool,
    /// Set for the last pass, when a stuck constraint may get a point added on its behalf.
    steiner_pass: bool,
}

impl<T: Copy> Rounds<T> {
    fn new(pending: Vec<T>) -> Self {
        Self {
            pending,
            retry: Vec::new(),
            cursor: 0,
            round: 0,
            progressed: false,
            steiner_pass: false,
        }
    }

    /// The next constraint, starting another round when one is finished and retrying still
    /// might help. `None` when the phase is over.
    fn next_item(&mut self) -> Option<T> {
        if self.cursor < self.pending.len() {
            let item = self.pending[self.cursor];
            self.cursor += 1;
            return Some(item);
        }
        if self.retry.is_empty() {
            return None;
        }
        if self.progressed && self.round + 1 < MAX_ROUNDS {
            self.round += 1;
        } else if self.steiner_pass {
            return None;
        } else {
            self.steiner_pass = true;
        }
        self.progressed = false;
        self.pending = std::mem::take(&mut self.retry);
        self.cursor = 0;
        self.next_item()
    }
}

/// Everything that exists once the Delaunay phase has finished.
struct Recovery {
    flipper: Flipper,
    edge_rounds: Rounds<(u32, u32)>,
    patch_rounds: Rounds<usize>,
    missing_edges: BTreeSet<(u32, u32)>,
    missing_patches: BTreeSet<usize>,
    region: SolidRegion,
    edge_triangles: HashMap<u64, [u32; 2]>,
    steiner_limit: usize,
    no_room: usize,
    settled: usize,
    triangles: Vec<[u32; 3]>,
    /// The stuck edge getting points on the last pass, between units of work.
    edge_work: Option<EdgeWork>,
    /// Why each edge and patch was last given up on a pass that may add points. The pass can
    /// repeat while it makes progress, so only the last reason of what stays missing is counted.
    edge_gave_up: HashMap<(u32, u32), GaveUp>,
    patch_gave_up: HashMap<usize, GaveUp>,
}

/// A stuck edge part way through getting points. Each stage is one unit, so a host can pause or
/// cancel between a point and the walks that follow it.
#[derive(Clone, Copy)]
struct EdgeWork {
    u: u32,
    v: u32,
    points_left: usize,
    stage: EdgeStage,
}

#[derive(Clone, Copy)]
enum EdgeStage {
    PlacePoint,
    WalkFromU,
    WalkFromV,
}

/// Builds the solid mesh of a closed surface in resumable steps.
pub struct SurfaceMesher {
    triangles: Vec<[u32; 3]>,
    edges: Vec<(u32, u32)>,
    triangulator: Option<Triangulator>,
    pending_patches: Option<PendingPatches>,
    recovery: Option<Recovery>,
    phase: SurfacePhase,
    report: RecoveryReport,
    solid: Option<SolidMesh>,
}

/// What surface analysis prepares before the Delaunay phase, used once it has finished.
struct PendingPatches {
    patches: Vec<Patch>,
    edge_triangles: HashMap<u64, [u32; 2]>,
    region: SolidRegion,
}

impl SurfaceMesher {
    /// Checks the surface and prepares the Delaunay phase. Nothing long runs here beyond the
    /// sort the point-set triangulator does in its own constructor.
    pub fn new(points: Vec<[f64; 3]>, triangles: Vec<[u32; 3]>) -> Result<Self, TetraError> {
        check_surface(&points, &triangles)?;
        let analysis = build_patches(&points, &triangles);
        let region = SolidRegion::new(&points, &triangles);
        let edges = analysis.feature_edges;
        let triangulator = Triangulator::new(points)?;
        let report = RecoveryReport {
            edges_total: edges.len(),
            faces_total: triangles.len(),
            flat_patches: analysis.patches.len(),
            ..RecoveryReport::default()
        };
        Ok(Self {
            triangles,
            edges,
            triangulator: Some(triangulator),
            pending_patches: Some(PendingPatches {
                patches: analysis.patches,
                edge_triangles: analysis.edge_triangles,
                region,
            }),
            recovery: None,
            phase: SurfacePhase::Delaunay,
            report,
            solid: None,
        })
    }

    /// Advances by up to `max_units` units (a point inserted, a surface edge or face recovered or
    /// set aside, or one stage of the work on a stuck edge), checking `cancel` before each.
    ///
    /// # Errors
    /// Any input error from the point set, [`TetraError::SurfaceNotRecovered`] when flips cannot
    /// finish, and the classification errors. A failure is final: `result` keeps refusing.
    pub fn step(
        &mut self,
        max_units: usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<StepStatus, TetraError> {
        let mut used = 0usize;
        loop {
            if self.phase == SurfacePhase::Done {
                return Ok(StepStatus::Done);
            }
            if used >= max_units {
                return Ok(StepStatus::Running);
            }
            if cancel() {
                return Ok(StepStatus::Cancelled);
            }
            used += 1;
            self.advance()?;
        }
    }

    /// One unit of work in the current phase.
    fn advance(&mut self) -> Result<(), TetraError> {
        match self.phase {
            SurfacePhase::Delaunay => self.advance_delaunay(),
            SurfacePhase::RecoveringEdges => self.advance_edges(),
            SurfacePhase::RecoveringFaces => self.advance_faces(),
            SurfacePhase::Classifying => self.classify(),
            SurfacePhase::Done => Ok(()),
        }
    }

    fn advance_delaunay(&mut self) -> Result<(), TetraError> {
        let Some(triangulator) = self.triangulator.as_mut() else {
            return Err(TetraError::SurfaceNotFinished);
        };
        if triangulator.step(1, &mut || false)? != StepStatus::Done {
            return Ok(());
        }
        let Some(triangulator) = self.triangulator.take() else {
            return Err(TetraError::SurfaceNotFinished);
        };
        let core = triangulator.into_core();
        let locked_edges: HashSet<u64> = self.edges.iter().map(|&(u, v)| edge_key(u, v)).collect();
        let Some(pending) = self.pending_patches.take() else {
            return Err(TetraError::SurfaceNotFinished);
        };
        // Single-triangle patches are real constraints from the start; a flat patch's triangles
        // are not, and its faces lock only once it has been recovered.
        let locked_faces: HashSet<[u32; 3]> = pending
            .patches
            .iter()
            .filter(|patch| patch.triangles.len() == 1)
            .map(|patch| {
                let t = patch.triangles[0];
                face_key(t[0], t[1], t[2])
            })
            .collect();
        let patch_count = pending.patches.len();
        let mut flipper = Flipper::new(core, locked_edges, locked_faces, pending.patches);
        let mut missing_edge = Vec::new();
        for &(u, v) in &self.edges {
            if flipper.find_edge(u, v).is_none() {
                missing_edge.push((u, v));
            }
        }
        self.report.edges_present_after_delaunay = self.edges.len() - missing_edge.len();
        let present = self
            .triangles
            .iter()
            .filter(|t| flipper.face_exists(t[0], t[1], t[2]))
            .count();
        self.report.faces_present_after_delaunay = present;
        self.recovery = Some(Recovery {
            flipper,
            edge_rounds: Rounds::new(missing_edge),
            patch_rounds: Rounds::new((0..patch_count).collect()),
            missing_edges: BTreeSet::new(),
            missing_patches: BTreeSet::new(),
            region: pending.region,
            edge_triangles: pending.edge_triangles,
            // A generous bound that still ends: a fair share of the triangles, plus room for
            // small surfaces.
            steiner_limit: self.triangles.len() / 4 + 64,
            no_room: 0,
            settled: 0,
            triangles: self.triangles.clone(),
            edge_work: None,
            edge_gave_up: HashMap::new(),
            patch_gave_up: HashMap::new(),
        });
        self.phase = SurfacePhase::RecoveringEdges;
        Ok(())
    }

    fn recovery_mut(&mut self) -> Result<&mut Recovery, TetraError> {
        self.recovery.as_mut().ok_or(TetraError::SurfaceNotFinished)
    }

    fn advance_edges(&mut self) -> Result<(), TetraError> {
        let recovery = self.recovery_mut()?;
        if let Some(work) = recovery.edge_work.take() {
            return advance_edge_work(recovery, work);
        }
        let Some((u, v)) = recovery.edge_rounds.next_item() else {
            recovery.missing_edges = recovery.edge_rounds.retry.iter().copied().collect();
            recovery.settled = 0;
            self.phase = SurfacePhase::RecoveringFaces;
            return Ok(());
        };
        // Walking in from the other end meets different blockers, so it can succeed where the first
        // end did not.
        if recover_edge(&mut recovery.flipper, u, v)? || recover_edge(&mut recovery.flipper, v, u)?
        {
            finish_edge(recovery, u, v, true);
        } else if recovery.edge_rounds.steiner_pass {
            recovery.edge_work = Some(EdgeWork {
                u,
                v,
                points_left: MAX_STEINER_PER_CONSTRAINT,
                stage: EdgeStage::PlacePoint,
            });
        } else {
            finish_edge(recovery, u, v, false);
        }
        Ok(())
    }

    fn advance_faces(&mut self) -> Result<(), TetraError> {
        let recovery = self.recovery_mut()?;
        let Some(id) = recovery.patch_rounds.next_item() else {
            recovery.missing_patches = recovery.patch_rounds.retry.iter().copied().collect();
            self.phase = SurfacePhase::Classifying;
            return Ok(());
        };
        // A patch whose boundary is not all there can only be missing: leave it for the count.
        let boundary_present = recovery.flipper.patches[id]
            .boundary
            .clone()
            .iter()
            .all(|&(u, v)| recovery.flipper.find_edge(u, v).is_some());
        let steiner_pass = recovery.patch_rounds.steiner_pass;
        if boundary_present && recover_patch_with_points(recovery, id, steiner_pass)? {
            recovery.patch_rounds.progressed = true;
            recovery.settled += 1;
        } else {
            if steiner_pass && !boundary_present {
                recovery.patch_gave_up.insert(id, GaveUp::BoundaryMissing);
            }
            recovery.patch_rounds.retry.push(id);
        }
        Ok(())
    }

    fn classify(&mut self) -> Result<(), TetraError> {
        let counts = {
            let recovery = self.recovery_mut()?;
            recovery.flipper.counts
        };
        self.report.flips_two_to_three = counts.two_three;
        self.report.flips_three_to_two = counts.three_two;
        self.report.flips_four_to_four = counts.four_four;
        self.report.flips_larger_ring = counts.larger;
        self.report.in_plane_flips = counts.in_plane_flips;
        self.report.in_plane_flips_refused = counts.in_plane_flips_refused;
        let recovery = self
            .recovery
            .as_mut()
            .ok_or(TetraError::SurfaceNotFinished)?;
        self.report.edges_failed = recovery.missing_edges.len();
        self.report.faces_failed = recovery
            .missing_patches
            .iter()
            .map(|&id| recovery.flipper.patches[id].triangles.len())
            .sum();
        self.report.steiner_points = recovery.flipper.kept_steiner_points();
        self.report.constraints_without_room = recovery.no_room;
        let mut gave_up = GaveUpCounts::default();
        for edge in &recovery.missing_edges {
            if let Some(&reason) = recovery.edge_gave_up.get(edge) {
                gave_up.add(reason);
            }
        }
        for id in &recovery.missing_patches {
            if let Some(&reason) = recovery.patch_gave_up.get(id) {
                gave_up.add(reason);
            }
        }
        self.report.gave_up = gave_up;
        if !recovery.missing_edges.is_empty() || !recovery.missing_patches.is_empty() {
            return Err(TetraError::SurfaceNotRecovered {
                missing_edges: recovery.missing_edges.len(),
                missing_faces: self.report.faces_failed,
                first_missing_edge: recovery.missing_edges.iter().next().map(|&(a, b)| [a, b]),
                first_missing_face: recovery
                    .missing_patches
                    .iter()
                    .next()
                    .map(|&id| recovery.flipper.patches[id].triangles[0]),
                constraints_without_room: recovery.no_room,
                steiner_points: recovery.flipper.kept_steiner_points(),
            });
        }
        // Every surface region is covered by locked faces now; a flood across the faces that are
        // not locked cannot leak, and each covering face must leave solid on its inner side.
        let labels = label_layers(&recovery.flipper)?;
        for id in 0..recovery.flipper.patches.len() {
            let plane = recovery.flipper.patches[id].triangles[0];
            for face in recovery.flipper.patch_faces[id].clone() {
                let sides = tets_of_face(&mut recovery.flipper, face);
                let mut solid_seen = false;
                let mut layers = [0i32; 2];
                for (slot, (tet, apex)) in sides.iter().enumerate().take(2) {
                    let layer = labels[*tet as usize];
                    layers[slot] = layer;
                    if layer % 2 == 1 {
                        solid_seen = true;
                        let points = &recovery.flipper.core.points;
                        let [a, b, c] = plane.map(|v| &points[v as usize]);
                        if *apex == GHOST || orient(a, b, c, &points[*apex as usize]) <= 0.0 {
                            return Err(TetraError::SurfaceWindingWrong { triangle: id });
                        }
                    }
                }
                if sides.len() != 2
                    || !solid_seen
                    || (layers[0] - layers[1]).abs() != 1
                    || layers.iter().any(|&l| l < 0)
                {
                    return Err(TetraError::SurfaceDoesNotSeparate { triangle: id });
                }
            }
        }
        self.report.surface_triangles_changed = self
            .triangles
            .iter()
            .filter(|t| {
                !recovery
                    .flipper
                    .locked_faces
                    .contains(&face_key(t[0], t[1], t[2]))
            })
            .count();
        self.solid = Some(extract_solid(&recovery.flipper, &labels));
        self.phase = SurfacePhase::Done;
        Ok(())
    }

    /// Where the run is now, for reporting.
    pub fn progress(&self) -> SurfaceProgress {
        match (&self.phase, &self.triangulator, &self.recovery) {
            (SurfacePhase::Delaunay, Some(t), _) => SurfaceProgress {
                phase: self.phase,
                done: t.inserted(),
                total: t.total(),
            },
            (SurfacePhase::RecoveringEdges, _, Some(r)) => SurfaceProgress {
                phase: self.phase,
                done: r.settled,
                total: self.edges.len() - self.report.edges_present_after_delaunay,
            },
            (SurfacePhase::RecoveringFaces, _, Some(r)) => SurfaceProgress {
                phase: self.phase,
                done: r.settled,
                total: r.flipper.patches.len(),
            },
            _ => SurfaceProgress {
                phase: self.phase,
                done: 0,
                total: 0,
            },
        }
    }

    /// The report of what happened, filled in as the phases complete.
    pub fn report(&self) -> RecoveryReport {
        self.report
    }

    /// The surface edges and the triangles of the flat patches still missing once recovery has
    /// ended, for diagnosing a [`TetraError::SurfaceNotRecovered`]; empty before then.
    pub fn missing_constraints(&self) -> (Vec<[u32; 2]>, Vec<[u32; 3]>) {
        let Some(recovery) = &self.recovery else {
            return (Vec::new(), Vec::new());
        };
        let edges = recovery
            .missing_edges
            .iter()
            .map(|&(a, b)| [a, b])
            .collect();
        let faces = recovery
            .missing_patches
            .iter()
            .flat_map(|&id| recovery.flipper.patches[id].triangles.iter().copied())
            .collect();
        (edges, faces)
    }

    /// The finished solid mesh.
    pub fn result(&self) -> Result<SolidMesh, TetraError> {
        self.solid.clone().ok_or(TetraError::SurfaceNotFinished)
    }
}

/// Records how a stuck edge ended: recovered, or back on the list for the next round.
fn finish_edge(recovery: &mut Recovery, u: u32, v: u32, recovered: bool) {
    if recovered {
        recovery.edge_rounds.progressed = true;
        recovery.settled += 1;
    } else {
        recovery.edge_rounds.retry.push((u, v));
    }
}

/// One unit of the work on a stuck edge: placing a point inside the solid near where its walk is
/// blocked, or the walk from one end after a point, whichever is next.
fn advance_edge_work(recovery: &mut Recovery, work: EdgeWork) -> Result<(), TetraError> {
    let EdgeWork {
        u,
        v,
        points_left,
        stage,
    } = work;
    let next = |stage, points_left| {
        Some(EdgeWork {
            u,
            v,
            points_left,
            stage,
        })
    };
    let following = match stage {
        EdgeStage::PlacePoint => match place_point_for_edge(recovery, u, v)? {
            PointOutcome::Recovered => {
                finish_edge(recovery, u, v, true);
                None
            }
            PointOutcome::Stopped(reason) => {
                recovery.edge_gave_up.insert((u, v), reason);
                finish_edge(recovery, u, v, false);
                None
            }
            PointOutcome::Placed => next(EdgeStage::WalkFromU, points_left),
        },
        EdgeStage::WalkFromU => {
            if recover_edge(&mut recovery.flipper, u, v)? {
                finish_edge(recovery, u, v, true);
                None
            } else {
                next(EdgeStage::WalkFromV, points_left)
            }
        }
        EdgeStage::WalkFromV => {
            if recover_edge(&mut recovery.flipper, v, u)? {
                finish_edge(recovery, u, v, true);
                None
            } else if points_left > 1 {
                next(EdgeStage::PlacePoint, points_left - 1)
            } else {
                recovery.edge_gave_up.insert((u, v), GaveUp::PointsSpent);
                finish_edge(recovery, u, v, false);
                None
            }
        }
    };
    recovery.edge_work = following;
    Ok(())
}

/// What placing a point for a stuck edge came to.
enum PointOutcome {
    /// The edge exists now.
    Recovered,
    /// A point went in; the walks from each end come next, as units of their own.
    Placed,
    /// No more points can go in for this edge, and why.
    Stopped(GaveUp),
}

/// Places one point inside the solid for stuck edge `(u, v)`. A point placed over a chord's ring
/// exists to make that ring small enough to flip, so that flip is tried here, with no relief so
/// the point's star stays. If the point's cavity took the chord or an end of it (an earlier added
/// point can be swallowed), the flip finds no such edge and does nothing.
fn place_point_for_edge(
    recovery: &mut Recovery,
    u: u32,
    v: u32,
) -> Result<PointOutcome, TetraError> {
    if recovery.flipper.steiner_points >= recovery.steiner_limit {
        return Ok(PointOutcome::Stopped(GaveUp::PointLimit));
    }
    let entry = match entry_at(&mut recovery.flipper, u, v)? {
        Some(entry) => Some(entry),
        None => entry_at(&mut recovery.flipper, v, u)?,
    };
    let Some(entry) = entry else {
        return Ok(PointOutcome::Stopped(GaveUp::NoEntry));
    };
    let points = &recovery.flipper.core.points;
    let (pu, pv) = (points[u as usize], points[v as usize]);
    let candidates = edge_obstructions(&mut recovery.flipper, entry, u, v);
    let directions = inward_directions(recovery, u, v);
    let scale = distance(pu, pv) * 0.5;
    let mut placed = None;
    let mut no_room = false;
    for (obstruction, base) in candidates {
        match recovery.flipper.insert_steiner(
            &recovery.region,
            obstruction,
            base,
            &directions,
            scale,
        )? {
            Placed::Added => {
                placed = Some(obstruction);
                break;
            }
            Placed::NotPlaced { no_room: none } => no_room = no_room || none,
        }
    }
    let Some(placed) = placed else {
        if no_room {
            recovery.no_room += 1;
        }
        return Ok(PointOutcome::Stopped(GaveUp::NoPlace));
    };
    if let Obstruction::RingArc { a, b, .. } = placed {
        let depth = recovery.flipper.relief_depth;
        recovery.flipper.relief_depth = 0;
        let flipped = recovery.flipper.remove_edge(a, b, Target::Edge(u, v))?;
        recovery.flipper.relief_depth = depth;
        if flipped && recovery.flipper.find_edge(u, v).is_some() {
            return Ok(PointOutcome::Recovered);
        }
    }
    Ok(PointOutcome::Placed)
}

/// Recovers one flat patch, by flips and then, on the last pass, with interior points beside
/// the edges that pierce it.
fn recover_patch_with_points(
    recovery: &mut Recovery,
    id: usize,
    steiner_pass: bool,
) -> Result<bool, TetraError> {
    if recover_patch(&mut recovery.flipper, id)? {
        return Ok(true);
    }
    if !steiner_pass {
        return Ok(false);
    }
    let plane = recovery.flipper.patches[id].triangles[0];
    let single = recovery.flipper.patches[id].triangles.len() == 1;
    let mut reason = GaveUp::PointsSpent;
    for _ in 0..MAX_STEINER_PER_CONSTRAINT {
        if recovery.flipper.steiner_points >= recovery.steiner_limit {
            reason = GaveUp::PointLimit;
            break;
        }
        let crossing = if single {
            crossing_edges(&mut recovery.flipper, plane)
        } else {
            crossing_edges_of_patch(&mut recovery.flipper, id)
        };
        let Some(&(p, q)) = crossing.first() else {
            reason = GaveUp::NothingCrossing;
            break;
        };
        let points = &recovery.flipper.core.points;
        let (pp, pq) = (points[p as usize], points[q as usize]);
        let [a, b, c] = plane.map(|v| points[v as usize]);
        let base = segment_plane_crossing(pp, pq, a, b, c);
        let directions = vec![negated(triangle_normal(a, b, c))];
        let scale = distance(pp, pq) * 0.5;
        match recovery.flipper.insert_steiner(
            &recovery.region,
            Obstruction::Edge(p, q),
            base,
            &directions,
            scale,
        )? {
            Placed::Added => {
                let depth = recovery.flipper.relief_depth;
                recovery.flipper.relief_depth = 0;
                let recovered = recover_patch(&mut recovery.flipper, id)?;
                recovery.flipper.relief_depth = depth;
                if recovered {
                    return Ok(true);
                }
            }
            Placed::NotPlaced { no_room } => {
                if no_room {
                    recovery.no_room += 1;
                }
                reason = GaveUp::NoPlace;
                break;
            }
        }
    }
    recovery.patch_gave_up.insert(id, reason);
    Ok(false)
}

/// The directions from a stuck edge into the solid: opposite the outward normals of the two
/// surface triangles on the edge, and their average, the direction least likely to hit anything.
fn inward_directions(recovery: &Recovery, u: u32, v: u32) -> Vec<[f64; 3]> {
    let points = &recovery.flipper.core.points;
    let Some(&[t, s]) = recovery.edge_triangles.get(&edge_key(u, v)) else {
        return Vec::new();
    };
    let normal = |index: u32| {
        let [a, b, c] = recovery.triangles[index as usize].map(|w| points[w as usize]);
        triangle_normal(a, b, c)
    };
    let (n1, n2) = (normal(t), normal(s));
    vec![
        negated(normalized([n1[0] + n2[0], n1[1] + n2[1], n1[2] + n2[2]])),
        negated(n1),
        negated(n2),
    ]
}

fn negated(v: [f64; 3]) -> [f64; 3] {
    [-v[0], -v[1], -v[2]]
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// The solid's tetrahedra, right-handed, in slot order, with non-solid neighbours marked [`HULL`].
fn extract_solid(flipper: &Flipper, labels: &[i32]) -> SolidMesh {
    let core = &flipper.core;
    let is_solid = |slot: usize| {
        labels[slot] % 2 == 1 && !core.tets[slot].contains(&GHOST) && labels[slot] >= 0
    };
    let mut output_index = vec![HULL; core.tets.len()];
    let mut count = 0u32;
    for (slot, index) in output_index.iter_mut().enumerate() {
        if is_solid(slot) {
            *index = count;
            count += 1;
        }
    }
    // A point added and later swallowed by another point's cavity is in no tetrahedron: dropping
    // it keeps every output point used. Surface vertices keep their numbers, all below the added
    // points.
    let surface_count = core.points.len() - flipper.steiner_points;
    let mut used = vec![false; core.points.len()];
    used[..surface_count].fill(true);
    for (slot, tet) in core.tets.iter().enumerate() {
        if is_solid(slot) {
            for &v in tet {
                used[v as usize] = true;
            }
        }
    }
    let mut point_index = vec![0u32; core.points.len()];
    let mut points = Vec::with_capacity(core.points.len());
    for (index, &keep) in used.iter().enumerate() {
        if keep {
            // The output point count fits u32 as the triangulation's own does.
            point_index[index] = points.len() as u32;
            points.push(core.points[index]);
        }
    }
    let mut tets = Vec::with_capacity(count as usize);
    let mut neighbors = Vec::with_capacity(count as usize);
    for (slot, tet) in core.tets.iter().enumerate() {
        if !is_solid(slot) {
            continue;
        }
        let adjacent = core.neighbors[slot];
        tets.push([tet[1], tet[0], tet[2], tet[3]].map(|v| point_index[v as usize]));
        neighbors.push([
            output_index[adjacent[1] as usize],
            output_index[adjacent[0] as usize],
            output_index[adjacent[2] as usize],
            output_index[adjacent[3] as usize],
        ]);
    }
    SolidMesh {
        points,
        tets,
        neighbors,
    }
}
