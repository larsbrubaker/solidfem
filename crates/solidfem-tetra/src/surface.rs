//! The public face of surface recovery: [`SurfaceMesher`] takes a closed surface, and through
//! resumable steps produces a [`SolidMesh`] whose boundary is that surface and which holds only
//! the solid's tetrahedra. It owns the phases: Delaunay of the surface's vertices, recovering
//! edges, recovering flat patches, then classifying which side of the surface is solid; the
//! flips and per-constraint recovery are in `flips.rs` and `recover.rs`.
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

use crate::core::{FACE, GHOST};
use crate::error::TetraError;
use crate::flips::{edge_key, face_key, Flipper};
use crate::patches::{build_patches, Patch};
use crate::predicates::orient;
use crate::recover::{
    crossing_edges, crossing_edges_of_patch, entry_at, label_layers, recover_edge, recover_patch,
    tets_of_face, Entry,
};
use crate::steiner::{Obstruction, Placed, SolidRegion, MAX_STEINER_PER_CONSTRAINT};
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

/// Which phase a [`SurfaceMesher`] is in, in the order they run.
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
    /// Points added strictly inside the solid.
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
    /// Edges inside flat patches flipped to make each patch Delaunay in its own plane.
    pub in_plane_flips: usize,
    /// Edges inside a flat patch that fail the in-circle test but could not be flipped: the
    /// patch keeps a valid triangulation that is not fully Delaunay at this many edges.
    pub in_plane_flips_refused: usize,
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

    /// Advances by up to `max_units` units (a point inserted, or a surface edge or face recovered
    /// or set aside), checking `cancel` before each.
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
        });
        self.phase = SurfacePhase::RecoveringEdges;
        Ok(())
    }

    fn recovery_mut(&mut self) -> Result<&mut Recovery, TetraError> {
        self.recovery.as_mut().ok_or(TetraError::SurfaceNotFinished)
    }

    fn advance_edges(&mut self) -> Result<(), TetraError> {
        let recovery = self.recovery_mut()?;
        let Some((u, v)) = recovery.edge_rounds.next_item() else {
            recovery.missing_edges = recovery.edge_rounds.retry.iter().copied().collect();
            recovery.settled = 0;
            self.phase = SurfacePhase::RecoveringFaces;
            return Ok(());
        };
        let steiner_pass = recovery.edge_rounds.steiner_pass;
        if recover_edge_with_points(recovery, u, v, steiner_pass)? {
            recovery.edge_rounds.progressed = true;
            recovery.settled += 1;
        } else {
            recovery.edge_rounds.retry.push((u, v));
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
        self.report.steiner_points = recovery.flipper.steiner_points;
        self.report.constraints_without_room = recovery.no_room;
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
                steiner_points: recovery.flipper.steiner_points,
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

    /// The finished solid mesh.
    pub fn result(&self) -> Result<SolidMesh, TetraError> {
        self.solid.clone().ok_or(TetraError::SurfaceNotFinished)
    }
}

/// Recovers one feature edge: by flips from either end, then on the last pass by adding points
/// inside the solid near where its walk is blocked, retrying flips after each.
fn recover_edge_with_points(
    recovery: &mut Recovery,
    u: u32,
    v: u32,
    steiner_pass: bool,
) -> Result<bool, TetraError> {
    // Walking in from the other end meets different blockers, so it can succeed where the first
    // end did not.
    if recover_edge(&mut recovery.flipper, u, v)? || recover_edge(&mut recovery.flipper, v, u)? {
        return Ok(true);
    }
    if !steiner_pass {
        return Ok(false);
    }
    for _ in 0..MAX_STEINER_PER_CONSTRAINT {
        if recovery.flipper.steiner_points >= recovery.steiner_limit {
            break;
        }
        let entry = match entry_at(&mut recovery.flipper, u, v)? {
            Some(entry) => Some(entry),
            None => entry_at(&mut recovery.flipper, v, u)?,
        };
        let Some(entry) = entry else {
            break;
        };
        let points = &recovery.flipper.core.points;
        let (pu, pv) = (points[u as usize], points[v as usize]);
        let (obstruction, base) = match entry {
            Entry::Face(t, slot) => {
                let face = FACE[slot].map(|p| recovery.flipper.core.tets[t as usize][p]);
                match recovery.flipper.hull_fold_over(face, u, v) {
                    Some((a, b)) => (Obstruction::HullFold(a, b), midpoint(pu, pv)),
                    None => {
                        let points = &recovery.flipper.core.points;
                        let [a, b, c] = face.map(|f| points[f as usize]);
                        (
                            Obstruction::Face(t, slot),
                            segment_plane_crossing(pu, pv, a, b, c),
                        )
                    }
                }
            }
            Entry::Edge(a, b) => {
                let (pa, pb) = (points[a as usize], points[b as usize]);
                (Obstruction::Edge(a, b), midpoint(pa, pb))
            }
        };
        let directions = inward_directions(recovery, u, v);
        let scale = distance(pu, pv) * 0.5;
        match recovery.flipper.insert_steiner(
            &recovery.region,
            obstruction,
            base,
            &directions,
            scale,
        )? {
            Placed::Added => {
                // Relief would flip the new point's star away again, undoing what it opened.
                let depth = recovery.flipper.relief_depth;
                recovery.flipper.relief_depth = 0;
                let recovered = recover_edge(&mut recovery.flipper, u, v)?
                    || recover_edge(&mut recovery.flipper, v, u)?;
                recovery.flipper.relief_depth = depth;
                if recovered {
                    return Ok(true);
                }
            }
            Placed::NotPlaced { no_room } => {
                if no_room {
                    recovery.no_room += 1;
                }
                break;
            }
        }
    }
    Ok(false)
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
    for _ in 0..MAX_STEINER_PER_CONSTRAINT {
        if recovery.flipper.steiner_points >= recovery.steiner_limit {
            break;
        }
        let crossing = if single {
            crossing_edges(&mut recovery.flipper, plane)
        } else {
            crossing_edges_of_patch(&mut recovery.flipper, id)
        };
        let Some(&(p, q)) = crossing.first() else {
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
                break;
            }
        }
    }
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

fn triangle_normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    normalized([
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ])
}

fn normalized(v: [f64; 3]) -> [f64; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if length > 0.0 {
        [v[0] / length, v[1] / length, v[2] / length]
    } else {
        [0.0; 3]
    }
}

fn negated(v: [f64; 3]) -> [f64; 3] {
    [-v[0], -v[1], -v[2]]
}

fn midpoint(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ]
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Where the segment `p q` meets the plane of `a b c`, to within rounding: only a position to
/// offset from, every decision about it is made exactly afterwards.
fn segment_plane_crossing(
    p: [f64; 3],
    q: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> [f64; 3] {
    let n = triangle_normal(a, b, c);
    let dot = |v: [f64; 3]| n[0] * v[0] + n[1] * v[1] + n[2] * v[2];
    let (dp, dq) = (
        dot([p[0] - a[0], p[1] - a[1], p[2] - a[2]]),
        dot([q[0] - a[0], q[1] - a[1], q[2] - a[2]]),
    );
    let denominator = dq - dp;
    let t = if denominator == 0.0 {
        0.5
    } else {
        (-dp / denominator).clamp(0.0, 1.0)
    };
    [
        p[0] + (q[0] - p[0]) * t,
        p[1] + (q[1] - p[1]) * t,
        p[2] + (q[2] - p[2]) * t,
    ]
}

/// Refuses anything not a closed, manifold, consistently wound surface using every vertex and
/// with no flat triangle, and returns its undirected edges in a fixed order.
fn check_surface(
    points: &[[f64; 3]],
    triangles: &[[u32; 3]],
) -> Result<Vec<(u32, u32)>, TetraError> {
    let mut directed: Vec<(u32, u32)> = Vec::with_capacity(triangles.len() * 3);
    let mut used = vec![false; points.len()];
    for (index, triangle) in triangles.iter().enumerate() {
        for &vertex in triangle {
            if vertex as usize >= points.len() {
                return Err(TetraError::SurfaceIndexOutOfRange {
                    triangle: index,
                    index: vertex,
                });
            }
            used[vertex as usize] = true;
        }
        let [a, b, c] = *triangle;
        if a == b || b == c || a == c || collinear(points, *triangle) {
            return Err(TetraError::DegenerateTriangle { triangle: index });
        }
        directed.extend_from_slice(&[(a, b), (b, c), (c, a)]);
    }
    if let Some(index) = used.iter().position(|&u| !u) {
        // `index` fits u32: `check_surface` runs before `Triangulator::new` refuses larger inputs,
        // and a set that large fails that test next, so a clamp here is only cosmetic.
        return Err(TetraError::UnusedVertex {
            index: u32::try_from(index).unwrap_or(u32::MAX),
        });
    }
    directed.sort_unstable();
    for pair in directed.windows(2) {
        if pair[0] == pair[1] {
            return Err(TetraError::SurfaceNotClosed {
                edge: [pair[0].0, pair[0].1],
            });
        }
    }
    let mut edges = Vec::with_capacity(directed.len() / 2);
    for &(a, b) in &directed {
        if directed.binary_search(&(b, a)).is_err() {
            return Err(TetraError::SurfaceNotClosed { edge: [a, b] });
        }
        if a < b {
            edges.push((a, b));
        }
    }
    Ok(edges)
}

/// Whether a triangle's corners are exactly on one line: all three 2D projections vanish.
fn collinear(points: &[[f64; 3]], triangle: [u32; 3]) -> bool {
    use robust::{orient2d, Coord};
    let [a, b, c] = triangle.map(|v| points[v as usize]);
    let area = |i: usize, j: usize| {
        let project = |p: &[f64; 3]| Coord { x: p[i], y: p[j] };
        orient2d(project(&a), project(&b), project(&c))
    };
    area(0, 1) == 0.0 && area(1, 2) == 0.0 && area(0, 2) == 0.0
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
    let mut tets = Vec::with_capacity(count as usize);
    let mut neighbors = Vec::with_capacity(count as usize);
    for (slot, tet) in core.tets.iter().enumerate() {
        if !is_solid(slot) {
            continue;
        }
        let adjacent = core.neighbors[slot];
        tets.push([tet[1], tet[0], tet[2], tet[3]]);
        neighbors.push([
            output_index[adjacent[1] as usize],
            output_index[adjacent[0] as usize],
            output_index[adjacent[2] as usize],
            output_index[adjacent[3] as usize],
        ]);
    }
    SolidMesh {
        points: core.points.clone(),
        tets,
        neighbors,
    }
}
