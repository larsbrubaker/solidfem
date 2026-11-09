//! The public face of surface recovery: [`SurfaceMesher`] takes a closed surface, and through
//! resumable steps produces a [`SolidMesh`] holding every surface triangle exactly and only the
//! solid's tetrahedra. It owns the phases: Delaunay of the surface's vertices, recovering edges,
//! recovering faces, then classifying which side of the surface is solid; the flips and the
//! per-constraint recovery are in `flips.rs` and `recover.rs`.
//!
//! No point is ever added: this stage only flips, and a surface flips cannot recover is reported
//! by [`TetraError::SurfaceNotRecovered`], never returned as a mesh with a triangle missing.
//!
//! **Which tetrahedra are solid.** Layers are counted from outside through the surface faces
//! (`recover.rs`): an odd layer is solid, whether a body, a void or a body in a void. The
//! triangles' winding only confirms this answer; it never decides it. Each is checked exactly
//! against the solid tetrahedron across it.

use std::collections::{BTreeSet, HashSet};

use crate::core::GHOST;
use crate::error::TetraError;
use crate::flips::{edge_key, face_key, Flipper};
use crate::predicates::orient;
use crate::recover::{label_layers, recover_edge, recover_face, tets_of_face};
use crate::triangulator::{StepStatus, Triangulator, HULL};

/// The most rounds of retrying constraints a pass makes after earlier recoveries opened up room.
const MAX_ROUNDS: usize = 8;

/// A closed surface's tetrahedralization: only the solid's tetrahedra.
#[derive(Debug, Clone, PartialEq)]
pub struct SolidMesh {
    /// The surface's vertices, unchanged, followed by any interior points added (none today).
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
    /// Points added strictly inside the solid. This stage adds none.
    pub steiner_points: usize,
    pub edges_failed: usize,
    pub faces_failed: usize,
}

/// One constraint phase: a work list retried in rounds.
struct Rounds<T> {
    pending: Vec<T>,
    retry: Vec<T>,
    cursor: usize,
    round: usize,
    progressed: bool,
}

impl<T: Copy> Rounds<T> {
    fn new(pending: Vec<T>) -> Self {
        Self {
            pending,
            retry: Vec::new(),
            cursor: 0,
            round: 0,
            progressed: false,
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
        if !self.progressed || self.retry.is_empty() || self.round + 1 >= MAX_ROUNDS {
            return None;
        }
        self.round += 1;
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
    face_rounds: Rounds<[u32; 3]>,
    missing_edges: BTreeSet<(u32, u32)>,
    missing_faces: BTreeSet<[u32; 3]>,
    settled: usize,
}

/// Builds the solid mesh of a closed surface in resumable steps.
pub struct SurfaceMesher {
    triangles: Vec<[u32; 3]>,
    edges: Vec<(u32, u32)>,
    triangulator: Option<Triangulator>,
    recovery: Option<Recovery>,
    phase: SurfacePhase,
    report: RecoveryReport,
    solid: Option<SolidMesh>,
}

impl SurfaceMesher {
    /// Checks the surface and prepares the Delaunay phase. Nothing long runs here beyond the
    /// sort the point-set triangulator does in its own constructor.
    pub fn new(points: Vec<[f64; 3]>, triangles: Vec<[u32; 3]>) -> Result<Self, TetraError> {
        let edges = check_surface(&points, &triangles)?;
        let triangulator = Triangulator::new(points)?;
        let report = RecoveryReport {
            edges_total: edges.len(),
            faces_total: triangles.len(),
            ..RecoveryReport::default()
        };
        Ok(Self {
            triangles,
            edges,
            triangulator: Some(triangulator),
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
        let locked_faces: HashSet<[u32; 3]> = self
            .triangles
            .iter()
            .map(|t| face_key(t[0], t[1], t[2]))
            .collect();
        let mut flipper = Flipper::new(core, locked_edges, locked_faces);
        let mut missing_edge = Vec::new();
        for &(u, v) in &self.edges {
            if flipper.find_edge(u, v).is_none() {
                missing_edge.push((u, v));
            }
        }
        self.report.edges_present_after_delaunay = self.edges.len() - missing_edge.len();
        let mut missing_face = Vec::new();
        for t in &self.triangles {
            if !flipper.face_exists(t[0], t[1], t[2]) {
                missing_face.push(*t);
            }
        }
        self.report.faces_present_after_delaunay = self.triangles.len() - missing_face.len();
        self.recovery = Some(Recovery {
            flipper,
            edge_rounds: Rounds::new(missing_edge),
            face_rounds: Rounds::new(missing_face),
            missing_edges: BTreeSet::new(),
            missing_faces: BTreeSet::new(),
            settled: 0,
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
        // Walking in from the other end meets different blockers, so it can succeed where the
        // first end did not.
        if recover_edge(&mut recovery.flipper, u, v)? || recover_edge(&mut recovery.flipper, v, u)?
        {
            recovery.edge_rounds.progressed = true;
            recovery.settled += 1;
        } else {
            recovery.edge_rounds.retry.push((u, v));
        }
        Ok(())
    }

    fn advance_faces(&mut self) -> Result<(), TetraError> {
        let recovery = self.recovery_mut()?;
        let Some(face) = recovery.face_rounds.next_item() else {
            recovery.missing_faces = recovery.face_rounds.retry.iter().copied().collect();
            self.phase = SurfacePhase::Classifying;
            return Ok(());
        };
        let [a, b, c] = face;
        // A face whose edge is gone can only be missing: leave it for the final error's count.
        let edges_present = [(a, b), (b, c), (c, a)]
            .iter()
            .all(|&(u, v)| recovery.flipper.find_edge(u, v).is_some());
        if edges_present && recover_face(&mut recovery.flipper, a, b, c)? {
            recovery.face_rounds.progressed = true;
            recovery.settled += 1;
        } else {
            recovery.face_rounds.retry.push(face);
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
        let recovery = self
            .recovery
            .as_mut()
            .ok_or(TetraError::SurfaceNotFinished)?;
        self.report.edges_failed = recovery.missing_edges.len();
        self.report.faces_failed = recovery.missing_faces.len();
        if !recovery.missing_edges.is_empty() || !recovery.missing_faces.is_empty() {
            return Err(TetraError::SurfaceNotRecovered {
                missing_edges: recovery.missing_edges.len(),
                missing_faces: recovery.missing_faces.len(),
                first_missing_edge: recovery.missing_edges.iter().next().map(|&(a, b)| [a, b]),
                first_missing_face: recovery.missing_faces.iter().next().copied(),
            });
        }
        // Every triangle is a face now; a flood across the faces that are not surface triangles
        // cannot leak, and each triangle's winding must leave solid on its inner side.
        let labels = label_layers(&recovery.flipper)?;
        for (index, t) in self.triangles.iter().enumerate() {
            let sides = tets_of_face(&mut recovery.flipper, face_key(t[0], t[1], t[2]));
            let mut solid_seen = false;
            let mut layers = [0i32; 2];
            for (slot, (tet, apex)) in sides.iter().enumerate().take(2) {
                let layer = labels[*tet as usize];
                layers[slot] = layer;
                if layer % 2 == 1 {
                    solid_seen = true;
                    let points = &recovery.flipper.core.points;
                    let [a, b, c] = t.map(|v| &points[v as usize]);
                    if *apex == GHOST || orient(a, b, c, &points[*apex as usize]) <= 0.0 {
                        return Err(TetraError::SurfaceWindingWrong { triangle: index });
                    }
                }
            }
            if sides.len() != 2
                || !solid_seen
                || (layers[0] - layers[1]).abs() != 1
                || layers.iter().any(|&l| l < 0)
            {
                return Err(TetraError::SurfaceDoesNotSeparate { triangle: index });
            }
        }
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
                total: self.triangles.len() - self.report.faces_present_after_delaunay,
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
