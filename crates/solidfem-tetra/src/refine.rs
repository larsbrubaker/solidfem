//! The public face of refinement: [`Refiner`] takes a finished [`SolidMesh`] and through resumable
//! steps puts points inside it, by through-thickness layers across thin walls (`layers.rs`, and
//! `wall_fill.rs` for walls whose faces have no vertices inside them) and the removal of every
//! edge still joining the walls' opposite skins (`skins.rs`), by a caller's
//! size function, then improves the worst tetrahedra (`quality.rs`, and a point for a sliver no
//! flip can remove, in `quality_points.rs`, with a last attempt there on any tetrahedron of four
//! surface vertices the 10-node upgrade would still refuse or that is flat to the surface's
//! coordinate rounding). `refine.rs`, `layers.rs` and `quality_points.rs` choose
//! where points go; the cavities of `interior.rs` are the only way a vertex is created, and only
//! strictly inside the solid.
//!
//! **What is fixed.** Every boundary face of the input stays a boundary face with the same bits
//! and vertices: surface triangles are never split and surface vertices never move. Boundary
//! faces and edges are locked; everything else is the mesher's to change.
//!
//! **Layers stay layers.** A point dividing a wall is smoothed only across that wall, in the plane
//! the wall's own normal defines, so a layer stays planar however the plane of the ray leans.
//!
//! **No interior edge `skins.rs` calls through a wall survives.** The layer stage ends by removing
//! every such edge (split where a point can be starred, flipped away where it cannot), and from
//! then on no later stage can restore one: size points add edges only to themselves and a flip
//! refuses such an edge. An edge no split or flip can remove ends the run in
//! [`TetraError::ThroughEdgeNotSplit`], so an `Ok` run has none left. The limits are in `skins.rs`:
//! the surface is fixed, so a tetrahedron can still bridge a wall through a surface edge at a
//! rim, and the classification is this library's own rule, checked over shapes by
//! `tests/refine/sweep.rs`.
//!
//! **Size-driven insertion.** A tetrahedron whose longest interior edge exceeds
//! [`SIZE_RATIO`]` * size(centroid)` gets a point: its circumcentre where a walk from it reaches
//! one inside the solid, else its centroid. A point within half the local size of a cavity vertex
//! is refused, so the points stay separated and the number added is finite; [`RefineOptions::
//! max_added_points`] bounds it regardless and ends in
//! [`TetraError::RefinementBudgetExhausted`] (a point the quality pass would have added is an
//! optional improvement, so at the limit the pass stops adding points and still returns `Ok`). An edge on the surface cannot be split in this
//! stage, so it is not counted as too long.
//!
//! **The tetrahedron budget.** [`RefineOptions::max_tets`] is a promise: an `Ok` run has at most
//! that many tetrahedra. The surface and the layers through its thin walls are what the caller
//! asked for, so a solid over the budget as given (checked in [`Refiner::new`]) or once its
//! layers are in is [`TetraError::TetBudgetExceeded`], with its count. After that every move that
//! adds tetrahedra asks the flipper whether it fits (`flips.rs`): the size stage ends at the first
//! point that would cross, with no smaller fallback for that slot
//! ([`RefineReport::size_stopped_at_budget`]), and the quality pass skips flips and points that
//! would. The stop leaves the interior coarser than the size function where the stage had not
//! reached yet; nothing about the surface changes. The count is checked once more before the
//! result is handed out (`phase: Done`), so a move added later without its gate cannot return
//! `Ok` over the budget.
//!
//! **Order and determinism.** Work runs in slot and index order, every choice is made by the exact
//! predicates or by floating-point values computed the same way each run, and the only hash maps
//! are read for membership. A slice boundary falls between two units, so slicing changes nothing.

use std::collections::{HashMap, HashSet};

use crate::core::{Core, DEAD, GHOST};
use crate::error::TetraError;
use crate::flips::{edge_key, face_key, FlipCounts, Flipper};
use crate::layers::{layer_run, min_layer_thickness, LayerRun};
use crate::quality::{histogram, min_dihedral_degrees, subtract, DihedralHistogram};
use crate::quality_points::surface_rounding;
use crate::refine_mesh::{boundary_faces, centroid_of, checked_size, circumcentre_of, extract};
pub use crate::refine_options::{RefineOptions, RefinePhase, RefineProgress, RefineReport};
use crate::skins::{incident_faces, OppositeSkins};
use crate::surface::SolidMesh;
use crate::triangulator::StepStatus;
use crate::wall_fill::{wall_seeds, WallColumn};

/// The longest edge a tetrahedron may have, as a multiple of the size function at its centroid.
/// Bisecting an edge longer than this leaves two pieces of at least `sqrt(2)/2` of the size, so
/// splitting always tends towards the requested size rather than past it.
pub const SIZE_RATIO: f64 = std::f64::consts::SQRT_2;

/// The bound on full passes over the mesh in the size stage; each pass ends with no insertion
/// or starts another, so this is only a backstop against a pathological size function.
const MAX_SIZE_ROUNDS: usize = 256;

/// The worst tetrahedra below this minimum dihedral angle are worked on by the quality pass.
const QUALITY_TARGET_DEGREES: f64 = 30.0;

/// Below this minimum dihedral angle, a tetrahedron that flips and smoothing cannot improve gets a
/// new point (`quality_points.rs`). Points cost elements, so this is the band where a 10-node
/// element loses most of its accuracy, well below the quality target.
const QUALITY_POINT_TARGET_DEGREES: f64 = 5.0;

/// How many second attempts a run makes. Each that changes the mesh is followed by the last quality
/// passes, which can leave a refused tetrahedron behind; this ends what could otherwise repeat.
const MAX_RESCUE_ROUNDS: usize = 4;

/// The quality passes run again after a second attempt changed the mesh, so its new points are
/// worked on like any other.
const PASSES_AFTER_RESCUE: usize = 2;

/// A caller's size function: the wanted edge length at a position.
pub type SizeFn = Box<dyn Fn([f64; 3]) -> f64>;

struct SplitRun {
    edges: Vec<(u32, u32)>,
    cursor: usize,
    /// Edges actually cut in this pass, so a pass that cuts none knows the rest are stuck.
    split_this_pass: usize,
}

struct SizeRun {
    cursor: usize,
    round_end: usize,
    round: usize,
    added_this_round: usize,
}

struct QualityRun {
    list: Vec<u32>,
    cursor: usize,
    pass: usize,
    improved: bool,
    /// The second attempt in progress, once the passes have ended.
    rescue: Option<RescueRun>,
    /// How many second attempts have run and changed the mesh.
    rescue_rounds: usize,
}

/// One second attempt over the flat tetrahedra of four surface vertices (`quality_points.rs`).
struct RescueRun {
    list: Vec<u32>,
    cursor: usize,
    changed: bool,
    /// The surface's coordinate rounding (`quality_points::surface_rounding`).
    rounding: f64,
}

/// Refines a finished solid mesh in resumable steps.
pub struct Refiner {
    flipper: Flipper,
    fixed: Vec<bool>,
    /// Which surface vertices stand on opposite skins; no edge may join two that do.
    skins: OppositeSkins,
    /// Points dividing a wall, with the wall's normal, which they are not smoothed along.
    axes: HashMap<u32, [f64; 3]>,
    size: SizeFn,
    options: RefineOptions,
    phase: RefinePhase,
    layers: Option<LayerRun>,
    splits: Option<SplitRun>,
    size_run: SizeRun,
    quality: Option<QualityRun>,
    report: RefineReport,
    added: usize,
    /// Tetrahedra, by sorted vertices, that no quality point could improve since the last one went
    /// in. Only read for membership, so its order cannot reach the output.
    refused_points: HashSet<[u32; 4]>,
    result: Option<SolidMesh>,
}

impl Refiner {
    /// Prepares refinement of `solid`. The surface is read from its boundary faces.
    ///
    /// # Errors
    /// [`TetraError::TetBudgetExceeded`] when `solid` already has more tetrahedra than
    /// [`RefineOptions::max_tets`], and the errors of rebuilding the triangulation.
    pub fn new(solid: SolidMesh, size: SizeFn, options: RefineOptions) -> Result<Self, TetraError> {
        let first_phase = if options.layers_through_walls >= 2 {
            RefinePhase::Layers
        } else {
            RefinePhase::Sizing
        };
        if solid.tets.len() > options.max_tets {
            return Err(TetraError::TetBudgetExceeded {
                tets: solid.tets.len(),
                budget: options.max_tets,
                phase: first_phase,
            });
        }
        let boundary = boundary_faces(&solid);
        let mut locked_edges = HashSet::with_capacity(boundary.len() * 2);
        let mut locked_faces = HashSet::with_capacity(boundary.len());
        let mut fixed = vec![false; solid.points.len()];
        for face in &boundary {
            let [a, b, c] = *face;
            locked_faces.insert(face_key(a, b, c));
            for (u, v) in [(a, b), (b, c), (c, a)] {
                locked_edges.insert(edge_key(u, v));
            }
            for v in face {
                fixed[*v as usize] = true;
            }
        }
        let core = Core::from_solid(solid.points.clone(), &solid.tets, &solid.neighbors)?;
        let mut flipper = Flipper::new(core, locked_edges, locked_faces, Vec::new());
        flipper.ghosts = boundary.len();
        if first_phase == RefinePhase::Sizing {
            flipper.tet_limit = options.max_tets;
        }
        let (layers, skins) = if options.layers_through_walls >= 2 {
            let run = layer_run(&solid.points, &boundary, &fixed);
            let skins = OppositeSkins::new(
                run.normals.clone(),
                run.edge_mean.clone(),
                incident_faces(&solid.points, &boundary, solid.points.len()),
                options.thin_wall_ratio,
            );
            (Some(run), skins)
        } else {
            (None, OppositeSkins::default())
        };
        let phase = first_phase;
        let round_end = flipper.core.tets.len();
        Ok(Self {
            flipper,
            fixed,
            skins,
            axes: HashMap::new(),
            size,
            options,
            phase,
            layers,
            splits: None,
            size_run: SizeRun {
                cursor: 0,
                round_end,
                round: 0,
                added_this_round: 0,
            },
            quality: None,
            report: RefineReport::default(),
            added: 0,
            refused_points: HashSet::new(),
            result: None,
        })
    }

    /// Advances by up to `max_units` units (a surface vertex, a slot looked at, or a tetrahedron
    /// worked on), checking `cancel` before each.
    ///
    /// # Errors
    /// [`TetraError::RefinementBudgetExhausted`] when the point budget runs out with layer or size
    /// work left (the quality pass stops adding points at the limit instead),
    /// [`TetraError::InvalidSizeFunction`] for a bad size value, and the invariant errors.
    pub fn step(
        &mut self,
        max_units: usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<StepStatus, TetraError> {
        let mut used = 0usize;
        loop {
            if self.phase == RefinePhase::Done {
                return Ok(StepStatus::Done);
            }
            if used >= max_units {
                return Ok(StepStatus::Running);
            }
            if cancel() {
                return Ok(StepStatus::Cancelled);
            }
            used += 1;
            match self.phase {
                RefinePhase::Layers => self.advance_layers()?,
                RefinePhase::Sizing => self.advance_size()?,
                RefinePhase::Quality => self.advance_quality()?,
                RefinePhase::Done => {}
            }
        }
    }

    fn budget_check(&self) -> Result<(), TetraError> {
        if self.added >= self.options.max_added_points {
            return Err(TetraError::RefinementBudgetExhausted {
                added: self.added,
                limit: self.options.max_added_points,
            });
        }
        Ok(())
    }

    fn advance_layers(&mut self) -> Result<(), TetraError> {
        if self.splits.is_some() {
            return self.advance_split();
        }
        let Some(run) = self.layers.as_mut() else {
            return self.begin_sizing();
        };
        let Some(&vertex) = run.vertices.get(run.cursor) else {
            return self.advance_fill();
        };
        run.cursor += 1;
        let normal = run.normals[vertex as usize];
        let edge = run.edge_mean[vertex as usize];
        let origin = self.flipper.core.points[vertex as usize];
        // Inward: against the outward normal of the surface at the vertex.
        let direction = [-normal[0], -normal[1], -normal[2]];
        let Some(hit) = run.rays.nearest(origin, direction, vertex) else {
            return Ok(());
        };
        if !hit.from_inside || hit.distance > self.options.thin_wall_ratio * edge {
            return Ok(());
        }
        self.report.thin_wall_vertices += 1;
        let column = WallColumn {
            origin,
            direction,
            distance: hit.distance,
            far_normal: hit.normal,
            across: 0.5 * edge,
        };
        let placed = self.place_layer_points(vertex, None, column)?;
        self.report.layer_points += placed;
        Ok(())
    }

    /// Places the layer points of one column of a wall, walking from a tetrahedron around
    /// `vertex`, on `face` when there is one. Returns how many were placed; a wall too thin to
    /// divide above rounding is counted and adds none.
    fn place_layer_points(
        &mut self,
        vertex: u32,
        face: Option<[u32; 3]>,
        column: WallColumn,
    ) -> Result<usize, TetraError> {
        let WallColumn {
            origin,
            direction,
            distance,
            far_normal,
            across,
        } = column;
        let count = self.options.layers_through_walls;
        let layer = distance / f64::from(count);
        let magnitude = self.layers.as_ref().map_or(0.0, |run| run.magnitude);
        if layer < min_layer_thickness(magnitude) {
            self.report.walls_too_thin_for_rounding += 1;
            return Ok(0);
        }
        let mut placed = 0;
        for k in 1..count {
            self.budget_check()?;
            let along = layer * f64::from(k);
            let candidate = [
                origin[0] + direction[0] * along,
                origin[1] + direction[1] * along,
                origin[2] + direction[2] * along,
            ];
            let Some(run) = self.layers.as_mut() else {
                return Ok(placed);
            };
            if !run
                .separation
                .accept(candidate, direction, across, 0.5 * layer)
            {
                self.report.layer_points_dropped += 1;
                continue;
            }
            if self.place_from_vertex(vertex, face, candidate)? {
                // Index fits u32: the count is checked against the point limit on insertion.
                let index = self.flipper.core.points.len() as u32 - 1;
                self.axes.insert(index, far_normal);
                placed += 1;
                self.added += 1;
            } else {
                self.report.layer_points_dropped += 1;
            }
        }
        Ok(placed)
    }

    /// Proposes the wall points of the next surface triangle (`wall_fill.rs`), ending the fill and
    /// starting the through-edge splits when none are left.
    fn advance_fill(&mut self) -> Result<(), TetraError> {
        let Some(run) = self.layers.as_mut() else {
            return Ok(());
        };
        let Some(&face) = run.boundary.get(run.fill_cursor) else {
            self.layers = None;
            self.begin_split_pass();
            return Ok(());
        };
        run.fill_cursor += 1;
        let seeds = wall_seeds(
            &run.rays,
            &self.flipper.core.points,
            face,
            self.options.thin_wall_ratio,
            self.options.layers_through_walls,
            run.magnitude,
        );
        for seed in seeds {
            let placed = self.place_layer_points(face[0], Some(face), seed)?;
            self.report.wall_fill_points += placed;
            self.report.layer_points += placed;
        }
        Ok(())
    }

    fn begin_split_pass(&mut self) {
        self.splits = Some(SplitRun {
            edges: self.skins.through_edges(&self.flipper),
            cursor: 0,
            split_this_pass: 0,
        });
    }

    /// Splits the next edge joining opposite skins. At the end of a pass, the edges still there are
    /// tried again while any split was made, the mesh around them having changed; an edge that is
    /// left when a whole pass splits nothing is the named error.
    fn advance_split(&mut self) -> Result<(), TetraError> {
        let Some(run) = self.splits.as_mut() else {
            return Ok(());
        };
        let Some(&edge) = run.edges.get(run.cursor) else {
            let split_this_pass = run.split_this_pass;
            let remaining = self.skins.through_edges(&self.flipper);
            let Some(&first) = remaining.first() else {
                self.splits = None;
                return self.begin_sizing();
            };
            if split_this_pass == 0 {
                self.report.through_edges_refused = remaining.len();
                return Err(TetraError::ThroughEdgeNotSplit {
                    edge: [first.0, first.1],
                    remaining: remaining.len(),
                });
            }
            self.splits = Some(SplitRun {
                edges: remaining,
                cursor: 0,
                split_this_pass: 0,
            });
            return Ok(());
        };
        run.cursor += 1;
        self.budget_check()?;
        let layers = self.options.layers_through_walls;
        let outcome = self
            .skins
            .split_through_edge(&mut self.flipper, edge, layers)?;
        for &point in &outcome.added {
            self.axes.insert(point, outcome.axis);
            self.added += 1;
        }
        if outcome.cut {
            self.report.through_edges_split += 1;
            if let Some(run) = self.splits.as_mut() {
                run.split_this_pass += 1;
            }
        }
        Ok(())
    }

    /// Inserts `candidate` starting the walk at a real tetrahedron around `vertex`, one holding the
    /// rest of `face` when one is given, so the walk does not leave a curved wall to get there.
    fn place_from_vertex(
        &mut self,
        vertex: u32,
        face: Option<[u32; 3]>,
        candidate: [f64; 3],
    ) -> Result<bool, TetraError> {
        let around = self.flipper.tets_around(vertex);
        let real = |t: &u32| !self.flipper.core.tets[*t as usize].contains(&GHOST);
        let on_face = |t: &u32| {
            face.is_some_and(|f| {
                let tet = self.flipper.core.tets[*t as usize];
                f.iter().all(|v| tet.contains(v))
            })
        };
        let Some(start) = around
            .iter()
            .copied()
            .find(|t| real(t) && on_face(t))
            .or_else(|| around.iter().copied().find(real))
        else {
            return Ok(false);
        };
        let Some(seeds) = self.flipper.locate_interior(start, candidate) else {
            return Ok(false);
        };
        self.flipper.insert_interior(&seeds, candidate, 0.0)
    }

    /// Ends the layer stage: the layers are what the caller asked for, so a solid they took over
    /// the budget is refused, and from here on no move may cross it.
    fn begin_sizing(&mut self) -> Result<(), TetraError> {
        let tets = self.flipper.real_tets();
        if tets > self.options.max_tets {
            return Err(TetraError::TetBudgetExceeded {
                tets,
                budget: self.options.max_tets,
                phase: RefinePhase::Sizing,
            });
        }
        self.flipper.tet_limit = self.options.max_tets;
        self.phase = RefinePhase::Sizing;
        Ok(())
    }

    fn advance_size(&mut self) -> Result<(), TetraError> {
        let run = &mut self.size_run;
        if run.cursor >= run.round_end {
            if run.added_this_round == 0 || run.round + 1 >= MAX_SIZE_ROUNDS {
                self.begin_quality();
                return Ok(());
            }
            run.round += 1;
            run.added_this_round = 0;
            run.cursor = 0;
            run.round_end = self.flipper.core.tets.len();
            return Ok(());
        }
        let slot = run.cursor;
        run.cursor += 1;
        let tet = self.flipper.core.tets[slot];
        if tet.iter().any(|&v| v >= DEAD) {
            return Ok(());
        }
        let centroid = centroid_of(&self.flipper.core.points, &tet);
        let wanted = checked_size(&self.size, centroid)?;
        if self.longest_free_edge(&tet) <= SIZE_RATIO * wanted {
            return Ok(());
        }
        self.budget_check()?;
        // `slot` fits u32: the triangulation counts its tetrahedra in u32.
        let slot = slot as u32;
        let circumcentre = circumcentre_of(&self.flipper.core.points, &tet);
        let candidates = [circumcentre, Some(centroid)];
        let refused_before = self.flipper.limit_refusals;
        for candidate in candidates.into_iter().flatten() {
            if !candidate.iter().all(|value| value.is_finite()) {
                continue;
            }
            let Some(seeds) = self.flipper.locate_interior(slot, candidate) else {
                continue;
            };
            let local = checked_size(&self.size, candidate)?;
            if self
                .flipper
                .insert_interior(&seeds, candidate, 0.5 * local)?
            {
                self.report.size_points += 1;
                self.added += 1;
                self.size_run.added_this_round += 1;
                return Ok(());
            }
            if self.flipper.limit_refusals > refused_before {
                // This point would cross the budget: the stage ends here, with no smaller fallback,
                // rather than fill the rest of the slots unevenly by whichever cavities still fit.
                self.report.size_points_refused += 1;
                self.report.size_stopped_at_budget = true;
                self.begin_quality();
                return Ok(());
            }
        }
        self.report.size_points_refused += 1;
        Ok(())
    }

    fn longest_free_edge(&self, tet: &[u32; 4]) -> f64 {
        let mut longest = 0.0_f64;
        for a in 0..4 {
            for b in a + 1..4 {
                if self
                    .flipper
                    .locked_edges
                    .contains(&edge_key(tet[a], tet[b]))
                {
                    continue;
                }
                let d = subtract(
                    self.flipper.core.points[tet[a] as usize],
                    self.flipper.core.points[tet[b] as usize],
                );
                longest = longest.max(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            }
        }
        longest.sqrt()
    }

    fn begin_quality(&mut self) {
        self.report.before = self.histogram_now();
        self.quality = Some(QualityRun {
            list: Vec::new(),
            cursor: 0,
            pass: 0,
            improved: true,
            rescue: None,
            rescue_rounds: 0,
        });
        self.phase = RefinePhase::Quality;
    }

    fn histogram_now(&self) -> DihedralHistogram {
        histogram(
            &self.flipper.core.points,
            self.flipper
                .core
                .tets
                .iter()
                .filter(|tet| tet.iter().all(|&v| v < DEAD)),
        )
    }

    fn advance_quality(&mut self) -> Result<(), TetraError> {
        let Some(mut run) = self.quality.take() else {
            return self.finish();
        };
        if run.rescue.is_some() {
            return self.advance_rescue(run);
        }
        if run.cursor >= run.list.len() {
            if !run.improved || run.pass >= self.options.quality_passes {
                return self.begin_rescue(run);
            }
            run.pass += 1;
            run.improved = false;
            run.cursor = 0;
            run.list = self.worst_first();
            self.quality = Some(run);
            return Ok(());
        }
        let slot = run.list[run.cursor];
        run.cursor += 1;
        let mut changed = self.flipper.improve_tet(
            slot,
            &self.fixed,
            &self.skins,
            &self.axes,
            QUALITY_TARGET_DEGREES,
        )?;
        if !changed && self.added < self.options.max_added_points && self.needs_point(slot) {
            changed = self.try_quality_point(slot)?;
        }
        if changed {
            run.improved = true;
            self.report.quality_improvements += 1;
        }
        self.quality = Some(run);
        Ok(())
    }

    /// Starts a second attempt over the flat tetrahedra of four surface vertices, or ends the run
    /// when there are none or the rounds are used up.
    fn begin_rescue(&mut self, mut run: QualityRun) -> Result<(), TetraError> {
        if run.rescue_rounds >= MAX_RESCUE_ROUNDS {
            return self.finish();
        }
        // A slot fits u32: the triangulation counts its tetrahedra in u32.
        let rounding = surface_rounding(&self.flipper.core.points, &self.fixed);
        let list: Vec<u32> = (0..self.flipper.core.tets.len() as u32)
            .filter(|&slot| self.flipper.needs_rescue(slot, &self.fixed, rounding))
            .collect();
        if list.is_empty() {
            return self.finish();
        }
        run.rescue = Some(RescueRun {
            list,
            cursor: 0,
            changed: false,
            rounding,
        });
        self.quality = Some(run);
        Ok(())
    }

    /// One tetrahedron of the second attempt, or the end of it: when any point went in, the last
    /// quality passes run again, and the next second attempt looks at what is left.
    fn advance_rescue(&mut self, mut run: QualityRun) -> Result<(), TetraError> {
        let Some(rescue) = run.rescue.as_mut() else {
            return self.finish();
        };
        let Some(&slot) = rescue.list.get(rescue.cursor) else {
            if !rescue.changed {
                return self.finish();
            }
            run.rescue = None;
            run.rescue_rounds += 1;
            run.pass = self
                .options
                .quality_passes
                .saturating_sub(PASSES_AFTER_RESCUE);
            run.improved = true;
            run.list.clear();
            run.cursor = 0;
            self.quality = Some(run);
            return Ok(());
        };
        rescue.cursor += 1;
        // An earlier point of this attempt may have replaced this tetrahedron already.
        if self.added < self.options.max_added_points
            && self
                .flipper
                .needs_rescue(slot, &self.fixed, rescue.rounding)
            && self.flipper.rescue_tet(slot)?
        {
            rescue.changed = true;
            self.added += 1;
            self.report.quality_points += 1;
            self.refused_points.clear();
        }
        self.quality = Some(run);
        Ok(())
    }

    /// Tries a point for tetrahedron `slot`, unless the same tetrahedron was already refused one:
    /// a sliver with no point to offer is looked at again on every pass, and a layered wall has
    /// thousands of them, so a refusal is remembered until a quality point goes in.
    fn try_quality_point(&mut self, slot: u32) -> Result<bool, TetraError> {
        let mut key = self.flipper.core.tets[slot as usize];
        key.sort_unstable();
        if self.refused_points.contains(&key) {
            return Ok(false);
        }
        if self.flipper.add_quality_point(slot)? {
            self.added += 1;
            self.report.quality_points += 1;
            // The tetrahedra around the new point changed, so a refusal may no longer hold.
            self.refused_points.clear();
            return Ok(true);
        }
        self.refused_points.insert(key);
        Ok(false)
    }

    /// Whether tetrahedron `slot` is still below the angle a new point is worth spending on.
    fn needs_point(&self, slot: u32) -> bool {
        let tet = self.flipper.core.tets[slot as usize];
        tet.iter().all(|&v| v < DEAD)
            && min_dihedral_degrees(&self.flipper.core.points, &tet) < QUALITY_POINT_TARGET_DEGREES
    }

    /// Slots of the tetrahedra below the quality target, worst first, ties by slot.
    fn worst_first(&self) -> Vec<u32> {
        let core = &self.flipper.core;
        let mut scored: Vec<(f64, u32)> = core
            .tets
            .iter()
            .enumerate()
            .filter(|(_, tet)| tet.iter().all(|&v| v < DEAD))
            .filter_map(|(slot, tet)| {
                let angle = min_dihedral_degrees(&core.points, tet);
                // `slot` fits u32: the triangulation counts its tetrahedra in u32.
                (angle < QUALITY_TARGET_DEGREES).then_some((angle, slot as u32))
            })
            .collect();
        scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        scored.into_iter().map(|(_, slot)| slot).collect()
    }

    /// Hands out the result, after a last check of the count against the budget: every move past
    /// the layers is gated, so this refuses only a move added later without its gate.
    fn finish(&mut self) -> Result<(), TetraError> {
        let tets = self.flipper.real_tets();
        if tets > self.options.max_tets {
            return Err(TetraError::TetBudgetExceeded {
                tets,
                budget: self.options.max_tets,
                phase: RefinePhase::Done,
            });
        }
        self.report.after = self.histogram_now();
        let FlipCounts {
            two_three,
            three_two,
            four_four,
            larger,
            ..
        } = self.flipper.counts;
        self.report.flips_two_to_three = two_three;
        self.report.flips_three_to_two = three_two;
        self.report.flips_four_to_four = four_four;
        self.report.flips_larger_ring = larger;
        self.report.moves_refused_for_budget = self.flipper.limit_refusals;
        self.result = Some(extract(&self.flipper));
        self.phase = RefinePhase::Done;
        Ok(())
    }

    /// Where the run is now, for reporting.
    pub fn progress(&self) -> RefineProgress {
        let (done, total) = match self.phase {
            RefinePhase::Layers => self.splits.as_ref().map_or_else(
                || {
                    self.layers
                        .as_ref()
                        .map_or((0, 0), |run| (run.cursor, run.vertices.len()))
                },
                |run| (run.cursor, run.edges.len()),
            ),
            RefinePhase::Sizing => (self.size_run.cursor, self.size_run.round_end),
            RefinePhase::Quality => self.quality.as_ref().map_or((0, 0), |run| {
                run.rescue
                    .as_ref()
                    .map_or((run.cursor, run.list.len()), |rescue| {
                        (rescue.cursor, rescue.list.len())
                    })
            }),
            RefinePhase::Done => (0, 0),
        };
        RefineProgress {
            phase: self.phase,
            done,
            total,
        }
    }

    /// What refinement did, filled in as the stages complete.
    pub fn report(&self) -> RefineReport {
        self.report
    }

    /// The finished mesh.
    pub fn result(&self) -> Result<SolidMesh, TetraError> {
        self.result.clone().ok_or(TetraError::RefinementNotFinished)
    }
}
