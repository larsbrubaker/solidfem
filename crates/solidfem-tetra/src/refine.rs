//! The public face of refinement: [`Refiner`] takes a finished [`SolidMesh`] and through resumable
//! steps puts points inside it, by through-thickness layers across thin walls (`layers.rs`) and
//! the removal of every edge still joining the walls' opposite skins (`skins.rs`), by a caller's
//! size function, then improves the worst tetrahedra (`quality.rs`). Points go in through
//! `interior.rs`, the only way a vertex is created, and only strictly inside the solid.
//!
//! **What is fixed.** Every boundary face of the input stays a boundary face with the same bits
//! and vertices: surface triangles are never split and surface vertices never move. Boundary
//! faces and edges are locked; everything else is the mesher's to change.
//!
//! **Layers stay layers.** A point dividing a wall is smoothed only across that wall, in the plane
//! the wall's own normal defines, so a layer stays planar however the plane of the ray leans.
//!
//! **No element joins opposite skins.** The layer stage ends by splitting every interior edge
//! between opposite skins of a thin wall, and from then on no later stage can restore one: size
//! points add edges only to themselves and a flip refuses such an edge. So every tetrahedron has
//! no vertex pair on opposite skins, and any column across a wall crosses at least two of them.
//!
//! **Size-driven insertion.** A tetrahedron whose longest interior edge exceeds
//! [`SIZE_RATIO`]` * size(centroid)` gets a point: its circumcentre where a walk from it reaches
//! one inside the solid, else its centroid. A point within half the local size of a cavity vertex
//! is refused, so the points stay separated and the number added is finite; [`RefineOptions::
//! max_added_points`] bounds it regardless and ends in
//! [`TetraError::RefinementBudgetExhausted`]. An edge on the surface cannot be split in this
//! stage, so it is not counted as too long.
//!
//! **Order and determinism.** Work runs in slot and index order, every choice is made by the exact
//! predicates or by floating-point values computed the same way each run, and the only hash maps
//! are read for membership. A slice boundary falls between two units, so slicing changes nothing.

use std::collections::{HashMap, HashSet};

use crate::core::{Core, DEAD, FACE, GHOST};
use crate::error::TetraError;
use crate::flips::{edge_key, face_key, FlipCounts, Flipper};
use crate::layers::{min_layer_thickness, Separation, SurfaceRays};
use crate::quality::{
    cross, histogram, length, min_dihedral_degrees, normalized, subtract, DihedralHistogram,
};
use crate::skins::OppositeSkins;
use crate::surface::SolidMesh;
use crate::triangulator::{StepStatus, HULL};

/// The longest edge a tetrahedron may have, as a multiple of the size function at its centroid.
/// Bisecting an edge longer than this leaves two pieces of at least `sqrt(2)/2` of the size, so
/// splitting always tends towards the requested size rather than past it.
pub const SIZE_RATIO: f64 = std::f64::consts::SQRT_2;

/// The bound on full passes over the mesh in the size stage; each pass ends with no insertion
/// or starts another, so this is only a backstop against a pathological size function.
const MAX_SIZE_ROUNDS: usize = 256;

/// The worst tetrahedra below this minimum dihedral angle are worked on by the quality pass.
const QUALITY_TARGET_DEGREES: f64 = 30.0;

/// A caller's size function: the wanted edge length at a position.
pub type SizeFn = Box<dyn Fn([f64; 3]) -> f64>;

/// What a [`Refiner`] should do.
#[derive(Clone, Copy, Debug)]
pub struct RefineOptions {
    /// Elements requested through each thin wall; less than 2 adds no layers.
    pub layers_through_walls: u32,
    /// A wall is thin when its thickness is at most this many local surface edge lengths.
    pub thin_wall_ratio: f64,
    /// Full passes of the quality improvement.
    pub quality_passes: usize,
    /// The most points refinement may add; reaching it with work left is a named error.
    pub max_added_points: usize,
}

impl Default for RefineOptions {
    fn default() -> Self {
        Self {
            layers_through_walls: 0,
            thin_wall_ratio: 2.0,
            quality_passes: 8,
            max_added_points: 50_000_000,
        }
    }
}

/// Which stage a [`Refiner`] is in, in the order they run.
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
    /// Points added inside thin walls.
    pub layer_points: usize,
    /// Points a thin wall proposed that the separation rule or exact placement dropped.
    pub layer_points_dropped: usize,
    /// Surface vertices that found a thin wall.
    pub thin_wall_vertices: usize,
    /// Interior edges joining opposite skins of a thin wall that were split.
    pub through_edges_split: usize,
    /// Such edges the exact tests refused at every offset, each left in place and counted.
    pub through_edges_refused: usize,
    /// Surface vertices across a wall too thin to divide above rounding, so no points were added.
    pub walls_too_thin_for_rounding: usize,
    /// Points added by the size function.
    pub size_points: usize,
    /// Size candidates that were refused: too close to a neighbour or with no legal cavity.
    pub size_points_refused: usize,
    /// Flips made by the quality pass, by kind.
    pub flips_two_to_three: usize,
    pub flips_three_to_two: usize,
    pub flips_four_to_four: usize,
    pub flips_larger_ring: usize,
    /// Tetrahedra the quality pass improved.
    pub quality_improvements: usize,
    /// Minimum dihedral angles before the quality pass.
    pub before: DihedralHistogram,
    /// Minimum dihedral angles after the quality pass.
    pub after: DihedralHistogram,
}

struct LayerRun {
    rays: SurfaceRays,
    separation: Separation,
    normals: Vec<[f64; 3]>,
    edge_mean: Vec<f64>,
    vertices: Vec<u32>,
    cursor: usize,
    magnitude: f64,
}

struct SplitRun {
    edges: Vec<(u32, u32)>,
    cursor: usize,
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
    result: Option<SolidMesh>,
}

impl Refiner {
    /// Prepares refinement of `solid`. The surface is read from its boundary faces.
    pub fn new(solid: SolidMesh, size: SizeFn, options: RefineOptions) -> Result<Self, TetraError> {
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
        let flipper = Flipper::new(core, locked_edges, locked_faces, Vec::new());
        let (layers, skins) = if options.layers_through_walls >= 2 {
            let run = layer_run(&solid.points, &boundary, &fixed);
            let skins = OppositeSkins::new(
                run.normals.clone(),
                run.edge_mean.clone(),
                options.thin_wall_ratio,
            );
            (Some(run), skins)
        } else {
            (None, OppositeSkins::default())
        };
        let phase = if layers.is_some() {
            RefinePhase::Layers
        } else {
            RefinePhase::Sizing
        };
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
            result: None,
        })
    }

    /// Advances by up to `max_units` units (a surface vertex, a slot looked at, or a tetrahedron
    /// worked on), checking `cancel` before each.
    ///
    /// # Errors
    /// [`TetraError::RefinementBudgetExhausted`] when the point budget runs out with work left,
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
            self.phase = RefinePhase::Sizing;
            return Ok(());
        };
        let Some(&vertex) = run.vertices.get(run.cursor) else {
            self.layers = None;
            self.splits = Some(SplitRun {
                edges: self.skins.through_edges(&self.flipper),
                cursor: 0,
            });
            return Ok(());
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
        let count = self.options.layers_through_walls;
        let layer = hit.distance / f64::from(count);
        if layer < min_layer_thickness(run.magnitude) {
            self.report.walls_too_thin_for_rounding += 1;
            return Ok(());
        }
        for k in 1..count {
            self.budget_check()?;
            let along = layer * f64::from(k);
            let candidate = [
                origin[0] + direction[0] * along,
                origin[1] + direction[1] * along,
                origin[2] + direction[2] * along,
            ];
            let Some(run) = self.layers.as_mut() else {
                return Ok(());
            };
            if !run
                .separation
                .accept(candidate, direction, 0.5 * edge, 0.5 * layer)
            {
                self.report.layer_points_dropped += 1;
                continue;
            }
            if self.place_from_vertex(vertex, candidate)? {
                // Index fits u32: the count is checked against the point limit on insertion.
                let index = self.flipper.core.points.len() as u32 - 1;
                self.axes.insert(index, hit.normal);
                self.report.layer_points += 1;
                self.added += 1;
            } else {
                self.report.layer_points_dropped += 1;
            }
        }
        Ok(())
    }

    /// Splits the next edge joining opposite skins, ending the layer stage when none are left.
    fn advance_split(&mut self) -> Result<(), TetraError> {
        let Some(run) = self.splits.as_mut() else {
            return Ok(());
        };
        let Some(&edge) = run.edges.get(run.cursor) else {
            self.splits = None;
            self.phase = RefinePhase::Sizing;
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
        } else {
            self.report.through_edges_refused += 1;
        }
        Ok(())
    }

    /// Inserts `candidate` starting the walk at a real tetrahedron around `vertex`.
    fn place_from_vertex(&mut self, vertex: u32, candidate: [f64; 3]) -> Result<bool, TetraError> {
        let around = self.flipper.tets_around(vertex);
        let Some(start) = around
            .iter()
            .copied()
            .find(|&t| !self.flipper.core.tets[t as usize].contains(&GHOST))
        else {
            return Ok(false);
        };
        let Some(seeds) = self.flipper.locate_interior(start, candidate) else {
            return Ok(false);
        };
        self.flipper.insert_interior(&seeds, candidate, 0.0)
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
            self.finish();
            return Ok(());
        };
        if run.cursor >= run.list.len() {
            if !run.improved || run.pass >= self.options.quality_passes {
                self.finish();
                return Ok(());
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
        let changed = self.flipper.improve_tet(
            slot,
            &self.fixed,
            &self.skins,
            &self.axes,
            QUALITY_TARGET_DEGREES,
        )?;
        if changed {
            run.improved = true;
            self.report.quality_improvements += 1;
        }
        self.quality = Some(run);
        Ok(())
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

    fn finish(&mut self) {
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
        self.result = Some(extract(&self.flipper));
        self.phase = RefinePhase::Done;
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
            RefinePhase::Quality => self
                .quality
                .as_ref()
                .map_or((0, 0), |run| (run.cursor, run.list.len())),
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

fn checked_size(size: &SizeFn, position: [f64; 3]) -> Result<f64, TetraError> {
    let value = size(position);
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(TetraError::InvalidSizeFunction { position, value })
    }
}

fn centroid_of(points: &[[f64; 3]], tet: &[u32; 4]) -> [f64; 3] {
    let mut sum = [0.0; 3];
    for &v in tet {
        let p = points[v as usize];
        for k in 0..3 {
            sum[k] += p[k];
        }
    }
    [sum[0] / 4.0, sum[1] / 4.0, sum[2] / 4.0]
}

/// The tetrahedron's circumcentre, or `None` when it is too flat for the formula to be finite.
fn circumcentre_of(points: &[[f64; 3]], tet: &[u32; 4]) -> Option<[f64; 3]> {
    let [a, b, c, d] = tet.map(|v| points[v as usize]);
    let (u, v, w) = (subtract(b, a), subtract(c, a), subtract(d, a));
    let denominator = 2.0 * crate::quality::dot(u, cross(v, w));
    if denominator == 0.0 {
        return None;
    }
    let (lu, lv, lw) = (
        crate::quality::dot(u, u),
        crate::quality::dot(v, v),
        crate::quality::dot(w, w),
    );
    let (vw, wu, uv) = (cross(v, w), cross(w, u), cross(u, v));
    Some([
        a[0] + (lu * vw[0] + lv * wu[0] + lw * uv[0]) / denominator,
        a[1] + (lu * vw[1] + lv * wu[1] + lw * uv[1]) / denominator,
        a[2] + (lu * vw[2] + lv * wu[2] + lw * uv[2]) / denominator,
    ])
}

/// Each boundary face of the solid, wound counterclockwise from outside.
fn boundary_faces(solid: &SolidMesh) -> Vec<[u32; 3]> {
    let mut faces = Vec::new();
    for (t, tet) in solid.tets.iter().enumerate() {
        // The output tet is right-handed; the internal tet swaps its first two vertices, so
        // FACE lists faces with the opposite vertex on their positive (inner) side.
        let internal = [tet[1], tet[0], tet[2], tet[3]];
        for (slot, neighbor) in solid.neighbors[t].iter().enumerate() {
            if *neighbor != HULL {
                continue;
            }
            let i = [1, 0, 2, 3][slot];
            let [p, q, r] = FACE[i].map(|position| internal[position]);
            // `orient` is positive when the fourth point is below the plane: the opposite
            // vertex is on that side, so `p q r` is counterclockwise from outside already.
            faces.push([p, q, r]);
        }
    }
    faces
}

/// Prepares the layer stage: the ray tree, the separation grid and each surface vertex's normal
/// and mean incident edge length.
fn layer_run(points: &[[f64; 3]], boundary: &[[u32; 3]], fixed: &[bool]) -> LayerRun {
    let mut normals = vec![[0.0; 3]; points.len()];
    let mut edge_total = vec![0.0; points.len()];
    let mut edge_count = vec![0u32; points.len()];
    let (mut total_edge, mut edges) = (0.0_f64, 0u32);
    for face in boundary {
        let p = face.map(|v| points[v as usize]);
        let normal = normalized(cross(subtract(p[1], p[0]), subtract(p[2], p[0])));
        for k in 0..3 {
            let (v, from, to) = (face[k], p[(k + 1) % 3], p[(k + 2) % 3]);
            let (e1, e2) = (subtract(from, p[k]), subtract(to, p[k]));
            let angle = crate::quality::dot(normalized(e1), normalized(e2))
                .clamp(-1.0, 1.0)
                .acos();
            let n = &mut normals[v as usize];
            n[0] += normal[0] * angle;
            n[1] += normal[1] * angle;
            n[2] += normal[2] * angle;
            let l = length(e1);
            edge_total[v as usize] += l;
            edge_count[v as usize] += 1;
            total_edge += l;
            edges += 1;
        }
    }
    let mean_edge = total_edge / f64::from(edges.max(1));
    let edge_mean: Vec<f64> = edge_total
        .iter()
        .zip(&edge_count)
        .map(|(total, count)| total / f64::from((*count).max(1)))
        .collect();
    let normals: Vec<[f64; 3]> = normals.into_iter().map(normalized).collect();
    let magnitude = points
        .iter()
        .flatten()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    let vertices: Vec<u32> = (0..points.len() as u32)
        .filter(|&v| fixed[v as usize])
        .collect();
    let rays = SurfaceRays::new(points, boundary);
    let mut separation = Separation::new(mean_edge);
    for &v in &vertices {
        separation.add_surface(points[v as usize]);
    }
    LayerRun {
        rays,
        separation,
        normals,
        edge_mean,
        vertices,
        cursor: 0,
        magnitude,
    }
}

/// The current mesh, real tetrahedra only, in the public orientation and order.
fn extract(flipper: &Flipper) -> SolidMesh {
    let core = &flipper.core;
    let is_real = |slot: usize| core.tets[slot].iter().all(|&v| v != GHOST && v != DEAD);
    let mut output_index = vec![HULL; core.tets.len()];
    let mut count = 0u32;
    for (slot, index) in output_index.iter_mut().enumerate() {
        if is_real(slot) {
            *index = count;
            count += 1;
        }
    }
    let mut tets = Vec::with_capacity(count as usize);
    let mut neighbors = Vec::with_capacity(count as usize);
    for slot in 0..core.tets.len() {
        if !is_real(slot) {
            continue;
        }
        let tet = core.tets[slot];
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
