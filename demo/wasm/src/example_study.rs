//! An example part from its surface file to a solved result: the surface is parsed, meshed by
//! [`SurfaceMesher`] and refined by [`Refiner`] in bounded steps, upgraded to 10-node elements and
//! held and loaded by the benchmark case's own `apply`, then solved by a [`DemoStudy`]. The mesh is
//! kept so the visitor changing the material solves again without meshing again; changing the load
//! size solves nothing at all, because the results of a linear solve scale with the load
//! (`DemoStudy::set_load_scale`).
//!
//! A part the mesher, the refiner or the case refuses ends in the failed state with the sentence
//! the page shows, so a refusal is never a blank view. Which parts are offered and what the page
//! says about them is `examples.rs`.

use std::collections::BTreeMap;

use solidfem::elastic::{
    Material, Model, ModelError, MultigridOptions, PreconditionerChoice, Quadrature,
};
use solidfem::mesh::QuadraticTetMesh;
use solidfem::meshing::quadratic_from_solid;
use solidfem::tetra::{RefineOptions, Refiner, SizeFn, SolidMesh, StepStatus, SurfaceMesher};
use solidfem_bench::benchmarks::cases::{Case, Density};
use solidfem_bench::benchmarks::pipeline::{parse_surface, Supports};

use crate::engine::{DemoStudy, PartStart, ReadyPart, StudyState};
use crate::examples::ExamplePart;

/// Mesher and refiner units per budget unit: one unit is a single point insertion or check, far
/// cheaper than an element or a solver iteration.
pub const MESH_UNITS_PER_BUDGET_UNIT: usize = 32;

/// The quadrature the benchmarks judge with, so the page solves what they solved.
const QUADRATURE: Quadrature = Quadrature::FourPoint;

/// The most held-node markers drawn: enough to read the held region without hiding the mesh.
pub const MAX_HELD_MARKERS: usize = 64;

/// The most load arrows drawn: enough to show where the load acts without hiding the part.
pub const MAX_LOAD_ARROWS: usize = 32;

/// The material a visitor chose for an example part.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExampleMaterial {
    pub material: Material,
    /// Absent for the benchmark's own material, which states no yield stress.
    pub yield_strength: Option<f64>,
}

/// A part meshed, held and loaded, kept across material changes.
#[derive(Debug, Clone)]
pub struct MeshedPart {
    pub id: &'static str,
    pub mesh: QuadraticTetMesh,
    pub supports: Supports,
}

/// Where an example study is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExampleState {
    Meshing,
    Refining,
    Preparing,
    Study(StudyState),
    Failed,
    Cancelled,
}

/// Where the run is. The preparing state is four passes, one per frame, because together they
/// are the longest unsliced work a part has: the 10-node upgrade, the case's holds and loads, the
/// surface and geometry check, and the matrix pattern.
enum Phase {
    Surface(Box<SurfaceMesher>),
    Refining(Box<Refiner>),
    Upgrading(Box<SolidMesh>),
    Holding(Box<QuadraticTetMesh>),
    Readying(Box<MeshedPart>),
    Starting(Box<ReadyPart>),
    Study(Box<DemoStudy>),
    Failed,
}

/// One example part's run, from its surface to its result.
pub struct ExampleStudy {
    part: &'static ExamplePart,
    case: Case,
    density: Density,
    material: ExampleMaterial,
    load_scale: f64,
    multigrid: bool,
    phase: Phase,
    cancelled: bool,
    failure: Option<String>,
    meshed: Option<MeshedPart>,
    held_points: Vec<f32>,
    load_arrows: Vec<f32>,
}

impl ExampleStudy {
    /// Starts meshing `surface_text` for a study solved with multigrid, or from `retained` when
    /// it holds this part already. Never
    /// fails: a refusal is the failed state with a sentence for the page.
    pub fn start(
        part: &'static ExamplePart,
        surface_text: &str,
        material: ExampleMaterial,
        load_scale: f64,
        retained: Option<MeshedPart>,
    ) -> Self {
        Self::start_with_preconditioner(part, surface_text, material, load_scale, retained, true)
    }

    /// [`ExampleStudy::start`] with the preconditioner stated, so the timing test can compare
    /// both on a part; the page always uses multigrid.
    pub fn start_with_preconditioner(
        part: &'static ExamplePart,
        surface_text: &str,
        material: ExampleMaterial,
        load_scale: f64,
        retained: Option<MeshedPart>,
        multigrid: bool,
    ) -> Self {
        let mut study = match part.case_and_density() {
            Some((case, density)) => ExampleStudy {
                part,
                case,
                density,
                material,
                load_scale,
                multigrid,
                phase: Phase::Failed,
                cancelled: false,
                failure: None,
                meshed: None,
                held_points: Vec::new(),
                load_arrows: Vec::new(),
            },
            None => panic!(
                "the example table names case {:?} and surface {:?}, which the benchmarks no \
                 longer have; the examples test holds them together",
                part.case_id, part.surface_file
            ),
        };
        match retained {
            Some(meshed) if meshed.id == part.id => {
                study.phase = Phase::Readying(Box::new(meshed));
            }
            _ => study.begin_mesh(surface_text),
        }
        study
    }

    fn begin_mesh(&mut self, surface_text: &str) {
        let (points, triangles) = match parse_surface(surface_text, &self.density.surface_file) {
            Ok(surface) => surface,
            Err(reason) => return self.refuse(format!("The part's surface file is bad: {reason}")),
        };
        match SurfaceMesher::new(points, triangles) {
            Ok(mesher) => self.phase = Phase::Surface(Box::new(mesher)),
            Err(error) => self.refuse(format!(
                "The mesher refused this part's surface: {error}. Nothing was solved."
            )),
        }
    }

    fn refuse(&mut self, sentence: String) {
        self.failure = Some(sentence);
        self.phase = Phase::Failed;
    }

    /// The table entry this study is of.
    pub fn part(&self) -> &'static ExamplePart {
        self.part
    }

    /// Where the study is.
    pub fn state(&self) -> ExampleState {
        if self.cancelled {
            return ExampleState::Cancelled;
        }
        match &self.phase {
            Phase::Surface(_) => ExampleState::Meshing,
            Phase::Refining(_) => ExampleState::Refining,
            Phase::Upgrading(_) | Phase::Holding(_) | Phase::Readying(_) | Phase::Starting(_) => {
                ExampleState::Preparing
            }
            Phase::Study(study) => ExampleState::Study(study.state()),
            Phase::Failed => ExampleState::Failed,
        }
    }

    /// How far the mesher or refiner is, as (done, total) of its current pass.
    pub fn mesh_progress(&self) -> (usize, usize) {
        match &self.phase {
            Phase::Surface(mesher) => {
                let progress = mesher.progress();
                (progress.done, progress.total)
            }
            Phase::Refining(refiner) => {
                let progress = refiner.progress();
                (progress.done, progress.total)
            }
            _ => (0, 0),
        }
    }

    /// The sentence for a refusal or a failed solve.
    pub fn failure(&self) -> Option<&str> {
        match &self.phase {
            Phase::Study(study) => study.failure(),
            _ => self.failure.as_deref(),
        }
    }

    /// The solve, once the part is meshed.
    pub fn study(&self) -> Option<&DemoStudy> {
        match &self.phase {
            Phase::Study(study) => Some(study),
            _ => None,
        }
    }

    /// The meshed part, held and loaded, once meshing is done.
    pub fn meshed(&self) -> Option<&MeshedPart> {
        self.meshed.as_ref()
    }

    /// Takes the meshed part, for a new study of the same part.
    pub fn take_meshed(self) -> Option<MeshedPart> {
        self.meshed
    }

    /// The positions of the held nodes, three numbers each.
    pub fn held_points(&self) -> &[f32] {
        &self.held_points
    }

    /// Where the load acts: a tail, a unit direction and a length, seven numbers per arrow.
    pub fn load_arrows(&self) -> &[f32] {
        &self.load_arrows
    }

    /// Shows the results at `factor` times the benchmark's load, without solving again.
    pub fn set_load_scale(&mut self, factor: f64) {
        self.load_scale = factor;
        if let Phase::Study(study) = &mut self.phase {
            study.set_load_scale(factor);
        }
    }

    /// Stops the study at once.
    pub fn cancel(&mut self) {
        self.cancelled = true;
        if let Phase::Study(study) = &mut self.phase {
            study.cancel();
        }
    }

    /// Runs one slice of at most `budget` units, returning the new state.
    pub fn step(&mut self, budget: u32) -> ExampleState {
        if self.cancelled {
            return ExampleState::Cancelled;
        }
        let units = (budget.max(1) as usize).saturating_mul(MESH_UNITS_PER_BUDGET_UNIT);
        let mut no_cancel = || false;
        match std::mem::replace(&mut self.phase, Phase::Failed) {
            Phase::Surface(mut mesher) => match mesher.step(units, &mut no_cancel) {
                Ok(StepStatus::Done) => match mesher.result() {
                    Ok(solid) => self.begin_refine(solid),
                    Err(error) => self.refuse_mesh(&error),
                },
                Ok(_) => self.phase = Phase::Surface(mesher),
                Err(error) => self.refuse_mesh(&error),
            },
            Phase::Refining(mut refiner) => match refiner.step(units, &mut no_cancel) {
                Ok(StepStatus::Done) => match refiner.result() {
                    Ok(solid) => self.phase = Phase::Upgrading(Box::new(solid)),
                    Err(error) => self.refuse_mesh(&error),
                },
                Ok(_) => self.phase = Phase::Refining(refiner),
                Err(error) => self.refuse_mesh(&error),
            },
            Phase::Upgrading(solid) => match quadratic_from_solid(&solid, QUADRATURE) {
                Ok(mesh) => self.phase = Phase::Holding(Box::new(mesh)),
                Err(error) => self.refuse(format!("The mesher refused this part: {error}.")),
            },
            Phase::Holding(mesh) => match (self.case.apply)(&mesh) {
                Ok(supports) => {
                    self.phase = Phase::Readying(Box::new(MeshedPart {
                        id: self.part.id,
                        mesh: *mesh,
                        supports,
                    }));
                }
                Err(reason) => self.refuse(format!(
                    "The part's holds and loads could not be placed: {reason}."
                )),
            },
            Phase::Readying(meshed) => self.ready(*meshed),
            Phase::Starting(ready) => match DemoStudy::start_part(*ready) {
                Ok(study) => self.phase = Phase::Study(Box::new(study)),
                Err(error) => self.refuse_solver(&error),
            },
            Phase::Study(mut study) => {
                study.step(budget);
                self.phase = Phase::Study(study);
            }
            Phase::Failed => {}
        }
        self.state()
    }

    fn refuse_mesh(&mut self, error: &solidfem::tetra::TetraError) {
        self.refuse(format!(
            "The mesher refused this part: {error}. Nothing was solved, so there is no result to \
             show for it."
        ));
    }

    fn begin_refine(&mut self, solid: SolidMesh) {
        let sizing = self.density.size;
        let size_function: SizeFn = Box::new(move |p| sizing.at(p));
        let options = RefineOptions {
            layers_through_walls: self.part.demo_layers.unwrap_or(self.case.layers),
            ..RefineOptions::default()
        };
        match Refiner::new(solid, size_function, options) {
            Ok(refiner) => self.phase = Phase::Refining(Box::new(refiner)),
            Err(error) => self.refuse_mesh(&error),
        }
    }

    /// Marks the part and gets its solve ready, keeping the mesh for a later material change.
    fn ready(&mut self, meshed: MeshedPart) {
        self.held_points = held_points(&meshed);
        self.load_arrows = load_arrows(&meshed, self.part.load_disk);
        let start = self.part_start(&meshed);
        match DemoStudy::ready_part(start) {
            Ok(ready) => {
                self.meshed = Some(meshed);
                self.phase = Phase::Starting(Box::new(ready));
            }
            Err(error) => self.refuse_solver(&error),
        }
    }

    fn refuse_solver(&mut self, error: &ModelError) {
        self.refuse(format!(
            "The solver refused this part before solving it: {error}."
        ));
    }

    fn part_start(&self, meshed: &MeshedPart) -> PartStart {
        let mut model = Model::new(meshed.mesh.clone(), self.material.material);
        model.set_quadrature(QUADRATURE);
        for node in &meshed.supports.fixed {
            model.fix_node(*node);
        }
        for (face, traction) in &meshed.supports.tractions {
            model.add_traction(*face, *traction);
        }
        for (node, axis, value) in &meshed.supports.prescribed {
            model.prescribe_displacement(*node, *axis, *value);
        }
        let (excluded, excluded_elements) = excluded_from_peak(meshed);
        PartStart {
            mesh: meshed.mesh.clone(),
            model,
            material: self.material.material,
            yield_strength: self.material.yield_strength,
            size: largest_extent(&meshed.mesh),
            excluded,
            excluded_elements,
            load_scale: self.load_scale,
            preconditioner: if self.multigrid {
                PreconditionerChoice::Multigrid(MultigridOptions::default())
            } else {
                PreconditionerChoice::Jacobi
            },
        }
    }
}

/// The largest bounding-box side of a mesh.
fn largest_extent(mesh: &QuadraticTetMesh) -> f64 {
    let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in mesh.nodes() {
        for axis in 0..3 {
            low[axis] = low[axis].min(p[axis]);
            high[axis] = high[axis].max(p[axis]);
        }
    }
    (high[0] - low[0])
        .max(high[1] - low[1])
        .max(high[2] - low[2])
}

/// The region a full fixture's stress singularity reaches, as (nodes, elements): every element
/// touching a fixed node and all of its nodes. Rollers (one direction held) are not singular, and
/// a real peak such as NAFEMS LE1's point D sits on one, so they stay in.
fn excluded_from_peak(meshed: &MeshedPart) -> (Vec<bool>, Vec<bool>) {
    let mut fixed = vec![false; meshed.mesh.nodes().len()];
    for node in &meshed.supports.fixed {
        fixed[*node as usize] = true;
    }
    let mut nodes = fixed.clone();
    let mut elements = vec![false; meshed.mesh.tets().len()];
    for (element, tet) in meshed.mesh.tets().iter().enumerate() {
        if tet.iter().any(|node| fixed[*node as usize]) {
            elements[element] = true;
            for node in tet {
                nodes[*node as usize] = true;
            }
        }
    }
    (nodes, elements)
}

/// At most `cap` of `points`, spread evenly over the region they cover: one per cell of a grid
/// coarsened until the occupied cells fit the cap, the point nearest its cell's centre, in index
/// order. Taking every Nth point instead would follow the mesher's numbering, which clusters.
pub fn spread_evenly(points: &[[f64; 3]], cap: usize) -> Vec<usize> {
    if points.len() <= cap {
        return (0..points.len()).collect();
    }
    let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in points {
        for axis in 0..3 {
            low[axis] = low[axis].min(p[axis]);
            high[axis] = high[axis].max(p[axis]);
        }
    }
    let extent = (high[0] - low[0])
        .max(high[1] - low[1])
        .max(high[2] - low[2]);
    // Starts near the spacing `cap` points would have over a square of that side.
    let mut cell = extent / (cap as f64).sqrt();
    loop {
        let mut cells: BTreeMap<[i64; 3], (f64, usize)> = BTreeMap::new();
        for (index, p) in points.iter().enumerate() {
            let mut key = [0i64; 3];
            let mut distance = 0.0;
            for axis in 0..3 {
                // Saturates harmlessly: the offset over the cell is at most about sqrt(cap).
                let at = ((p[axis] - low[axis]) / cell).floor();
                key[axis] = at as i64;
                let centre = low[axis] + (at + 0.5) * cell;
                distance += (p[axis] - centre) * (p[axis] - centre);
            }
            let slot = cells.entry(key).or_insert((distance, index));
            if distance < slot.0 {
                *slot = (distance, index);
            }
        }
        if cells.len() <= cap {
            let mut chosen: Vec<usize> = cells.values().map(|(_, index)| *index).collect();
            chosen.sort_unstable();
            return chosen;
        }
        cell *= 1.25;
    }
}

/// The positions of up to [`MAX_HELD_MARKERS`] of the nodes held in any direction by a zero
/// displacement or a full fixture, spread evenly over the held region.
fn held_points(meshed: &MeshedPart) -> Vec<f32> {
    let mut held = vec![false; meshed.mesh.nodes().len()];
    for node in &meshed.supports.fixed {
        held[*node as usize] = true;
    }
    for (node, _, value) in &meshed.supports.prescribed {
        if *value == 0.0 {
            held[*node as usize] = true;
        }
    }
    let positions: Vec<[f64; 3]> = held
        .iter()
        .enumerate()
        .filter(|(_, is_held)| **is_held)
        .map(|(node, _)| meshed.mesh.nodes()[node])
        .collect();
    let mut points = Vec::new();
    for index in spread_evenly(&positions, MAX_HELD_MARKERS) {
        let p = positions[index];
        points.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32]);
    }
    points
}

/// Where arrows land on a disk load: a triangular grid at half the radius, inside the disk.
fn disk_landing_points(centre: [f64; 2], radius: f64, z: f64) -> Vec<[f64; 3]> {
    let spacing = 0.5 * radius;
    let row_height = spacing * 3.0_f64.sqrt() / 2.0;
    let mut points = Vec::new();
    for row in -2i32..=2 {
        let shift = if row.rem_euclid(2) == 1 { 0.5 } else { 0.0 };
        for column in -3i32..=3 {
            let x = (f64::from(column) + shift) * spacing;
            let y = f64::from(row) * row_height;
            if (x * x + y * y).sqrt() <= radius {
                points.push([centre[0] + x, centre[1] + y, z]);
            }
        }
    }
    points
}

/// Up to [`MAX_LOAD_ARROWS`] arrows spread evenly over the loaded faces and over the nodes with
/// an enforced displacement, or inside `load_disk` when the part has one, each a tail, a unit direction and a length: seven numbers. A load
/// pressing into the part has its arrow outside the surface, pointing at it; a pull starts on
/// the surface and points away. Each arrow is a twelfth of the part's largest extent long.
fn load_arrows(meshed: &MeshedPart, load_disk: Option<([f64; 2], f64)>) -> Vec<f32> {
    let mesh = &meshed.mesh;
    let length = (largest_extent(mesh) / 12.0) as f32;
    let mut sources: Vec<([f64; 3], [f64; 3], bool)> = meshed
        .supports
        .tractions
        .iter()
        .map(|(face, traction)| {
            let corners =
                [face.nodes[0], face.nodes[1], face.nodes[2]].map(|n| mesh.nodes()[n as usize]);
            let mut centre = [0.0; 3];
            for p in &corners {
                centre = [centre[0] + p[0], centre[1] + p[1], centre[2] + p[2]];
            }
            let edge_u = [
                corners[1][0] - corners[0][0],
                corners[1][1] - corners[0][1],
                corners[1][2] - corners[0][2],
            ];
            let edge_v = [
                corners[2][0] - corners[0][0],
                corners[2][1] - corners[0][1],
                corners[2][2] - corners[0][2],
            ];
            let normal = [
                edge_u[1] * edge_v[2] - edge_u[2] * edge_v[1],
                edge_u[2] * edge_v[0] - edge_u[0] * edge_v[2],
                edge_u[0] * edge_v[1] - edge_u[1] * edge_v[0],
            ];
            let pressing =
                normal[0] * traction[0] + normal[1] * traction[1] + normal[2] * traction[2] < 0.0;
            (centre.map(|c| c / 3.0), *traction, pressing)
        })
        .collect();
    for (node, axis, value) in &meshed.supports.prescribed {
        if *value != 0.0 {
            let mut direction = [0.0; 3];
            direction[*axis] = *value;
            sources.push((mesh.nodes()[*node as usize], direction, true));
        }
    }
    // A disk smaller than its faces is drawn where the load is, not at those faces' centroids.
    if let (Some((centre, radius)), Some(first)) = (load_disk, sources.first().copied()) {
        sources = disk_landing_points(centre, radius, first.0[2])
            .into_iter()
            .map(|point| (point, first.1, first.2))
            .collect();
    }
    let positions: Vec<[f64; 3]> = sources.iter().map(|source| source.0).collect();
    let mut arrows = Vec::new();
    for (position, direction, pressing) in spread_evenly(&positions, MAX_LOAD_ARROWS)
        .into_iter()
        .map(|index| &sources[index])
    {
        let magnitude = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        if magnitude <= 0.0 {
            continue;
        }
        let unit = direction.map(|d| (d / magnitude) as f32);
        let tail = if *pressing {
            [
                position[0] as f32 - unit[0] * length,
                position[1] as f32 - unit[1] * length,
                position[2] as f32 - unit[2] * length,
            ]
        } else {
            [position[0] as f32, position[1] as f32, position[2] as f32]
        };
        arrows.extend_from_slice(&[tail[0], tail[1], tail[2], unit[0], unit[1], unit[2], length]);
    }
    arrows
}
