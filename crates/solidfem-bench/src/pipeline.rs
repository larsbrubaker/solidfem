//! One benchmark run: sample the shape's distance field, mesh it with Quartet, upgrade to 10-node
//! elements, hold and load the named regions, and solve through the study's `step` slices. A
//! capped run reports the iterations and residual reached. Memory is estimated from the sizes of
//! the main allocations, and a run over the caller's limit is refused before assembly.

use quartet_rust::{make_tet_mesh, SdfGrid};
use solidfem_elastic::{
    Clock, Material, Model, Report, Solution, StaticStudy, StepOutcome, StudyOptions,
};
use solidfem_mesh::{BoundaryFace, QuadraticTetMesh, TetMesh};

/// Cells of empty grid the mesher needs around the shape.
const MARGIN_CELLS: usize = 8;

/// Fraction of `dx` within which a node counts as on a named flat face.
pub const ON_FACE_FRACTION: f64 = 0.02;

/// Assumed average count of coupled node pairs per node of a 10-node mesh, used to estimate the
/// matrix size before it is built. An assumption, not a measurement: the printed memory is an estimate.
pub const NODE_PAIRS_PER_NODE: u64 = 40;

/// What a run builds, described by closures so the same runner serves every shape.
pub struct Case {
    /// The name printed in the tables.
    pub name: String,
    /// Signed distance of the part, negative inside.
    pub sdf: Box<dyn Fn([f64; 3]) -> f64>,
    /// The part's bounding box, low corner.
    pub low: [f64; 3],
    /// The part's bounding box, high corner.
    pub high: [f64; 3],
    /// Grid spacing.
    pub dx: f64,
    /// True for a node that is held, given the on-face tolerance.
    pub fixed: Box<dyn Fn([f64; 3], f64) -> bool>,
    /// True for a boundary-face corner inside the loaded region, given the on-face tolerance.
    pub loaded: Box<dyn Fn([f64; 3], f64) -> bool>,
    /// The total force applied over the loaded faces, so it is independent of the mesh.
    pub total_force: [f64; 3],
    /// Offset of the sampling grid against the part, in cells, so a part is not accidentally aligned
    /// with the mesher.s lattice.
    pub grid_shift_cells: f64,
    /// The part's material.
    pub material: Material,
}

/// Why a run did not produce numbers.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// The memory estimate exceeds the caller's limit.
    TooMuchMemory { estimated_gib: f64 },
    /// The mesher or model refused the input, or the solve failed.
    Failed(String),
}

/// A case meshed and assembled, ready to step.
pub struct Prepared {
    /// The study to step.
    pub study: StaticStudy<Material>,
    /// The mesh, for evaluating the result.
    pub mesh: QuadraticTetMesh,
    /// Grid spacing used.
    pub dx: f64,
    /// Grid cells sampled.
    pub cells: u64,
    /// Linear tetrahedra.
    pub tets: usize,
    /// Mesh nodes after the 10-node upgrade.
    pub nodes: usize,
    /// Seconds in the mesher and the upgrade.
    pub mesh_seconds: f64,
    /// Estimated peak bytes of the main allocations.
    pub estimated_bytes: u64,
    /// The largest dihedral in the mesh, degrees.
    pub max_dihedral: f64,
}

/// The main allocations' total bytes, before the matrix exists: positions, elements, the symbolic
/// sort's packed pairs, the sparse matrix, the solver's seven vectors and the results.
pub fn estimate_bytes(tets: u64, nodes: u64, points: u64) -> u64 {
    let dofs = 3 * nodes;
    let pairs = NODE_PAIRS_PER_NODE * nodes;
    let nonzeros = 9 * pairs;
    let mesh = nodes * 24 + tets * (40 + 16);
    let symbolic_sort = 100 * tets * 8;
    let matrix = nonzeros * (8 + 4) + (dofs + 1) * 8;
    let solver_vectors = 7 * dofs * 8;
    let results = tets * points * 88 + nodes * 8 + dofs * 8;
    mesh + symbolic_sort + matrix + solver_vectors + results
}

/// Samples, meshes and assembles `case`; the clock is read for phase times.
pub fn prepare(
    case: &Case,
    new_clock: &mut dyn FnMut() -> Clock,
    byte_limit: u64,
) -> Result<Prepared, Refusal> {
    let mut timer = new_clock();
    let start = timer();
    let grid = sample_grid(case)?;
    let cells = {
        let dims = grid.dims();
        (dims[0] as u64 - 1) * (dims[1] as u64 - 1) * (dims[2] as u64 - 1)
    };
    let quartet = make_tet_mesh(&grid, || false).map_err(|e| Refusal::Failed(format!("{e}")))?;
    drop(grid);
    let max_dihedral = quartet.max_dihedral_angle_degrees();
    let linear = TetMesh::new(quartet.vertices, quartet.tets)
        .map_err(|e| Refusal::Failed(format!("the mesher's output was refused: {e}")))?;
    let mesh = QuadraticTetMesh::from_linear(&linear);
    let mesh_seconds = timer() - start;

    let tets = mesh.tets().len();
    let nodes = mesh.nodes().len();
    let estimated_bytes = estimate_bytes(tets as u64, nodes as u64, 4);
    if estimated_bytes > byte_limit {
        return Err(Refusal::TooMuchMemory {
            estimated_gib: estimated_bytes as f64 / (1u64 << 30) as f64,
        });
    }

    let tolerance = ON_FACE_FRACTION * case.dx;
    let mut model = Model::new(mesh.clone(), case.material);
    let held: Vec<u32> = (0..nodes as u32)
        .filter(|n| (case.fixed)(mesh.nodes()[*n as usize], tolerance))
        .collect();
    for node in held {
        model.fix_node(node);
    }
    let faces: Vec<BoundaryFace> = mesh
        .boundary_faces()
        .into_iter()
        .filter(|f| {
            f.nodes[..3]
                .iter()
                .all(|n| (case.loaded)(mesh.nodes()[*n as usize], tolerance))
        })
        .collect();
    let area: f64 = faces.iter().map(|f| face_area(&mesh, f)).sum();
    if area <= 0.0 {
        return Err(Refusal::Failed(
            "no boundary face lies in the loaded region".to_string(),
        ));
    }
    let traction = case.total_force.map(|f| f / area);
    for face in faces {
        model.add_traction(face, traction);
    }

    let options = StudyOptions {
        clock: Some(new_clock()),
        ..StudyOptions::default()
    };
    let study = StaticStudy::new(model, options, &mut || false)
        .map_err(|e| Refusal::Failed(format!("{e}")))?;
    Ok(Prepared {
        study,
        mesh,
        dx: case.dx,
        cells,
        tets,
        nodes,
        mesh_seconds,
        estimated_bytes,
        max_dihedral,
    })
}

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ending {
    /// The solve converged and results were recovered.
    Converged,
    /// The wall-clock cap stopped the solve.
    Capped,
}

/// What was measured.
pub struct Measured {
    /// The study's sizes and times.
    pub report: Report,
    /// How the run ended.
    pub ending: Ending,
    /// Results, when the solve converged.
    pub solution: Option<Solution>,
    /// Peak of the per-node von Mises stress, or zero when capped.
    pub peak_von_mises: f64,
}

/// Advances the solve by `slice` iterations; `true` when there is more to do.
pub fn advance(prepared: &mut Prepared, slice: usize) -> Result<bool, Refusal> {
    match prepared
        .study
        .step(slice, &mut || false)
        .map_err(|e| Refusal::Failed(format!("{e}")))?
    {
        StepOutcome::Running(_) => Ok(true),
        StepOutcome::Converged(_) => Ok(false),
        StepOutcome::Cancelled(_) => Err(Refusal::Failed("cancelled".to_string())),
    }
}

/// Collects the outcome of a run that has stopped stepping.
pub fn finish(prepared: &mut Prepared, ending: Ending) -> Result<Measured, Refusal> {
    let solution = match ending {
        Ending::Converged => Some(
            prepared
                .study
                .solution(&mut || false)
                .map_err(|e| Refusal::Failed(format!("{e}")))?,
        ),
        Ending::Capped => None,
    };
    let peak_von_mises = solution.as_ref().map_or(0.0, |s| {
        s.nodal_von_mises.iter().cloned().fold(0.0, f64::max)
    });
    Ok(Measured {
        report: solution
            .as_ref()
            .map_or(prepared.study.report(), |s| s.report),
        ending,
        solution,
        peak_von_mises,
    })
}

fn sample_grid(case: &Case) -> Result<SdfGrid, Refusal> {
    let origin: [f64; 3] = std::array::from_fn(|axis| {
        case.low[axis] - (MARGIN_CELLS as f64 + case.grid_shift_cells) * case.dx
    });
    let dims: [usize; 3] = std::array::from_fn(|axis| {
        // The ceiling cannot overflow usize: bounds over dx give a modest cell count here.
        let span = ((case.high[axis] - case.low[axis]) / case.dx).ceil() as usize;
        span + 2 * MARGIN_CELLS + 1
    });
    let mut values = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let p = [
                    origin[0] + i as f64 * case.dx,
                    origin[1] + j as f64 * case.dx,
                    origin[2] + k as f64 * case.dx,
                ];
                values.push((case.sdf)(p));
            }
        }
    }
    SdfGrid::new(origin, case.dx, dims, values).map_err(|e| Refusal::Failed(format!("{e}")))
}

fn face_area(mesh: &QuadraticTetMesh, face: &BoundaryFace) -> f64 {
    let [a, b, c] = [face.nodes[0], face.nodes[1], face.nodes[2]].map(|n| mesh.nodes()[n as usize]);
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
}
