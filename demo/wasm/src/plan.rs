//! Turns a request into everything a solve needs: the mesh, supports and loads, and the textbook
//! comparison the page shows beside the result. The formulas are first-principles beam theory on
//! the section numbers `section.rs` derives, and each compared value says where it was measured.
//!
//! What is compared where, stated once:
//! - Push deflection: the mean z displacement over the loaded section (the whole section plane
//!   under the load, so the local indentation under the load has averaged out). Textbook:
//!   Timoshenko `F L^3 / (3 E I) + F L / (k G A)` for the end-loaded bar;
//!   for the bar fixed at both ends and loaded on a band of half-width `a` at mid-span,
//!   Timoshenko bending by superposing the unit-load deflection at mid-span
//!   `g(s) = s^2 (3L - 4s) / (48 E I)` over the band plus shear `F (L - a) / (4 k G A)`.
//! - Push stress: the outer-fibre bending stress from a straight-line fit of the sigma_xx
//!   integration values across a section-spanning slab, at a station away from the clamp and the
//!   load (the clamp's corners are a singularity, so the clamp itself is never compared).
//!   Textbook `M c / I`, where `M` is `F (L - x)` for the end-loaded bar and, for the
//!   both-ends bar, `M(x) = M0 + F x / 2` with `M0 = -F L / 8 + F a^2 / (6 L)`, the clamp moment
//!   of the band load from equal end slopes.
//! - Twist: the twist rate between two sections away from both ends (end effects decay
//!   exponentially), against Saint-Venant `T / (G J)`; a bar fixed at both ends carries half the
//!   torque in each span by symmetry, `(T / 2) / (G J)`. The closed tube has no closed-form
//!   torsion constant and returns no textbook answer: thin-walled formulas misjudge a chunky wall.

use solidfem::elastic::{point_geometry, Material, Model, ModelError, PointGeometry};
use solidfem::mesh::{BoundaryFace, MeshError, QuadraticTetMesh};

use crate::catalog::{LoadKind, RequestError, Restraint, StudyRequest, MATERIALS};
use crate::part::{build_mesh, kept_cells, plan_grid, PartGrid};
use crate::section::Section;

/// Maximum elements a study may have: the largest mesh any offered slider position needs, so a
/// position inside the catalog's ranges is never refused (a test walks every slider to hold that).
/// The time limit is set by the ranges, not by this number: `tests/demo/timing.rs` times the
/// slowest offered studies in the native release build, the proxy for a browser run.
pub const MAX_ELEMENTS: usize = 3_072;

/// Six tetrahedra per kept hexahedral cell.
const TETS_PER_CELL: usize = 6;

/// What a start request can be refused with.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanError {
    Request(RequestError),
    /// The mesh needed for this request would exceed the size the demo can solve in time.
    TooLarge {
        elements: usize,
        limit: usize,
    },
    Mesh(MeshError),
    /// A validated request failed to build a model; a defect of the demo, named for the page.
    Model(ModelError),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::Request(error) => write!(f, "{error}"),
            PlanError::TooLarge { elements, limit } => write!(
                f,
                "this would need {elements} elements, more than the {limit} this demo can solve \
                 in a moment: choose a shorter part or a coarser mesh"
            ),
            PlanError::Mesh(error) => write!(f, "{error}"),
            PlanError::Model(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for PlanError {}

/// What a compared value measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantity {
    Deflection,
    BendingStress,
    TwistRate,
}

/// How a compared value is measured from a displacement field.
#[derive(Debug, Clone)]
pub enum Probe {
    /// Mean z displacement over these nodes, in the push direction.
    Deflection { nodes: Vec<usize> },
    /// Straight-line fit of sigma_xx over the section slab, read at the outer fibre.
    Fibre {
        points: Vec<StationPoint>,
        centre_z: f64,
        station_x: f64,
        half_height: f64,
    },
    /// Rate of twist about the axis through the section centre between two node planes.
    TwistRate {
        first: Vec<usize>,
        second: Vec<usize>,
        first_x: f64,
        second_x: f64,
        centre: [f64; 2],
    },
}

/// One integration point of a tetrahedron near a station.
#[derive(Debug, Clone)]
pub struct StationPoint {
    pub element: u32,
    pub geometry: PointGeometry,
}

/// One value the page compares against the textbook.
#[derive(Debug, Clone)]
pub struct Comparison {
    pub quantity: Quantity,
    pub unit: &'static str,
    /// Where it is measured, in plain words for the page.
    pub place: String,
    pub textbook: f64,
    pub probe: Probe,
}

/// Why a study has no textbook column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoTextbookAnswer {
    /// Saint-Venant has no closed form for the torsion of this closed section.
    TubeTwist,
}

impl NoTextbookAnswer {
    /// A sentence for the page.
    pub fn explanation(self) -> &'static str {
        match self {
            NoTextbookAnswer::TubeTwist => {
                "A closed thick-walled tube in twist has no simple textbook formula, so only the \
                 simulated result is shown."
            }
        }
    }
}

/// The textbook side of a study.
#[derive(Debug, Clone)]
pub enum Textbook {
    Answers(Vec<Comparison>),
    NoTextbook(NoTextbookAnswer),
}

/// A request turned into what the solve needs.
pub struct PlannedStudy {
    pub request: StudyRequest,
    pub material: Material,
    pub yield_strength: f64,
    pub mesh: QuadraticTetMesh,
    pub model: Model<Material>,
    pub textbook: Textbook,
    /// x positions of the clamped planes: the peak stress skips nodes near them.
    pub clamps: Vec<f64>,
    /// How far from a clamp nodes are left out of the peak: the larger section side, past which
    /// the clamp's corner singularity has decayed away.
    pub clamp_zone: f64,
}

const ON_PLANE_RELATIVE: f64 = 1e-9;

/// Plans a request, refusing a bad or oversized one before any mesh is built.
pub fn plan_study(request: &StudyRequest) -> Result<PlannedStudy, PlanError> {
    plan_study_within(request, MAX_ELEMENTS)
}

/// How many elements a validated request's mesh has.
pub fn planned_elements(request: &StudyRequest) -> usize {
    kept_cells(&plan_grid(request)) * TETS_PER_CELL
}

/// [`plan_study`] against an explicit element limit.
pub fn plan_study_within(request: &StudyRequest, limit: usize) -> Result<PlannedStudy, PlanError> {
    request.validate().map_err(PlanError::Request)?;
    let grid = plan_grid(request);
    let elements = kept_cells(&grid) * TETS_PER_CELL;
    if elements > limit {
        return Err(PlanError::TooLarge { elements, limit });
    }
    let mesh = build_mesh(&grid).map_err(PlanError::Mesh)?;
    let picked = MATERIALS[request.material];
    let material = Material {
        youngs_modulus: picked.youngs_modulus,
        poisson_ratio: picked.poisson_ratio,
    };
    let section = request.section_shape();
    let length = request.length;

    let mut model = Model::new(mesh.clone(), material);
    let tolerance = ON_PLANE_RELATIVE * length;
    let mut clamps = vec![0.0];
    if request.restraint == Restraint::FixedBothEnds {
        clamps.push(length);
    }
    for (node, p) in mesh.nodes().iter().enumerate() {
        if clamps.iter().any(|x| (p[0] - x).abs() < tolerance) {
            model.fix_node(node as u32);
        }
    }
    let ctx = Context {
        grid: &grid,
        mesh: &mesh,
        section: &section,
        request,
        material,
        tolerance,
    };
    let textbook = match request.load {
        LoadKind::Push => ctx.apply_push(&mut model),
        LoadKind::Twist => ctx.apply_twist(&mut model),
    };
    Ok(PlannedStudy {
        request: *request,
        material,
        yield_strength: picked.yield_strength,
        model,
        textbook,
        clamps,
        clamp_zone: section.width().max(section.height()),
        mesh,
    })
}

struct Context<'a> {
    grid: &'a PartGrid,
    mesh: &'a QuadraticTetMesh,
    section: &'a Section,
    request: &'a StudyRequest,
    material: Material,
    tolerance: f64,
}

impl Context<'_> {
    fn node(&self, n: u32) -> [f64; 3] {
        self.mesh.nodes()[n as usize]
    }

    /// Boundary faces all of whose corners satisfy `pred`.
    fn faces_where(&self, pred: impl Fn([f64; 3]) -> bool) -> Vec<BoundaryFace> {
        self.mesh
            .boundary_faces()
            .into_iter()
            .filter(|f| f.nodes[..3].iter().all(|n| pred(self.node(*n))))
            .collect()
    }

    /// The three corners of a face.
    fn corners(&self, face: &BoundaryFace) -> [[f64; 3]; 3] {
        [face.nodes[0], face.nodes[1], face.nodes[2]].map(|n| self.node(n))
    }

    /// Node indices on the plane at `x`.
    fn plane_nodes(&self, x: f64) -> Vec<usize> {
        self.mesh
            .nodes()
            .iter()
            .enumerate()
            .filter(|(_, p)| (p[0] - x).abs() < self.tolerance)
            .map(|(i, _)| i)
            .collect()
    }

    /// The grid plane `numerator / 8` of the way along the bar. The cell count along the bar is
    /// a multiple of 8, so this is an exact grid line.
    fn station(&self, numerator: usize) -> f64 {
        self.grid.x.0[self.grid.x.cells() * numerator / 8]
    }

    /// The straight-line sigma_xx probe over the slab of elements one cell to each side of the
    /// section at `x`.
    fn fibre_probe(&self, x: f64) -> Probe {
        let dx = self.grid.x.0[1] - self.grid.x.0[0];
        let mut points = Vec::new();
        for (element, tet) in self.mesh.tets().iter().enumerate() {
            let nodes = tet.map(|n| self.mesh.nodes()[n as usize]);
            let centroid = nodes[..4].iter().fold(0.0, |sum, p| sum + p[0]) * 0.25;
            if (centroid - x).abs() >= dx {
                continue;
            }
            // The grid's elements are valid and straight, so geometry at the rule's points exists.
            for (index, q) in solidfem::elastic::Quadrature::default()
                .points()
                .iter()
                .enumerate()
            {
                if let Ok(geometry) = point_geometry(&nodes, q.natural, index) {
                    points.push(StationPoint {
                        element: element as u32,
                        geometry,
                    });
                }
            }
        }
        Probe::Fibre {
            points,
            centre_z: 0.5 * self.section.height(),
            station_x: x,
            half_height: 0.5 * self.section.height(),
        }
    }

    fn apply_push(&self, model: &mut Model<Material>) -> Textbook {
        let force = self.request.magnitude;
        let length = self.request.length;
        let e = self.material.youngs_modulus;
        let g = e / (2.0 * (1.0 + self.material.poisson_ratio));
        let inertia = self.section.inertia();
        let area = self.section.area();
        let kappa = self.section.shear_coefficient();
        let (deflection, moment, stress_x, place_loaded) = match self.request.restraint {
            Restraint::FixedOneEnd => {
                let end_faces = self.faces_where(|p| (p[0] - length).abs() < self.tolerance);
                let total: f64 = end_faces.iter().map(|f| face_area(self.corners(f))).sum();
                let traction = force / total;
                for face in end_faces {
                    model.add_traction(face, [0.0, 0.0, -traction]);
                }
                let stress_x = self.station(4);
                let deflection = force * length * length * length / (3.0 * e * inertia)
                    + force * length / (kappa * g * area);
                let moment = force * (length - stress_x);
                (
                    deflection,
                    moment,
                    stress_x,
                    "mean vertical displacement of the free end section".to_string(),
                )
            }
            Restraint::FixedBothEnds => {
                let mid = self.station(4);
                // One cell either side of the mid-span plane carries the push on the top surface.
                let half_band = self.grid.x.0[1] - self.grid.x.0[0];
                let top = self.section.height();
                let band: Vec<BoundaryFace> = self.faces_where(|p| {
                    (p[2] - top).abs() < self.tolerance
                        && (p[0] - mid).abs() <= half_band + self.tolerance
                });
                let total: f64 = band.iter().map(|f| face_area(self.corners(f))).sum();
                let traction = force / total;
                for face in band {
                    model.add_traction(face, [0.0, 0.0, -traction]);
                }
                let a = half_band;
                let deflection = force * band_bending_compliance(length, a) / (e * inertia)
                    + force * (length - a) / (4.0 * kappa * g * area);
                let stress_x = self.station(3);
                let m0 = -force * length / 8.0 + force * a * a / (6.0 * length);
                (
                    deflection,
                    m0 + 0.5 * force * stress_x,
                    stress_x,
                    "mean vertical displacement of the mid-span section".to_string(),
                )
            }
        };
        let load_x = match self.request.restraint {
            Restraint::FixedOneEnd => length,
            Restraint::FixedBothEnds => self.station(4),
        };
        let half_height = 0.5 * self.section.height();
        Textbook::Answers(vec![
            Comparison {
                quantity: Quantity::Deflection,
                unit: "mm",
                place: place_loaded,
                textbook: deflection,
                probe: Probe::Deflection {
                    nodes: self.plane_nodes(load_x),
                },
            },
            Comparison {
                quantity: Quantity::BendingStress,
                unit: "MPa",
                place: format!(
                    "outer-fibre bending stress at x = {stress_x:.1} mm, fitted across the \
                     section away from the clamp and the load"
                ),
                textbook: moment.abs() * half_height / inertia,
                probe: self.fibre_probe(stress_x),
            },
        ])
    }

    fn apply_twist(&self, model: &mut Model<Material>) -> Textbook {
        let torque = self.request.magnitude;
        let length = self.request.length;
        let g = self.material.youngs_modulus / (2.0 * (1.0 + self.material.poisson_ratio));
        let centre = [0.5 * self.section.width(), 0.5 * self.section.height()];
        let (faces_raw, tractions): (Vec<BoundaryFace>, Vec<[f64; 3]>) =
            match self.request.restraint {
                Restraint::FixedOneEnd => {
                    let faces = self.faces_where(|p| (p[0] - length).abs() < self.tolerance);
                    let tractions = faces
                        .iter()
                        .map(|f| {
                            let c = centroid(self.corners(f));
                            [0.0, -(c[2] - centre[1]), c[1] - centre[0]]
                        })
                        .collect();
                    (faces, tractions)
                }
                Restraint::FixedBothEnds => {
                    let mid = self.station(4);
                    let half_band = self.grid.x.0[1] - self.grid.x.0[0];
                    let (w, h) = (self.section.width(), self.section.height());
                    let band =
                        self.faces_where(|p| (p[0] - mid).abs() <= half_band + self.tolerance);
                    let faces: Vec<BoundaryFace> = band
                        .into_iter()
                        .filter(|f| {
                            let n = unit_normal(self.corners(f));
                            let c = centroid(self.corners(f));
                            n[0].abs() < 0.5
                                && (c[1].abs() < self.tolerance
                                    || (c[1] - w).abs() < self.tolerance
                                    || c[2].abs() < self.tolerance
                                    || (c[2] - h).abs() < self.tolerance)
                        })
                        .collect();
                    let tractions = faces
                        .iter()
                        .map(|f| {
                            let n = unit_normal(self.corners(f));
                            // The tangent around the bar axis: x cross the outward normal.
                            [0.0, -n[2], n[1]]
                        })
                        .collect();
                    (faces, tractions)
                }
            };
        // The delivered torque per unit strength, from what each uniform face traction applies: a
        // uniform traction on a straight face has its resultant at the face's centroid.
        let mut unit_torque = 0.0;
        for (face, t) in faces_raw.iter().zip(&tractions) {
            let c = centroid(self.corners(face));
            let area = face_area(self.corners(face));
            unit_torque += area * ((c[1] - centre[0]) * t[2] - (c[2] - centre[1]) * t[1]);
        }
        let scale = torque / unit_torque;
        for (face, t) in faces_raw.into_iter().zip(tractions) {
            model.add_traction(face, [t[0] * scale, t[1] * scale, t[2] * scale]);
        }
        let Some(j) = self.section.torsion_constant() else {
            return Textbook::NoTextbook(NoTextbookAnswer::TubeTwist);
        };
        let (first_x, second_x, carried) = match self.request.restraint {
            // Away from both the clamp and the loaded end: a quarter and three quarters along.
            Restraint::FixedOneEnd => (self.station(2), self.station(6), torque),
            // Each half carries half the torque; measured inside the half from the clamp to the
            // mid-span load, away from both.
            Restraint::FixedBothEnds => (self.station(1), self.station(3), 0.5 * torque),
        };
        Textbook::Answers(vec![Comparison {
            quantity: Quantity::TwistRate,
            unit: "rad/mm",
            place: format!(
                "twist rate between the sections at x = {first_x:.1} mm and x = {second_x:.1} mm, \
                 away from both ends"
            ),
            textbook: carried / (g * j),
            probe: Probe::TwistRate {
                first: self.plane_nodes(first_x),
                second: self.plane_nodes(second_x),
                first_x,
                second_x,
                centre,
            },
        }])
    }
}

/// The bending part of the mid-span deflection of a fixed-fixed beam per unit `F / (E I)`,
/// loaded uniformly over `[L/2 - a, L/2 + a]`: the unit-load influence
/// `g(s) = s^2 (3L - 4s) / (48 E I)` averaged over the band. Simpson's rule on the cubic is exact.
fn band_bending_compliance(length: f64, a: f64) -> f64 {
    let g = |s: f64| s * s * (3.0 * length - 4.0 * s) / 48.0;
    let start = 0.5 * length - a;
    let middle = 0.5 * length - 0.5 * a;
    let end = 0.5 * length;
    (g(start) + 4.0 * g(middle) + g(end)) / 6.0
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn face_area(corners: [[f64; 3]; 3]) -> f64 {
    let n = cross(sub(corners[1], corners[0]), sub(corners[2], corners[0]));
    0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
}

/// Unit normal from a face's corners: outward by the boundary face convention.
fn unit_normal(corners: [[f64; 3]; 3]) -> [f64; 3] {
    let n = cross(sub(corners[1], corners[0]), sub(corners[2], corners[0]));
    let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    [n[0] / length, n[1] / length, n[2] / length]
}

fn centroid(corners: [[f64; 3]; 3]) -> [f64; 3] {
    [
        (corners[0][0] + corners[1][0] + corners[2][0]) / 3.0,
        (corners[0][1] + corners[1][1] + corners[2][1]) / 3.0,
        (corners[0][2] + corners[1][2] + corners[2][2]) / 3.0,
    ]
}
