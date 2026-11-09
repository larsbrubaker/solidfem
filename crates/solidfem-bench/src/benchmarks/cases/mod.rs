//! The real-part benchmark cases, one file per part: what each part is, which surface files it uses and at what mesh
//! sizes, where it is held and loaded, what is measured, the reference it is compared with and its
//! limits. Each limit is derived in the comment on its judge, written before the case first ran.
//!
//! The surface files come from `crate::surfaces`, built on solidmesh's CSG, extrude, revolve and
//! `remesh_for_analysis` (`solidfem-surfaces` writes them). Each case's `Sizing` is the size
//! function that generator was given, so the interior of the volume mesh grades the same way as
//! its surface; the `surfaces` tests hold the two together. Selection regions are
//! geometric, with a tolerance above the rounding of the remeshed coordinates; a chord across a
//! curved surface is allowed for where the region is curved.

use solidfem::elastic::{
    point_geometry, strain_at, stress_at, von_mises, Material, PointGeometry, Solution, NODE_COUNT,
    NODE_NATURAL,
};
use solidfem::mesh::{BoundaryFace, QuadraticTetMesh};

use super::pipeline::{face_area, face_normal, DensityResult, Supports};
use super::verdict::{
    classify_peak, last_change, within, within_limit, worst_of, PeakBehaviour, Verdict,
};

/// Young's modulus for every case, MPa; lengths are mm and forces N.
const E: f64 = 200_000.0;
/// How far from a flat region a surface vertex may lie and still belong to it.
const PLANE_TOL: f64 = 1e-6;

/// A case's mesh-size function, the same one its surface generator used. Copied into the closure
/// the refiner keeps, and a plain value so it can be stated beside the surface file it matches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sizing {
    /// One edge length everywhere.
    Uniform(f64),
    /// `near` within `r0` of `centre` in the xy plane, growing at slope `grade` out to `far`.
    AroundPoint {
        centre: [f64; 2],
        r0: f64,
        near: f64,
        grade: f64,
        far: f64,
    },
}

impl Sizing {
    /// The wanted edge length at `p`.
    pub fn at(self, p: [f64; 3]) -> f64 {
        match self {
            Sizing::Uniform(h) => h,
            Sizing::AroundPoint {
                centre,
                r0,
                near,
                grade,
                far,
            } => {
                let d = ((p[0] - centre[0]) * (p[0] - centre[0])
                    + (p[1] - centre[1]) * (p[1] - centre[1]))
                    .sqrt();
                (near + grade * (d - r0).max(0.0)).min(far)
            }
        }
    }
}

/// One mesh density of a case.
#[derive(Debug, Clone)]
pub struct Density {
    /// A short label.
    pub label: String,
    /// The surface file's name in the generator's output directory.
    pub surface_file: String,
    /// The size function for the volume mesh.
    pub size: Sizing,
    /// The characteristic size this density is measured at, for convergence rates.
    pub series_size: f64,
}

/// Measures a solved part: named quantities, in the order the table shows them.
pub type Measure = fn(&QuadraticTetMesh, &Solution) -> Result<Vec<(String, f64)>, String>;

/// A benchmark case.
pub struct Case {
    /// The id it is selected by on the command line.
    pub id: &'static str,
    /// A line for the report: what the part is.
    pub title: &'static str,
    /// The material.
    pub material: Material,
    /// The densities, coarse to fine.
    pub densities: Vec<Density>,
    /// Elements requested through thin walls.
    pub layers: u32,
    /// Picks the supports and loads on a meshed part, by geometric region.
    pub apply: fn(&QuadraticTetMesh) -> Result<Supports, String>,
    /// Measures the quantities the judge reads, from a solved mesh.
    pub measure: Measure,
    /// A solve taking more conjugate gradient iterations than this fails the case by name.
    pub iteration_budget: Option<u64>,
    /// Compares the rows with the reference and the limits.
    pub judge: fn(&[DensityResult]) -> Verdict,
}

fn density(label: &str, file: &str, size: Sizing, series_size: f64) -> Density {
    Density {
        label: label.to_string(),
        surface_file: format!("{file}.txt"),
        size,
        series_size,
    }
}

/// `value` against `reference`, a named failure when off by more than the `limit` fraction.
pub fn check_error(value: f64, reference: f64, limit: f64, what: &str) -> Result<f64, String> {
    let error = (value - reference).abs() / reference.abs();
    if !within_limit(error, limit) {
        return Err(format!(
            "{what}: measured {value:.6e}, reference {reference:.6e}, error {:.3}% over the {:.3}% limit",
            100.0 * error,
            100.0 * limit
        ));
    }
    Ok(error)
}

fn series(rows: &[DensityResult], name: &str) -> Vec<f64> {
    rows.iter().map(|row| row.value(name)).collect()
}

fn sizes(rows: &[DensityResult]) -> Vec<f64> {
    rows.iter().map(|row| row.size).collect()
}

fn stable(rows: &[DensityResult], name: &str, limit: f64, what: &str) -> Result<(), String> {
    let values = series(rows, name);
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{what}: a value in the series is not finite: {values:?}"
        ));
    }
    let change = last_change(&values).ok_or_else(|| format!("{what}: no last change"))?;
    if !within_limit(change, limit) {
        return Err(format!(
            "{what}: the last refinement moved it by {:.3}%, over the {:.3}% allowed; series {values:?}",
            100.0 * change,
            100.0 * limit
        ));
    }
    Ok(())
}

fn verdict_of(result: Result<(), String>) -> Verdict {
    match result {
        Ok(()) => Verdict::Pass,
        Err(reason) => Verdict::Fail(reason),
    }
}

/// Mean of `-u[axis]` over the nodes in the plane `coordinate == value`.
fn mean_deflection(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
    coordinate: usize,
    value: f64,
    axis: usize,
) -> Result<f64, String> {
    let (mut sum, mut count) = (0.0, 0usize);
    for (node, position) in mesh.nodes().iter().enumerate() {
        if (position[coordinate] - value).abs() <= PLANE_TOL {
            sum += -solution.displacements[node][axis];
            count += 1;
        }
    }
    if count == 0 {
        return Err(format!("no node on the plane x[{coordinate}] = {value}"));
    }
    Ok(sum / count as f64)
}

/// Boundary faces with every corner in the plane `coordinate == value`, and their total area.
fn plane_faces(mesh: &QuadraticTetMesh, coordinate: usize, value: f64) -> (Vec<BoundaryFace>, f64) {
    let faces: Vec<BoundaryFace> = mesh
        .boundary_faces()
        .into_iter()
        .filter(|face| {
            face.nodes[..3]
                .iter()
                .all(|n| (mesh.nodes()[*n as usize][coordinate] - value).abs() <= PLANE_TOL)
        })
        .collect();
    let area = faces.iter().map(|face| face_area(mesh, face)).sum();
    (faces, area)
}

/// All nodes on the plane `coordinate == value`.
fn plane_nodes(mesh: &QuadraticTetMesh, coordinate: usize, value: f64) -> Vec<u32> {
    (0..mesh.nodes().len() as u32)
        .filter(|node| (mesh.nodes()[*node as usize][coordinate] - value).abs() <= PLANE_TOL)
        .collect()
}

/// Refuses a selected region whose area is not what the geometry says it is, so a wrong selection
/// cannot quietly apply a different load.
fn check_area(area: f64, expected: f64, what: &str) -> Result<(), String> {
    // The selected faces are flat and exactly the region, so the only departure is rounding.
    if !within_limit((area - expected).abs(), 1e-6 * expected) {
        return Err(format!("{what}: selected area {area}, expected {expected}"));
    }
    Ok(())
}

/// The largest nodal von Mises stress at nodes within the (x, y) box `low..high`.
fn peak_in_box(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
    low: [f64; 2],
    high: [f64; 2],
) -> Result<f64, String> {
    let mut peak = f64::MIN;
    for (node, position) in mesh.nodes().iter().enumerate() {
        if position[0] >= low[0]
            && position[0] <= high[0]
            && position[1] >= low[1]
            && position[1] <= high[1]
        {
            peak = worst_of(peak, solution.nodal_von_mises[node]);
        }
    }
    if peak == f64::MIN {
        return Err(format!("no node in the box {low:?}..{high:?}"));
    }
    Ok(peak)
}

fn steel(nu: f64) -> Material {
    Material {
        youngs_modulus: E,
        poisson_ratio: nu,
    }
}

/// A volume-weighted average over the incident elements of `value`, a tensor evaluated at the
/// node in each element, for each node in `targets`, in that order. The display von Mises is the
/// same recovery of a scalar; a benchmark comparing a stress component needs the tensor.
fn nodal_tensor(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
    targets: &[u32],
    value: impl Fn(&PointGeometry, &[[f64; 3]; NODE_COUNT]) -> [f64; 6],
) -> Result<Vec<[f64; 6]>, String> {
    let mut slot = vec![usize::MAX; mesh.nodes().len()];
    for (index, node) in targets.iter().enumerate() {
        slot[*node as usize] = index;
    }
    let mut sum = vec![[0.0; 6]; targets.len()];
    let mut weight = vec![0.0; targets.len()];
    for tet in mesh.tets() {
        if tet.iter().all(|n| slot[*n as usize] == usize::MAX) {
            continue;
        }
        let nodes = tet.map(|n| mesh.nodes()[n as usize]);
        let u = tet.map(|n| solution.displacements[n as usize]);
        for local in 0..NODE_COUNT {
            let target = slot[tet[local] as usize];
            if target == usize::MAX {
                continue;
            }
            let geometry = point_geometry(&nodes, NODE_NATURAL[local], local)
                .map_err(|fault| format!("nodal tensor: {fault:?}"))?;
            let tensor = value(&geometry, &u);
            for k in 0..6 {
                sum[target][k] += geometry.det_j * tensor[k];
            }
            weight[target] += geometry.det_j;
        }
    }
    for (index, total) in weight.iter().enumerate() {
        if *total <= 0.0 {
            return Err(format!("no element carries target node {}", targets[index]));
        }
        for value in sum[index].iter_mut() {
            *value /= *total;
        }
    }
    Ok(sum)
}

/// The stress tensor at the nodes `targets`, averaged over their elements by volume.
fn nodal_stress(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
    material: Material,
    targets: &[u32],
) -> Result<Vec<[f64; 6]>, String> {
    nodal_tensor(mesh, solution, targets, |geometry, u| {
        stress_at(&material, &strain_at(geometry, u))
    })
}

/// The smallest dihedral angle of the corners of a tetrahedron, in degrees.
fn min_dihedral_degrees(p: [[f64; 3]; 4]) -> f64 {
    const EDGES: [(usize, usize, usize, usize); 6] = [
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 3, 1, 2),
        (1, 2, 0, 3),
        (1, 3, 0, 2),
        (2, 3, 0, 1),
    ];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut smallest = 180.0_f64;
    for (a, b, c, d) in EDGES {
        let edge = sub(p[b], p[a]);
        let n1 = cross(edge, sub(p[c], p[a]));
        let n2 = cross(edge, sub(p[d], p[a]));
        let cosine = dot(n1, n2) / (dot(n1, n1) * dot(n2, n2)).sqrt();
        smallest = smallest.min(cosine.clamp(-1.0, 1.0).acos().to_degrees());
    }
    smallest
}

/// All nodes matching `keep`, in node order.
fn nodes_where(mesh: &QuadraticTetMesh, keep: impl Fn([f64; 3]) -> bool) -> Vec<u32> {
    (0..mesh.nodes().len() as u32)
        .filter(|n| keep(mesh.nodes()[*n as usize]))
        .collect()
}

/// Boundary faces whose three corners all match `keep`.
fn faces_where(mesh: &QuadraticTetMesh, keep: impl Fn([f64; 3]) -> bool) -> Vec<BoundaryFace> {
    mesh.boundary_faces()
        .into_iter()
        .filter(|face| {
            face.nodes[..3]
                .iter()
                .all(|n| keep(mesh.nodes()[*n as usize]))
        })
        .collect()
}

mod bracket;
mod column;
mod gear;
mod hollow_box;
mod hook;
mod nafems;
mod plate;
mod ring;
mod snapfit;
mod strip;

pub use gear::{fillet_concentration, lewis_stress};
pub use hook::winkler_inner_stress;
pub use plate::ktn_finite_width;

/// The real-part cases, by id, in part order.
pub fn all_cases() -> Vec<Case> {
    let mut cases = strip::cases();
    cases.extend(plate::cases());
    cases.extend(ring::cases());
    cases.extend(bracket::cases());
    cases.extend(nafems::cases());
    cases.extend(hook::cases());
    cases.extend(snapfit::cases());
    cases.extend(hollow_box::cases());
    cases.extend(column::cases());
    cases.extend(gear::cases());
    cases
}

/// The worst element of a mesh by minimum dihedral angle and where it is, shown and never
/// judged: its angle, its centroid, how many of its four corners lie on the surface, how many
/// of its faces are boundary faces, and its longest over shortest edge.
fn worst_element(mesh: &QuadraticTetMesh) -> Vec<(String, f64)> {
    let mut on_surface = vec![false; mesh.nodes().len()];
    let mut boundary_faces_of = vec![0u32; mesh.tets().len()];
    for face in mesh.boundary_faces() {
        boundary_faces_of[face.tet as usize] += 1;
        for node in &face.nodes[..3] {
            on_surface[*node as usize] = true;
        }
    }
    let mut worst = (180.0_f64, 0usize);
    for (index, tet) in mesh.tets().iter().enumerate() {
        let p = [tet[0], tet[1], tet[2], tet[3]].map(|n| mesh.nodes()[n as usize]);
        let angle = min_dihedral_degrees(p);
        if angle < worst.0 {
            worst = (angle, index);
        }
    }
    let tet = mesh.tets()[worst.1];
    let corners = [tet[0], tet[1], tet[2], tet[3]];
    let points = corners.map(|n| mesh.nodes()[n as usize]);
    let centroid = [
        (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0,
        (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0,
        (points[0][2] + points[1][2] + points[2][2] + points[3][2]) / 4.0,
    ];
    let (mut shortest, mut longest) = (f64::MAX, 0.0_f64);
    for a in 0..4 {
        for b in a + 1..4 {
            let d = ((points[a][0] - points[b][0]).powi(2)
                + (points[a][1] - points[b][1]).powi(2)
                + (points[a][2] - points[b][2]).powi(2))
            .sqrt();
            shortest = shortest.min(d);
            longest = longest.max(d);
        }
    }
    let surface_corners = corners.iter().filter(|n| on_surface[**n as usize]).count();
    vec![
        ("worst_dihedral_deg".to_string(), worst.0),
        ("worst_x".to_string(), centroid[0]),
        ("worst_y".to_string(), centroid[1]),
        ("worst_z".to_string(), centroid[2]),
        ("worst_surface_corners".to_string(), surface_corners as f64),
        (
            "worst_boundary_faces".to_string(),
            f64::from(boundary_faces_of[worst.1]),
        ),
        ("worst_edge_ratio".to_string(), longest / shortest),
    ]
}
