//! The real-part benchmark cases: what each part is, which surface files it uses and at what mesh
//! sizes, where it is held and loaded, what is measured, the reference it is compared with and its
//! limits. Each limit is derived in the comment on its judge, written before the case first ran.
//!
//! The surface files come from a separate generator built on solidmesh's CSG, extrude, revolve and
//! `remesh_for_analysis`, kept out of this repository because that remesher is not a dependency
//! this workspace takes. Each case's `Sizing` repeats the size function the generator was given,
//! so the interior of the volume mesh grades the same way as its surface. Selection regions are
//! geometric, with a tolerance above the rounding of the remeshed coordinates; a chord across a
//! curved surface is allowed for where the region is curved.

use solidfem::elastic::{Material, Solution};
use solidfem::mesh::{BoundaryFace, QuadraticTetMesh};

use super::pipeline::{face_area, face_normal, DensityResult, Supports};
use super::verdict::{classify_peak, last_change, within, PeakBehaviour, Verdict};

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
    if error > limit {
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
    let change = last_change(&values).ok_or_else(|| format!("{what}: no last change"))?;
    if change > limit {
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
    if (area - expected).abs() > 1e-6 * expected {
        return Err(format!("{what}: selected area {area}, expected {expected}"));
    }
    Ok(())
}

// --- Part 1: a thin cantilever strip with a tip load ---------------------------------------------

const STRIP_L: f64 = 100.0;
const STRIP_W: f64 = 10.0;
const STRIP_T: f64 = 1.0;
const STRIP_F: f64 = 1.0;

fn strip_apply(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
    let (faces, area) = plane_faces(mesh, 0, STRIP_L);
    check_area(area, STRIP_W * STRIP_T, "tip face")?;
    Ok(Supports {
        fixed: plane_nodes(mesh, 0, 0.0),
        tractions: faces
            .into_iter()
            .map(|f| (f, [0.0, 0.0, -STRIP_F / area]))
            .collect(),
    })
}

fn strip_measure(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
) -> Result<Vec<(String, f64)>, String> {
    let tip = mean_deflection(mesh, solution, 0, STRIP_L, 2)?;
    Ok(vec![("tip".to_string(), tip)])
}

/// Bending and shear parts of the Timoshenko tip deflection for the strip.
fn strip_reference(nu: f64) -> (f64, f64) {
    let inertia = STRIP_W * STRIP_T * STRIP_T * STRIP_T / 12.0;
    let area = STRIP_W * STRIP_T;
    let g = E / (2.0 * (1.0 + nu));
    (
        STRIP_F * STRIP_L * STRIP_L * STRIP_L / (3.0 * E * inertia),
        STRIP_F * STRIP_L / ((5.0 / 6.0) * g * area),
    )
}

/// **Limit, nu = 0.** Tip deflection against `F L^3/(3 E I) + F L/(k G A)` with `k = 5/6`, within
/// 1% at the finest mesh, and the last refinement moving it by at most 1%. With nu = 0 there is no
/// anticlastic restraint at the clamp, so beam theory is exact to the shear term, which is itself
/// about 0.6 (t/L)^2 of the bending term (6e-5 here): the reference's own uncertainty is under
/// 0.01%, and the whole 1% is what a two-layer 10-node strip is allowed for discretization.
fn strip_nu0_judge(rows: &[DensityResult]) -> Verdict {
    verdict_of((|| {
        let last = rows.last().ok_or("no rows")?;
        let (bending, shear) = strip_reference(0.0);
        check_error(last.value("tip"), bending + shear, 0.01, "tip deflection")?;
        stable(rows, "tip", 0.01, "tip deflection")
    })())
}

/// **Limit, nu = 0.3.** For a strip of width b = 10 t the anticlastic curvature is partly
/// restrained and the root clamp restrains the transverse strain, so the true stiffness lies
/// between the narrow beam (no restraint, `E I`) and the plate strip restrained everywhere
/// (`E I / (1 - nu^2)`): the bending part of the deflection is in
/// `[(1 - nu^2) d_b, d_b]`, with the Timoshenko shear term added to each bound. This is a
/// bracket from the two limiting beam theories, not a single closed form, so the only claim the
/// reference makes is that interval. The measurement is allowed 1% of `d_b` outside each bound
/// for discretization, as the nu = 0 case is, and the last refinement moving it by at most 1%.
fn strip_nu03_judge(rows: &[DensityResult]) -> Verdict {
    verdict_of((|| {
        let last = rows.last().ok_or("no rows")?;
        let nu = 0.3;
        let (bending, shear) = strip_reference(nu);
        let tol = 0.01 * bending;
        let (low, high) = (
            (1.0 - nu * nu) * bending + shear - tol,
            bending + shear + tol,
        );
        let tip = last.value("tip");
        if !within(tip, low, high) {
            return Err(format!(
                "tip deflection {tip:.6e} outside the bracket [{low:.6e}, {high:.6e}]"
            ));
        }
        stable(rows, "tip", 0.01, "tip deflection")
    })())
}

fn strip_densities() -> Vec<Density> {
    vec![
        density("h=2.0", "strip_h2.0", Sizing::Uniform(2.0), 2.0),
        density("h=1.4", "strip_h1.4", Sizing::Uniform(1.4), 1.4),
        density("h=1.0", "strip_h1.0", Sizing::Uniform(1.0), 1.0),
    ]
}

// --- Part 2: a plate with a central hole in tension ----------------------------------------------

const PLATE_L: f64 = 300.0;
const PLATE_W: f64 = 100.0;
const PLATE_T: f64 = 2.0;
const PLATE_D: f64 = 10.0;
const PLATE_SIGMA: f64 = 100.0;

fn plate_apply(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
    let (faces, area) = plane_faces(mesh, 0, PLATE_L);
    check_area(area, PLATE_W * PLATE_T, "loaded end")?;
    Ok(Supports {
        fixed: plane_nodes(mesh, 0, 0.0),
        tractions: faces
            .into_iter()
            .map(|f| (f, [PLATE_SIGMA, 0.0, 0.0]))
            .collect(),
    })
}

fn plate_measure(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
) -> Result<Vec<(String, f64)>, String> {
    let peak = peak_in_box(mesh, solution, [120.0, 20.0], [180.0, 80.0])?;
    // At the hole edge the only stress is the hoop stress (a free surface, and nu = 0), so von
    // Mises is that stress; sigma_net = F / ((W - d) t) with F = sigma W t.
    let sigma_net = PLATE_SIGMA * PLATE_W / (PLATE_W - PLATE_D);
    let (band, theta) = hole_band(mesh, solution, peak);
    Ok(vec![
        ("ktn".to_string(), peak / sigma_net),
        // Shown beside the peak, never judged: the mean over the hole-edge nodes within five
        // degrees of the top and bottom of the hole, and where the peak node is. A peak well above
        // that band's mean is a local spike at one node, not the hole's stress state; the band
        // mean itself reads about 0.3% under the true peak, because the Kirsch hoop stress falls
        // that far on average over the band.
        ("ktn_hole_band".to_string(), band / sigma_net),
        ("peak_theta_deg".to_string(), theta),
    ])
}

/// The mean nodal von Mises over the nodes on the hole edge within five degrees of its top and
/// bottom (where the hoop stress peaks), and the angle round the hole of the node holding `peak`.
fn hole_band(mesh: &QuadraticTetMesh, solution: &Solution, peak: f64) -> (f64, f64) {
    let (mut sum, mut count, mut theta) = (0.0, 0usize, f64::NAN);
    for (node, p) in mesh.nodes().iter().enumerate() {
        let (dx, dy) = (p[0] - 150.0, p[1] - 50.0);
        let radius = (dx * dx + dy * dy).sqrt();
        if radius > 0.5 * PLATE_D + 0.05 {
            continue;
        }
        let vm = solution.nodal_von_mises[node];
        let angle = dy.atan2(dx).to_degrees();
        if vm >= peak {
            theta = angle;
        }
        if (angle.abs() - 90.0).abs() <= 5.0 {
            sum += vm;
            count += 1;
        }
    }
    (sum / count.max(1) as f64, theta)
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
            peak = peak.max(solution.nodal_von_mises[node]);
        }
    }
    if peak == f64::MIN {
        return Err(format!("no node in the box {low:?}..{high:?}"));
    }
    Ok(peak)
}

/// The published finite-width net-section stress concentration for a central circular hole in
/// tension. `delta` is d/W.
///
/// Peterson's Stress Concentration Factors (Pilkey and Pilkey, 3rd edition, chart 4.1), the curve
/// fit `Ktn = 2 + 0.284 (1 - delta) - 0.600 (1 - delta)^2 + 1.32 (1 - delta)^3`, over Howland's
/// 1930 series solution. Cited from memory of that text, not checked against the page. At delta =
/// 0.1 it gives 2.732; Heywood's formula `2 + (1 - delta)^3` gives 2.729, 0.1% away, and at
/// delta -> 0 it gives 3.004, 0.13% above the exact 3.
pub fn ktn_finite_width(delta: f64) -> f64 {
    let one_minus = 1.0 - delta;
    2.0 + 0.284 * one_minus - 0.600 * one_minus * one_minus
        + 1.32 * one_minus * one_minus * one_minus
}

/// **Limit.** Net-section Ktn within 3% of the published curve at the finest mesh, and the peak
/// changing by at most 3% in the last refinement, not a singularity. The 3% is three parts. The
/// reference: the curve fit and Heywood differ by 0.1% here and the fit is 0.13% from the exact
/// infinite-plate value, so 0.5% is allowed for a second-hand formula cited from memory. Plate
/// length: the hole disturbs the far field like `(a/x)^2`, 0.1% at the 150 mm to the loaded end.
/// Discretization: the rest, 2.4%, is what 10-node nodal recovery may still under-read at the
/// finest mesh of about 57 chords round the hole. nu = 0 makes the 3D solution exactly the plane
/// stress one (no Poisson coupling across the thickness), so the plane-stress chart carries no
/// through-thickness uncertainty, and the clamped end imposes no transverse restraint.
fn plate_judge(rows: &[DensityResult]) -> Verdict {
    verdict_of((|| {
        let last = rows.last().ok_or("no rows")?;
        let reference = ktn_finite_width(PLATE_D / PLATE_W);
        check_error(last.value("ktn"), reference, 0.03, "net-section Kt")?;
        match classify_peak(&sizes(rows), &series(rows, "ktn"), 0.03) {
            Some(PeakBehaviour::Converged { .. }) => Ok(()),
            other => Err(format!(
                "the peak at the hole did not converge: {other:?}, series {:?}",
                series(rows, "ktn")
            )),
        }
    })())
}

// --- Part 3: a thick ring under internal pressure --------------------------------------------------

const RING_RI: f64 = 5.0;
const RING_RO: f64 = 10.0;
const RING_H: f64 = 8.0;
const RING_P: f64 = 10.0;
const RING_NU: f64 = 0.3;
/// A remeshed cylinder's vertices are within the generator's 0.03 mm deviation of its radius.
const RING_RADIAL_TOL: f64 = 0.05;
/// The integration points used for the stress comparison are at least this far round from the
/// held cluster at angle zero, far beyond the reach of its self-equilibrated disturbance.
const RING_MIN_ANGLE: f64 = std::f64::consts::FRAC_PI_3;

/// The Lamé plane-stress coefficients `(A, B)` with `sigma_r = A - B/r^2`, `sigma_t = A + B/r^2`.
fn lame() -> (f64, f64) {
    let (a2, b2) = (RING_RI * RING_RI, RING_RO * RING_RO);
    let k = RING_P * a2 / (b2 - a2);
    (k, k * b2)
}

fn ring_apply(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
    // Three corner nodes clustered on the outer surface at angle zero, mid-height, hold the rigid
    // motion. Fixing all three components of each over-constrains the part by three, but only
    // locally, across an edge length or two; the ring's expansion is otherwise free.
    let target = [RING_RO, 0.0, 0.5 * RING_H];
    let distance = |p: [f64; 3]| {
        (p[0] - target[0]) * (p[0] - target[0])
            + (p[1] - target[1]) * (p[1] - target[1])
            + (p[2] - target[2]) * (p[2] - target[2])
    };
    let mut near: Vec<u32> = (0..mesh.vertex_count() as u32).collect();
    near.sort_by(|a, b| {
        distance(mesh.nodes()[*a as usize])
            .total_cmp(&distance(mesh.nodes()[*b as usize]))
            .then(a.cmp(b))
    });
    let mut fixed: Vec<u32> = near.iter().take(2).copied().collect();
    let (a, b) = (
        mesh.nodes()[fixed[0] as usize],
        mesh.nodes()[fixed[1] as usize],
    );
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let third = near.iter().copied().find(|node| {
        let p = mesh.nodes()[*node as usize];
        let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
        let cross = [
            ab[1] * ap[2] - ab[2] * ap[1],
            ab[2] * ap[0] - ab[0] * ap[2],
            ab[0] * ap[1] - ab[1] * ap[0],
        ];
        let span = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
        let scale = (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2])
            * (ap[0] * ap[0] + ap[1] * ap[1] + ap[2] * ap[2]);
        span > 1e-3 * scale
    });
    fixed.push(third.ok_or("no third held node off the line of the first two")?);

    let mut tractions = Vec::new();
    for face in bore_faces(mesh)? {
        let n = face_normal(mesh, &face);
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        // The outward normal of the part points towards the axis here; a pressure pushes opposite
        // to it, into the material.
        tractions.push((
            face,
            [
                -RING_P * n[0] / length,
                -RING_P * n[1] / length,
                -RING_P * n[2] / length,
            ],
        ));
    }
    Ok(Supports { fixed, tractions })
}

/// The boundary faces of the bore: every corner on its radius, and a normal mostly radial so an
/// end-face sliver between two bore-rim vertices is not taken. Refuses a bore whose outward normal
/// points away from the axis, which would be a wound-inside-out part.
fn bore_faces(mesh: &QuadraticTetMesh) -> Result<Vec<BoundaryFace>, String> {
    let mut faces = Vec::new();
    for face in mesh.boundary_faces() {
        let on_bore = face.nodes[..3].iter().all(|n| {
            let p = mesh.nodes()[*n as usize];
            (p[0] * p[0] + p[1] * p[1]).sqrt() <= RING_RI + RING_RADIAL_TOL
        });
        if !on_bore {
            continue;
        }
        let n = face_normal(mesh, &face);
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if n[2].abs() > 0.2 * length {
            continue;
        }
        let c = centroid(mesh, &face);
        if n[0] * c[0] + n[1] * c[1] >= 0.0 {
            return Err("a bore face's outward normal points away from the axis".to_string());
        }
        faces.push(face);
    }
    if faces.is_empty() {
        return Err("no bore faces found".to_string());
    }
    Ok(faces)
}

fn centroid(mesh: &QuadraticTetMesh, face: &BoundaryFace) -> [f64; 3] {
    let [a, b, c] = [face.nodes[0], face.nodes[1], face.nodes[2]].map(|n| mesh.nodes()[n as usize]);
    [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ]
}

/// Bore radial displacement, area-weighted over the bore so rigid translation of the part (it is
/// held at one spot on the outside) cancels; measured values are the bore mean, the worst hoop and
/// radial stress errors at integration points away from the held cluster, and how many points.
fn ring_measure(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
) -> Result<Vec<(String, f64)>, String> {
    let (mut sum, mut area) = (0.0, 0.0);
    for face in bore_faces(mesh)? {
        let measure = face_area(mesh, &face);
        // A quadratic triangle's integral weights are zero at the corners and 1/3 at each midside.
        for node in &face.nodes[3..] {
            let p = mesh.nodes()[*node as usize];
            let radius = (p[0] * p[0] + p[1] * p[1]).sqrt();
            let u = solution.displacements[*node as usize];
            sum += measure * (u[0] * p[0] + u[1] * p[1]) / radius / 3.0;
        }
        area += measure;
    }
    let bore = sum / area;

    let (coefficient_a, coefficient_b) = lame();
    let peak_hoop = coefficient_a + coefficient_b / (RING_RI * RING_RI);
    let (mut hoop_error, mut radial_error, mut points) = (0.0_f64, 0.0_f64, 0usize);
    for point in &solution.point_stresses {
        let [x, y, _] = point.position;
        let angle = y.atan2(x).abs();
        if angle < RING_MIN_ANGLE {
            continue;
        }
        let r2 = x * x + y * y;
        let radius = r2.sqrt();
        let (nx, ny) = (x / radius, y / radius);
        let s = point.stress;
        let radial = nx * nx * s[0] + ny * ny * s[1] + 2.0 * nx * ny * s[3];
        let hoop = ny * ny * s[0] + nx * nx * s[1] - 2.0 * nx * ny * s[3];
        hoop_error = hoop_error.max((hoop - (coefficient_a + coefficient_b / r2)).abs());
        radial_error = radial_error.max((radial - (coefficient_a - coefficient_b / r2)).abs());
        points += 1;
    }
    if points == 0 {
        return Err("no integration point away from the held cluster".to_string());
    }
    Ok(vec![
        ("bore_ur".to_string(), bore),
        ("hoop_err".to_string(), hoop_error / peak_hoop),
        ("radial_err".to_string(), radial_error / RING_P),
    ])
}

/// `u_r(r)` of the Lamé solution with free ends, `sigma_z = 0`: exact in 3D for any height.
fn lame_radial_displacement(r: f64) -> f64 {
    let (a, b) = lame();
    ((1.0 - RING_NU) * a * r + (1.0 + RING_NU) * b / r) / E
}

/// **Limits.** The free-ended tube's stresses and displacement are the exact Lamé solution with
/// `sigma_z = 0`: the field `u_r(r)`, `u_z = -2 nu A z / E` satisfies 3D equilibrium with no
/// traction on the ends, so no plane-stress or plane-strain caveat applies at any height. The mean
/// bore radial displacement must be within 1% of the reference at the finest mesh and moved by at
/// most 1% by the last refinement: the chord of a remeshed bore of radius 5 at edge 0.75 sags by
/// `h^2 / (8 r)` = 0.014 mm, 0.28% of radius, and the three held nodes' local disturbance is
/// small, so most of the 1% is for discretization. The worst pointwise hoop stress error, at
/// integration points 60 degrees or more from the held cluster, must be at most 3% of the peak
/// hoop stress, and the worst radial stress error 3% of the pressure: 10-node elements across a
/// 5 mm wall resolve the `1/r^2` field with a few percent of pointwise error at the integration
/// points of the coarsest element next to the bore, and this limit is on the worst single point
/// rather than a mean. The stresses are at points, not extrapolated, so no recovery is involved.
fn ring_judge(rows: &[DensityResult]) -> Verdict {
    verdict_of((|| {
        let last = rows.last().ok_or("no rows")?;
        check_error(
            last.value("bore_ur"),
            lame_radial_displacement(RING_RI),
            0.01,
            "bore radial displacement",
        )?;
        stable(rows, "bore_ur", 0.01, "bore radial displacement")?;
        let (hoop, radial) = (last.value("hoop_err"), last.value("radial_err"));
        if hoop > 0.03 {
            return Err(format!(
                "worst hoop stress error {:.3}% of the peak hoop, over 3%",
                100.0 * hoop
            ));
        }
        if radial > 0.03 {
            return Err(format!(
                "worst radial stress error {:.3}% of the pressure, over 3%",
                100.0 * radial
            ));
        }
        Ok(())
    })())
}

// --- Part 4: an L-bracket with a sharp re-entrant corner, and with a fillet ------------------------

const L_LEG: f64 = 100.0;
const L_DEPTH: f64 = 10.0;
const L_THICK: f64 = 5.0;
const L_F: f64 = 100.0;

fn bracket_apply(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
    let (faces, area) = plane_faces(mesh, 0, L_LEG);
    check_area(area, L_DEPTH * L_THICK, "bracket tip")?;
    Ok(Supports {
        fixed: plane_nodes(mesh, 1, L_LEG),
        tractions: faces
            .into_iter()
            .map(|f| (f, [0.0, -L_F / area, 0.0]))
            .collect(),
    })
}

/// A rough frame estimate of the tip deflection, shown beside the measurement and never judged:
/// two straight beams on centre lines through a rigid joint,
/// `F L^3 / (3 E I) + F L^2 H / (E I) + F H / (E A)`, L = H = 95. A real corner is more flexible
/// than a rigid joint, so a few percent above this is expected.
fn frame_estimate() -> f64 {
    let inertia = L_THICK * L_DEPTH * L_DEPTH * L_DEPTH / 12.0;
    let area = L_THICK * L_DEPTH;
    let arm = L_LEG - 0.5 * L_DEPTH;
    L_F * arm * arm * arm / (3.0 * E * inertia)
        + L_F * arm * arm * arm / (E * inertia)
        + L_F * arm / (E * area)
}

fn bracket_measure(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
) -> Result<Vec<(String, f64)>, String> {
    Ok(vec![
        (
            "tip".to_string(),
            mean_deflection(mesh, solution, 0, L_LEG, 1)?,
        ),
        // The window excludes the clamp and the loaded end, which have stress raisers of their own.
        (
            "peak".to_string(),
            peak_in_box(mesh, solution, [0.0, 0.0], [30.0, 30.0])?,
        ),
        ("frame_estimate".to_string(), frame_estimate()),
    ])
}

/// **Limit, sharp corner.** The tip deflection is an energy quantity, and the corner singularity
/// costs strain energy only like `h^(2 lambda)` with `lambda` = 0.5445, so the last refinement must
/// move it by at most 1%. The corner's peak stress has no value to compare with anything: the
/// right outcome is that it keeps growing and is reported as a detected singularity, a number the
/// reader must not use. Failing to detect it, here, is a failure of this benchmark.
fn l_sharp_judge(rows: &[DensityResult]) -> Verdict {
    verdict_of((|| {
        stable(rows, "tip", 0.01, "tip deflection")?;
        let peaks = series(rows, "peak");
        match classify_peak(&sizes(rows), &peaks, 0.03) {
            Some(PeakBehaviour::Singular { .. }) => Ok(()),
            other => Err(format!(
                "the sharp corner's peak did not grow like a singularity: {other:?}, series {peaks:?}"
            )),
        }
    })())
}

/// **Limit, fillet.** Tip deflection moved by at most 1% in the last refinement, as for the sharp
/// corner. The fillet's peak is a real number and must be converged, not singular: the last
/// refinement moves it by at most 3%. With error shrinking about as `h^2` and a refinement ratio
/// near 1.6, a 3% last change means the coarser mesh is still 5% off and the finest about 2%, the
/// usual engineering target for a fillet peak; there is no closed-form value to hold it to here.
fn l_fillet_judge(rows: &[DensityResult]) -> Verdict {
    verdict_of((|| {
        stable(rows, "tip", 0.01, "tip deflection")?;
        let peaks = series(rows, "peak");
        match classify_peak(&sizes(rows), &peaks, 0.03) {
            Some(PeakBehaviour::Converged { .. }) => Ok(()),
            other => Err(format!(
                "the fillet's peak did not converge: {other:?}, series {peaks:?}"
            )),
        }
    })())
}

fn steel(nu: f64) -> Material {
    Material {
        youngs_modulus: E,
        poisson_ratio: nu,
    }
}

/// The real-part cases, by id.
pub fn all_cases() -> Vec<Case> {
    let plate_size = |near: f64| Sizing::AroundPoint {
        centre: [150.0, 50.0],
        r0: 5.0,
        near,
        grade: 0.3,
        far: 6.0,
    };
    let sharp_size = |near: f64| Sizing::AroundPoint {
        centre: [10.0, 10.0],
        r0: 0.0,
        near,
        grade: 0.25,
        far: 4.0,
    };
    let fillet_size = |near: f64| Sizing::AroundPoint {
        centre: [15.0, 15.0],
        r0: 6.0,
        near,
        grade: 0.25,
        far: 4.0,
    };
    vec![
        Case {
            id: "strip-nu0",
            title: "Thin cantilever strip 100 x 10 x 1, tip load, nu = 0",
            material: steel(0.0),
            densities: strip_densities(),
            layers: 2,
            apply: strip_apply,
            measure: strip_measure,
            judge: strip_nu0_judge,
        },
        Case {
            id: "strip-nu03",
            title: "Thin cantilever strip 100 x 10 x 1, tip load, nu = 0.3",
            material: steel(0.3),
            densities: strip_densities(),
            layers: 2,
            apply: strip_apply,
            measure: strip_measure,
            judge: strip_nu03_judge,
        },
        Case {
            id: "plate-hole",
            title: "Plate 300 x 100 x 2 with d = 10 hole in tension, nu = 0",
            material: steel(0.0),
            densities: vec![
                density("near=1.2", "plate_n1.2", plate_size(1.2), 1.2),
                density("near=0.8", "plate_n0.8", plate_size(0.8), 0.8),
                density("near=0.55", "plate_n0.55", plate_size(0.55), 0.55),
            ],
            layers: 2,
            apply: plate_apply,
            measure: plate_measure,
            judge: plate_judge,
        },
        Case {
            id: "ring",
            title: "Thick ring 5/10 x 8 under internal pressure, nu = 0.3",
            material: steel(RING_NU),
            densities: vec![
                density("h=1.4", "ring_h1.4", Sizing::Uniform(1.4), 1.4),
                density("h=1.0", "ring_h1.0", Sizing::Uniform(1.0), 1.0),
                density("h=0.75", "ring_h0.75", Sizing::Uniform(0.75), 0.75),
            ],
            layers: 2,
            apply: ring_apply,
            measure: ring_measure,
            judge: ring_judge,
        },
        Case {
            id: "l-sharp",
            title: "L-bracket, sharp re-entrant corner, tip load, nu = 0.3",
            material: steel(0.3),
            densities: vec![
                density("c=1.0", "lsharp_c1.0", sharp_size(1.0), 1.0),
                density("c=0.5", "lsharp_c0.5", sharp_size(0.5), 0.5),
                density("c=0.25", "lsharp_c0.25", sharp_size(0.25), 0.25),
            ],
            layers: 2,
            apply: bracket_apply,
            measure: bracket_measure,
            judge: l_sharp_judge,
        },
        Case {
            id: "l-fillet",
            title: "L-bracket, 5 mm fillet at the corner, tip load, nu = 0.3",
            material: steel(0.3),
            densities: vec![
                density("f=1.2", "lfillet_f1.2", fillet_size(1.2), 1.2),
                density("f=0.8", "lfillet_f0.8", fillet_size(0.8), 0.8),
                density("f=0.5", "lfillet_f0.5", fillet_size(0.5), 0.5),
            ],
            layers: 2,
            apply: bracket_apply,
            measure: bracket_measure,
            judge: l_fillet_judge,
        },
    ]
}
