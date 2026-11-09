//! Part 3: a thick ring under internal pressure, against the Lame solution.

use super::*;

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
    Ok(Supports {
        fixed,
        tractions,
        prescribed: Vec::new(),
    })
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
    let mut values = vec![
        ("bore_ur".to_string(), bore),
        ("hoop_err".to_string(), hoop_error / peak_hoop),
        ("radial_err".to_string(), radial_error / RING_P),
    ];
    values.extend(worst_element(mesh));
    Ok(values)
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

pub(super) fn cases() -> Vec<Case> {
    vec![Case {
        id: "ring",
        title: "Part 3: thick ring 5/10 x 8 under internal pressure, nu = 0.3",
        material: steel(RING_NU),
        densities: vec![
            density("h=1.4", "ring_h1.4", Sizing::Uniform(1.4), 1.4),
            density("h=1.0", "ring_h1.0", Sizing::Uniform(1.0), 1.0),
            density("h=0.75", "ring_h0.75", Sizing::Uniform(0.75), 0.75),
        ],
        layers: 2,
        apply: ring_apply,
        measure: ring_measure,
        iteration_budget: None,
        judge: ring_judge,
    }]
}
