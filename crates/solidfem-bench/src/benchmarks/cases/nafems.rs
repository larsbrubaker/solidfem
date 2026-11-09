//! Part 5: NAFEMS LE1 (elliptic membrane) and LE10 (thick plate under pressure).
//!
//! **Source and its limits.** The problem statements come from the openly available FeenoX
//! documentation (`seamplex.com/feenox/examples/mechanical.html`, which gives both problems'
//! loads, supports and the LE1 reference 92.7 MPa at D), not from the paywalled NAFEMS text. The
//! LE10 target -5.38 MPa was supplied second hand and that page's text states only the point D
//! (2000, 0, 300) and the quantity, not this number. Both targets are therefore second hand, so
//! each limit carries an allowance for that. The geometry is the elliptical annulus quarter both problems use: inner ellipse
//! semi-axes 2000 and 1000 mm, outer 3250 and 2750 mm, E = 210 GPa, nu = 0.3, LE1 thickness 100 mm
//! and an outward 10 MPa on the outer edge, LE10 thickness 600 mm with a 1 MPa pressure on the
//! upper face and the outer face held in x and y. The surfaces are extruded and remeshed through
//! the real pipeline, so the quarter is faceted and the mid-plane support line of LE10, which is
//! not a mesh feature of an extruded surface, is realised as the nodes of the outer face within a
//! band about the mid-plane, stated in the judge below.

use super::*;

const LE_E: f64 = 210_000.0;
const LE_NU: f64 = 0.3;
const OUTER: [f64; 2] = [3250.0, 2750.0];
const INNER: [f64; 2] = [2000.0, 1000.0];
/// How far a point may be from the ellipse in its implicit value, generous enough to hold the
/// sagitta of the 200-segment polygon and the remesher deviation at these sizes (a few tenths of
/// a millimetre on a radius of thousands).
const ELLIPSE_TOL: f64 = 2e-3;

fn le_material() -> Material {
    Material {
        youngs_modulus: LE_E,
        poisson_ratio: LE_NU,
    }
}

fn ellipse_value(p: [f64; 3], axes: [f64; 2]) -> f64 {
    (p[0] / axes[0]).powi(2) + (p[1] / axes[1]).powi(2)
}

fn on_outer(p: [f64; 3]) -> bool {
    (ellipse_value(p, OUTER) - 1.0).abs() <= ELLIPSE_TOL
}

/// Faces of the outer curved surface: every corner on the ellipse, normal mostly in the plane.
fn outer_faces(mesh: &QuadraticTetMesh) -> Vec<BoundaryFace> {
    faces_where(mesh, on_outer)
        .into_iter()
        .filter(|face| {
            let n = face_normal(mesh, face);
            n[2].abs() <= 0.2 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
        })
        .collect()
}

/// The node nearest `target`, ties to the lower index.
fn nearest_node(mesh: &QuadraticTetMesh, target: [f64; 3]) -> u32 {
    let mut best = (f64::MAX, 0u32);
    for (index, p) in mesh.nodes().iter().enumerate() {
        let d =
            (p[0] - target[0]).powi(2) + (p[1] - target[1]).powi(2) + (p[2] - target[2]).powi(2);
        if d < best.0 {
            best = (d, index as u32);
        }
    }
    best.1
}

fn le1_apply(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
    let faces = outer_faces(mesh);
    if faces.is_empty() {
        return Err("no outer faces found".to_string());
    }
    let mut supports = Supports::default();
    for face in faces {
        let n = face_normal(mesh, &face);
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        // An outward pressure of 10 MPa pulls along the outward normal.
        supports.tractions.push((
            face,
            [
                10.0 * n[0] / length,
                10.0 * n[1] / length,
                10.0 * n[2] / length,
            ],
        ));
    }
    for node in plane_nodes(mesh, 0, 0.0) {
        supports.prescribed.push((node, 0, 0.0));
    }
    for node in plane_nodes(mesh, 1, 0.0) {
        supports.prescribed.push((node, 1, 0.0));
    }
    // One node held in z fixes the rigid z translation; the rollers fix every other rigid motion.
    supports
        .prescribed
        .push((nearest_node(mesh, [2625.0, 0.0, 50.0]), 2, 0.0));
    Ok(supports)
}

fn le1_measure(mesh: &QuadraticTetMesh, solution: &Solution) -> Result<Vec<(String, f64)>, String> {
    // The stress at point D (2000, 0) is taken as the recovered nodal stress averaged over the
    // line of nodes through the thickness at that point, the 2D benchmark being uniform in z.
    let targets = nodes_where(mesh, |p| {
        (p[0] - INNER[0]).abs() <= 1.0 && p[1].abs() <= PLANE_TOL
    });
    if targets.is_empty() {
        return Err("no nodes on the line through point D".to_string());
    }
    let stresses = nodal_stress(solution, &targets);
    let mean = stresses.iter().map(|s| s[1]).sum::<f64>() / stresses.len() as f64;
    Ok(vec![
        ("sigma_y_D".to_string(), mean),
        ("line_nodes".to_string(), stresses.len() as f64),
    ])
}

const LE1_TARGET: f64 = 92.7;

/// **Limit, LE1.** `sigma_y` at D, within 3% of the second-hand 92.7 MPa at the finest mesh and
/// the last refinement moving it by at most 3%. The 3% is: 1% for a target read second hand from
/// open documentation of a paywalled publication, with no printed tolerance seen; 0.5% for a 3D
/// model of the plane-stress statement; 1.5% for discretization at a nodal average on the
/// quadratic mesh. The 3D allowance: the radius of curvature of the inner ellipse at D is
/// `b^2 / a = 1000^2 / 2000 = 500`, so the thickness is `t / rho = 100 / 500 = 0.2` of it. The
/// quantity is the through-thickness mean, for which the leading departure from plane stress cancels
/// and the next is of order `nu (t / rho)^2 / 12` = 0.1%; 0.5% stands as five times that order of
/// magnitude, an estimate rather than a published value. The limit is unchanged, and the measured
/// +0.95% passes it.
fn le1_judge(rows: &[DensityResult]) -> Verdict {
    verdict_of((|| {
        let last = rows.last().ok_or("no rows")?;
        check_error(last.value("sigma_y_D"), LE1_TARGET, 0.03, "sigma_y at D")?;
        stable(rows, "sigma_y_D", 0.03, "sigma_y at D")
    })())
}

fn le10_apply(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
    let mut supports = Supports::default();
    // The upper face (z = 300), pressure 1 MPa pushing down: traction opposite the outward normal.
    for face in faces_where(mesh, |p| (p[2] - 300.0).abs() <= PLANE_TOL) {
        let n = face_normal(mesh, &face);
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        supports
            .tractions
            .push((face, [-n[0] / length, -n[1] / length, -n[2] / length]));
    }
    if supports.tractions.is_empty() {
        return Err("no upper face found".to_string());
    }
    for node in plane_nodes(mesh, 0, 0.0) {
        supports.prescribed.push((node, 0, 0.0));
    }
    for node in plane_nodes(mesh, 1, 0.0) {
        supports.prescribed.push((node, 1, 0.0));
    }
    // The outer face is held in x and y, every node on the ellipse.
    for node in nodes_where(mesh, on_outer) {
        supports.prescribed.push((node, 0, 0.0));
        supports.prescribed.push((node, 1, 0.0));
    }
    // The mid-plane support line of the outer edge is not a mesh feature: the nodes of the outer
    // face within this band of z = 0 stand in for it. They all lie on one radial position, so a
    // band of them restrains the vertical shear but adds no rotation about the edge.
    let band = 0.5 * LE10_FAR;
    let held = nodes_where(mesh, |p| on_outer(p) && p[2].abs() <= band);
    if held.is_empty() {
        return Err("no outer-face node near the mid-plane".to_string());
    }
    for node in held {
        supports.prescribed.push((node, 2, 0.0));
    }
    Ok(supports)
}

fn le10_measure(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
) -> Result<Vec<(String, f64)>, String> {
    let node = nearest_node(mesh, [INNER[0], 0.0, 300.0]);
    let p = mesh.nodes()[node as usize];
    if (p[0] - INNER[0]).abs() > 1.0 || p[1].abs() > 1.0 || (p[2] - 300.0).abs() > 1.0 {
        return Err(format!(
            "no node at point D (2000, 0, 300); nearest is at {p:?}"
        ));
    }
    let stress = nodal_stress(solution, &[node]);
    Ok(vec![("sigma_y_D".to_string(), stress[0][1])])
}

const LE10_FAR: f64 = 150.0;
const LE10_TARGET: f64 = -5.38;

/// **Limit, LE10.** `sigma_y` at D (2000, 0, 300), within 5% of the second-hand -5.38 MPa at the
/// finest mesh, and the last refinement moving it by at most 5%. The 5% is: 1.5% for the second
/// hand target, as for LE1, with more spread between published solutions at this singular-prone
/// point; 1% for the mid-plane support line realised as a band of outer-face nodes, because the
/// surface pipeline gives no line of mesh edges at z = 0 on an extruded wall; 2.5% for the
/// discretization at a single node, the corner where the top face meets the elliptical hole.
fn le10_judge(rows: &[DensityResult]) -> Verdict {
    verdict_of((|| {
        let last = rows.last().ok_or("no rows")?;
        check_error(last.value("sigma_y_D"), LE10_TARGET, 0.05, "sigma_y at D")?;
        stable(rows, "sigma_y_D", 0.05, "sigma_y at D")
    })())
}

pub(super) fn cases() -> Vec<Case> {
    let size = |near: f64, grade: f64, far: f64| Sizing::AroundPoint {
        centre: [2000.0, 0.0],
        r0: 0.0,
        near,
        grade,
        far,
    };
    vec![
        Case {
            id: "le1",
            title: "Part 5: NAFEMS LE1 elliptic membrane (stress at D), nu = 0.3",
            material: le_material(),
            densities: vec![
                density("near=60", "le1_n60", size(60.0, 0.2, 120.0), 60.0),
                density("near=40", "le1_n40", size(40.0, 0.2, 120.0), 40.0),
                density("near=28", "le1_n28", size(28.0, 0.2, 120.0), 28.0),
            ],
            layers: 2,
            apply: le1_apply,
            measure: le1_measure,
            iteration_budget: None,
            judge: le1_judge,
        },
        Case {
            id: "le10",
            title: "Part 5: NAFEMS LE10 thick plate (stress at D), nu = 0.3",
            material: le_material(),
            densities: vec![
                density("near=120", "le10_n120", size(120.0, 0.25, LE10_FAR), 120.0),
                density("near=85", "le10_n85", size(85.0, 0.25, LE10_FAR), 85.0),
                density("near=60", "le10_n60", size(60.0, 0.25, LE10_FAR), 60.0),
            ],
            layers: 2,
            apply: le10_apply,
            measure: le10_measure,
            iteration_budget: None,
            judge: le10_judge,
        },
    ]
}
