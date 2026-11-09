//! Part 6: a hook under a hanging weight, against Winkler-Bach curved-beam stress at the throat
//! and the stability of its tip deflection.

use super::*;

const RI: f64 = 20.0;
const RO: f64 = 40.0;
const WIDTH: f64 = 10.0;
const F: f64 = 1000.0;

/// The hanging weight, as a downward traction over a 30 degree patch of the bore at its lowest
/// point, centred on the line x = 0 that passes through the shank and the centre of curvature.
fn apply(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
    let faces: Vec<BoundaryFace> = faces_where(mesh, |p| {
        let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
        let angle = p[1].atan2(p[0]).to_degrees();
        // The lowest part of the bore: angle about -90 degrees.
        (r - RI).abs() <= 0.05 && (angle + 90.0).abs() <= 15.0
    });
    let area: f64 = faces.iter().map(|f| face_area(mesh, f)).sum();
    if faces.is_empty() || area <= 0.0 {
        return Err("no bore faces found under the load patch".to_string());
    }
    Ok(Supports {
        fixed: plane_nodes(mesh, 1, 110.0),
        tractions: faces
            .into_iter()
            .map(|f| (f, [0.0, -F / area, 0.0]))
            .collect(),
        prescribed: Vec::new(),
    })
}

fn measure(mesh: &QuadraticTetMesh, solution: &Solution) -> Result<Vec<(String, f64)>, String> {
    // Inner fibre of the throat, on the left side at y = 0, within a 1.5 mm window; the hoop
    // stress at that section is sigma_y.
    let targets = nodes_where(mesh, |p| {
        let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
        p[0] < 0.0 && p[1].abs() <= 1.5 && (r - RI).abs() <= 0.05
    });
    if targets.is_empty() {
        return Err("no nodes at the inner fibre of the throat".to_string());
    }
    let stress = nodal_stress(solution, &targets);
    let inner = stress.iter().map(|s| s[1]).sum::<f64>() / stress.len() as f64;
    // Tip deflection: the hook tip at 300 degrees, along the load.
    let tip_nodes = nodes_where(mesh, |p| {
        let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
        let angle = p[1].atan2(p[0]).to_degrees();
        (angle + 60.0).abs() <= 1.0 && (RI - 0.5..=RO + 0.5).contains(&r)
    });
    if tip_nodes.is_empty() {
        return Err("no nodes at the hook tip".to_string());
    }
    let tip = tip_nodes
        .iter()
        .map(|n| -solution.displacements[*n as usize][1])
        .sum::<f64>()
        / tip_nodes.len() as f64;
    Ok(vec![
        ("inner_sigma".to_string(), inner),
        ("tip_uy".to_string(), tip),
    ])
}

/// The Winkler-Bach inner-fibre stress at the throat of a rectangular curved section: `F/A`
/// plus the moment `F R` over `A e r_i`, with the neutral radius `h / ln(r_o / r_i)`.
pub fn winkler_inner_stress(force: f64, ri: f64, ro: f64, width: f64) -> f64 {
    let h = ro - ri;
    let area = h * width;
    let centroid = 0.5 * (ri + ro);
    let neutral = h / (ro / ri).ln();
    let e = centroid - neutral;
    let moment = force * centroid;
    force / area + moment * (neutral - ri) / (area * e * ri)
}

/// **Limit.** The inner-fibre stress at the throat, averaged over a 1.5 mm window, within 2.4% of
/// Winkler-Bach at the finest mesh, and tip deflection moved by at most 1% in the last refinement.
/// The formula assumes plane sections, a constant section, no radial stress, an axis of constant
/// curvature, and this hook has `R / h = 1.5`, strongly curved, but the elasticity closed form for
/// pure bending of a curved bar gives `7.755 M / h^2` at the inner fibre of `r_o / r_i = 2`,
/// against Winkler-Bach's `7.725 M / h^2`: a 0.4% difference at this very shape, not the larger
/// figure usually quoted for curved beams in general. The throat also carries the direct stress
/// `F / A` (5 of 62.9 MPa), taken at its Winkler value; the bending elasticity comparison does not
/// cover that small share, which sits inside the next 1%. The throat is one quarter of a circle
/// from the load patch (47 mm of arc, 2.4 section depths), and the shank union is on the other
/// side of the ring, a real departure from a uniform arc, allowed 1%. The stress is a nodal
/// average in a window over which the moment and curvature change by under 0.5%; with
/// discretization that is 1%. 2.4% is 0.4% for the formula, 1% for the load and shank departures,
/// 1% for discretization. The finest mesh measures +2.1% against this reference.
fn judge(rows: &[DensityResult]) -> Verdict {
    verdict_of((|| {
        let last = rows.last().ok_or("no rows")?;
        let reference = winkler_inner_stress(F, RI, RO, WIDTH);
        check_error(
            last.value("inner_sigma"),
            reference,
            0.024,
            "inner fibre stress at the throat",
        )?;
        stable(rows, "tip_uy", 0.01, "tip deflection")
    })())
}

pub(super) fn cases() -> Vec<Case> {
    vec![Case {
        id: "hook",
        title: "Part 6: hook, hanging weight, Winkler-Bach at the throat, nu = 0.3",
        material: steel(0.3),
        densities: vec![
            density("h=3.0", "hook_h3.0", Sizing::Uniform(3.0), 3.0),
            density("h=2.2", "hook_h2.2", Sizing::Uniform(2.2), 2.2),
            density("h=1.6", "hook_h1.6", Sizing::Uniform(1.6), 1.6),
        ],
        layers: 2,
        apply,
        measure,
        iteration_budget: None,
        judge,
    }]
}
