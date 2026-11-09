//! Part 10: a single gear tooth on a sector of its rim, under a tangential load at the tip,
//! against the Lewis bending stress with the Dolan-Broghamer fillet concentration.
//!
//! The tooth: 20 teeth, module 4 mm, 20 degree pressure angle, 20 mm face width; involute flanks
//! from the base circle to the addendum circle, a radial flank below the base circle, a 1.2 mm
//! root fillet, a root circle at 35 mm, and a rim sector out to a 24 mm bore, clamped there. A
//! single tooth in a section rather than a full gear: Lewis is a single-tooth formula.
//!
//! **The formula's known limitations, in the limit.** Lewis: `sigma = F / (b m Y)` treats the
//! tooth as a cantilever of constant strength with the load at the tip and Y a shape factor, no
//! stress concentration, and no rim. Its factor `Y(N = 20) = 0.322` for full-depth 20 degree teeth
//! is read from a standard design-text table (Shigley, Lewis form factor table), second hand from
//! memory, unchecked against the page in this run. The concentration `Kf` is the Dolan and
//! Broghamer photoelastic fit, `Kf = H + (t / r)^L (t / l)^M`, with `H = 0.34 - 0.4583662 phi`,
//! `L = 0.316 - 0.4583662 phi`, `M = 0.290 + 0.4583662 phi`, phi in radians, also second hand and
//! stated with a scatter of the order of 10 to 15% against test data in design texts. It uses the
//! thickness `t` at the critical section, the fillet radius `r` and the load height `l`.

use super::*;

const MODULE: f64 = 4.0;
const TEETH: f64 = 20.0;
const FACE: f64 = 20.0;
const FORCE: f64 = 1000.0;
const LEWIS_Y: f64 = 0.322;
const PHI: f64 = 0.349_065_850_398_865_9;
const RP: f64 = 0.5 * TEETH * MODULE;
const RA: f64 = RP + MODULE;
const RR: f64 = RP - 1.25 * MODULE;
const RF: f64 = 0.3 * MODULE;
/// The radial limit of the clamped bore and the tolerance on its surface.
const RIN: f64 = 24.0;

fn radius(p: [f64; 3]) -> f64 {
    (p[0] * p[0] + p[1] * p[1]).sqrt()
}

fn apply(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
    let faces = faces_where(mesh, |p| (radius(p) - RA).abs() <= 0.05);
    let tip: Vec<BoundaryFace> = faces
        .into_iter()
        .filter(|face| {
            let n = face_normal(mesh, face);
            let c = face.nodes[..3].iter().fold([0.0; 2], |sum, node| {
                let p = mesh.nodes()[*node as usize];
                [sum[0] + p[0], sum[1] + p[1]]
            });
            n[0] * c[0] + n[1] * c[1] > 0.0
        })
        .collect();
    let area: f64 = tip.iter().map(|f| face_area(mesh, f)).sum();
    if tip.is_empty() || area <= 0.0 {
        return Err("no faces on the tooth tip".to_string());
    }
    Ok(Supports {
        fixed: nodes_where(mesh, |p| radius(p) <= RIN + 0.5),
        tractions: tip
            .into_iter()
            .map(|f| (f, [FORCE / area, 0.0, 0.0]))
            .collect(),
        prescribed: Vec::new(),
    })
}

fn measure(mesh: &QuadraticTetMesh, solution: &Solution) -> Result<Vec<(String, f64)>, String> {
    // The tension-side root fillet is on the side opposite the load; +x load, so x < 0, within a
    // window round the fillet radially.
    let mut peak = f64::MIN;
    for (node, p) in mesh.nodes().iter().enumerate() {
        let r = radius(*p);
        if p[0] < 0.0 && (RR - 0.5..=RR + 3.0 * RF).contains(&r) {
            peak = peak.max(solution.nodal_von_mises[node]);
        }
    }
    if peak == f64::MIN {
        return Err("no nodes in the root fillet window".to_string());
    }
    Ok(vec![("root_peak".to_string(), peak)])
}

/// The Lewis bending stress, MPa: `F / (b m Y)`.
pub fn lewis_stress() -> f64 {
    FORCE / (FACE * MODULE * LEWIS_Y)
}

/// The Dolan-Broghamer fillet concentration for this tooth's geometry.
pub fn fillet_concentration() -> f64 {
    let half_tooth = 0.25 * std::f64::consts::PI * MODULE / RP;
    let inv_phi = PHI.tan() - PHI;
    let theta_b = half_tooth + inv_phi;
    let alpha = (RF / (RR + RF)).asin();
    let critical_radius = (RR + RF) * alpha.cos();
    let t = 2.0 * critical_radius * theta_b.sin();
    let l = RA - critical_radius * theta_b.cos();
    let h = 0.34 - 0.458_366_2 * PHI;
    let ll = 0.316 - 0.458_366_2 * PHI;
    let m = 0.290 + 0.458_366_2 * PHI;
    h + (t / RF).powf(ll) * (t / l).powf(m)
}

/// **Limit.** The peak von Mises at the tension-side root fillet within 25% of `Kf` times the
/// Lewis stress at the finest mesh, and the peak moved by at most 5% in the last refinement. The
/// 25% is: up to 15% scatter of the Dolan-Broghamer fit against tests, stated in design texts; 5%
/// for a 3D tooth (a 20 mm face against the plane assumption) and a load spread over the tip land
/// rather than a line; the rest for the second-hand formula and Y from memory and discretization
/// at a fillet peak. This is a formula known to be approximate, and the limit says only that the
/// model and the formula agree to engineering accuracy, not that either is exact.
fn judge(rows: &[DensityResult]) -> Verdict {
    verdict_of((|| {
        let last = rows.last().ok_or("no rows")?;
        let reference = fillet_concentration() * lewis_stress();
        check_error(
            last.value("root_peak"),
            reference,
            0.25,
            "root fillet peak against Kf x Lewis",
        )?;
        stable(rows, "root_peak", 0.05, "root fillet peak")
    })())
}

pub(super) fn cases() -> Vec<Case> {
    let size = |near: f64| Sizing::AroundPoint {
        centre: [0.0, 38.0],
        r0: 10.0,
        near,
        grade: 0.3,
        far: 3.0,
    };
    vec![Case {
        id: "gear-tooth",
        title: "Part 10: single gear tooth, Lewis with Dolan-Broghamer, nu = 0.3",
        material: steel(0.3),
        densities: vec![
            density("near=0.9", "gear_n0.9", size(0.9), 0.9),
            density("near=0.6", "gear_n0.6", size(0.6), 0.6),
            density("near=0.4", "gear_n0.4", size(0.4), 0.4),
        ],
        layers: 2,
        apply,
        measure,
        iteration_budget: None,
        judge,
    }]
}
