//! Part 10: a single gear tooth on a sector of its rim, under a tangential load at the tip,
//! against an independent structured solve of the same tooth.
//!
//! The tooth: 20 teeth, module 4 mm, 20 degree pressure angle, 20 mm face width; involute flanks
//! from the base circle to the addendum circle, a radial flank below the base circle, a 1.2 mm
//! root fillet, a root circle at 35 mm, and a rim sector out to a 24 mm bore, clamped there.
//!
//! **The reference.** The root fillet peak has no closed form for this shape, so it is measured:
//! `gear_reference.rs` solves the same load and clamp on a structured body-fitted grid of the true
//! circular fillet, refined together in all three directions, and [`ROOT_PEAK_REFERENCE`] is that
//! study extrapolated. Its weakness is that it shares this library's elements, solver and stress
//! recovery, so it can catch the mesher but not an error in those; the section stresses of a
//! solved tooth against beam theory (`the_flank_section_stresses_follow_beam_theory`) are the check
//! that does not, and they hold the bending load the fillet sees to about 1.5%.
//!
//! **The textbook figure, printed and never judged.** Lewis, `F / (b m Y)`, with the table's
//! `Y(N = 20) = 0.322` (Shigley's form factor table for full-depth 20 degree teeth, from memory)
//! and the Dolan and Broghamer photoelastic fit `Kf = H + (t / r)^L (t / l)^M`, with
//! `H = 0.34 - 0.4583662 phi`, `L = 0.316 - 0.4583662 phi`, `M = 0.290 + 0.4583662 phi`, phi in
//! radians (also from memory, with a stated scatter of 10 to 15%), gives 54.1 MPa here. It is not
//! this tooth's stress, for two reasons. The table's `Y` is for a generated tooth, which thickens
//! towards its root; this outline's radial flank necks to 6.75 mm at the fillet, so its own form
//! factor, the least `t^2 / (6 l m)` over the outline, is 0.231, and `6 F l / (b t^2)` at the
//! fillet is 52.7 MPa rather than the table's 38.8. And the fit is for generated trochoidal
//! fillets, while this one is a circular arc between a radial flank and the root circle, near a
//! 90 degree shoulder, outside the fit's domain: against that 52.7 MPa the solved peak is a
//! concentration of about 1.8, where the fit says 1.39.

use super::*;

const MODULE: f64 = 4.0;
const TEETH: f64 = 20.0;
const FACE: f64 = 20.0;
const FORCE: f64 = 1000.0;
const LEWIS_Y: f64 = 0.322;
const PHI: f64 = 0.349_065_850_398_865_9;

/// The converged root fillet peak, von Mises MPa, from the structured refinement study of
/// `gear_reference.rs`. The grids of `ROOT_PEAK_REFERENCE_GRIDS` (139 thousand, 396 thousand and
/// 1.29 million free values) gave 91.381, 92.255 and 92.658, an observed rate of 1.91 and an
/// extrapolation of 93.00; a rate of 1.5 or 2 instead puts it between 92.98 and 93.14. The
/// `gear_reference_extrapolates_...` test redoes this by a real solve in release.
pub const ROOT_PEAK_REFERENCE: f64 = 93.0;
const RP: f64 = 0.5 * TEETH * MODULE;
const RA: f64 = RP + MODULE;
const RR: f64 = RP - 1.25 * MODULE;
const RF: f64 = 0.3 * MODULE;
/// The radial limit of the clamped bore and the tolerance on its surface.
const RIN: f64 = 24.0;

fn radius(p: [f64; 3]) -> f64 {
    (p[0] * p[0] + p[1] * p[1]).sqrt()
}

pub(super) fn apply(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
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

pub(super) fn measure(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
) -> Result<Vec<(String, f64)>, String> {
    // The tension-side root fillet is on the side opposite the load; +x load, so x < 0, within a
    // window round the fillet radially.
    let peak = solution
        .peak_von_mises(mesh, |p| {
            p[0] < 0.0 && (RR - 0.5..=RR + 3.0 * RF).contains(&radius(p))
        })
        .ok_or("no node or integration point in the root fillet window")?;
    Ok(vec![
        ("root_peak".to_string(), peak),
        ("lewis_kf_info".to_string(), lewis_kf_information()),
    ])
}

/// The Lewis bending stress for the table's form factor, MPa: `F / (b m Y)`. Information only;
/// see the header for why it is not this tooth's stress.
pub fn lewis_stress() -> f64 {
    FORCE / (FACE * MODULE * LEWIS_Y)
}

/// The figure the textbook formulas give, MPa, printed in the case's table and never judged:
/// [`fillet_concentration`] times [`lewis_stress`].
pub fn lewis_kf_information() -> f64 {
    fillet_concentration() * lewis_stress()
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

/// **Limit.** The peak von Mises at the tension-side root fillet within 5% of
/// [`ROOT_PEAK_REFERENCE`] at the finest mesh, and the peak moved by at most 5% in the last
/// refinement. The 5% was fixed before any structured row was read, as the room for a peak read
/// over the discrete nodes and integration points of an unstructured mesh to sit a few percent
/// either side of a smooth peak, depending on which of them land near it and on its surface's
/// chords across a fillet the reference resolves as a true arc, plus the reference's own
/// uncertainty of well under 1%.
fn judge(rows: &[DensityResult]) -> Verdict {
    verdict_of((|| {
        let last = rows.last().ok_or("no rows")?;
        check_error(
            last.value("root_peak"),
            ROOT_PEAK_REFERENCE,
            0.05,
            "root fillet peak against the structured reference",
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
        title: "Part 10: single gear tooth against a structured reference solve, nu = 0.3",
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
