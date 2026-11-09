//! Part 4: an L-bracket with a sharp re-entrant corner, and with a fillet. Neither shape has an
//! accuracy reference: what these cases test is mesh stability of the tip deflection, that the
//! sharp corner's peak is reported as a singularity, and that the fillet's peak converges. The
//! verdict says so on its row, never a bare PASS.

use super::*;

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
        ..Supports::default()
    })
}

/// A rough frame estimate of the tip deflection, shown beside the measurement and never judged,
/// because it is not an accuracy reference: two centre-line beams, each carried at full beam
/// flexibility all the way to where the centre lines meet, `F L^3 / (3 E I) + F L^2 H / (E I) +
/// F H / (E A)`, L = H = 95, giving 1.3728 mm. A solid corner is stiffer than beams running
/// through it, so a true deflection is expected below this. The stiff side is a rigid 10 x 10
/// corner square with both legs bending only outside it, 1.266 mm by hand: `F 90^3 / (3 E I)` for
/// the horizontal leg plus the corner's rotation `F 95 * 90 / (E I)` times the 95 mm arm. The
/// measured tips, 2.1% and 5.7% below 1.3728, lie between the two, the fillet being the stiffer.
/// So this is the soft side of a hand estimate and nothing more; the deflection is not checked
/// against it, and there is no closed form to check it against.
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
/// reader must not use. Failing to detect it, here, is a failure of this benchmark. A pass claims
/// mesh stability and the detection, and no accuracy against any reference.
fn l_sharp_judge(rows: &[DensityResult]) -> Verdict {
    let outcome = (|| -> Result<f64, String> {
        stable(rows, "tip", 0.01, "tip deflection")?;
        let peaks = series(rows, "peak");
        match classify_peak(&sizes(rows), &peaks, 0.03) {
            Some(PeakBehaviour::Singular { exponent }) => Ok(exponent),
            other => Err(format!(
                "the sharp corner's peak did not grow like a singularity: {other:?}, series {peaks:?}"
            )),
        }
    })();
    match outcome {
        Ok(exponent) => Verdict::PassLimited(format!(
            "tip deflection mesh-stable, no accuracy reference; corner peak: singularity detected, \
             growing as h^-{exponent:.2}, not a usable stress"
        )),
        Err(reason) => Verdict::Fail(reason),
    }
}

/// **Limit, fillet.** Tip deflection moved by at most 1% in the last refinement, as for the sharp
/// corner. The fillet's peak is a real number and must be converged, not singular: the last
/// refinement moves it by at most 3%. With error shrinking about as `h^2` and a refinement ratio
/// near 1.6, a 3% last change means the coarser mesh is still 5% off and the finest about 2%. That
/// reading is from the mesh sequence alone: there is no closed-form value and no measured
/// reference to hold the fillet to, so a pass claims mesh stability and not accuracy.
fn l_fillet_judge(rows: &[DensityResult]) -> Verdict {
    let outcome = (|| -> Result<(), String> {
        stable(rows, "tip", 0.01, "tip deflection")?;
        let peaks = series(rows, "peak");
        match classify_peak(&sizes(rows), &peaks, 0.03) {
            Some(PeakBehaviour::Converged { .. }) => Ok(()),
            other => Err(format!(
                "the fillet's peak did not converge: {other:?}, series {peaks:?}"
            )),
        }
    })();
    match outcome {
        Ok(()) => Verdict::PassLimited("mesh-stable, no accuracy reference".to_string()),
        Err(reason) => Verdict::Fail(reason),
    }
}

pub(super) fn cases() -> Vec<Case> {
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
            id: "l-sharp",
            title: "Part 4: L-bracket, sharp re-entrant corner, tip load, nu = 0.3",
            material: steel(0.3),
            densities: vec![
                density("c=1.0", "lsharp_c1.0", sharp_size(1.0), 1.0),
                density("c=0.5", "lsharp_c0.5", sharp_size(0.5), 0.5),
                density("c=0.25", "lsharp_c0.25", sharp_size(0.25), 0.25),
            ],
            layers: 2,
            apply: bracket_apply,
            measure: bracket_measure,
            iteration_budget: None,
            judge: l_sharp_judge,
        },
        Case {
            id: "l-fillet",
            title: "Part 4: L-bracket, 5 mm fillet at the corner, tip load, nu = 0.3",
            material: steel(0.3),
            densities: vec![
                density("f=1.2", "lfillet_f1.2", fillet_size(1.2), 1.2),
                density("f=0.8", "lfillet_f0.8", fillet_size(0.8), 0.8),
                density("f=0.5", "lfillet_f0.5", fillet_size(0.5), 0.5),
            ],
            layers: 2,
            apply: bracket_apply,
            measure: bracket_measure,
            iteration_budget: None,
            judge: l_fillet_judge,
        },
    ]
}
