//! Part 2: a plate with a central hole in tension, against the published finite-width net-section
//! stress concentration.

use super::*;

const PLATE_L: f64 = 300.0;
const PLATE_W: f64 = 100.0;
const PLATE_T: f64 = 2.0;
const PLATE_D: f64 = 10.0;
const PLATE_SIGMA: f64 = 100.0;

/// The window the plate's peak is read in, away from the clamp and the loaded end.
const WINDOW_LOW: [f64; 2] = [120.0, 20.0];
const WINDOW_HIGH: [f64; 2] = [180.0, 80.0];

fn in_window(p: [f64; 3]) -> bool {
    p[0] >= WINDOW_LOW[0]
        && p[0] <= WINDOW_HIGH[0]
        && p[1] >= WINDOW_LOW[1]
        && p[1] <= WINDOW_HIGH[1]
}

fn plate_apply(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
    let (faces, area) = plane_faces(mesh, 0, PLATE_L);
    check_area(area, PLATE_W * PLATE_T, "loaded end")?;
    Ok(Supports {
        fixed: plane_nodes(mesh, 0, 0.0),
        tractions: faces
            .into_iter()
            .map(|f| (f, [PLATE_SIGMA, 0.0, 0.0]))
            .collect(),
        ..Supports::default()
    })
}

fn plate_measure(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
) -> Result<Vec<(String, f64)>, String> {
    let peak = peak_in_box(mesh, solution, WINDOW_LOW, WINDOW_HIGH)?;
    // At the hole edge the only stress is the hoop stress (a free surface, and nu = 0), so von
    // Mises is that stress; sigma_net = F / ((W - d) t) with F = sigma W t.
    let sigma_net = PLATE_SIGMA * PLATE_W / (PLATE_W - PLATE_D);
    // The diagnostics locate the largest node value, which can sit below a peak set by an
    // integration point.
    let nodal_peak = mesh
        .nodes()
        .iter()
        .zip(&solution.nodal_von_mises)
        .filter(|(p, _)| in_window(**p))
        .fold(f64::MIN, |peak, (_, value)| worst_of(peak, *value));
    let (band, theta) = hole_band(mesh, solution, nodal_peak);
    let mut values = spike_values(mesh, solution, nodal_peak, sigma_net)?;
    values.extend(worst_element(mesh));
    let mut head = vec![
        ("ktn".to_string(), peak / sigma_net),
        // Shown beside the peak, never judged: the mean over the hole-edge nodes within five
        // degrees of the top and bottom of the hole, and where the peak node is. A peak well above
        // that band's mean is a local spike at one node, not the hole's stress state; the band
        // mean reads about 0.3% under the true peak, because the Kirsch hoop stress falls that far
        // on average over the band, and a further percent or so under it from the recovery reading
        // low on a hole edge's interior nodes (`solidfem-elastic`'s recovery header).
        ("ktn_hole_band".to_string(), band / sigma_net),
        ("peak_theta_deg".to_string(), theta),
    ];
    head.append(&mut values);
    Ok(head)
}

/// Diagnostics of the hole-edge peak, shown and never judged: where the peak node is (whether it
/// is a corner or a midside node, and its radius from the hole centre), the worst shape among
/// the elements meeting it, the spread of its element-by-element values before averaging, and
/// the largest integration-point von Mises in the window. A recovery spike at a polygon vertex
/// or a poor element shows as a wide spread across elements with the mean or integration-point
/// value far below the peak; a real peak shows little spread and a high integration-point value.
fn spike_values(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
    peak: f64,
    sigma_net: f64,
) -> Result<Vec<(String, f64)>, String> {
    let mut node = usize::MAX;
    for (index, p) in mesh.nodes().iter().enumerate() {
        if in_window(*p) && solution.nodal_von_mises[index] >= peak {
            node = index;
            break;
        }
    }
    if node == usize::MAX {
        return Err("the peak node was not found".to_string());
    }
    let is_corner = if node < mesh.vertex_count() { 1.0 } else { 0.0 };
    let p = mesh.nodes()[node];
    let radius = ((p[0] - 150.0).powi(2) + (p[1] - 50.0).powi(2)).sqrt();
    let (mut count, mut low, mut high, mut sum) = (0usize, f64::MAX, f64::MIN, 0.0);
    let mut worst_shape = 180.0_f64;
    let material = steel(0.0);
    for tet in mesh.tets() {
        let Some(local) = tet.iter().position(|n| *n as usize == node) else {
            continue;
        };
        let nodes = tet.map(|n| mesh.nodes()[n as usize]);
        let u = tet.map(|n| solution.displacements[n as usize]);
        let geometry = point_geometry(&nodes, NODE_NATURAL[local], local)
            .map_err(|fault| format!("peak diagnostics: {fault:?}"))?;
        let value = von_mises(&stress_at(&material, &strain_at(&geometry, &u)));
        low = low.min(value);
        high = high.max(value);
        sum += value;
        count += 1;
        worst_shape = worst_shape.min(min_dihedral_degrees([
            nodes[0], nodes[1], nodes[2], nodes[3],
        ]));
    }
    let ip_peak = solution
        .point_stresses
        .iter()
        .filter(|s| in_window(s.position))
        .map(|s| s.von_mises)
        .fold(f64::MIN, f64::max);
    Ok(vec![
        ("peak_is_corner_node".to_string(), is_corner),
        ("peak_radius".to_string(), radius),
        ("peak_elements".to_string(), count as f64),
        ("peak_element_ktn_low".to_string(), low / sigma_net),
        ("peak_element_ktn_high".to_string(), high / sigma_net),
        (
            "peak_element_ktn_mean".to_string(),
            sum / count as f64 / sigma_net,
        ),
        ("peak_min_dihedral_deg".to_string(), worst_shape),
        ("ip_peak_ktn".to_string(), ip_peak / sigma_net),
    ])
}

pub(super) fn cases() -> Vec<Case> {
    let plate_size = |near: f64| Sizing::AroundPoint {
        centre: [150.0, 50.0],
        r0: 5.0,
        near,
        grade: 0.3,
        far: 6.0,
    };
    vec![Case {
        id: "plate-hole",
        title: "Part 2: plate 300 x 100 x 2 with d = 10 hole in tension, nu = 0",
        material: steel(0.0),
        densities: vec![
            density("near=1.2", "plate_n1.2", plate_size(1.2), 1.2),
            density("near=0.8", "plate_n0.8", plate_size(0.8), 0.8),
            density("near=0.55", "plate_n0.55", plate_size(0.55), 0.55),
        ],
        layers: 2,
        apply: plate_apply,
        measure: plate_measure,
        iteration_budget: None,
        judge: plate_judge,
    }]
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
