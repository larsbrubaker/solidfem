//! Part 7: a uniform snap-fit cantilever under an enforced tip deflection, judged by the
//! permissible-strain method for the peak strain at the root.
//!
//! The strip is 20 long, 5 wide, 2 thick, ABS-like (E = 2000 MPa, nu = 0.35). The method's strain
//! at the root of a uniform rectangular cantilever is `1.5 * delta * T / L^2`. That is small
//! deflection beam theory. Two cases: `delta = 1 mm` (`delta / L = 0.05`, root rotation about
//! 0.075 radians, within small-rotation linear statics, geometric effects of order `theta^2`,
//! well under 1%), which is judged; and `delta = 5 mm` (`delta / L = 0.25`), where a solver of
//! linear statics is outside its validity and the verdict must say so instead of giving a number.

use super::*;

const L: f64 = 20.0;
const WIDTH: f64 = 5.0;
const T: f64 = 2.0;
const SNAP_E: f64 = 2000.0;
const SNAP_NU: f64 = 0.35;
const DELTA: f64 = 1.0;
/// Enforced deflections beyond this fraction of the length are outside linear statics.
const VALIDITY_RATIO: f64 = 0.1;

fn material() -> Material {
    Material {
        youngs_modulus: SNAP_E,
        poisson_ratio: SNAP_NU,
    }
}

fn apply(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
    let mut supports = Supports {
        fixed: plane_nodes(mesh, 0, 0.0),
        ..Supports::default()
    };
    // Every node of the tip face is pushed to the enforced deflection in z only, so the tip may
    // shorten and rotate as the beam bends.
    for node in plane_nodes(mesh, 0, L) {
        supports.prescribed.push((node, 2, -DELTA));
    }
    if supports.prescribed.is_empty() {
        return Err("no nodes on the tip face".to_string());
    }
    Ok(supports)
}

fn measure(mesh: &QuadraticTetMesh, solution: &Solution) -> Result<Vec<(String, f64)>, String> {
    // The bending strain along the surface at a section a small distance from the root is
    // measured, then referred back to the root through the linear strain of a tip-loaded beam: the
    // root line itself is the clamp corner, whose nodal value is not a usable strain.
    let targets = nodes_where(mesh, |p| {
        p[0] >= 0.15 * L
            && p[0] <= 0.25 * L
            && (p[2].abs() <= PLANE_TOL || (p[2] - T).abs() <= PLANE_TOL)
            && p[1] > 0.1 * WIDTH
            && p[1] < 0.9 * WIDTH
    });
    if targets.is_empty() {
        return Err("no surface nodes near the root".to_string());
    }
    let strains = nodal_tensor(mesh, solution, &targets, strain_at)?;
    let (mut root, mut count) = (0.0, 0usize);
    for (node, strain) in targets.iter().zip(&strains) {
        let x = mesh.nodes()[*node as usize][0];
        root += strain[0].abs() / (1.0 - x / L);
        count += 1;
    }
    Ok(vec![("root_strain".to_string(), root / count as f64)])
}

/// The permissible-strain formula for a uniform rectangular cantilever.
fn root_strain_formula(delta: f64) -> f64 {
    1.5 * delta * T / (L * L)
}

/// **Limit.** The surface bending strain referred to the root within 4% of `1.5 delta T / L^2` at
/// the finest mesh, and it moved by at most 3% by the last refinement. The formula is Euler-Bernoulli
/// bending of a slender beam: at L/T = 10 the shear deflection is about 0.8% of the bending
/// deflection (so 0.8% less strain than the formula for the same enforced deflection), and the
/// clamped root adds an elastic flexibility of order a percent or two. 4% is that, 1%
/// for the extrapolation from a band away from the root to the root, and 2% for discretization.
fn judge(rows: &[DensityResult]) -> Verdict {
    verdict_of((|| {
        if DELTA / L > VALIDITY_RATIO {
            return Err(format!(
                "linear statics is outside its validity at delta / L = {}",
                DELTA / L
            ));
        }
        let last = rows.last().ok_or("no rows")?;
        check_error(
            last.value("root_strain"),
            root_strain_formula(DELTA),
            0.04,
            "root strain",
        )?;
        stable(rows, "root_strain", 0.03, "root strain")
    })())
}

fn outside_validity(_: &[DensityResult]) -> Verdict {
    Verdict::OutOfScope(
        "an enforced tip deflection of 5 mm on a 20 mm snap-fit (delta / L = 0.25) is outside \
         linear statics: large rotation and stress stiffening (geometric nonlinearity) are not \
         modelled, so no number is given"
            .to_string(),
    )
}

pub(super) fn cases() -> Vec<Case> {
    vec![
        Case {
            id: "snapfit",
            title: "Part 7: snap-fit strip 20 x 5 x 2, enforced tip deflection 1 mm",
            material: material(),
            densities: vec![
                density("h=1.0", "snap_h1.0", Sizing::Uniform(1.0), 1.0),
                density("h=0.7", "snap_h0.7", Sizing::Uniform(0.7), 0.7),
                density("h=0.5", "snap_h0.5", Sizing::Uniform(0.5), 0.5),
            ],
            layers: 2,
            apply,
            measure,
            iteration_budget: None,
            judge,
        },
        Case {
            id: "snapfit-large",
            title: "Part 7: snap-fit, enforced tip deflection 5 mm (outside linear statics)",
            material: material(),
            densities: Vec::new(),
            layers: 2,
            apply,
            measure,
            iteration_budget: None,
            judge: outside_validity,
        },
    ]
}
