//! Part 8: a 1 mm walled hollow box pressed at the centre of its top face, on a real unaligned
//! remeshed surface. There is no closed form; the reference is the mesh-refinement study itself.
//!
//! The box is 40 x 40 x 20 mm with 1 mm walls and a closed internal void, clamped on its bottom
//! face, with a 10 N downward load spread over a 3 mm radius patch at the centre of the top face.
//! The reported quantity is the mean deflection of the patch. The multigrid must solve this thin
//! closed shell of unaligned skins within the iteration budget.

use super::*;

const SIDE: f64 = 40.0;
const HEIGHT: f64 = 20.0;
const PATCH: f64 = 3.0;
const FORCE: f64 = 10.0;
/// The iteration budget, enforced: a solve over this count is a named failure. The hardest plate
/// here took about 540 iterations at 170 thousand free values; 3000 is five times
/// that for a harder thin closed shell.
const ITERATION_BUDGET: u64 = 3000;

/// The faces of the top skin whose centroid is within the patch radius of the centre: a patch of
/// faces, so it exists at every density, unlike one requiring every corner inside the radius.
fn patch_faces(mesh: &QuadraticTetMesh) -> Vec<BoundaryFace> {
    faces_where(mesh, |p| (p[2] - HEIGHT).abs() <= PLANE_TOL)
        .into_iter()
        .filter(|face| {
            let mut c = [0.0; 2];
            for node in &face.nodes[..3] {
                let p = mesh.nodes()[*node as usize];
                c[0] += p[0] / 3.0;
                c[1] += p[1] / 3.0;
            }
            let (dx, dy) = (c[0] - 0.5 * SIDE, c[1] - 0.5 * SIDE);
            (dx * dx + dy * dy).sqrt() <= PATCH
        })
        .collect()
}

fn apply(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
    let faces = patch_faces(mesh);
    let area: f64 = faces.iter().map(|f| face_area(mesh, f)).sum();
    if faces.is_empty() || area <= 0.0 {
        return Err("no faces in the top patch".to_string());
    }
    Ok(Supports {
        fixed: plane_nodes(mesh, 2, 0.0),
        tractions: faces
            .into_iter()
            .map(|f| (f, [0.0, 0.0, -FORCE / area]))
            .collect(),
        prescribed: Vec::new(),
    })
}

fn measure(mesh: &QuadraticTetMesh, solution: &Solution) -> Result<Vec<(String, f64)>, String> {
    // The mean deflection of the loaded faces, weighted by area: a quadratic triangle's integral
    // weights are zero at its corners and a third at each midside node.
    let (mut weighted, mut total_area) = (0.0, 0.0);
    for face in patch_faces(mesh) {
        let area = face_area(mesh, &face);
        for node in &face.nodes[3..] {
            weighted += area * -solution.displacements[*node as usize][2] / 3.0;
        }
        total_area += area;
    }
    if total_area <= 0.0 {
        return Err("no faces in the top patch".to_string());
    }
    Ok(vec![("patch_w".to_string(), weighted / total_area)])
}

/// **Criterion.** With no closed form, the reference is the mesh-refinement study: the patch
/// deflection must have moved by at most 2% in the last refinement, with the iteration budget of
/// [`ITERATION_BUDGET`] holding at every density. A 2% change over a halving of element
/// size bounds the error on a slowly converging thin-wall response to about 2 to 4% of the
/// finest value, a meaningful engineering answer; a larger one fails.
fn judge(rows: &[DensityResult]) -> Verdict {
    verdict_of(stable(rows, "patch_w", 0.02, "patch deflection"))
}

pub(super) fn cases() -> Vec<Case> {
    vec![Case {
        id: "hollow-box",
        title: "Part 8: hollow box 40 x 40 x 20 with 1 mm walls, centre push",
        material: steel(0.3),
        densities: vec![
            density("h=4.0", "hollow_h4.0", Sizing::Uniform(4.0), 4.0),
            density("h=3.0", "hollow_h3.0", Sizing::Uniform(3.0), 3.0),
            density("h=2.0", "hollow_h2.0", Sizing::Uniform(2.0), 2.0),
        ],
        layers: 2,
        apply,
        measure,
        iteration_budget: Some(ITERATION_BUDGET),
        judge,
    }]
}
