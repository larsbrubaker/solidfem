//! Part 9: a thin printed column under compression. Euler buckling is a loss of stability, an
//! eigenvalue problem on the deformed geometry, and linear statics has no geometric stiffness to
//! compute it, so a linear solve would report a smooth, wrong, safe-looking compression. This
//! case never solves: it reports OUT OF SCOPE and names what is missing.

use super::*;

fn apply(_: &QuadraticTetMesh) -> Result<Supports, String> {
    Err("this part is out of scope; it is not meshed".to_string())
}

fn measure(_: &QuadraticTetMesh, _: &Solution) -> Result<Vec<(String, f64)>, String> {
    Err("this part is out of scope; it is not measured".to_string())
}

fn judge(_: &[DensityResult]) -> Verdict {
    Verdict::OutOfScope(
        "a thin column in compression fails by Euler buckling, a loss of stability: linear statics \
         has no geometric stiffness and no eigenvalue analysis, so it cannot predict the critical \
         load and would report a smooth compression instead; no number is given"
            .to_string(),
    )
}

pub(super) fn cases() -> Vec<Case> {
    vec![Case {
        id: "column-buckling",
        title: "Part 9: thin printed column under compression",
        material: steel(0.3),
        densities: Vec::new(),
        layers: 2,
        apply,
        measure,
        iteration_budget: None,
        judge,
    }]
}
