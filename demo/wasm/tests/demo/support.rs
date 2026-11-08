// Shared by the demo engine tests: building requests from what the catalog offers, running a
// study to the end through the public `step`, and listing the deviation of each compared value.

use solidfem_wasm::catalog::{
    ranges, Combination, LoadKind, Restraint, SectionKind, StudyRequest, MATERIALS,
};
use solidfem_wasm::engine::{DemoStudy, StudyState};

/// A request for `combination` at the given range level (0 = bottom, 1 = middle, 2 = top) of
/// length, section and magnitude, for `material`.
pub fn request_at(combination: Combination, material: usize, level: usize) -> StudyRequest {
    let r = ranges(combination.section, combination.restraint, combination.load);
    StudyRequest {
        section: combination.section,
        length: r.length.points()[level],
        width: r.width.points()[level],
        height: r.height.points()[level],
        wall_fraction: r.wall_fraction.map_or(0.0, |w| w.points()[level]),
        material,
        restraint: combination.restraint,
        load: combination.load,
        magnitude: r.magnitude.points()[level],
    }
}

/// A short readable name for one case, printed with its result.
pub fn case_name(combination: Combination, material: usize, level: &str) -> String {
    format!(
        "{}/{}/{}/{}/{}",
        match combination.section {
            SectionKind::Solid => "solid",
            SectionKind::Tube => "tube",
        },
        match combination.restraint {
            Restraint::FixedOneEnd => "one_end",
            Restraint::FixedBothEnds => "both_ends",
        },
        match combination.load {
            LoadKind::Push => "push",
            LoadKind::Twist => "twist",
        },
        MATERIALS[material].name,
        level
    )
}

/// Runs a started study to its end with `budget` per step, panicking on a failure.
pub fn run_to_done(study: &mut DemoStudy, budget: u32) {
    loop {
        match study.step(budget) {
            StudyState::Done => return,
            StudyState::Failed => panic!("study failed: {:?}", study.failure()),
            StudyState::Cancelled => panic!("study cancelled"),
            _ => {}
        }
    }
}

/// Every offered material index.
pub fn material_indices() -> std::ops::Range<usize> {
    0..MATERIALS.len()
}
