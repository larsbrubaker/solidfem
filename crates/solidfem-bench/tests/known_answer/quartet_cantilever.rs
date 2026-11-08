// A cantilever described by a distance field, meshed by Quartet and solved, against Timoshenko. No
// percentage limit is committed here: the only available derivation assumed Quartet's elements at
// a given grid match a structured mesh's element size, and measurement showed its tets are about
// two grid cells across, so a limit derived that way would be unsupported. The claim that survives
// is the one that needs no such assumption: a finer grid gives a closer answer, measured here
// across three grids with the end selection as written.

use solidfem_bench::known_answer::{
    cantilever_case, evaluate, timoshenko_deflection, EndSelection,
};
use solidfem_bench::pipeline::{advance, finish, prepare, Ending};
use solidfem_elastic::Clock;

fn clock() -> Clock {
    let start = std::time::Instant::now();
    Box::new(move || start.elapsed().as_secs_f64())
}

/// The relative tip-deflection error at `cells_across` grid cells through the depth.
fn deflection_error(cells_across: u32) -> f64 {
    let case = cantilever_case(cells_across);
    let mut prepared = prepare(&case, &mut clock, u64::MAX).expect("meshes and assembles");
    while advance(&mut prepared, 500).expect("solves") {}
    let measured = finish(&mut prepared, Ending::Converged).expect("recovers");
    let solution = measured.solution.expect("converged");
    let (tip, _) = evaluate(&prepared.mesh, &solution, prepared.dx, EndSelection::Plane);
    (tip / timoshenko_deflection() - 1.0).abs()
}

#[test]
fn a_finer_quartet_grid_brings_the_cantilever_closer_to_timoshenko() {
    let errors = [4, 6, 8].map(deflection_error);
    assert!(
        errors[1] < errors[0] && errors[2] < errors[1],
        "errors at h/dx 4, 6, 8 did not fall: {errors:?}"
    );
}
