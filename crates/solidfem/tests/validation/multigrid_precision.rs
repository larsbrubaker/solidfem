// The multigrid hierarchy stored in f32, with the cycle's arithmetic in f64 (SingleStorage) or in
// f32 (Single), under the f64 conjugate gradient solve: on a stocky cantilever and on a thin plate
// each reaches 1e-9 or its own rounding floor, in iteration counts pinned to measured values,
// gives the same answer to within what that residual allows, and gives identical bits however the
// build and the solve are sliced. The default f64 path's
// iterations and displacement bits are pinned by a fingerprint.

use solidfem::elastic::{
    Material, Model, MultigridOptions, PcgOptions, Precision, PreconditionerChoice, Solution,
    StepOutcome, StudyBuilder, StudyOptions,
};

use crate::support::{bar, faces_at, fix_root};

const TOLERANCE: f64 = 1e-9;

fn steel() -> Material {
    Material {
        youngs_modulus: 200_000.0,
        poisson_ratio: 0.3,
    }
}

/// Clamped at x = 0 and pulled sideways and down at the far end.
fn loaded(size: [f64; 3], divisions: [usize; 3]) -> Model<Material> {
    let mesh = bar(size, divisions);
    let mut model = Model::new(mesh.clone(), steel());
    fix_root(&mut model);
    for face in faces_at(&mesh, size[0]) {
        model.add_traction(face, [0.0, 3.0, -7.0]);
    }
    model
}

/// Builds and solves in slices of `slice` rows and iterations, to a 1e-9 true residual.
fn solve(model: Model<Material>, precision: Precision, slice: usize) -> Solution {
    let options = StudyOptions {
        pcg: PcgOptions {
            tolerance: TOLERANCE,
            max_iterations: None,
        },
        preconditioner: PreconditionerChoice::Multigrid(MultigridOptions {
            coarse_dofs: 100,
            precision,
            ..MultigridOptions::default()
        }),
        ..StudyOptions::default()
    };
    let mut builder = StudyBuilder::new(model, options).expect("valid");
    loop {
        let progress = builder.step(slice, &mut || false).expect("builds");
        if progress.elements_done == progress.elements_total && progress.setup_finished {
            break;
        }
    }
    let mut study = builder.finish().expect("finished");
    loop {
        match study.step(slice, &mut || false).expect("solves") {
            StepOutcome::Converged(_) => break,
            StepOutcome::Running(_) => {}
            StepOutcome::Cancelled(_) => panic!("nothing cancelled"),
        }
    }
    study.solution(&mut || false).expect("converged")
}

/// Solves in f64 and in `precision` and checks the latter reached the tolerance in `expected`
/// iterations (f64, `precision`), with the same answer. Tolerance on the answer: both solves stop at a
/// relative true residual of 1e-9, and the Jacobi-scaled condition number of these meshes is
/// below 1e6, so each displacement is within 1e-3 of the exact discrete one relative to the
/// largest; twice that bounds their difference.
fn assert_matches_double(
    name: &str,
    model: impl Fn() -> Model<Material>,
    precision: Precision,
    expected: (u64, u64),
    double_floor: bool,
) {
    let double = solve(model(), Precision::Double, usize::MAX);
    let single = solve(model(), precision, usize::MAX);
    println!(
        "VALIDATION multigrid precision {name} {precision:?}: iterations f64 {} f32 {}, residual f64 {:.3e} f32 {:.3e}, hierarchy bytes f64 {} f32 {}",
        double.report.iterations,
        single.report.iterations,
        double.report.relative_residual,
        single.report.relative_residual,
        double.report.multigrid.map_or(0, |m| m.memory_bytes),
        single.report.multigrid.map_or(0, |m| m.memory_bytes),
    );
    // Each solve stops at 1e-9 or, where f64 cannot resolve 1e-9 for the system, at its own
    // rounding floor by the solver's rule. The plate is such a system and the cantilever is not,
    // which the f64 run must show for this test to measure what it says.
    assert_eq!(
        double.report.at_rounding_floor, double_floor,
        "{name}: f64 stopped at {}",
        double.report.relative_residual
    );
    for (label, run) in [("f64", &double), ("f32", &single)] {
        assert!(
            run.report.relative_residual <= TOLERANCE || run.report.at_rounding_floor,
            "{name}: {label} stopped at {} short of its floor",
            run.report.relative_residual
        );
    }
    assert_eq!(
        (double.report.iterations, single.report.iterations),
        expected,
        "{name}: (f64, f32) iterations"
    );
    let largest = double
        .displacements
        .iter()
        .flat_map(|u| u.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(largest > 0.0, "{name}: nothing moved");
    for (a, b) in double.displacements.iter().zip(&single.displacements) {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() <= 2e-3 * largest,
                "{name}: {} against {}",
                a[axis],
                b[axis]
            );
        }
    }
    let single_bytes = single.report.multigrid.map_or(0, |m| m.memory_bytes);
    let double_bytes = double.report.multigrid.map_or(0, |m| m.memory_bytes);
    assert!(
        single_bytes < double_bytes,
        "{name}: f32 hierarchy {single_bytes} bytes, f64 {double_bytes}"
    );
}

#[test]
fn an_f32_cycle_reaches_the_f64_tolerance_on_a_stocky_cantilever() {
    assert_matches_double(
        "stocky cantilever",
        || loaded([10.0, 1.0, 1.0], [12, 3, 3]),
        Precision::Single,
        (66, 86),
        false,
    );
}

#[test]
fn an_f32_cycle_reaches_the_f64_tolerance_on_a_thin_plate() {
    assert_matches_double(
        "thin plate",
        || loaded([20.0, 20.0, 1.0], [8, 8, 2]),
        Precision::Single,
        (84, 90),
        true,
    );
}

// Rounding only the stored operators gives exactly the f64 iteration counts on both cases. That
// is measured, not guaranteed: the preconditioner differs from the f64 one by about 6e-8
// relative, which happens to move no iteration's stopping test here. The counts themselves are
// the same on every platform, because every operation is IEEE with a fixed order and no fused
// multiply-add; the exception is 32-bit x86 without SSE2, whose x87 unit rounds intermediates
// differently.
#[test]
fn an_f32_stored_hierarchy_keeps_the_f64_iterations_on_a_stocky_cantilever() {
    assert_matches_double(
        "stocky cantilever",
        || loaded([10.0, 1.0, 1.0], [12, 3, 3]),
        Precision::SingleStorage,
        (66, 66),
        false,
    );
}

#[test]
fn an_f32_stored_hierarchy_keeps_the_f64_iterations_on_a_thin_plate() {
    assert_matches_double(
        "thin plate",
        || loaded([20.0, 20.0, 1.0], [8, 8, 2]),
        Precision::SingleStorage,
        (84, 84),
        true,
    );
}

#[test]
fn an_f32_hierarchy_gives_identical_bits_however_build_and_solve_are_sliced() {
    for precision in [Precision::SingleStorage, Precision::Single] {
        sliced_bits_match(precision);
    }
}

fn sliced_bits_match(precision: Precision) {
    let model = || loaded([6.0, 1.0, 1.0], [6, 2, 2]);
    let one = solve(model(), precision, usize::MAX);
    let again = solve(model(), precision, usize::MAX);
    assert!(one.report.iterations > 0);
    for slice in [1usize, 7, 53] {
        let sliced = solve(model(), precision, slice);
        for other in [&again, &sliced] {
            assert_eq!(
                other.report.iterations, one.report.iterations,
                "{precision:?} slice {slice}"
            );
            for (a, b) in one.displacements.iter().zip(&other.displacements) {
                for axis in 0..3 {
                    assert_eq!(
                        a[axis].to_bits(),
                        b[axis].to_bits(),
                        "{precision:?} slice {slice}"
                    );
                }
            }
        }
    }
}

/// The iteration count and an FNV-1a hash of every displacement's bits.
fn fingerprint(solution: &Solution) -> (u64, u64) {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in solution.displacements.iter().flat_map(|u| u.iter()) {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    (solution.report.iterations, hash)
}

/// Pinned because the f64 cycle promises the same bits whatever precision options exist beside
/// it: a change that moves these must say so and why, never pass by accident.
#[test]
fn the_default_f64_cycle_keeps_its_bits() {
    let cantilever = solve(
        loaded([10.0, 1.0, 1.0], [12, 3, 3]),
        Precision::Double,
        usize::MAX,
    );
    let plate = solve(
        loaded([20.0, 20.0, 1.0], [8, 8, 2]),
        Precision::Double,
        usize::MAX,
    );
    println!(
        "VALIDATION multigrid f64 fingerprint: cantilever {:?} plate {:?}",
        fingerprint(&cantilever),
        fingerprint(&plate)
    );
    assert_eq!(fingerprint(&cantilever), (66, 14157103597919219128));
    assert_eq!(fingerprint(&plate), (84, 9110598921777769261));
}
