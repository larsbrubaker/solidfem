// The multigrid's worker threads against one thread, through the production study: on a stocky
// cantilever and a thin plate, in every precision, solving on 2 or 7 threads gives the iterations,
// residual, displacements and stresses of a one-thread solve to the bit, also when the build and
// the solve are sliced. A wrong answer on some thread count would be invisible on another machine,
// which is why the comparison is on bits, and why the pool's size is asserted: a hierarchy too
// small to use the workers would pass without testing anything.

use solidfem::elastic::{
    Material, Model, MultigridOptions, PcgOptions, Precision, PreconditionerChoice, Solution,
    StepOutcome, StudyBuilder, StudyOptions,
};

use crate::support::{bar, faces_at, fix_root};

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

/// Solves to 1e-9 on `threads` threads in slices of `slice`, returning the threads the hierarchy
/// actually runs on with the solution.
fn solve(
    model: Model<Material>,
    precision: Precision,
    threads: usize,
    slice: usize,
) -> (usize, Solution) {
    let options = StudyOptions {
        pcg: PcgOptions {
            tolerance: 1e-9,
            max_iterations: None,
        },
        preconditioner: PreconditionerChoice::Multigrid(MultigridOptions {
            coarse_dofs: 100,
            precision,
            threads,
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
    let used = study.multigrid().expect("multigrid chosen").threads();
    loop {
        match study.step(slice, &mut || false).expect("solves") {
            StepOutcome::Converged(_) => break,
            StepOutcome::Running(_) => {}
            StepOutcome::Cancelled(_) => panic!("nothing cancelled"),
        }
    }
    (used, study.solution(&mut || false).expect("converged"))
}

fn assert_same_bits(one: &Solution, other: &Solution, what: &str) {
    assert_eq!(
        one.report.iterations, other.report.iterations,
        "{what}: iterations"
    );
    assert_eq!(
        one.report.relative_residual.to_bits(),
        other.report.relative_residual.to_bits(),
        "{what}: residual"
    );
    assert_eq!(one.displacements.len(), other.displacements.len());
    for (node, (a, b)) in one
        .displacements
        .iter()
        .zip(&other.displacements)
        .enumerate()
    {
        for axis in 0..3 {
            assert_eq!(
                a[axis].to_bits(),
                b[axis].to_bits(),
                "{what}: node {node} axis {axis}"
            );
        }
    }
    for (node, (a, b)) in one
        .nodal_von_mises
        .iter()
        .zip(&other.nodal_von_mises)
        .enumerate()
    {
        assert_eq!(a.to_bits(), b.to_bits(), "{what}: von Mises at node {node}");
    }
}

fn thread_counts_match(name: &str, model: impl Fn() -> Model<Material>) {
    for precision in [
        Precision::Double,
        Precision::SingleStorage,
        Precision::Single,
    ] {
        let (used, one) = solve(model(), precision, 1, usize::MAX);
        assert_eq!(used, 1, "{name} {precision:?}: one thread asked for");
        for threads in [2, 7] {
            let (used, many) = solve(model(), precision, threads, usize::MAX);
            assert_eq!(
                used, threads,
                "{name} {precision:?}: the hierarchy must be large enough to use the workers"
            );
            assert_same_bits(
                &one,
                &many,
                &format!("{name} {precision:?} {threads} threads"),
            );
        }
        let (_, sliced) = solve(model(), precision, 7, 5);
        assert_same_bits(
            &one,
            &sliced,
            &format!("{name} {precision:?} 7 threads sliced"),
        );
    }
}

/// `usize::MAX` is a natural way to ask for "no limit": it must start the fixed ceiling of workers
/// and not try to allocate a thread for every value of `usize`. The ceiling does not depend on the
/// machine's processors, so 64 here holds on a small runner too.
#[test]
fn an_unbounded_thread_count_starts_the_ceiling_and_keeps_the_bits() {
    let model = || loaded([10.0, 1.0, 1.0], [12, 3, 3]);
    let (_, one) = solve(model(), Precision::SingleStorage, 1, usize::MAX);
    let (used, many) = solve(model(), Precision::SingleStorage, usize::MAX, usize::MAX);
    assert_eq!(used, 64);
    assert_same_bits(&one, &many, "usize::MAX threads");
}

#[test]
fn a_cantilever_has_the_same_bits_on_every_thread_count() {
    thread_counts_match("cantilever", || loaded([10.0, 1.0, 1.0], [12, 3, 3]));
}

#[test]
fn a_plate_has_the_same_bits_on_every_thread_count() {
    thread_counts_match("plate", || loaded([20.0, 20.0, 1.0], [8, 8, 2]));
}
