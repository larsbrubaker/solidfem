// The multigrid preconditioner against Jacobi on real 10-node stiffness: symmetry and positive
// definiteness, the rigid-body modes at every node, the closed-form cases reaching the same
// answers, identical bits across repeated runs and however setup and solve are sliced, and
// cancellation during setup and during the solve. Iteration counts staying flat under refinement
// is the one ignored test (the meshes are too large for a debug run):
// `cargo test --release -p solidfem --test validation multigrid::iterations -- --ignored --nocapture`

use solidfem::elastic::{
    assemble_stiffness, rigid_body_modes, Material, Model, ModelError, MultigridOptions,
    PcgOptions, Precision, PreconditionerChoice, Quadrature, StaticStudy, StepOutcome,
    StudyBuilder, StudyOptions,
};
use solidfem::mesh::QuadraticTetMesh;
use solidfem::meshing::quadratic_from_solid;
use solidfem::solve::{CsrMatrix, Multigrid, MultigridSetup, Preconditioner, SetupStatus};
use solidfem::tetra::{RefineOptions, Refiner, StepStatus, SurfaceMesher};

use crate::support::{bar, faces_at, fix_root, mean_displacement};

const MULTIGRID: PreconditionerChoice = PreconditionerChoice::Multigrid(MultigridOptions {
    neighbour_reach: 2.0,
    coarse_dofs: 1500,
    max_levels: 12,
    smoother_degree: 3,
    smoother_range: 30.0,
    precision: Precision::Double,
    threads: 1,
});

fn steel() -> Material {
    Material {
        youngs_modulus: 200_000.0,
        poisson_ratio: 0.3,
    }
}

/// The assembled stiffness of `mesh_size` shifted to be positive definite, with its rigid modes.
fn shifted_stiffness() -> (CsrMatrix, Vec<f64>, Vec<[f64; 3]>) {
    let mesh = bar([10.0, 1.0, 1.0], [10, 3, 3]);
    let material = steel();
    let mut matrix =
        assemble_stiffness(&mesh, &material, Quadrature::FourPoint, &mut || false).expect("valid");
    // The stiffness alone has the rigid-body null space, so the shift makes it definite without
    // touching its near-null structure: a thousandth of the largest diagonal entry.
    let diagonal = matrix.diagonal();
    let largest = diagonal.iter().cloned().fold(0.0, f64::max);
    for (row, _) in diagonal.iter().enumerate() {
        matrix.add(row, row as u32, 1e-3 * largest);
    }
    let fixed = vec![false; mesh.nodes().len()];
    let modes = rigid_body_modes(mesh.nodes(), &fixed).expect("one flag per node");
    (matrix, modes, mesh.nodes().to_vec())
}

fn built_multigrid(
    matrix: &CsrMatrix,
    modes: &[f64],
    positions: &[[f64; 3]],
    precision: Precision,
    threads: usize,
) -> Multigrid {
    let options = MultigridOptions {
        precision,
        threads,
        ..MultigridOptions::default()
    };
    let mut setup = MultigridSetup::new(matrix, modes, positions, options).expect("valid setup");
    assert_eq!(
        setup.step(usize::MAX, &mut || false),
        Ok(SetupStatus::Finished)
    );
    setup.finish().expect("finished")
}

/// A reproducible vector with entries in `[-1, 1]`.
fn sample(size: usize, seed: u64) -> Vec<f64> {
    (0..size)
        .map(|i| {
            let mixed = (i as u64 + seed).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            (mixed >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Both precisions whose cycle arithmetic is `f64`. `SingleStorage` rounds the stored operators
/// entry by entry, so two equal mirrored entries stay equal; its cycle runs the same `f64`
/// arithmetic, so the same derived bound applies to both.
#[test]
fn the_preconditioner_is_symmetric_and_positive_definite() {
    for precision in [Precision::Double, Precision::SingleStorage] {
        symmetric_and_positive_definite(precision);
    }
}

fn symmetric_and_positive_definite(precision: Precision) {
    let (matrix, modes, positions) = shifted_stiffness();
    let threads = MultigridOptions::default().threads;
    let multigrid = built_multigrid(&matrix, &modes, &positions, precision, threads);
    assert_eq!(multigrid.precision(), precision);
    let n = matrix.size();
    // Tolerance: the cycle performs at most about 20 products over the matrix's scalar entries,
    // each entry contributing at most one rounding unit relative to the magnitudes combined, so
    // the asymmetry of the bilinear form is bounded by 256 * nonzeros * eps * ||x|| * ||Py||;
    // the factor of 256 covers the Cholesky substitutions and the prolongations.
    let bound = |x: &[f64], y_image: &[f64], y: &[f64], x_image: &[f64]| {
        256.0
            * matrix.nonzeros() as f64
            * f64::EPSILON
            * (dot(x, x).sqrt() * dot(y_image, y_image).sqrt())
                .max(dot(y, y).sqrt() * dot(x_image, x_image).sqrt())
    };
    for seed in 1..=6u64 {
        let x = sample(n, seed);
        let y = sample(n, seed * 101);
        let mut px = vec![0.0; n];
        let mut py = vec![0.0; n];
        multigrid.apply(&x, &mut px);
        multigrid.apply(&y, &mut py);
        let (xpy, ypx) = (dot(&x, &py), dot(&y, &px));
        assert!(
            (xpy - ypx).abs() <= bound(&x, &py, &y, &px),
            "{precision:?} seed {seed}: x.Py = {xpy}, y.Px = {ypx}"
        );
        // Positive definiteness: x.Px is the energy of the preconditioner on a non-zero vector.
        let energy = dot(&x, &px);
        assert!(energy > 0.0, "{precision:?} seed {seed}: x.Px = {energy}");
    }
}

/// A residual three values short is out of bounds only in the last chunk of rows, which runs on a
/// worker thread, so the panic has to cross from the worker to the caller. The pool and the cycle's
/// work vectors must come out of it usable. `Single` is not exercised: it rounds the residual into
/// its own full-length vector first, so a short one does not panic there.
#[test]
fn a_panic_on_a_worker_reaches_the_caller_and_the_pool_keeps_working() {
    let (matrix, modes, positions) = shifted_stiffness();
    let n = matrix.size();
    let residual = sample(n, 3);
    for precision in [Precision::Double, Precision::SingleStorage] {
        let multigrid = built_multigrid(&matrix, &modes, &positions, precision, 4);
        assert_eq!(multigrid.threads(), 4, "the level must use the workers");
        let mut before = vec![0.0; n];
        multigrid.apply(&residual, &mut before);

        let mut output = vec![0.0; n];
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            multigrid.apply(&residual[..n - 3], &mut output);
        }));
        assert!(outcome.is_err(), "{precision:?}: a short residual panics");

        let mut after = vec![0.0; n];
        multigrid.apply(&residual, &mut after);
        for (row, (a, b)) in before.iter().zip(&after).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{precision:?}: row {row}");
        }
    }
}

#[test]
fn rigid_body_modes_lie_in_the_null_space_at_every_node_including_midsides() {
    // Tolerance: each row of K is a sum of about 100 products of terms no larger than the
    // largest entry times the mode's size (near 1 after centring), so its rounding is bounded by
    // 256 * eps * largest entry, relative to the same entry here.
    let mesh = bar([10.0, 1.0, 1.0], [6, 2, 2]);
    let material = steel();
    let matrix =
        assemble_stiffness(&mesh, &material, Quadrature::FourPoint, &mut || false).expect("valid");
    let fixed = vec![false; mesh.nodes().len()];
    let modes = rigid_body_modes(mesh.nodes(), &fixed).expect("one flag per node");
    assert_eq!(modes.len(), 3 * mesh.nodes().len() * 6);
    let largest = matrix.diagonal().iter().cloned().fold(0.0, f64::max);
    let mut product = vec![0.0; matrix.size()];
    for mode in 0..6 {
        let vector: Vec<f64> = (0..matrix.size()).map(|i| modes[i * 6 + mode]).collect();
        matrix.multiply(&vector, &mut product);
        for (row, value) in product.iter().enumerate() {
            assert!(
                value.abs() <= 256.0 * f64::EPSILON * largest * 100.0,
                "mode {mode} row {row}: K b = {value}"
            );
        }
    }
}

#[test]
fn rigid_body_modes_are_restricted_to_free_unknowns_and_checked() {
    let mesh = bar([5.0, 1.0, 1.0], [5, 1, 1]);
    let mut fixed = vec![false; mesh.nodes().len()];
    fixed[0] = true;
    fixed[1] = true;
    let modes = rigid_body_modes(mesh.nodes(), &fixed).expect("one flag per node");
    assert_eq!(modes.len(), 18 * (mesh.nodes().len() - 2));
    assert_eq!(
        rigid_body_modes(mesh.nodes(), &fixed[1..]),
        Err(ModelError::RigidBodyData {
            nodes: mesh.nodes().len(),
            flags: mesh.nodes().len() - 1,
        })
    );
}

/// Solves `model` with `choice` and returns (mean axial or transverse tip motion, iterations).
fn solve(model: Model<Material>, choice: PreconditionerChoice) -> (Vec<[f64; 3]>, u64) {
    // A 1e-11 residual: the comparison below is against the solver error, not its tolerance.
    let options = StudyOptions {
        pcg: PcgOptions {
            tolerance: 1e-11,
            max_iterations: None,
        },
        preconditioner: choice,
        ..StudyOptions::default()
    };
    let mut study = StaticStudy::new(model, options, &mut || false).expect("valid");
    loop {
        match study.step(500, &mut || false).expect("the solve succeeds") {
            StepOutcome::Converged(_) => break,
            StepOutcome::Running(_) => {}
            StepOutcome::Cancelled(_) => panic!("nothing cancelled this solve"),
        }
    }
    let iterations = study.report().iterations;
    (study.displacements(), iterations)
}

/// Compares the displacement fields of the same model under both preconditioners. Tolerance: at
/// relative residual 1e-11 and a Jacobi-scaled condition number below 5e5 (the bars here have
/// L/h of at most 20), each solve's displacement error is at most kappa * tolerance = 5e-6 of the
/// displacement, so the two answers differ by at most twice that plus rounding: limit 2e-5.
fn assert_same_answers(name: &str, model: impl Fn() -> Model<Material>) {
    let (jacobi, _) = solve(model(), PreconditionerChoice::Jacobi);
    let (multigrid, _) = solve(model(), MULTIGRID);
    let max_motion = jacobi
        .iter()
        .flat_map(|u| u.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(max_motion > 0.0, "{name}: nothing moved");
    for (a, b) in jacobi.iter().zip(&multigrid) {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() <= 2e-5 * max_motion,
                "{name}: {} against {}",
                a[axis],
                b[axis]
            );
        }
    }
}

fn clamped_end_loaded(
    size: [f64; 3],
    divisions: [usize; 3],
    material: Material,
    traction: [f64; 3],
) -> Model<Material> {
    let mesh = bar(size, divisions);
    let mut model = Model::new(mesh.clone(), material);
    fix_root(&mut model);
    for face in faces_at(&mesh, size[0]) {
        model.add_traction(face, traction);
    }
    model
}

#[test]
fn tension_cantilever_torsion_and_two_materials_agree_with_jacobi() {
    let nu0 = Material {
        youngs_modulus: 2100.0,
        poisson_ratio: 0.0,
    };
    let bar_size = [20.0, 1.0, 1.0];
    assert_same_answers("tension", || {
        clamped_end_loaded(bar_size, [20, 2, 2], nu0, [1000.0, 0.0, 0.0])
    });
    assert_same_answers("cantilever", || {
        clamped_end_loaded([10.0, 1.0, 1.0], [20, 2, 2], nu0, [0.0, 0.0, -1.0])
    });
    // Torsion: the same clamp with a traction circulating about the bar axis.
    assert_same_answers("torsion", || {
        let material = Material {
            youngs_modulus: 2100.0,
            poisson_ratio: 0.3,
        };
        let mesh = bar([10.0, 1.0, 1.0], [20, 3, 3]);
        let mut model = Model::new(mesh.clone(), material);
        fix_root(&mut model);
        for face in faces_at(&mesh, 10.0) {
            let [a, b, c] =
                [face.nodes[0], face.nodes[1], face.nodes[2]].map(|n| mesh.nodes()[n as usize]);
            let (cy, cz) = (
                (a[1] + b[1] + c[1]) / 3.0 - 0.5,
                (a[2] + b[2] + c[2]) / 3.0 - 0.5,
            );
            model.add_traction(face, [0.0, -cz, cy]);
        }
        model
    });
}

#[test]
fn two_materials_in_series_agree_with_jacobi() {
    // A material field cannot be a plain Material, so this case compares through the study
    // directly, at the same 1e-11 residual and the same derived 2e-5 limit.
    let field = |position: [f64; 3]| Material {
        youngs_modulus: if position[0] < 5.0 { 1000.0 } else { 250.0 },
        poisson_ratio: 0.0,
    };
    let mesh = bar([10.0, 1.0, 1.0], [20, 2, 2]);
    let build = |choice| {
        let mut model = Model::new(mesh.clone(), field);
        fix_root(&mut model);
        for face in faces_at(&mesh, 10.0) {
            model.add_traction(face, [1.0, 0.0, 0.0]);
        }
        let mut study = StaticStudy::new(
            model,
            StudyOptions {
                pcg: PcgOptions {
                    tolerance: 1e-11,
                    max_iterations: None,
                },
                preconditioner: choice,
                ..StudyOptions::default()
            },
            &mut || false,
        )
        .expect("valid");
        loop {
            match study.step(500, &mut || false).expect("solves") {
                StepOutcome::Converged(_) => break,
                StepOutcome::Running(_) => {}
                StepOutcome::Cancelled(_) => panic!("nothing cancelled"),
            }
        }
        study.displacements()
    };
    let (jacobi, multigrid) = (build(PreconditionerChoice::Jacobi), build(MULTIGRID));
    let max_motion = jacobi
        .iter()
        .flat_map(|u| u.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    for (a, b) in jacobi.iter().zip(&multigrid) {
        for axis in 0..3 {
            assert!((a[axis] - b[axis]).abs() <= 2e-5 * max_motion);
        }
    }
}

/// Iterations and level count of the multigrid at residual 1e-9 on a clamped end-loaded box,
/// coarsened down to 100 values so even the smallest mesh has real levels.
fn iterations_on(size: [f64; 3], divisions: [usize; 3]) -> (u64, usize) {
    let model = clamped_end_loaded(size, divisions, steel(), [0.0, 0.0, -1.0]);
    let choice = PreconditionerChoice::Multigrid(MultigridOptions {
        coarse_dofs: 100,
        ..MultigridOptions::default()
    });
    let mut study = StaticStudy::new(
        model,
        StudyOptions {
            preconditioner: choice,
            ..StudyOptions::default()
        },
        &mut || false,
    )
    .expect("valid");
    loop {
        match study.step(500, &mut || false).expect("solves") {
            StepOutcome::Converged(_) => break,
            StepOutcome::Running(_) => {}
            StepOutcome::Cancelled(_) => panic!("nothing cancelled"),
        }
    }
    let report = study.report();
    (report.iterations, report.multigrid.map_or(0, |m| m.levels))
}

#[test]
#[ignore = "large meshes; run in release, see the header of this file"]
fn iterations_stay_roughly_flat_under_refinement_on_a_cantilever_and_a_thin_plate() {
    // The bound, stated before measuring: the multigrid's convergence factor must not depend on
    // the mesh size, so iterations are bounded as the mesh refines, where Jacobi's roughly double
    // per refinement. The finest count may be at most twice the coarsest plus twenty: a factor
    // for the proportion of boundary layer changing with the mesh, the slack for the smallest
    // mesh being few iterations, and neither enough to hide growth with refinement.
    for (name, size, refinements) in [
        (
            "cantilever",
            [10.0, 1.0, 1.0],
            [[12, 3, 3], [24, 6, 6], [36, 9, 9]],
        ),
        (
            "thin plate",
            [100.0, 100.0, 0.8],
            [[10, 10, 2], [20, 20, 2], [40, 40, 2]],
        ),
    ] {
        let counts: Vec<(u64, usize)> = refinements
            .iter()
            .map(|divisions| iterations_on(size, *divisions))
            .collect();
        println!("VALIDATION multigrid iterations {name}: (iterations, levels) {counts:?}");
        assert!(
            counts[0].1 >= 3,
            "{name}: the smallest mesh has too few levels: {counts:?}"
        );
        assert!(
            counts[2].0 <= 2 * counts[0].0 + 20,
            "{name}: iterations {counts:?} grow with the mesh"
        );
    }
}

#[test]
fn repeated_runs_give_identical_bits() {
    let run = || {
        let mesh = bar([6.0, 1.0, 1.0], [8, 3, 3]);
        let mut model = Model::new(mesh.clone(), steel());
        fix_root(&mut model);
        for face in faces_at(&mesh, 6.0) {
            model.add_traction(face, [0.0, 3.0, -7.0]);
        }
        let options = StudyOptions {
            preconditioner: MULTIGRID,
            ..StudyOptions::default()
        };
        let mut study = StaticStudy::new(model, options, &mut || false).expect("valid");
        loop {
            match study.step(100, &mut || false).expect("solves") {
                StepOutcome::Converged(_) => break,
                StepOutcome::Running(_) => {}
                StepOutcome::Cancelled(_) => panic!("nothing cancelled"),
            }
        }
        study.solution(&mut || false).expect("converged")
    };
    let (first, second) = (run(), run());
    assert_eq!(first.report.iterations, second.report.iterations);
    assert_eq!(first.displacements.len(), second.displacements.len());
    for (a, b) in first.displacements.iter().zip(&second.displacements) {
        for axis in 0..3 {
            assert_eq!(a[axis].to_bits(), b[axis].to_bits());
        }
    }
}

#[test]
fn setup_and_solve_are_bit_identical_however_they_are_sliced() {
    let run = |slice: usize| {
        let mesh = bar([6.0, 1.0, 1.0], [6, 2, 2]);
        let mut model = Model::new(mesh.clone(), steel());
        fix_root(&mut model);
        for face in faces_at(&mesh, 6.0) {
            model.add_traction(face, [0.0, 3.0, -7.0]);
        }
        let options = StudyOptions {
            preconditioner: MULTIGRID,
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
        let solution = study.solution(&mut || false).expect("converged");
        (
            solution.displacements,
            solution.report.iterations,
            mesh.nodes().len(),
        )
    };
    let (one, iterations, _) = run(usize::MAX);
    assert!(iterations > 0);
    for slice in [1usize, 7, 53] {
        let (sliced, iterations_sliced, _) = run(slice);
        assert_eq!(iterations_sliced, iterations, "slice {slice}");
        assert_eq!(one.len(), sliced.len());
        for (a, b) in one.iter().zip(&sliced) {
            for axis in 0..3 {
                assert_eq!(a[axis].to_bits(), b[axis].to_bits(), "slice {slice}");
            }
        }
    }
}

#[test]
fn cancelling_during_setup_and_during_the_solve_stops_each_promptly() {
    let build = || {
        let mesh = bar([20.0, 2.0, 2.0], [40, 3, 3]);
        let mut model = Model::new(mesh.clone(), steel());
        fix_root(&mut model);
        for face in faces_at(&mesh, 20.0) {
            model.add_traction(face, [0.0, 0.0, -1.0]);
        }
        (
            StudyBuilder::new(
                model,
                StudyOptions {
                    preconditioner: MULTIGRID,
                    ..StudyOptions::default()
                },
            )
            .expect("valid"),
            mesh,
        )
    };
    let (mut builder, mesh) = build();
    // Run assembly in full, then cancel once setup has been polled ten times.
    let elements = mesh.tets().len();
    let mut polls = 0usize;
    let cancel_setup = builder.step(elements, &mut || {
        polls += 1;
        polls > elements + 10
    });
    assert_eq!(cancel_setup.err(), Some(ModelError::Cancelled));
    assert!(polls > elements, "setup was reached before the cancel");

    let (mut builder, _) = build();
    builder.step(usize::MAX, &mut || false).expect("builds");
    let mut study = builder.finish().expect("finished");
    let mut iterations = 0;
    let outcome = study
        .step(usize::MAX, &mut || {
            iterations += 1;
            iterations > 2
        })
        .expect("the cancel is not an error");
    assert!(matches!(outcome, StepOutcome::Cancelled(_)), "{outcome:?}");
    assert_eq!(study.report().iterations, 2);
    let displacements = study.displacements();
    assert!(displacements.iter().any(|u| *u != [0.0; 3]));
    // A nonzero estimate before convergence is a preview, never a result.
    assert_eq!(
        study.solution(&mut || false).err(),
        Some(ModelError::NotFinished)
    );
    let _ = mean_displacement;
}

/// A 2.5 mm-celled 40 x 20 x 0.8 plate whose interior skin vertices are moved within their cells
/// differently on the two skins, so no column lines up: the wall a remeshed surface really has.
fn unaligned_thin_plate() -> QuadraticTetMesh {
    let (cells, rows) = (16usize, 8usize);
    let (length, width, thick) = (40.0, 20.0, 0.8);
    let (hx, hy) = (length / cells as f64, width / rows as f64);
    let at = |i: usize, j: usize, layer: usize| ((layer * (cells + 1) + i) * (rows + 1) + j) as u32;
    let mut points = Vec::new();
    for layer in 0..2 {
        for i in 0..=cells {
            for j in 0..=rows {
                let on_rim = i == 0 || j == 0 || i == cells || j == rows;
                let sign = if layer == 1 { 1.0 } else { -1.0 };
                let (dx, dy) = if on_rim {
                    (0.0, 0.0)
                } else {
                    (
                        0.31 * hx * sign * if (i + j).is_multiple_of(2) { 1.0 } else { -1.0 },
                        0.23 * hy * sign * if i.is_multiple_of(2) { 1.0 } else { -1.0 },
                    )
                };
                points.push([i as f64 * hx + dx, j as f64 * hy + dy, layer as f64 * thick]);
            }
        }
    }
    let mut triangles = Vec::new();
    let mut quad = |a: u32, b: u32, c: u32, d: u32| {
        triangles.push([a, b, c]);
        triangles.push([a, c, d]);
    };
    for i in 0..cells {
        for j in 0..rows {
            quad(
                at(i, j, 1),
                at(i + 1, j, 1),
                at(i + 1, j + 1, 1),
                at(i, j + 1, 1),
            );
            quad(
                at(i, j, 0),
                at(i, j + 1, 0),
                at(i + 1, j + 1, 0),
                at(i + 1, j, 0),
            );
        }
        quad(at(i, 0, 0), at(i + 1, 0, 0), at(i + 1, 0, 1), at(i, 0, 1));
        quad(
            at(i + 1, rows, 0),
            at(i, rows, 0),
            at(i, rows, 1),
            at(i + 1, rows, 1),
        );
    }
    for j in 0..rows {
        quad(at(0, j + 1, 0), at(0, j, 0), at(0, j, 1), at(0, j + 1, 1));
        quad(
            at(cells, j, 0),
            at(cells, j + 1, 0),
            at(cells, j + 1, 1),
            at(cells, j, 1),
        );
    }
    let mut mesher = SurfaceMesher::new(points, triangles).expect("valid surface");
    assert!(matches!(
        mesher.step(usize::MAX, &mut || false),
        Ok(StepStatus::Done)
    ));
    let solid = mesher.result().expect("recovered");
    let size = 1.5 * hx;
    let options = RefineOptions {
        layers_through_walls: 2,
        ..RefineOptions::default()
    };
    let mut refiner = Refiner::new(solid, Box::new(move |_| size), options).expect("valid");
    assert!(matches!(
        refiner.step(usize::MAX, &mut || false),
        Ok(StepStatus::Done)
    ));
    let refined = refiner.result().expect("refined");
    quadratic_from_solid(&refined, Quadrature::FourPoint).expect("upgrades")
}

/// The iteration count and the finest aggregates' count and undersized count for `reach`.
fn unaligned_plate_iterations(reach: f64) -> (u64, usize, usize) {
    let mesh = unaligned_thin_plate();
    let material = Material {
        youngs_modulus: 200_000.0,
        poisson_ratio: 0.0,
    };
    let mut model = Model::new(mesh.clone(), material);
    fix_root(&mut model);
    for face in faces_at(&mesh, 40.0) {
        model.add_traction(face, [0.0, 0.0, -1.0]);
    }
    let choice = PreconditionerChoice::Multigrid(MultigridOptions {
        neighbour_reach: reach,
        ..MultigridOptions::default()
    });
    let mut study = StaticStudy::new(
        model,
        StudyOptions {
            preconditioner: choice,
            ..StudyOptions::default()
        },
        &mut || false,
    )
    .expect("valid");
    let aggregates = study.multigrid().expect("multigrid").aggregates(0).to_vec();
    let count = 1 + aggregates.iter().copied().max().unwrap_or(0) as usize;
    let mut size = vec![0usize; count];
    for a in &aggregates {
        size[*a as usize] += 1;
    }
    let undersized = size.iter().filter(|s| **s < 3).count();
    loop {
        match study.step(500, &mut || false).expect("solves") {
            StepOutcome::Converged(_) => break,
            StepOutcome::Running(_) => {}
            StepOutcome::Cancelled(_) => panic!("nothing cancelled"),
        }
    }
    (study.report().iterations, count, undersized)
}

#[test]
fn the_default_multigrid_handles_unaligned_two_layer_walls_better_than_the_narrow_reach() {
    let narrow = 2.0;
    let default = MultigridOptions::default().neighbour_reach;
    let (narrow_iterations, narrow_count, narrow_small) = unaligned_plate_iterations(narrow);
    let (iterations, count, small) = unaligned_plate_iterations(default);
    eprintln!(
        "reach {narrow}: {narrow_iterations} iterations, {narrow_small} of {narrow_count} aggregates under three nodes; reach {default}: {iterations} iterations, {small} of {count}"
    );
    // An aggregate of fewer than three nodes cannot carry the six rigid-body modes, so it leaves
    // coarse values dead; a solver user's wall should have almost none. The share is a budget: one
    // in twenty still leaves nineteen of twenty aggregates able to carry the wall's bending.
    assert!(
        small * 20 <= count,
        "{small} of {count} aggregates are under three nodes"
    );
    assert!(
        iterations < narrow_iterations,
        "the default took {iterations} iterations, the narrow reach {narrow_iterations}"
    );
}
