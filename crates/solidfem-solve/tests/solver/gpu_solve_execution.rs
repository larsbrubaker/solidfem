// A whole conjugate gradient solve preconditioned by the V-cycle on this machine's GPU (native
// only): the production `Pcg` stays in f64 on the CPU with its own matrix product and vectors, and
// its preconditioner is `DeviceCycle` through the public `Preconditioner` trait, rounding the
// residual to f32, running the cycle on the device with the coarsest solve on the CPU, and
// widening the correction back. The small test converges to the CPU solve's answer within a
// tolerance derived from the requested residual. `measure_a_gpu_solve_near_a_million_unknowns`
// (ignored; run it in release with `--ignored --nocapture`) reports iterations, per-cycle phases
// and total solve time against the CPU at one and eight threads.
//
// The measurement requests the adapter's own limits, so the finest level's values array fits one
// storage binding. A browser's default limits would refuse this size (`GpuError::BindingTooLarge`,
// asserted below), and this code has no row split to run it there.

#![cfg(not(target_arch = "wasm32"))]

use std::time::Instant;

use solidfem_solve::gpu::{BlockMatrix, GpuError, GUARANTEED_BINDING_BYTES};
use solidfem_solve::{
    CsrMatrix, Multigrid, MultigridOptions, Pcg, PcgOptions, Precision, Preconditioner, StepOutcome,
};

use super::gpu_execution::gpu_or_skip;
use super::gpu_hierarchy_execution::{DeviceCycle, DeviceHierarchy, Phases};
use super::gpu_kernels::varied_vector;
use super::gpu_transfer::built_hierarchy;

/// A CPU cycle by reference, so a hierarchy can serve a solve and then the device's upload.
struct OnCpu<'a>(&'a Multigrid);

impl Preconditioner for OnCpu<'_> {
    fn apply(&self, residual: &[f64], output: &mut [f64]) {
        self.0.apply(residual, output);
    }

    fn system_multiply(&self, matrix: &CsrMatrix, x: &[f64], y: &mut [f64]) {
        self.0.system_multiply(matrix, x, y);
    }
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// `|| b - A x || / || b ||`, computed directly.
fn true_residual(csr: &CsrMatrix, rhs: &[f64], x: &[f64]) -> f64 {
    let mut residual = vec![0.0; rhs.len()];
    csr.multiply(x, &mut residual);
    for (r, b) in residual.iter_mut().zip(rhs) {
        *r = b - *r;
    }
    norm(&residual) / norm(rhs)
}

/// A finished solve with what it took.
struct Solved<P: Preconditioner> {
    pcg: Pcg<P>,
    iterations: u64,
    seconds: f64,
}

/// Runs a solve of `csr x = rhs` to the tolerance, which must be reached within `cap` iterations
/// (`None` is the solver's own cap): the `Converged` status recomputes the true residual, and a
/// stop at the rounding floor would not meet the tolerance.
fn solve<P: Preconditioner>(
    csr: &CsrMatrix,
    preconditioner: P,
    rhs: &[f64],
    tolerance: f64,
    cap: Option<u64>,
) -> Solved<P> {
    let options = PcgOptions {
        tolerance,
        max_iterations: cap,
    };
    let mut pcg = Pcg::new(csr.clone(), preconditioner, rhs.to_vec(), options);
    let clock = Instant::now();
    loop {
        let outcome = pcg
            .step(1000, &mut || false)
            .unwrap_or_else(|error| panic!("the solve, capped at {cap:?} iterations: {error}"));
        if let StepOutcome::Converged(progress) = outcome {
            assert!(
                !progress.at_rounding_floor,
                "the solve stopped at the rounding floor, not at the requested residual"
            );
            return Solved {
                iterations: progress.iteration,
                seconds: clock.elapsed().as_secs_f64(),
                pcg,
            };
        }
    }
}

/// The hierarchy options of the solves here: single precision, as the device runs.
fn options(threads: usize) -> MultigridOptions {
    MultigridOptions {
        precision: Precision::Single,
        threads,
        ..MultigridOptions::default()
    }
}

#[test]
fn a_gpu_preconditioned_solve_reaches_the_answer_of_the_cpu_solve() {
    let Some(gpu) = gpu_or_skip("the GPU preconditioned solve") else {
        return;
    };
    // 16^3 nodes, 12,288 unknowns.
    let small = MultigridOptions {
        coarse_dofs: 30,
        ..options(1)
    };
    let (csr, multigrid) = built_hierarchy(16, small);
    assert!(multigrid.stats().levels >= 3, "a multilevel hierarchy");
    let rhs = varied_vector(csr.size());
    let tolerance = 1e-9;
    let cpu = solve(&csr, OnCpu(&multigrid), &rhs, tolerance, None);
    let cpu_x = cpu.pcg.estimate().to_vec();

    let limit = gpu.device.limits().max_storage_buffer_binding_size;
    let device = DeviceHierarchy::upload(&gpu, &multigrid, limit, false)
        .expect("the hierarchy fits the adapter's binding");
    let (down, up) = device.commands(&gpu);

    // The trait's `apply`, which the solve reaches, is the device cycle of the rounded input with
    // the correction widened, to the bit: a cycle repeats to the bit, and rounding and widening
    // are exact functions of their inputs. The values are not all f32s, so the rounding is real.
    assert!(rhs.iter().any(|&v| f64::from(v as f32) != v));
    let cycle = DeviceCycle::new(&gpu, &device, &multigrid, &down, &up);
    let mut applied = vec![0.0; rhs.len()];
    cycle.apply(&rhs, &mut applied);
    let rounded: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
    let direct = cycle.apply_f32(&rounded);
    assert!(
        applied
            .iter()
            .zip(&direct)
            .all(|(a, d)| a.to_bits() == f64::from(*d).to_bits()),
        "{}: Preconditioner::apply is not the widened device cycle of the rounded input",
        gpu.name
    );

    // The device's cycle differs from the CPU's only by rounding. If its relative difference in
    // the energy norm is at most `d`, the preconditioned spectrum moves by at most a factor
    // `1 +- d`, so the condition number grows by at most `(1 + d) / (1 - d)` and the conjugate
    // gradient's iteration estimate, which goes as its square root, by at most that root. The cap
    // allows `d = 3/5`, a root of exactly 2, far above the rounding the cycle comparisons measure
    // (about 3e-7 of the largest entry), plus one iteration to round up and one for the restart
    // after a drifted recursive residual. A cycle that is wrong stops here with the solver's error
    // instead of running to the test timeout.
    let cap = 2 * cpu.iterations + 2;
    let gpu_solve = solve(
        &csr,
        DeviceCycle::new(&gpu, &device, &multigrid, &down, &up),
        &rhs,
        tolerance,
        Some(cap),
    );
    let gpu_x = gpu_solve.pcg.estimate();

    let gpu_residual = true_residual(&csr, &rhs, gpu_x);
    let cpu_residual = true_residual(&csr, &rhs, &cpu_x);
    assert!(gpu_residual <= tolerance, "GPU residual {gpu_residual}");
    assert!(cpu_residual <= tolerance, "CPU residual {cpu_residual}");
    // Derived from the requested residual: `gpu - cpu = A^-1 (r_cpu - r_gpu)` for the two true
    // residuals, so `||gpu - cpu|| <= (||r_gpu|| + ||r_cpu||) / lambda_min(A)`, and the lattice is
    // a sum of `G^T G` terms plus 0.05 on the diagonal, so `lambda_min(A) >= 0.05`.
    let difference: Vec<f64> = gpu_x.iter().zip(&cpu_x).map(|(a, b)| a - b).collect();
    let bound = (gpu_residual + cpu_residual) * norm(&rhs) / 0.05;
    assert!(
        norm(&difference) <= bound,
        "{}: the GPU and CPU answers differ by {} where the residuals allow {bound}",
        gpu.name,
        norm(&difference)
    );
    println!(
        "{}: {} levels; iterations {} on the GPU, {} on the CPU; residual {gpu_residual:.3e} GPU, \
         {cpu_residual:.3e} CPU; answers differ by {:.3e}, allowed {bound:.3e}",
        gpu.name,
        multigrid.stats().levels,
        gpu_solve.iterations,
        cpu.iterations,
        norm(&difference)
    );
}

/// Prints one solve's phase times per cycle, in milliseconds.
fn report_phases(label: &str, phases: &Phases) {
    let per = 1000.0 / f64::from(phases.cycles.max(1));
    println!(
        "{label}: {} cycles, per cycle ms: rhs write + down dispatches {:.3}, coarse read-back \
         {:.3}, coarse solve {:.3}, coarse write + up dispatches {:.3}, fine read-back {:.3}, \
         f32/f64 rounding {:.3}",
        phases.cycles,
        phases.down * per,
        phases.coarse_read * per,
        phases.coarse_solve * per,
        phases.up * per,
        phases.fine_read * per,
        phases.rounding * per,
    );
}

fn timed_setup(n: usize, threads: usize) -> (CsrMatrix, Multigrid, f64) {
    let clock = Instant::now();
    let (csr, multigrid) = built_hierarchy(n, options(threads));
    (csr, multigrid, clock.elapsed().as_secs_f64())
}

/// Prints one solve's summary line and returns its true residual.
fn report_solve<P: Preconditioner>(
    label: &str,
    solved: &Solved<P>,
    csr: &CsrMatrix,
    rhs: &[f64],
) -> f64 {
    let residual = true_residual(csr, rhs, solved.pcg.estimate());
    println!(
        "{label}: {} iterations in {:.2} s ({:.1} ms per iteration), true residual {residual:.3e}",
        solved.iterations,
        solved.seconds,
        solved.seconds * 1000.0 / solved.iterations as f64
    );
    residual
}

#[test]
#[ignore = "a measurement: about a million unknowns, run in release with --ignored --nocapture"]
fn measure_a_gpu_solve_near_a_million_unknowns() {
    let Some(gpu) = gpu_or_skip("the GPU solve measurement") else {
        return;
    };
    // 70^3 = 343,000 block rows, 1,029,000 unknowns, the size the smoother was measured at.
    let n = 70;
    let (csr, eight, setup_eight) = timed_setup(n, 8);
    println!(
        "{}: {} unknowns, {} stored values; levels {:?}, setup at 8 threads {setup_eight:.1} s",
        gpu.name,
        csr.size(),
        csr.nonzeros(),
        eight.level_sizes()
    );
    let rhs = varied_vector(csr.size());
    let tolerance = 1e-9;

    let limits = gpu.device.limits();
    let limit = limits.max_storage_buffer_binding_size;
    println!(
        "adapter storage binding limit {limit} bytes, buffer limit {} bytes",
        limits.max_buffer_size
    );
    let finest = BlockMatrix::<f32>::from_level(&eight.level_matrix(0).expect("a finest level"))
        .expect("the finest level packs");
    let refused = finest.check_binding_size(GUARANTEED_BINDING_BYTES);
    assert!(
        matches!(refused, Err(GpuError::BindingTooLarge { .. })),
        "a million unknowns exceed the guaranteed 128 MiB binding: {refused:?}"
    );
    if let Err(error) = refused {
        println!("the guaranteed binding refuses it: {error}");
    }
    drop(finest);

    let clock = Instant::now();
    let device = DeviceHierarchy::upload(&gpu, &eight, limit, false)
        .expect("the adapter's own limits hold the hierarchy");
    let upload = clock.elapsed().as_secs_f64();
    let clock = Instant::now();
    let (down, up) = device.commands(&gpu);
    println!(
        "upload (reading levels, packing bytes, creating buffers) {upload:.3} s; recording bind \
         groups {:.3} s",
        clock.elapsed().as_secs_f64()
    );

    let (first_smoothers, first_transfers) = device.dispatch_seconds(&gpu, 1);
    println!(
        "first submissions after upload (they also wait for the upload's copies): smoothers \
         {:.2} ms, transfers {:.2} ms",
        first_smoothers * 1000.0,
        first_transfers * 1000.0
    );
    let (smoothers, transfers) = device.dispatch_seconds(&gpu, 20);
    println!(
        "dispatches alone per cycle, resident, {} smoothed levels: smoothers {:.2} ms, transfers \
         {:.2} ms",
        device.cycle_levels.len(),
        smoothers * 1000.0,
        transfers * 1000.0
    );

    let eight_gpu = solve(
        &csr,
        DeviceCycle::new(&gpu, &device, &eight, &down, &up),
        &rhs,
        tolerance,
        None,
    );
    report_solve(
        "GPU cycle, CG product on 8 CPU threads",
        &eight_gpu,
        &csr,
        &rhs,
    );
    report_phases(
        "GPU cycle within that solve",
        &eight_gpu.pcg.preconditioner().phases(),
    );
    let cg_iterations = eight_gpu.iterations;
    drop(eight_gpu);

    let eight_cpu = solve(&csr, OnCpu(&eight), &rhs, tolerance, None);
    report_solve("CPU Single cycle, 8 threads", &eight_cpu, &csr, &rhs);
    drop(eight_cpu);
    drop(eight);

    let (_, one, setup_one) = timed_setup(n, 1);
    println!("setup at 1 thread {setup_one:.1} s");
    let one_gpu = solve(
        &csr,
        DeviceCycle::new(&gpu, &device, &one, &down, &up),
        &rhs,
        tolerance,
        None,
    );
    report_solve(
        "GPU cycle, CG product on 1 CPU thread",
        &one_gpu,
        &csr,
        &rhs,
    );
    assert_eq!(
        one_gpu.iterations, cg_iterations,
        "the same device cycle repeats its bits, so the same iterations"
    );
    drop(one_gpu);
    let one_cpu = solve(&csr, OnCpu(&one), &rhs, tolerance, None);
    report_solve("CPU Single cycle, 1 thread", &one_cpu, &csr, &rhs);
}
