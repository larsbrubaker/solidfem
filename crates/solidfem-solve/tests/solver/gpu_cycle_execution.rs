// The V-cycle transfer kernels and one composed two-level cycle on this machine's GPU through wgpu
// (native only), on two-level hierarchies the multigrid builds from the spring lattice, whose
// fine blocks are not symmetric. Each kernel is compared with the f32 CPU reference on the same
// inputs; the composed cycle (pre-smooth, residual, restrict, the coarse solve on the CPU from a
// readback, prolong and correct, residual, post-smooth, add) is checked stage by stage against
// the reference applied to the device's own input to that stage, and must repeat to the bit. The
// kernels also run on level 1 of a three-level hierarchy, whose matrix and prolongator are 6x6
// blocks, and on a record that counts fewer rows than the dispatch starts threads for, which must
// leave the output past the count unchanged.
//
// Tolerances, all derived and never fitted: a row sum of `k` products in f32 is within
// `k * u * sum |a_ij x_j|` of the exact sum (`u` the f32 unit roundoff, here f32::EPSILON, an
// upper bound on it), the bound `reference::multiply_rounding_bound` computes; one more addition
// or subtraction adds at most `u` times its result, bounded to first order by
// `u * (|c| + sum |a_ij x_j|)` for the added value `c`. A device and the reference are each
// within that bound of the exact result on the same f32 inputs, so they differ by at most twice
// it; a device that contracts to fused multiply-adds rounds less, never more. A single f32
// addition is correctly rounded in WGSL, so `add_main` must match the reference to the bit.
//
// A missing adapter skips, or fails under SOLIDFEM_REQUIRE_GPU=1, as in `gpu_execution.rs`.

use std::time::Instant;

use solidfem_solve::gpu::reference::{self, CycleLevel, SmootherVectors};
use solidfem_solve::gpu::{
    dispatch_size, f32_le_bytes, BlockMatrix, GpuParams, SMOOTHER_WGSL, TRANSFER_WGSL,
};
use solidfem_solve::{Multigrid, MultigridOptions, Precision};

use super::gpu_execution::{bind, gpu_or_skip, pipeline, read, storage, uniform, Gpu};
use super::gpu_kernels::{spring_lattice, varied_vector};
use super::gpu_transfer::{built_hierarchy, coarse_solve, cycle_level, two_level_hierarchy};

const UNIT: f64 = f32::EPSILON as f64;

/// A matrix resident on the device with its uniform record.
pub(super) struct Resident {
    starts: wgpu::Buffer,
    columns: wgpu::Buffer,
    values: wgpu::Buffer,
    params: wgpu::Buffer,
    threads: u32,
}

pub(super) fn resident(gpu: &Gpu, matrix: &BlockMatrix<f32>) -> Resident {
    Resident {
        starts: storage(gpu, "starts", &matrix.starts_le_bytes(), false),
        columns: storage(gpu, "columns", &matrix.columns_le_bytes(), false),
        values: storage(gpu, "values", &matrix.values_le_bytes(), false),
        params: uniform(gpu, &GpuParams::for_matrix(matrix, false)),
        // Fits u32: the layout refuses 2^32 scalar rows.
        threads: (matrix.rows() * matrix.row_block()) as u32,
    }
}

/// A vector the device writes and the host reads and writes.
pub(super) fn vector(gpu: &Gpu, label: &str, values: &[f32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    gpu.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: &f32_le_bytes(values),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
}

pub(super) fn read_vector(gpu: &Gpu, buffer: &wgpu::Buffer) -> Vec<f32> {
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    read(gpu, buffer, &readback)
}

/// One recorded command: a dispatch over `threads` threads, or a copy between equal buffers.
pub(super) enum Command<'a> {
    Dispatch(&'a wgpu::ComputePipeline, wgpu::BindGroup, u32),
    Copy(&'a wgpu::Buffer, &'a wgpu::Buffer),
}

/// Records `commands` in one submission, in order, and waits for them.
pub(super) fn submit(gpu: &Gpu, commands: &[Command<'_>]) {
    submit_repeated(gpu, commands, 1);
}

/// Records `commands` `repeats` times in one submission and waits for them, so a timing of the
/// dispatches alone does not count one wait per repeat.
pub(super) fn submit_repeated(gpu: &Gpu, commands: &[Command<'_>], repeats: usize) {
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    for command in std::iter::repeat_n(commands, repeats).flatten() {
        match command {
            Command::Dispatch(pipeline, group, threads) => {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                let [x, y] = dispatch_size(*threads);
                pass.dispatch_workgroups(x, y, 1);
            }
            Command::Copy(from, to) => encoder.copy_buffer_to_buffer(from, 0, to, 0, from.size()),
        }
    }
    gpu.queue.submit([encoder.finish()]);
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device finishes");
}

/// The pipelines of both shaders.
pub(super) struct Pipelines {
    start: wgpu::ComputePipeline,
    multiply: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    pub(super) residual: wgpu::ComputePipeline,
    pub(super) restrict: wgpu::ComputePipeline,
    pub(super) prolong: wgpu::ComputePipeline,
    pub(super) add: wgpu::ComputePipeline,
}

pub(super) fn pipelines(gpu: &Gpu) -> Pipelines {
    let module = |source: &str| {
        gpu.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
    };
    let (smoother, transfer) = (module(SMOOTHER_WGSL), module(TRANSFER_WGSL));
    Pipelines {
        start: pipeline(gpu, &smoother, "chebyshev_start_main"),
        multiply: pipeline(gpu, &smoother, "multiply_main"),
        update: pipeline(gpu, &smoother, "chebyshev_update_main"),
        residual: pipeline(gpu, &transfer, "residual_main"),
        restrict: pipeline(gpu, &transfer, "restrict_main"),
        prolong: pipeline(gpu, &transfer, "prolong_correct_main"),
        add: pipeline(gpu, &transfer, "add_main"),
    }
}

/// A transfer kernel's dispatch over `matrix`'s scalar rows with its uniform, its arrays bound
/// when `with_matrix` (`add_main` reads only the uniform), and `input`, `base` and `output`.
pub(super) fn transfer<'a>(
    gpu: &Gpu,
    pipeline: &'a wgpu::ComputePipeline,
    matrix: &Resident,
    input: &wgpu::Buffer,
    base: Option<&wgpu::Buffer>,
    output: &wgpu::Buffer,
    with_matrix: bool,
) -> Command<'a> {
    let mut entries = vec![(0, &matrix.params), (4, input), (6, output)];
    if with_matrix {
        entries.extend([
            (1, &matrix.starts),
            (2, &matrix.columns),
            (3, &matrix.values),
        ]);
    }
    if let Some(base) = base {
        entries.push((5, base));
    }
    Command::Dispatch(pipeline, bind(gpu, pipeline, &entries), matrix.threads)
}

/// The smoother's dispatches from a zero guess on `rhs`, writing `solution`.
#[allow(clippy::too_many_arguments)] // The smoother's buffers; a struct would only rename them.
pub(super) fn smoother<'a>(
    gpu: &Gpu,
    p: &'a Pipelines,
    level: &CycleLevel<f32>,
    a: &Resident,
    inverse: &wgpu::Buffer,
    work: &(wgpu::Buffer, wgpu::Buffer),
    rhs: &wgpu::Buffer,
    solution: &wgpu::Buffer,
) -> Vec<Command<'a>> {
    let (residual, direction) = work;
    let blocks = level.matrix.rows() as u32;
    let start = uniform(gpu, &level.coefficients.start_params(&level.matrix));
    let mut commands = vec![Command::Dispatch(
        &p.start,
        bind(
            gpu,
            &p.start,
            &[
                (0, &start),
                (4, inverse),
                (5, rhs),
                (6, residual),
                (7, direction),
                (8, solution),
            ],
        ),
        blocks,
    )];
    for step in 0..level.coefficients.steps.len() {
        let params = level.coefficients.step_params(&level.matrix, step);
        let params = uniform(gpu, &params.expect("a step below steps.len()"));
        let multiply = [
            (0, &params),
            (1, &a.starts),
            (2, &a.columns),
            (3, &a.values),
            (6, residual),
            (7, direction),
        ];
        let update = [
            (0, &params),
            (4, inverse),
            (6, residual),
            (7, direction),
            (8, solution),
        ];
        commands.push(Command::Dispatch(
            &p.multiply,
            bind(gpu, &p.multiply, &multiply),
            a.threads,
        ));
        commands.push(Command::Dispatch(
            &p.update,
            bind(gpu, &p.update, &update),
            blocks,
        ));
    }
    commands
}

/// Per row, `sum |a_ij x_j|`.
fn abs_product(matrix: &BlockMatrix<f32>, x: &[f32]) -> Vec<f64> {
    let wide = matrix.convert::<f64>();
    let abs: Vec<f64> = wide.values().iter().map(|v| v.abs()).collect();
    let (rb, cb) = (wide.row_block(), wide.col_block());
    let mut sums = vec![0.0; wide.rows() * rb];
    for (row, sum) in sums.iter_mut().enumerate() {
        let block_row = row / rb;
        let p = row % rb;
        for at in wide.starts()[block_row] as usize..wide.starts()[block_row + 1] as usize {
            let column = wide.columns()[at] as usize * cb;
            for q in 0..cb {
                *sum += abs[at * rb * cb + p * cb + q] * f64::from(x[column + q].abs());
            }
        }
    }
    sums
}

/// The bound of a product followed by one addition of `added` (zero for a plain product).
pub(super) fn product_bound(
    matrix: &BlockMatrix<f32>,
    x: &[f32],
    added: Option<&[f32]>,
) -> Vec<f64> {
    let product = reference::multiply_rounding_bound(matrix, x);
    let abs = abs_product(matrix, x);
    (0..product.len())
        .map(|i| match added {
            Some(c) => product[i] + UNIT * (f64::from(c[i].abs()) + abs[i]),
            None => product[i],
        })
        .collect()
}

/// Asserts the device's `found` is within twice `bound` of the reference's `expected`, returning
/// how many entries are identical.
pub(super) fn within(
    gpu: &Gpu,
    what: &str,
    found: &[f32],
    expected: &[f32],
    bound: &[f64],
) -> usize {
    assert_eq!(found.len(), expected.len(), "{what}");
    let mut identical = 0;
    for (i, ((device, cpu), bound)) in found.iter().zip(expected).zip(bound).enumerate() {
        let difference = (f64::from(*device) - f64::from(*cpu)).abs();
        assert!(
            difference <= 2.0 * bound,
            "{}: {what} entry {i} is {device} on the GPU and {cpu} on the CPU, bound {}",
            gpu.name,
            2.0 * bound
        );
        identical += usize::from(device.to_bits() == cpu.to_bits());
    }
    identical
}

fn hierarchy(n: usize) -> (Multigrid, CycleLevel<f32>) {
    let (_, multigrid) = two_level_hierarchy(n, Precision::Single);
    let level = cycle_level::<f32>(&multigrid, 0);
    (multigrid, level)
}

pub(super) fn varied_f32(length: usize, shift: usize) -> Vec<f32> {
    varied_vector(length + shift)[shift..]
        .iter()
        .map(|&v| v as f32)
        .collect()
}

/// Runs every transfer kernel of `level` on the device and compares each with the reference.
fn compare_kernels(gpu: &Gpu, label: &str, level: &CycleLevel<f32>) {
    let p = pipelines(gpu);
    let fine = level.matrix.rows() * level.matrix.row_block();
    let coarse = level.prolongator.block_columns() * level.prolongator.col_block();
    let (x, b, e) = (
        varied_f32(fine, 0),
        varied_f32(fine, 7),
        varied_f32(coarse, 3),
    );
    let a = resident(gpu, &level.matrix);
    let prolongator = resident(gpu, &level.prolongator);
    let restriction = resident(gpu, &level.restriction);
    let (x_buffer, b_buffer, e_buffer) = (
        vector(gpu, "x", &x),
        vector(gpu, "b", &b),
        vector(gpu, "e", &e),
    );
    // Filled with a value no kernel gives, so an unwritten row shows.
    let r_buffer = vector(gpu, "r", &vec![1.0e30; fine]);
    let c_buffer = vector(gpu, "coarse", &vec![1.0e30; coarse]);
    let corrected = vector(gpu, "corrected", &x);
    let summed = vector(gpu, "summed", &x);
    submit(
        gpu,
        &[
            transfer(
                gpu,
                &p.residual,
                &a,
                &x_buffer,
                Some(&b_buffer),
                &r_buffer,
                true,
            ),
            transfer(
                gpu,
                &p.restrict,
                &restriction,
                &x_buffer,
                None,
                &c_buffer,
                true,
            ),
            transfer(
                gpu,
                &p.prolong,
                &prolongator,
                &e_buffer,
                None,
                &corrected,
                true,
            ),
            transfer(gpu, &p.add, &a, &b_buffer, None, &summed, false),
        ],
    );

    let mut expected = vec![0.0f32; fine];
    reference::residual(&level.matrix, &b, &x, &mut expected);
    let bound = product_bound(&level.matrix, &x, Some(&b));
    let residual_same = within(
        gpu,
        "residual",
        &read_vector(gpu, &r_buffer),
        &expected,
        &bound,
    );

    let mut expected = vec![0.0f32; coarse];
    reference::restrict(&level.restriction, &x, &mut expected);
    let bound = product_bound(&level.restriction, &x, None);
    let restrict_same = within(
        gpu,
        "restrict",
        &read_vector(gpu, &c_buffer),
        &expected,
        &bound,
    );

    let mut expected = x.clone();
    reference::prolong_correct(&level.prolongator, &e, &mut expected);
    let bound = product_bound(&level.prolongator, &e, Some(&x));
    let prolong_same = within(
        gpu,
        "prolong-correct",
        &read_vector(gpu, &corrected),
        &expected,
        &bound,
    );

    let mut expected = x.clone();
    reference::add(&mut expected, &b);
    assert_eq!(read_vector(gpu, &summed), expected, "{}", gpu.name);
    println!(
        "{}: {label}, {fine} fine and {coarse} coarse values; identical to the CPU: \
         residual {residual_same}, restrict {restrict_same}, prolong-correct {prolong_same}",
        gpu.name
    );
}

#[test]
fn the_transfer_kernels_run_on_the_gpu_as_the_cpu_reference() {
    let Some(gpu) = gpu_or_skip("the GPU transfer kernels") else {
        return;
    };
    for n in [8, 12] {
        let (_, level) = hierarchy(n);
        compare_kernels(&gpu, &format!("{n}^3 lattice, finest level"), &level);
    }
}

#[test]
fn the_transfer_kernels_read_six_by_six_blocks_on_the_gpu_as_the_cpu_reference() {
    let Some(gpu) = gpu_or_skip("the GPU transfer kernels on a coarse level") else {
        return;
    };
    let options = MultigridOptions {
        max_levels: 3,
        coarse_dofs: 30,
        precision: Precision::Single,
        ..MultigridOptions::default()
    };
    let (_, multigrid) = built_hierarchy(12, options);
    assert_eq!(multigrid.stats().levels, 3);
    let shape = multigrid
        .level_prolongator(1)
        .expect("level 1 is not the coarsest");
    assert_eq!((shape.row_block, shape.col_block), (6, 6));
    let level = cycle_level::<f32>(&multigrid, 1);
    assert_eq!((level.matrix.row_block(), level.matrix.col_block()), (6, 6));
    assert_eq!(
        (level.restriction.row_block(), level.restriction.col_block()),
        (6, 6)
    );
    // A transposed read of a block shows only where a block is not symmetric.
    let unsymmetric = level
        .prolongator
        .values()
        .chunks(36)
        .any(|block| (0..6).any(|p| (0..6).any(|q| block[6 * p + q] != block[6 * q + p])));
    assert!(unsymmetric, "the 6x6 prolongator has an unsymmetric block");
    compare_kernels(&gpu, "12^3 lattice, level 1 of 3", &level);
}

/// One guard case: the kernel's name, pipeline, whether it reads a matrix, its `base` and `input`
/// buffers, the expected output and the output's rounding bound.
type GuardCase<'a> = (
    &'a str,
    &'a wgpu::ComputePipeline,
    bool,
    Option<&'a wgpu::Buffer>,
    &'a wgpu::Buffer,
    Vec<f32>,
    Vec<f64>,
);

#[test]
fn threads_past_the_row_count_leave_the_rest_of_the_output_alone() {
    let Some(gpu) = gpu_or_skip("the GPU kernels' row guards") else {
        return;
    };
    let p = pipelines(&gpu);
    // 64 block rows, 192 values, three whole workgroups. The record counts one block row fewer,
    // so the dispatch still starts threads for the last three values: real matrix rows and real
    // vector entries that a kernel without its guard would compute and write.
    let matrix = BlockMatrix::<f32>::from_nodal(&spring_lattice(4)).expect("nodal blocks");
    let full = matrix.rows() * 3;
    let counted = full - 3;
    let bound_matrix = Resident {
        params: uniform(
            &gpu,
            &GpuParams {
                // Fits u32: 63.
                rows: (matrix.rows() - 1) as u32,
                ..GpuParams::for_matrix(&matrix, false)
            },
        ),
        // Fits u32: 189.
        threads: counted as u32,
        ..resident(&gpu, &matrix)
    };
    assert_eq!(dispatch_size(189), [3, 1], "the dispatch rounds up to 192");
    let (x, b) = (varied_f32(full, 0), varied_f32(full, 7));
    let (x_buffer, b_buffer) = (vector(&gpu, "x", &x), vector(&gpu, "b", &b));
    // Not a value any kernel gives, and small enough that an added term would change it.
    const SENTINEL: f32 = -12345.5;
    let mut seeded = x[..counted].to_vec();
    seeded.resize(full, SENTINEL);

    let mut residual = vec![0.0; full];
    reference::residual(&matrix, &b, &x, &mut residual);
    let mut restricted = vec![0.0; full];
    reference::restrict(&matrix, &x, &mut restricted);
    let mut corrected = x.clone();
    reference::prolong_correct(&matrix, &x, &mut corrected);
    let mut summed = x.clone();
    reference::add(&mut summed, &b);
    let cases: [GuardCase<'_>; 4] = [
        (
            "residual",
            &p.residual,
            true,
            Some(&b_buffer),
            &x_buffer,
            residual,
            product_bound(&matrix, &x, Some(&b)),
        ),
        (
            "restrict",
            &p.restrict,
            true,
            None,
            &x_buffer,
            restricted,
            product_bound(&matrix, &x, None),
        ),
        (
            "prolong-correct",
            &p.prolong,
            true,
            None,
            &x_buffer,
            corrected,
            product_bound(&matrix, &x, Some(&x)),
        ),
        // One correctly rounded addition, so no difference is allowed.
        (
            "add",
            &p.add,
            false,
            None,
            &b_buffer,
            summed,
            vec![0.0; full],
        ),
    ];
    for (name, kernel, with_matrix, base, input, expected, bound) in cases {
        let output = vector(&gpu, "output", &seeded);
        let command = transfer(
            &gpu,
            kernel,
            &bound_matrix,
            input,
            base,
            &output,
            with_matrix,
        );
        submit(&gpu, &[command]);
        let found = read_vector(&gpu, &output);
        within(
            &gpu,
            name,
            &found[..counted],
            &expected[..counted],
            &bound[..counted],
        );
        assert_eq!(found[counted..], [SENTINEL; 3], "{}: {name}", gpu.name);
    }
}

#[test]
fn one_two_level_cycle_runs_on_the_gpu_stage_by_stage_as_the_cpu_reference() {
    let Some(gpu) = gpu_or_skip("the GPU two-level cycle") else {
        return;
    };
    let (multigrid, level) = hierarchy(12);
    let p = pipelines(&gpu);
    let fine = level.matrix.rows() * 3;
    let coarse = level.prolongator.block_columns() * level.prolongator.col_block();
    let b = varied_f32(fine, 0);
    let a = resident(&gpu, &level.matrix);
    let prolongator = resident(&gpu, &level.prolongator);
    let restriction = resident(&gpu, &level.restriction);
    let inverse = storage(
        &gpu,
        "inverse",
        &f32_le_bytes(&level.inverse_diagonal),
        false,
    );
    let zeros = vec![0.0f32; fine];
    let work = (
        vector(&gpu, "work r", &zeros),
        vector(&gpu, "work d", &zeros),
    );
    let [b_buffer, x, r, post] = ["b", "x", "r", "post"].map(|label| vector(&gpu, label, &zeros));
    gpu.queue.write_buffer(&b_buffer, 0, &f32_le_bytes(&b));
    let [x_pre, r_pre, x_corrected, r_corrected] =
        ["x pre", "r pre", "x corrected", "r corrected"].map(|l| vector(&gpu, l, &zeros));
    let coarse_rhs = vector(&gpu, "coarse rhs", &vec![0.0; coarse]);
    let coarse_x = vector(&gpu, "coarse x", &vec![0.0; coarse]);

    let mut down = smoother(&gpu, &p, &level, &a, &inverse, &work, &b_buffer, &x);
    down.push(Command::Copy(&x, &x_pre));
    down.push(transfer(
        &gpu,
        &p.residual,
        &a,
        &x,
        Some(&b_buffer),
        &r,
        true,
    ));
    down.push(Command::Copy(&r, &r_pre));
    down.push(transfer(
        &gpu,
        &p.restrict,
        &restriction,
        &r,
        None,
        &coarse_rhs,
        true,
    ));
    let mut up = vec![
        transfer(&gpu, &p.prolong, &prolongator, &coarse_x, None, &x, true),
        Command::Copy(&x, &x_corrected),
        transfer(&gpu, &p.residual, &a, &x, Some(&b_buffer), &r, true),
        Command::Copy(&r, &r_corrected),
    ];
    up.extend(smoother(&gpu, &p, &level, &a, &inverse, &work, &r, &post));
    up.push(transfer(&gpu, &p.add, &a, &post, None, &x, false));

    let run = || {
        submit(&gpu, &down);
        let device_rhs = read_vector(&gpu, &coarse_rhs);
        let mut solved = vec![0.0f32; coarse];
        coarse_solve(&multigrid, &device_rhs, &mut solved);
        gpu.queue.write_buffer(&coarse_x, 0, &f32_le_bytes(&solved));
        submit(&gpu, &up);
        (device_rhs, solved, read_vector(&gpu, &x))
    };
    let (device_rhs, solved, first) = run();
    let clock = Instant::now();
    let (_, _, second) = run();
    let elapsed = clock.elapsed();
    // One thread owns every sum and the coarse solve is the same CPU arithmetic, so a second
    // cycle repeats the first to the bit.
    assert_eq!(second, first, "{}", gpu.name);

    // Each stage against the reference on the device's own input to it.
    let smooth = |rhs: &[f32]| {
        let mut vectors = SmootherVectors::<f32>::zeros(fine);
        let m = &level.matrix;
        reference::chebyshev_smooth(
            m,
            &level.inverse_diagonal,
            &level.coefficients,
            rhs,
            &mut vectors,
        );
        let bound = reference::smoother_rounding_bound(
            m,
            &level.inverse_diagonal,
            &level.coefficients,
            rhs,
        );
        (vectors.solution, bound)
    };
    let x_pre = read_vector(&gpu, &x_pre);
    let (expected, bound) = smooth(&b);
    within(&gpu, "pre-smooth", &x_pre, &expected, &bound);
    let r_pre = read_vector(&gpu, &r_pre);
    let mut expected = vec![0.0; fine];
    reference::residual(&level.matrix, &b, &x_pre, &mut expected);
    let bound = product_bound(&level.matrix, &x_pre, Some(&b));
    within(&gpu, "residual", &r_pre, &expected, &bound);
    let mut expected = vec![0.0; coarse];
    reference::restrict(&level.restriction, &r_pre, &mut expected);
    let bound = product_bound(&level.restriction, &r_pre, None);
    within(&gpu, "restrict", &device_rhs, &expected, &bound);
    let x_corrected = read_vector(&gpu, &x_corrected);
    let mut expected = x_pre.clone();
    reference::prolong_correct(&level.prolongator, &solved, &mut expected);
    let bound = product_bound(&level.prolongator, &solved, Some(&x_pre));
    within(&gpu, "prolong-correct", &x_corrected, &expected, &bound);
    let r_corrected = read_vector(&gpu, &r_corrected);
    let mut expected = vec![0.0; fine];
    reference::residual(&level.matrix, &b, &x_corrected, &mut expected);
    let bound = product_bound(&level.matrix, &x_corrected, Some(&b));
    within(&gpu, "second residual", &r_corrected, &expected, &bound);
    let post = read_vector(&gpu, &post);
    let (expected, bound) = smooth(&r_corrected);
    within(&gpu, "post-smooth", &post, &expected, &bound);
    let mut expected = x_corrected.clone();
    reference::add(&mut expected, &post);
    assert_eq!(first, expected, "{}: the final sum", gpu.name);

    // The whole cycle against the reference from the same right-hand side: measured, not
    // asserted, because the coarse solve amplifies the stages' rounding by the coarse matrix's
    // conditioning, which no bound here carries.
    let whole = reference::two_level_cycle(&level, &b, &mut |rhs, out| {
        coarse_solve(&multigrid, rhs, out)
    });
    let largest = whole.iter().fold(0.0f64, |m, v| m.max(f64::from(v.abs())));
    let difference = first.iter().zip(&whole).fold(0.0f64, |m, (d, c)| {
        m.max((f64::from(*d) - f64::from(*c)).abs())
    });
    println!(
        "{}: two-level cycle on {fine} fine and {coarse} coarse values in {elapsed:?} with all \
         operators resident (two submissions, one readback and one upload of the coarse \
         vector); largest difference from the CPU cycle {:.3e} of the largest entry",
        gpu.name,
        difference / largest
    );
}
