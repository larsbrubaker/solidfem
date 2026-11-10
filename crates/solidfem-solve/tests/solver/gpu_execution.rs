// The GPU smoother kernels run on this machine's GPU through wgpu (native only) and are compared
// with the f32 CPU reference within its derived rounding bound: the smoother on a small and a
// medium lattice with unsymmetric blocks, and the plain product on rectangular 6x3 blocks.
// `measure_one_smoother_*` (ignored by default, run with `--ignored`) times one application at a
// million unknowns.
//
// A machine without a GPU adapter, or whose adapter gives no device, skips each test with a
// printed `SKIPPED:` line, which `.config/nextest.toml` shows for passing tests here. Set
// `SOLIDFEM_REQUIRE_GPU=1` to make a missing adapter or device a failure instead, so a machine
// that must run these kernels cannot pass without running them.
#![cfg(not(target_arch = "wasm32"))]

use std::time::Instant;

use solidfem_solve::gpu::reference::{self, SmootherVectors};
use solidfem_solve::gpu::{
    dispatch_size, f32_from_le_bytes, f32_le_bytes, BlockMatrix, ChebyshevCoefficients, GpuParams,
    SMOOTHER_WGSL,
};
use wgpu::util::DeviceExt;

use super::gpu_kernels::{coefficients, six_by_three, spring_lattice, varied_vector};

/// The environment variable that turns a skipped GPU test into a failure.
const REQUIRE_GPU: &str = "SOLIDFEM_REQUIRE_GPU";

pub(super) struct Gpu {
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) name: String,
}

/// The default adapter with its own limits, or the reason there is none.
fn gpu() -> Result<Gpu, String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .map_err(|e| format!("no GPU adapter: {e}"))?;
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("smoother test"),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .map_err(|e| format!("the adapter {} gave no device: {e}", info.name))?;
    Ok(Gpu {
        device,
        queue,
        name: format!("{} ({:?})", info.name, info.backend),
    })
}

/// The device, or `None` after printing why `test` is skipped; fails when [`REQUIRE_GPU`] is 1.
pub(super) fn gpu_or_skip(test: &str) -> Option<Gpu> {
    match gpu() {
        Ok(gpu) => Some(gpu),
        Err(reason) => {
            let required = std::env::var(REQUIRE_GPU).is_ok_and(|value| value == "1");
            assert!(
                !required,
                "{test} needs a GPU and {REQUIRE_GPU}=1: {reason}"
            );
            println!("SKIPPED: {test} needs a GPU: {reason}");
            None
        }
    }
}

pub(super) fn shader(gpu: &Gpu) -> wgpu::ShaderModule {
    gpu.device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smoother"),
            source: wgpu::ShaderSource::Wgsl(SMOOTHER_WGSL.into()),
        })
}

pub(super) fn pipeline(
    gpu: &Gpu,
    module: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    gpu.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
}

/// One smoother problem resident on the device.
struct Resident {
    pipelines: [wgpu::ComputePipeline; 3],
    start: wgpu::BindGroup,
    steps: Vec<(wgpu::BindGroup, wgpu::BindGroup)>,
    solution: wgpu::Buffer,
    readback: wgpu::Buffer,
    rows: u32,
    dofs: u32,
}

pub(super) fn storage(gpu: &Gpu, label: &str, bytes: &[u8], writable: bool) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE;
    if writable {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    gpu.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage,
        })
}

pub(super) fn uniform(gpu: &Gpu, params: &GpuParams) -> wgpu::Buffer {
    gpu.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: &params.to_le_bytes(),
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

pub(super) fn bind(
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    entries: &[(u32, &wgpu::Buffer)],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = entries
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

fn upload(
    gpu: &Gpu,
    matrix: &BlockMatrix<f32>,
    diagonal: &[f32],
    coefficients: &ChebyshevCoefficients,
    rhs: &[f32],
) -> Resident {
    let module = shader(gpu);
    let pipelines = [
        pipeline(gpu, &module, "chebyshev_start_main"),
        pipeline(gpu, &module, "multiply_main"),
        pipeline(gpu, &module, "chebyshev_update_main"),
    ];
    let starts = storage(gpu, "starts", &matrix.starts_le_bytes(), false);
    let columns = storage(gpu, "columns", &matrix.columns_le_bytes(), false);
    let values = storage(gpu, "values", &matrix.values_le_bytes(), false);
    let inverse = storage(gpu, "inverse diagonal", &f32_le_bytes(diagonal), false);
    let rhs_buffer = storage(gpu, "rhs", &f32_le_bytes(rhs), false);
    let zeros = vec![0u8; rhs.len() * 4];
    let residual = storage(gpu, "residual", &zeros, true);
    let direction = storage(gpu, "direction", &zeros, true);
    let solution = storage(gpu, "solution", &zeros, true);
    let start_params = uniform(gpu, &coefficients.start_params(matrix));
    let start = bind(
        gpu,
        &pipelines[0],
        &[
            (0, &start_params),
            (4, &inverse),
            (5, &rhs_buffer),
            (6, &residual),
            (7, &direction),
            (8, &solution),
        ],
    );
    let steps = (0..coefficients.steps.len())
        .map(|step| {
            let step_params = coefficients.step_params(matrix, step);
            let params = uniform(gpu, &step_params.expect("a step below steps.len()"));
            let multiply = bind(
                gpu,
                &pipelines[1],
                &[
                    (0, &params),
                    (1, &starts),
                    (2, &columns),
                    (3, &values),
                    (6, &residual),
                    (7, &direction),
                ],
            );
            let update = bind(
                gpu,
                &pipelines[2],
                &[
                    (0, &params),
                    (4, &inverse),
                    (6, &residual),
                    (7, &direction),
                    (8, &solution),
                ],
            );
            (multiply, update)
        })
        .collect();
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: zeros.len() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    Resident {
        pipelines,
        start,
        steps,
        solution,
        readback,
        // Fit in u32: the layout refuses a matrix with 2^32 scalar rows.
        rows: matrix.rows() as u32,
        dofs: rhs.len() as u32,
    }
}

/// Records `applications` smoother applications in one submission and waits for them.
fn run(gpu: &Gpu, resident: &Resident, applications: usize) {
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        let [bx, by] = dispatch_size(resident.rows);
        let [sx, sy] = dispatch_size(resident.dofs);
        for _ in 0..applications {
            pass.set_pipeline(&resident.pipelines[0]);
            pass.set_bind_group(0, &resident.start, &[]);
            pass.dispatch_workgroups(bx, by, 1);
            for (multiply, update) in &resident.steps {
                pass.set_pipeline(&resident.pipelines[1]);
                pass.set_bind_group(0, multiply, &[]);
                pass.dispatch_workgroups(sx, sy, 1);
                pass.set_pipeline(&resident.pipelines[2]);
                pass.set_bind_group(0, update, &[]);
                pass.dispatch_workgroups(bx, by, 1);
            }
        }
    }
    gpu.queue.submit([encoder.finish()]);
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device finishes");
}

fn read_solution(gpu: &Gpu, resident: &Resident) -> Vec<f32> {
    read(gpu, &resident.solution, &resident.readback)
}

/// Copies `source` into the mappable `readback` and reads it as f32s.
pub(super) fn read(gpu: &Gpu, source: &wgpu::Buffer, readback: &wgpu::Buffer) -> Vec<f32> {
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(source, 0, readback, 0, readback.size());
    gpu.queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| {
        result.expect("the readback maps");
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device finishes");
    let values = f32_from_le_bytes(&slice.get_mapped_range());
    readback.unmap();
    values
}

struct Problem {
    matrix: BlockMatrix<f32>,
    diagonal: Vec<f32>,
    coefficients: ChebyshevCoefficients,
    rhs: Vec<f32>,
}

fn problem(n: usize) -> Problem {
    let wide = BlockMatrix::<f64>::from_nodal(&spring_lattice(n)).expect("nodal blocks");
    let diagonal = reference::inverse_diagonal(&wide).expect("positive definite");
    let rhs = varied_vector(wide.rows() * 3);
    Problem {
        matrix: wide.convert(),
        diagonal: diagonal.iter().map(|&v| v as f32).collect(),
        coefficients: coefficients(3),
        rhs: rhs.iter().map(|&v| v as f32).collect(),
    }
}

fn compare_on_gpu(n: usize) {
    let Some(gpu) = gpu_or_skip("the GPU smoother comparison") else {
        return;
    };
    let p = problem(n);
    let resident = upload(&gpu, &p.matrix, &p.diagonal, &p.coefficients, &p.rhs);
    run(&gpu, &resident, 1);
    let found = read_solution(&gpu, &resident);
    let mut cpu = SmootherVectors::<f32>::zeros(p.rhs.len());
    reference::chebyshev_smooth(&p.matrix, &p.diagonal, &p.coefficients, &p.rhs, &mut cpu);
    // Device and reference each sit within the rounding bound of the exact result on the same
    // f32 inputs, so they differ by at most twice it. A device that contracts to fused
    // multiply-adds rounds less, never more.
    let bound = reference::smoother_rounding_bound(&p.matrix, &p.diagonal, &p.coefficients, &p.rhs);
    let mut identical = 0usize;
    for (i, ((device, cpu), bound)) in found.iter().zip(&cpu.solution).zip(&bound).enumerate() {
        let difference = (f64::from(*device) - f64::from(*cpu)).abs();
        assert!(
            difference <= 2.0 * bound,
            "{}: entry {i} of {} block rows is {device} on the GPU and {cpu} on the CPU, bound {}",
            gpu.name,
            p.matrix.rows(),
            2.0 * bound
        );
        identical += usize::from(device.to_bits() == cpu.to_bits());
    }
    // A second run on the device repeats to the bit: one thread owns every sum.
    run(&gpu, &resident, 1);
    assert_eq!(read_solution(&gpu, &resident), found, "{}", gpu.name);
    println!(
        "{}: {} block rows, {identical} of {} entries identical to the CPU reference",
        gpu.name,
        p.matrix.rows(),
        found.len()
    );
}

#[test]
fn the_smoother_runs_on_the_gpu_as_the_cpu_reference_on_a_small_lattice() {
    compare_on_gpu(8);
}

#[test]
fn the_smoother_runs_on_the_gpu_as_the_cpu_reference_on_tens_of_thousands_of_rows() {
    // 32^3 = 32,768 block rows, 98,304 unknowns.
    compare_on_gpu(32);
}

#[test]
fn the_plain_product_runs_on_the_gpu_with_rectangular_blocks() {
    let Some(gpu) = gpu_or_skip("the GPU rectangular product") else {
        return;
    };
    let (starts, columns, values) = six_by_three();
    let matrix = BlockMatrix::<f32>::from_parts(3, 6, 3, &starts, &columns, &values).unwrap();
    let x: Vec<f32> = (0..9).map(|v| v as f32 * 0.5 - 1.75).collect();
    let module = shader(&gpu);
    let multiply = pipeline(&gpu, &module, "multiply_main");
    let params = uniform(&gpu, &GpuParams::for_matrix(&matrix, false));
    let starts_buffer = storage(&gpu, "starts", &matrix.starts_le_bytes(), false);
    let columns_buffer = storage(&gpu, "columns", &matrix.columns_le_bytes(), false);
    let values_buffer = storage(&gpu, "values", &matrix.values_le_bytes(), false);
    let input = storage(&gpu, "x", &f32_le_bytes(&x), true);
    // Filled with a value the product never gives, so an unwritten row shows.
    let output = storage(&gpu, "y", &f32_le_bytes(&[1.0e30; 12]), true);
    let group = bind(
        &gpu,
        &multiply,
        &[
            (0, &params),
            (1, &starts_buffer),
            (2, &columns_buffer),
            (3, &values_buffer),
            (6, &output),
            (7, &input),
        ],
    );
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&multiply);
        pass.set_bind_group(0, &group, &[]);
        let [groups_x, groups_y] = dispatch_size(12);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }
    gpu.queue.submit([encoder.finish()]);
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: 48,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let found = read(&gpu, &output, &readback);
    let mut expected = vec![0.0f32; 12];
    reference::multiply(&matrix, &x, &mut expected);
    // Each is within the product's rounding bound of the exact result on the same inputs.
    let bound = reference::multiply_rounding_bound(&matrix, &x);
    for (row, ((device, cpu), bound)) in found.iter().zip(&expected).zip(&bound).enumerate() {
        let difference = (f64::from(*device) - f64::from(*cpu)).abs();
        assert!(
            difference <= 2.0 * bound,
            "{}: row {row} is {device} on the GPU and {cpu} on the CPU, bound {}",
            gpu.name,
            2.0 * bound
        );
    }
    println!(
        "{}: the 6x3 product agrees with the CPU reference",
        gpu.name
    );
}

#[test]
#[ignore = "a measurement: about a million unknowns, run with --ignored"]
fn measure_one_smoother_application_at_a_million_unknowns() {
    let Some(gpu) = gpu_or_skip("the GPU measurement") else {
        return;
    };
    // 70^3 = 343,000 block rows, 1,029,000 unknowns.
    let p = problem(70);
    let upload_start = Instant::now();
    let resident = upload(&gpu, &p.matrix, &p.diagonal, &p.coefficients, &p.rhs);
    run(&gpu, &resident, 1);
    let first = upload_start.elapsed().as_secs_f64();
    let repeats = 20;
    let resident_start = Instant::now();
    run(&gpu, &resident, repeats);
    let per_application = resident_start.elapsed().as_secs_f64() / repeats as f64;
    let mut cpu = SmootherVectors::<f32>::zeros(p.rhs.len());
    let cpu_start = Instant::now();
    reference::chebyshev_smooth(&p.matrix, &p.diagonal, &p.coefficients, &p.rhs, &mut cpu);
    let cpu_seconds = cpu_start.elapsed().as_secs_f64();
    println!(
        "{}: {} block rows, {} stored values ({} MB); upload, pipelines and one application \
         {first:.4} s; one application resident {per_application:.6} s; CPU f32 reference one \
         application {cpu_seconds:.4} s",
        gpu.name,
        p.matrix.rows(),
        p.matrix.values().len(),
        p.matrix.values().len() * 4 / 1_000_000
    );
}
