// One multigrid V-cycle over a whole hierarchy on this machine's GPU through wgpu (native only).
// `DeviceHierarchy` holds every smoothed level's operators and vectors on the device; `commands`
// records the down and up legs once, each a single submission, so a cycle is: write the fine
// right-hand side, submit down, read back the coarsest right-hand side, solve it on the CPU, write
// the coarse correction, submit up, read back the fine correction. `DeviceCycle` is that cycle
// behind the production `Preconditioner` trait, which `gpu_solve_execution.rs` hands to `Pcg`.
//
// Every level's smoother and transfers are compared stage by stage with the f32 CPU reference on
// the device's own input to that stage, within the derived bounds `gpu_cycle_execution.rs` states
// (6x6 smoothers included), and a cycle must repeat to the bit. No per-level device limit is
// reached: a dispatch binds one group of at most six storage buffers and one uniform, and WebGPU
// does not limit how many buffers or bind groups exist, only their size. Each matrix array is one
// buffer in one binding, so `maxStorageBufferBindingSize` and `maxBufferSize` both apply; the
// guaranteed minimums are 128 MiB and 256 MiB and the adapter measured allows more of both, so
// the binding is the limit `BlockMatrix::check_binding_size` checks for every matrix here.
//
// A hierarchy whose coarsest level has no factorization panics in the coarse solve's `expect`: the
// device driver does not run a coarsest-level smoother.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::time::Instant;

use solidfem_solve::gpu::reference::{self, CycleLevel, SmootherVectors};
use solidfem_solve::gpu::{f32_le_bytes, GpuError, GUARANTEED_BINDING_BYTES};
use solidfem_solve::{CsrMatrix, Multigrid, MultigridOptions, Precision, Preconditioner};

use super::gpu_cycle_execution::{
    pipelines, product_bound, read_vector, resident, smoother, submit, submit_repeated, transfer,
    varied_f32, vector, within, Command, Pipelines, Resident,
};
use super::gpu_execution::{gpu_or_skip, read, storage, Gpu};
use super::gpu_transfer::{built_hierarchy, coarse_solve, cycle_level};

/// One smoothed level on the device.
pub(super) struct DeviceLevel {
    a: Resident,
    prolongator: Resident,
    restriction: Resident,
    inverse: wgpu::Buffer,
    work: (wgpu::Buffer, wgpu::Buffer),
    b: wgpu::Buffer,
    x: wgpu::Buffer,
    r: wgpu::Buffer,
    post: wgpu::Buffer,
    /// Copies of `x` after the pre-smoother, `r` before restriction, `x` after the correction and
    /// `r` after it, when the commands are recorded for a stage-by-stage comparison.
    trace: Option<[wgpu::Buffer; 4]>,
}

/// A hierarchy on the device: every smoothed level and the coarsest level's two vectors.
pub(super) struct DeviceHierarchy {
    pipelines: Pipelines,
    /// The levels as the reference reads them; `cycle_levels[i]` is `levels[i]`.
    pub(super) cycle_levels: Vec<CycleLevel<f32>>,
    levels: Vec<DeviceLevel>,
    coarse_b: wgpu::Buffer,
    coarse_x: wgpu::Buffer,
    coarse_readback: wgpu::Buffer,
    fine_readback: wgpu::Buffer,
}

fn zeros(gpu: &Gpu, label: &str, length: usize) -> wgpu::Buffer {
    vector(gpu, label, &vec![0.0; length])
}

fn readback_of(gpu: &Gpu, length: usize) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        // Four bytes per f32; a usize length times four fits u64.
        size: 4 * length as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

impl DeviceHierarchy {
    /// Uploads every smoothed level, refusing a matrix larger than one binding of `limit` bytes.
    pub(super) fn upload(
        gpu: &Gpu,
        multigrid: &Multigrid,
        limit: u64,
        trace: bool,
    ) -> Result<Self, GpuError> {
        let count = multigrid.stats().levels;
        assert!(count >= 2, "a hierarchy of one level has no device work");
        let cycle_levels: Vec<CycleLevel<f32>> = (0..count - 1)
            .map(|index| cycle_level::<f32>(multigrid, index))
            .collect();
        let mut levels = Vec::with_capacity(cycle_levels.len());
        for level in &cycle_levels {
            for matrix in [&level.matrix, &level.prolongator, &level.restriction] {
                matrix.check_binding_size(limit)?;
            }
            let length = level.matrix.rows() * level.matrix.row_block();
            levels.push(DeviceLevel {
                a: resident(gpu, &level.matrix),
                prolongator: resident(gpu, &level.prolongator),
                restriction: resident(gpu, &level.restriction),
                inverse: storage(
                    gpu,
                    "inverse diagonal",
                    &f32_le_bytes(&level.inverse_diagonal),
                    false,
                ),
                work: (zeros(gpu, "work r", length), zeros(gpu, "work d", length)),
                b: zeros(gpu, "b", length),
                x: zeros(gpu, "x", length),
                r: zeros(gpu, "r", length),
                post: zeros(gpu, "post", length),
                trace: trace.then(|| {
                    ["x pre", "r pre", "x corrected", "r corrected"]
                        .map(|label| zeros(gpu, label, length))
                }),
            });
        }
        let last = &cycle_levels[cycle_levels.len() - 1].prolongator;
        let coarse = last.block_columns() * last.col_block();
        let fine = cycle_levels[0].matrix.rows() * cycle_levels[0].matrix.row_block();
        Ok(DeviceHierarchy {
            pipelines: pipelines(gpu),
            cycle_levels,
            levels,
            coarse_b: zeros(gpu, "coarse b", coarse),
            coarse_x: zeros(gpu, "coarse x", coarse),
            coarse_readback: readback_of(gpu, coarse),
            fine_readback: readback_of(gpu, fine),
        })
    }

    /// The right-hand side and solution of the level below `levels[index]`.
    fn coarser(&self, index: usize) -> (&wgpu::Buffer, &wgpu::Buffer) {
        match self.levels.get(index + 1) {
            Some(next) => (&next.b, &next.x),
            None => (&self.coarse_b, &self.coarse_x),
        }
    }

    /// The cycle's down and up legs, fine to coarse and back. A bind group holds buffers, so each
    /// is made once and every later cycle reuses it.
    pub(super) fn commands(&self, gpu: &Gpu) -> (Vec<Command<'_>>, Vec<Command<'_>>) {
        let p = &self.pipelines;
        let mut down = Vec::new();
        for (index, (level, cycle)) in self.levels.iter().zip(&self.cycle_levels).enumerate() {
            let (next_b, _) = self.coarser(index);
            let (a, inverse, work) = (&level.a, &level.inverse, &level.work);
            down.extend(smoother(
                gpu, p, cycle, a, inverse, work, &level.b, &level.x,
            ));
            if let Some(trace) = &level.trace {
                down.push(Command::Copy(&level.x, &trace[0]));
            }
            down.push(transfer(
                gpu,
                &p.residual,
                a,
                &level.x,
                Some(&level.b),
                &level.r,
                true,
            ));
            if let Some(trace) = &level.trace {
                down.push(Command::Copy(&level.r, &trace[1]));
            }
            down.push(transfer(
                gpu,
                &p.restrict,
                &level.restriction,
                &level.r,
                None,
                next_b,
                true,
            ));
        }
        let mut up = Vec::new();
        for (index, (level, cycle)) in self.levels.iter().zip(&self.cycle_levels).enumerate().rev()
        {
            let (_, next_x) = self.coarser(index);
            let (a, inverse, work) = (&level.a, &level.inverse, &level.work);
            up.push(transfer(
                gpu,
                &p.prolong,
                &level.prolongator,
                next_x,
                None,
                &level.x,
                true,
            ));
            if let Some(trace) = &level.trace {
                up.push(Command::Copy(&level.x, &trace[2]));
            }
            up.push(transfer(
                gpu,
                &p.residual,
                a,
                &level.x,
                Some(&level.b),
                &level.r,
                true,
            ));
            if let Some(trace) = &level.trace {
                up.push(Command::Copy(&level.r, &trace[3]));
            }
            up.extend(smoother(
                gpu,
                p,
                cycle,
                a,
                inverse,
                work,
                &level.r,
                &level.post,
            ));
            up.push(transfer(gpu, &p.add, a, &level.post, None, &level.x, false));
        }
        (down, up)
    }

    /// Times the smoother dispatches and the transfer dispatches of one whole cycle apart, in
    /// seconds per cycle, on whatever the buffers hold: each is `repeats` repetitions of its
    /// dispatches in one submission so a wait is not counted per dispatch. Neither depends on the
    /// values. The first submission after an upload also waits for the upload's own copies to the
    /// device, so a timing of the dispatches alone must not be the first.
    pub(super) fn dispatch_seconds(&self, gpu: &Gpu, repeats: usize) -> (f64, f64) {
        let p = &self.pipelines;
        let mut smooth = Vec::new();
        let mut transfers = Vec::new();
        for (index, (level, cycle)) in self.levels.iter().zip(&self.cycle_levels).enumerate() {
            let (next_b, next_x) = self.coarser(index);
            let (a, inverse, work) = (&level.a, &level.inverse, &level.work);
            smooth.extend(smoother(
                gpu, p, cycle, a, inverse, work, &level.b, &level.x,
            ));
            smooth.extend(smoother(
                gpu,
                p,
                cycle,
                a,
                inverse,
                work,
                &level.r,
                &level.post,
            ));
            let residual = |x: &wgpu::Buffer, b: &wgpu::Buffer, r: &wgpu::Buffer| {
                transfer(gpu, &p.residual, a, x, Some(b), r, true)
            };
            transfers.push(residual(&level.x, &level.b, &level.r));
            transfers.push(transfer(
                gpu,
                &p.restrict,
                &level.restriction,
                &level.r,
                None,
                next_b,
                true,
            ));
            transfers.push(transfer(
                gpu,
                &p.prolong,
                &level.prolongator,
                next_x,
                None,
                &level.x,
                true,
            ));
            transfers.push(residual(&level.x, &level.b, &level.r));
            transfers.push(transfer(gpu, &p.add, a, &level.post, None, &level.x, false));
        }
        let timed = |commands: &[Command<'_>]| {
            let clock = Instant::now();
            submit_repeated(gpu, commands, repeats);
            clock.elapsed().as_secs_f64() / repeats as f64
        };
        (timed(&smooth), timed(&transfers))
    }
}

/// Seconds a device cycle spent in each phase, summed over `cycles` cycles.
#[derive(Default, Clone, Copy)]
pub(super) struct Phases {
    pub(super) cycles: u32,
    pub(super) rounding: f64,
    pub(super) down: f64,
    pub(super) coarse_read: f64,
    pub(super) coarse_solve: f64,
    pub(super) up: f64,
    pub(super) fine_read: f64,
}

/// A device V-cycle, the production preconditioner seam: the conjugate gradient in `f64` on the
/// CPU, the cycle in `f32` on the device with the coarsest solve on the CPU in `f64`.
pub(super) struct DeviceCycle<'a> {
    gpu: &'a Gpu,
    hierarchy: &'a DeviceHierarchy,
    multigrid: &'a Multigrid,
    down: &'a [Command<'a>],
    up: &'a [Command<'a>],
    phases: RefCell<Phases>,
}

impl<'a> DeviceCycle<'a> {
    pub(super) fn new(
        gpu: &'a Gpu,
        hierarchy: &'a DeviceHierarchy,
        multigrid: &'a Multigrid,
        down: &'a [Command<'a>],
        up: &'a [Command<'a>],
    ) -> Self {
        DeviceCycle {
            gpu,
            hierarchy,
            multigrid,
            down,
            up,
            phases: RefCell::new(Phases::default()),
        }
    }

    /// The phase times accumulated so far.
    pub(super) fn phases(&self) -> Phases {
        *self.phases.borrow()
    }

    /// One device cycle: the `f32` right-hand side in, the `f32` correction out.
    pub(super) fn apply_f32(&self, rhs: &[f32]) -> Vec<f32> {
        let (gpu, h) = (self.gpu, self.hierarchy);
        let mut phases = self.phases.borrow_mut();
        let clock = Instant::now();
        gpu.queue
            .write_buffer(&h.levels[0].b, 0, &f32_le_bytes(rhs));
        submit(gpu, self.down);
        phases.down += clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let coarse_rhs = read(gpu, &h.coarse_b, &h.coarse_readback);
        phases.coarse_read += clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let mut solved = vec![0.0f32; coarse_rhs.len()];
        coarse_solve(self.multigrid, &coarse_rhs, &mut solved);
        phases.coarse_solve += clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        gpu.queue
            .write_buffer(&h.coarse_x, 0, &f32_le_bytes(&solved));
        submit(gpu, self.up);
        phases.up += clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let correction = read(gpu, &h.levels[0].x, &h.fine_readback);
        phases.fine_read += clock.elapsed().as_secs_f64();
        phases.cycles += 1;
        correction
    }
}

impl Preconditioner for DeviceCycle<'_> {
    fn apply(&self, residual: &[f64], output: &mut [f64]) {
        let clock = Instant::now();
        // Rounds to nearest; a residual beyond f32's range would be infinite and show as a failed
        // solve, as in the CPU's Single cycle.
        let rhs: Vec<f32> = residual.iter().map(|&v| v as f32).collect();
        let rounding = clock.elapsed().as_secs_f64();
        let correction = self.apply_f32(&rhs);
        let clock = Instant::now();
        for (out, value) in output.iter_mut().zip(&correction) {
            *out = f64::from(*value);
        }
        self.phases.borrow_mut().rounding += rounding + clock.elapsed().as_secs_f64();
    }

    /// The conjugate gradient's product, on the hierarchy's worker threads as a CPU solve runs it.
    fn system_multiply(&self, matrix: &CsrMatrix, x: &[f64], y: &mut [f64]) {
        self.multigrid.system_multiply(matrix, x, y);
    }
}

fn multigrid(n: usize, coarse_dofs: usize, levels: usize) -> Multigrid {
    let options = MultigridOptions {
        coarse_dofs,
        max_levels: levels,
        precision: Precision::Single,
        ..MultigridOptions::default()
    };
    let (_, multigrid) = built_hierarchy(n, options);
    assert_eq!(multigrid.stats().levels, levels);
    multigrid
}

#[test]
fn a_six_by_six_smoother_runs_on_the_gpu_as_the_cpu_reference() {
    let Some(gpu) = gpu_or_skip("the GPU 6x6 smoother") else {
        return;
    };
    let hierarchy = multigrid(12, 30, 3);
    let level = cycle_level::<f32>(&hierarchy, 1);
    assert_eq!((level.matrix.row_block(), level.matrix.col_block()), (6, 6));
    let p = pipelines(&gpu);
    let length = level.matrix.rows() * 6;
    let a = resident(&gpu, &level.matrix);
    let inverse = storage(
        &gpu,
        "inverse",
        &f32_le_bytes(&level.inverse_diagonal),
        false,
    );
    let work = (zeros(&gpu, "work r", length), zeros(&gpu, "work d", length));
    let rhs = varied_f32(length, 5);
    let b = vector(&gpu, "b", &rhs);
    let x = zeros(&gpu, "x", length);
    let commands = smoother(&gpu, &p, &level, &a, &inverse, &work, &b, &x);
    submit(&gpu, &commands);
    let found = read_vector(&gpu, &x);
    let mut expected = SmootherVectors::<f32>::zeros(length);
    reference::chebyshev_smooth(
        &level.matrix,
        &level.inverse_diagonal,
        &level.coefficients,
        &rhs,
        &mut expected,
    );
    let bound = reference::smoother_rounding_bound(
        &level.matrix,
        &level.inverse_diagonal,
        &level.coefficients,
        &rhs,
    );
    let same = within(&gpu, "6x6 smoother", &found, &expected.solution, &bound);
    println!(
        "{}: 6x6 smoother over {length} values, {same} identical to the CPU",
        gpu.name
    );
}

/// Runs one traced device cycle over all of `hierarchy`'s levels and compares every stage of every
/// level with the reference on the device's own input to that stage.
fn compare_every_stage(gpu: &Gpu, hierarchy: &Multigrid) {
    let device = DeviceHierarchy::upload(gpu, hierarchy, GUARANTEED_BINDING_BYTES, true)
        .expect("the test hierarchies fit the guaranteed binding");
    let (down, up) = device.commands(gpu);
    let fine = &device.cycle_levels[0].matrix;
    let b = varied_f32(fine.rows() * fine.row_block(), 0);
    let cycle = DeviceCycle::new(gpu, &device, hierarchy, &down, &up);
    let first = cycle.apply_f32(&b);
    let second = cycle.apply_f32(&b);
    // One thread owns every sum and the coarse solve is the same CPU arithmetic.
    assert_eq!(second, first, "{}: a cycle repeats to the bit", gpu.name);
    for index in 0..device.levels.len() {
        let (level, stages) = (&device.levels[index], &device.cycle_levels[index]);
        let trace = level.trace.as_ref().expect("traced for this comparison");
        let name = |stage: &str| format!("level {index} {stage}");
        let device_b = if index == 0 {
            b.clone()
        } else {
            read_vector(gpu, &level.b)
        };
        let smooth = |rhs: &[f32]| {
            let mut vectors = SmootherVectors::<f32>::zeros(rhs.len());
            let (m, d, c) = (
                &stages.matrix,
                &stages.inverse_diagonal,
                &stages.coefficients,
            );
            reference::chebyshev_smooth(m, d, c, rhs, &mut vectors);
            (
                vectors.solution,
                reference::smoother_rounding_bound(m, d, c, rhs),
            )
        };
        let [x_pre, r_pre, x_corrected, r_corrected] = [
            read_vector(gpu, &trace[0]),
            read_vector(gpu, &trace[1]),
            read_vector(gpu, &trace[2]),
            read_vector(gpu, &trace[3]),
        ];
        let (next_b, next_x) = device.coarser(index);
        let (next_b, next_x) = (read_vector(gpu, next_b), read_vector(gpu, next_x));
        let (expected, bound) = smooth(&device_b);
        within(gpu, &name("pre-smooth"), &x_pre, &expected, &bound);
        let mut expected = vec![0.0; device_b.len()];
        reference::residual(&stages.matrix, &device_b, &x_pre, &mut expected);
        let bound = product_bound(&stages.matrix, &x_pre, Some(&device_b));
        within(gpu, &name("residual"), &r_pre, &expected, &bound);
        let mut expected = vec![0.0; next_b.len()];
        reference::restrict(&stages.restriction, &r_pre, &mut expected);
        let bound = product_bound(&stages.restriction, &r_pre, None);
        within(gpu, &name("restrict"), &next_b, &expected, &bound);
        let mut expected = x_pre.clone();
        reference::prolong_correct(&stages.prolongator, &next_x, &mut expected);
        let bound = product_bound(&stages.prolongator, &next_x, Some(&x_pre));
        within(
            gpu,
            &name("prolong-correct"),
            &x_corrected,
            &expected,
            &bound,
        );
        let mut expected = vec![0.0; device_b.len()];
        reference::residual(&stages.matrix, &device_b, &x_corrected, &mut expected);
        let bound = product_bound(&stages.matrix, &x_corrected, Some(&device_b));
        within(
            gpu,
            &name("second residual"),
            &r_corrected,
            &expected,
            &bound,
        );
        let post = read_vector(gpu, &level.post);
        let (expected, bound) = smooth(&r_corrected);
        within(gpu, &name("post-smooth"), &post, &expected, &bound);
        let mut expected = x_corrected.clone();
        reference::add(&mut expected, &post);
        // A single f32 addition is correctly rounded in WGSL, so no difference is allowed.
        assert_eq!(
            read_vector(gpu, &level.x),
            expected,
            "{}: level {index} final sum",
            gpu.name
        );
    }
    // The coarsest level's correction is the CPU's solve of the device's right-hand side, written
    // verbatim.
    let coarse_rhs = read_vector(gpu, &device.coarse_b);
    let mut solved = vec![0.0f32; coarse_rhs.len()];
    coarse_solve(hierarchy, &coarse_rhs, &mut solved);
    assert_eq!(read_vector(gpu, &device.coarse_x), solved, "{}", gpu.name);
    // The whole cycle against the reference: measured, not asserted, because the coarse solve
    // amplifies the stages' rounding by the coarse matrix's conditioning, which no bound carries.
    let whole = reference::v_cycle(&device.cycle_levels, &b, &mut |rhs, out| {
        coarse_solve(hierarchy, rhs, out)
    });
    let largest = whole.iter().fold(0.0f64, |m, v| m.max(f64::from(v.abs())));
    let difference = first.iter().zip(&whole).fold(0.0f64, |m, (d, c)| {
        m.max((f64::from(*d) - f64::from(*c)).abs())
    });
    println!(
        "{}: {}-level cycle over {} fine values; largest difference from the CPU cycle {:.3e} of \
         the largest entry",
        gpu.name,
        device.levels.len() + 1,
        first.len(),
        difference / largest
    );
}

#[test]
fn a_three_level_cycle_runs_on_the_gpu_stage_by_stage_as_the_cpu_reference() {
    let Some(gpu) = gpu_or_skip("the GPU three-level cycle") else {
        return;
    };
    compare_every_stage(&gpu, &multigrid(12, 30, 3));
}

#[test]
fn a_four_level_cycle_runs_on_the_gpu_stage_by_stage_as_the_cpu_reference() {
    let Some(gpu) = gpu_or_skip("the GPU four-level cycle") else {
        return;
    };
    compare_every_stage(&gpu, &multigrid(24, 30, 4));
}
