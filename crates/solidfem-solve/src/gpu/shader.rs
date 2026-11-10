//! The WGSL of the smoother's kernels, [`SMOOTHER_WGSL`], reading the buffers `layout.rs` packs.
//! Bindings (group 0), which a pipeline built from one entry point uses only in part:
//!
//! | binding | name               | type                  | read by                         |
//! |---------|--------------------|-----------------------|---------------------------------|
//! | 0       | `params`           | uniform `Params`      | every kernel                    |
//! | 1       | `starts`           | storage `array<u32>`  | `multiply_main`                 |
//! | 2       | `columns`          | storage `array<u32>`  | `multiply_main`                 |
//! | 3       | `values`           | storage `array<f32>`  | `multiply_main`                 |
//! | 4       | `inverse_diagonal` | storage `array<f32>`  | start and update                |
//! | 5       | `rhs`              | storage `array<f32>`  | start                           |
//! | 6       | `residual`         | read-write            | all three (multiply's output)   |
//! | 7       | `direction`        | read-write            | all three (multiply's input)    |
//! | 8       | `solution`         | read-write            | start and update                |
//!
//! A plain product `y = A x` binds `x` as `direction` and `y` as `residual` with `subtract` 0.
//! Each kernel uses at most five storage buffers, under WebGPU's guaranteed eight per stage.

/// The largest block dimension the kernels handle: 6, one value per rigid-body mode on a coarse
/// level.
pub const MAX_BLOCK: usize = 6;

/// The block product and the block-Jacobi Chebyshev steps: entry points `multiply_main`,
/// `chebyshev_start_main` and `chebyshev_update_main`, workgroups of
/// [`super::WORKGROUP_SIZE`] threads dispatched as [`super::dispatch_size`] says.
pub const SMOOTHER_WGSL: &str = r#"
struct Params {
    rows: u32,
    row_block: u32,
    col_block: u32,
    subtract: u32,
    theta: f32,
    direction_scale: f32,
    scaled_scale: f32,
    pad: u32,
}

const WORKGROUP_SIZE: u32 = 64u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> starts: array<u32>;
@group(0) @binding(2) var<storage, read> columns: array<u32>;
@group(0) @binding(3) var<storage, read> values: array<f32>;
@group(0) @binding(4) var<storage, read> inverse_diagonal: array<f32>;
@group(0) @binding(5) var<storage, read> rhs: array<f32>;
@group(0) @binding(6) var<storage, read_write> residual: array<f32>;
@group(0) @binding(7) var<storage, read_write> direction: array<f32>;
@group(0) @binding(8) var<storage, read_write> solution: array<f32>;

fn thread_index(id: vec3<u32>, groups: vec3<u32>) -> u32 {
    return id.x + id.y * groups.x * WORKGROUP_SIZE;
}

// residual = A direction, or residual - A direction when params.subtract is 1. One thread per
// scalar row, summing its blocks in stored order and each block's columns in order from zero.
@compute @workgroup_size(64)
fn multiply_main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let row = thread_index(id, groups);
    let rb = params.row_block;
    if (row >= params.rows * rb) {
        return;
    }
    let block_row = row / rb;
    let p = row - block_row * rb;
    let cb = params.col_block;
    let block_len = rb * cb;
    var sum: f32 = 0.0;
    let end = starts[block_row + 1u];
    for (var at: u32 = starts[block_row]; at < end; at = at + 1u) {
        let base = at * block_len + p * cb;
        let column = columns[at] * cb;
        for (var q: u32 = 0u; q < cb; q = q + 1u) {
            sum = sum + values[base + q] * direction[column + q];
        }
    }
    if (params.subtract == 1u) {
        residual[row] = residual[row] - sum;
    } else {
        residual[row] = sum;
    }
}

// The smoother's first step from a zero guess: residual = rhs, direction = solution =
// (D^-1 rhs) / theta. One thread per block row.
@compute @workgroup_size(64)
fn chebyshev_start_main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let block_row = thread_index(id, groups);
    if (block_row >= params.rows) {
        return;
    }
    let n = params.row_block;
    let first = block_row * n;
    let diagonal = first * n;
    for (var p: u32 = 0u; p < n; p = p + 1u) {
        var sum: f32 = 0.0;
        for (var q: u32 = 0u; q < n; q = q + 1u) {
            sum = sum + inverse_diagonal[diagonal + p * n + q] * rhs[first + q];
        }
        let d = sum / params.theta;
        residual[first + p] = rhs[first + p];
        direction[first + p] = d;
        solution[first + p] = d;
    }
}

// A later step, after multiply_main has subtracted A direction from the residual:
// direction = direction_scale * direction + scaled_scale * (D^-1 residual), solution += direction.
// One thread per block row.
@compute @workgroup_size(64)
fn chebyshev_update_main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let block_row = thread_index(id, groups);
    if (block_row >= params.rows) {
        return;
    }
    let n = params.row_block;
    let first = block_row * n;
    let diagonal = first * n;
    for (var p: u32 = 0u; p < n; p = p + 1u) {
        var sum: f32 = 0.0;
        for (var q: u32 = 0u; q < n; q = q + 1u) {
            sum = sum + inverse_diagonal[diagonal + p * n + q] * residual[first + q];
        }
        let d = params.direction_scale * direction[first + p] + params.scaled_scale * sum;
        direction[first + p] = d;
        solution[first + p] = solution[first + p] + d;
    }
}
"#;
