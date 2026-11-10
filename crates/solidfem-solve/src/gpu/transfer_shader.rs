//! The WGSL of the rest of one V-cycle level, [`TRANSFER_WGSL`]: the residual, the restriction,
//! the prolongation with its correction, and the vector sum that adds the post-smoother's
//! correction. It reads the same block layout and `Params` record as [`super::SMOOTHER_WGSL`]
//! (`rows`, `row_block` and `col_block` of the bound matrix; the other fields are unused).
//! Bindings (group 0):
//!
//! | binding | name      | type                  | read by                                        |
//! |---------|-----------|-----------------------|------------------------------------------------|
//! | 0       | `params`  | uniform `Params`      | every kernel                                   |
//! | 1       | `starts`  | storage `array<u32>`  | residual, restrict, prolong-correct            |
//! | 2       | `columns` | storage `array<u32>`  | residual, restrict, prolong-correct            |
//! | 3       | `values`  | storage `array<f32>`  | residual, restrict, prolong-correct            |
//! | 4       | `input`   | storage `array<f32>`  | every kernel: the vector the matrix multiplies |
//! | 5       | `base`    | storage `array<f32>`  | `residual_main`: the right-hand side `b`       |
//! | 6       | `output`  | read-write            | every kernel                                   |
//!
//! - `residual_main`: `output = base - A input`, the level's matrix bound.
//! - `restrict_main`: `output = P^T input`, the prolongator's stored transpose
//!   ([`super::BlockMatrix::transpose`]) bound, so each coarse scalar row gathers its terms in
//!   ascending fine row order and no two threads write one value.
//! - `prolong_correct_main`: `output = output + P input`, the prolongator bound.
//! - `add_main`: `output = output + input` over `rows * row_block` values.
//!
//! One thread per scalar row; each sums from zero over its blocks in stored order and each
//! block's columns in order, then does its single subtraction or addition, the order the CPU
//! cycle uses, so results repeat to the bit on one device.

/// The residual, restriction, prolongation and correction kernels: entry points `residual_main`,
/// `restrict_main`, `prolong_correct_main` and `add_main`, workgroups of
/// [`super::WORKGROUP_SIZE`] threads dispatched over scalar rows as [`super::dispatch_size`] says.
pub const TRANSFER_WGSL: &str = r#"
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
@group(0) @binding(4) var<storage, read> input: array<f32>;
@group(0) @binding(5) var<storage, read> base: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

fn thread_index(id: vec3<u32>, groups: vec3<u32>) -> u32 {
    return id.x + id.y * groups.x * WORKGROUP_SIZE;
}

// Row `row` of the bound matrix times `input`, summed from zero over the row's blocks in stored
// order and each block's columns in order.
fn row_product(row: u32) -> f32 {
    let rb = params.row_block;
    let block_row = row / rb;
    let p = row - block_row * rb;
    let cb = params.col_block;
    let block_len = rb * cb;
    var sum: f32 = 0.0;
    let end = starts[block_row + 1u];
    for (var at: u32 = starts[block_row]; at < end; at = at + 1u) {
        let first = at * block_len + p * cb;
        let column = columns[at] * cb;
        for (var q: u32 = 0u; q < cb; q = q + 1u) {
            sum = sum + values[first + q] * input[column + q];
        }
    }
    return sum;
}

// output = base - A input.
@compute @workgroup_size(64)
fn residual_main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let row = thread_index(id, groups);
    if (row >= params.rows * params.row_block) {
        return;
    }
    output[row] = base[row] - row_product(row);
}

// output = P^T input, with the stored transpose of P bound as the matrix.
@compute @workgroup_size(64)
fn restrict_main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let row = thread_index(id, groups);
    if (row >= params.rows * params.row_block) {
        return;
    }
    output[row] = row_product(row);
}

// output = output + P input, with P bound as the matrix.
@compute @workgroup_size(64)
fn prolong_correct_main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let row = thread_index(id, groups);
    if (row >= params.rows * params.row_block) {
        return;
    }
    output[row] = output[row] + row_product(row);
}

// output = output + input over rows * row_block values.
@compute @workgroup_size(64)
fn add_main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let row = thread_index(id, groups);
    if (row >= params.rows * params.row_block) {
        return;
    }
    output[row] = output[row] + input[row];
}
"#;
