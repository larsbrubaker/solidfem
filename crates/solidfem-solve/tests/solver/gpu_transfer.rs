// The V-cycle transfer kernels without a GPU: the transfer WGSL validates with its entry points and
// shares the smoother's uniform record; the stored transpose is a true transpose; a matrix too big
// for one binding is refused by name with how many nodes fit; and the CPU reference of one
// cycle, built from a real hierarchy's stored levels at two, three and four levels, is the
// multigrid's own cycle bit for bit, in f32 against `Precision::Single` (and at two levels in f64
// against `Precision::Double`).

use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{Module, ShaderStage, TypeInner};
use solidfem_solve::gpu::reference::{self, CycleLevel};
use solidfem_solve::gpu::{
    BlockMatrix, ChebyshevCoefficients, GpuError, Real, GUARANTEED_BINDING_BYTES, SMOOTHER_WGSL,
    TRANSFER_WGSL, WORKGROUP_SIZE,
};
use solidfem_solve::{
    CsrMatrix, Multigrid, MultigridError, MultigridOptions, MultigridSetup, Precision,
    Preconditioner, SetupStatus, NEAR_NULL_MODES,
};

use super::gpu_kernels::{six_by_three, spring_lattice, varied_vector};

fn parsed(source: &str) -> Module {
    naga::front::wgsl::parse_str(source).expect("the shader parses as WGSL")
}

/// The member names and offsets of the shader's `Params`, and its size.
fn params_layout(module: &Module) -> (Vec<(String, u32)>, u32) {
    let (_, params) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("Params"))
        .expect("the shader has a Params struct");
    let TypeInner::Struct { members, span } = &params.inner else {
        panic!("Params is not a struct");
    };
    let fields = members
        .iter()
        .map(|m| (m.name.clone().unwrap_or_default(), m.offset))
        .collect();
    (fields, *span)
}

#[test]
fn the_transfer_shader_validates_with_its_four_entry_points() {
    let module = parsed(TRANSFER_WGSL);
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .expect("the shader validates");
    let entries: Vec<(&str, [u32; 3])> = module
        .entry_points
        .iter()
        .filter(|e| e.stage == ShaderStage::Compute)
        .map(|e| (e.name.as_str(), e.workgroup_size))
        .collect();
    let size = [WORKGROUP_SIZE, 1, 1];
    assert_eq!(
        entries,
        [
            ("residual_main", size),
            ("restrict_main", size),
            ("prolong_correct_main", size),
            ("add_main", size),
        ]
    );
    // One uniform record serves both shaders.
    assert_eq!(
        params_layout(&module),
        params_layout(&parsed(SMOOTHER_WGSL))
    );
}

#[test]
fn the_stored_transpose_is_the_transpose_of_every_block() {
    let (starts, columns, values) = six_by_three();
    let matrix = BlockMatrix::<f64>::from_parts(3, 6, 3, &starts, &columns, &values).unwrap();
    let transpose = matrix.transpose();
    assert_eq!(
        (transpose.rows(), transpose.block_columns()),
        (matrix.block_columns(), matrix.rows())
    );
    assert_eq!((transpose.row_block(), transpose.col_block()), (3, 6));
    // Entry (i, j) of the transpose is entry (j, i) of the matrix, read through the products of
    // unit vectors.
    let (rows, cols) = (matrix.rows() * 6, matrix.block_columns() * 3);
    for j in 0..cols {
        let mut unit = vec![0.0; cols];
        unit[j] = 1.0;
        let mut column = vec![0.0; rows];
        reference::multiply(&matrix, &unit, &mut column);
        for (i, &value) in column.iter().enumerate() {
            let mut unit_row = vec![0.0; rows];
            unit_row[i] = 1.0;
            let mut row = vec![0.0; cols];
            reference::restrict(&transpose, &unit_row, &mut row);
            assert_eq!(row[j], value, "entry ({i}, {j})");
        }
    }
    assert_eq!(transpose.transpose(), matrix);
}

#[test]
fn a_matrix_larger_than_one_binding_is_refused_with_how_many_nodes_fit() {
    let matrix = BlockMatrix::<f32>::from_nodal(&spring_lattice(6)).unwrap();
    assert_eq!(matrix.check_binding_size(GUARANTEED_BINDING_BYTES), Ok(()));
    let bytes = 4 * matrix.values().len() as u64;
    assert_eq!(matrix.check_binding_size(bytes), Ok(()));
    let limit = bytes / 3;
    let error = matrix.check_binding_size(limit).unwrap_err();
    let GpuError::BindingTooLarge {
        bytes: found,
        limit: found_limit,
        rows,
        rows_that_fit,
    } = error
    else {
        panic!("expected BindingTooLarge, got {error:?}");
    };
    assert_eq!((found, found_limit, rows), (bytes, limit, 216));
    // A third of the bytes holds a third of the rows, rounded down.
    assert_eq!(
        rows_that_fit,
        (u128::from(limit) * 216 / u128::from(bytes)) as usize
    );
    assert!(rows_that_fit < 216 / 3 + 1 && rows_that_fit + 1 >= 216 / 3);
    let message = error.to_string();
    assert!(
        message.contains(&format!("about {rows_that_fit} of its 216 block rows fit")),
        "{message}"
    );
}

/// The rigid-body modes of nodes at `positions`: per scalar row, three translations and three
/// rotations, the near-null space of an unsupported elastic body.
pub(super) fn rigid_body_modes(positions: &[[f64; 3]]) -> Vec<f64> {
    let mut modes = Vec::with_capacity(positions.len() * 3 * NEAR_NULL_MODES);
    for &[x, y, z] in positions {
        modes.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, z, -y]);
        modes.extend_from_slice(&[0.0, 1.0, 0.0, -z, 0.0, x]);
        modes.extend_from_slice(&[0.0, 0.0, 1.0, y, -x, 0.0]);
    }
    modes
}

/// The `n^3` spring lattice with its rigid-body modes as a hierarchy built with `options`.
pub(super) fn built_hierarchy(n: usize, options: MultigridOptions) -> (CsrMatrix, Multigrid) {
    let csr = spring_lattice(n);
    let positions: Vec<[f64; 3]> = (0..n * n * n)
        .map(|i| [(i % n) as f64, ((i / n) % n) as f64, (i / (n * n)) as f64])
        .collect();
    let modes = rigid_body_modes(&positions);
    let mut setup = MultigridSetup::new(&csr, &modes, &positions, options).unwrap();
    while setup.step(usize::MAX, &mut || false).unwrap() != SetupStatus::Finished {}
    (csr, setup.finish().unwrap())
}

/// The lattice as a two-level hierarchy in `precision`: an aggregated fine level with a 3x6-block
/// prolongator and a coarse level of 6x6 blocks factored directly.
pub(super) fn two_level_hierarchy(n: usize, precision: Precision) -> (CsrMatrix, Multigrid) {
    let options = MultigridOptions {
        max_levels: 2,
        precision,
        ..MultigridOptions::default()
    };
    let (csr, multigrid) = built_hierarchy(n, options);
    assert_eq!(multigrid.stats().levels, 2);
    (csr, multigrid)
}

/// Level `index` of `multigrid`, which must have a coarser level, as the device reads it, in `T`.
pub(super) fn cycle_level<T: Real>(multigrid: &Multigrid, index: usize) -> CycleLevel<T> {
    let wide = BlockMatrix::<f64>::from_level(&multigrid.level_matrix(index).unwrap()).unwrap();
    let prolongator =
        BlockMatrix::<f64>::from_level(&multigrid.level_prolongator(index).unwrap()).unwrap();
    let (lambda_min, lambda_max) = multigrid.smoother_spectrum(index).unwrap();
    let degree = MultigridOptions::default().smoother_degree;
    CycleLevel {
        matrix: wide.convert(),
        inverse_diagonal: multigrid
            .level_inverse_diagonal(index)
            .unwrap()
            .iter()
            .map(|&v| T::from_f64(v))
            .collect(),
        coefficients: ChebyshevCoefficients::new(lambda_min, lambda_max, degree).unwrap(),
        restriction: prolongator.transpose().convert(),
        prolongator: prolongator.convert(),
    }
}

/// The hierarchy's own coarse solve on values of `T`: widened, solved in `f64`, rounded back, as
/// the cycle does it.
pub(super) fn coarse_solve<T: Real>(multigrid: &Multigrid, rhs: &[T], x: &mut [T]) {
    let mut values: Vec<f64> = rhs.iter().map(|v| v.to_f64()).collect();
    multigrid
        .solve_coarsest(&mut values)
        .expect("a factored coarsest level");
    for (to, from) in x.iter_mut().zip(&values) {
        *to = T::from_f64(*from);
    }
}

#[test]
fn the_levels_read_back_with_the_shapes_the_kernels_handle() {
    let (_, multigrid) = two_level_hierarchy(8, Precision::Single);
    let level = cycle_level::<f32>(&multigrid, 0);
    let coarse = multigrid.level_matrix(1).unwrap();
    assert_eq!((level.matrix.row_block(), level.matrix.col_block()), (3, 3));
    assert_eq!(
        (level.prolongator.row_block(), level.prolongator.col_block()),
        (3, NEAR_NULL_MODES)
    );
    assert_eq!((coarse.row_block, coarse.col_block), (6, 6));
    assert_eq!(level.prolongator.block_columns(), coarse.starts.len() - 1);
    assert!(multigrid.level_prolongator(1).is_none() && multigrid.level_matrix(2).is_none());
}

#[test]
fn a_coarse_solve_of_the_wrong_length_is_refused_by_name_and_leaves_the_values_alone() {
    let (_, multigrid) = two_level_hierarchy(8, Precision::Single);
    let expected = multigrid.stats().coarse_dofs;
    let mut values = [1.5; 5];
    assert_eq!(
        multigrid.solve_coarsest(&mut values),
        Err(MultigridError::CoarseLength { expected, found: 5 })
    );
    assert_eq!(values, [1.5; 5]);
}

#[test]
fn a_smoothed_coarsest_level_is_named_apart_from_a_wrong_length() {
    // 11^3 nodes, 3,993 unknowns: above the dense factorization limit, so a one-level hierarchy's
    // coarsest level has no factor and a host must run its smoother instead.
    let options = MultigridOptions {
        max_levels: 1,
        ..MultigridOptions::default()
    };
    let (csr, multigrid) = built_hierarchy(11, options);
    assert_eq!(multigrid.stats().levels, 1);
    let mut values = varied_vector(csr.size());
    let untouched = values.clone();
    let error = multigrid.solve_coarsest(&mut values).unwrap_err();
    assert_eq!(
        error,
        MultigridError::CoarsestNotFactored { values: csr.size() }
    );
    assert_eq!(values, untouched);
    assert!(error.to_string().contains("apply its smoother"), "{error}");
}

#[test]
fn the_f32_reference_cycle_is_the_single_precision_multigrid_bit_for_bit() {
    let (csr, multigrid) = two_level_hierarchy(8, Precision::Single);
    let level = cycle_level::<f32>(&multigrid, 0);
    // f32 values, so the hierarchy's rounding of its input is exact.
    let rhs: Vec<f32> = varied_vector(csr.size())
        .iter()
        .map(|&v| v as f32)
        .collect();
    let found =
        reference::two_level_cycle(&level, &rhs, &mut |b, x| coarse_solve(&multigrid, b, x));
    let mut expected = vec![0.0; csr.size()];
    let wide_rhs: Vec<f64> = rhs.iter().map(|&v| f64::from(v)).collect();
    multigrid.apply(&wide_rhs, &mut expected);
    let found: Vec<f64> = found.iter().map(|&v| f64::from(v)).collect();
    assert_eq!(found, expected);
}

#[test]
fn the_f64_reference_cycle_is_the_double_precision_multigrid_bit_for_bit() {
    let (csr, multigrid) = two_level_hierarchy(8, Precision::Double);
    let level = cycle_level::<f64>(&multigrid, 0);
    let rhs = varied_vector(csr.size());
    let found =
        reference::two_level_cycle(&level, &rhs, &mut |b, x| coarse_solve(&multigrid, b, x));
    let mut expected = vec![0.0; csr.size()];
    multigrid.apply(&rhs, &mut expected);
    assert_eq!(found, expected);
}

/// Asserts the f32 reference cycle over every smoothed level equals the single-precision cycle to
/// the bit, on a hierarchy of `expect_levels` levels built from the `n^3` lattice.
fn assert_f32_reference_matches(n: usize, coarse_dofs: usize, expect_levels: usize) {
    let options = MultigridOptions {
        coarse_dofs,
        max_levels: expect_levels,
        precision: Precision::Single,
        ..MultigridOptions::default()
    };
    let (csr, multigrid) = built_hierarchy(n, options);
    assert_eq!(multigrid.stats().levels, expect_levels);
    let levels: Vec<CycleLevel<f32>> = (0..expect_levels - 1)
        .map(|index| cycle_level::<f32>(&multigrid, index))
        .collect();
    let rhs: Vec<f32> = varied_vector(csr.size())
        .iter()
        .map(|&v| v as f32)
        .collect();
    let found = reference::v_cycle(&levels, &rhs, &mut |b, x| coarse_solve(&multigrid, b, x));
    let wide: Vec<f64> = rhs.iter().map(|&v| f64::from(v)).collect();
    let mut expected = vec![0.0; csr.size()];
    multigrid.apply(&wide, &mut expected);
    let found: Vec<f64> = found.iter().map(|&v| f64::from(v)).collect();
    assert_eq!(
        found, expected,
        "{expect_levels} levels on the {n}^3 lattice"
    );
}

#[test]
fn the_f32_reference_cycle_is_the_single_precision_multigrid_bit_for_bit_at_three_levels() {
    assert_f32_reference_matches(12, 30, 3);
}

#[test]
fn the_f32_reference_cycle_is_the_single_precision_multigrid_bit_for_bit_at_four_levels() {
    assert_f32_reference_matches(24, 30, 4);
}
