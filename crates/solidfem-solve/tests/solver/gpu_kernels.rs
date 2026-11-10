// The GPU smoother kernels without a GPU: the WGSL parses and validates (a shader that does not
// compile fails here and not only on a device), its uniform struct is byte for byte the Rust
// record, the packed layout reads an assembled stiffness-like matrix with unsymmetric blocks, the
// f64 reference is the multigrid's own smoother bit for bit, and the f32 reference agrees with f64
// within its derived rounding bound.

use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{Module, ShaderStage, TypeInner};
use solidfem_solve::gpu::reference::{self, SmootherVectors};
use solidfem_solve::gpu::{
    dispatch_size, BlockMatrix, ChebyshevCoefficients, GpuError, GpuParams, MAX_BLOCK,
    PARAMS_STRIDE, SMOOTHER_WGSL, WORKGROUP_SIZE,
};
use solidfem_solve::{
    CsrMatrix, MultigridOptions, MultigridSetup, Precision, Preconditioner, SetupStatus,
    NEAR_NULL_MODES,
};

/// A lattice of `n^3` nodes whose 3x3 blocks are dense and not symmetric, while the matrix is
/// symmetric positive definite: every pair of nodes `a`, `b` in each unit cell adds `G^T G` for
/// `G = [U, -V]`, so block `(a, b)` gets `-U^T V` and block `(b, a)` its transpose. `U` and `V`
/// are an axial spring along the pair's offset plus two different fixed twists, so `U^T V` is not
/// symmetric and a kernel that reads a block transposed gets a different answer. A light ground
/// term on the diagonal makes the matrix definite; the stiffness varies cell to cell, so no two
/// rows are alike. An interior node couples to 26 neighbours, as in a 10-node tetrahedral mesh.
pub(super) fn spring_lattice(n: usize) -> CsrMatrix {
    let index = |x: usize, y: usize, z: usize| (x + n * (y + n * z)) as u32;
    let mut cells: Vec<[u32; 8]> = Vec::new();
    for z in 0..n - 1 {
        for y in 0..n - 1 {
            for x in 0..n - 1 {
                let mut cell = [0u32; 8];
                for (corner, slot) in cell.iter_mut().enumerate() {
                    *slot = index(x + (corner & 1), y + ((corner >> 1) & 1), z + (corner >> 2));
                }
                cells.push(cell);
            }
        }
    }
    let nodes = n * n * n;
    let mut matrix = CsrMatrix::from_block_connectivity(nodes, 3, cells.iter().map(|c| &c[..]));
    let position = |node: u32| {
        let node = node as usize;
        [
            (node % n) as f64,
            ((node / n) % n) as f64,
            (node / (n * n)) as f64,
        ]
    };
    for (number, cell) in cells.iter().enumerate() {
        let stiffness = 1.0 + (number % 7) as f64 * 0.25;
        for (i, &a) in cell.iter().enumerate() {
            for &b in &cell[i + 1..] {
                let (pa, pb) = (position(a), position(b));
                let offset = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                let length2 = offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2];
                let mut u = [[0.0f64; 3]; 3];
                let mut v = [[0.0f64; 3]; 3];
                for p in 0..3 {
                    for q in 0..3 {
                        let axial = stiffness * offset[p] * offset[q] / length2;
                        u[p][q] = axial + ((p * 3 + q + number) % 5) as f64 * 0.04 - 0.08;
                        v[p][q] = axial + ((p + 2 * q + number + 1) % 7) as f64 * 0.03 - 0.09;
                    }
                }
                let (a, b) = (a as usize, b as usize);
                for p in 0..3 {
                    for q in 0..3 {
                        let (mut aa, mut bb, mut ab) = (0.0, 0.0, 0.0);
                        for k in 0..3 {
                            aa += u[k][p] * u[k][q];
                            bb += v[k][p] * v[k][q];
                            ab += u[k][p] * v[k][q];
                        }
                        matrix.add(3 * a + p, (3 * a + q) as u32, aa);
                        matrix.add(3 * b + p, (3 * b + q) as u32, bb);
                        matrix.add(3 * a + p, (3 * b + q) as u32, -ab);
                        // Block (b, a) is the transpose of block (a, b): entry (q, p) there.
                        matrix.add(3 * b + q, (3 * a + p) as u32, -ab);
                    }
                }
            }
        }
    }
    for row in 0..3 * nodes {
        matrix.add(row, row as u32, 0.05);
    }
    matrix
}

/// Two block rows of 6x3 blocks over three block columns, as a prolongator's transpose is shaped:
/// `(starts, columns, values)`, every value different.
pub(super) fn six_by_three() -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    let values = (0..3 * 18).map(|v| v as f64 * 0.75 - 20.0).collect();
    (vec![0, 2, 3], vec![0, 2, 1], values)
}

/// A right-hand side with every entry different and of both signs.
pub(super) fn varied_vector(length: usize) -> Vec<f64> {
    (0..length)
        .map(|i| ((i * 37 % 101) as f64 - 50.0) / 25.0)
        .collect()
}

/// Smoother scalars for the comparisons, which check arithmetic and not convergence, so any valid
/// interval serves; this one has the default range of thirty.
pub(super) fn coefficients(degree: usize) -> ChebyshevCoefficients {
    let lambda_max = 4.0;
    ChebyshevCoefficients::new(lambda_max / 30.0, lambda_max, degree).expect("valid bounds")
}

fn parsed() -> Module {
    naga::front::wgsl::parse_str(SMOOTHER_WGSL).expect("the shader parses as WGSL")
}

#[test]
fn the_smoother_shader_validates_with_its_three_entry_points() {
    let module = parsed();
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
            ("multiply_main", size),
            ("chebyshev_start_main", size),
            ("chebyshev_update_main", size)
        ]
    );
}

#[test]
fn the_params_struct_is_the_rust_record_byte_for_byte() {
    let module = parsed();
    let (_, params) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("Params"))
        .expect("the shader has a Params struct");
    let TypeInner::Struct { members, span } = &params.inner else {
        panic!("Params is not a struct");
    };
    let fields: Vec<(&str, u32)> = members
        .iter()
        .map(|m| (m.name.as_deref().unwrap_or(""), m.offset))
        .collect();
    assert_eq!(
        fields,
        [
            ("rows", 0),
            ("row_block", 4),
            ("col_block", 8),
            ("subtract", 12),
            ("theta", 16),
            ("direction_scale", 20),
            ("scaled_scale", 24),
            ("pad", 28),
        ]
    );
    assert_eq!(*span as usize, PARAMS_STRIDE);
    let record = GpuParams {
        rows: 7,
        row_block: 3,
        col_block: 6,
        subtract: 1,
        theta: 1.5,
        direction_scale: -0.25,
        scaled_scale: 2.0,
        pad: 0,
    };
    let bytes = record.to_le_bytes();
    assert_eq!(&bytes[8..12], &6u32.to_le_bytes());
    assert_eq!(&bytes[20..24], &(-0.25f32).to_le_bytes());
    let workgroup_constant = module
        .constants
        .iter()
        .find(|(_, c)| c.name.as_deref() == Some("WORKGROUP_SIZE"))
        .map(|(_, c)| module.global_expressions[c.init].clone());
    assert_eq!(
        workgroup_constant,
        Some(naga::Expression::Literal(naga::Literal::U32(
            WORKGROUP_SIZE
        )))
    );
}

#[test]
fn dispatches_stay_within_webgpu_workgroup_counts() {
    assert_eq!(dispatch_size(1), [1, 1]);
    assert_eq!(dispatch_size(64 * 65_535), [65_535, 1]);
    for threads in [64 * 65_535 + 1, 3_000_000_000u32, u32::MAX] {
        let [x, y] = dispatch_size(threads);
        assert!(x <= 65_535 && y <= 65_535, "{threads} threads: {x} x {y}");
        assert!(
            u64::from(x) * u64::from(y) * u64::from(WORKGROUP_SIZE) >= u64::from(threads),
            "{threads} threads are covered"
        );
    }
}

#[test]
fn the_layout_refuses_what_the_kernels_cannot_run() {
    let too_wide = BlockMatrix::<f32>::from_parts(1, MAX_BLOCK + 1, 1, &[0, 0], &[], &[]);
    assert_eq!(
        too_wide,
        Err(GpuError::BlockSize {
            row_block: MAX_BLOCK + 1,
            col_block: 1
        })
    );
    let outside = BlockMatrix::<f32>::from_parts(1, 1, 1, &[0, 1], &[1], &[1.0]);
    assert_eq!(outside, Err(GpuError::Structure { row: 0 }));
    let missing_diagonal =
        BlockMatrix::<f64>::from_parts(2, 1, 1, &[0, 1, 2], &[1, 0], &[1.0, 1.0]).unwrap();
    assert_eq!(
        reference::inverse_diagonal(&missing_diagonal),
        Err(GpuError::NotPositiveDefinite { row: 0, pivot: 0.0 })
    );
    assert!(ChebyshevCoefficients::new(2.0, 1.0, 3).is_err());
}

#[test]
fn a_rectangular_six_by_three_product_reads_its_blocks_row_major() {
    let (starts, columns, values) = six_by_three();
    let matrix = BlockMatrix::<f64>::from_parts(3, 6, 3, &starts, &columns, &values).unwrap();
    let x: Vec<f64> = (0..9).map(|v| v as f64 * 0.5 + 1.0).collect();
    let mut y = vec![0.0; 12];
    reference::multiply(&matrix, &x, &mut y);
    for (row, &found) in y.iter().enumerate() {
        let (block_row, p) = (row / 6, row % 6);
        let mut expected = 0.0;
        for at in starts[block_row]..starts[block_row + 1] {
            for q in 0..3 {
                expected += values[at * 18 + p * 3 + q] * x[columns[at] as usize * 3 + q];
            }
        }
        assert_eq!(found, expected, "row {row}");
    }
}

#[test]
fn the_nodal_layout_multiplies_exactly_as_the_assembled_matrix() {
    let csr = spring_lattice(5);
    let matrix = BlockMatrix::<f64>::from_nodal(&csr).unwrap();
    let x = varied_vector(csr.size());
    let (mut block_y, mut csr_y) = (vec![0.0; csr.size()], vec![0.0; csr.size()]);
    reference::multiply(&matrix, &x, &mut block_y);
    csr.multiply(&x, &mut csr_y);
    // Both sum each row's stored entries in column order from zero: the same bits.
    assert_eq!(block_y, csr_y);
    // The off-diagonal blocks are not symmetric, so a transposed read of a block shows.
    let unsymmetric = (0..matrix.columns().len())
        .filter(|&at| {
            let block = &matrix.values()[9 * at..9 * at + 9];
            (0..3).any(|p| (0..3).any(|q| block[3 * p + q] != block[3 * q + p]))
        })
        .count();
    assert!(
        unsymmetric * 2 > matrix.columns().len(),
        "{unsymmetric} of {} blocks are not symmetric",
        matrix.columns().len()
    );
}

#[test]
fn the_f64_reference_is_the_multigrid_smoother_bit_for_bit() {
    // 11^3 nodes, 3,993 unknowns: above the dense factorization limit, so a one-level hierarchy
    // applies its Chebyshev smoother alone.
    let n = 11;
    let csr = spring_lattice(n);
    let positions: Vec<[f64; 3]> = (0..n * n * n)
        .map(|i| [(i % n) as f64, ((i / n) % n) as f64, (i / (n * n)) as f64])
        .collect();
    let modes: Vec<f64> = (0..csr.size() * NEAR_NULL_MODES)
        .map(|i| (i % 11) as f64 - 5.0)
        .collect();
    let options = MultigridOptions {
        max_levels: 1,
        precision: Precision::Double,
        ..MultigridOptions::default()
    };
    let mut setup = MultigridSetup::new(&csr, &modes, &positions, options).unwrap();
    while setup.step(usize::MAX, &mut || false).unwrap() != SetupStatus::Finished {}
    let multigrid = setup.finish().unwrap();
    assert_eq!(multigrid.stats().levels, 1);
    let rhs = varied_vector(csr.size());
    let mut expected = vec![0.0; csr.size()];
    multigrid.apply(&rhs, &mut expected);

    let (lambda_min, lambda_max) = multigrid.smoother_spectrum(0).unwrap();
    let coefficients =
        ChebyshevCoefficients::new(lambda_min, lambda_max, options.smoother_degree).unwrap();
    let matrix = BlockMatrix::<f64>::from_nodal(&csr).unwrap();
    let diagonal = reference::inverse_diagonal(&matrix).unwrap();
    let mut found = SmootherVectors::<f64>::zeros(csr.size());
    reference::chebyshev_smooth(&matrix, &diagonal, &coefficients, &rhs, &mut found);
    assert_eq!(found.solution, expected);
}

#[test]
fn the_f32_reference_agrees_with_f64_to_single_precision() {
    // The f64 run on the same f32-rounded inputs is the exact result to f32's resolution; the f32
    // run may differ from it only by its own rounding, which the bound carries entry by entry.
    let csr = spring_lattice(6);
    let wide = BlockMatrix::<f64>::from_nodal(&csr).unwrap();
    let narrow = wide.convert::<f32>();
    let diagonal = reference::inverse_diagonal(&wide).unwrap();
    let narrow_diagonal: Vec<f32> = diagonal.iter().map(|&v| v as f32).collect();
    let rhs = varied_vector(csr.size());
    let narrow_rhs: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
    let coefficients = coefficients(3);

    let mut narrow_vectors = SmootherVectors::<f32>::zeros(csr.size());
    reference::chebyshev_smooth(
        &narrow,
        &narrow_diagonal,
        &coefficients,
        &narrow_rhs,
        &mut narrow_vectors,
    );
    let bound =
        reference::smoother_rounding_bound(&narrow, &narrow_diagonal, &coefficients, &narrow_rhs);
    let mut on_f32_inputs = SmootherVectors::<f64>::zeros(csr.size());
    let rounded_wide = narrow.convert::<f64>();
    let rounded_diagonal: Vec<f64> = narrow_diagonal.iter().map(|&v| f64::from(v)).collect();
    let rounded_rhs: Vec<f64> = narrow_rhs.iter().map(|&v| f64::from(v)).collect();
    let rounded_coefficients = ChebyshevCoefficients {
        theta: f64::from(coefficients.theta as f32),
        steps: coefficients
            .steps
            .iter()
            .map(|&(a, b)| (f64::from(a as f32), f64::from(b as f32)))
            .collect(),
    };
    reference::chebyshev_smooth(
        &rounded_wide,
        &rounded_diagonal,
        &rounded_coefficients,
        &rounded_rhs,
        &mut on_f32_inputs,
    );
    for (i, ((exact, found), bound)) in on_f32_inputs
        .solution
        .iter()
        .zip(&narrow_vectors.solution)
        .zip(&bound)
        .enumerate()
    {
        assert!(
            (exact - f64::from(*found)).abs() <= *bound,
            "entry {i}: f32 {found} against {exact}, bound {bound}"
        );
    }
}
