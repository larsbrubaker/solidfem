//! A deliberately wrong case through the production pipeline: a stubby cantilever solved for real
//! and judged against a reference made wrong by a factor of two must come out FAIL, and the right
//! reference must not, so a FAIL means something and a PASS is not automatic.

use solidfem::elastic::{Material, Solution};
use solidfem::mesh::QuadraticTetMesh;
use solidfem_bench::benchmarks::cases::{check_error, Case, Density, Sizing};
use solidfem_bench::benchmarks::pipeline::{face_area, run_case, DensityResult, Supports, Surface};
use solidfem_bench::benchmarks::verdict::{Verdict, LOAD_BALANCE_LIMIT};

const L: f64 = 12.0;
const W: f64 = 3.0;
const H: f64 = 3.0;
const E: f64 = 200_000.0;
const F: f64 = 30.0;

/// A closed box surface `L x W x H` with `n` cells along each edge of every face, wound outward.
fn box_surface(nx: usize, ny: usize, nz: usize) -> Surface {
    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut index = std::collections::HashMap::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut vertex = |i: usize, j: usize, k: usize, points: &mut Vec<[f64; 3]>| -> u32 {
        *index.entry((i, j, k)).or_insert_with(|| {
            points.push([
                L * i as f64 / nx as f64,
                W * j as f64 / ny as f64,
                H * k as f64 / nz as f64,
            ]);
            (points.len() - 1) as u32
        })
    };
    let quad = |a: u32, b: u32, c: u32, d: u32, triangles: &mut Vec<[u32; 3]>| {
        triangles.push([a, b, c]);
        triangles.push([a, c, d]);
    };
    for i in 0..nx {
        for j in 0..ny {
            let (a, b, c, d) = (
                vertex(i, j, 0, &mut points),
                vertex(i, j + 1, 0, &mut points),
                vertex(i + 1, j + 1, 0, &mut points),
                vertex(i + 1, j, 0, &mut points),
            );
            quad(a, b, c, d, &mut triangles);
            let (a, b, c, d) = (
                vertex(i, j, nz, &mut points),
                vertex(i + 1, j, nz, &mut points),
                vertex(i + 1, j + 1, nz, &mut points),
                vertex(i, j + 1, nz, &mut points),
            );
            quad(a, b, c, d, &mut triangles);
        }
    }
    for i in 0..nx {
        for k in 0..nz {
            let (a, b, c, d) = (
                vertex(i, 0, k, &mut points),
                vertex(i + 1, 0, k, &mut points),
                vertex(i + 1, 0, k + 1, &mut points),
                vertex(i, 0, k + 1, &mut points),
            );
            quad(a, b, c, d, &mut triangles);
            let (a, b, c, d) = (
                vertex(i, ny, k, &mut points),
                vertex(i, ny, k + 1, &mut points),
                vertex(i + 1, ny, k + 1, &mut points),
                vertex(i + 1, ny, k, &mut points),
            );
            quad(a, b, c, d, &mut triangles);
        }
    }
    for j in 0..ny {
        for k in 0..nz {
            let (a, b, c, d) = (
                vertex(0, j, k, &mut points),
                vertex(0, j, k + 1, &mut points),
                vertex(0, j + 1, k + 1, &mut points),
                vertex(0, j + 1, k, &mut points),
            );
            quad(a, b, c, d, &mut triangles);
            let (a, b, c, d) = (
                vertex(nx, j, k, &mut points),
                vertex(nx, j + 1, k, &mut points),
                vertex(nx, j + 1, k + 1, &mut points),
                vertex(nx, j, k + 1, &mut points),
            );
            quad(a, b, c, d, &mut triangles);
        }
    }
    (points, triangles)
}

fn apply(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
    let fixed = (0..mesh.nodes().len() as u32)
        .filter(|n| mesh.nodes()[*n as usize][0].abs() < 1e-9)
        .collect();
    let faces: Vec<_> = mesh
        .boundary_faces()
        .into_iter()
        .filter(|f| {
            f.nodes[..3]
                .iter()
                .all(|n| (mesh.nodes()[*n as usize][0] - L).abs() < 1e-9)
        })
        .collect();
    let area: f64 = faces.iter().map(|f| face_area(mesh, f)).sum();
    Ok(Supports {
        fixed,
        tractions: faces
            .into_iter()
            .map(|f| (f, [0.0, 0.0, -F / area]))
            .collect(),
        ..Supports::default()
    })
}

fn measure(mesh: &QuadraticTetMesh, solution: &Solution) -> Result<Vec<(String, f64)>, String> {
    let (mut sum, mut count) = (0.0, 0usize);
    for (node, p) in mesh.nodes().iter().enumerate() {
        if (p[0] - L).abs() < 1e-9 {
            sum -= solution.displacements[node][2];
            count += 1;
        }
    }
    Ok(vec![("tip".to_string(), sum / count as f64)])
}

/// The tip deflection a beam of this shape really has, shear included.
fn right_reference() -> f64 {
    let inertia = W * H * H * H / 12.0;
    let g = E / 2.0;
    F * L * L * L / (3.0 * E * inertia) + F * L / ((5.0 / 6.0) * g * W * H)
}

fn judge_against(rows: &[DensityResult], factor: f64) -> Verdict {
    let tip = rows.last().expect("one row").value("tip");
    // The stubby beam's tip is within a few percent of the Timoshenko value at this coarse mesh;
    // the 10% limit is wide enough for the right reference and far short of the wrong one.
    match check_error(tip, factor * right_reference(), 0.10, "tip deflection") {
        Ok(_) => Verdict::Pass,
        Err(reason) => Verdict::Fail(reason),
    }
}

fn case(judge: fn(&[DensityResult]) -> Verdict) -> Case {
    Case {
        id: "stubby-beam",
        title: "a small cantilever box",
        material: Material {
            youngs_modulus: E,
            poisson_ratio: 0.0,
        },
        densities: vec![Density {
            label: "only".to_string(),
            surface_file: String::new(),
            size: Sizing::Uniform(1.0),
            series_size: 1.0,
        }],
        layers: 0,
        apply,
        measure,
        iteration_budget: None,
        judge,
    }
}

fn run(judge: fn(&[DensityResult]) -> Verdict) -> (Verdict, Vec<DensityResult>) {
    let case = case(judge);
    let mut surface_of = |_: &Density| Ok(box_surface(8, 3, 3));
    let result = run_case(&case, &mut surface_of, 120.0, &mut |_: &str| {});
    (result.verdict, result.rows)
}

#[test]
fn a_wrong_reference_comes_out_as_fail_through_the_real_pipeline() {
    let (verdict, rows) = run(|rows| judge_against(rows, 2.0));
    assert!(matches!(verdict, Verdict::Fail(_)), "{verdict}");
    assert!(verdict.to_string().contains("tip deflection"), "{verdict}");
    // The run reached the judge, so the mesh solved and its loads balanced.
    assert!(rows[0].balance.error <= LOAD_BALANCE_LIMIT);
    assert!((rows[0].balance.applied[2] + F).abs() < 1e-9 * F);
}

#[test]
fn the_right_reference_passes_through_the_same_pipeline() {
    let (verdict, _) = run(|rows| judge_against(rows, 1.0));
    assert_eq!(verdict, Verdict::Pass);
}

#[test]
fn a_part_the_mesher_refuses_fails_by_name_instead_of_producing_a_number() {
    let case = case(|_| Verdict::Pass);
    let mut surface_of = |_: &Density| Ok((vec![[0.0, 0.0, 0.0]], vec![[0, 0, 0]]));
    let result = run_case(&case, &mut surface_of, 120.0, &mut |_: &str| {});
    match &result.verdict {
        Verdict::Fail(reason) => assert!(reason.contains("refused"), "{reason}"),
        other => panic!("expected FAIL, got {other}"),
    }
}

#[test]
fn a_support_on_a_loaded_face_is_a_named_refusal() {
    // Fixing the loaded end itself would send its load straight to the support unseen.
    let mut case = case(|_| Verdict::Pass);
    case.apply = |mesh| {
        let mut supports = apply(mesh)?;
        supports.fixed = (0..mesh.nodes().len() as u32)
            .filter(|n| (mesh.nodes()[*n as usize][0] - L).abs() < 1e-9)
            .collect();
        Ok(supports)
    };
    let mut surface_of = |_: &Density| Ok(box_surface(8, 3, 3));
    let result = run_case(&case, &mut surface_of, 120.0, &mut |_: &str| {});
    match &result.verdict {
        Verdict::Fail(reason) => assert!(reason.contains("fixed node"), "{reason}"),
        other => panic!("expected FAIL, got {other}"),
    }
}

#[test]
fn a_solve_past_its_iteration_budget_fails_by_name() {
    let mut case = case(|_| Verdict::Pass);
    case.iteration_budget = Some(2);
    let mut surface_of = |_: &Density| Ok(box_surface(8, 3, 3));
    let result = run_case(&case, &mut surface_of, 120.0, &mut |_: &str| {});
    match &result.verdict {
        Verdict::Fail(reason) => assert!(reason.contains("iteration budget"), "{reason}"),
        other => panic!("expected FAIL, got {other}"),
    }
}
