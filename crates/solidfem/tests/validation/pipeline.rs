// A block meshed by this crate's own mesher and solved: the whole pipeline from a surface to a
// Timoshenko deflection, and the named refusal when a solid cannot become 10-node elements.

use solidfem::elastic::{Material, Model, Quadrature};
use solidfem::meshing::{quadratic_from_solid, MeshingError};
use solidfem::tetra::{RefineOptions, Refiner, SolidMesh, StepStatus, SurfaceMesher, HULL};

use crate::support::{faces_at, fix_root, mean_displacement, report_case, run};

/// A closed box surface with `n[axis]` cells along each axis and outward-wound triangles.
fn box_surface(size: [f64; 3], n: [usize; 3]) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut index = std::collections::BTreeMap::new();
    let mut vertex = |x: usize, y: usize, z: usize| -> u32 {
        *index.entry((x, y, z)).or_insert_with(|| {
            let at = |i: usize, axis: usize| size[axis] * i as f64 / n[axis] as f64;
            points.push([at(x, 0), at(y, 1), at(z, 2)]);
            (points.len() - 1) as u32
        })
    };
    let mut triangles = Vec::new();
    let mut quad = |a: u32, b: u32, c: u32, d: u32| {
        triangles.push([a, b, c]);
        triangles.push([a, c, d]);
    };
    for i in 0..n[0] {
        for j in 0..n[1] {
            quad(
                vertex(i, j, 0),
                vertex(i, j + 1, 0),
                vertex(i + 1, j + 1, 0),
                vertex(i + 1, j, 0),
            );
            quad(
                vertex(i, j, n[2]),
                vertex(i + 1, j, n[2]),
                vertex(i + 1, j + 1, n[2]),
                vertex(i, j + 1, n[2]),
            );
        }
        for k in 0..n[2] {
            quad(
                vertex(i, 0, k),
                vertex(i + 1, 0, k),
                vertex(i + 1, 0, k + 1),
                vertex(i, 0, k + 1),
            );
            quad(
                vertex(i, n[1], k),
                vertex(i, n[1], k + 1),
                vertex(i + 1, n[1], k + 1),
                vertex(i + 1, n[1], k),
            );
        }
    }
    for j in 0..n[1] {
        for k in 0..n[2] {
            quad(
                vertex(0, j, k),
                vertex(0, j, k + 1),
                vertex(0, j + 1, k + 1),
                vertex(0, j + 1, k),
            );
            quad(
                vertex(n[0], j, k),
                vertex(n[0], j + 1, k),
                vertex(n[0], j + 1, k + 1),
                vertex(n[0], j, k + 1),
            );
        }
    }
    (points, triangles)
}

fn meshed(size: [f64; 3], n: [usize; 3], element: f64) -> SolidMesh {
    let (points, triangles) = box_surface(size, n);
    let mut mesher = SurfaceMesher::new(points, triangles).expect("a valid surface");
    assert_eq!(mesher.step(usize::MAX, &mut || false), Ok(StepStatus::Done));
    let solid = mesher.result().expect("recovered");
    let options = RefineOptions::default();
    let mut refiner =
        Refiner::new(solid, Box::new(move |_| element), options).expect("a valid solid");
    assert_eq!(
        refiner.step(usize::MAX, &mut || false),
        Ok(StepStatus::Done)
    );
    refiner.result().expect("refined")
}

#[test]
fn a_cantilever_meshed_by_the_pipeline_matches_timoshenko() {
    // Timoshenko at L/h = 10, nu = 0: the same modelling gap as the structured validation, under
    // 0.5% (clamp warping and the uniform end traction, each a fraction of the 0.8% shear term).
    // Discretization: the structured case carries a few tenths of a percent at four cells through
    // the depth; this mesh has the same four surface cells through the depth with interior
    // elements of about that size but unstructured tetrahedra, which for quadratic elements
    // cost roughly twice the structured error at the same spacing, so under 1% with the gap.
    // Limit: 1.5%, the sum of those bounds rounded up, fixed before the first run.
    let (length, depth) = (10.0, 1.0);
    let material = Material {
        youngs_modulus: 1000.0,
        poisson_ratio: 0.0,
    };
    let solid = meshed([length, depth, depth], [40, 4, 4], 0.35);
    let mesh = quadratic_from_solid(&solid, Quadrature::FourPoint).expect("10-node mesh");
    let mut model = Model::new(mesh.clone(), material);
    fix_root(&mut model);
    let load = 1.0;
    for face in faces_at(&mesh, length) {
        model.add_traction(face, [0.0, 0.0, -load / (depth * depth)]);
    }
    let solution = run(model);
    let tip = -mean_displacement(&mesh, &solution, length, 2);
    let inertia = depth.powi(4) / 12.0;
    let area = depth * depth;
    let e = material.youngs_modulus;
    let g = e / 2.0;
    let expected =
        load * length.powi(3) / (3.0 * e * inertia) + load * length / (5.0 / 6.0 * g * area);
    let error = ((tip - expected) / expected).abs();
    report_case("pipeline_cantilever", error * 100.0, &solution);
    assert!(
        error < 0.015,
        "pipeline tip {tip}, expected {expected}, error {error}"
    );
}

#[test]
fn a_tetrahedron_the_solver_cannot_integrate_is_a_named_error() {
    // A tetrahedron with no volume: refused with the mesh's own named error, before assembly.
    let solid = SolidMesh {
        points: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        tets: vec![[0, 1, 2, 3]],
        neighbors: vec![[HULL; 4]],
    };
    let error = quadratic_from_solid(&solid, Quadrature::FourPoint).expect_err("flat");
    assert!(matches!(error, MeshingError::Mesh(_)), "got {error}");
}
