// What the recovered nodal stress reads against known answers: a uniform field and a linear field
// reproduced at every node (the one-sided boundary patches included), the Lamé hoop stress at the
// bore of a thick ring, and the Kirsch 3-times peak at a hole. The two curved cases print the
// signed error at each density on a `VALIDATION` line (run with --no-capture); the recovery's
// module header says what a peak read this way may under-read.

use solidfem::elastic::{
    von_mises, Material, Model, Solution, StaticStudy, StepOutcome, StudyOptions,
};
use solidfem::mesh::{box_tet_mesh, QuadraticTetMesh};

use crate::support::run;

const STEEL: Material = Material {
    youngs_modulus: 200_000.0,
    poisson_ratio: 0.3,
};

const CUBE_SIZE: f64 = 2.0;

/// A cube of 4 x 4 x 4 cells from the origin, its node positions kept exactly on their planes.
fn cube() -> QuadraticTetMesh {
    QuadraticTetMesh::from_linear(&box_tet_mesh([0.0; 3], [CUBE_SIZE; 3], [4, 4, 4]).expect("box"))
}

/// Every node on the cube's surface.
fn surface_nodes(mesh: &QuadraticTetMesh) -> Vec<u32> {
    let mut marked = vec![false; mesh.nodes().len()];
    for face in mesh.boundary_faces() {
        for node in face.nodes {
            marked[node as usize] = true;
        }
    }
    (0..mesh.nodes().len() as u32)
        .filter(|n| marked[*n as usize])
        .collect()
}

/// Solves the cube with `field` prescribed on its surface. A field inside the 10-node element's
/// displacement space, in equilibrium with no body force, is the exact solution.
fn solve_prescribed(
    field: impl Fn([f64; 3]) -> [f64; 3],
    material: Material,
) -> (Solution, Vec<[f64; 3]>) {
    let mesh = cube();
    let nodes = mesh.nodes().to_vec();
    let surface = surface_nodes(&mesh);
    let mut model = Model::new(mesh, material);
    for node in surface {
        let value = field(nodes[node as usize]);
        for (axis, component) in value.iter().enumerate() {
            model.prescribe_displacement(node, axis, *component);
        }
    }
    (run(model), nodes)
}

#[test]
fn a_uniform_stress_is_reproduced_at_every_node() {
    // A uniform strain state, with a shear, inside the element's displacement space. The solver's
    // relative residual is 1e-9; on a cube of this size the displacement error it leaves is
    // bounded near 1e-8 of the stress scale, so 1e-6 is a limit on recovery, not on the solver.
    let (solution, _) = solve_prescribed(
        |p| {
            [
                1e-3 * p[0] + 2e-4 * p[1],
                -3e-4 * p[1],
                5e-4 * p[2] + 1e-4 * p[0],
            ]
        },
        STEEL,
    );
    let reference = solution.nodal_stress[0];
    let scale = von_mises(&reference);
    assert!(scale > 0.0);
    for (node, stress) in solution.nodal_stress.iter().enumerate() {
        for k in 0..6 {
            assert!(
                (stress[k] - reference[k]).abs() < 1e-6 * scale,
                "node {node}, component {k}: {stress:?} against {reference:?}"
            );
        }
    }
}

/// Pure bending about y, nu = 0: `sigma_xx = E kappa z` and no other stress, linear through the
/// cube; a quadratic displacement field, so it is inside the element and is the exact solution.
#[test]
fn a_linear_stress_field_is_reproduced_at_boundary_nodes() {
    let kappa = 1e-3;
    let material = Material {
        youngs_modulus: 200_000.0,
        poisson_ratio: 0.0,
    };
    let (solution, nodes) = solve_prescribed(
        |p| [kappa * p[0] * p[2], 0.0, -0.5 * kappa * p[0] * p[0]],
        material,
    );
    let scale = material.youngs_modulus * kappa * CUBE_SIZE;
    // Corners, edges and faces all sit on one-sided patches here, a box corner on a patch of a
    // single element. A linear field is in the fit's own space, so what is left is the solver's
    // residual, bounded as in the uniform case.
    for (node, stress) in solution.nodal_stress.iter().enumerate() {
        let expected = material.youngs_modulus * kappa * nodes[node][2];
        assert!(
            (stress[0] - expected).abs() < 1e-6 * scale,
            "node {node}: sigma_xx {} against {expected}",
            stress[0]
        );
        for k in 1..6 {
            assert!(
                stress[k].abs() < 1e-6 * scale,
                "node {node}, component {k}: {stress:?}"
            );
        }
    }
}

/// A quarter annulus from `a` to `b`, thickness `t`, `nr` x `nt` cells with one through the
/// thickness, the radial spacing graded so the cells stay about square.
fn quarter_ring(a: f64, b: f64, t: f64, nr: usize, nt: usize) -> QuadraticTetMesh {
    let unit = box_tet_mesh([0.0; 3], [1.0, 1.0, 1.0], [nr, nt, 1]).expect("box");
    let ratio = b / a;
    let mapped = unit
        .with_mapped_positions(|p| {
            let radius = a * ratio.powf(p[0]);
            let angle = 0.5 * std::f64::consts::PI * p[1];
            [radius * angle.cos(), radius * angle.sin(), t * p[2]]
        })
        .expect("a positive mapping");
    QuadraticTetMesh::from_linear(&mapped)
}

/// Holds a quarter annulus on its two symmetry planes and against rigid motion along z.
fn hold_quarter(model: &mut Model<Material>, nodes: &[[f64; 3]]) {
    for (node, p) in nodes.iter().enumerate() {
        if p[1].abs() < 1e-9 {
            model.prescribe_displacement(node as u32, 1, 0.0);
        }
        if p[0].abs() < 1e-9 {
            model.prescribe_displacement(node as u32, 0, 0.0);
        }
        if p[2].abs() < 1e-9 {
            model.prescribe_displacement(node as u32, 2, 0.0);
        }
    }
}

/// The face centroid of a boundary face's three corners.
fn face_centre(mesh: &QuadraticTetMesh, face: &solidfem::mesh::BoundaryFace) -> [f64; 3] {
    let n = face.nodes[..3]
        .iter()
        .map(|i| mesh.nodes()[*i as usize])
        .collect::<Vec<_>>();
    [
        (n[0][0] + n[1][0] + n[2][0]) / 3.0,
        (n[0][1] + n[1][1] + n[2][1]) / 3.0,
        (n[0][2] + n[1][2] + n[2][2]) / 3.0,
    ]
}

/// Whether all three corners of a face lie on the cylinder of radius `radius`: true of the bore's
/// and the outer arc's faces, and of no face on a symmetry plane or end.
fn corners_at_radius(
    mesh: &QuadraticTetMesh,
    face: &solidfem::mesh::BoundaryFace,
    radius: f64,
) -> bool {
    face.nodes[..3].iter().all(|i| {
        let p = mesh.nodes()[*i as usize];
        ((p[0] * p[0] + p[1] * p[1]).sqrt() - radius).abs() < 1e-9
    })
}

fn polar_stress(s: &[f64; 6], p: [f64; 3]) -> (f64, f64) {
    let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
    let (c, n) = (p[0] / r, p[1] / r);
    let radial = c * c * s[0] + n * n * s[1] + 2.0 * c * n * s[3];
    let hoop = n * n * s[0] + c * c * s[1] - 2.0 * c * n * s[3];
    (radial, hoop)
}

/// Bore radius of the ring.
const RING_A: f64 = 5.0;
/// Outer radius of the ring.
const RING_B: f64 = 10.0;
/// Internal pressure on the ring.
const RING_P: f64 = 10.0;
/// Ring thickness: a free-ended tube's Lamé solution holds at any height.
const RING_T: f64 = 2.0;

/// Solves a quarter ring under internal pressure and returns the worst signed percentage error of
/// the recovered hoop stress against Lamé over the bore nodes, the error of largest size, and the
/// signed percentage error of the reported von Mises peak over the body.
fn ring_hoop_error(nr: usize, nt: usize) -> (f64, f64) {
    let mesh = quarter_ring(RING_A, RING_B, RING_T, nr, nt);
    let nodes = mesh.nodes().to_vec();
    let mut model = Model::new(mesh.clone(), STEEL);
    hold_quarter(&mut model, &nodes);
    for face in mesh.boundary_faces() {
        if !corners_at_radius(&mesh, &face, RING_A) {
            continue;
        }
        let c = face_centre(&mesh, &face);
        let r = (c[0] * c[0] + c[1] * c[1]).sqrt();
        // A pressure pushes outward from the axis into the material.
        model.add_traction(face, [RING_P * c[0] / r, RING_P * c[1] / r, 0.0]);
    }
    let solution = run(model);
    let k = RING_P * RING_A * RING_A / (RING_B * RING_B - RING_A * RING_A);
    let hoop_at = |r: f64| k * (1.0 + RING_B * RING_B / (r * r));
    let mut worst = 0.0f64;
    let (mut imin, mut imax, mut peak) = (f64::MAX, f64::MIN, f64::MIN);
    for (node, p) in nodes.iter().enumerate() {
        let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
        if r > RING_A + 1e-6 {
            continue;
        }
        let (_, hoop) = polar_stress(&solution.nodal_stress[node], *p);
        let error = 100.0 * (hoop - hoop_at(r)) / hoop_at(r);
        if error.abs() > worst.abs() {
            worst = error;
        }
        peak = peak.max(hoop);
        let angle = p[1].atan2(p[0]).to_degrees();
        if (20.0..=70.0).contains(&angle) && p[2] > 0.25 * RING_T && p[2] < 0.75 * RING_T {
            imin = imin.min(error);
            imax = imax.max(error);
        }
    }
    let reported = solution
        .peak_von_mises(&mesh, |_| true)
        .expect("a mesh has nodes");
    let (bore_hoop, radial) = (hoop_at(RING_A), -RING_P);
    let exact = (bore_hoop * bore_hoop - bore_hoop * radial + radial * radial).sqrt();
    let peak_error = 100.0 * (reported - exact) / exact;
    println!(
        "VALIDATION ring nt={nt}: reported von Mises peak {reported:.4} against Lame {exact:.4}, {peak_error:+.3}%"
    );
    println!(
        "VALIDATION ring nt={nt}: bore nodes away from the edges err [{imin:+.3}%, {imax:+.3}%], peak hoop {:+.3}% against Lame",
        100.0 * (peak - hoop_at(RING_A)) / hoop_at(RING_A)
    );
    (worst, peak_error)
}

#[test]
fn a_thick_ring_reads_the_lame_hoop_stress_at_the_bore() {
    // The limit is the ring's bore budget: at these sizes a bore chord sags by h^2 / (8 a), under
    // 0.3% of the radius, and hoop stress depends on the square of the radius, under 0.6%. 10-node
    // elements across the bore resolve the 1/r^2 field at about 2.4% by the same reasoning as the
    // plate's discretization allowance, so 3% total.
    let (mut last, mut peak_error) = (0.0, 0.0);
    for (nr, nt) in [(6, 12), (9, 18), (12, 24)] {
        (last, peak_error) = ring_hoop_error(nr, nt);
        println!("VALIDATION ring nr={nr} nt={nt}: worst bore hoop error {last:+.3}%");
    }
    assert!(last.abs() <= 3.0, "worst bore hoop error {last:.3}%");
    // The reported von Mises peak over the body, against the exact peak at the bore, under the
    // same budget as the hoop stress.
    assert!(
        peak_error.abs() <= 3.0,
        "reported peak error {peak_error:.3}%"
    );
}

/// Hole radius.
const HOLE_A: f64 = 5.0;
/// Remote stress.
const HOLE_SIGMA: f64 = 100.0;

/// The Kirsch stresses at `(r, cos theta, sin theta)`, in polar components, for uniaxial tension
/// along x at infinity.
fn kirsch(r: f64, c: f64, n: f64) -> (f64, f64, f64) {
    let a2 = (HOLE_A / r).powi(2);
    let a4 = a2 * a2;
    let c2 = c * c - n * n;
    let s2 = 2.0 * c * n;
    let h = 0.5 * HOLE_SIGMA;
    (
        h * (1.0 - a2) + h * (1.0 - 4.0 * a2 + 3.0 * a4) * c2,
        h * (1.0 + a2) - h * (1.0 + 3.0 * a4) * c2,
        -h * (1.0 + 2.0 * a2 - 3.0 * a4) * s2,
    )
}

/// Solves a quarter annulus, thickness giving plane stress with nu = 0, with the exact Kirsch
/// traction on its outer arc, and returns the peak recovered von Mises over all nodes and the
/// exact 3 sigma.
fn hole_peak(nr: usize, nt: usize) -> (f64, f64) {
    let outer = 20.0;
    let mesh = quarter_ring(HOLE_A, outer, 0.5, nr, nt);
    let nodes = mesh.nodes().to_vec();
    let material = Material {
        youngs_modulus: 200_000.0,
        poisson_ratio: 0.0,
    };
    let mut model = Model::new(mesh.clone(), material);
    hold_quarter(&mut model, &nodes);
    for face in mesh.boundary_faces() {
        if !corners_at_radius(&mesh, &face, outer) {
            continue;
        }
        let m = face_centre(&mesh, &face);
        let r = (m[0] * m[0] + m[1] * m[1]).sqrt();
        let (c, n) = (m[0] / r, m[1] / r);
        let (radial, _, shear) = kirsch(r, c, n);
        // The traction on a face of outward normal e_r: sigma_rr e_r + sigma_rt e_t.
        let traction = [radial * c - shear * n, radial * n + shear * c, 0.0];
        model.add_traction(face, traction);
    }
    let solution = run(model);
    let peak = solution
        .peak_von_mises(&mesh, |_| true)
        .expect("a mesh has nodes");
    (peak, 3.0 * HOLE_SIGMA)
}

#[test]
fn the_peak_at_a_hole_reads_the_kirsch_value() {
    // The exact traction and nu = 0 make the 3D problem exactly plane stress, with no
    // finite-width term. The limit is the same 3% the plate benchmark derives: 0.5% for per-face
    // traction and the chord of the hole (the sag h^2 / (8 a) is under 0.2% of the radius at
    // these sizes), the rest for discretization of a stress concentration by 10-node elements.
    let mut last = (0.0, 0.0);
    for (nr, nt) in [(9, 12), (15, 20), (22, 30)] {
        let (peak, exact) = hole_peak(nr, nt);
        last = (peak, exact);
        println!(
            "VALIDATION hole nr={nr} nt={nt}: peak {peak:.4}, exact {exact:.4}, error {:+.3}%",
            100.0 * (peak - exact) / exact
        );
    }
    let error = (last.0 - last.1) / last.1;
    assert!(error.abs() <= 0.03, "peak error {:.3}%", 100.0 * error);
}

#[test]
fn a_recovery_run_in_slices_gives_the_same_bits() {
    // A host with no threads recovers in slices; the patch fits must not change with the slicing.
    let mesh = QuadraticTetMesh::from_linear(
        &box_tet_mesh([0.0; 3], [2.0, 1.0, 1.0], [6, 3, 3]).expect("box"),
    );
    let nodes = mesh.nodes().to_vec();
    let mut model = Model::new(mesh, STEEL);
    for (node, p) in nodes.iter().enumerate() {
        if p[0] == 0.0 {
            model.fix_node(node as u32);
        }
        if p[0] == 2.0 {
            model.add_nodal_force(node as u32, [0.0, 0.0, -1.0]);
        }
    }
    let mut study =
        StaticStudy::new(model, StudyOptions::default(), &mut || false).expect("a valid model");
    while !matches!(
        study.step(1000, &mut || false).expect("the solve succeeds"),
        StepOutcome::Converged(_)
    ) {}
    let whole = study.solution(&mut || false).expect("a converged solve");
    let mut recovery = study.begin_recovery();
    while !study
        .continue_recovery(&mut recovery, 7, &mut || false)
        .expect("the slice succeeds")
    {}
    let sliced = study
        .finish_recovery(recovery)
        .expect("a finished recovery");
    assert_eq!(sliced.nodal_stress.len(), whole.nodal_stress.len());
    for (node, (a, b)) in sliced
        .nodal_stress
        .iter()
        .zip(&whole.nodal_stress)
        .enumerate()
    {
        assert_eq!(
            a.map(f64::to_bits),
            b.map(f64::to_bits),
            "node {node}: {a:?} against {b:?}"
        );
    }
}

/// A bar half of one material, half of a stiffer one, in tension with nu = 0: the stress is `F / A`
/// in both halves (no lateral contraction to mismatch at the interface), and the patches there hold
/// two materials, so they fall back to the worst incident element, which must read each element's
/// own stress. The solver tolerance with this contrast bounds the error near 1e-6 of the stress, so
/// 1e-4 is a limit on the recovery.
#[test]
fn a_stiffness_jump_in_tension_reads_the_uniform_stress_at_the_interface() {
    let mesh = QuadraticTetMesh::from_linear(
        &box_tet_mesh([0.0; 3], [4.0, 1.0, 1.0], [8, 2, 2]).expect("box"),
    );
    let nodes = mesh.nodes().to_vec();
    let boundary = mesh.boundary_faces();
    let field = |p: [f64; 3]| Material {
        youngs_modulus: if p[0] < 2.0 { 1_000.0 } else { 5_000.0 },
        poisson_ratio: 0.0,
    };
    let mut model = Model::new(mesh, field);
    // Held against rigid motion without restraining the (zero) lateral strain.
    for (node, p) in nodes.iter().enumerate() {
        if p[0] == 0.0 {
            model.prescribe_displacement(node as u32, 0, 0.0);
        }
    }
    for (node, p) in nodes.iter().enumerate() {
        if p[0] == 0.0 && p[2] == 0.0 {
            model.prescribe_displacement(node as u32, 2, 0.0);
        }
        if p[0] == 0.0 && p[1] == 0.0 {
            model.prescribe_displacement(node as u32, 1, 0.0);
        }
    }
    let force = 100.0;
    for face in boundary {
        if face.nodes[..3].iter().all(|n| nodes[*n as usize][0] == 4.0) {
            model.add_traction(face, [force, 0.0, 0.0]);
        }
    }
    let solution = run(model);
    for (node, p) in nodes.iter().enumerate() {
        if (p[0] - 2.0).abs() < 1e-9 {
            let value = solution.nodal_von_mises[node];
            assert!(
                ((value - force) / force).abs() < 1e-4,
                "interface node {node}: {value} against {force}"
            );
        }
    }
}
