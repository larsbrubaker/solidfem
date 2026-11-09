//! Measures the preconditioners on the two problem classes that matter: a stocky cantilever and a
//! thin plate. `solidfem-preconditioner <cantilever|plate> <jacobi|multigrid> <nx> <ny> <nz>
//! <cap-seconds>` builds a structured box of 10-node elements, clamps x = 0, loads the x = L face
//! and prints the free unknowns, iterations, setup and solve seconds, operator complexity and
//! hierarchy memory. Run in release; a capped solve reports the residual it reached.

use std::time::Instant;

use solidfem::elastic::{
    Clock, Material, Model, MultigridOptions, PreconditionerChoice, StaticStudy, StepOutcome,
    StudyOptions,
};
use solidfem::mesh::{box_tet_mesh, QuadraticTetMesh};

const ON_PLANE: f64 = 1e-9;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let parsed = (|| {
        let kind = args.first()?.clone();
        let choice = args.get(1)?.clone();
        let divisions = [
            args.get(2)?.parse::<usize>().ok()?,
            args.get(3)?.parse::<usize>().ok()?,
            args.get(4)?.parse::<usize>().ok()?,
        ];
        let cap = args.get(5)?.parse::<f64>().ok()?;
        Some((kind, choice, divisions, cap))
    })();
    let Some((kind, choice, divisions, cap)) = parsed else {
        eprintln!(
            "usage: solidfem-preconditioner <cantilever|plate> <jacobi|multigrid> <nx> <ny> <nz> <cap-seconds>"
        );
        std::process::exit(2);
    };
    let (size, nu) = match kind.as_str() {
        "cantilever" => ([10.0, 1.0, 1.0], 0.3),
        "plate" => ([100.0, 100.0, 0.8], 0.3),
        other => {
            eprintln!("unknown problem {other}");
            std::process::exit(2);
        }
    };
    let linear = box_tet_mesh([0.0; 3], size, divisions).expect("a valid box");
    let mesh = QuadraticTetMesh::from_linear(&linear);
    let material = Material {
        youngs_modulus: 200_000.0,
        poisson_ratio: nu,
    };
    let mut model = Model::new(mesh.clone(), material);
    for node in 0..mesh.nodes().len() as u32 {
        if mesh.nodes()[node as usize][0].abs() < ON_PLANE {
            model.fix_node(node);
        }
    }
    for face in mesh.boundary_faces() {
        let at_end = face.nodes[..3]
            .iter()
            .all(|n| (mesh.nodes()[*n as usize][0] - size[0]).abs() < ON_PLANE);
        if at_end {
            model.add_traction(face, [0.0, 0.0, -1.0]);
        }
    }
    let preconditioner = match choice.as_str() {
        "jacobi" => PreconditionerChoice::Jacobi,
        "multigrid" => {
            let env_or = |name: &str, default: f64| {
                std::env::var(name)
                    .ok()
                    .and_then(|v| v.parse::<f64>().ok())
                    .unwrap_or(default)
            };
            let mut options = MultigridOptions::default();
            options.strength_threshold = env_or("MG_THETA", options.strength_threshold);
            options.smoother_range = env_or("MG_RANGE", options.smoother_range);
            options.smoother_degree = env_or("MG_DEGREE", options.smoother_degree as f64) as usize;
            PreconditionerChoice::Multigrid(options)
        }
        other => {
            eprintln!("unknown preconditioner {other}");
            std::process::exit(2);
        }
    };
    let start = Instant::now();
    let clock: Clock = Box::new(move || start.elapsed().as_secs_f64());
    let mut study = StaticStudy::new(
        model,
        StudyOptions {
            clock: Some(clock),
            preconditioner,
            ..StudyOptions::default()
        },
        &mut || false,
    )
    .expect("the study builds");
    let solve_start = Instant::now();
    let mut converged = false;
    loop {
        match study.step(200, &mut || false).expect("the solve runs") {
            StepOutcome::Converged(_) => {
                converged = true;
                break;
            }
            StepOutcome::Running(_) => {}
            StepOutcome::Cancelled(_) => unreachable!("nothing cancels this solve"),
        }
        if solve_start.elapsed().as_secs_f64() >= cap {
            break;
        }
    }
    let r = study.report();
    let (levels, complexity, memory) = r.multigrid.map_or((0, 0.0, 0.0), |m| {
        (m.levels, m.operator_complexity, m.memory_bytes as f64 / 1e6)
    });
    println!(
        "{kind} {choice} {divisions:?}: free unknowns {}, nonzeros {}, converged {converged}, iterations {}, residual {:.3e}, assembly {:.3} s, setup {:.3} s, solve {:.3} s, levels {levels}, operator complexity {complexity:.2}, hierarchy {memory:.0} MB",
        r.free_dofs, r.nonzeros, r.iterations, r.relative_residual, r.assembly_seconds, r.preconditioner_seconds, r.solve_seconds,
    );
}
