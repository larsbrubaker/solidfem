//! Measures the preconditioners on the two problem classes that matter: a stocky cantilever and a
//! thin plate. `solidfem-preconditioner <cantilever|plate> <jacobi|multigrid> <nx> <ny> <nz>
//! <cap-seconds>` builds a structured box of 10-node elements, clamps x = 0, loads the x = L face
//! and prints the free unknowns, iterations, setup and solve seconds, operator complexity and
//! hierarchy memory. Run in release; a capped solve reports the residual it reached. `MG_INSPECT`
//! prints the levels and the finest aggregates' geometry, `MG_REACH` and the like override the
//! options, and `MG_SLICE_ROWS` assembles and sets up in slices of that many rows, printing the
//! longest single call.

use std::time::Instant;

use solidfem::elastic::{
    Clock, Material, Model, MultigridOptions, PreconditionerChoice, StaticStudy, StepOutcome,
    StudyBuilder, StudyOptions,
};
use solidfem::mesh::{box_tet_mesh, QuadraticTetMesh};
use solidfem::solve::Multigrid;

const ON_PLANE: f64 = 1e-9;

/// An aggregate's lowest corner, highest corner and member count.
type Extent = ([f64; 3], [f64; 3], usize);

/// Prints every level's size, and what the finest level's aggregates look like geometrically:
/// how many nodes they hold, and how far they reach through the wall and across it.
fn inspect(multigrid: &Multigrid, mesh: &QuadraticTetMesh, size: &[f64; 3]) {
    let levels = multigrid.level_sizes();
    let finest = levels[0].2 as f64;
    for (level, (rows, block, stored)) in levels.iter().enumerate() {
        println!(
            "  level {level}: {rows} block rows of {block}, stored values {stored} ({:.2} of finest)",
            *stored as f64 / finest
        );
    }
    let free: Vec<usize> = (0..mesh.nodes().len())
        .filter(|n| mesh.nodes()[*n][0].abs() >= ON_PLANE)
        .collect();
    let aggregates = multigrid.aggregates(0);
    if aggregates.is_empty() {
        return;
    }
    let count = 1 + aggregates.iter().copied().max().unwrap_or(0) as usize;
    let mut reach: Vec<Extent> = vec![([f64::MAX; 3], [f64::MIN; 3], 0usize); count];
    for (row, aggregate) in aggregates.iter().enumerate() {
        let p = mesh.nodes()[free[row]];
        let entry = &mut reach[*aggregate as usize];
        for (axis, value) in p.iter().enumerate() {
            entry.0[axis] = entry.0[axis].min(*value);
            entry.1[axis] = entry.1[axis].max(*value);
        }
        entry.2 += 1;
    }
    let mean = |pick: &dyn Fn(&Extent) -> f64| reach.iter().map(pick).sum::<f64>() / count as f64;
    println!(
        "  level 0 aggregates: {count} for {} rows, mean size {:.1}, largest {}, singletons {}, mean extent x {:.3} y {:.3} z {:.3} (part {} x {} x {})",
        aggregates.len(),
        aggregates.len() as f64 / count as f64,
        reach.iter().map(|r| r.2).max().unwrap_or(0),
        reach.iter().filter(|r| r.2 == 1).count(),
        mean(&|r| r.1[0] - r.0[0]),
        mean(&|r| r.1[1] - r.0[1]),
        mean(&|r| r.1[2] - r.0[2]),
        size[0],
        size[1],
        size[2],
    );
}

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
            options.neighbour_reach = env_or("MG_REACH", options.neighbour_reach);
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
    let options = StudyOptions {
        clock: Some(clock),
        preconditioner,
        ..StudyOptions::default()
    };
    let slice = std::env::var("MG_SLICE_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let mut study = if let Some(rows) = slice {
        let mut builder = StudyBuilder::new(model, options).expect("the study builds");
        let (mut longest, mut calls) = (0.0_f64, 0usize);
        loop {
            let begun = Instant::now();
            let progress = builder.step(rows, &mut || false).expect("builds");
            longest = longest.max(begun.elapsed().as_secs_f64());
            calls += 1;
            if progress.elements_done == progress.elements_total && progress.setup_finished {
                break;
            }
        }
        println!(
            "longest single build slice {:.1} ms over {calls} calls of {rows} rows",
            longest * 1000.0
        );
        builder.finish().expect("finished")
    } else {
        StaticStudy::new(model, options, &mut || false).expect("the study builds")
    };
    if std::env::var("MG_INSPECT").is_ok() {
        if let Some(multigrid) = study.multigrid() {
            inspect(multigrid, &mesh, &size);
        }
    }
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
