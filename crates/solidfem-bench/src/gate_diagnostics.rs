//! The measurements the mesher gate prints on request: the shape of the finest multigrid
//! aggregates (`MG_INSPECT`) and multigrid options overridden from the environment for
//! experiments (`MG_REACH`, `MG_RANGE`, `MG_DEGREE`, `MG_COARSE`). Kept apart from the gate binary
//! (`bin/gate.rs`), which owns the verdict, so its size stays bounded.

use solidfem::elastic::MultigridOptions;
use solidfem::mesh::QuadraticTetMesh;
use solidfem::solve::Multigrid;

/// Prints every level's size and what the finest aggregates look like: their node counts, how much
/// of the wall each one spans, and how wide each is across the wall (`MG_INSPECT`).
pub fn inspect_aggregates(multigrid: &Multigrid, mesh: &QuadraticTetMesh) {
    let levels = multigrid.level_sizes();
    let finest = levels[0].2 as f64;
    for (level, (rows, block, stored)) in levels.iter().enumerate() {
        println!(
            "  level {level}: {rows} block rows of {block}, stored values {stored} ({:.2} of finest)",
            *stored as f64 / finest
        );
    }
    let free: Vec<usize> = (0..mesh.nodes().len())
        .filter(|n| mesh.nodes()[*n][0].abs() >= 1e-9)
        .collect();
    let aggregates = multigrid.aggregates(0);
    if aggregates.is_empty() {
        return;
    }
    let count = 1 + aggregates.iter().copied().max().unwrap_or(0) as usize;
    let mut low = vec![[f64::MAX; 3]; count];
    let mut high = vec![[f64::MIN; 3]; count];
    let mut size = vec![0usize; count];
    for (row, aggregate) in aggregates.iter().enumerate() {
        let p = mesh.nodes()[free[row]];
        let a = *aggregate as usize;
        for axis in 0..3 {
            low[a][axis] = low[a][axis].min(p[axis]);
            high[a][axis] = high[a][axis].max(p[axis]);
        }
        size[a] += 1;
    }
    let wall = mesh.nodes().iter().map(|n| n[2]).fold(f64::MIN, f64::max);
    let mut spans = [0usize; 4];
    let (mut in_plane, mut through) = (0.0, 0.0);
    for a in 0..count {
        let dz = high[a][2] - low[a][2];
        through += dz;
        in_plane += ((high[a][0] - low[a][0]).powi(2) + (high[a][1] - low[a][1]).powi(2)).sqrt();
        spans[if dz >= 0.99 * wall {
            3
        } else if dz >= 0.5 * wall {
            2
        } else if dz > 0.0 {
            1
        } else {
            0
        }] += 1;
    }
    let small = size.iter().filter(|s| **s < 3).count();
    println!(
        "  level 0 aggregates: {count} for {} rows, mean size {:.2}, largest {}, smaller than 3 nodes {small}, mean z extent {:.3}, mean in-plane extent {:.3}; aggregates by wall spanned: flat {}, up to half {}, over half {}, full {}",
        aggregates.len(),
        aggregates.len() as f64 / count as f64,
        size.iter().max().copied().unwrap_or(0),
        through / count as f64,
        in_plane / count as f64,
        spans[0],
        spans[1],
        spans[2],
        spans[3],
    );
}

/// The default options, with `MG_REACH`, `MG_RANGE`, `MG_DEGREE`, `MG_COARSE` and `MG_THREADS`
/// overriding them for measurements.
pub fn multigrid_options() -> MultigridOptions {
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
    options.coarse_dofs = env_or("MG_COARSE", options.coarse_dofs as f64) as usize;
    options.threads = env_or("MG_THREADS", options.threads as f64) as usize;
    options
}
