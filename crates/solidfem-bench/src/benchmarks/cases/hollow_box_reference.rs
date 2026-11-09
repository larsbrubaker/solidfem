//! The hollow-box benchmark's independent reference: a structured quarter-symmetry mesh of the
//! same box, load and clamp as `hollow_box.rs`, solved by a real run of the same 10-node elements
//! and multigrid. The structure is what makes it independent of the surface remesher and the
//! volume mesher the case tests: every wall has 3 layers of aligned cells and the in-plane size
//! is one number.
//!
//! The quarter `0..20` by `0..20` uses the two planes through the box's centre as symmetry planes,
//! so one quarter of the load disk and of the force is applied and the load-weighted mean over
//! the quadrant is the mean over the whole disk. Each cell is cut into six tetrahedra around its
//! lowest-to-highest diagonal, the same way in every cell so they conform.

use std::time::Instant;

use solidfem::elastic::{Quadrature, Report};
use solidfem::mesh::TetMesh;

use super::hollow_box::{disk_patch, disk_tractions, patch_deflection, FORCE, HEIGHT};
use super::*;
use crate::benchmarks::pipeline::{solve, SolveEnd};

/// The wall thickness and the layers through it.
const WALL: f64 = 1.0;
const WALL_LAYERS: usize = 3;
/// The plane each symmetry cut lies on, and the half length of the box's side.
const SYMMETRY: f64 = 20.0;
/// A reference solve runs until it converges; this only stops one gone wrong.
const CAP_SECONDS: f64 = 3600.0;

/// One structured solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceRow {
    /// The in-plane element size asked for, mm.
    pub h: f64,
    /// Solved degrees of freedom.
    pub free_dofs: usize,
    /// Conjugate gradient iterations.
    pub iterations: u64,
    /// The load-weighted mean deflection of the disk, mm.
    pub patch_w: f64,
    /// The whole pipeline's time, seconds.
    pub seconds: f64,
}

/// The grid lines of one in-plane axis: `WALL_LAYERS` layers in the wall, then cells of `h` out
/// to the symmetry plane.
fn side_lines(h: f64) -> Result<Vec<f64>, String> {
    let cells = whole_cells(SYMMETRY - WALL, h)?;
    let mut lines = wall_lines(0.0);
    lines.extend((1..=cells).map(|k| WALL + (SYMMETRY - WALL) * k as f64 / cells as f64));
    Ok(lines)
}

/// The grid lines through the height: the bottom wall, cells of `h`, then the top wall.
fn height_lines(h: f64) -> Result<Vec<f64>, String> {
    let span = HEIGHT - 2.0 * WALL;
    let cells = whole_cells(span, h)?;
    let mut lines = wall_lines(0.0);
    lines.extend((1..cells).map(|k| WALL + span * k as f64 / cells as f64));
    lines.extend(wall_lines(HEIGHT - WALL));
    Ok(lines)
}

/// The lines of the `WALL_LAYERS` layers of a wall starting at `from`, including both faces.
fn wall_lines(from: f64) -> Vec<f64> {
    (0..=WALL_LAYERS)
        .map(|k| from + WALL * k as f64 / WALL_LAYERS as f64)
        .collect()
}

/// How many cells of size `h` fill `span`, refused unless `h` divides it.
fn whole_cells(span: f64, h: f64) -> Result<usize, String> {
    if !h.is_finite() || !(0.05..=span).contains(&h) {
        return Err(format!("element size {h} is outside 0.05..{span}"));
    }
    // In range: the check above bounds the quotient by 20 / 0.05.
    let cells = (span / h).round() as usize;
    if (cells as f64 * h - span).abs() > 1e-9 {
        return Err(format!("element size {h} does not divide {span}"));
    }
    Ok(cells)
}

/// Whether the cell with centre `c` is in the void, which is open towards the symmetry planes.
fn in_void(c: [f64; 3]) -> bool {
    c[0] > WALL && c[1] > WALL && c[2] > WALL && c[2] < HEIGHT - WALL
}

/// The walks from a cell's lowest corner to its highest along each axis order: one tetrahedron
/// each, whose node order keeps its volume positive when `even`.
const AXIS_ORDERS: [([usize; 3], bool); 6] = [
    ([0, 1, 2], true),
    ([1, 2, 0], true),
    ([2, 0, 1], true),
    ([0, 2, 1], false),
    ([2, 1, 0], false),
    ([1, 0, 2], false),
];

/// The quarter box meshed in 10-node tetrahedra at in-plane size `h`.
fn quarter_box(h: f64) -> Result<QuadraticTetMesh, String> {
    let (x, z) = (side_lines(h)?, height_lines(h)?);
    let (nx, nz) = (x.len(), z.len());
    let slot = |[i, j, k]: [usize; 3]| i + nx * (j + nx * k);
    let mut index = vec![u32::MAX; nx * nx * nz];
    let mut positions: Vec<[f64; 3]> = Vec::new();
    let mut tets: Vec<[u32; 4]> = Vec::new();
    let mut node = |at: [usize; 3], positions: &mut Vec<[f64; 3]>| -> u32 {
        let id = &mut index[slot(at)];
        if *id == u32::MAX {
            // A quarter box of these sizes has well under four billion nodes.
            *id = positions.len() as u32;
            positions.push([x[at[0]], x[at[1]], z[at[2]]]);
        }
        *id
    };
    for k in 0..nz - 1 {
        for j in 0..nx - 1 {
            for i in 0..nx - 1 {
                let centre = [
                    0.5 * (x[i] + x[i + 1]),
                    0.5 * (x[j] + x[j + 1]),
                    0.5 * (z[k] + z[k + 1]),
                ];
                if in_void(centre) {
                    continue;
                }
                for (order, even) in AXIS_ORDERS {
                    let mut at = [i, j, k];
                    let v0 = node(at, &mut positions);
                    at[order[0]] += 1;
                    let v1 = node(at, &mut positions);
                    at[order[1]] += 1;
                    let v2 = node(at, &mut positions);
                    at[order[2]] += 1;
                    let v3 = node(at, &mut positions);
                    tets.push(if even {
                        [v0, v1, v2, v3]
                    } else {
                        [v0, v2, v1, v3]
                    });
                }
            }
        }
    }
    let linear = TetMesh::new(positions, tets).map_err(|e| format!("structured mesh: {e}"))?;
    Ok(QuadraticTetMesh::from_linear(&linear))
}

/// The clamp and the two symmetry planes, whose normal components are prescribed zero. A node on
/// the clamped plane is fully fixed already and a study refuses it being prescribed as well.
fn quarter_supports(
    mesh: &QuadraticTetMesh,
    patch: &[(BoundaryFace, f64)],
) -> Result<Supports, String> {
    let fixed = plane_nodes(mesh, 2, 0.0);
    let mut prescribed = Vec::new();
    for axis in 0..2 {
        prescribed.extend(
            nodes_where(mesh, |p| {
                (p[axis] - SYMMETRY).abs() <= PLANE_TOL && p[2].abs() > PLANE_TOL
            })
            .into_iter()
            .map(|node| (node, axis, 0.0)),
        );
    }
    Ok(Supports {
        fixed,
        tractions: disk_tractions(mesh, patch, 0.25)?,
        prescribed,
    })
}

/// Solves the quarter box at in-plane size `h` and returns the disk's load-weighted mean
/// deflection, after checking that the patch carries exactly a quarter of [`FORCE`].
///
/// # Errors
/// A message when `h` does not divide the box, the mesh or the patch is refused, or the solve
/// fails or reaches its time cap.
pub fn structured_reference(h: f64) -> Result<ReferenceRow, String> {
    let started = Instant::now();
    let mesh = quarter_box(h)?;
    let patch = disk_patch(&mesh, [SYMMETRY, SYMMETRY]);
    let supports = quarter_supports(&mesh, &patch)?;
    let applied: f64 = supports
        .tractions
        .iter()
        .map(|(face, traction)| -traction[2] * face_area(&mesh, face))
        .sum();
    let force_check = 0.25 * FORCE;
    if !within_limit((applied - force_check).abs(), 1e-9 * force_check) {
        return Err(format!("applied {applied} N, expected {force_check} N"));
    }
    let (solution, report) = solve(
        mesh,
        steel(0.3),
        &supports,
        CAP_SECONDS,
        None,
        Quadrature::FourPoint,
    )
    .map_err(|end| match end {
        SolveEnd::Refused(reason) => reason,
        SolveEnd::Capped(Report { iterations, .. }) => {
            format!("the solve hit its {CAP_SECONDS} s cap at iteration {iterations}")
        }
    })?;
    Ok(ReferenceRow {
        h,
        free_dofs: report.free_dofs,
        iterations: report.iterations,
        patch_w: patch_deflection(&patch, &solution.displacements)?,
        seconds: started.elapsed().as_secs_f64(),
    })
}
