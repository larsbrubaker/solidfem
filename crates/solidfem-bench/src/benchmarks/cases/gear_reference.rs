//! The gear tooth benchmark's independent reference: the same tooth outline, load and clamp as
//! `gear.rs`, meshed by a structured grid fitted to the outline instead of the surface remesher
//! and the volume mesher the case tests, and solved by the same 10-node elements and multigrid.
//!
//! The grid is one logical rectangle in the tooth's plane: `i` across from the left side to the
//! right side, `j` from the bore to the tip, with the boundary nodes on the outline's own polyline.
//! The interior is the Winslow elliptic grid (the harmonic map from the part to the logical
//! rectangle, which is one to one because the rectangle is convex), started from a Coons patch and
//! iterated to a stated tolerance: a grid stopped early keeps the patch's distortion at the
//! concave fillets, which shows up in the stress there. The grid is extruded through half the
//! face width, `z = 0..10`, with `z = 10` a symmetry plane, half the force, and the same clamp.
//! Each cell is cut into six tetrahedra around its diagonal, the same way in every cell, so a
//! refinement study scales the cells in all three directions together.
//!
//! This module is native only, because `crate::surfaces` is.

use std::time::Instant;

use solidfem::elastic::{Quadrature, Report, Solution};
use solidfem::mesh::TetMesh;

use super::gear::{apply, measure};
use super::*;
use crate::benchmarks::pipeline::{solve, SolveEnd};
use crate::surfaces::shapes::{gear_fillet, gear_tooth_outline};

/// The face width and the symmetry plane at its middle.
const FACE_WIDTH: f64 = 20.0;
const SYMMETRY_Z: f64 = 0.5 * FACE_WIDTH;
/// A reference solve runs until it converges; this only stops one gone wrong.
const CAP_SECONDS: f64 = 3600.0;
/// The Winslow iteration stops when no node moved further than this in a sweep, mm: three orders
/// below the smallest cell of the finest grid, so the rest of the iteration cannot move a stress.
const WINSLOW_TOLERANCE: f64 = 1e-6;
/// Over-relaxation of the Winslow sweeps: shortens the slow convergence on long thin grids.
const WINSLOW_RELAXATION: f64 = 1.7;
/// A grid that has not settled after this many sweeps is refused rather than used.
const WINSLOW_SWEEPS: usize = 400_000;

/// The right side of the outline as stretches of `outline[from..=to]`: the sector side, the root
/// arc, the fillet and the flank.
const SIDE_STRETCHES: [(usize, usize); 4] = [(0, 1), (1, 9), (9, 23), (23, 44)];
/// Which of the stretches is the fillet.
const FILLET_STRETCH: usize = 2;
/// The index just past the last point of the right side in the outline.
const RIGHT_SIDE_POINTS: usize = 45;

/// The cells of one structured grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToothGrid {
    /// Cells across the tooth.
    pub across: usize,
    /// Cells along each of the outline's stretches: sector side, root arc, fillet, flank.
    pub side_cells: [usize; 4],
    /// Layers through the half face width.
    pub half_layers: usize,
}

/// One structured solve of the tooth.
#[derive(Debug, Clone, PartialEq)]
pub struct GearReferenceRow {
    /// The grid solved.
    pub grid: ToothGrid,
    /// Winslow sweeps the grid took to settle.
    pub sweeps: usize,
    /// Free degrees of freedom.
    pub free_dofs: usize,
    /// Conjugate gradient iterations.
    pub iterations: u64,
    /// The root peak, measured as the benchmark measures it, MPa.
    pub root_peak: f64,
    /// Seconds for the whole solve.
    pub seconds: f64,
}

/// A polyline resampled at `n` equal arc-length steps, including both ends.
fn resample(points: &[[f64; 2]], n: usize) -> Vec<[f64; 2]> {
    let mut lengths = vec![0.0];
    for w in points.windows(2) {
        let (dx, dy) = (w[1][0] - w[0][0], w[1][1] - w[0][1]);
        lengths.push(lengths[lengths.len() - 1] + (dx * dx + dy * dy).sqrt());
    }
    let total = lengths[lengths.len() - 1];
    (0..=n)
        .map(|k| {
            let s = total * k as f64 / n as f64;
            let at = lengths
                .partition_point(|l| *l < s)
                .clamp(1, points.len() - 1);
            let span = lengths[at] - lengths[at - 1];
            let t = if span > 0.0 {
                (s - lengths[at - 1]) / span
            } else {
                0.0
            };
            [
                points[at - 1][0] + t * (points[at][0] - points[at - 1][0]),
                points[at - 1][1] + t * (points[at][1] - points[at - 1][1]),
            ]
        })
        .collect()
}

/// The (x, y) of the outline's right side, bore corner to tip corner, resampled. The fillet is
/// sampled on its true circular arc rather than the outline's chords: a grid finer than the
/// chords would resolve their corners as real stress raisers, and the peak would climb with
/// refinement towards a polyline's answer rather than the circle's.
fn right_side(cells: [usize; 4]) -> Vec<[f64; 2]> {
    let outline: Vec<[f64; 2]> = gear_tooth_outline().iter().map(|p| [p.x, p.y]).collect();
    let right = &outline[..RIGHT_SIDE_POINTS];
    let mut side = vec![right[0]];
    for (index, (stretch, n)) in SIDE_STRETCHES.iter().zip(cells).enumerate() {
        if index == FILLET_STRETCH {
            let fillet = gear_fillet();
            side.extend((1..=n).map(|k| {
                let a = fillet.start + (fillet.end - fillet.start) * k as f64 / n as f64;
                [
                    fillet.centre[0] + fillet.radius * a.cos(),
                    fillet.centre[1] + fillet.radius * a.sin(),
                ]
            }));
        } else {
            side.extend(
                resample(&right[stretch.0..=stretch.1], n)
                    .into_iter()
                    .skip(1),
            );
        }
    }
    side
}

/// The logical grid and how many Winslow sweeps it took.
struct PlaneGrid {
    points: Vec<[f64; 2]>,
    nx: usize,
    ny: usize,
    sweeps: usize,
}

/// The body-fitted (x, y) grid.
fn plane_grid(across: usize, side_cells: [usize; 4]) -> Result<PlaneGrid, String> {
    let right = right_side(side_cells);
    let ny = right.len() - 1;
    let nx = across;
    let at = |i: usize, j: usize| i + (nx + 1) * j;
    let mut g = vec![[0.0; 2]; (nx + 1) * (ny + 1)];
    let polar = |p: [f64; 2]| ((p[0] * p[0] + p[1] * p[1]).sqrt(), p[0].atan2(p[1]));
    let (bottom_r, bottom_t) = polar(right[0]);
    let (tip_r, tip_t) = polar(right[ny]);
    for i in 0..=nx {
        let f = -1.0 + 2.0 * i as f64 / nx as f64;
        g[at(i, 0)] = [
            bottom_r * (f * bottom_t).sin(),
            bottom_r * (f * bottom_t).cos(),
        ];
        g[at(i, ny)] = [tip_r * (f * tip_t).sin(), tip_r * (f * tip_t).cos()];
    }
    for (j, p) in right.iter().enumerate() {
        g[at(nx, j)] = *p;
        g[at(0, j)] = [-p[0], p[1]];
    }
    for j in 1..ny {
        for i in 1..nx {
            let (u, v) = (i as f64 / nx as f64, j as f64 / ny as f64);
            let coons = |c: usize| {
                (1.0 - u) * g[at(0, j)][c]
                    + u * g[at(nx, j)][c]
                    + (1.0 - v) * g[at(i, 0)][c]
                    + v * g[at(i, ny)][c]
                    - (1.0 - u) * (1.0 - v) * g[at(0, 0)][c]
                    - u * (1.0 - v) * g[at(nx, 0)][c]
                    - (1.0 - u) * v * g[at(0, ny)][c]
                    - u * v * g[at(nx, ny)][c]
            };
            let point = [coons(0), coons(1)];
            g[at(i, j)] = point;
        }
    }
    for sweep in 1..=WINSLOW_SWEEPS {
        let mut largest = 0.0_f64;
        for j in 1..ny {
            for i in 1..nx {
                let xe = [
                    0.5 * (g[at(i + 1, j)][0] - g[at(i - 1, j)][0]),
                    0.5 * (g[at(i + 1, j)][1] - g[at(i - 1, j)][1]),
                ];
                let xn = [
                    0.5 * (g[at(i, j + 1)][0] - g[at(i, j - 1)][0]),
                    0.5 * (g[at(i, j + 1)][1] - g[at(i, j - 1)][1]),
                ];
                let alpha = xn[0] * xn[0] + xn[1] * xn[1];
                let gamma = xe[0] * xe[0] + xe[1] * xe[1];
                let beta = xe[0] * xn[0] + xe[1] * xn[1];
                let denom = 2.0 * (alpha + gamma);
                if denom <= 0.0 {
                    return Err(format!("the grid collapsed at cell ({i}, {j})"));
                }
                let target = |c: usize| {
                    (alpha * (g[at(i + 1, j)][c] + g[at(i - 1, j)][c])
                        + gamma * (g[at(i, j + 1)][c] + g[at(i, j - 1)][c])
                        - 0.5
                            * beta
                            * (g[at(i + 1, j + 1)][c]
                                - g[at(i + 1, j - 1)][c]
                                - g[at(i - 1, j + 1)][c]
                                + g[at(i - 1, j - 1)][c]))
                        / denom
                };
                let (target_x, target_y) = (target(0), target(1));
                let point = &mut g[at(i, j)];
                let step = [
                    WINSLOW_RELAXATION * (target_x - point[0]),
                    WINSLOW_RELAXATION * (target_y - point[1]),
                ];
                point[0] += step[0];
                point[1] += step[1];
                largest = largest.max(step[0].abs()).max(step[1].abs());
            }
        }
        if largest <= WINSLOW_TOLERANCE {
            return Ok(PlaneGrid {
                points: g,
                nx,
                ny,
                sweeps: sweep,
            });
        }
    }
    Err(format!(
        "the Winslow grid did not settle in {WINSLOW_SWEEPS} sweeps"
    ))
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

/// The half tooth, `z = 0..10`, meshed on `grid`, and the Winslow sweeps it took.
///
/// # Errors
/// A message when the grid does not settle or the mesh is refused.
pub fn structured_tooth(grid: ToothGrid) -> Result<(QuadraticTetMesh, usize), String> {
    let plane = plane_grid(grid.across, grid.side_cells)?;
    let (nx, ny, layers) = (plane.nx, plane.ny, grid.half_layers);
    let slot = |[i, j, k]: [usize; 3]| i + (nx + 1) * (j + (ny + 1) * k);
    let mut positions = vec![[0.0; 3]; (nx + 1) * (ny + 1) * (layers + 1)];
    for k in 0..=layers {
        let z = SYMMETRY_Z * k as f64 / layers as f64;
        for j in 0..=ny {
            for i in 0..=nx {
                let p = plane.points[i + (nx + 1) * j];
                positions[slot([i, j, k])] = [p[0], p[1], z];
            }
        }
    }
    let mut tets = Vec::with_capacity(nx * ny * layers * 6);
    for k in 0..layers {
        for j in 0..ny {
            for i in 0..nx {
                for (order, even) in AXIS_ORDERS {
                    let mut c = [i, j, k];
                    // A tooth of these sizes has well under four billion nodes.
                    let v0 = slot(c) as u32;
                    c[order[0]] += 1;
                    let v1 = slot(c) as u32;
                    c[order[1]] += 1;
                    let v2 = slot(c) as u32;
                    c[order[2]] += 1;
                    let v3 = slot(c) as u32;
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
    Ok((QuadraticTetMesh::from_linear(&linear), plane.sweeps))
}

/// The case's supports on the half tooth: its clamp and half its load, because `apply` spreads
/// the whole force over whatever tip area it finds, and the mid-plane held in `z`.
///
/// # Errors
/// A message when the tip is not found.
pub fn half_model_supports(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
    let mut supports = apply(mesh)?;
    for (_, traction) in supports.tractions.iter_mut() {
        for value in traction.iter_mut() {
            *value *= 0.5;
        }
    }
    let mut is_fixed = vec![false; mesh.nodes().len()];
    for node in &supports.fixed {
        is_fixed[*node as usize] = true;
    }
    supports.prescribed = plane_nodes(mesh, 2, SYMMETRY_Z)
        .into_iter()
        .filter(|node| !is_fixed[*node as usize])
        .map(|node| (node, 2, 0.0))
        .collect();
    Ok(supports)
}

/// Solves the half tooth on `grid`.
///
/// # Errors
/// A message when the grid, the mesh or the solve is refused or capped.
pub fn solve_structured(
    grid: ToothGrid,
) -> Result<(QuadraticTetMesh, Solution, Report, usize), String> {
    let (mesh, sweeps) = structured_tooth(grid)?;
    let supports = half_model_supports(&mesh)?;
    let (solution, report) = solve(
        mesh.clone(),
        steel(0.3),
        &supports,
        CAP_SECONDS,
        None,
        Quadrature::FourPoint,
    )
    .map_err(|end| match end {
        SolveEnd::Refused(reason) => reason,
        SolveEnd::Capped(report) => format!("capped at iteration {}", report.iterations),
    })?;
    Ok((mesh, solution, report, sweeps))
}

/// One structured solve's root peak, measured as the benchmark measures it.
///
/// # Errors
/// A message when the solve or the measurement fails.
pub fn gear_structured_reference(grid: ToothGrid) -> Result<GearReferenceRow, String> {
    let started = Instant::now();
    let (mesh, solution, report, sweeps) = solve_structured(grid)?;
    let values = measure(&mesh, &solution)?;
    let root_peak = values
        .iter()
        .find(|(name, _)| name == "root_peak")
        .map(|(_, value)| *value)
        .ok_or_else(|| "no root peak".to_string())?;
    Ok(GearReferenceRow {
        grid,
        sweeps,
        free_dofs: report.free_dofs,
        iterations: report.iterations,
        root_peak,
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// How far above and below a section its integration points are taken, mm.
const SECTION_BAND: f64 = 0.3;
/// How far from the free face at `z = 0` a section's points must be, mm, so the fit reads the
/// section's bending and not the local stress change at the free face.
const FREE_FACE_MARGIN: f64 = 0.5;

/// For each section height in `sections`, the ratio of the section's bending stress, a straight
/// line fitted to `sigma_yy` against `x` over the integration points within [`SECTION_BAND`] of
/// it and read at the tension-side surface, to beam theory's `6 F l / (b t^2)`. The force and its
/// height come from the supports the solve used, `t` from the outline, so the expected value owes
/// nothing to the elements.
///
/// # Errors
/// A message when a section has too few points or is off the outline.
pub fn beam_section_ratios(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
    sections: &[f64],
) -> Result<Vec<(f64, f64)>, String> {
    let supports = half_model_supports(mesh)?;
    let (mut force, mut force_y) = (0.0, 0.0);
    for (face, traction) in &supports.tractions {
        let share = traction[0] * face_area(mesh, face);
        let y = face.nodes[..3]
            .iter()
            .map(|n| mesh.nodes()[*n as usize][1])
            .sum::<f64>()
            / 3.0;
        force += share;
        force_y += share * y;
    }
    let load_height = force_y / force;
    let outline: Vec<[f64; 2]> = gear_tooth_outline().iter().map(|p| [p.x, p.y]).collect();
    let flank = &outline[SIDE_STRETCHES[2].0..RIGHT_SIDE_POINTS];
    let mut ratios = Vec::with_capacity(sections.len());
    for y0 in sections {
        let half_width = flank
            .windows(2)
            .find(|w| (w[0][1] - y0) * (w[1][1] - y0) <= 0.0)
            .map(|w| w[0][0] + (y0 - w[0][1]) / (w[1][1] - w[0][1]) * (w[1][0] - w[0][0]))
            .ok_or_else(|| format!("section y = {y0} is off the flank"))?;
        let (mut n, mut sum_x, mut sum_xx, mut sum_s, mut sum_xs) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for point in &solution.point_stresses {
            let [x, y, z] = point.position;
            if (y - y0).abs() <= SECTION_BAND && z >= FREE_FACE_MARGIN {
                n += 1.0;
                sum_x += x;
                sum_xx += x * x;
                sum_s += point.stress[1];
                sum_xs += x * point.stress[1];
            }
        }
        let determinant = n * sum_xx - sum_x * sum_x;
        if n < 6.0 || determinant <= 0.0 {
            return Err(format!("too few integration points at y = {y0}"));
        }
        let slope = (n * sum_xs - sum_x * sum_s) / determinant;
        let t = 2.0 * half_width;
        let beam = 6.0 * force * (load_height - y0) / (SYMMETRY_Z * t * t);
        ratios.push((*y0, -slope * half_width / beam));
    }
    Ok(ratios)
}

/// The grids of the refinement study that sets `ROOT_PEAK_REFERENCE`: the cells across the tooth
/// grow by 1.5 from each to the next, and the cells along the outline and through the width grow
/// in step, rounded to whole cells.
pub const ROOT_PEAK_REFERENCE_GRIDS: [ToothGrid; 3] = [
    ToothGrid {
        across: 24,
        side_cells: [6, 8, 24, 30],
        half_layers: 3,
    },
    ToothGrid {
        across: 36,
        side_cells: [8, 12, 36, 45],
        half_layers: 4,
    },
    ToothGrid {
        across: 54,
        side_cells: [12, 18, 54, 68],
        half_layers: 6,
    },
];
