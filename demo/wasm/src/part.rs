//! Builds the bar mesh for a request from solidfem's box builder: a unit-spaced grid of tetrahedra
//! whose cells are mapped to the wall-following grid lines, with the hollow cells of a tube
//! removed (a block with removed cells conforms because each cell is cut the same way).

use solidfem::mesh::{box_tet_mesh, MeshError, QuadraticTetMesh, TetMesh};

use crate::catalog::{SectionKind, StudyRequest};

/// The grid lines of one axis, ascending.
#[derive(Debug, Clone)]
pub struct GridLines(pub Vec<f64>);

impl GridLines {
    /// Cells along this axis.
    pub fn cells(&self) -> usize {
        self.0.len() - 1
    }
}

/// How the bar is cut into cells.
#[derive(Debug, Clone)]
pub struct PartGrid {
    /// Along the bar, a multiple of 8 cells so sections at L/8, L/4, 3L/8, L/2 and 3L/4 are grid
    /// planes (the textbook stations must be exact mesh planes).
    pub x: GridLines,
    pub y: GridLines,
    pub z: GridLines,
    /// Cells across the wall on each side, 0 for a solid section.
    pub wall_cells: usize,
}

/// Cells through a solid section's sides.
const SOLID_CELLS: usize = 4;

/// Cells through a tube's wall: two, so a wall's bending is carried by more than one element.
const WALL_CELLS: usize = 2;

/// Longest cell along the bar, as a fraction of the section's smaller side. What the bar's bending
/// needs along its length depends on the section depth, not on how thin a wall is, so a thin wall
/// does not force more cells along the bar.
const ALONG_CELL_OF_SIDE: f64 = 0.8;

/// Fewest cells along a bar: enough to carry a bar's bending, and enough that the load band of a
/// bar held at both ends (one cell either side of mid-span) stays clear of the sections the
/// textbook is read at.
const MIN_CELLS_ALONG: usize = 16;

/// Grid lines of one cross-section axis: solid sections cut evenly, tubes get `wall_cells` cells
/// across each wall and evenly sized cells across the interior.
fn cross_lines(
    request: &StudyRequest,
    along: f64,
    solid_cells: usize,
    wall_cells: usize,
) -> GridLines {
    match request.section {
        SectionKind::Solid => {
            let lines = (0..=solid_cells)
                .map(|i| along * i as f64 / solid_cells as f64)
                .collect();
            GridLines(lines)
        }
        SectionKind::Tube => {
            let wall = request.wall_fraction * request.width.min(request.height);
            let inner = along - 2.0 * wall;
            let wall_step = wall / wall_cells as f64;
            // The hollow span is cut as the wall is: it carries no material of its own, and a
            // positive span is what the wall range guarantees. A bar's bending lives in the
            // walls' stations along the bar, so finer cells across here only slow the solve.
            let inner_cells = wall_cells;
            let mut lines = Vec::with_capacity(2 * wall_cells + inner_cells + 1);
            for i in 0..=wall_cells {
                lines.push(wall_step * i as f64);
            }
            for i in 1..inner_cells {
                lines.push(wall + inner * i as f64 / inner_cells as f64);
            }
            for i in 0..=wall_cells {
                lines.push(along - wall + wall_step * i as f64);
            }
            GridLines(lines)
        }
    }
}

/// Plans the grid for a request. Geometry is validated by the caller.
pub fn plan_grid(request: &StudyRequest) -> PartGrid {
    let (solid_cells, wall_cells) = (SOLID_CELLS, WALL_CELLS);
    let y = cross_lines(request, request.width, solid_cells, wall_cells);
    let z = cross_lines(request, request.height, solid_cells, wall_cells);
    let side = request.width.min(request.height);
    let wanted = (request.length / (ALONG_CELL_OF_SIDE * side)).ceil() as usize;
    let along = wanted.max(MIN_CELLS_ALONG).div_ceil(8) * 8;
    let x = GridLines(
        (0..=along)
            .map(|i| request.length * i as f64 / along as f64)
            .collect(),
    );
    PartGrid {
        x,
        y,
        z,
        wall_cells: if request.section == SectionKind::Tube {
            wall_cells
        } else {
            0
        },
    }
}

/// Whether a section cell lies in the tube's hollow.
fn is_hollow(grid: &PartGrid, j: usize, k: usize) -> bool {
    let walls = grid.wall_cells;
    walls > 0
        && j >= walls
        && j < grid.y.cells() - walls
        && k >= walls
        && k < grid.z.cells() - walls
}

/// How many cells the mesh would keep.
pub fn kept_cells(grid: &PartGrid) -> usize {
    let mut section = 0;
    for j in 0..grid.y.cells() {
        for k in 0..grid.z.cells() {
            if !is_hollow(grid, j, k) {
                section += 1;
            }
        }
    }
    section * grid.x.cells()
}

/// Builds the 10-node mesh for a planned grid.
pub fn build_mesh(grid: &PartGrid) -> Result<QuadraticTetMesh, MeshError> {
    let cells = [grid.x.cells(), grid.y.cells(), grid.z.cells()];
    // A unit grid makes a tetrahedron's cell the integer floor of its centroid, and every grid
    // node the integer coordinates of a grid-line pick.
    let unit = box_tet_mesh(
        [0.0; 3],
        [cells[0] as f64, cells[1] as f64, cells[2] as f64],
        cells,
    )?;
    let keep: Vec<[u32; 4]> = unit
        .tets()
        .iter()
        .filter(|tet| {
            let centroid =
                tet.iter()
                    .map(|n| unit.positions()[*n as usize])
                    .fold([0.0; 3], |sum, p| {
                        [
                            sum[0] + p[0] * 0.25,
                            sum[1] + p[1] * 0.25,
                            sum[2] + p[2] * 0.25,
                        ]
                    });
            !is_hollow(
                grid,
                centroid[1].floor() as usize,
                centroid[2].floor() as usize,
            )
        })
        .copied()
        .collect();
    let mut remap = vec![u32::MAX; unit.positions().len()];
    let mut positions = Vec::new();
    let tets = keep
        .iter()
        .map(|tet| {
            tet.map(|node| {
                let slot = &mut remap[node as usize];
                if *slot == u32::MAX {
                    *slot = positions.len() as u32;
                    let p = unit.positions()[node as usize];
                    positions.push([
                        grid.x.0[p[0].round() as usize],
                        grid.y.0[p[1].round() as usize],
                        grid.z.0[p[2].round() as usize],
                    ]);
                }
                *slot
            })
        })
        .collect();
    Ok(QuadraticTetMesh::from_linear(&TetMesh::new(
        positions, tets,
    )?))
}
