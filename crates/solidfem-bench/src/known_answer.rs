//! The cantilever with a known answer, through the real mesher: a 100 x 10 x 10 bar described by a
//! signed-distance field, one end clamped, a total force on the other. Poisson's ratio is zero so
//! the clamp restrains nothing beam theory does not model. Deflection is compared with Timoshenko
//! and mid-span stress with Mc/I.

use solidfem_elastic::{Material, Solution};
use solidfem_mesh::{tet_volume, QuadraticTetMesh};

use crate::pipeline::{Case, ON_FACE_FRACTION};

/// How the clamped and loaded ends of the bar are chosen from the mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EndSelection {
    /// Only nodes on the end planes, within a small tolerance: what a part with planar ends gives.
    Plane,
    /// Nodes within half a grid cell of the end planes: the mesher's surface is only approximately
    /// planar, so this is what clamping "the end" means on its mesh.
    Band,
}

impl EndSelection {
    /// The distance from an end plane within which a node belongs to that end.
    pub fn half_width(self, dx: f64) -> f64 {
        match self {
            EndSelection::Plane => ON_FACE_FRACTION * dx,
            EndSelection::Band => 0.5 * dx,
        }
    }
}
use crate::shapes::box_sdf;

/// Bar length.
pub const LENGTH: f64 = 100.0;
/// Section side.
pub const SIDE: f64 = 10.0;
/// Total transverse force.
pub const FORCE: f64 = 100.0;
/// Young's modulus.
pub const YOUNGS: f64 = 2000.0;

/// The bar meshed with `cells_across` grid cells through its depth, aligned with the grid.
pub fn cantilever_case(cells_across: u32) -> Case {
    cantilever_variant(cells_across, 0.0, EndSelection::Plane)
}

/// The bar with a grid shift and an end selection, to separate what each costs.
pub fn cantilever_variant(
    cells_across: u32,
    grid_shift_cells: f64,
    selection: EndSelection,
) -> Case {
    let dx = SIDE / f64::from(cells_across);
    let half_width = selection.half_width(dx);
    Case {
        name: format!("h/dx={cells_across} shift={grid_shift_cells} {selection:?}"),
        sdf: Box::new(|p| box_sdf(p, [0.0; 3], [LENGTH, SIDE, SIDE])),
        low: [0.0; 3],
        high: [LENGTH, SIDE, SIDE],
        dx,
        fixed: Box::new(move |p, _| p[0].abs() < half_width),
        loaded: Box::new(move |p, _| (p[0] - LENGTH).abs() < half_width),
        total_force: [0.0, 0.0, -FORCE],
        material: Material {
            youngs_modulus: YOUNGS,
            poisson_ratio: 0.0,
        },
        grid_shift_cells,
    }
}

/// Timoshenko tip deflection of the bar: bending plus shear with coefficient 5/6.
pub fn timoshenko_deflection() -> f64 {
    let area = SIDE * SIDE;
    let inertia = SIDE.powi(4) / 12.0;
    let shear_modulus = YOUNGS / 2.0;
    FORCE * LENGTH.powi(3) / (3.0 * YOUNGS * inertia)
        + FORCE * LENGTH / (5.0 / 6.0 * shear_modulus * area)
}

/// Peak bending stress at mid-span: M c / I with M = F * L / 2.
pub fn mid_span_bending_stress() -> f64 {
    let inertia = SIDE.powi(4) / 12.0;
    FORCE * 0.5 * LENGTH * (0.5 * SIDE) / inertia
}

/// The measured tip deflection and mid-span peak von Mises stress of a solved bar.
pub fn evaluate(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
    dx: f64,
    selection: EndSelection,
) -> (f64, f64) {
    let tolerance = selection.half_width(dx);
    let mut sum = 0.0;
    let mut count = 0.0;
    let mut peak: f64 = 0.0;
    for (node, p) in mesh.nodes().iter().enumerate() {
        if (p[0] - LENGTH).abs() < tolerance {
            sum += solution.displacements[node][2];
            count += 1.0;
        }
        if (p[0] - 0.5 * LENGTH).abs() < dx {
            peak = peak.max(solution.nodal_von_mises[node]);
        }
    }
    (-sum / count, peak)
}

/// What the mesher made of the bar: volume, the section's second moment between stations, and the
/// area of the faces selected as the clamped root and the loaded tip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    /// Mesh volume over the exact volume.
    pub volume_ratio: f64,
    /// Second moment of the section about its centre, averaged over the middle of the bar, over the
    /// exact value h^4 / 12. Bending compliance scales with its inverse.
    pub second_moment_ratio: f64,
    /// Boundary area on the root plane over the section's area.
    pub root_area_ratio: f64,
    /// Boundary area on the tip plane over the section's area.
    pub tip_area_ratio: f64,
}

/// Measures the mesh against the exact bar.
pub fn measure_geometry(mesh: &QuadraticTetMesh, dx: f64, selection: EndSelection) -> Geometry {
    let tolerance = selection.half_width(dx);
    let (band_low, band_high) = (0.2 * LENGTH, 0.8 * LENGTH);
    let mut volume = 0.0;
    let mut band_second_moment = 0.0;
    for tet in mesh.tets() {
        let corners = [tet[0], tet[1], tet[2], tet[3]].map(|n| mesh.nodes()[n as usize]);
        let v = tet_volume(corners[0], corners[1], corners[2], corners[3]);
        volume += v;
        let centroid_x = 0.25 * (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]);
        if centroid_x > band_low && centroid_x < band_high {
            let z = corners.map(|c| c[2] - 0.5 * SIDE);
            let sum: f64 = z.iter().sum();
            let squares: f64 = z.iter().map(|c| c * c).sum();
            // The integral of z^2 over a tetrahedron from its corner values.
            band_second_moment += v / 20.0 * (squares + sum * sum);
        }
    }
    let exact_second_moment = SIDE.powi(4) / 12.0;
    let on_plane = |face: &[u32; 6], x: f64| {
        face[..3]
            .iter()
            .all(|n| (mesh.nodes()[*n as usize][0] - x).abs() < tolerance)
    };
    let area_at = |x: f64| -> f64 {
        mesh.boundary_faces()
            .iter()
            .filter(|f| on_plane(&f.nodes, x))
            .map(|f| corner_area(mesh, &f.nodes))
            .sum()
    };
    let section = SIDE * SIDE;
    Geometry {
        volume_ratio: volume / (LENGTH * section),
        second_moment_ratio: band_second_moment / (band_high - band_low) / exact_second_moment,
        root_area_ratio: area_at(0.0) / section,
        tip_area_ratio: area_at(LENGTH) / section,
    }
}

fn corner_area(mesh: &QuadraticTetMesh, face: &[u32; 6]) -> f64 {
    let [a, b, c] = [face[0], face[1], face[2]].map(|n| mesh.nodes()[n as usize]);
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
}
