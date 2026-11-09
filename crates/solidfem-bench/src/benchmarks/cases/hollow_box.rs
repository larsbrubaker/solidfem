//! Part 8: a 1 mm walled hollow box pressed at the centre of its top face, on a real unaligned
//! remeshed surface. There is no closed form; the reference is a structured quarter-symmetry
//! refinement study, kept beside this file as `hollow_box_reference.rs` and made by a real solve.
//!
//! The box is 40 x 40 x 20 mm with 1 mm walls and a closed internal void, clamped on its bottom
//! face, with a 10 N downward load spread over a 3 mm radius disk at the centre of the top face.
//! The reported quantity is the load-weighted mean deflection of the disk. The multigrid must
//! solve this thin closed shell of unaligned skins within the iteration budget.
//!
//! **Why these densities.** A 6 mm patch on a 1 mm skin needs about 7 elements across it before
//! the local plate bending under the load is resolved; elements of 3 to 4 mm put one or two across
//! the patch and the series is pre-asymptotic, so its last change says nothing about its error.
//! The densities are 2.0, 1.5 and 1.0 mm.
//!
//! **Why the load is clipped exactly.** Choosing faces by their centroids makes the loaded area
//! swing by up to 18% between meshes, which moves the deflection by about 1% with a sign that
//! varies, enough to flatter a coarse row. Each top-skin face is instead loaded by the exact area
//! of its overlap with the disk ([`disk_triangle_overlap`]), so the area and the total force are
//! the disk's at every density; the deflection is averaged with the same weights, which makes it
//! the work-conjugate of the applied load.

use std::f64::consts::PI;

use super::*;

const SIDE: f64 = 40.0;
pub(super) const HEIGHT: f64 = 20.0;
const PATCH: f64 = 3.0;
pub(super) const FORCE: f64 = 10.0;
const PATCH_AREA: f64 = PI * PATCH * PATCH;

/// The disk the load acts on: its centre on the top face and its radius.
pub const LOAD_DISK: ([f64; 2], f64) = ([0.5 * SIDE, 0.5 * SIDE], PATCH);

/// The iteration budget, enforced: a solve over this count is a named failure. The hardest plate
/// here took about 540 iterations at 170 thousand free values; 3000 is five times
/// that for a harder thin closed shell.
const ITERATION_BUDGET: u64 = 3000;

/// The converged load-weighted mean deflection of the disk, mm, from the structured
/// quarter-symmetry study of `hollow_box_reference.rs`: 3 layers through every wall with in-plane
/// element sizes of 1.0, 0.5 and 0.25 mm gave 5.4344, 5.4676 and 5.4776 x 1e-3 mm at 140 thousand,
/// 510 thousand and 1.9 million free values, an observed rate of 1.73; Richardson extrapolation of
/// the last two to zero size gives 5.482e-3 mm. The `structured_reference_...` test in the
/// benchmarks area redoes this by a real solve in release.
pub const PATCH_W_REFERENCE: f64 = 5.48e-3;

/// The area of the overlap of a disk with a triangle, in closed form: the sum over the triangle's
/// directed edges of the signed area of the disk's overlap with the triangle from the disk's
/// centre to that edge (Green's theorem), each a chord triangle plus up to two circular sectors.
/// The corners may wind either way.
pub fn disk_triangle_overlap(centre: [f64; 2], radius: f64, triangle: [[f64; 2]; 3]) -> f64 {
    let p = triangle.map(|v| [v[0] - centre[0], v[1] - centre[1]]);
    let signed = edge_overlap(p[0], p[1], radius)
        + edge_overlap(p[1], p[2], radius)
        + edge_overlap(p[2], p[0], radius);
    signed.abs()
}

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn sector(a: [f64; 2], b: [f64; 2], radius: f64) -> f64 {
    0.5 * radius * radius * cross(a, b).atan2(a[0] * b[0] + a[1] * b[1])
}

/// The signed area of the disk of `radius` at the origin overlapping the triangle (origin, a, b).
fn edge_overlap(a: [f64; 2], b: [f64; 2], radius: f64) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let qa = d[0] * d[0] + d[1] * d[1];
    if qa == 0.0 {
        return 0.0;
    }
    let qb = 2.0 * (a[0] * d[0] + a[1] * d[1]);
    let qc = a[0] * a[0] + a[1] * a[1] - radius * radius;
    let discriminant = qb * qb - 4.0 * qa * qc;
    if discriminant <= 0.0 {
        return sector(a, b, radius);
    }
    let root = discriminant.sqrt();
    let (t0, t1) = ((-qb - root) / (2.0 * qa), (-qb + root) / (2.0 * qa));
    if t1 <= 0.0 || t0 >= 1.0 {
        return sector(a, b, radius);
    }
    let along = |t: f64| [a[0] + t * d[0], a[1] + t * d[1]];
    let (enter, leave) = (along(t0.max(0.0)), along(t1.min(1.0)));
    let mut area = 0.5 * cross(enter, leave);
    if t0 > 0.0 {
        area += sector(a, enter, radius);
    }
    if t1 < 1.0 {
        area += sector(leave, b, radius);
    }
    area
}

/// The faces of the top skin overlapping the load disk at `centre`, each with its overlap area.
pub(super) fn disk_patch(mesh: &QuadraticTetMesh, centre: [f64; 2]) -> Vec<(BoundaryFace, f64)> {
    faces_where(mesh, |p| (p[2] - HEIGHT).abs() <= PLANE_TOL)
        .into_iter()
        .filter_map(|face| {
            let corners = [face.nodes[0], face.nodes[1], face.nodes[2]].map(|n| {
                let p = mesh.nodes()[n as usize];
                [p[0], p[1]]
            });
            let overlap = disk_triangle_overlap(centre, PATCH, corners);
            (overlap > 0.0).then_some((face, overlap))
        })
        .collect()
}

/// The uniform traction on each face of `patch` carrying `share` of the load: its force is
/// proportional to its overlap with the disk, and spreading that over the whole face keeps the
/// face's uniform traction. The overlaps must add up to the disk's share of the area, which
/// refuses a patch that misses part of the disk.
pub(super) fn disk_tractions(
    mesh: &QuadraticTetMesh,
    patch: &[(BoundaryFace, f64)],
    share: f64,
) -> Result<Vec<(BoundaryFace, [f64; 3])>, String> {
    let overlap: f64 = patch.iter().map(|(_, area)| area).sum();
    check_area(overlap, share * PATCH_AREA, "the load disk on the top skin")?;
    let force = share * FORCE;
    Ok(patch
        .iter()
        .map(|(face, area)| {
            let traction = -force * area / (face_area(mesh, face) * overlap);
            (*face, [0.0, 0.0, traction])
        })
        .collect())
}

/// The load-weighted mean downward deflection of `patch`. A uniform traction on a straight 6-node
/// face loads each midside node with a third of the face's force and its corners with none, so
/// these weights make the result the deflection the applied load does its work through.
pub(super) fn patch_deflection(
    patch: &[(BoundaryFace, f64)],
    displacements: &[[f64; 3]],
) -> Result<f64, String> {
    let (mut weighted, mut total) = (0.0, 0.0);
    for (face, overlap) in patch {
        for node in &face.nodes[3..] {
            weighted += overlap * -displacements[*node as usize][2] / 3.0;
        }
        total += overlap;
    }
    if total <= 0.0 {
        return Err("no faces in the top patch".to_string());
    }
    Ok(weighted / total)
}

fn apply(mesh: &QuadraticTetMesh) -> Result<Supports, String> {
    let patch = disk_patch(mesh, LOAD_DISK.0);
    Ok(Supports {
        fixed: plane_nodes(mesh, 2, 0.0),
        tractions: disk_tractions(mesh, &patch, 1.0)?,
        prescribed: Vec::new(),
    })
}

fn measure(mesh: &QuadraticTetMesh, solution: &Solution) -> Result<Vec<(String, f64)>, String> {
    let patch = disk_patch(mesh, [0.5 * SIDE, 0.5 * SIDE]);
    let deflection = patch_deflection(&patch, &solution.displacements)?;
    Ok(vec![("patch_w".to_string(), deflection)])
}

/// **Criterion.** The disk deflection must have moved by at most 2% in the last refinement, and
/// the finest density must lie within 2% of the independent structured reference
/// [`PATCH_W_REFERENCE`], with the iteration budget of [`ITERATION_BUDGET`] holding at every
/// density. The refinement limit alone cannot tell a converged series from a pre-asymptotic one
/// standing still, which the absolute check can.
fn judge(rows: &[DensityResult]) -> Verdict {
    verdict_of(
        stable(rows, "patch_w", 0.02, "patch deflection").and_then(|()| {
            let finest = rows.last().map_or(f64::NAN, |row| row.value("patch_w"));
            check_error(
                finest,
                PATCH_W_REFERENCE,
                0.02,
                "finest patch deflection against the structured reference",
            )
            .map(|_| ())
        }),
    )
}

pub(super) fn cases() -> Vec<Case> {
    vec![Case {
        id: "hollow-box",
        title: "Part 8: hollow box 40 x 40 x 20 with 1 mm walls, centre push",
        material: steel(0.3),
        densities: vec![
            density("h=2.0", "hollow_h2.0", Sizing::Uniform(2.0), 2.0),
            density("h=1.5", "hollow_h1.5", Sizing::Uniform(1.5), 1.5),
            density("h=1.0", "hollow_h1.0", Sizing::Uniform(1.0), 1.0),
        ],
        layers: 2,
        apply,
        measure,
        iteration_budget: Some(ITERATION_BUDGET),
        judge,
    }]
}
