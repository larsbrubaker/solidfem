//! Points through the middle of a wall whose skins are large faces. `layers.rs` proposes points
//! only from surface vertices, so a face with vertices only at its rim gets none in its interior
//! and the elements across it are not divided, whatever `skins.rs` splits: the tetrahedra there
//! are free to bridge the wall through a surface edge at the rim, which the fixed surface does not
//! allow to be cut.
//!
//! **What it adds.** A surface triangle much wider than the wall behind it (its longest edge at
//! least [`WALL_WIDTH_RATIO`] wall thicknesses) proposes interior
//! points: a barycentric grid across the triangle, at a spacing of twice the wall or a sixteenth
//! of the triangle's longest edge, whichever is larger (see [`wall_seeds`]), each looking straight across the
//! wall and naming the same layer points a vertex would, kept apart by the same separation rule. A
//! triangle narrower than that is left to the vertex rays and to the through-edge split, so an
//! ordinary meshed surface gets nothing from here.
//!
//! **What it costs and what it leaves.** A face gets at most about
//! [`MAX_GRID_STEPS`]`^2 / 2` candidates, a handful of points per wall layer per face for typical
//! CAD surfaces. The rim keeps a band about one grid step wide where no point is proposed, and an
//! element there may still bridge the wall through a surface edge: that is the surface's own edge
//! and is reachable only by changing the surface. How far such an element reaches is measured,
//! not bounded: the tests check the plain box and the end-ring tube at two grid steps from the
//! rim, where no sampled mid-wall point is inside a tetrahedron joining the skins. Nothing here
//! decides which edges are through a wall; that rule is `skins.rs` and does not use the width.

use crate::layers::SurfaceRays;
use crate::quality::{cross, dot, length, normalized, subtract};
use crate::skins::OPPOSED_COSINE;

/// The most grid steps along the longest edge of one triangle: enough to keep the rim band a small
/// part of a large face, while keeping the rays and points for a face bounded. A judgement.
const MAX_GRID_STEPS: f64 = 16.0;

/// A triangle proposes columns only when its longest edge is at least this many walls wide. At
/// two-wall spacing that is four grid steps, the coarsest grid with columns apart from the one
/// at the triangle's centre; a narrower triangle is left to the vertex rays and the edge splits.
/// A judgement, not derived further. It only decides where points are added: the classification
/// of opposite skins (`skins.rs`) does not use it.
const WALL_WIDTH_RATIO: f64 = 8.0;

/// One proposed column across a wall: where it starts, inward, the far side, and how far it
/// keeps from its neighbours.
#[derive(Clone, Copy)]
pub(crate) struct WallColumn {
    pub(crate) origin: [f64; 3],
    pub(crate) direction: [f64; 3],
    pub(crate) distance: f64,
    pub(crate) far_normal: [f64; 3],
    /// Half the column's lateral separation from its neighbours, which the separation rule applies.
    pub(crate) across: f64,
}

/// The columns a surface triangle proposes across the wall behind it, in a stable barycentric
/// order. Empty for a triangle no wider than the wall behind its centre allows.
pub(crate) fn wall_seeds(
    rays: &SurfaceRays,
    points: &[[f64; 3]],
    face: [u32; 3],
    thin_ratio: f64,
) -> Vec<WallColumn> {
    let [a, b, c] = face.map(|v| points[v as usize]);
    let (ab, ac) = (subtract(b, a), subtract(c, a));
    let normal = normalized(cross(ab, ac));
    let longest = [length(ab), length(ac), length(subtract(c, b))]
        .into_iter()
        .fold(0.0_f64, f64::max);
    let inward = [-normal[0], -normal[1], -normal[2]];
    let centroid = [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ];
    // The grid is sized from the wall behind the triangle's middle, when there is one: a triangle
    // with a wall behind only part of it still lays the finest grid its width allows, so whatever
    // part of it does face a wall is still seen.
    let wall = rays
        .nearest_off_face(centroid, inward, face)
        .filter(|hit| hit.from_inside && dot(hit.normal, normal) <= OPPOSED_COSINE)
        .map(|hit| hit.distance);
    if wall.is_some_and(|t| longest < WALL_WIDTH_RATIO * t) {
        return Vec::new();
    }
    let spacing = wall.map_or(0.0, |t| 2.0 * t).max(longest / MAX_GRID_STEPS);
    // `as` saturates; the step count is capped just above by construction.
    let steps = (longest / spacing).ceil().min(MAX_GRID_STEPS) as usize;
    let steps_f = steps as f64;
    let mut seeds = Vec::new();
    for i in 1..steps {
        for j in 1..steps - i {
            let (u, v) = (i as f64 / steps_f, j as f64 / steps_f);
            let w = 1.0 - u - v;
            let origin = [
                a[0] * w + b[0] * u + c[0] * v,
                a[1] * w + b[1] * u + c[1] * v,
                a[2] * w + b[2] * u + c[2] * v,
            ];
            let Some(hit) = rays.nearest_off_face(origin, inward, face) else {
                continue;
            };
            if !hit.from_inside
                || dot(hit.normal, normal) > OPPOSED_COSINE
                || longest < WALL_WIDTH_RATIO * hit.distance
                || hit.distance > thin_ratio * longest
            {
                continue;
            }
            seeds.push(WallColumn {
                origin,
                direction: inward,
                distance: hit.distance,
                far_normal: hit.normal,
                across: 0.5 * spacing,
            });
        }
    }
    seeds
}
