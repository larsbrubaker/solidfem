//! Points through the middle of a wall whose skins are large faces. `layers.rs` proposes points
//! only from surface vertices, so a face with vertices only at its rim gets none in its interior
//! and the elements across it are not divided, whatever `skins.rs` splits: the tetrahedra there
//! are free to bridge the wall through a surface edge at the rim, which the fixed surface does not
//! allow to be cut.
//!
//! **What it adds.** A surface triangle much wider than the wall behind it (its longest edge at
//! least [`WALL_WIDTH_RATIO`] wall thicknesses) proposes interior points: a barycentric grid
//! across the triangle, at a spacing of twice the wall or a sixteenth of the triangle's longest
//! edge, whichever is larger (see [`wall_seeds`]), each looking straight across the wall and
//! naming the same layer points a vertex would, kept apart by the same separation rule. A
//! triangle narrower than that is left to the vertex rays and to the through-edge split, so an
//! ordinary meshed surface gets nothing from here.
//!
//! **A wedge is not a wall.** Two faces meeting at a sharp edge face each other too, and a column
//! beside the edge meets the other face at a distance set by how far it is from the edge, not by
//! any wall. A column is left out when the wedge its face and the far triangle make thins too
//! fast to hold the column's points across the patch it stands for, within `layers / 2` grid
//! steps of the edge, and the surface really is that wedge there: two rays find both skins on
//! their planes out to where the far skin would cut the column's deepest point (see
//! [`WedgeTest::holds_wedge`]). That does not depend on how the faces are cut into triangles.
//! Without it, a row of columns along the edge puts points nearly in line with the edge's
//! vertices, and the elements between them are flat. A faceted curved wall keeps the columns
//! whose skins turn away from their triangles' planes before that crossing; the crossing is
//! nearer with more layers, so a coarse curved wall gives up more of its columns at more layers,
//! where it really does thin past their deepest points.
//!
//! **What it costs and what it leaves.** A face gets at most about
//! [`MAX_GRID_STEPS`]`^2 / 2` candidates, a handful of points per wall layer per face for typical
//! CAD surfaces, each one ray across the wall; a column failing the thinning test casts the two
//! wedge rays as well. The rim keeps a band about one grid step wide where no point is proposed,
//! and an element there may still bridge the wall through a surface edge: that is the surface's
//! own edge and is reachable only by changing the surface. How far such an element reaches is
//! measured, not bounded: the tests check the plain box and the end-ring tube at two grid steps
//! from the rim, where no sampled mid-wall point is inside a tetrahedron joining the skins.
//! Nothing here decides which edges are through a wall; that rule is `skins.rs` and does not use
//! the width.

use crate::layers::{Hit, SurfaceRays};
use crate::quality::{cross, dot, length, normalized, subtract};
use crate::skins::OPPOSED_COSINE;

/// The most grid steps along the longest edge of one triangle: enough to keep the rim band a small
/// part of a large face, while keeping the rays and points for a face bounded. A judgement.
const MAX_GRID_STEPS: f64 = 16.0;

/// A triangle proposes columns only when its longest edge is at least this many walls wide. At
/// two-wall spacing that is four grid steps, the coarsest grid with columns apart from the one
/// at the triangle's centre; a narrower triangle is left to the vertex rays and the edge splits.
/// A judgement, not derived further. It decides where points are added, and so how far apart two
/// thin-wall measures may be and still be one wall (`thin_walls.rs`): the classification of
/// opposite skins (`skins.rs`) does not use it.
pub(crate) const WALL_WIDTH_RATIO: f64 = 8.0;

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
    /// The surface triangle the column ends on, across the wall.
    pub(crate) far: [u32; 3],
}

/// The columns a surface triangle proposes across the wall behind it, in a stable barycentric
/// order. Empty for a triangle no wider than the wall behind its centre allows.
pub(crate) fn wall_seeds(
    rays: &SurfaceRays,
    points: &[[f64; 3]],
    face: [u32; 3],
    thin_ratio: f64,
    layers: u32,
    magnitude: f64,
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
    // Surface coordinates read from single-precision files are each within half a unit in the
    // last place of their own value, so a point within `sqrt(3) * 2^-24 * magnitude` of where it
    // was meant. The probe's crossing lies on a triangle whose corners are each that far from the
    // plane they share with the far triangle, and the far triangle's plane is set by three more
    // such corners: four such roundings apart at most on the far triangle itself. A crossing
    // strips away extrapolates the far corners' rounding and may miss the plane by more, which
    // keeps the column: the safe side.
    let rounding = 3.0_f64.sqrt() * 0.5 * f64::from(f32::EPSILON) * magnitude;
    let wedge = WedgeTest {
        rays,
        spacing,
        layers: f64::from(layers),
        on_plane: 4.0 * rounding,
    };
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
            let facing = dot(hit.normal, normal);
            if !hit.from_inside
                || facing > OPPOSED_COSINE
                || longest < WALL_WIDTH_RATIO * hit.distance
                || hit.distance > thin_ratio * longest
                || wedge.holds_wedge(origin, normal, &hit, facing)
            {
                continue;
            }
            seeds.push(WallColumn {
                origin,
                direction: inward,
                distance: hit.distance,
                far_normal: hit.normal,
                across: 0.5 * spacing,
                far: hit.face,
            });
        }
    }
    seeds
}

/// What decides whether a column of one face stands in the corner of a wedge: the rays, the
/// face's grid step, the layers asked for and how close to a plane its points can be told apart.
struct WedgeTest<'a> {
    rays: &'a SurfaceRays,
    spacing: f64,
    layers: f64,
    on_plane: f64,
}

impl WedgeTest<'_> {
    /// Whether the column from `origin`, on a face of outward unit `normal`, that met `hit`, whose
    /// normal has cosine `facing` with the face's, stands in a wedge's corner rather than a wall.
    ///
    /// **Thinning across the patch.** The column meets the far triangle at thickness `t`. The two
    /// planes meet along a line `t / tan(angle)` from the column, the angle between them having
    /// cosine `-facing`, so across the patch, `spacing / 2` either side, the thickness falls by
    /// `tan(angle)` per unit towards that line. With `layers` layers the column's deepest point is
    /// at `(layers - 1) / layers` of `t`; on the patch's thin side the wall,
    /// `t - tan(angle) * spacing / 2`, must still hold it, or the points there sit in the corner
    /// beside the edge, not between two skins. That holds exactly when
    /// `t >= tan(angle) * spacing * layers / 2`, the line at least `layers / 2` grid steps away.
    /// Parallel skins never meet and always pass, and a column passing is kept with no more work.
    ///
    /// **Only where the surface is that wedge.** The planes are extended from two triangles and
    /// describe the surface only where the far skin really carries on in its triangle's plane. So
    /// a probe leaves the deepest point parallel to the face, towards the line: in a wedge it
    /// crosses the far skin, on the far triangle's plane, after `1 / layers` of the line's
    /// distance, inside the patch exactly when the test above fails. The column is in a wedge
    /// only when the probe does cross a triangle on that plane inside the patch, and a ray from
    /// halfway up the wall there, straight out, leaves through the face's own plane: both skins
    /// are then the two planes out to the crossing, whichever of their triangles those are, so a
    /// face split on either diagonal, or cut into strips along the edge, gives the same answer.
    /// On a faceted curved wall a skin turns away from its triangle's plane at the next facet: a
    /// column whose crossing lies beyond that facet has one of the two rays leave off its plane
    /// and is kept, as a wall missed is worse than a flat element. A column whose crossing comes
    /// before the skin turns is left out, since the wall really does thin past its deepest point
    /// there; with more layers the crossing is nearer, `1 / layers` of the line's distance, so a
    /// coarse curved wall gives up more of its columns at more layers.
    fn holds_wedge(&self, origin: [f64; 3], normal: [f64; 3], hit: &Hit, facing: f64) -> bool {
        // `facing` is at most `OPPOSED_COSINE` here; rounding can put it a hair below -1.
        let sine = (1.0 - facing * facing).max(0.0).sqrt();
        let t = hit.distance;
        if t * -facing >= 0.5 * self.layers * self.spacing * sine {
            return false;
        }
        // Towards the line: the far normal's part along the face, of length `sine`, not zero here.
        let toward = [0, 1, 2].map(|k| (hit.normal[k] - facing * normal[k]) / sine);
        let deepest = t * (self.layers - 1.0) / self.layers;
        let start = [0, 1, 2].map(|k| origin[k] - normal[k] * deepest);
        let Some(exit) = self.rays.nearest_any(start, toward) else {
            return false;
        };
        if !exit.from_inside || exit.distance > 0.5 * self.spacing {
            return false;
        }
        let far = [0, 1, 2].map(|k| origin[k] - normal[k] * t);
        let at = [0, 1, 2].map(|k| start[k] + toward[k] * exit.distance);
        if dot(subtract(at, far), hit.normal).abs() > self.on_plane {
            return false;
        }
        // The near skin must carry on in the face's plane out to the crossing too: from halfway
        // up the wall there, straight out, the ray leaves through a triangle on that plane.
        let within = [0, 1, 2].map(|k| at[k] + normal[k] * 0.5 * deepest);
        let Some(near) = self.rays.nearest_any(within, normal) else {
            return false;
        };
        let top = [0, 1, 2].map(|k| within[k] + normal[k] * near.distance);
        near.from_inside && dot(subtract(top, origin), normal).abs() <= self.on_plane
    }
}
