// A flat face whose corners were rounded is two triangles folded by a rounding-sized angle, and the
// other diagonal of each of its quads runs along the face, not through the solid. Measured on a
// wedge whose sharp edge is 30 degrees, so the faces either side of it face nearly away from each
// other, and on a thin wall whose top is a strip one quad across between two acute edges, its side
// faces first in triangle order; each turned and rounded to single precision the way a mesh file
// stores it.

use crate::shapes::{rotated_by, Surface};
use crate::support::{constant_size, dot, options_with_layers, recovered, run_refine, sub};
use solidfem_tetra::RefineOptions;

fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// The cross-section's three corners: a sharp edge at the origin with sides at 15 degrees either
/// side of the x axis, `depth` long, and a flat back.
fn section(depth: f64) -> [[f64; 2]; 3] {
    // tan(15 degrees), so the sharp edge is 30 degrees.
    let w = depth * 0.267_949_192_431_122_7;
    [[0.0, 0.0], [depth, -w], [depth, w]]
}

/// A prism on `section` along z, each side split into `m x m` quads and each end fanned from its
/// centroid, wound outward.
fn wedge(depth: f64, length: f64, m: usize) -> Surface {
    let corners = section(depth);
    // The rim, counterclockwise seen from +z, `m` pieces per side, the corners exact.
    let mut rim: Vec<[f64; 2]> = Vec::new();
    for side in 0..3 {
        let (a, b) = (corners[side], corners[(side + 1) % 3]);
        rim.extend((0..m).map(|i| {
            if i == 0 {
                a
            } else {
                lerp(a, b, i as f64 / m as f64)
            }
        }));
    }
    let z = |j: usize| {
        if j == m {
            length
        } else {
            length * j as f64 / m as f64
        }
    };
    let at = |p: [f64; 2], height: f64| [p[0], p[1], height];
    let mut s = Surface::default();
    let n = rim.len();
    for i in 0..n {
        let (a, b) = (rim[i], rim[(i + 1) % n]);
        for j in 0..m {
            s.quad(at(a, z(j)), at(b, z(j)), at(b, z(j + 1)), at(a, z(j + 1)));
        }
    }
    let centre = [depth * 2.0 / 3.0, 0.0];
    for i in 0..n {
        let (a, b) = (rim[i], rim[(i + 1) % n]);
        s.triangle(at(centre, length), at(a, length), at(b, length));
        s.triangle(at(centre, 0.0), at(b, 0.0), at(a, 0.0));
    }
    s
}

#[test]
fn a_rounded_flat_face_gets_no_point_on_a_quads_other_diagonal() {
    let (depth, length, m) = (40.0, 60.0, 8);
    let base = wedge(depth, length, m);
    // The three sides' unrounded planes, as a point and an outward normal before turning.
    let corners = section(depth);
    let sides = Surface {
        points: (0..3)
            .flat_map(|side| {
                let (a, b) = (corners[side], corners[(side + 1) % 3]);
                let along = [b[0] - a[0], b[1] - a[1]];
                let reach = (along[0] * along[0] + along[1] * along[1]).sqrt();
                [
                    [a[0], a[1], 0.0],
                    [along[1] / reach, -along[0] / reach, 0.0],
                ]
            })
            .collect(),
        ..Surface::default()
    };
    for k in 0..6 {
        let angles = [
            0.37 + 0.31 * k as f64,
            0.61 + 0.17 * k as f64,
            0.23 * k as f64,
        ];
        let (turned, triangles) = rotated_by(&base, angles);
        let (planes, _) = rotated_by(&sides, angles);
        // Single precision, as mesh files store positions: each coordinate rounds by at most
        // 2^-24 of the largest magnitude `M`, so a point is within `e = sqrt(3) 2^-24 M` of the
        // plane it was meant on, and so is any point on a segment between two such points.
        let points: Vec<[f64; 3]> = turned
            .iter()
            .map(|p| p.map(|c| f64::from(c as f32)))
            .collect();
        let magnitude = points.iter().flatten().fold(0.0_f64, |m, c| m.max(c.abs()));
        let e = 3.0_f64.sqrt() * magnitude / 16_777_216.0;
        let surface = Surface {
            points,
            triangles,
            ..Surface::default()
        };
        let surface_points = surface.points.len();
        let (refined, _) = run_refine(
            recovered(&surface),
            constant_size(depth / m as f64),
            options_with_layers(2),
        )
        .expect("refines");
        // A point added inside the solid stands well off the surface: layer points at a fraction
        // of the wall, size points inside. One within twice the rounding of a side's plane is on a
        // chord along that side, and with the side's triangle it makes an element flat to rounding.
        for (index, &point) in refined.points.iter().enumerate().skip(surface_points) {
            for pair in planes.chunks(2) {
                let off = dot(sub(point, pair[0]), pair[1]).abs();
                assert!(
                    off > 2.0 * e,
                    "turn {k}: point {index} at {point:?} is {off:e} off a flat side, within \
                     twice its rounding {e:e}"
                );
            }
        }
    }
}

/// A wall `length` long along z on the section (-0.3, 0), (0.3, 0), (0.4, 10), (-0.4, 10), the
/// four faces one quad across at each of `stations` steps, the two sides first in triangle order
/// so a strip vertex on the top's acute edges sees a side triangle first; the ends are two
/// triangles each.
fn dovetail(length: f64, stations: usize) -> Surface {
    let section = [[-0.3, 0.0], [0.3, 0.0], [0.4, 10.0], [-0.4, 10.0]];
    let z = |j: usize| {
        if j == stations {
            length
        } else {
            length * j as f64 / stations as f64
        }
    };
    let at = |p: [f64; 2], height: f64| [p[0], p[1], height];
    let mut s = Surface::default();
    // Sides (+x, then -x), then the top, then the bottom: each face runs from section corner `i`
    // to `i + 1`, counterclockwise seen from +z.
    for i in [1usize, 3, 2, 0] {
        let (a, b) = (section[i], section[(i + 1) % 4]);
        for j in 0..stations {
            s.quad(at(a, z(j)), at(b, z(j)), at(b, z(j + 1)), at(a, z(j + 1)));
        }
    }
    let [p0, p1, p2, p3] = section;
    s.quad(
        at(p0, length),
        at(p1, length),
        at(p2, length),
        at(p3, length),
    );
    s.quad(at(p0, 0.0), at(p3, 0.0), at(p2, 0.0), at(p1, 0.0));
    s
}

#[test]
fn a_strip_one_quad_across_between_acute_edges_gets_no_point_on_its_other_diagonal() {
    let base = dovetail(20.0, 8);
    // The four faces' unrounded planes, as a point and an outward normal before turning.
    let section = [[-0.3, 0.0], [0.3, 0.0], [0.4, 10.0], [-0.4, 10.0]];
    let faces = Surface {
        points: (0..4)
            .flat_map(|i| {
                let (a, b): ([f64; 2], [f64; 2]) = (section[i], section[(i + 1) % 4]);
                let along = [b[0] - a[0], b[1] - a[1]];
                let reach = (along[0] * along[0] + along[1] * along[1]).sqrt();
                [
                    [a[0], a[1], 0.0],
                    [along[1] / reach, -along[0] / reach, 0.0],
                ]
            })
            .collect(),
        ..Surface::default()
    };
    for k in 0..8 {
        let angles = [0.37 * k as f64, 0.61 * k as f64, 0.23 * k as f64];
        let (turned, triangles) = rotated_by(&base, angles);
        let (planes, _) = rotated_by(&faces, angles);
        // Single precision, as for the wedge above: a point is within `e` of its plane.
        let points: Vec<[f64; 3]> = turned
            .iter()
            .map(|p| p.map(|c| f64::from(c as f32)))
            .collect();
        let magnitude = points.iter().flatten().fold(0.0_f64, |m, c| m.max(c.abs()));
        let e = 3.0_f64.sqrt() * magnitude / 16_777_216.0;
        let surface = Surface {
            points,
            triangles,
            ..Surface::default()
        };
        let surface_points = surface.points.len();
        let (refined, _) = run_refine(
            recovered(&surface),
            constant_size(2.5),
            // No quality passes: smoothing can move a split point off the face it was put on,
            // hiding a chord wrongly called through, and it is that call this test is about.
            RefineOptions {
                quality_passes: 0,
                ..options_with_layers(2)
            },
        )
        .expect("refines");
        for (index, &point) in refined.points.iter().enumerate().skip(surface_points) {
            for pair in planes.chunks(2) {
                let off = dot(sub(point, pair[0]), pair[1]).abs();
                assert!(
                    off > 2.0 * e,
                    "turn {k}: point {index} at {point:?} is {off:e} off a flat face, within \
                     twice its rounding {e:e}"
                );
            }
        }
    }
}
