//! The shape of each real-part benchmark part, built through solidmesh's operations into a closed
//! triangle mesh for `surfaces::remesh_density` to remesh. Dimensions are in the part's own units;
//! what each part is measured against is in `benchmarks/cases/`. Each builder takes the cancel
//! token its CSG operations check, though the extrudes and revolves do not need one.

use std::f64::consts::{PI, TAU};

use solidmesh_math::Vector2;
use solidmesh_mesh::csg::csg_operations;
use solidmesh_mesh::processors::outline_to_mesh::{extrude_contours, revolve_contours};
use solidmesh_mesh::{CancelToken, Mesh};

use super::SurfaceError;

type Build = Result<Mesh, SurfaceError>;

fn v(x: f64, y: f64) -> Vector2 {
    Vector2 { x, y }
}

fn shape_error(part: &'static str, reason: impl std::fmt::Display) -> SurfaceError {
    SurfaceError::Shape {
        part,
        reason: reason.to_string(),
    }
}

fn extrude(part: &'static str, outline: Vec<Vector2>, height: f64) -> Build {
    extrude_contours(&[outline], height).map_err(|e| shape_error(part, e))
}

fn rectangle(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Vector2> {
    vec![v(x0, y0), v(x1, y0), v(x1, y1), v(x0, y1)]
}

fn subtract(part: &'static str, a: &Mesh, b: &Mesh, cancel: &CancelToken) -> Build {
    csg_operations::subtract(a, b, cancel).map_err(|e| shape_error(part, e))
}

/// The mesh moved along z: extrudes start at z = 0, and a cutter or a centred part does not.
fn shifted(mut mesh: Mesh, dz: f32) -> Mesh {
    for p in mesh.vertices.iter_mut() {
        p.z += dz;
    }
    mesh
}

/// A strip 100 x 10 x 1.
pub fn strip(_cancel: &CancelToken) -> Build {
    extrude("strip", rectangle(0.0, 0.0, 100.0, 10.0), 1.0)
}

/// A plate 300 x 100 x 2 with a diameter 10 hole at its centre.
pub fn plate_with_hole(cancel: &CancelToken) -> Build {
    let plate = extrude("plate", rectangle(0.0, 0.0, 300.0, 100.0), 2.0)?;
    let circle: Vec<Vector2> = (0..256)
        .map(|i| {
            let a = TAU * i as f64 / 256.0;
            v(150.0 + 5.0 * a.cos(), 50.0 + 5.0 * a.sin())
        })
        .collect();
    let cutter = shifted(extrude("plate", circle, 4.0)?, -1.0);
    subtract("plate", &plate, &cutter, cancel)
}

/// A ring of bore 10, outer diameter 20 and height 8, revolved.
pub fn ring(_cancel: &CancelToken) -> Build {
    revolve_contours(&[rectangle(5.0, 0.0, 10.0, 8.0)], 240, 0.0, TAU, true)
        .map_err(|e| shape_error("ring", e))
}

/// An L bracket outline: 100 by 100 legs 10 wide, joined by `tail` at the inner corner.
fn l_bracket(tail: &[Vector2]) -> Vec<Vector2> {
    let mut outline = vec![v(0.0, 0.0), v(100.0, 0.0), v(100.0, 10.0)];
    outline.extend_from_slice(tail);
    outline.extend_from_slice(&[v(10.0, 100.0), v(0.0, 100.0)]);
    outline
}

/// An L bracket 5 thick with a sharp inner corner.
pub fn l_bracket_sharp(_cancel: &CancelToken) -> Build {
    extrude("lsharp", l_bracket(&[v(10.0, 10.0)]), 5.0)
}

/// The same L bracket with a radius 5 fillet at the inner corner.
pub fn l_bracket_filleted(_cancel: &CancelToken) -> Build {
    let mut tail = Vec::new();
    for i in 0..=48 {
        let a = 1.5 * PI - (0.5 * PI) * i as f64 / 48.0;
        tail.push(v(15.0 + 5.0 * a.cos(), 15.0 + 5.0 * a.sin()));
    }
    extrude("lfillet", l_bracket(&tail), 5.0)
}

/// The quarter elliptic membrane of NAFEMS LE1 and LE10: outer semi-axes 3250 and 2750, inner
/// 2000 and 1000.
fn quarter_ellipse() -> Vec<Vector2> {
    let ellipse = |a: f64, b: f64, t: f64| v(a * t.cos(), b * t.sin());
    let mut outline = Vec::new();
    for i in 0..=200 {
        outline.push(ellipse(3250.0, 2750.0, 0.5 * PI * i as f64 / 200.0));
    }
    for i in (0..=200).rev() {
        outline.push(ellipse(2000.0, 1000.0, 0.5 * PI * i as f64 / 200.0));
    }
    outline
}

/// NAFEMS LE1: the membrane 100 thick.
pub fn nafems_le1(_cancel: &CancelToken) -> Build {
    extrude("le1", quarter_ellipse(), 100.0)
}

/// NAFEMS LE10: the membrane 600 thick, centred on z = 0.
pub fn nafems_le10(_cancel: &CancelToken) -> Build {
    Ok(shifted(extrude("le10", quarter_ellipse(), 600.0)?, -300.0))
}

/// A crane hook: a ring sector of radii 20 and 40 from 90 to 300 degrees, with a straight shank,
/// 10 thick.
pub fn hook(cancel: &CancelToken) -> Build {
    let arc = |r: f64, a0: f64, a1: f64, n: usize| -> Vec<Vector2> {
        (0..=n)
            .map(|i| {
                let a = a0 + (a1 - a0) * i as f64 / n as f64;
                v(r * a.to_radians().cos(), r * a.to_radians().sin())
            })
            .collect()
    };
    let mut outline = arc(40.0, 90.0, 300.0, 180);
    outline.extend(arc(20.0, 300.0, 90.0, 90));
    let ring = extrude("hook", outline, 10.0)?;
    let shank = extrude("hook", rectangle(-10.0, 25.0, 10.0, 110.0), 10.0)?;
    csg_operations::union(&ring, &shank, cancel).map_err(|e| shape_error("hook", e))
}

/// A snap-fit beam 20 x 5 x 2.
pub fn snap_fit(_cancel: &CancelToken) -> Build {
    extrude("snap", rectangle(0.0, 0.0, 20.0, 5.0), 2.0)
}

/// A closed box 40 x 40 x 20 with walls 1 thick.
pub fn hollow_box(cancel: &CancelToken) -> Build {
    let outer = extrude("hollow", rectangle(0.0, 0.0, 40.0, 40.0), 20.0)?;
    let void = shifted(
        extrude("hollow", rectangle(1.0, 1.0, 39.0, 39.0), 18.0)?,
        1.0,
    );
    subtract("hollow", &outer, &void, cancel)
}

/// The outline of one tooth of a 20-tooth, module 4, 20 degree pressure angle involute gear: a
/// rim sector with the tooth centred on the y axis and a root fillet of radius 0.3 times the
/// module.
pub fn gear_tooth_outline() -> Vec<Vector2> {
    let (n, m, phi) = (20.0_f64, 4.0_f64, 20.0_f64.to_radians());
    let rp = 0.5 * n * m;
    let rb = rp * phi.cos();
    let ra = rp + m;
    let rr = rp - 1.25 * m;
    let rf = 0.3 * m;
    let rin = rr - 11.0;
    let theta_s = 1.25 * PI / n;
    let inv = |a: f64| a.tan() - a;
    let half_tooth = 0.25 * PI * m / rp;
    let flank = |r: f64| half_tooth + inv(phi) - inv((rb / r).min(1.0).acos());
    let theta_b = half_tooth + inv(phi);
    let alpha = (rf / (rr + rf)).asin();
    let polar = |r: f64, t: f64| [r * t.sin(), r * t.cos()];
    let mut right: Vec<[f64; 2]> = vec![polar(rin, theta_s), polar(rr, theta_s)];
    // Root arc from the sector side to the fillet tangent.
    let root_end = theta_b + alpha;
    for i in 1..=8 {
        let t = theta_s + (root_end - theta_s) * i as f64 / 8.0;
        right.push(polar(rr, t));
    }
    // Fillet arc about its centre, from the root tangent to the flank tangent.
    let c = polar(rr + rf, root_end);
    let p_root = polar(rr, root_end);
    let p_line = polar((rr + rf) * alpha.cos(), theta_b);
    let a0 = (p_root[1] - c[1]).atan2(p_root[0] - c[0]);
    let mut a1 = (p_line[1] - c[1]).atan2(p_line[0] - c[0]);
    while a1 > a0 + PI {
        a1 -= TAU;
    }
    while a1 < a0 - PI {
        a1 += TAU;
    }
    for i in 1..=14 {
        let a = a0 + (a1 - a0) * i as f64 / 14.0;
        right.push([c[0] + rf * a.cos(), c[1] + rf * a.sin()]);
    }
    right.push(polar(rb, theta_b));
    for i in 1..=20 {
        let r = rb + (ra - rb) * i as f64 / 20.0;
        right.push(polar(r, flank(r)));
    }
    let mut profile: Vec<Vector2> = right.iter().map(|p| v(p[0], p[1])).collect();
    // Across the tip and down the mirror image, then the rim arc back.
    let tip_right = flank(ra);
    for i in 1..=8 {
        let t = tip_right - 2.0 * tip_right * i as f64 / 8.0;
        profile.push(v(ra * t.sin(), ra * t.cos()));
    }
    for p in right.iter().rev() {
        profile.push(v(-p[0], p[1]));
    }
    for i in 1..16 {
        let t = -theta_s + 2.0 * theta_s * i as f64 / 16.0;
        profile.push(v(rin * t.sin(), rin * t.cos()));
    }
    profile
}

/// The gear tooth extruded 20 wide.
pub fn gear_tooth(_cancel: &CancelToken) -> Build {
    extrude("gear", gear_tooth_outline(), 20.0)
}
