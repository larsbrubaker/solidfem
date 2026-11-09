// A sweep over thin-walled shapes, independent of the library's own idea of a wall: each shape's
// generator says which surface vertices stand on which skin of which wall, from how it built them,
// and the test reads the refined tetrahedra for an interior edge joining the two skins of one wall.
// Such an edge in a run that returned `Ok` is a false success. Edges both of whose ends are on the
// same free rim side run along that rim, not through the wall, and are not counted. A named error
// is an honest outcome and is reported, not counted.

use std::collections::HashSet;
use std::f64::consts::TAU;

use crate::check::mesh;
use crate::shapes::{rotated_by, sin_cos, Surface};
use solidfem_tetra::RefineOptions;

use crate::support::{constant_size, options_with_layers, run_refine};

/// One skin membership of one surface vertex: which wall, which of its two skins, and which of the
/// wall's free rim sides the vertex is on, as a bit mask.
#[derive(Clone, Copy)]
struct Tag {
    wall: u16,
    side: u8,
    rim: u8,
}

/// A surface and each vertex's skin memberships, recorded by its generator.
#[derive(Default, Clone)]
struct Probe {
    surface: Surface,
    tags: Vec<Vec<Tag>>,
}

impl Probe {
    /// The welded vertex at `point`, adding `tags` the first time it is seen.
    fn vertex(&mut self, point: [f64; 3], tags: impl FnOnce() -> Vec<Tag>) -> u32 {
        let before = self.surface.points.len();
        let index = self.surface.vertex(point);
        if self.surface.points.len() > before {
            self.tags.push(tags());
        }
        index
    }

    fn triangle(&mut self, a: u32, b: u32, c: u32) {
        self.surface.triangles.push([a, b, c]);
    }

    fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.triangle(a, b, c);
        self.triangle(a, c, d);
    }

    /// Triangles between two rows of vertices that run the same way, `low` then `high` being
    /// counterclockwise from outside; each row's parameter runs over 0 to 1 and the next triangle
    /// advances whichever row is behind.
    fn zip(&mut self, low: &[(f64, u32)], high: &[(f64, u32)]) {
        let (mut i, mut j) = (0, 0);
        while i + 1 < low.len() || j + 1 < high.len() {
            let advance_low =
                j + 1 == high.len() || (i + 1 < low.len() && low[i + 1].0 <= high[j + 1].0);
            if advance_low {
                self.triangle(low[i].1, low[i + 1].1, high[j].1);
                i += 1;
            } else {
                self.triangle(low[i].1, high[j + 1].1, high[j].1);
                j += 1;
            }
        }
    }

    fn rotated(&self, angles: [f64; 3]) -> Probe {
        let (points, triangles) = rotated_by(&self.surface, angles);
        Probe {
            surface: Surface {
                points,
                triangles,
                ..Surface::default()
            },
            tags: self.tags.clone(),
        }
    }
}

/// Plate rim sides, one bit each.
const X_LOW: u8 = 1;
const X_HIGH: u8 = 2;
const Y_LOW: u8 = 4;
const Y_HIGH: u8 = 8;

/// A plate of `size` whose two skins are divided differently, its height at a plan position given
/// by `at(x, y, top)`.
fn plate(
    size: [f64; 2],
    top: [usize; 2],
    bottom: [usize; 2],
    at: &dyn Fn(f64, f64, bool) -> f64,
) -> Probe {
    let mut probe = Probe::default();
    let rim_of = |x: f64, y: f64| {
        let mut rim = 0;
        for (on, bit) in [
            (x == 0.0, X_LOW),
            (x == size[0], X_HIGH),
            (y == 0.0, Y_LOW),
            (y == size[1], Y_HIGH),
        ] {
            if on {
                rim |= bit;
            }
        }
        rim
    };
    let mut vertex = |p: &mut Probe, x: f64, y: f64, upper: bool| {
        let z = at(x, y, upper);
        p.vertex([x, y, z], || {
            vec![Tag {
                wall: 0,
                side: u8::from(upper),
                rim: rim_of(x, y),
            }]
        })
    };
    // `nx` cells of `size[0]` place vertex `i` at exactly the side at `i == nx`.
    let line = |n: usize, length: f64, i: usize| length * i as f64 / n as f64;
    for (grid, upper) in [(top, true), (bottom, false)] {
        for i in 0..grid[0] {
            for j in 0..grid[1] {
                let (x0, x1) = (line(grid[0], size[0], i), line(grid[0], size[0], i + 1));
                let (y0, y1) = (line(grid[1], size[1], j), line(grid[1], size[1], j + 1));
                let a = vertex(&mut probe, x0, y0, upper);
                let b = vertex(&mut probe, x1, y0, upper);
                let c = vertex(&mut probe, x1, y1, upper);
                let d = vertex(&mut probe, x0, y1, upper);
                if upper {
                    probe.quad(a, b, c, d);
                } else {
                    probe.quad(a, d, c, b);
                }
            }
        }
    }
    // Each side strip, its two rows running counterclockwise as seen from outside the plate.
    let mut strip = |p: &mut Probe, along_x: bool, at_far: bool, reversed: bool| {
        let row = |p: &mut Probe,
                   grid: [usize; 2],
                   upper: bool,
                   vertex: &mut dyn FnMut(&mut Probe, f64, f64, bool) -> u32| {
            let n = if along_x { grid[0] } else { grid[1] };
            let length = if along_x { size[0] } else { size[1] };
            let fixed = if along_x { size[1] } else { size[0] };
            let fixed = if at_far { fixed } else { 0.0 };
            let mut out: Vec<(f64, u32)> = (0..=n)
                .map(|i| {
                    let run = line(n, length, i);
                    let (x, y) = if along_x { (run, fixed) } else { (fixed, run) };
                    (i as f64 / n as f64, vertex(p, x, y, upper))
                })
                .collect();
            if reversed {
                out.reverse();
                for (t, _) in &mut out {
                    *t = 1.0 - *t;
                }
            }
            out
        };
        let lower = row(p, bottom, false, &mut vertex);
        let upper = row(p, top, true, &mut vertex);
        p.zip(&lower, &upper);
    };
    strip(&mut probe, true, false, false);
    strip(&mut probe, false, true, false);
    strip(&mut probe, true, true, true);
    strip(&mut probe, false, false, true);
    probe
}

/// One thin wall of an extruded section: its thin axis (0 for x, 1 for z in the profile), the two
/// lattice planes that are its skins, each skin's lattice range along the other profile axis (a
/// skin stops where another wall meets it, so the two ranges can differ), and whether the low and
/// high ends of that range are free rims.
struct Wall {
    thin_axis: usize,
    planes: [usize; 2],
    ranges: [[usize; 2]; 2],
    free_ends: [bool; 2],
}

/// A section extruded along y: the profile is the occupied cells of a lattice `xs` by `zs`, swept
/// over `stations`. Rim bit 1 and 2 are the two ends of the extrusion, 4 and 8 a wall's free ends.
fn section(
    xs: &[f64],
    zs: &[f64],
    ys: &[f64],
    filled: &dyn Fn(usize, usize) -> bool,
    walls: &[Wall],
) -> Probe {
    let mut probe = Probe::default();
    let tags = |i: usize, j: usize, k: usize| {
        let mut out = Vec::new();
        for (index, wall) in walls.iter().enumerate() {
            let (thin, run) = if wall.thin_axis == 0 { (i, k) } else { (k, i) };
            for (side, &plane) in wall.planes.iter().enumerate() {
                let range = wall.ranges[side];
                if thin == plane && run >= range[0] && run <= range[1] {
                    let mut rim = 0;
                    if j == 0 {
                        rim |= 1;
                    }
                    if j + 1 == ys.len() {
                        rim |= 2;
                    }
                    if wall.free_ends[0] && run == range[0] {
                        rim |= 4;
                    }
                    if wall.free_ends[1] && run == range[1] {
                        rim |= 8;
                    }
                    out.push(Tag {
                        wall: index as u16,
                        side: side as u8,
                        rim,
                    });
                }
            }
        }
        out
    };
    let at = |p: &mut Probe, i: usize, j: usize, k: usize| {
        p.vertex([xs[i], ys[j], zs[k]], || tags(i, j, k))
    };
    let (nx, nz) = (xs.len() - 1, zs.len() - 1);
    let solid = |i: isize, k: isize| {
        i >= 0 && k >= 0 && (i as usize) < nx && (k as usize) < nz && filled(i as usize, k as usize)
    };
    for k in 0..nz {
        for i in 0..nx {
            if !filled(i, k) {
                continue;
            }
            let (ii, kk) = (i as isize, k as isize);
            for j in 0..ys.len() - 1 {
                if !solid(ii - 1, kk) {
                    let (a, b, c, d) = (
                        at(&mut probe, i, j, k),
                        at(&mut probe, i, j, k + 1),
                        at(&mut probe, i, j + 1, k + 1),
                        at(&mut probe, i, j + 1, k),
                    );
                    probe.quad(a, b, c, d);
                }
                if !solid(ii + 1, kk) {
                    let (a, b, c, d) = (
                        at(&mut probe, i + 1, j, k),
                        at(&mut probe, i + 1, j + 1, k),
                        at(&mut probe, i + 1, j + 1, k + 1),
                        at(&mut probe, i + 1, j, k + 1),
                    );
                    probe.quad(a, b, c, d);
                }
                if !solid(ii, kk - 1) {
                    let (a, b, c, d) = (
                        at(&mut probe, i, j, k),
                        at(&mut probe, i, j + 1, k),
                        at(&mut probe, i + 1, j + 1, k),
                        at(&mut probe, i + 1, j, k),
                    );
                    probe.quad(a, b, c, d);
                }
                if !solid(ii, kk + 1) {
                    let (a, b, c, d) = (
                        at(&mut probe, i, j, k + 1),
                        at(&mut probe, i + 1, j, k + 1),
                        at(&mut probe, i + 1, j + 1, k + 1),
                        at(&mut probe, i, j + 1, k + 1),
                    );
                    probe.quad(a, b, c, d);
                }
            }
            // The two ends of the extrusion.
            let last = ys.len() - 1;
            let (a, b, c, d) = (
                at(&mut probe, i, 0, k),
                at(&mut probe, i + 1, 0, k),
                at(&mut probe, i + 1, 0, k + 1),
                at(&mut probe, i, 0, k + 1),
            );
            probe.quad(a, b, c, d);
            let (a, b, c, d) = (
                at(&mut probe, i, last, k),
                at(&mut probe, i, last, k + 1),
                at(&mut probe, i + 1, last, k + 1),
                at(&mut probe, i + 1, last, k),
            );
            probe.quad(a, b, c, d);
        }
    }
    probe
}

fn uniform(from: f64, to: f64, cells: usize) -> Vec<f64> {
    (0..=cells)
        .map(|i| from + (to - from) * i as f64 / cells as f64)
        .collect()
}

fn with_cut(from: f64, cut_low: f64, cut_high: f64, to: f64, each: usize) -> Vec<f64> {
    let mut out = uniform(from, cut_low, each);
    out.push(cut_high);
    out.extend(uniform(cut_high, to, each).into_iter().skip(1));
    out
}

/// A plain tube along z with `outer` and `inner` segments per ring and `rings` stations.
fn tube(radius: f64, wall: f64, height: f64, outer: usize, inner: usize, rings: usize) -> Probe {
    let mut probe = Probe::default();
    let at = |p: &mut Probe, radius: f64, n: usize, i: usize, ring: usize, side: u8| {
        let (s, c) = sin_cos(TAU * (i % n) as f64 / n as f64);
        let z = height * ring as f64 / rings as f64;
        let rim = u8::from(ring == 0) | (u8::from(ring == rings) << 1);
        p.vertex([radius * c, radius * s, z], || {
            vec![Tag { wall: 0, side, rim }]
        })
    };
    for ring in 0..rings {
        for i in 0..outer {
            let (a, b, c, d) = (
                at(&mut probe, radius, outer, i, ring, 1),
                at(&mut probe, radius, outer, i + 1, ring, 1),
                at(&mut probe, radius, outer, i + 1, ring + 1, 1),
                at(&mut probe, radius, outer, i, ring + 1, 1),
            );
            probe.quad(a, b, c, d);
        }
        for i in 0..inner {
            let inner_radius = radius - wall;
            let (a, b, c, d) = (
                at(&mut probe, inner_radius, inner, i + 1, ring, 0),
                at(&mut probe, inner_radius, inner, i, ring, 0),
                at(&mut probe, inner_radius, inner, i, ring + 1, 0),
                at(&mut probe, inner_radius, inner, i + 1, ring + 1, 0),
            );
            probe.quad(a, b, c, d);
        }
    }
    // The annular ends: each ring's vertices in angle order, the wider one's turn decided by angle.
    let rows = |p: &mut Probe, ring: usize, r: f64, n: usize, side: u8| -> Vec<(f64, u32)> {
        (0..=n)
            .map(|i| (i as f64 / n as f64, at(p, r, n, i, ring, side)))
            .collect()
    };
    let outer_row = rows(&mut probe, 0, radius, outer, 1);
    let inner_row = rows(&mut probe, 0, radius - wall, inner, 0);
    probe.zip(&inner_row, &outer_row);
    let outer_row = rows(&mut probe, rings, radius, outer, 1);
    let inner_row = rows(&mut probe, rings, radius - wall, inner, 0);
    probe.zip(&outer_row, &inner_row);
    probe
}

/// The sweep's shapes, each named so a false success can be read off.
fn shapes() -> Vec<(String, Probe, f64)> {
    let rotations: [[f64; 3]; 5] = [
        [0.0, 0.0, 0.0],
        [0.37, 0.61, 0.0],
        [1.1, 0.23, 0.47],
        [0.71, 1.13, 0.37],
        [2.3, 0.9, 1.7],
    ];
    let mut base: Vec<(String, Probe, f64)> = Vec::new();
    // Flat, wavy and tapered plates, the skins divided alike or crosswise.
    for (name, top, bottom) in [
        ("even", [8, 8], [8, 8]),
        ("mismatched", [10, 6], [6, 10]),
        ("cross", [5, 30], [30, 5]),
        ("cross2", [40, 3], [3, 40]),
    ] {
        let thick = 0.8;
        base.push((
            format!("plate {name} flat"),
            plate([30.0, 30.0], top, bottom, &|_, _, upper| {
                if upper {
                    thick
                } else {
                    0.0
                }
            }),
            2.0,
        ));
        base.push((
            format!("plate {name} wavy"),
            plate([30.0, 30.0], top, bottom, &|x, y, upper| {
                // Plain sine-free ripple: a parabola in each direction, the same on both skins.
                let ripple = 0.8 * (x / 30.0) * (1.0 - x / 30.0) * (y / 30.0) * (1.0 - y / 30.0);
                ripple * 4.0 + if upper { thick } else { 0.0 }
            }),
            2.0,
        ));
        base.push((
            format!("plate {name} tapered"),
            plate([30.0, 30.0], top, bottom, &|x, _, upper| {
                if upper {
                    thick * (1.0 + 0.5 * x / 30.0)
                } else {
                    0.0
                }
            }),
            2.0,
        ));
    }
    // Sections: L, T, U, stepped, thick next to thin, each with two wall thicknesses.
    for t in [0.8, 1.2] {
        let ys = uniform(0.0, 18.0, 5);
        // L: a base along x at the bottom, a stem up its low end.
        let xs_l = {
            let mut v = vec![0.0, t];
            v.extend((1..=6).map(|i| t + i as f64 * 3.0));
            v
        };
        let zs_l = {
            let mut v = vec![0.0, t];
            v.extend((1..=5).map(|i| t + i as f64 * 3.0));
            v
        };
        let (l_nx, l_nz) = (xs_l.len() - 1, zs_l.len() - 1);
        base.push((
            format!("L section {t}"),
            section(
                &xs_l,
                &zs_l,
                &ys,
                &|i, k| k == 0 || i == 0,
                &[
                    Wall {
                        thin_axis: 1,
                        planes: [0, 1],
                        ranges: [[0, l_nx], [1, l_nx]],
                        free_ends: [false, true],
                    },
                    Wall {
                        thin_axis: 0,
                        planes: [0, 1],
                        ranges: [[0, l_nz], [1, l_nz]],
                        free_ends: [false, true],
                    },
                ],
            ),
            3.0,
        ));
        // T: a base along x and a stem up its middle.
        let xs_t = with_cut(0.0, 10.8, 10.8 + t, 22.4 + t, 4);
        let zs_t = {
            let mut v = vec![0.0, t];
            v.extend((1..=5).map(|i| t + i as f64 * 3.0));
            v
        };
        let stem = xs_t
            .iter()
            .position(|&x| x == 10.8)
            .expect("the stem's plane is a lattice value");
        let (t_nx, t_nz) = (xs_t.len() - 1, zs_t.len() - 1);
        base.push((
            format!("T section {t}"),
            section(
                &xs_t,
                &zs_t,
                &ys,
                &|i, k| k == 0 || i == stem,
                &[
                    Wall {
                        thin_axis: 1,
                        planes: [0, 1],
                        ranges: [[0, t_nx], [0, t_nx]],
                        free_ends: [true, true],
                    },
                    Wall {
                        thin_axis: 0,
                        planes: [stem, stem + 1],
                        ranges: [[1, t_nz], [1, t_nz]],
                        free_ends: [false, true],
                    },
                ],
            ),
            3.0,
        ));
        // U: a base and a wall up each end.
        let xs_u = {
            let mut v = vec![0.0, t];
            v.extend((1..=6).map(|i| t + i as f64 * 3.0));
            let last = *v.last().expect("not empty");
            v.push(last + t);
            v
        };
        let u_nx = xs_u.len() - 1;
        let (right, zs_u) = (u_nx - 1, zs_t.clone());
        let u_nz = zs_u.len() - 1;
        base.push((
            format!("U section {t}"),
            section(
                &xs_u,
                &zs_u,
                &ys,
                &|i, k| k == 0 || i == 0 || i == right,
                &[
                    Wall {
                        thin_axis: 1,
                        planes: [0, 1],
                        ranges: [[0, u_nx], [1, u_nx - 1]],
                        free_ends: [false, false],
                    },
                    Wall {
                        thin_axis: 0,
                        planes: [0, 1],
                        ranges: [[0, u_nz], [1, u_nz]],
                        free_ends: [false, true],
                    },
                    Wall {
                        thin_axis: 0,
                        planes: [right, right + 1],
                        ranges: [[1, u_nz], [0, u_nz]],
                        free_ends: [false, true],
                    },
                ],
            ),
            3.0,
        ));
        // Stepped: thin on the high x end, one and a half times as thick on the low; and thick next
        // to thin, a block several walls deep on the low end with only the thin part a wall.
        let xs_step = uniform(0.0, 24.0, 8);
        let middle = 4;
        let zs_step = vec![0.0, t, 1.5 * t];
        base.push((
            format!("stepped {t}"),
            section(
                &xs_step,
                &zs_step,
                &ys,
                &|i, k| k == 0 || i < middle,
                &[
                    Wall {
                        thin_axis: 1,
                        planes: [0, 2],
                        ranges: [[0, middle], [0, middle]],
                        free_ends: [true, false],
                    },
                    Wall {
                        thin_axis: 1,
                        planes: [0, 1],
                        ranges: [[middle, 8], [middle, 8]],
                        free_ends: [false, true],
                    },
                ],
            ),
            3.0,
        ));
        let zs_deep = vec![0.0, t, 5.0 + t];
        base.push((
            format!("thick and thin {t}"),
            section(
                &xs_step,
                &zs_deep,
                &ys,
                &|i, k| k == 0 || i < middle,
                &[Wall {
                    thin_axis: 1,
                    planes: [0, 1],
                    ranges: [[middle, 8], [middle, 8]],
                    free_ends: [false, true],
                }],
            ),
            3.0,
        ));
    }
    // Tubes: matched and mismatched ring densities.
    for (radius, wall, height, outer, inner, rings) in [
        (30.0, 1.0, 10.0, 48, 24, 6),
        (30.0, 1.0, 10.0, 36, 36, 4),
        (20.0, 0.8, 16.0, 32, 40, 5),
        (10.0, 1.0, 40.0, 24, 24, 1),
    ] {
        base.push((
            format!("tube R{radius} wall {wall} {outer}/{inner}x{rings}"),
            tube(radius, wall, height, outer, inner, rings),
            TAU * radius / outer as f64,
        ));
    }
    let mut all = Vec::new();
    for (name, probe, edge) in base {
        for rotation in rotations {
            let rotated = if rotation == [0.0; 3] {
                probe.clone()
            } else {
                probe.rotated(rotation)
            };
            all.push((format!("{name} rot {rotation:?}"), rotated, edge));
        }
    }
    all
}

/// The interior edges of `tets` joining the two skins of one wall, away from a shared free rim.
fn false_edges(probe: &Probe, tets: &[[u32; 4]]) -> Vec<(u32, u32)> {
    let surface_points = probe.surface.points.len();
    let mut surface_edges = HashSet::new();
    for t in &probe.surface.triangles {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            surface_edges.insert((a.min(b), a.max(b)));
        }
    }
    let mut found = Vec::new();
    let mut seen = HashSet::new();
    for tet in tets {
        for a in 0..4 {
            for b in a + 1..4 {
                let (u, v) = (tet[a].min(tet[b]), tet[a].max(tet[b]));
                if (v as usize) >= surface_points
                    || surface_edges.contains(&(u, v))
                    || !seen.insert((u, v))
                {
                    continue;
                }
                let joins = probe.tags[u as usize].iter().any(|one| {
                    probe.tags[v as usize].iter().any(|other| {
                        one.wall == other.wall && one.side != other.side && one.rim & other.rim == 0
                    })
                });
                if joins {
                    found.push((u, v));
                }
            }
        }
    }
    found
}

/// The pinned count of shapes for which refinement returned `Ok` while an interior edge joining
/// opposite skins remains (`cargo nextest run --workspace --test refine sweep`).
const PINNED_FALSE_SUCCESSES: usize = 0;

#[test]
#[ignore = "about 140 shapes, minutes of work: cargo test --release --workspace --test refine sweep -- --ignored --nocapture"]
fn every_shape_of_the_sweep_returns_either_no_edge_across_a_wall_or_a_named_error() {
    let shapes = shapes();
    let total = shapes.len();
    let (mut clean, mut refused, mut unrecovered) = (0usize, 0usize, 0usize);
    let mut false_successes = Vec::new();
    for (name, probe, edge) in &shapes {
        let unrefined = match mesh(&probe.surface.points, &probe.surface.triangles) {
            Ok((solid, _)) => solid,
            Err(error) => {
                unrecovered += 1;
                eprintln!("surface not recovered, {name}: {error}");
                continue;
            }
        };
        let options = RefineOptions {
            quality_passes: 1,
            ..options_with_layers(2)
        };
        match run_refine(unrefined, constant_size(1.5 * edge), options) {
            Err(error) => {
                refused += 1;
                eprintln!("named error, {name}: {error}");
            }
            Ok((refined, _)) => {
                let edges = false_edges(probe, &refined.tets);
                if edges.is_empty() {
                    clean += 1;
                } else {
                    false_successes.push(format!(
                        "{name}: {} edges, first {:?}",
                        edges.len(),
                        edges[0]
                    ));
                }
            }
        }
    }
    eprintln!(
        "sweep: {total} shapes, {clean} clean, {refused} named errors, {unrecovered} not recovered, \
         {} false successes",
        false_successes.len()
    );
    for line in &false_successes {
        eprintln!("false success: {line}");
    }
    assert_eq!(
        false_successes.len(),
        PINNED_FALSE_SUCCESSES,
        "false successes: {false_successes:#?}"
    );
}
